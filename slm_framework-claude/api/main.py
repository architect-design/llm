"""
FastAPI Backend — FinancialSLM API Server.

Endpoints:
  POST /api/validate          — Upload & validate a financial file
  POST /api/generate          — Generate a synthetic spec-valid file
  GET  /api/specs             — List available spec metadata
  GET  /api/specs/{name}/rules — Export full field rules for a spec
  GET  /api/specs/{name}/record-types — List record types + field tables
  POST /api/train             — Kick off a background training run
  GET  /api/train/status      — Poll training progress
  POST /api/config/override   — Set a runtime field-level rule override
  DELETE /api/config/override — Reset overrides
  GET  /api/health            — Health check

All inference is local. Zero external LLM calls.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Make project root importable ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from memory.config_engine import ConfigEngine
from memory.seeder        import DataSeeder
from slm.tokenizer        import make_tokenizer
from slm.validator        import FinancialValidator
from slm.generator        import FinancialGenerator, GenerationConfig

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("slm.api")

# ─────────────────────── Shared Singletons ────────────────────────────────────

engine   = ConfigEngine()
seeder   = DataSeeder(engine)

# Lazy model cache: {spec_name: (model, model_cfg)}
_model_cache: Dict[str, Any] = {}

def _get_model(spec_name: str):
    """Load or retrieve cached model for a spec. Returns (model, cfg) or (None, None)."""
    if spec_name in _model_cache:
        return _model_cache[spec_name]
    ckpt_dir = ROOT / "checkpoints"
    ckpt     = ckpt_dir / f"{spec_name}_best.pt"
    if not ckpt.exists():
        return None, None
    try:
        import torch
        from slm.model import build_model
        from slm.trainer import SLMTrainer
        model, cfg = build_model(spec_name)
        SLMTrainer.load_checkpoint(str(ckpt), model)
        model.eval()
        _model_cache[spec_name] = (model, cfg)
        log.info(f"Loaded checkpoint: {ckpt}")
        return model, cfg
    except Exception as e:
        log.warning(f"Could not load model for {spec_name}: {e}")
        return None, None


# ─────────────────────── Training State ──────────────────────────────────────

_train_state: Dict[str, Any] = {
    "running"  : False,
    "spec"     : None,
    "step"     : 0,
    "max_steps": 0,
    "metrics"  : [],
    "error"    : None,
    "started"  : None,
    "finished" : None,
}
_train_lock = threading.Lock()


# ─────────────────────── Pydantic Models ─────────────────────────────────────

class GenerateRequest(BaseModel):
    spec_name  : str = "ACH_NACHA"
    n_entries  : int = 3
    strategy   : str = "temperature"     # greedy | temperature | top_k
    temperature: float = 0.7
    top_k      : int   = 10
    use_model  : bool  = False

class TrainRequest(BaseModel):
    spec_name       : str   = "ACH_NACHA"
    max_steps       : int   = 2000
    batch_size      : int   = 16
    learning_rate   : float = 3e-4
    corruption_prob : float = 0.3

class OverrideRequest(BaseModel):
    spec_name   : str
    record_type : str
    field_name  : str
    overrides   : Dict[str, Any]

class ResetRequest(BaseModel):
    spec_name   : str
    record_type : Optional[str] = None


# ─────────────────────── App Init ────────────────────────────────────────────

app = FastAPI(
    title       = "FinancialSLM API",
    description = "Local SLM for financial file parsing, validation & generation",
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# Serve static frontend
frontend_dir = ROOT / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


# ─────────────────────── Health ──────────────────────────────────────────────

@app.get("/api/health")
async def health():
    models_loaded = list(_model_cache.keys())
    return {
        "status"        : "ok",
        "specs_loaded"  : [s["name"] for s in engine.get_all_specs()],
        "models_loaded" : models_loaded,
        "train_running" : _train_state["running"],
    }


# ─────────────────────── Specs & Rules ───────────────────────────────────────

@app.get("/api/specs")
async def list_specs():
    """Return metadata for all registered specs."""
    return {"specs": engine.get_all_specs()}


@app.get("/api/specs/{spec_name}/rules")
async def get_rules(spec_name: str):
    """Export full field rule set for a spec (JSON-serialisable)."""
    spec_name = spec_name.upper()
    rules = engine.export_rules(spec_name)
    if not rules.get("spec"):
        raise HTTPException(404, f"Spec '{spec_name}' not found")
    return rules


@app.get("/api/specs/{spec_name}/record-types")
async def get_record_types(spec_name: str):
    spec_name = spec_name.upper()
    rts = engine.get_record_types(spec_name)
    if not rts:
        raise HTTPException(404, f"Spec '{spec_name}' not found")
    result = {}
    for rt in rts:
        fields = engine.get_fields(spec_name, rt)
        result[rt] = [
            {k: v for k, v in f.items() if k != "descriptor"}
            for f in fields
        ]
    return {"spec_name": spec_name, "record_types": result}


@app.get("/api/specs/{spec_name}/describe/{record_type}")
async def describe_record(spec_name: str, record_type: str):
    spec_name   = spec_name.upper()
    record_type = record_type.upper()
    text = engine.describe(spec_name, record_type)
    return PlainTextResponse(text)


# ─────────────────────── Validation ──────────────────────────────────────────

@app.post("/api/validate")
async def validate_file(
    spec_name: str = "ACH_NACHA",
    file: UploadFile = File(...),
):
    """
    Upload a financial file for validation.
    Returns a full ValidationReport as JSON.
    """
    spec_name = spec_name.upper()
    if spec_name not in [s["name"] for s in engine.get_all_specs()]:
        raise HTTPException(400, f"Unknown spec: {spec_name}")

    raw_bytes = await file.read()
    try:
        raw_text = raw_bytes.decode("ascii")
    except UnicodeDecodeError:
        raw_text = raw_bytes.decode("latin-1")

    tok   = make_tokenizer(spec_name)
    model, mcfg = _get_model(spec_name)

    validator = FinancialValidator(
        spec_name     = spec_name,
        config_engine = engine,
        tokenizer     = tok,
        model         = model,
    )
    report = validator.validate(raw_text)
    return report.to_dict()


@app.post("/api/validate/text")
async def validate_text_body(body: Dict[str, str]):
    """Validate raw text sent in JSON body (for browser paste/demo)."""
    spec_name = body.get("spec_name", "ACH_NACHA").upper()
    raw_text  = body.get("content", "")
    if not raw_text:
        raise HTTPException(400, "No content provided")

    tok = make_tokenizer(spec_name)
    model, _ = _get_model(spec_name)
    validator = FinancialValidator(
        spec_name     = spec_name,
        config_engine = engine,
        tokenizer     = tok,
        model         = model,
    )
    report = validator.validate(raw_text)
    return report.to_dict()


# ─────────────────────── Generation ──────────────────────────────────────────

@app.post("/api/generate")
async def generate_file(req: GenerateRequest):
    """
    Generate a synthetic spec-valid financial file.
    Returns the raw file content as plain text plus metadata.
    """
    spec_name = req.spec_name.upper()
    if spec_name not in [s["name"] for s in engine.get_all_specs()]:
        raise HTTPException(400, f"Unknown spec: {spec_name}")

    model, mcfg = (None, None)
    if req.use_model:
        model, mcfg = _get_model(spec_name)
        if model is None:
            log.info(f"No trained model for {spec_name}, falling back to rule-based generation")

    tok = make_tokenizer(spec_name) if (model or True) else None

    gen_cfg = GenerationConfig(
        strategy    = req.strategy,
        temperature = req.temperature,
        top_k       = req.top_k,
    )
    generator = FinancialGenerator(
        spec_name   = spec_name,
        config_engine = engine,
        seeder      = seeder,
        tokenizer   = tok,
        model       = model,
        model_cfg   = mcfg,
    )

    t0      = time.time()
    content = generator.generate_file(gen_cfg, n_entries=req.n_entries)
    elapsed = round(time.time() - t0, 3)

    lines     = [l for l in content.split("\n") if l.strip()]
    line_len  = engine.get_line_length(spec_name)

    return {
        "spec_name"   : spec_name,
        "content"     : content,
        "line_count"  : len(lines),
        "line_length" : line_len,
        "generated_ms": int(elapsed * 1000),
        "used_model"  : model is not None,
        "n_entries"   : req.n_entries,
    }


@app.post("/api/generate/record")
async def generate_record(body: Dict[str, str]):
    """Generate a single record line for a given spec + record type."""
    spec_name   = body.get("spec_name", "ACH_NACHA").upper()
    record_type = body.get("record_type", "RT6").upper()

    gen = FinancialGenerator(
        spec_name     = spec_name,
        config_engine = engine,
        seeder        = seeder,
        tokenizer     = make_tokenizer(spec_name),
    )
    line, ctx = seeder.generate_line(spec_name, record_type, return_context=True)
    fields    = engine.get_fields(spec_name, record_type)

    # Annotate each field with its generated value
    annotated = []
    for fd in fields:
        s, e = fd["start"] - 1, fd["end"]
        annotated.append({
            "name"  : fd["name"],
            "start" : fd["start"],
            "end"   : fd["end"],
            "value" : line[s:e],
            "type"  : fd["field_type"],
        })

    return {
        "line"        : line,
        "record_type" : record_type,
        "spec_name"   : spec_name,
        "fields"      : annotated,
        "context"     : {k: v for k, v in (ctx or {}).items() if isinstance(v, (str, int, float))},
    }


# ─────────────────────── Training ────────────────────────────────────────────

def _run_training(req: TrainRequest):
    """Background thread entry point for model training."""
    global _train_state
    with _train_lock:
        _train_state.update({
            "running"  : True,
            "spec"     : req.spec_name,
            "step"     : 0,
            "max_steps": req.max_steps,
            "metrics"  : [],
            "error"    : None,
            "started"  : time.time(),
            "finished" : None,
        })

    try:
        import torch
        from slm.model   import build_model
        from slm.trainer import SLMTrainer, TrainConfig

        spec_name = req.spec_name.upper()
        tok       = make_tokenizer(spec_name)
        model, mcfg = build_model(spec_name)

        cfg = TrainConfig(
            spec_name       = spec_name,
            batch_size      = req.batch_size,
            learning_rate   = req.learning_rate,
            max_steps       = req.max_steps,
            corruption_prob = req.corruption_prob,
            checkpoint_dir  = str(ROOT / "checkpoints"),
            device          = "cuda" if torch.cuda.is_available() else "cpu",
        )

        def progress_cb(step, metrics):
            with _train_lock:
                _train_state["step"] = step
                _train_state["metrics"].append({"step": step, **metrics})

        trainer = SLMTrainer(model, mcfg, tok, engine, seeder, cfg)
        trainer.train(progress_callback=progress_cb)

        # Cache the freshly trained model
        model.eval()
        _model_cache[spec_name] = (model, mcfg)

    except Exception as e:
        log.exception("Training error")
        with _train_lock:
            _train_state["error"] = str(e)
    finally:
        with _train_lock:
            _train_state["running"]  = False
            _train_state["finished"] = time.time()


@app.post("/api/train")
async def start_training(req: TrainRequest, bg: BackgroundTasks):
    if _train_state["running"]:
        raise HTTPException(409, "Training already in progress")
    bg.add_task(_run_training, req)
    return {"status": "started", "spec_name": req.spec_name, "max_steps": req.max_steps}


@app.get("/api/train/status")
async def train_status():
    with _train_lock:
        state = dict(_train_state)
    if state["started"]:
        state["elapsed_s"] = round(time.time() - state["started"], 1)
    return state


# ─────────────────────── Config Overrides ────────────────────────────────────

@app.post("/api/config/override")
async def set_override(req: OverrideRequest):
    """
    Set a runtime field-level rule override.
    Example body:
    {
      "spec_name": "ACH_NACHA",
      "record_type": "RT6",
      "field_name": "Amount",
      "overrides": { "allowed": ["0000000100", "0000001000"] }
    }
    """
    engine.set_custom_rule(
        req.spec_name.upper(),
        req.record_type.upper(),
        req.field_name,
        req.overrides,
    )
    return {"status": "ok", "message": f"Override applied for {req.field_name}"}


@app.delete("/api/config/override")
async def reset_overrides(req: ResetRequest):
    engine.reset_custom_rules(
        req.spec_name.upper(),
        req.record_type.upper() if req.record_type else None,
    )
    return {"status": "ok", "message": "Overrides reset"}


# ─────────────────────── Root / Docs Redirect ────────────────────────────────

@app.get("/")
async def root():
    """Serve frontend index or redirect to docs."""
    idx = ROOT / "frontend" / "index.html"
    if idx.exists():
        from fastapi.responses import FileResponse
        return FileResponse(str(idx))
    return {"message": "FinancialSLM API", "docs": "/docs"}


# ─────────────────────── Entrypoint ──────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host       = "0.0.0.0",
        port       = 8000,
        reload     = False,
        log_level  = "info",
    )
