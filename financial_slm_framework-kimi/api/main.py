"""
financial_slm_framework/api/main.py
FastAPI backend for managing SLM inference and in-memory database state.
Lightweight, offline-capable server.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import torch
import json
import tempfile

from config import spec_store, initialize_all_specs
from config.store import FileSpec, RecordSpec, FieldRule, FieldType, PaddingType
from slm_core.model import FinancialSLM
from slm_core.tokenizer import FinancialTokenizer
from slm_core.trainer import FinancialSLMTrainer
from validation import FinancialValidator, ValidationStatus
from generation import FinancialGenerator


# Initialize specifications on startup
initialize_all_specs()

app = FastAPI(
    title="Financial SLM Framework",
    description="Standalone Small Language Model for financial file parsing, validation, and generation",
    version="1.0.0"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model and tokenizer instances
_model = None
_tokenizer = None
_validator = None
_generator = None


def get_model():
    """Lazy-load the SLM model."""
    global _model
    if _model is None:
        vocab_size = _tokenizer.vocab_size if _tokenizer else 300
        _model = FinancialSLM(
            vocab_size=vocab_size,
            d_model=256,
            n_layers=6,
            n_heads=8,
            d_ff=1024,
            max_seq_len=2048
        )
    return _model


def get_tokenizer():
    """Lazy-load the tokenizer."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = FinancialTokenizer(max_record_types=50)
    return _tokenizer


def get_validator():
    """Lazy-load the validator."""
    global _validator
    if _validator is None:
        _validator = FinancialValidator(
            spec_store=spec_store,
            model=get_model(),
            tokenizer=get_tokenizer()
        )
    return _validator


def get_generator():
    """Lazy-load the generator."""
    global _generator
    if _generator is None:
        _generator = FinancialGenerator(
            spec_store=spec_store,
            model=get_model(),
            tokenizer=get_tokenizer()
        )
    return _generator


# Pydantic models
class GenerateRequest(BaseModel):
    spec_id: str
    num_records: int = 10
    use_slm: bool = False
    seed: Optional[int] = None


class ValidateRequest(BaseModel):
    content: str
    spec_id: str
    filename: str = "uploaded_file"


class TrainRequest(BaseModel):
    spec_id: str
    num_samples: int = 1000
    epochs: int = 5
    batch_size: int = 16


class SpecInfo(BaseModel):
    spec_id: str
    name: str
    description: str
    version: str
    record_types: List[Dict[str, Any]]


# API Routes

@app.get("/")
async def root():
    return {"message": "Financial SLM Framework API", "version": "1.0.0", "status": "running"}


@app.get("/api/specs", response_model=List[SpecInfo])
async def list_specs():
    """List all available financial specifications."""
    specs = []
    for spec_id in spec_store.list_specs():
        spec = spec_store.get_spec(spec_id)
        record_types = []
        for rt, rs in spec.record_specs.items():
            record_types.append({
                "code": rt,
                "id": rs.record_type_id,
                "name": rs.name,
                "length": rs.total_length,
                "mandatory": rs.mandatory
            })
        specs.append(SpecInfo(
            spec_id=spec.spec_id,
            name=spec.name,
            description=spec.description,
            version=spec.version,
            record_types=record_types
        ))
    return specs


@app.get("/api/specs/{spec_id}")
async def get_spec_detail(spec_id: str):
    """Get detailed specification information."""
    spec = spec_store.get_spec(spec_id)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"Specification '{spec_id}' not found")

    return spec.to_dict()


@app.post("/api/validate")
async def validate_file(request: ValidateRequest):
    """
    Validate file content against a specification.
    Returns detailed validation results.
    """
    validator = get_validator()

    result = validator.validate_file(
        file_content=request.content,
        spec_id=request.spec_id,
        filename=request.filename
    )

    # Convert to JSON-serializable format
    records_json = []
    for rec in result.records:
        records_json.append({
            "record_type": rec.record_type,
            "record_number": rec.record_number,
            "status": rec.status.value,
            "parsed_fields": rec.parsed_fields,
            "results": [
                {
                    "field_name": r.field_name,
                    "position": r.position,
                    "expected": r.expected,
                    "actual": r.actual,
                    "message": r.message,
                    "severity": r.severity.value,
                    "rule_type": r.rule_type
                }
                for r in rec.results
            ]
        })

    return {
        "spec_id": result.spec_id,
        "filename": result.filename,
        "overall_status": result.overall_status.value,
        "checksum_valid": result.checksum_valid,
        "summary": result.summary,
        "records": records_json
    }


@app.post("/api/validate/upload")
async def validate_upload(
    file: UploadFile = File(...),
    spec_id: str = Form(...)
):
    """Validate an uploaded file."""
    content = await file.read()
    try:
        text_content = content.decode('utf-8')
    except UnicodeDecodeError:
        text_content = content.decode('latin-1')

    request = ValidateRequest(
        content=text_content,
        spec_id=spec_id,
        filename=file.filename
    )
    return await validate_file(request)


@app.post("/api/generate")
async def generate_file(request: GenerateRequest):
    """
    Generate a syntactically correct test file.
    """
    generator = get_generator()

    try:
        generated_content = generator.generate_file(
            spec_id=request.spec_id,
            num_records=request.num_records,
            use_slm=request.use_slm,
            seed=request.seed
        )

        # Also validate the generated file
        validator = get_validator()
        validation = validator.validate_file(
            file_content=generated_content,
            spec_id=request.spec_id,
            filename="generated_file"
        )

        return {
            "spec_id": request.spec_id,
            "content": generated_content,
            "num_records": request.num_records,
            "validation_status": validation.overall_status.value,
            "checksum_valid": validation.checksum_valid
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/train")
async def train_model(request: TrainRequest):
    """
    Train the SLM on synthetic specification data.
    """
    from slm_core.trainer import FinancialDataset
    from torch.utils.data import DataLoader
    import random

    spec = spec_store.get_spec(request.spec_id)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"Specification '{request.spec_id}' not found")

    # Generate synthetic training data
    generator = get_generator()
    records = []

    for _ in range(request.num_samples):
        content = generator.generate_file(
            spec_id=request.spec_id,
            num_records=random.randint(1, 20),
            use_slm=False
        )

        lines = content.split('\n')
        for line in lines:
            if line.strip():
                rt = line[0] if line else ""
                rs = spec.get_record_spec(rt)
                rt_id = rs.record_type_id if rs else 0
                records.append({
                    'text': line,
                    'record_type': rt_id,
                    'is_valid': 1
                })

    # Create dataset and dataloader
    tokenizer = get_tokenizer()
    dataset = FinancialDataset(records, tokenizer, max_seq_len=512, augment=True)
    dataloader = DataLoader(dataset, batch_size=request.batch_size, shuffle=True)

    # Train
    model = get_model()
    trainer = FinancialSLMTrainer(model, tokenizer)

    save_dir = f"./checkpoints/{request.spec_id}"
    os.makedirs(save_dir, exist_ok=True)

    trainer.train(
        train_dataloader=dataloader,
        epochs=request.epochs,
        save_dir=save_dir,
        save_every=1
    )

    return {
        "message": "Training completed",
        "spec_id": request.spec_id,
        "num_samples": len(records),
        "epochs": request.epochs,
        "final_train_loss": trainer.history['train_loss'][-1] if trainer.history['train_loss'] else None,
        "checkpoint_dir": save_dir
    }


@app.get("/api/model/status")
async def model_status():
    """Get current model status."""
    model = get_model()
    tokenizer = get_tokenizer()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "model_loaded": True,
        "architecture": "FinancialTransformer",
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "vocab_size": tokenizer.vocab_size,
        "device": str(next(model.parameters()).device)
    }


@app.post("/api/model/load")
async def load_model_checkpoint(checkpoint_path: str):
    """Load a model checkpoint."""
    try:
        model = get_model()
        trainer = FinancialSLMTrainer(model, get_tokenizer())
        trainer.load_checkpoint(checkpoint_path)
        return {"message": "Checkpoint loaded successfully", "path": checkpoint_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load checkpoint: {str(e)}")


@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "specs_loaded": len(spec_store.list_specs())}


# Mount static files (frontend)
frontend_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
