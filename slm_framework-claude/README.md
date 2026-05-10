# FinancialSLM — Specification Intelligence Engine

> A custom Small Language Model (SLM) built from scratch for parsing, validating, and generating financial file formats — with **zero external LLM API calls**, fully offline, privacy-first.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                        FinancialSLM Stack                            │
│                                                                      │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────────────┐    │
│  │  Frontend   │    │  FastAPI     │    │  SLM Core Engine     │    │
│  │  SPA        │◄──►│  Backend     │◄──►│                      │    │
│  │  (HTML/JS)  │    │  api/main.py │    │  ┌────────────────┐  │    │
│  └─────────────┘    └──────────────┘    │  │  Tokenizer     │  │    │
│                                         │  │  (fixed-width) │  │    │
│  ┌─────────────────────────────────┐    │  ├────────────────┤  │    │
│  │  In-Memory Config Engine        │    │  │  Transformer   │  │    │
│  │  (Singleton Source of Truth)    │◄──►│  │  4-6 layers    │  │    │
│  │                                 │    │  ├────────────────┤  │    │
│  │  ACH_NACHA  │ VISA_VCF  │  GL  │    │  │  Gen Head +    │  │    │
│  │  RT1-RT9    │ VH/DT/VF  │JH/JE │    │  │  Val Head      │  │    │
│  └─────────────────────────────────┘    │  └────────────────┘  │    │
│                                         └──────────────────────┘    │
│  ┌──────────────┐    ┌───────────────┐                              │
│  │  Validator   │    │  Generator    │                              │
│  │  Rule+Model  │    │  Field-Mask   │                              │
│  │  Hybrid      │    │  Constrained  │                              │
│  └──────────────┘    └───────────────┘                              │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
slm_framework/
│
├── run.py                    # CLI entry point
├── requirements.txt
│
├── slm/                      # Core SLM engine
│   ├── tokenizer.py          # Domain-specific fixed-width tokenizer
│   ├── model.py              # Custom Transformer (dual-head)
│   ├── trainer.py            # Dual-objective training pipeline
│   ├── validator.py          # Hybrid rule + model validator
│   └── generator.py          # Constrained autoregressive generation
│
├── memory/                   # In-memory configuration engine
│   ├── config_engine.py      # Singleton Source of Truth
│   └── seeder.py             # Mock data injection pipeline
│
├── specs/                    # Financial format definitions
│   ├── ach_nacha.py          # ACH NACHA field schemas (RT1–RT9)
│   ├── visa_vcf.py           # VISA VCF + General Ledger schemas
│   └── general_ledger.py     # GL re-export shim
│
├── api/                      # FastAPI backend
│   └── main.py               # All routes: validate / generate / train
│
├── frontend/
│   └── index.html            # Single-page application
│
├── checkpoints/              # Auto-created; model weights saved here
├── tests/
│   └── test_suite.py         # 40+ unit + integration tests
└── data/samples/             # Sample files for testing
```

---

## Supported Specifications

| Spec | Record Types | Line Width | Key Rules |
|------|-------------|-----------|-----------|
| **ACH NACHA** | RT1, RT5, RT6, RT7, RT8, RT9 | 94 chars | Mod-10 routing check, hash totals, blocking factor of 10 |
| **VISA VCF** | VH, DT, TR, VF | 80 chars | Volume header/trailer, transaction codes |
| **General Ledger** | JH, JE, GL | 120 chars | Double-entry balance (Dr = Cr), period codes |

---

## Quick Start

### 1. Install Dependencies

```bash
cd slm_framework
pip install -r requirements.txt
```

PyTorch is required for model training/inference. Rule-based generation and validation work without it.

### 2. Generate a Sample File

```bash
# Generate ACH NACHA file with 5 entry records
python run.py generate --spec ACH_NACHA --entries 5 --out test.ach

# Generate VISA VCF file
python run.py generate --spec VISA_VCF --entries 10 --out test.vcf

# Generate General Ledger journal
python run.py generate --spec GENERAL_LEDGER --entries 8 --out test.gl
```

### 3. Validate a File

```bash
# Validate with full error output
python run.py validate --spec ACH_NACHA --file test.ach

# Summary only
python run.py validate --spec ACH_NACHA --file test.ach --summary

# Export JSON report
python run.py validate --spec ACH_NACHA --file test.ach --json-out report.json
```

Exit code 0 = fully valid, exit code 1 = errors found.

### 4. Explore Spec Definitions

```bash
# List all record types for a spec
python run.py explore --spec ACH_NACHA

# Show field schema for a specific record type
python run.py explore --spec ACH_NACHA --rt RT6
```

Output:
```
ACH NACHA File Format

  Line length   : 94 chars
  Record types  : RT1, RT5, RT6, RT7, RT8, RT9

ACH_NACHA / RT6 — Field Schema

Field Name                          Start  End  Len  Type           Req
───────────────────────────────────────────────────────────────────────
Record Type Code                        1    1    1  RECORD_TYPE    Yes
Transaction Code                        2    3    2  NUMERIC        Yes
RDFI Routing Transit                    4   11    8  ROUTING        Yes
Check Digit                            12   12    1  NUMERIC        Yes
RDFI Account Number                    13   29   17  ACCOUNT        Yes
Amount                                 30   39   10  AMOUNT         Yes
Individual ID Number                   40   54   15  ALPHANUMERIC   No
Individual Name                        55   76   22  ALPHANUMERIC   Yes
Discretionary Data                     77   78    2  ALPHANUMERIC   No
Addenda Record Indicator               79   79    1  NUMERIC        Yes
Trace Number                           80   94   15  NUMERIC        Yes
```

### 5. Train the SLM

```bash
# Train on ACH NACHA (2000 steps, ~5 min on CPU)
python run.py train --spec ACH_NACHA --steps 2000 --batch 16

# Longer training run
python run.py train --spec GENERAL_LEDGER --steps 10000 --batch 32 --lr 1e-4
```

Training generates infinite synthetic records from the ConfigEngine — no external datasets needed.

### 6. Start the Web Interface

```bash
python run.py serve --port 8000
```

Open `http://localhost:8000` for the SPA, or `http://localhost:8000/docs` for the API explorer.

### 7. Run Tests

```bash
python run.py test
# or
python -m pytest tests/test_suite.py -v
```

---

## Model Architecture

### FinancialSLM Transformer

```
Input Line (94 chars for ACH)
         │
         ▼
┌──────────────────────────────┐
│  Triple Embedding            │
│  ┌────────┐ ┌───────────┐   │
│  │ Char   │ │ Field-Slot │   │   ← Which named field this char belongs to
│  │ Embed  │ │ Embed (PE) │   │
│  └────────┘ └───────────┘   │
│  ┌──────────────────────┐   │
│  │ Record-Type Embed     │   │   ← Broadcast: same RT for all chars in line
│  └──────────────────────┘   │
│  → Concat → Linear proj      │
└──────────────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│  Sinusoidal Positional Enc.  │   ← Column-aware (col 4 = always same field)
└──────────────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│  Transformer Blocks × N      │
│  ┌─────────────────────────┐ │
│  │ Multi-Head Attention    │ │   Mode: causal (generate) / full (validate)
│  │ Feed-Forward (GELU)     │ │
│  │ Layer Norm (pre-norm)   │ │
│  └─────────────────────────┘ │
└──────────────────────────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌─────────────────────┐
│ Gen    │ │ Val Head            │
│ Head   │ │ AdaptiveAvgPool1d   │
│        │ │ → per-field (0/1)   │
│ (B,T,V)│ │ → global confidence │
└────────┘ └─────────────────────┘
```

**Model sizes by spec:**

| Spec | Layers | d_model | Heads | ~Params |
|------|--------|---------|-------|---------|
| ACH NACHA | 4 | 128 | 4 | ~850K |
| VISA VCF | 4 | 128 | 4 | ~850K |
| General Ledger | 6 | 192 | 6 | ~2.1M |

### Training Objectives

```
Loss = α × CLM_Loss + β × VAL_Loss

CLM_Loss: CrossEntropy(next-char prediction)
          → Teaches syntactic structure of records

VAL_Loss: BCEWithLogits(per-field binary validity)
          → Uses deliberately corrupted records (30% of training)
          → Teaches field-level constraint awareness
```

---

## In-Memory Config Engine

The `ConfigEngine` is the **Source of Truth** for all spec rules:

```python
from memory.config_engine import ConfigEngine

engine = ConfigEngine()   # Singleton — always same instance

# Query field rules
fields = engine.get_fields("ACH_NACHA", "RT6")
# → [{name: "Amount", start: 30, end: 39, field_type: "AMOUNT", required: True, ...}]

# Runtime override (e.g., for test-data constraints)
engine.set_custom_rule(
    "ACH_NACHA", "RT6", "Amount",
    {"allowed": ["0000000100", "0000001000"]}
)

# Export all rules as JSON
rules = engine.export_rules("ACH_NACHA")
```

---

## Constrained Generation

The `ConstraintResolver` ensures every generated character is spec-legal:

```
For each character position col during generation:
  1. Query ConfigEngine: which field does col belong to?
  2. Determine allowed character class for that field type
  3. Zero-out all illegal token logits (hard mask)
  4. Sample from constrained distribution

Result: Syntactically perfect output even from untrained model
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/validate` | Upload file for validation |
| POST | `/api/validate/text` | Validate raw text in JSON body |
| POST | `/api/generate` | Generate synthetic file |
| POST | `/api/generate/record` | Generate single annotated record |
| GET | `/api/specs` | List all supported specs |
| GET | `/api/specs/{name}/rules` | Export full field rules |
| GET | `/api/specs/{name}/record-types` | Field tables per record type |
| POST | `/api/train` | Start background training |
| GET | `/api/train/status` | Poll training progress |
| POST | `/api/config/override` | Set runtime field override |
| DELETE | `/api/config/override` | Reset overrides |
| GET | `/api/health` | Health check |

---

## Privacy & Compliance

- **Zero external calls**: No OpenAI, Anthropic, HuggingFace, or any cloud LLM API
- **Fully offline**: All inference runs locally; financial data never leaves your machine
- **No persistent storage**: The ConfigEngine uses in-memory Python dicts (no database)
- **Deterministic**: With a fixed seed, generation is fully reproducible
- **Auditable**: All field rules are explicit Python code in `specs/` — no black-box weights for validation logic

---

## Extending the Framework

### Add a New Spec

1. Define field schemas in `specs/yourspec.py` using `FieldDescriptor`
2. Register in `memory/config_engine.py` `_bootstrap()` method
3. Add line-length to `FinancialTokenizer.LINE_LENGTH_MAP`
4. Add seeding hints in `memory/seeder.py` `_by_type()`
5. Add structural checks in `slm/validator.py` `_check_structure()`

### Add a Custom Field Type

1. Add to `TokenType` enum in `slm/tokenizer.py`
2. Add generation logic in `memory/seeder.py`
3. Add validation rule in `slm/validator.py` `_check_line_rules()`
4. Add constraint class in `slm/generator.py` `ConstraintResolver`
