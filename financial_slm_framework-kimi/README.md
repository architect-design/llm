# Financial SLM Framework

A **standalone, specialized Small Language Model (SLM)** built from scratch for parsing, validating, and generating complex financial file formats — with **zero external LLM API dependencies**.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     FRONTEND (SPA)                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ Validation   │  │ Generation   │  │ Model Status &       │   │
│  │ Console      │  │ Studio       │  │ Training Controls    │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTP/REST
┌──────────────────────────▼──────────────────────────────────────┐
│                     FASTAPI BACKEND                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ /validate    │  │ /generate    │  │ /train               │   │
│  │ /validate/   │  │              │  │ /model/status        │   │
│  │ upload       │  │              │  │ /model/load          │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                     SLM CORE ENGINE                              │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  FinancialSLM (PyTorch Transformer)                       │   │
│  │  - Character-level embeddings                             │   │
│  │  - Positional encoding for fixed-width awareness          │   │
│  │  - Causal self-attention for auto-regressive generation   │   │
│  │  - Dual heads: Generation + Validation                    │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ Financial    │  │ Financial    │  │ FinancialSLM         │   │
│  │ Tokenizer    │  │ Dataset      │  │ Trainer              │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│              CONFIG STORE (SOURCE OF TRUTH)                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ ACH NACHA    │  │ VISA VCF     │  │ General Ledger       │   │
│  │ Specification│  │ Specification│  │ Specification        │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
│                                                                  │
│  Thread-safe singleton with O(1) field rule lookups             │
│  Maps: spec_id -> record_type -> field_name -> FieldRule        │
└─────────────────────────────────────────────────────────────────┘
```

## Key Features

### Domain-Specific Tokenizer
- Character-level vocabulary optimized for financial data
- Special tokens for record types (`<RT_1>`, `<RT_2>`...) and field boundaries (`<FB>`, `<RB>`)
- Fixed-width record encoding with explicit position markers
- Handles padding, delimiters, and structural syntax natively

### Dual-Head Architecture
- **Generation Head**: Auto-regressive next-character prediction for file generation
- **Validation Head**: Classifies records as VALID / INVALID_SYNTAX / INVALID_SEMANTIC
- **Field Boundary Head**: Detects structural boundaries in fixed-width records

### Constrained Generation
- Rule-based generation with specification-aware mock data seeding
- SLM-guided generation with real-time character constraints
- Automatic checksum and total computation for control records

### Real-Time Validation
- Structural validation (record length, order, padding)
- Field-level validation (type, format, allowed values, regex)
- Semantic validation (checksums, totals, cross-record consistency)
- SLM-based anomaly detection

## Supported Formats

| Format | Spec ID | Record Types | Record Length |
|--------|---------|-------------|---------------|
| ACH NACHA | `ach_nacha` | File Header, Batch Header, Entry Detail, Batch Control, File Control | 94 chars |
| VISA VCF | `visa_vcf` | Header, Detail, Trailer | 80 chars |
| General Ledger | `general_ledger` | Header, Detail, Trailer | 120 chars |

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Demo
```bash
python demo.py
```

### 3. Start the API Server
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Open the Web Interface
Navigate to `http://localhost:8000` in your browser.

## API Endpoints

### Validation
```bash
POST /api/validate
{
  "content": "101 091000019...",
  "spec_id": "ach_nacha",
  "filename": "myfile.txt"
}

POST /api/validate/upload  # multipart/form-data
```

### Generation
```bash
POST /api/generate
{
  "spec_id": "ach_nacha",
  "num_records": 10,
  "use_slm": false,
  "seed": 42
}
```

### Training
```bash
POST /api/train
{
  "spec_id": "ach_nacha",
  "num_samples": 1000,
  "epochs": 5,
  "batch_size": 16
}
```

### Model Status
```bash
GET /api/model/status
GET /api/health
GET /api/specs
```

## Project Structure

```
financial_slm_framework/
├── api/
│   ├── __init__.py
│   └── main.py              # FastAPI server
├── config/
│   ├── __init__.py
│   ├── store.py             # In-memory specification store (Source of Truth)
│   └── specs.py             # Pre-built ACH, VCF, GL specifications
├── slm_core/
│   ├── __init__.py
│   ├── model.py             # FinancialSLM (Transformer) + FinancialLSTM
│   ├── tokenizer.py         # Domain-specific character-level tokenizer
│   └── trainer.py           # Training pipeline with data augmentation
├── validation/
│   ├── __init__.py
│   └── validator.py         # Multi-layer validation engine
├── generation/
│   ├── __init__.py
│   └── generator.py         # Constrained generation + mock data seeding
├── frontend/
│   ├── index.html           # SPA shell
│   ├── styles.css           # Dark theme UI
│   └── app.js               # Vanilla JS frontend logic
├── demo.py                  # Interactive demonstration script
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Compliance & Privacy

- **Zero External Dependencies**: No calls to OpenAI, Anthropic, Hugging Face, or any cloud LLM API
- **Fully Offline**: The entire ecosystem runs locally without internet connectivity
- **Financial Data Security**: All file processing happens in-memory; no data leaves the local environment
- **Custom Architecture**: Built from scratch with PyTorch — no pre-trained model weights required

## Training

The model can be trained on synthetic data generated from the specification store:

```python
from slm_core.model import FinancialSLM
from slm_core.tokenizer import FinancialTokenizer
from slm_core.trainer import FinancialSLMTrainer

# Initialize
tokenizer = FinancialTokenizer()
model = FinancialSLM(vocab_size=tokenizer.vocab_size)
trainer = FinancialSLMTrainer(model, tokenizer)

# Train
trainer.train(train_dataloader, epochs=10)
```

The trainer includes:
- Data augmentation (random corruptions for validation head training)
- Combined loss: generation loss + validation loss
- Gradient clipping and checkpointing

## Adding New Specifications

```python
from config.store import FileSpec, RecordSpec, FieldRule, FieldType, PaddingType, spec_store

my_spec = FileSpec(
    spec_id="my_format",
    name="My Custom Format",
    description="...",
    version="1.0"
)

record = RecordSpec(
    record_type_code="H",
    record_type_id=100,
    name="Header",
    total_length=80
)
record.fields = [
    FieldRule("Field1", 0, 10, FieldType.NUMERIC, 10, padding=PaddingType.LEFT_ZERO),
    FieldRule("Field2", 10, 30, FieldType.ALPHANUMERIC, 20, padding=PaddingType.RIGHT_SPACE),
]

my_spec.add_record_spec(record)
spec_store.register_spec(my_spec)
```

## License

MIT License — Built for financial data processing with privacy and compliance in mind.
