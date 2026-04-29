# FinSLM — Financial Small Language Model
### ACH NACHA & VISA VCF Generator · Validator · Intelligence Platform

---

## Overview

FinSLM is a production-ready Python system that trains a **custom Small Language Model (SLM)** on synthetic ACH NACHA and VISA VCF financial files, enabling:

| Capability | Description |
|---|---|
| **File Generation** | Rule-based (always available) + SLM-powered synthesis |
| **Full Validation** | NACHA spec compliance, VISA VCF spec, Luhn checks, ABA routing validation |
| **Model Training** | Transformer (PyTorch) or Bigram fallback (numpy) |
| **Web UI** | Dark fintech dashboard — upload, validate, generate, train |
| **REST API** | JSON endpoints for all operations |

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Flask Web Application                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐  │
│  │ /generate│  │/validate │  │  /train  │  │  /spec  │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └─────────┘  │
└───────┼──────────────┼─────────────┼────────────────────-─┘
        │              │             │
        ▼              ▼             ▼
┌───────────┐   ┌────────────┐  ┌──────────────────────────┐
│Data Layer │   │ Validators │  │     Model Layer           │
│           │   │            │  │                           │
│ACHGenerator   │ACHValidator│  │  FinancialSLM             │
│VCFGenerator   │VCFValidator│  │  (Transformer, ~4M params)│
│           │   │            │  │       ─── OR ───          │
│500+ rules │   │NACHA spec  │  │  BigramSLM                │
│Real routing   │VISA spec   │  │  (numpy fallback, fast)   │
│Luhn PAN   │   │ABA routing │  │                           │
│SEC codes  │   │Luhn check  │  │  FinancialTokenizer       │
└───────────┘   └────────────┘  │  (char-level, vocab ~100) │
                                └──────────────────────────┘
```

---

## Project Structure

```
slm_ach_vcf/
├── app.py                    # Flask web application + REST API
├── trainer.py                # Training pipeline (auto-detects backend)
├── requirements.txt
│
├── model/
│   ├── transformer.py        # GPT-style decoder transformer (~1–10M params)
│   ├── tokenizer.py          # Character-level tokenizer with special tokens
│   └── bigram_fallback.py    # Numpy n-gram model (runs without PyTorch)
│
├── validators/
│   ├── ach_validator.py      # Full NACHA ACH validator (94-char records)
│   └── vcf_validator.py      # VISA VCF validator (pipe-delimited + fixed-width)
│
├── data/
│   └── generator.py          # Synthetic ACH + VCF file generator
│
├── templates/
│   └── index.html            # Dark fintech web UI
│
└── trained_models/           # Saved model checkpoints (.pt or .pkl)
```

---

## Quick Start

### 1. Install Dependencies

```bash
# Minimal (numpy backend — no GPU needed, trains fast)
pip install flask numpy werkzeug

# Full (Transformer backend — much better generation quality)
pip install flask numpy werkzeug torch
```

### 2. Start the Web App

```bash
cd slm_ach_vcf
python app.py
# Open http://localhost:5000
```

### 3. Train from Command Line

```bash
# Train both models (Transformer if torch available, Bigram otherwise)
python trainer.py --type both --epochs 10 --files 500

# Train only ACH model, nano size (fast)
python trainer.py --type ACH --config nano --epochs 5

# Train only VCF model, medium quality
python trainer.py --type VCF --config medium --epochs 20 --files 1000
```

---

## ACH NACHA Validation Rules

The validator enforces the complete **NACHA Operating Rules**:

### Record Structure
| Record Type | Code | Length | Validates |
|---|---|---|---|
| File Header | `1` | 94 chars | Priority code, routing format, YYMMDD date, record size=094, blocking=10 |
| Batch Header | `5` | 94 chars | Service class (200/220/225), SEC code, ODFI routing (8 digits) |
| Entry Detail | `6` | 94 chars | Transaction code, ABA check digit (Luhn-like), amount (10 digits), trace |
| Addenda | `7` | 94 chars | Addenda type code, sequence number, entry link |
| Batch Control | `8` | 94 chars | Entry count, entry hash (sum of routing numbers mod 10^10), debit/credit totals |
| File Control | `9` | 94 chars | Batch count, block count (records ÷ 10), entry hash, grand totals |

### Cross-Record Validation
- **Entry Hash**: Sum of all RDFI routing numbers (mod 10^10) must match batch and file control
- **Debit/Credit Totals**: Summed from transaction codes, must match batch control
- **Record Count**: Must be a multiple of 10 (blocking factor)
- **ABA Routing Check Digit**: `(3×d1 + 7×d2 + d3) + (3×d4 + 7×d5 + d6) + (3×d7 + 7×d8 + d9) ≡ 0 mod 10`

### Valid SEC Codes
`PPD`, `CCD`, `CTX`, `WEB`, `TEL`, `COR/NOC`, `RCK`, `ARC`, `BOC`, `POP`, `XCK`, `IAT`, `ENR`, `MTE`, `SHR`

---

## VISA VCF Validation Rules

Supports both **pipe-delimited** (`|`) and **fixed-width** VCF formats.

### Field Validation
| Field | Position | Format | Validation |
|---|---|---|---|
| Transaction Code | 0 | 2N | Must be in `{05, 06, 10, 12, 15, 25, 26, ...}` |
| PAN | 1 | 13–19N | Luhn algorithm check |
| Processing Code | 2 | 6N | ISO processing code lookup |
| Amount | 3 | N.NN | Numeric, non-negative, ≤ 999,999,999.99 |
| Currency Code | 4 | 3N | ISO 4217 lookup (840=USD, 978=EUR, etc.) |
| Transaction DateTime | 5 | 14N | `YYYYMMDDHHmmSS` format |
| MCC | 6 | 4N | Merchant Category Code (4 digits) |
| POS Entry Mode | 7 | 2N | `{00, 01, 02, 05, 07, 90, 91, ...}` |
| Response Code | 8 | 2AN | `{00=Approved, 05=DNH, 51=NSF, ...}` |
| Auth Code | 9 | 6AN | 6 alphanumeric characters |
| Merchant ID | 10 | 15AN | 1–15 alphanumeric |
| ARN | 14 | 23N | Acquirer Reference Number (23 digits) |

### Luhn Algorithm (PAN Validation)
```python
digits = [int(d) for d in pan]
total = sum(digits[-1::-2])  # Odd positions
for d in digits[-2::-2]:    # Even positions doubled
    total += sum(divmod(d * 2, 10))
assert total % 10 == 0
```

---

## SLM Architecture (Transformer)

### Model Sizes
| Config | d_model | Heads | Layers | d_ff | Params |
|---|---|---|---|---|---|
| `nano` | 128 | 4 | 4 | 512 | ~1M |
| `small` | 256 | 8 | 6 | 1024 | ~4M |
| `medium` | 384 | 6 | 8 | 1536 | ~10M |

### Architecture Details
- **Decoder-only** GPT-style transformer
- **Weight tying**: Token embedding = LM head (saves ~vocab_size × d_model params)
- **Causal masking**: Prevents future token leakage
- **Positional embedding**: Learned (not sinusoidal)
- **Sampling**: Top-k + Nucleus (top-p) sampling for controlled generation
- **Tokenizer**: Character-level, vocab ~100 chars (all printable ASCII + special tokens)

### Training
- Optimizer: AdamW with β₁=0.9, β₂=0.95
- Schedule: Cosine annealing with linear warmup
- Gradient clipping: 1.0
- Loss: Cross-entropy on next-character prediction

---

## REST API

### POST /api/generate
```json
{
  "file_type": "ACH",
  "use_model": false,
  "num_batches": 2,
  "entries_per_batch": 5,
  "sec_code": "PPD"
}
```
**Response**: Generated file content + metadata

### POST /api/validate
Multipart form with `file` + `file_type` field.
**Response**: Full validation report with errors, warnings, statistics

### POST /api/validate/text
```json
{
  "content": "1 021000021...",
  "file_type": "ACH"
}
```

### POST /api/train
```json
{
  "file_type": "ACH",
  "model_config": "small",
  "n_files": 500,
  "max_epochs": 10,
  "batch_size": 8,
  "learning_rate": 0.0003
}
```

### GET /api/train/status
Polls training progress: `{ "running": true, "progress": 67, "message": "Epoch 5/10..." }`

### GET /api/models
Returns available trained models and their status.

### GET /api/spec
Returns ACH and VCF specification summaries.

---

## Best Solution Recommendations

### For Production Use

1. **Use PyTorch Transformer** (not Bigram): Train the `small` or `medium` model on 1000–5000 synthetic files. The Transformer learns real structural patterns; the Bigram is for prototyping only.

2. **Fine-tune on Real Files**: If you have actual ACH/VCF files (even anonymized/masked), fine-tune the pre-trained model on them. Character-level models transfer well.

3. **Validate Generated Files**: Always pipe generated output through the validator before use. The rule-based generator (no model) produces 100% valid files; the SLM may occasionally deviate.

4. **Scale Training**: On a GPU (A100), train the `medium` model for 50 epochs on 5000+ files. Expected perplexity: < 1.1 (near-perfect ACH structure learned).

5. **Hybrid Approach**: Use the rule-based generator as the primary source and the SLM for augmentation/variation. The SLM adds statistical realism to amounts, company names, routing patterns.

### Recommended Hardware
| Config | Hardware | Training Time | Quality |
|---|---|---|---|
| Bigram (numpy) | Any CPU | 1–2 min | Basic |
| Nano Transformer | 8-core CPU | 15–30 min | Good |
| Small Transformer | GPU (T4) | 5–10 min | Very Good |
| Medium Transformer | GPU (A100) | 10–20 min | Excellent |

---

## Security Notes

- PAN numbers in generated VCF files use Luhn-valid but **synthetic card numbers**
- Routing numbers use real ABA routing numbers (public data) for realism
- **Do not use generated files in production payment systems**
- The validator does not store or log file contents
