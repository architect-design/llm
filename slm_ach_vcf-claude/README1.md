# FinSLM — Financial Small Language Model
### ACH NACHA & VISA VCF · Generator · Validator · ChromaDB Integration · Intelligence Platform

---

## Overview

FinSLM is a production-ready Python system that trains a **custom Small Language Model (SLM)** on ChromaDB-stored transaction data and synthetic ACH NACHA / VISA VCF financial files. It provides a complete pipeline for generating, validating, and learning from structured financial files.

| Capability | Description |
|---|---|
| **ChromaDB-backed Generation** | Reads `PENDING` transactions from ChromaDB collections, assembles spec-compliant NACHA files |
| **Synthetic Generation** | Rule-based ACH & VCF generator — works with no database |
| **Full Validation** | NACHA spec, VISA VCF spec, Luhn PAN, ABA routing check-digit, entry hash, batch totals |
| **SLM Training** | GPT-style Transformer (PyTorch) or Bigram n-gram (numpy) — auto-detected |
| **ChromaDB-aware Training** | Merges ChromaDB corpus files with live-generated and synthetic data; vector similarity selects diverse examples |
| **Web UI** | Wells Fargo–branded white dashboard — Generate, Validate, Train, ChromaDB, Spec |
| **REST API** | 11 JSON endpoints covering all operations |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         Flask Web Application  :5000                         │
│                                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
│  │/generate │  │/validate │  │  /train  │  │  /chroma │  │    /spec     │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────────────┘  │
└───────┼──────────────┼─────────────┼──────────────┼──────────────────────────┘
        │              │             │              │
        ▼              ▼             ▼              ▼
┌────────────────┐  ┌───────────┐  ┌───────────────────────────────────────────┐
│  Data Layer    │  │Validators │  │              Model Layer                  │
│                │  │           │  │                                           │
│ ACHGenerator   │  │ACHValidator  │  FinancialSLM  (Transformer, ~1–10M)    │
│ VCFGenerator   │  │VCFValidator  │       --- OR ---                        │
│ ChromaACH      │  │           │  │  BigramSLM    (numpy fallback)           │
│  Generator     │  │NACHA spec │  │                                           │
│                │  │VISA spec  │  │  FinancialTokenizer  (char-level ~100)   │
│ 30 real ABA    │  │ABA routing│  │                                           │
│ routing numbers│  │Luhn check │  │  ChromaAwareTrainer                      │
│ Luhn-valid PANs│  │Entry hash │  │  (ChromaDB corpus + synthetic blending)  │
└────────────────┘  └───────────┘  └───────────────────────────────────────────┘
        │                                         │
        ▼                                         ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                          ChromaDB Vector Store Layer                         │
│                                                                              │
│  ChromaDBClient  (PersistentClient — data stored in db/chromadb_store/)     │
│                                                                              │
│  ACHRepository  -->  ach_transactions  (denormalised ChromaDB collection)   │
│                       ├── ach_odfi_config         (bank routing config)      │
│                       ├── ach_companies           (originator master)        │
│                       ├── ach_accounts            (RDFI account master)      │
│                       └── ach_transactions        (pending queue)            │
│                                                                              │
│  Audit & Training  --> ach_file_log               (generated file audit)    │
│                         ach_validation_log         (validation history)      │
│                         ach_training_corpus        (SLM training data)       │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
slm_ach_vcf/
│
├── app.py                        # Flask web app + 11 REST endpoints
├── trainer.py                    # Synthetic-only training pipeline
├── requirements.txt
├── README.md
│
├── db/                           # ChromaDB Database Layer
│   ├── __init__.py
│   ├── chroma_client.py          # PersistentClient, offline embedding, health check
│   ├── chroma_schema.py          # Collection metadata contracts (analogous to DDL)
│   ├── ach_repository.py         # All ChromaDB queries, typed domain objects
│   └── chromadb_store/           # On-disk ChromaDB data files (auto-created)
│
├── data/                         # Data Generation
│   ├── generator.py              # Synthetic ACH + VCF file generator
│   ├── chroma_ach_generator.py   # ChromaDB-backed NACHA file builder
│   └── chroma_trainer.py         # ChromaDB-aware SLM training pipeline
│
├── model/                        # SLM Model Layer
│   ├── transformer.py            # GPT-style decoder transformer (~1–10M params)
│   ├── tokenizer.py              # Character-level tokenizer, vocab ~105
│   └── bigram_fallback.py        # N-gram model (no PyTorch needed)
│
├── validators/                   # Validation
│   ├── ach_validator.py          # Full NACHA ACH validator (94-char records)
│   └── vcf_validator.py          # VISA VCF validator (pipe-delimited + fixed-width)
│
├── templates/
│   └── index.html                # Wells Fargo-branded white UI (5 panels)
│
└── trained_models/               # Saved model checkpoints
    ├── ach_model.pt / .pkl       # ACH SLM (transformer or bigram)
    └── vcf_model.pt / .pkl       # VCF SLM (transformer or bigram)
```

---

## Quick Start

### 1. Install Dependencies

```bash
# Minimum — numpy Bigram SLM, no database server required
pip install flask numpy werkzeug

# Add ChromaDB (no server install needed — fully embedded)
pip install flask numpy werkzeug chromadb

# Full stack — Transformer SLM + ChromaDB
pip install flask numpy werkzeug chromadb torch
```

### 2. ChromaDB Store (automatic)

ChromaDB is fully **embedded** — no separate server, no connection string, no credentials.
The store is created automatically at `db/chromadb_store/` on first run.

```bash
# The store directory is created for you — no setup required
python app.py
# ChromaDB store initialised at: slm_ach_vcf/db/chromadb_store/
```

To **seed** initial transaction data for generation, use the web UI's
**ChromaDB → Seed Transactions** button, or call the API:

```bash
curl -X POST http://localhost:5000/api/chroma/seed \
     -H "Content-Type: application/json" \
     -d '{"n": 100, "sec_code": "PPD"}'
```

### 4. Start the Web App

```bash
cd slm_ach_vcf
python app.py
# Open http://localhost:5000
```

### 5. Train from the Command Line

```bash
# Synthetic-only (always works, no database server needed)
python trainer.py --type ACH --config nano --epochs 5
python trainer.py --type VCF --config small --epochs 10

# ChromaDB-aware training (pulls corpus + generates from ChromaDB transactions)
python -c "
from data.chroma_trainer import ChromaAwareTrainer
trainer = ChromaAwareTrainer(
    store_path='db/chromadb_store',
    train_config={
        'model_config':      'small',
        'n_corpus_files':    300,
        'n_chroma_files':    100,
        'n_synthetic_files': 100,
        'max_epochs':        10,
    }
)
results = trainer.train()
print('Best val loss:', results['best_val_loss'])
"
```

---

## ChromaDB Collections

All data is stored in `db/chromadb_store/` — a local persistent ChromaDB directory
that lives inside the project and requires no external server.

### Collections

| Collection | Analogous to | Purpose |
|---|---|---|
| `ach_odfi_config` | ODFI config table | ABA routing, bank name, NACHA File Header fields |
| `ach_companies` | Companies table | Originator — SEC code, service class, company ID |
| `ach_accounts` | Accounts table | Receiver (RDFI) — routing, account number, payee name |
| `ach_transactions` | Transactions table | Pending payments — transaction code, amount in cents, effective date |
| `ach_file_log` | File audit table | Every generated file — batch/entry counts, totals, full NACHA text |
| `ach_validation_log` | Validation log | Validation history — JSON reports, error/warning counts |
| `ach_training_corpus` | Training corpus | ACH file text with TRAIN/VAL/TEST split labels |

### ChromaDB Document Model

Each collection entry has three parts:

```
id        unique string identifier   (e.g. "txn_a4f3b2c1d0e9")
document  primary searchable text    (payee name, file content, etc.)
metadata  dict of filterable fields  (amount_cents, sec_code, status, …)
```

ChromaDB uses the `document` field for vector similarity search (e.g. find ACH files
structurally similar to a seed). Metadata fields drive exact filtering (replacing SQL WHERE).

### Transaction Lifecycle

```
POST /api/chroma/seed  -->  ach_transactions (status=PENDING)
       |
       v
ChromaACHGenerator.generate()
  |-- repo.get_pending_transactions()  [ChromaDB where status=PENDING]
  |-- Build 94-char NACHA records
  |-- repo.log_file()         -->  ach_file_log        (audit trail)
  |-- repo.save_corpus_entry() --> ach_training_corpus  (SLM training)
  |-- repo.mark_batched()     -->  ach_transactions     (status=BATCHED)
```

### Embedding Function

All collections use an **offline hash-based embedding function** (`FinancialHashEmbedding`)
that requires no model download and no network access. It uses character n-gram hashing
(n=2,3,4) to produce 256-dimensional L2-normalised vectors — sufficient for structural
similarity retrieval from the training corpus. Replace with `sentence-transformers` in
production for higher-quality semantic search.

### Amount Storage

Amounts are stored as **integer cents** in the `amount_cents` metadata field
(e.g. `$19.99 → 1999`). The generator zero-pads to 10 digits for the NACHA
implied-decimal field: `0000001999`.

---

## Web UI — Five Panels

The UI uses a **white background with black text and Wells Fargo red (#D71E28) accents**
including buttons, active nav indicators, progress bars, header border, and panel title underlines.

| Panel | Icon | Key Actions |
|---|---|---|
| **Generate** | ⚡ | Rule-based or SLM-powered ACH/VCF file generation; inline copy/download/validate |
| **Validate** | ✓ | Drag-and-drop file upload or paste content; error/warning table with line numbers |
| **Train Model** | 🧠 | Configure model size/epochs; live progress bar and scrolling training log |
| **ChromaDB** | 🗄 | Inspect collections, seed transactions, generate from ChromaDB data, train on corpus |
| **Specification** | 📋 | NACHA record layout reference; VISA VCF field map; transaction code tables |

### ChromaDB Panel Features (legacy reference removed)

- Connection form — host, port, service, username, password, schema, with mock mode toggle
- **Test Connection** — displays driver name, DSN, pool open/busy counts
- **Load Companies** — populates dropdown from `ACH_COMPANIES` table live
- **Generate from ChromaDB** — filter by company, SEC code, effective date, max rows
- **Audit toggle** — writes generated file metadata to `ACH_FILE_LOG`
- **Corpus toggle** — saves file content to `ACH_TRAINING_CORPUS` for future SLM training
- **Recent File Log** — live table showing last 10 generated files with batch/entry counts and totals
- **ChromaDB-aware training** — pulls corpus files first, generates fresh from transactions, blends with synthetic; real-time progress log

---

## ACH NACHA Validation Rules

### Record Structure

| Record | Code | Length | Fields Validated |
|---|---|---|---|
| File Header | `1` | 94 | Priority code `01`, routing format, `YYMMDD` date, record size `094`, blocking factor `10`, format code `1` |
| Batch Header | `5` | 94 | Service class `200/220/225`, SEC code (16 valid), ODFI 8-digit routing, effective date `YYMMDD` |
| Entry Detail | `6` | 94 | Transaction code (20 valid), ABA check digit (3-7-1), 10-digit zero-padded amount, trace number |
| Addenda | `7` | 94 | Addenda type code, sequence number, entry detail sequence link |
| Batch Control | `8` | 94 | Entry/addenda count, entry hash (mod 10¹⁰), debit total, credit total |
| File Control | `9` | 94 | Batch count, block count (records ÷ 10), grand entry hash, grand totals |

### Cross-Record Validation

- **Entry Hash** — Sum of all RDFI routing numbers, last 10 digits, must match Batch Control and File Control exactly
- **Debit/Credit Totals** — Derived from transaction codes per entry; must match batch and file control fields
- **Record Count** — Total records must be a multiple of 10 (blocking factor)
- **ABA Check Digit** — `(3×d₁ + 7×d₂ + d₃) + (3×d₄ + 7×d₅ + d₆) + (3×d₇ + 7×d₈ + d₉) ≡ 0 mod 10`
- **Addenda Indicator** — Entry Detail position 79 must be `1` when an Addenda record follows

### Valid SEC Codes

`PPD`  `CCD`  `CTX`  `WEB`  `TEL`  `COR`  `NOC`  `RCK`  `ARC`  `BOC`  `POP`  `XCK`  `IAT`  `ENR`  `MTE`  `SHR`

### Transaction Code Reference

| Code | Account Type | Direction |
|---|---|---|
| `22` | Checking | Credit — live |
| `23` | Checking | Credit — prenote |
| `27` | Checking | Debit — live |
| `28` | Checking | Debit — prenote |
| `32` | Savings | Credit — live |
| `37` | Savings | Debit — live |
| `42` | GL | Credit — live |
| `47` | GL | Debit — live |

---

## VISA VCF Validation Rules

Supports both **pipe-delimited** (`|`) and **fixed-width** VCF formats.

### Field Validation

| Field | Position | Format | Validation |
|---|---|---|---|
| Transaction Code | 0 | 2N | `{05, 06, 10, 12, 15, 25, 26, 30, 40, 92, 96}` |
| PAN | 1 | 13–19N | Luhn algorithm; may be masked with `*` or `X` |
| Processing Code | 2 | 6N | ISO processing code lookup |
| Amount | 3 | N.NN | Numeric, non-negative, ≤ 999,999,999.99 |
| Currency Code | 4 | 3N | ISO 4217 (`840`=USD, `978`=EUR, `356`=INR …) |
| Transaction DateTime | 5 | 14N | `YYYYMMDDHHmmSS` |
| MCC | 6 | 4N | Exactly 4 digits |
| POS Entry Mode | 7 | 2N | `{00, 01, 02, 05, 07, 10, 90, 91}` |
| Response Code | 8 | 2AN | `{00=Approved, 05=DNH, 51=NSF, 54=Expired …}` |
| Auth Code | 9 | 6AN | 6 alphanumeric characters |
| Merchant ID | 10 | 15AN | 1–15 alphanumeric |
| Terminal ID | 11 | 8AN | 1–8 alphanumeric |
| ARN | 14 | 23N | Acquirer Reference Number — exactly 23 digits |
| Interchange Fee | 16 | N.NN | Numeric when present |

### Luhn Algorithm (PAN Validation)

```python
digits = [int(d) for d in pan]
total  = sum(digits[-1::-2])          # sum odd positions right-to-left
for d in digits[-2::-2]:              # double even positions
    total += sum(divmod(d * 2, 10))
assert total % 10 == 0                # Luhn-valid PAN
```

---

## SLM Architecture

### Model Sizes

| Config | d_model | Heads | Layers | d_ff | Parameters |
|---|---|---|---|---|---|
| `nano` | 128 | 4 | 4 | 512 | ~1M |
| `small` | 256 | 8 | 6 | 1,024 | ~4M |
| `medium` | 384 | 6 | 8 | 1,536 | ~10M |

### Architecture Details

- **Decoder-only** GPT-style causal transformer — predicts next character autoregressively
- **Weight tying** — token embedding matrix shared with LM head (saves `vocab × d_model` params)
- **Causal mask** — upper-triangular `-inf` fill prevents attending to future positions
- **Positional embedding** — learned (not sinusoidal), max sequence length 2,048
- **Sampling** — Top-k + Nucleus (Top-p) combined for controlled, diverse generation
- **Tokenizer** — character-level, vocabulary of ~105 (printable ASCII + 7 special tokens)

### Special Tokens

| Token | ID | Purpose |
|---|---|---|
| `<PAD>` | 0 | Padding |
| `<BOS>` | 1 | Begin of sequence |
| `<EOS>` | 2 | End of sequence |
| `<UNK>` | 3 | Unknown character |
| `<REC>` | 4 | Record separator |
| `<ACH>` | 5 | ACH file type marker |
| `<VCF>` | 6 | VCF file type marker |

### Training Hyperparameters

| Parameter | Default | Notes |
|---|---|---|
| Optimizer | AdamW | β₁=0.9, β₂=0.95 |
| LR Schedule | Cosine annealing | Linear warmup from 0 to peak LR |
| Warmup steps | 50–100 | Shorter for smaller datasets |
| Gradient clip | 1.0 | Prevents exploding gradients |
| Loss | Cross-entropy | Next-character prediction |
| Block size | 512 | Training context window |

### ChromaDB-Aware Training Data Blend

```
Data priority (highest quality first):

  1. ach_training_corpus      <- ACH files saved by previous ChromaACHGenerator runs
  2. ChromaACHGenerator       <- freshly built from ChromaDB PENDING transactions
  3. ACHGenerator             <- purely synthetic (diversity and fallback)

Default blend:  60% ChromaDB corpus + live,  40% synthetic
Similarity search: ChromaDB vector index selects structurally diverse training files
```

---

## REST API Reference

### Generation

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/generate` | Generate synthetic ACH or VCF (rule-based or SLM) |
| `POST` | `/api/chroma/generate` | Generate ACH from ChromaDB PENDING transactions |
| `POST` | `/api/download` | Stream generated content as a downloadable file |

#### `POST /api/generate`
```json
{
  "file_type": "ACH",
  "use_model": false,
  "num_batches": 2,
  "entries_per_batch": 5,
  "sec_code": "PPD"
}
```

#### `POST /api/chroma/generate`
```json
{
  "company_id": "co_abc123",
  "sec_code": "PPD",
  "effective_date": "260502",
  "max_transactions": 500,
  "save_audit": true,
  "save_corpus": true
}
```

---

### Validation

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/validate` | Validate uploaded file (multipart form) |
| `POST` | `/api/validate/text` | Validate raw text body |

**Validation response:**
```json
{
  "is_valid": true,
  "summary": { "errors": 0, "warnings": 1, "info": 0 },
  "errors": [],
  "warnings": [{ "severity": "WARNING", "field": "blocking_factor", "line": 0, "message": "..." }],
  "statistics": { "total_records": 30, "total_batches": 2, "sec_codes": ["PPD"] }
}
```

---

### Training

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/train` | Synthetic-only SLM training (ACH or VCF) |
| `POST` | `/api/chroma/train` | ChromaDB-aware ACH training (corpus + live + synthetic) |
| `GET` | `/api/train/status` | Poll training progress |

#### `POST /api/chroma/train`
```json
{
  "model_config": "small",
  "n_corpus_files": 300,
  "n_chroma_files": 100,
  "n_synthetic_files": 100,
  "n_val_files": 50,
  "max_epochs": 10,
  "batch_size": 8
}
```

#### `GET /api/train/status`
```json
{
  "running": true,
  "progress": 67,
  "message": "Epoch 5/10 | train=1.2341 val=1.3102",
  "file_type": "ACH",
  "source": "chromadb"
}
```

---

### ChromaDB Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/chroma/test` | Verify ChromaDB client and all 7 collections |
| `GET` | `/api/chroma/health` | Collection counts and disk usage |
| `POST` | `/api/chroma/seed` | Insert synthetic transactions (`n`, `sec_code`) |
| `GET` | `/api/chroma/companies` | List companies (`?sec_code=PPD`) |
| `GET` | `/api/chroma/odfi` | List ODFI configurations |
| `GET` | `/api/chroma/corpus/stats` | TRAIN/VAL/TEST split counts |
| `GET` | `/api/chroma/file-log` | Recent generated file audit entries |

#### `POST /api/chroma/test` — Response
```json
{
  "success": true,
  "message": "ChromaDB OK — 128 total documents across 7 collections",
  "health": {
    "status": "ok",
    "store_path": "/path/to/slm_ach_vcf/db/chromadb_store",
    "store_size_mb": 2.4,
    "collections": {
      "ach_odfi_config": 3,
      "ach_companies": 5,
      "ach_accounts": 0,
      "ach_transactions": 50,
      "ach_file_log": 12,
      "ach_validation_log": 8,
      "ach_training_corpus": 50
    }
  }
}
```

---

### Utility

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/models` | Status of trained ACH and VCF models |
| `GET` | `/api/spec` | Embedded ACH and VCF specification summaries |

---

## Production Recommendations

### Five Key Principles

**1. ChromaDB corpus first, synthetic second.** ChromaDB stores full ACH file text with vector embeddings — every file you generate grows the corpus. The similarity search selects structurally diverse training examples automatically.

**2. Always validate generated output.** The rule-based generator produces 100% valid files. The SLM generates statistically realistic output that must be piped through `ACHValidator` before use in any downstream system.

**3. Save everything to `ach_training_corpus`.** Generated files are automatically saved when `save_corpus=True`. The corpus grows with every generation run and model quality improves continuously.

**4. Use the `medium` Transformer on GPU.** The Bigram model trains in minutes but has limited structural memory. The `medium` Transformer on a T4/A100 learns near-perfect ACH structure (perplexity < 1.1) and produces output that consistently passes NACHA validation.

**5. Rotate File ID Modifier.** The `ach_file_log` tracks the `file_id_modifier` (`A`–`Z`) value. The generator handles this automatically within a session; query the latest modifier from `ach_file_log` at startup to continue the sequence across restarts.

### Hardware Guide

| Setup | Hardware | Train Time | Output Quality |
|---|---|---|---|
| Bigram (numpy) | Any CPU | 1–3 min | Basic — structural patterns only |
| Nano Transformer | 8-core CPU | 15–30 min | Good — learns SEC code patterns |
| Small Transformer | GPU T4 | 5–10 min | Very good — learns amount and routing distributions |
| Medium Transformer | GPU A100 | 10–20 min | Excellent — near-NACHA-perfect generation |
| Medium + ChromaDB corpus | GPU A100 | 15–25 min | Best — real corpus patterns learned |

### Useful Monitoring Queries

```python
# Monitor corpus growth via REST API
import requests

# Collection document counts
health = requests.get('http://localhost:5000/api/chroma/health').json()
print(health['collections'])  # {'ach_training_corpus': 450, 'ach_transactions': 200, ...}

# Corpus TRAIN/VAL/TEST split
stats = requests.get('http://localhost:5000/api/chroma/corpus/stats').json()
print(stats)  # {'total': 450, 'train': 380, 'val': 50, 'test': 20}

# Recent file log (last 20 generated files)
log = requests.get('http://localhost:5000/api/chroma/file-log?limit=20').json()
for entry in log:
    print(entry['file_name'], entry['entry_count'], entry['created_at'])
```

Or query the ChromaDB collections directly in Python:

```python
from db.chroma_client import get_client
from db.ach_repository import ACHRepository

repo = ACHRepository('db/chromadb_store')

# Check pending transactions by SEC code
txns = repo.get_pending_transactions(sec_code='PPD', max_rows=10000)
print(f"Pending PPD transactions: {len(txns)}")

# Corpus stats
print(repo.corpus_stats())

# Find similar ACH files to a seed (vector similarity search)
seed = open('my_reference.ach').read()
similar = repo.fetch_corpus(split='TRAIN', similar_to=seed, limit=10)
print(f"Found {len(similar)} structurally similar training files")
```

---

## ChromaDB Configuration

ChromaDB requires **no environment variables** — it is fully configured by the store path
(`db/chromadb_store/`) which is set in `app.py` as `CHROMA_STORE`. No credentials,
no server, no network access required.

| Setting | Where | Default | Description |
|---|---|---|---|
| Store path | `app.py:CHROMA_STORE` | `db/chromadb_store/` | On-disk persistent store location |
| Embedding fn | `db/chroma_client.py` | `FinancialHashEmbedding` | Offline n-gram hash (256-dim) |
| Anonymised telemetry | `Settings` | `False` | Disabled — no data sent to Chroma cloud |
| Allow reset | `Settings` | `True` | Allows store to be wiped via `client.reset()` |

To relocate the store (e.g. to a network drive or larger disk), change `CHROMA_STORE` in `app.py`:

```python
CHROMA_STORE = '/mnt/data/finslm_chromadb'   # custom path
```

The store directory is created automatically if it does not exist.

---

## Security Notes

- **No credentials needed** — ChromaDB is an embedded local store with no authentication layer; restrict filesystem access via OS permissions if the store contains sensitive data
- **PAN numbers** in generated VCF files are Luhn-valid but entirely synthetic — they are not real card numbers
- **ABA routing numbers** used in ACH files are real public routing numbers included for structural realism only
- **`ach_file_log` documents** contain the full NACHA file text — apply filesystem encryption (e.g. LUKS, BitLocker, or cloud-provider KMS) to the `db/chromadb_store/` directory if handling sensitive production data
- **Do not submit generated files to live payment rails** — FinSLM is a training, testing, and validation tool
- The in-memory validator never persists file content to disk; validation results exist only in memory and the optional `ach_validation_log` collection
- Back up `db/chromadb_store/` regularly — ChromaDB uses SQLite internally and is safe to copy while the app is not writing
