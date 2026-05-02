# FinSLM — Financial Small Language Model
### ACH NACHA & VISA VCF · Generator · Validator · Oracle Integration · Intelligence Platform

---

## Overview

FinSLM is a production-ready Python system that trains a **custom Small Language Model (SLM)** on real Oracle transaction data and synthetic ACH NACHA / VISA VCF financial files. It provides a complete pipeline for generating, validating, and learning from structured financial files.

| Capability | Description |
|---|---|
| **Oracle-backed Generation** | Reads live `PENDING` transactions from Oracle, assembles spec-compliant NACHA files |
| **Synthetic Generation** | Rule-based ACH & VCF generator — works with no database |
| **Full Validation** | NACHA spec, VISA VCF spec, Luhn PAN, ABA routing check-digit, entry hash, batch totals |
| **SLM Training** | GPT-style Transformer (PyTorch) or Bigram n-gram (numpy) — auto-detected |
| **Oracle-aware Training** | Merges real Oracle corpus files with synthetic data for richer model training |
| **Web UI** | Wells Fargo–branded white dashboard — Generate, Validate, Train, Oracle DB, Spec |
| **REST API** | 11 JSON endpoints covering all operations |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         Flask Web Application  :5000                         │
│                                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
│  │/generate │  │/validate │  │  /train  │  │  /oracle │  │    /spec     │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────────────┘  │
└───────┼──────────────┼─────────────┼──────────────┼──────────────────────────┘
        │              │             │              │
        ▼              ▼             ▼              ▼
┌────────────────┐  ┌───────────┐  ┌───────────────────────────────────────────┐
│  Data Layer    │  │Validators │  │              Model Layer                  │
│                │  │           │  │                                           │
│ ACHGenerator   │  │ACHValidator  │  FinancialSLM  (Transformer, ~1–10M)    │
│ VCFGenerator   │  │VCFValidator  │       --- OR ---                        │
│ OracleACH      │  │           │  │  BigramSLM    (numpy fallback)           │
│  Generator     │  │NACHA spec │  │                                           │
│                │  │VISA spec  │  │  FinancialTokenizer  (char-level ~100)   │
│ 30 real ABA    │  │ABA routing│  │                                           │
│ routing numbers│  │Luhn check │  │  OracleAwareTrainer                      │
│ Luhn-valid PANs│  │Entry hash │  │  (Oracle corpus + synthetic blending)    │
└────────────────┘  └───────────┘  └───────────────────────────────────────────┘
        │                                         │
        ▼                                         ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           Oracle Database Layer                              │
│                                                                              │
│  OracleConnectionPool  (oracledb thin → thick → cx_Oracle → MOCK)          │
│                                                                              │
│  ACHRepository  -->  V_ACH_PENDING_TRANSACTIONS  (denormalised view)        │
│                       ├── ACH_ODFI_CONFIG         (bank routing config)      │
│                       ├── ACH_COMPANIES           (originator master)        │
│                       ├── ACH_ACCOUNTS            (RDFI account master)      │
│                       └── ACH_TRANSACTIONS        (pending queue)            │
│                                                                              │
│  Audit & Training  --> ACH_FILE_LOG               (generated file audit)    │
│                         ACH_VALIDATION_LOG         (validation history)      │
│                         ACH_TRAINING_CORPUS        (SLM training data)       │
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
├── db/                           # Oracle Database Layer
│   ├── __init__.py
│   ├── oracle_connector.py       # Connection pool, mock fallback, health check
│   ├── ach_repository.py         # All SQL queries, typed domain objects
│   └── schema.sql                # Full Oracle DDL (7 tables, 1 view, 1 proc)
│
├── data/                         # Data Generation
│   ├── generator.py              # Synthetic ACH + VCF file generator
│   ├── oracle_ach_generator.py   # Oracle-backed NACHA file builder
│   └── oracle_trainer.py         # Oracle-aware SLM training pipeline
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
# Minimum — numpy Bigram SLM, no Oracle client required
pip install flask numpy werkzeug

# Add Oracle support (thin driver — no Oracle Client install needed)
pip install flask numpy werkzeug python-oracledb

# Full stack — Transformer SLM + Oracle
pip install flask numpy werkzeug python-oracledb torch
```

### 2. Configure Oracle Connection (optional)

Set environment variables — the app runs in **Mock mode** if these are absent:

```bash
export ORACLE_HOST=db.example.com
export ORACLE_PORT=1521
export ORACLE_SERVICE=ORCL
export ORACLE_USER=finslm
export ORACLE_PASSWORD=yourpassword
export ORACLE_SCHEMA=FINSLM

# Optional — mTLS wallet (Oracle Autonomous Database / OCI)
export ORACLE_WALLET_DIR=/path/to/wallet
export ORACLE_WALLET_PWD=walletpassword

# Force mock mode even if DB is reachable (useful for CI/CD pipelines)
export ORACLE_MOCK=false
```

### 3. Set Up the Oracle Schema

```bash
# Run once against your Oracle instance as DBA or the finslm user
sqlplus finslm/password@ORCL @slm_ach_vcf/db/schema.sql
```

This creates all 7 tables, the `V_ACH_PENDING_TRANSACTIONS` view, and the
`SP_MARK_BATCHED` stored procedure under the `FINSLM` schema.

### 4. Start the Web App

```bash
cd slm_ach_vcf
python app.py
# Open http://localhost:5000
```

### 5. Train from the Command Line

```bash
# Synthetic-only (always works, no Oracle needed)
python trainer.py --type ACH --config nano --epochs 5
python trainer.py --type VCF --config small --epochs 10

# Oracle-aware training (reads corpus + generates from live data)
python -c "
from db.oracle_connector import OracleConfig
from data.oracle_trainer import OracleAwareTrainer
trainer = OracleAwareTrainer(OracleConfig(), {
    'model_config': 'small',
    'n_oracle_files': 300,
    'n_synthetic_files': 100,
    'max_epochs': 10,
})
results = trainer.train()
print('Best val loss:', results['best_val_loss'])
"
```

---

## Oracle Database Schema

### Tables

| Table | Purpose |
|---|---|
| `ACH_ODFI_CONFIG` | ABA routing numbers, bank name, NACHA File Header fields (immediate dest/origin) |
| `ACH_COMPANIES` | Originator companies — SEC code, service class code, company ID number (10 chars) |
| `ACH_ACCOUNTS` | Receiver (RDFI) accounts — routing, account number, individual name, prenote status |
| `ACH_TRANSACTIONS` | Pending payments queue — transaction code, amount in cents, effective date, lifecycle status |
| `ACH_FILE_LOG` | Audit trail for every generated file — batch/entry counts, totals, full CLOB content |
| `ACH_VALIDATION_LOG` | Validation history — full JSON report, error/warning counts per file |
| `ACH_TRAINING_CORPUS` | SLM training data store — ACH files saved with TRAIN/VAL/TEST split labels |

### Key View

```sql
-- Denormalised join of all 4 master tables — used by OracleACHGenerator
SELECT * FROM FINSLM.V_ACH_PENDING_TRANSACTIONS
WHERE STATUS = 'PENDING';
```

### Transaction Lifecycle

```
INSERT into ACH_TRANSACTIONS (STATUS='PENDING')
       |
       v
OracleACHGenerator.generate()
  |-- SELECT from V_ACH_PENDING_TRANSACTIONS
  |-- Build 94-char NACHA records
  |-- INSERT into ACH_FILE_LOG        (audit trail)
  |-- INSERT into ACH_TRAINING_CORPUS (feeds SLM training)
       |
       v
SP_MARK_BATCHED  -->  bulk UPDATE STATUS='BATCHED'
```

### Amount Storage

Amounts are stored as **integer cents** in `AMOUNT_CENTS` (e.g. `$19.99 → 1999`).
The generator zero-pads to 10 digits for the NACHA implied-decimal field: `0000001999`.

---

## Oracle Driver Auto-Detection

The connector selects the best available Oracle driver at startup with no configuration:

```
Priority 1  oracledb thin mode   (pure Python, pip install python-oracledb)
Priority 2  oracledb thick mode  (if Oracle Instant Client libraries on PATH)
Priority 3  cx_Oracle            (legacy thick-mode driver)
Priority 4  MOCK mode            (synthetic data, always works — zero dependencies)
```

The MOCK fallback means **the entire application runs without any Oracle installation**.
Every UI feature and API endpoint works in mock mode using synthetic data.

---

## Web UI — Five Panels

The UI uses a **white background with black text and Wells Fargo red (#D71E28) accents**
including buttons, active nav indicators, progress bars, header border, and panel title underlines.

| Panel | Icon | Key Actions |
|---|---|---|
| **Generate** | ⚡ | Rule-based or SLM-powered ACH/VCF file generation; inline copy/download/validate |
| **Validate** | ✓ | Drag-and-drop file upload or paste content; error/warning table with line numbers |
| **Train Model** | 🧠 | Configure model size/epochs; live progress bar and scrolling training log |
| **Oracle DB** | 🗄 | Connect, test, load companies, generate from live Oracle data, train on corpus |
| **Specification** | 📋 | NACHA record layout reference; VISA VCF field map; transaction code tables |

### Oracle DB Panel Features

- Connection form — host, port, service, username, password, schema, with mock mode toggle
- **Test Connection** — displays driver name, DSN, pool open/busy counts
- **Load Companies** — populates dropdown from `ACH_COMPANIES` table live
- **Generate from Oracle** — filter by company, SEC code, effective date, max rows
- **Audit toggle** — writes generated file metadata to `ACH_FILE_LOG`
- **Corpus toggle** — saves file content to `ACH_TRAINING_CORPUS` for future SLM training
- **Oracle-aware training** — pulls corpus files first, then generates from live transactions, blends with synthetic; real-time progress log

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

### Oracle-Aware Training Data Blend

```
Data priority (highest quality first):

  1. ACH_TRAINING_CORPUS   <- real files saved by previous OracleACHGenerator runs
  2. OracleACHGenerator    <- freshly built from live PENDING transactions
  3. ACHGenerator          <- purely synthetic (diversity and fallback)

Default blend:  75% Oracle-sourced,  25% synthetic
```

---

## REST API Reference

### Generation

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/generate` | Generate synthetic ACH or VCF (rule-based or SLM) |
| `POST` | `/api/oracle/generate` | Generate ACH from Oracle PENDING transactions |
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

#### `POST /api/oracle/generate`
```json
{
  "host": "db.example.com",
  "port": 1521,
  "service": "ORCL",
  "user": "finslm",
  "password": "secret",
  "schema": "FINSLM",
  "company_id": 3,
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
| `POST` | `/api/oracle/train` | Oracle-aware ACH training (corpus + live + synthetic) |
| `GET` | `/api/train/status` | Poll training progress |

#### `POST /api/oracle/train`
```json
{
  "model_config": "small",
  "n_oracle_files": 300,
  "n_synthetic_files": 100,
  "n_val_files": 50,
  "max_epochs": 10,
  "batch_size": 8,
  "host": "db.example.com",
  "user": "finslm",
  "password": "secret"
}
```

#### `GET /api/train/status`
```json
{
  "running": true,
  "progress": 67,
  "message": "Epoch 5/10 | train=1.2341 val=1.3102",
  "file_type": "ACH",
  "source": "oracle"
}
```

---

### Oracle Connectivity

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/oracle/test` | Test connection, return driver + pool health |
| `GET` | `/api/oracle/health` | Lightweight status — no new connection opened |
| `GET` | `/api/oracle/companies` | List companies from `ACH_COMPANIES` (`?sec_code=PPD`) |

#### `POST /api/oracle/test` — Response
```json
{
  "success": true,
  "message": "Oracle connection successful",
  "health": {
    "driver": "oracledb-thin",
    "mode": "live",
    "dsn": "localhost:1521/ORCL",
    "schema": "FINSLM",
    "pool_open": 2,
    "pool_busy": 0
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

**1. Oracle first, synthetic second.** Oracle corpus files represent real business patterns — payment amounts, company distributions, routing number frequency. Always prioritise real data when training.

**2. Always validate generated output.** The rule-based generator produces 100% valid files. The SLM generates statistically realistic output that must be piped through `ACHValidator` before use in any downstream system.

**3. Save everything to `ACH_TRAINING_CORPUS`.** Every Oracle-sourced file generated is automatically saved when `save_corpus=True`. The corpus grows over time and model quality improves with each retraining cycle.

**4. Use the `medium` Transformer on GPU.** The Bigram model trains in minutes but has limited structural memory. The `medium` Transformer on a T4/A100 learns near-perfect ACH structure (perplexity < 1.1) and produces output that consistently passes NACHA validation.

**5. Rotate File ID Modifier.** The `ACH_FILE_LOG` tracks the `FILE_ID_MODIFIER` (`A`–`Z`) value. The generator handles this automatically within a session; in production, persist the last modifier in Oracle and seed the generator on startup.

### Hardware Guide

| Setup | Hardware | Train Time | Output Quality |
|---|---|---|---|
| Bigram (numpy) | Any CPU | 1–3 min | Basic — structural patterns only |
| Nano Transformer | 8-core CPU | 15–30 min | Good — learns SEC code patterns |
| Small Transformer | GPU T4 | 5–10 min | Very good — learns amount and routing distributions |
| Medium Transformer | GPU A100 | 10–20 min | Excellent — near-NACHA-perfect generation |
| Medium + Oracle corpus | GPU A100 | 15–25 min | Best — real business patterns learned |

### Useful Monitoring Queries

```sql
-- Monitor corpus growth over time
SELECT SPLIT_TYPE, SEC_CODE, COUNT(*) AS FILE_COUNT, MAX(CREATED_AT) AS LATEST
FROM FINSLM.ACH_TRAINING_CORPUS
GROUP BY SPLIT_TYPE, SEC_CODE
ORDER BY SPLIT_TYPE, FILE_COUNT DESC;

-- Check pending transactions available for next generation run
SELECT c.COMPANY_NAME, c.SEC_CODE,
       COUNT(*)               AS PENDING_COUNT,
       SUM(t.AMOUNT_CENTS)/100 AS TOTAL_USD
FROM FINSLM.ACH_TRANSACTIONS t
JOIN FINSLM.ACH_COMPANIES c ON t.COMPANY_ID = c.COMPANY_ID
WHERE t.STATUS = 'PENDING'
GROUP BY c.COMPANY_NAME, c.SEC_CODE
ORDER BY PENDING_COUNT DESC;

-- Validation failure rate by file source
SELECT f.GENERATION_METHOD,
       COUNT(*)                                    AS TOTAL_FILES,
       SUM(CASE WHEN v.IS_VALID = 'N' THEN 1 END) AS FAILED,
       ROUND(AVG(v.ERROR_COUNT), 2)                AS AVG_ERRORS
FROM FINSLM.ACH_FILE_LOG f
JOIN FINSLM.ACH_VALIDATION_LOG v ON v.FILE_ID = f.FILE_ID
GROUP BY f.GENERATION_METHOD;
```

---

## Environment Variable Reference

| Variable | Default | Description |
|---|---|---|
| `ORACLE_HOST` | `localhost` | Oracle DB hostname or IP address |
| `ORACLE_PORT` | `1521` | Oracle listener port |
| `ORACLE_SERVICE` | `ORCL` | Oracle service name or SID |
| `ORACLE_USER` | `finslm` | Database username |
| `ORACLE_PASSWORD` | _(empty)_ | Database password |
| `ORACLE_WALLET_DIR` | _(empty)_ | Wallet directory for mTLS (OCI Autonomous Database) |
| `ORACLE_WALLET_PWD` | _(empty)_ | Wallet password |
| `ORACLE_POOL_MIN` | `2` | Minimum connections kept open in pool |
| `ORACLE_POOL_MAX` | `10` | Maximum connections allowed in pool |
| `ORACLE_POOL_INC` | `1` | How many connections to add when pool is exhausted |
| `ORACLE_TIMEOUT_S` | `30` | Connection and query timeout in seconds |
| `ORACLE_SCHEMA` | `FINSLM` | Schema/owner prefix for all table and view references |
| `ORACLE_MOCK` | `false` | Set `true` to force mock mode (no real DB connection attempted) |

---

## Security Notes

- **Oracle credentials** are read from environment variables only — never hardcode passwords in source files or API request bodies that may be logged
- **PAN numbers** in generated VCF files are Luhn-valid but entirely synthetic — they are not real card numbers
- **ABA routing numbers** used in ACH files are real public routing numbers included for structural realism only
- **`ACH_FILE_LOG.FILE_CONTENT`** stores the full NACHA file as a CLOB — apply Oracle TDE (Transparent Data Encryption) or column-level encryption on this column in production environments
- **Do not submit generated files to live payment rails** — FinSLM is a training, testing, and validation tool
- The in-memory validator never persists file content to disk; validation results exist only in memory and the optional `ACH_VALIDATION_LOG` table
- Rotate `ORACLE_PASSWORD` via your secrets management system (HashiCorp Vault, AWS Secrets Manager, or OCI Vault); the application re-reads the environment variable on pool reset via `OracleConnectionPool.reset()`
