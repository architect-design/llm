# HELP.MD - Complete Code Documentation

## Overview
This project is a **Financial Structured Language Model (SLM)** web application designed to generate, validate, and train AI models on ACH (NACHA) and VCF (VisaNet Custom File) financial file formats. It supports both synthetic and **ChromaDB**-backed data, with a web UI for user interaction.

**Key Technologies:**
- Flask (web framework)
- PyTorch (deep learning, optional)
- NumPy (scientific computing)
- **ChromaDB** (vector database with local persistence - REPLACES Oracle)
- Custom Embedding Functions (offline n-gram hashing)

---

## Main Application

### `app.py` (541 lines)
The Flask web server providing REST APIs and web UI endpoints.

#### Configuration
```python
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload
app.config['UPLOAD_FOLDER'] = 'uploads/'             # Temp file storage
app.config['MODEL_DIR'] = 'trained_models/'          # Saved models location
```

#### Global State
```python
training_state = {
    "running": False,
    "file_type": None,    # "ACH" or "VCF"
    "message": str,       # Status message
    "progress": int,      # 0-100
    "result": dict,       # Training results
    "error": str,         # Error message if failed
    "started_at": str,    # ISO timestamp
}
```

#### Key Endpoints

**Validation Endpoints:**
- `POST /api/validate` - Upload and validate ACH/VCF file
  - Parameters: `file` (multipart), `file_type` ("ACH" or "VCF")
  - Returns: Validation report with errors, warnings, statistics
  
- `POST /api/validate/text` - Validate raw text content
  - Parameters: `content` (string), `file_type` ("ACH" or "VCF")
  - Returns: Same as `/api/validate`

**Generation Endpoints:**
- `POST /api/generate` - Generate synthetic ACH/VCF file
  - Parameters: `file_type`, `use_model` (bool), `temperature`, `max_tokens`
  - Returns: Generated file content + metadata
  
- `POST /api/download` - Download generated content as file
  - Parameters: `content`, `file_type`, `filename`
  - Returns: File download

**Training Endpoints:**
- `POST /api/train` - Start model training (async, runs in background thread)
  - Parameters: `file_type`, `model_config`, `n_files`, `batch_size`, `max_epochs`, `learning_rate`
  - Returns: Training job started confirmation
  
- `GET /api/train/status` - Get current training status
  - Returns: Current `training_state`

**Model Management:**
- `GET /api/models` - List available trained models
  - Returns: Dict with ACH/VCF models, sizes, timestamps

**Specification:**
- `GET /api/spec` - Get format specifications
  - Returns: ACH and VCF format details

**Oracle Integration:**
- `POST /api/oracle/test` - Test Oracle DB connection
- `GET /api/oracle/health` - Check Oracle connection health
- `POST /api/oracle/generate` - Generate ACH from Oracle pending transactions
- `GET /api/oracle/companies` - List companies from Oracle
- `POST /api/oracle/train` - Train on Oracle data + synthetic augmentation

#### Helper Functions
- `_get_oracle_config(data)` - Parse Oracle config from API request

---

## Model Training and Generation

### `trainer.py` (281 lines)
Implements the training pipeline for financial SLM models with automatic backend selection.

#### Automatic Backend Detection
```python
try:
    import torch
    BACKEND = "transformer"  # PyTorch-based Transformer
except ImportError:
    BACKEND = "bigram"       # NumPy-based fallback
```

#### Class: `Trainer`
Orchestrates model training and data generation.

**Constructor:**
```python
def __init__(self, file_type: str = "ACH", config: dict = None):
    # file_type: "ACH" or "VCF"
    # config: Training hyperparameters (uses defaults if None)
```

**Key Methods:**

- `generate_training_data(callback=None)` 
  - Generates synthetic ACH/VCF files for training/validation
  - Returns: `(train_files: List[str], val_files: List[str])`
  - Callback: `callback(message, progress_pct)`

- `train(callback=None, status_file=None)`
  - Main training loop. Auto-selects Transformer or Bigram backend
  - Returns: `{"model_path": str, "backend": str, "best_val_loss": float, "train_losses": List[float]}`

- `_train_transformer(train_files, val_files, report)` (Private)
  - Trains PyTorch Transformer model
  - Uses DataLoader, AdamW optimizer, CosineAnnealing LR scheduler
  - Saves best model on lowest validation loss
  - Returns: Training metadata dict

- `_train_bigram(train_files, val_files, report)` (Private)
  - Trains n-gram fallback model using numpy
  - Faster but less powerful than Transformer

**Default Configuration:**
```python
DEFAULT_CONFIG = {
    "ach": {
        "model_config": "small",
        "n_training_files": 300,
        "n_val_files": 40,
        "block_size": 512,              # Sequence length
        "batch_size": 4,
        "max_epochs": 8,
        "learning_rate": 3e-4,
        "weight_decay": 0.1,
        "grad_clip": 1.0,
        "warmup_steps": 50,
        "eval_every": 50,
        "save_dir": "trained_models/",
    },
    "vcf": { ... }  # Similar for VCF
}
```

#### Class: `FileGenerator`
Loads and uses trained models to generate new files.

**Constructor:**
```python
def __init__(self, file_type: str, model_dir: str = None):
    # Auto-detects and loads either .pt (Transformer) or .pkl (Bigram) model
```

**Key Methods:**

- `generate_with_seed(temperature=0.8, max_tokens=2000, **kwargs) -> str`
  - Generates new file content from trained model
  - `temperature`: Controls randomness (0.5=deterministic, 1.0=normal, >1=random)
  - Returns: Generated file as string
  - Handles both Transformer (with sampling) and Bigram (with backoff) generation

---

## Model Architecture

### `model/transformer.py` (208 lines)
GPT-style decoder-only Transformer for character-level text generation.

#### Class: `MultiHeadAttention(nn.Module)`
Computes scaled dot-product attention across multiple heads.

**Parameters:**
```python
def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
    # d_model: Hidden dimension (must be divisible by n_heads)
    # n_heads: Number of attention heads
```

**Forward Pass:**
```python
def forward(self, x: Tensor, mask=None) -> Tensor:
    # x shape: (batch_size, seq_len, d_model)
    # mask: Causal mask for autoregressive generation
    # Returns: Attention output same shape as input
```

#### Class: `FeedForward(nn.Module)`
Two-layer MLP with GELU activation and dropout.

```python
def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
    # d_model: Hidden dimension
    # d_ff: Intermediate dimension (usually 4x d_model)
```

#### Class: `TransformerBlock(nn.Module)`
Single Transformer block combining attention + feedforward with residuals.

```python
def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
    # Combines MultiHeadAttention + FeedForward
    # Uses pre-normalization: LayerNorm → Attention → Residual
```

#### Class: `FinancialSLM(nn.Module)` - Main Model
Character-level Transformer for ACH/VCF generation.

**Model Configurations:**
```python
CONFIGS = {
    "nano":   {"d_model": 128, "n_heads": 4, "n_layers": 4, "d_ff": 512},    # ~1M params
    "small":  {"d_model": 256, "n_heads": 8, "n_layers": 6, "d_ff": 1024},   # ~4M params
    "medium": {"d_model": 384, "n_heads": 6, "n_layers": 8, "d_ff": 1536},   # ~10M params
}
```

**Constructor:**
```python
def __init__(self, vocab_size: int, max_seq_len: int = 2048, config: str = "small", dropout: float = 0.1):
    # vocab_size: Number of unique characters
    # max_seq_len: Maximum sequence length (context window)
    # config: "nano", "small", or "medium"
```

**Key Methods:**

- `forward(idx: Tensor, targets: Tensor = None) -> (logits, loss)`
  - `idx`: Input token IDs, shape (batch_size, seq_len)
  - `targets`: Target token IDs (optional, needed for loss computation)
  - Returns: `(logits, loss)` or `(logits, None)` if no targets
  - Logits shape: (batch_size, seq_len, vocab_size)

- `generate(idx: Tensor, max_new_tokens: int, temperature=1.0, top_k=50, top_p=0.9, stop_token=None) -> Tensor`
  - Autoregressive generation with sampling
  - `temperature`: Softness of probability distribution
  - `top_k`: Only sample from top-k most likely tokens
  - `top_p`: Nucleus sampling threshold
  - `stop_token`: Stop generation token ID
  - Returns: Generated token sequence

- `save(path: str, tokenizer=None, file_type=None)`
  - Saves model checkpoint with vocabulary and metadata

- `@classmethod load(cls, path: str, config="small")`
  - Loads checkpoint and reconstructs model

**Key Features:**
- Weight tying: Embedding and output layer share weights
- Causal masking: Prevents attending to future tokens
- Pre-normalization: LayerNorm before attention/feedforward
- Positional embeddings: Fixed sinusoidal embeddings

---

### `model/tokenizer.py` (114 lines)
Character-level tokenizer for financial files.

#### Class: `FinancialTokenizer`
Maps characters ↔ token IDs with special tokens for financial structure.

**Special Tokens:**
```python
SPECIAL_TOKENS = {
    "<PAD>":  0,    # Padding
    "<BOS>":  1,    # Begin of sequence
    "<EOS>":  2,    # End of sequence
    "<UNK>":  3,    # Unknown character
    "<REC>":  4,    # Record separator
    "<ACH>":  5,    # ACH file marker
    "<VCF>":  6,    # VCF file marker
}
```

**Key Methods:**

- `encode(text: str, add_special_tokens: bool = True) -> List[int]`
  - Converts text to token IDs
  - If `add_special_tokens=True`, wraps with `<BOS>...<EOS>`
  - Unknown characters → `<UNK>` token

- `decode(ids: List[int], skip_special: bool = True) -> str`
  - Converts token IDs back to text
  - If `skip_special=True`, omits special token strings

- `encode_file(content: str, file_type: str = "ACH") -> List[int]`
  - Encodes entire file with type marker
  - File marker added at start

- `create_training_batch(texts: List[str], block_size: int, file_type: str = "ACH")`
  - Creates (input, target) pairs for training
  - Groups tokens into overlapping windows of size `block_size`
  - Returns: `(torch.Tensor, torch.Tensor)` of shapes `(n_samples, block_size)`

**Properties:**
- `vocab_size`: Total number of tokens
- `vocab`: Dictionary mapping char → token_id
- `char2idx`, `idx2char`: Bidirectional mappings

---

### `model/bigram_fallback.py` (108 lines)
NumPy-based n-gram language model (fallback when PyTorch unavailable).

#### Class: `BigramSLM`
Simple character-level n-gram model.

**Constructor:**
```python
def __init__(self, order: int = 4):
    # order: N-gram order (4 = quadrigram context)
```

**Key Methods:**

- `train(texts: List[str], callback=None)`
  - Builds n-gram counts from training texts
  - Callback: `callback(message, progress)`
  - Stores vocabulary and transition counts

- `generate(seed: str = None, max_chars: int = 2000, temperature: float = 1.0) -> str`
  - Generates text from trained model
  - Uses best n-gram match; falls back to shorter contexts if needed
  - Temperature controls softness of probability distribution
  - Returns: Generated text string

- `save(path: str)` / `@classmethod load(path: str)`
  - Persists/loads model using pickle

**Key Attributes:**
- `counts`: Dict[context → Dict[char → count]]
  - Stores n-gram transition statistics
- `vocab`: Set of all observed characters
- `file_type`: "ACH" or "VCF"

---

## Database Access (ChromaDB)

### `db/chroma_client.py` (243 lines)
Thread-safe ChromaDB client managing persistent local vector database.

**Purpose:**
- Replaces Oracle Database as the primary data store
- All data stored locally in `db/chromadb_store/` folder
- Zero external dependencies for database connectivity

**Collections (Like Oracle Tables):**
```python
ach_odfi_config      # ODFI bank routing configuration
ach_companies        # Originator company master data
ach_accounts         # Receiver (RDFI) account details
ach_transactions     # Pending payment transactions
ach_file_log         # Generated ACH file audit trail
ach_validation_log   # Validation run results
ach_training_corpus  # SLM training examples (ACH file content)
```

**Key Features:**
- **FinancialHashEmbedding**: Lightweight offline embedding using character n-gram hashing
  - No model download required
  - Dimension: 256-dimensional vectors
  - Works entirely offline
- **Thread-safe singleton pattern**: Reuses single client instance
- **Automatic collection creation**: Creates collections on first use

**Key Methods:**

- `@classmethod get_instance() -> ChromaDBClient`
  - Returns singleton client instance
  - Creates if needed

- `add_documents(collection_name, ids, documents, metadatas, embedding_texts=None)`
  - Adds records to a collection
  - `embedding_texts`: Text to embed (defaults to documents)

- `query(collection_name, query_texts, n_results=10, where=None)`
  - Searches collection with semantic similarity
  - Optional metadata filters via `where` dict

- `get(collection_name, ids=None, where=None, limit=None) -> List[Dict]`
  - Retrieves records by ID or metadata filter
  - Returns full record dicts with metadata

- `update(collection_name, ids, documents, metadatas, embedding_texts=None)`
  - Updates existing records

- `delete(collection_name, ids=None, where=None)`
  - Deletes records by ID or metadata filter

- `reset_collection(collection_name) -> bool`
  - Clears all records from a collection
  - Useful for resetting state

**Example Usage:**
```python
client = ChromaDBClient.get_instance()

# Add a transaction
client.add_documents(
    "ach_transactions",
    ids=["txn_123"],
    documents=["TRANSACTION_DETAILS"],
    metadatas=[{
        "company_id": "ACME",
        "amount": 10000,
        "status": "PENDING"
    }]
)

# Query by metadata
results = client.get(
    "ach_transactions",
    where={"company_id": "ACME", "status": "PENDING"},
    limit=100
)

# Semantic search
similar = client.query(
    "ach_training_corpus",
    query_texts=["PPD payroll direct deposit"],
    n_results=5
)
```

---

### `db/chroma_schema.py`
Schema definitions and initialization scripts for ChromaDB collections.

**Contains:**
- Collection metadata schemas
- Index definitions
- Data validation rules
- Sample seed data for each collection
- Initialization function to populate demo data

---

## Data Generation

### `data/generator.py` (416 lines)
Generates realistic synthetic ACH and VCF files for training.

#### Reference Data (Realistic Constants)
```python
COMPANY_NAMES          # 20 realistic company names
INDIVIDUAL_NAMES      # 30+ realistic person names
BANK_NAMES           # UK and US major bank names
VALID_ROUTING_NUMBERS # ~30 real ABA routing numbers
SEC_CODES            # ["PPD", "CCD", "CTX", "WEB", "TEL"]
TRANSACTION_CODES_CREDIT  # ["22", "32"] 
TRANSACTION_CODES_DEBIT   # ["27", "37"]
MERCHANT_NAMES       # Common merchant names
MCC_CODES            # Merchant category codes (4-digit)
VALID_PROCESSING_CODES # VISA/VCF transaction types
```

#### Class: `ACHGenerator`
Generates syntactically and semantically valid ACH NACHA files.

**Key Methods:**

- `generate_file(num_batches=None, entries_per_batch=None, sec_code=None, file_date=None) -> str`
  - Generates complete 94-char-per-record NACHA file
  - Automatically calculates checksums and entry counts
  - Returns: Full ACH file as multi-line string
  - Records padded to multiples of 10 (blocking factor)

- `generate_dataset(n: int = 100) -> List[str]`
  - Generates `n` ACH files for training

**ACH Record Types Generated:**
- **Type 1:** File Header (once per file)
- **Type 5:** Batch Header (once per batch)
- **Type 6:** Entry Detail (one per transaction)
- **Type 7:** Addenda (optional, one per entry with addenda)
- **Type 8:** Batch Control (once per batch)
- **Type 9:** File Control (once per file)
- **Padding:** 9-filled records to reach next multiple of 10

**Example Usage:**
```python
gen = ACHGenerator()
file_content = gen.generate_file(num_batches=2, entries_per_batch=5, sec_code="PPD")
# Returns a valid 94-char-per-record ACH file
```

**NACHA Field Details (Internal Methods):**
- `_file_header()`: Record type 1 (0-indexed positions)
- `_batch_header()`: Record type 5
- `_entry_detail()`: Record type 6
- `_batch_control()`: Record type 8
- `_file_control()`: Record type 9

#### Class: `VCFGenerator`
Generates VISA VCF (VisaNet Custom File) format.

**Key Methods:**

- `generate_file(num_transactions=None, file_date=None, acquirer_bin=None) -> str`
  - Generates pipe-delimited VCF file
  - Formats: Header | many Transaction records | Trailer
  - Header: `VCF|version|acq_bin|date|time|bank_name|env|seq`
  - Each transaction: 20+ pipe-delimited fields

- `generate_dataset(n: int = 100) -> List[str]`
  - Generates `n` VCF files

**VCF Transaction Fields:**
```
0: Transaction Code (05=Auth, 06=Financial, 10=Reversal, etc.)
1: PAN (16-digit Visa card number)
2: Processing Code
3: Amount (decimal)
4: Currency Code (ISO 4217, e.g., "840"=USD)
5: Transaction DateTime
6: Merchant Category Code (4-digit MCC)
7: POS Entry Mode
8: Response Code
9: Authorization Code
10: Merchant ID (15 chars)
11: Terminal ID (8 chars)
... (more fields)
16: Interchange Fee
18: Country Code
19: Additional fields
```

**Special Feature:**
- `_generate_pan()`: Generates valid Visa PAN with correct Luhn checksum

**Example Usage:**
```python
gen = VCFGenerator()
file_content = gen.generate_file(num_transactions=50)
# Returns pipe-delimited VCF with valid transaction records
```

---

### `data/chroma_ach_generator.py` (408 lines)
Generates ACH NACHA files from ChromaDB transaction data (replaces `oracle_ach_generator.py`).

**Purpose:**
- Reads pending transactions from ChromaDB `ach_transactions` collection
- Assembles NACHA-compliant 94-character-per-record files
- Falls back to synthetic generation if ChromaDB is empty
- Logs generated files to ChromaDB for audit trail

**Constructor:**
```python
def __init__(self, store_path: Optional[str] = None):
    # store_path: Optional custom path to chromadb_store/
    # Defaults to db/chromadb_store/
```

**Key Methods:**

- `generate(company_id=None, sec_code=None, effective_date=None, max_transactions=5000, save_audit=True, save_corpus=True) -> Dict`
  - Main entry point: generates ACH from ChromaDB
  - Returns dict with:
    ```python
    {
        "content": str,              # Full NACHA file
        "file_name": str,            # Generated filename
        "batch_count": int,          # Number of batches
        "entry_count": int,          # Total transactions
        "total_debit": int,          # Debit sum (cents)
        "total_credit": int,         # Credit sum (cents)
        "source": str,               # "chromadb" or "synthetic"
        "transactions": List[int],   # Transaction IDs included
        "file_id": Optional[str],    # ChromaDB audit log ID
    }
    ```
  - Falls back to synthetic data if ChromaDB corpus empty
  - Audit logs to `ach_file_log` collection
  - Saves to `ach_training_corpus` if `save_corpus=True`

- `generate_training_batch(n_files=200, sec_code=None, callback=None) -> List[str]`
  - Generates multiple files for SLM training
  - Each file created by calling `generate()` internally
  - Useful for training-aware data generation

- `fetch_corpus_files(split="TRAIN", limit=500) -> List[str]`
  - Retrieves previously generated files from ChromaDB corpus
  - `split`: "TRAIN" or "VAL" (validation)

**Internal NACHA Builders (Similar Structure to ACHGenerator):**
- `_build_records()`: Assembles NACHA records from transaction objects
- `_file_header()`, `_batch_header()`, `_entry_detail()`, `_batch_control()`, `_file_control()`

**Key Differences from `oracle_ach_generator.py`:**
- Data source: ChromaDB collections instead of SQL queries
- Repository calls use ChromaDB `get`/`where` filters instead of SQL WHERE clauses
- Generated files stored back into ChromaDB collections
- Native support for offline operation
- Embeddings enable semantic search of transaction corpus

---

### `data/chroma_trainer.py` (263 lines)
ChromaDB-aware training pipeline that blends corpus, live, and synthetic data (replaces `oracle_trainer.py`).

**Data Priority (Highest Quality First):**
1. **ach_training_corpus** - High-quality ACH files previously saved in ChromaDB
2. **ChromaACHGenerator** - Fresh files built from ChromaDB transactions (real data patterns)
3. **ACHGenerator** - Purely synthetic files (diversity + fallback)

**Constructor:**
```python
def __init__(self, store_path: Optional[str] = None, train_config: dict = None):
    # store_path: Path to chromadb_store/ (defaults to db/chromadb_store/)
    # train_config: Override DEFAULT_CONFIG
```

**Default Configuration:**
```python
DEFAULT_CONFIG = {
    "model_config":       "small",
    "n_corpus_files":     300,   # from ChromaDB corpus collection
    "n_chroma_files":     100,   # generated fresh from ChromaDB transactions
    "n_synthetic_files":  100,   # always include some synthetic for diversity
    "n_val_files":        50,
    "corpus_weight":      0.6,   # fraction from ChromaDB corpus
    "block_size":         512,
    "batch_size":         4,
    "max_epochs":         10,
    "learning_rate":      3e-4,
    "weight_decay":       0.1,
    "grad_clip":          1.0,
    "warmup_steps":       50,
    "save_dir":           "trained_models/",
}
```

**Key Methods:**

- `collect_training_data(callback=None) -> dict`
  - Gathers training data from all three sources
  - Returns: `{"corpus": n, "chroma_live": n, "synthetic": n}`
  - Callback for progress reporting

- `train(callback=None) -> Dict`
  - Main training loop
  - Auto-selects Transformer (PyTorch) or Bigram (NumPy) backend
  - Returns: `{"model_path", "backend", "best_val_loss", "train_losses"}`

**Example Usage:**
```python
from data.chroma_trainer import ChromaAwareTrainer

trainer = ChromaAwareTrainer(train_config={
    "n_corpus_files": 200,
    "max_epochs": 5,
})

def progress_callback(msg, pct):
    print(f"[{pct}%] {msg}")

result = trainer.train(callback=progress_callback)
print(f"Model: {result['model_path']}")
print(f"Val Loss: {result['best_val_loss']:.4f}")
```

---
Repository pattern for all ACH-related database queries.

#### Domain Objects (Dataclasses)
All represent Oracle table rows as typed Python objects.

**`ACHRepository.ODFIConfig`** (Originating Depository Financial Institution)
```python
routing_number: str          # 9 digits
bank_name: str
immediate_dest: str          # File header destination
immediate_origin: str        # File header origin
dest_short_name: str         # ≤23 characters
origin_short_name: str       # ≤23 characters
```

**`ACHRepository.CompanyRecord`**
```python
company_id: int
company_name: str            # ≤16 chars
company_id_number: str       # ≤10 chars (IRS EIN equivalent)
company_entry_desc: str      # ≤10 chars (e.g., "PAYROLL")
sec_code: str                # "PPD", "CCD", etc. (3 chars)
service_class_code: str      # "200", "220", or "225"
discretionary_data: str      # ≤2 chars
```

**`ACHRepository.AccountRecord`**
```python
account_id: int
individual_name: str         # ≤22 chars
individual_id: str           # ≤15 chars
rdfi_routing: str            # RDFI check digit excluded (8 digits)
rdfi_check_digit: str        # ABA check digit (1 digit)
account_number: str          # ≤17 chars
account_type: str            # "C"=Checking, "S"=Savings, etc.
```

**`ACHRepository.TransactionRecord`** (Complete transaction with all contexts)
```python
transaction_id: int
account_id: int
company_id: int
transaction_code: str        # "22"=credit, "27"=debit, etc. (2 digits)
amount_cents: int            # Stored as cents, not dollars
effective_date_str: str      # "YYMMDD"
individual_id: str
individual_name: str
rdfi_routing: str            # 8 digits
rdfi_check_digit: str
account_number: str
account_type: str
discretionary_data: str
addenda_info: Optional[str]  # Payment details if present
company_name: str
company_id_number: str
company_entry_desc: str
sec_code: str
service_class_code: str
company_disc_data: str       # ≤20 chars
odfi_routing: str            # 9 digits
immediate_dest: str
immediate_origin: str
dest_short_name: str
origin_short_name: str

# Useful properties:
@property
def amount_str(self) -> str:
    return str(self.amount_cents).zfill(10)  # 10-digit padded amount
```

#### Class: `ACHRepository`
All database queries for ACH data access. Falls back to mock data if DB unavailable.

**Constructor:**
```python
def __init__(self, config: Optional[OracleConfig] = None):
    self.pool = get_pool(config)
    self.schema = config.schema or "FINSLM"
    self._mock = self.pool.is_mock  # True if no DB connection
```

**Key Methods:**

- `get_pending_transactions(company_id=None, sec_code=None, effective_date=None, max_rows=5000) -> List[TransactionRecord]`
  - Queries `V_ACH_PENDING_TRANSACTIONS` view
  - Optional filters by company, SEC code, or date
  - Returns list of typed TransactionRecord objects
  - Falls back to mock data if query fails

- `get_companies(sec_code: Optional[str] = None) -> List[CompanyRecord]`
  - Queries `ACH_COMPANIES` table
  - Optional filter by SEC code
  - Returns only active companies (IS_ACTIVE='Y')

- `log_file(file_name, modifier, odfi_id, batch_count, entry_count, block_count, total_debit, total_credit, content) -> Optional[int]`
  - Inserts generated ACH file to `ACH_FILE_LOG` table
  - Returns file_id (Oracle-assigned) or None in mock mode
  - Audit trail: when file was generated, who generated it, etc.

- `save_corpus_entry(content, sec_code, scc, batches, entries, split="TRAIN", file_log_id=None)`
  - Inserts file to `ACH_TRAINING_CORPUS` table
  - `split`: "TRAIN" or "VAL" (80/20 split)
  - Used for SLM model training curation

- `fetch_corpus(split="TRAIN", sec_code=None, limit=500) -> List[str]`
  - Retrieves stored training files from corpus
  - Returns file contents (handles CLOB columns)
  - Optional filter by SEC code

**Internal Methods:**
- `_sql(template, **filters)`: Injects schema name and WHERE clauses
- `_row_to_txn(r)`: Converts query result dict → TransactionRecord
- `_mock_transactions(n, sec_code)`: Generates synthetic data when DB unavailable
- `_mock_companies(sec_code)`: Generates synthetic company list

**SQL Queries (Parameterized, No Injection Risk):**
```python
_SQL_PENDING_TRANSACTIONS  # SELECT pending txns
_SQL_COMPANIES             # SELECT active companies
_SQL_ODFI                  # SELECT bank config
_SQL_LOG_FILE              # INSERT file audit log
_SQL_SAVE_CORPUS           # INSERT training corpus
_SQL_FETCH_CORPUS          # SELECT training files
```

**Example Usage:**
```python
repo = ACHRepository()

# Get transactions for specific company
txns = repo.get_pending_transactions(company_id=123, sec_code="PPD", max_rows=100)
for txn in txns:
    print(f"Transaction {txn.transaction_id}: {txn.amount_str}")

# Log generated file
file_id = repo.log_file(
    file_name="ACH_20260502_A.ach",
    modifier="A",
    odfi_id=1,
    batch_count=2,
    entry_count=10,
    block_count=2,
    total_debit=50000,
    total_credit=0,
    content=file_content
)
```

---

## Validation

### `validators/ach_validator.py` (427 lines)
Validates ACH NACHA files against NACHA Operating Rules and Guidelines (ORG).

#### Severity Levels
```python
class Severity(Enum):
    ERROR   # File fails validation, cannot be processed
    WARNING # Issue but file may still processable
    INFO    # Informational message
```

#### Report Classes

**`ValidationResult`**
Single validation issue.
```python
severity: Severity
field: str                   # Field name
record_type: str            # Which record type (File Header, Entry Detail, etc.)
line_number: int            # Line in file
message: str                # Human-readable error message
value: Optional[str]        # Actual value found
expected: Optional[str]     # Expected value(s)
```

**`ACHValidationReport`**
Complete validation report. Includes:
```python
is_valid: bool              # True if no errors
errors: List[ValidationResult]
warnings: List[ValidationResult]
info: List[ValidationResult]
statistics: Dict            # Summary stats
```

#### NACHA Field Specifications (Constants)
```python
FILE_HEADER_FIELDS = {
    "record_type": (1, 1, "N", "1"),        # Position 1, numeric, must be "1"
    "priority_code": (2, 3, "N", "01"),
    "immediate_destination": (4, 13, "AN", None),  # 1-based positions
    ...
}
# Similar specs for BATCH_HEADER_FIELDS, ENTRY_DETAIL_FIELDS, etc.

VALID_SERVICE_CLASSES = {"200", "220", "225"}
VALID_TRANSACTION_CODES = {
    "22", "23", "24", "27", "28", "29",    # Checking
    "32", "33", "34", "37", "38", "39",    # Savings
    "42", "43", "44", "47", "48", "49",    # GL
    "52", "53", "54", "55",                # Loan
}
VALID_SEC_CODES = {"PPD", "CCD", "CTX", "WEB", "TEL", "COR", "NOC", "RCK", "ARC", ...}
VALID_ADDENDA_TYPE_CODES = {"02", "05", "98", "99"}
```

#### Class: `ACHValidator`
Main validator class for ACH files.

**Key Method:**

- `validate(content: str) -> ACHValidationReport`
  - Main entry point. Validates entire ACH file.
  - Steps:
    1. Parse lines and remove padding (9-filled lines)
    2. Check overall structure (starts with 1, ends with 9)
    3. Validate each record type individually
    4. Cross-validate batch/file totals
    5. Build statistics
  - Returns: Complete ACHValidationReport

**Validation Steps (Private Methods):**

- `_validate_structure(lines, report)`
  - File must start with Record Type 1 (File Header)
  - File must end with Record Type 9 (File Control)
  - Total records should be multiple of 10 (blocking factor)

- `_validate_file_header(line, ln, report)`
  - Record size must be 94
  - Blocking factor must be 10
  - File creation date format: YYMMDD
  - File creation time format: HHMM (optional)
  - Immediate destination: 9-10 digit ABA routing

- `_validate_batch_header(line, ln, report)`
  - Service class code: 200/220/225
  - SEC code: valid (PPD, CCD, etc.)
  - Effective entry date: YYMMDD format
  - ODFI routing: 8 digits
  - Company ID: 10 chars, alphanumeric

- `_validate_entry_detail(line, ln, report)`
  - Transaction code: valid (22, 27, etc.)
  - RDFI routing: 8 digits
  - Check digit: 0-9, passes ABA validation
  - Amount: exactly 10 digits (cents, no decimal)
  - Addenda indicator: 0 or 1
  - Trace number: 15 digits

- `_validate_addenda(line, ln, report)`
  - Addenda type code: 02, 05, 98, 99
  - Payment info: up to 80 characters
  - Sequence number: 4 digits

- `_validate_batch_control(line, ln, report)`
  - Service class code: valid
  - Entry/addenda count: numeric
  - Entry hash (routing sum): 10 digits
  - Debit/credit amounts: 12 digits each

- `_validate_file_control(line, ln, report)`
  - Batch count: numeric, 6 digits
  - Block count: numeric, 6 digits
  - Entry/addenda count: 8 digits
  - Entry hash: 10 digits
  - Debit/credit totals: 12 digits each

- `_cross_validate_batch(batch, ln, report)`
  - Entry count in control matches actual entries
  - Entry hash calculated from routing numbers matches
  - Debit/credit totals match entries

- `_cross_validate_file(file_header, file_control, batches, total_lines, report)`
  - Batch count in control matches actual batches
  - File total matches sum of batch totals

**Helper Methods:**

- `_validate_routing_check_digit(routing: str) -> bool`
  - ABA routing number validation using check digit formula
  - Formula: (3*d0 + 7*d1 + 1*d2 + 3*d3 + 7*d4 + 1*d5 + 3*d6 + 7*d7 + 1*d8) % 10 == 0

- `_build_statistics(file_header, batches, file_control, lines) -> Dict`
  - Returns: total records, batches, entries, file_id, creation_date, SEC codes

**Example Usage:**
```python
validator = ACHValidator()
report = validator.validate(file_content)

if report.is_valid:
    print("File is valid!")
    print(f"Batches: {len(report.statistics['total_batches'])}")
else:
    for error in report.errors:
        print(f"Line {error.line_number}: {error.message}")
```

---

### `validators/vcf_validator.py` (453 lines)
Validates VISA VCF (VisaNet Custom File) format.

#### VCF Specifications (Constants)
```python
VALID_TRANSACTION_CODES = {
    "05": "Authorization Request",
    "06": "Financial Transaction",
    "10": "Full Reversal",
    "25": "Chargeback",
    ...  # 13 codes total
}

VALID_CURRENCY_CODES = {
    "840": "USD", "978": "EUR", "826": "GBP", "392": "JPY", ...
}  # ISO 4217 currency codes

VALID_PROCESSING_CODES = {
    "000000": "Purchase",
    "200000": "Credit/Refund",
    ...
}

VALID_POS_ENTRY_MODES = {
    "00": "Unknown",
    "02": "Magnetic stripe (no CVV)",
    "05": "Chip (ICC)",
    "90": "Magnetic stripe",
    ...
}

VALID_RESPONSE_CODES = {
    "00": "Approved",
    "01": "Refer to issuer",
    "51": "Insufficient funds",
    "54": "Expired card",
    ...  # 20+ codes
}
```

#### Report Classes (Similar to ACH)
- `VCFValidationResult`
- `VCFValidationReport`

#### Class: `VCFValidator`
Main validator for VCF files. Supports two formats.

**Key Method:**

- `validate(content: str) -> VCFValidationReport`
  - Auto-detects format (delimited vs. fixed-width)
  - Delegates to appropriate validator

**Format Detection:**
```python
is_delimited = '|' in lines[0] or ',' in lines[0] or '\t' in lines[0]
# If contains pipe, comma, or tab → delimited
# Otherwise → fixed-width
```

**Delimited Format Validation:**

- `_validate_delimited(lines, report)`
  - Delimiter auto-detected: pipe (|), comma, or tab
  - Line 1: Header record
  - Lines 2..N-1: Transaction records (minimum 20 fields each)
  - Line N: Trailer record

- `_validate_vcf_header_delimited(fields, ln, report)`
  - Field 0: File type ("VCF", "VISA", "VISANET", etc.)
  - Field 1: Version (format: "X.Y")
  - Field 2: Acquirer BIN (6 digits)
  - Field 3: Creation date (YYYYMMDD)

- `_validate_transaction_record(fields, ln, report)`
  - Field 0: Transaction Code (valid values)
  - Field 1: PAN (13-19 digits, may be masked with *, passes Luhn if not masked)
  - Field 2: Processing Code (optional)
  - Field 3: Amount (non-negative, < 999999999.99)
  - Field 4: Currency (ISO 4217 code)
  - Field 5: DateTime (YYYYMMDDHHMMSS format)
  - Field 6: MCC (4 digits)
  - Field 7: POS Entry Mode (validated)
  - Field 8: Response Code (validated)
  - Field 9: Authorization Code (6 alphanumeric)
  - Field 10: Merchant ID (≤15 alphanumeric)
  - Field 11: Terminal ID (≤8 alphanumeric)
  - Field 14: ARN (23 digits)
  - Field 16: Interchange Fee (numeric)

- `_validate_vcf_trailer_delimited(fields, ln, expected_count, report)`
  - Must be labeled "TRAILER", "TRL", "99", or "EOF"
  - Transaction count should match header

**Fixed-Width Format Validation:**

- `_validate_fixed_header(line, ln, report)`
  - First 2 chars: "00"
  - Bytes 4-12: File date (YYYYMMDD)

- `_validate_fixed_transaction(line, ln, tc, report)`
  - First 2 chars: Valid transaction code
  - Record length ≥ 50 characters

- `_validate_fixed_trailer(line, ln, report)`
  - First 2 chars: "99"
  - Record length ≥ 20 characters

**Helper Methods:**

- `_luhn_check(card_number: str) -> bool`
  - Validates PAN (card number) using Luhn algorithm
  - Used for unmasked PANs only

- `_build_stats(lines, is_delimited) -> Dict`
  - Returns: format, delimiter, transaction counts, code distribution

**Example Usage:**
```python
validator = VCFValidator()
report = validator.validate(vcf_content)

if report.is_valid:
    print("VCF file is valid!")
    print(f"Transactions: {report.statistics['transaction_count']}")
    print(f"Format: {report.statistics['format']}")
else:
    for error in report.errors[:5]:  # Show first 5 errors
        print(f"Line {error.line_number}: {error.message}")
```

---

## Package Initializers

- `data/__init__.py`: Marks the data package.
- `db/__init__.py`: Marks the database package.
- `model/__init__.py`: Marks the model package.
- `validators/__init__.py`: Marks the validators package.

---

## Quick Start for Developers

### Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Optional: Install PyTorch for GPU acceleration**
   ```bash
   # CPU version (recommended for beginners)
   pip install torch torchvision torchaudio

   # GPU version (CUDA 12.1)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Set up Oracle (optional)**
   ```bash
   # If you have an Oracle database:
   export ORACLE_HOST=your-host
   export ORACLE_USER=your-user
   export ORACLE_PASSWORD=your-password
   # See db/oracle_connector.py for all env vars
   ```

### Running the Application

```bash
# Start the Flask web server
python app.py

# Server runs on http://localhost:5000
```

### Common Tasks for Developers

#### Task 1: Validate an ACH File Programmatically
```python
from validators.ach_validator import ACHValidator

# Read your ACH file
with open('my_file.ach', 'r') as f:
    content = f.read()

# Validate
validator = ACHValidator()
report = validator.validate(content)

# Check results
if report.is_valid:
    print("✓ File is valid!")
else:
    print(f"✗ Found {len(report.errors)} errors")
    for error in report.errors:
        print(f"  Line {error.line_number}: {error.message}")

# View statistics
print(f"Batches: {report.statistics['total_batches']}")
print(f"Entries: {report.statistics['total_entries']}")
```

#### Task 2: Generate Synthetic ACH Files
```python
from data.generator import ACHGenerator

gen = ACHGenerator()

# Generate a single file
file1 = gen.generate_file(
    num_batches=2,           # 2 batches per file
    entries_per_batch=5,     # 5 transactions per batch
    sec_code="PPD"           # Payroll Direct Deposit
)

# Generate a dataset for training
dataset = gen.generate_dataset(n=200)  # 200 files
print(f"Generated {len(dataset)} files for training")
```

#### Task 3: Train a Language Model
```python
from trainer import Trainer

trainer = Trainer(file_type="ACH", config={
    "model_config": "small",
    "n_training_files": 300,
    "n_val_files": 40,
    "batch_size": 4,
    "max_epochs": 5,
    "learning_rate": 3e-4,
    "save_dir": "trained_models/",
})

# Optional: Set up callback to track progress
def progress_callback(message, progress):
    print(f"[{progress}%] {message}")

# Train
result = trainer.train(callback=progress_callback)
print(f"Best validation loss: {result['best_val_loss']:.4f}")
print(f"Model saved to: {result['model_path']}")
```

#### Task 4: Generate Files Using a Trained Model
```python
from trainer import FileGenerator

# Load trained model
gen = FileGenerator(file_type="ACH", model_dir="trained_models/")

# Generate new files using the trained model
new_file = gen.generate_with_seed(
    temperature=0.8,      # Some randomness
    max_tokens=2000       # Max length
)

# Save to file
with open('generated_ach.txt', 'w') as f:
    f.write(new_file)
```

#### Task 5: Using the Oracle Integration
```python
from db.oracle_connector import OracleConfig, get_pool
from data.oracle_ach_generator import OracleACHGenerator

# Configure Oracle connection
config = OracleConfig()
# Or set environment variables: ORACLE_HOST, ORACLE_USER, ORACLE_PASSWORD

# Generate ACH from real Oracle data
gen = OracleACHGenerator(config)

try:
    result = gen.generate(
        company_id=None,           # All companies
        sec_code="PPD",            # Payroll
        max_transactions=100,
        save_audit=True,           # Log to ACH_FILE_LOG table
        save_corpus=True,          # Save to ACH_TRAINING_CORPUS
    )
    
    print(f"Generated file: {result['file_name']}")
    print(f"Entries: {result['entry_count']}")
    print(f"Total debit: ${result['total_debit']/100:.2f}")
    print(f"Total credit: ${result['total_credit']/100:.2f}")
    
except Exception as e:
    print(f"Note: {e}")
    print("(Falls back to synthetic data if DB unavailable)")
```

---

## API Examples (Using curl or Postman)

### Validate an ACH File
```bash
curl -X POST http://localhost:5000/api/validate \
  -F "file=@sample.ach" \
  -F "file_type=ACH"
```

Response:
```json
{
  "is_valid": true,
  "summary": {
    "errors": 0,
    "warnings": 0,
    "info": 0
  },
  "statistics": {
    "total_records": 42,
    "total_batches": 2,
    "total_entries": 10,
    "sec_codes": ["PPD"]
  }
}
```

### Generate a Synthetic ACH File
```bash
curl -X POST http://localhost:5000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "file_type": "ACH",
    "use_model": false,
    "num_batches": 2,
    "entries_per_batch": 5,
    "sec_code": "PPD"
  }'
```

### Start Model Training
```bash
curl -X POST http://localhost:5000/api/train \
  -H "Content-Type: application/json" \
  -d '{
    "file_type": "ACH",
    "model_config": "small",
    "n_files": 200,
    "batch_size": 4,
    "max_epochs": 5
  }'
```

### Check Training Status
```bash
curl http://localhost:5000/api/train/status
```

Response:
```json
{
  "running": true,
  "file_type": "ACH",
  "message": "Epoch 3/5 | train=2.1234 | val=2.3456",
  "progress": 60
}
```

### Get Specification
```bash
curl http://localhost:5000/api/spec
```

---

## Understanding the Architecture

### Data Flow: File Generation
```
request (file_type="ACH", use_model=true)
    ↓
app.py/generate_file()
    ├→ If use_model: Load FileGenerator
    │   ├→ Load trained model (.pt or .pkl)
    │   └→ Generate with model.generate_with_seed()
    │
    └→ If !use_model: Use ACHGenerator
        ├→ Generate synthetic data with rules
        └→ Return 94-char-per-record NACHA file
    ↓
response (content, file_type, char_count, line_count)
```

### Data Flow: Model Training
```
request (file_type="ACH", config...)
    ↓
trainer.py: Trainer.train()
    ├→ Generate synthetic files (ACHGenerator)
    ├→ Tokenize with FinancialTokenizer
    ├→ Train Model (Transformer or Bigram)
    │   ├→ For each epoch:
    │   │   ├→ Forward pass
    │   │   ├→ Backward (gradients)
    │   │   ├→ Optimizer step
    │   │   ├→ Validation evaluation
    │   │   └→ Save best model
    │   └→ Return best_loss and model_path
    ├→ Save tokenizer and model
    └→ Return result dict
    ↓
response (model_path, best_val_loss, train_losses)
```

### Data Flow: Database-Backed Generation
```
request (company_id, sec_code, max_transactions...)
    ↓
OracleACHGenerator.generate()
    ├→ ACHRepository.get_pending_transactions()
    │   └→ Oracle SELECT or mock data
    ├→ Group by (company_id, sec_code, service_class_code)
    ├→ _build_records()
    │   ├→ File Header (type 1)
    │   ├→ For each batch:
    │   │   ├→ Batch Header (type 5)
    │   │   ├→ Entry Details (type 6)
    │   │   ├→ Addenda (type 7, if present)
    │   │   └→ Batch Control (type 8)
    │   ├→ File Control (type 9)
    │   └→ Padding to multiples of 10
    ├→ ACHRepository.log_file() → ACH_FILE_LOG table
    ├→ ACHRepository.save_corpus_entry() → ACH_TRAINING_CORPUS table
    └→ Return result dict
    ↓
response (content, file_name, batch_count, entry_count, file_id)
```

---

## Troubleshooting

### PyTorch Not Installed
**Problem:** "ModuleNotFoundError: No module named 'torch'"  
**Solution:** Falls back to Bigram model automatically. To use Transformer:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Oracle Connection Fails
**Problem:** "Connection failed: ORA-12514: TNS:listener does not recognize..."
**Solution:** App automatically falls back to mock mode. Provides warning but continues:
```python
# In app output:
# Oracle pool failed (localhost:1521/ORCL) — falling back to MOCK mode
```

### Model Takes Too Long to Train
**Problem:** Training on CPU is slow  
**Solution:**
- Use a smaller config: `"nano"` instead of `"medium"`
- Reduce dataset size: `n_training_files=100`
- Reduce epochs: `max_epochs=3`
- Use GPU if available (install GPU PyTorch)

### Generated Files Fail Validation
**Problem:** Generated ACH file doesn't validate  
**Solution:**
- Check that model was trained on valid files
- Try rule-based generator first (use_model=false)
- Compare to sample files in `trained_models/` folder

### ORM/Memory Issues
**Problem:** "Out of memory" during training  
**Solution:**
- Reduce `block_size` (sequence length)
- Reduce `batch_size`
- Use `"nano"` config instead of `"small"`

---

## Dependencies

### Required
- **Flask**: Web framework
- **Werkzeug**: File upload handling
- **NumPy**: Scientific computing (fallback model)

### Optional
- **PyTorch**: Transformer model (recommended)
- **python-oracledb** or **cx_Oracle**: Oracle DB (optional)

See `requirements.txt` for versions.

---

## File Structure Summary
```
slm_ach_vcf-claude/
├── app.py                      # Flask web server (486 lines)
├── trainer.py                  # Training pipeline (281 lines)
├── model/
│   ├── transformer.py          # Transformer model (208 lines)
│   ├── tokenizer.py            # Tokenizer (114 lines)
│   └── bigram_fallback.py      # Bigram fallback (108 lines)
├── data/
│   ├── generator.py            # Synthetic data (416 lines)
│   └── oracle_ach_generator.py # Oracle-backed generation (382 lines)
├── db/
│   ├── oracle_connector.py     # DB connection pooling (319 lines)
│   └── ach_repository.py       # Data access layer (457 lines)
├── validators/
│   ├── ach_validator.py        # ACH validation (427 lines)
│   └── vcf_validator.py        # VCF validation (453 lines)
├── templates/
│   └── index.html              # Web UI
├── trained_models/             # Saved models (auto-created)
├── uploads/                    # Temp file storage (auto-created)
├── HELP.MD                     # This file
├── README.md                   # Setup guide
└── requirements.txt            # Pip dependencies
```

---

## For Novice Developers

### Key Concepts

1. **ACH (Automated Clearing House)**
   - Electronic bank transfers
   - Fixed 94-character record format
   - Batches group transactions by company and purpose
   - Check digits validate data integrity

2. **VCF (VISA Custom File)**
   - Payment card transaction file format
   - Pipe or comma delimited
   - Contains authorization codes, merchant info, amounts

3. **Language Models for Finance**
   - Train on valid file examples
   - Learn patterns and constraints
   - Generate new valid files
   - Both deep learning (Transformer) and statistical (Bigram) approaches

4. **Tokenization**
   - Convert text to numbers (tokens)
   - Character-level: one token per character
   - Enables models to process text

5. **Validation**
   - Check structure (record types, field positions)
   - Validate formats (dates, amounts, codes)
   - Cross-check totals and counts
   - Return errors, warnings, statistics

### Learning Path
1. Start with `data/generator.py` to understand file structures
2. Use `validators/ach_validator.py` to learn validation rules
3. Explore `db/chroma_client.py` to understand ChromaDB integration
4. Read `data/chroma_ach_generator.py` for transaction-to-file generation
5. Try `data/chroma_trainer.py` to understand training with real data
6. Explore `trainer.py` to understand basic (synthetic-only) training pipeline
7. Try `app.py` endpoints via curl or Postman
8. Read model classes `transformer.py` and `bigram_fallback.py`

---

*This comprehensive documentation is designed to help novice developers understand and work with the FinSLM project. Start with the Quick Start section, then dive into specific areas based on your task.*

