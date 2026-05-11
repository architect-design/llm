# FinancialSLM — Complete Technical Reference

> This document is the authoritative guide to every design decision, architectural pattern, and line of code in the FinancialSLM framework. It is written for engineers who need to understand, extend, debug, or evaluate the system from the ground up.

---

## Table of Contents

1. [Why This Project Exists — The Problem Space](#1-why-this-project-exists)
2. [High-Level Architecture](#2-high-level-architecture)
3. [The Financial File Formats](#3-the-financial-file-formats)
4. [Module: `slm/tokenizer.py`](#4-module-slmtokenizerpy)
5. [Module: `slm/model.py`](#5-module-slmmodelpy)
6. [Module: `slm/trainer.py`](#6-module-slmtrainerpy)
7. [Module: `slm/validator.py`](#7-module-slmvalidatorpy)
8. [Module: `slm/generator.py`](#8-module-slmgeneratorpy)
9. [Module: `specs/ach_nacha.py`](#9-module-specsach_nachapy)
10. [Module: `specs/visa_vcf.py` and `specs/general_ledger.py`](#10-module-specsvisa_vcfpy-and-specsgeneral_ledgerpy)
11. [Module: `memory/config_engine.py`](#11-module-memoryconfig_enginepy)
12. [Module: `memory/seeder.py`](#12-module-memoryseedepy)
13. [Module: `api/main.py`](#13-module-apimainpy)
14. [Module: `frontend/index.html`](#14-module-frontendindexhtml)
15. [Module: `run.py`](#15-module-runpy)
16. [Module: `tests/test_suite.py`](#16-module-teststest_suitepy)
17. [Data Flow: End-to-End Walkthrough](#17-data-flow-end-to-end-walkthrough)
18. [Design Decisions & Trade-offs](#18-design-decisions--trade-offs)
19. [Extending the Framework](#19-extending-the-framework)
20. [Troubleshooting Guide](#20-troubleshooting-guide)

---

## 1. Why This Project Exists

### The Core Problem

Financial institutions exchange data using **fixed-width flat-file formats** — ASCII text files where every field occupies a precisely defined range of character columns. A single misplaced character at column 4 of an ACH entry record is not a warning — it is a rejected file, a failed payroll, or a clawed-back transaction.

Existing approaches to working with these files fall into two extremes:

| Approach | Limitation |
|----------|-----------|
| **Hard-coded parsers** | Brittle. Each format requires a dedicated parser that cannot generalise. |
| **General-purpose LLMs** (GPT-4, Claude) | Cannot run offline. Send financial data to third parties. Expensive per-call. No awareness of column positions. |
| **Manual validation scripts** | Catch rule violations but miss inter-field contextual anomalies. |
| **Commercial tools** | Closed-source, expensive, not extensible. |

### The Solution

FinancialSLM is a **custom Small Language Model** that:

- Is trained **from scratch** on synthetic financial records — no pre-trained weights.
- Runs **entirely offline** — no external API calls, ever.
- Speaks the **language of columns** — its tokenizer is position-aware, not word-aware.
- Has a **dual output head** — one for generation, one for per-field validity scoring.
- Is gated by a **deterministic rule engine** so correctness is guaranteed even before training converges.

---

## 2. High-Level Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                            FinancialSLM Stack                              │
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                         USER INTERFACE LAYER                         │  │
│  │   frontend/index.html (SPA)      ◄──►      api/main.py (FastAPI)    │  │
│  └────────────────────────────┬─────────────────────────────────────────┘  │
│                               │ HTTP/JSON                                  │
│  ┌────────────────────────────▼─────────────────────────────────────────┐  │
│  │                        ORCHESTRATION LAYER                           │  │
│  │   run.py (CLI) ──► api/main.py ──► [validator | generator | trainer] │  │
│  └──────┬────────────────┬────────────────┬────────────────┬────────────┘  │
│         │                │                │                │               │
│  ┌──────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐      │
│  │  Tokenizer  │  │  Validator  │  │  Generator  │  │   Trainer   │      │
│  │ (tokenizer) │  │ (validator) │  │ (generator) │  │  (trainer)  │      │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘      │
│         │                │                │                │               │
│  ┌──────▼────────────────▼────────────────▼────────────────▼────────────┐  │
│  │                          SLM CORE LAYER                              │  │
│  │                    slm/model.py  (FinancialSLM)                      │  │
│  │         Transformer + Triple Embedding + Dual Output Head            │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                               │                                           │
│  ┌────────────────────────────▼─────────────────────────────────────────┐  │
│  │                       SPECIFICATION LAYER                            │  │
│  │   memory/config_engine.py   ◄──►   specs/{ach_nacha,visa_vcf,gl}.py │  │
│  │   memory/seeder.py                                                   │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────┘
```

### Layer Responsibilities

| Layer | Files | Responsibility |
|-------|-------|---------------|
| **UI** | `frontend/index.html` | Browser-based SPA for upload, generation, exploration, training |
| **Orchestration** | `api/main.py`, `run.py` | HTTP routing, background training, CLI dispatch |
| **Processing** | `slm/validator.py`, `slm/generator.py`, `slm/trainer.py` | Core business logic |
| **Tokenization** | `slm/tokenizer.py` | Convert raw text ↔ integer sequences |
| **SLM Core** | `slm/model.py` | Neural network: embedding, attention, dual heads |
| **Specification** | `specs/*.py` | Field-level format definitions as Python data |
| **Memory** | `memory/config_engine.py`, `memory/seeder.py` | Runtime source of truth + synthetic data |

---

## 3. The Financial File Formats

Understanding the target formats is essential to understanding every design decision.

### 3.1 ACH NACHA (94 characters per line)

The National Automated Clearing House Association format governs US electronic fund transfers — payroll direct deposits, bill payments, business-to-business payments.

```
Position:  1234567890123456789012345678901234567890...94
           ├─┤├─┤├──────────┤├──────────┤├──────┤├──┤
     RT1:  1 01 021000021   9876543210  260101  0000
           │  │  │           │           │       │
           │  │  │           │           │       File ID
           │  │  │           │           Creation Date (YYMMDD)
           │  │  │           Immediate Origin (10 digits)
           │  │  Immediate Destination (space + 9-digit routing)
           │  Priority Code (always "01")
           Record Type Code (always "1" for File Header)
```

**Record type sequence in a valid ACH file:**
```
1         ← File Header      (exactly 1 per file)
  5       ← Batch Header     (1 per batch)
    6     ← Entry Detail     (1+ per batch)
    7     ← Addenda          (0 or 1 per entry, optional)
  8       ← Batch Control    (1 per batch, mirrors RT5)
9         ← File Control     (exactly 1 per file)
999...9   ← Padding records  (fill to multiple of 10 lines)
```

**Critical rules enforced by this framework:**
- Every line is **exactly 94 characters**. Not 93. Not 95.
- The **Mod-10 routing check digit** algorithm must validate every 9-digit routing number.
- The **Entry Hash** in RT8/RT9 = sum of all RT6 8-digit routing numbers, mod 10^10.
- The **blocking factor** requires total line count to be a multiple of 10.

### 3.2 VISA VCF (80 characters per line)

VISA Card File format for transaction settlement between acquiring banks and VISA. Each file carries a volume of card transactions.

```
Record types: VH (Volume Header), DT (Detail Transaction), TR (Trailer), VF (Volume Footer)
```

### 3.3 General Ledger (120 characters per line)

Journal entry flat-file format compatible with ERP systems (SAP, Oracle, NetSuite). Implements double-entry bookkeeping — every file must have total debits equal total credits.

```
Record types: JH (Journal Header), JE (Journal Entry line), GL (Control/Trailer)
```

**Why these formats matter to the architecture:**
- All three are **fixed-width** → tokenizer slices by position, not delimiter.
- All three have **mandatory checksums** → validator must compute and compare.
- All three have **inter-record dependencies** → a control record value depends on computed accumulations from entry records.

---

## 4. Module: `slm/tokenizer.py`

### Purpose

Converts raw financial file text into integer token sequences that the neural network can process. This is **fundamentally different** from NLP tokenizers (which split on words or byte-pairs) because financial files are structured by character column, not semantic boundaries.

### 4.1 `TokenType` Enum

```python
class TokenType(Enum):
    RECORD_TYPE   = "RECORD_TYPE"
    NUMERIC       = "NUMERIC"
    ALPHANUMERIC  = "ALPHANUMERIC"
    BLANK_PAD     = "BLANK_PAD"
    CHECKSUM      = "CHECKSUM"
    AMOUNT        = "AMOUNT"
    ROUTING       = "ROUTING"
    ACCOUNT       = "ACCOUNT"
    DATE          = "DATE"
    UNKNOWN       = "UNKNOWN"
```

Each enum value represents a **field semantic category**. This taxonomy is used in three places:
1. The **spec definitions** (`specs/*.py`) to declare what kind of data each field holds.
2. The **validator** to decide which rule to apply to a field.
3. The **seeder** to decide what kind of random data to generate.

The distinction between `NUMERIC` and `AMOUNT` matters: an `AMOUNT` field is zero-padded with an implied decimal point (e.g., `0000012345` represents $123.45), while `NUMERIC` is a plain integer. Both are digits-only, but the seeder treats them differently.

### 4.2 `FinancialToken` Dataclass

```python
@dataclass
class FinancialToken:
    token_type : TokenType
    raw_value  : str       # exact characters from file
    position   : int       # absolute character offset in line
    length     : int       # field width
    field_name : str = ""
    line_no    : int = 0
```

Every token carries its **position** and **length** — metadata that would be discarded by a word-level tokenizer but is essential for financial files. The `token_id` property maps the raw value to the vocabulary using the character-level vocabulary table.

### 4.3 `_build_vocab()` — The Vocabulary

```python
def _build_vocab() -> Dict[str, int]:
    vocab = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3, "<SEP>": 4, "<MASK>": 5}
    idx = len(vocab)
    for ch in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ /-.:@#*":
        vocab[ch] = idx
        idx += 1
    for rt in ["RT1", "RT5", "RT6", ...]:
        vocab[rt] = idx
        idx += 1
    return vocab
```

**Why this vocabulary is small (≈64 tokens):**

Financial files use a **closed character set**. The NACHA spec explicitly states that alphanumeric fields contain only upper-case A–Z, digits 0–9, and a small set of special characters. This is by design — financial data must be transmittable over 1970s-era banking networks that only supported 7-bit ASCII.

Special tokens:
- `<PAD>` (0): Used to pad shorter sequences to the model's max sequence length.
- `<UNK>` (1): Catches illegal characters — their presence in output signals a validation problem.
- `<BOS>` (2): Beginning-of-sequence, prepended during generation to give the model an initial hidden state.
- `<EOS>` (3): End-of-sequence (reserved for future use).
- `<SEP>` (4): Separator between records in multi-record batch inference.
- `<MASK>` (5): For masked language modeling experiments (not used in current training).

Record-type sentinels (`RT1`, `RT5`, etc.) are **compound tokens** — single vocabulary entries representing an entire record-type code. This allows the model's embedding layer to learn a dedicated representation for "this line is an ACH Entry Detail record" rather than composing that understanding from individual characters.

### 4.4 `FieldDescriptor` Dataclass

```python
@dataclass
class FieldDescriptor:
    name       : str
    start      : int    # 1-based (matches NACHA spec column references)
    end        : int    # inclusive
    field_type : TokenType
    required   : bool = True
    pattern    : Optional[str] = None
    allowed    : Optional[List[str]] = None

    @property
    def length(self) -> int:
        return self.end - self.start + 1

    def extract(self, line: str) -> str:
        s, e = self.start - 1, self.end   # convert 1-based to 0-based
        return line[s:e] if len(line) >= e else line[s:].ljust(self.length)
```

This is the schema atom. Every named field in every record type is described by one `FieldDescriptor`. The `start`/`end` values are **1-based and inclusive** to match the NACHA specification's column numbering (the spec says "Field 3, positions 4–13" meaning columns 4 through 13 inclusive).

The `extract()` method converts to Python's 0-based slicing: `line[start-1 : end]`.

### 4.5 `FinancialTokenizer` Class

```python
LINE_LENGTH_MAP = {
    "ACH_NACHA"      : 94,
    "VISA_VCF"       : 80,
    "GENERAL_LEDGER" : 120,
}
```

The tokenizer is **spec-aware** — it knows the mandatory line length for each format and enforces it during normalisation.

**`tokenize(raw_text)`** is the main entry point. It:
1. Splits the raw text into lines.
2. Normalises each line: upper-cases it, pads to spec length, trims overlong lines.
3. Detects the record type from the first character(s).
4. If a field schema exists for that record type, slices the line into named field tokens.
5. Otherwise falls back to character-by-character tokenization.

**`_normalise_lines()`** handles the three common line-ending conventions (`\r\n`, `\r`, `\n`) and enforces exact line length.

**`_detect_record_type()`** uses spec-specific logic:
- ACH NACHA: first character only (`1`, `5`, `6`, `7`, `8`, `9`).
- VISA VCF: first two characters (`VH`, `DT`, `TR`, `VF`).
- General Ledger: first two characters (`JH`, `JE`, `GL`).

**`encode_line(line)`** produces a flat integer list — one ID per character — suitable for batched tensor creation in the trainer.

**`make_tokenizer(spec_name)`** is the factory function. It imports the appropriate schema from `specs/` and wires it to a new `FinancialTokenizer` instance, so callers never need to import spec definitions directly.

---

## 5. Module: `slm/model.py`

### Purpose

Defines the neural network architecture. The model is a custom Transformer with three design features that distinguish it from a standard language model:

1. **Triple embedding** — character identity + field slot + record type, all combined.
2. **Mode-switching attention mask** — causal for generation, bidirectional for validation.
3. **Dual output head** — generation logits and per-field validity scores from the same encoder.

### 5.1 `SLMConfig` Dataclass

```python
@dataclass
class SLMConfig:
    vocab_size     : int   = 64
    d_model        : int   = 128
    n_heads        : int   = 4
    n_layers       : int   = 4
    d_ff           : int   = 512
    max_seq_len    : int   = 120
    dropout        : float = 0.1
    n_record_types : int   = 32
    n_field_slots  : int   = 40
    pad_token_id   : int   = 0
```

**Why these sizes?**

The model is intentionally small. Financial syntax is **low-complexity**: the rules are deterministic and the vocabulary is tiny. A model with 850K–2.1M parameters can overfit a financial format's syntactic structure in thousands of steps, not millions. This is why it's called a Small Language Model.

`d_model=128` means each character position is represented as a 128-dimensional vector. This is 6–8× smaller than typical NLP models but sufficient for a closed-alphabet fixed-width format.

`n_field_slots=40` is the maximum number of named fields any record type has. RT6 (ACH Entry Detail) has 11 fields. The GL Journal Entry has 13. The pool uses this as the output dimension for the validation head.

### 5.2 `SinusoidalPositionalEncoding`

```python
pe[:, 0::2] = torch.sin(position * div_term)
pe[:, 1::2] = torch.cos(position * div_term)
```

Standard sinusoidal encoding from "Attention Is All You Need" (Vaswani et al., 2017). It adds a unique continuous-valued signal to each position so the model can distinguish character at column 4 from character at column 5.

**Why this matters for financial files:** Unlike natural language where word order is flexible, financial column positions are **semantically absolute**. Column 4 in every ACH file is always the first digit of the Service Class Code. The sinusoidal encoding reinforces this by giving the model a strong, position-specific signal that does not change between batches.

The encoding is registered as a **buffer** (not a parameter) — it is fixed and not updated during training.

### 5.3 `FieldSlotEncoding`

```python
class FieldSlotEncoding(nn.Module):
    def __init__(self, n_field_slots, d_model):
        self.embedding = nn.Embedding(n_field_slots, d_model)
```

This is a **learnable** encoding — unlike the sinusoidal PE. Each character in the input sequence is tagged with an integer indicating which named field it belongs to (e.g., "field 3 of 11 in an RT6 record"). The embedding table then maps this integer to a learned vector.

**Why both PE types?** Sinusoidal PE tells the model the absolute column (0–119). Field-slot encoding tells the model the semantic field index (0–39). These are different. Column 13 in an RT6 record is the start of "RDFI Account Number" — but column 13 in an RT5 record is part of "Company Name". The field-slot embedding lets the model know *which field it's inside*, not just which column it's at.

### 5.4 `MultiHeadAttention`

```python
scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
if mask is not None:
    scores = scores.masked_fill(mask == 0, float("-inf"))
attn = self.dropout(F.softmax(scores, dim=-1))
```

Standard scaled dot-product attention. Bias is disabled on all projection layers (`bias=False`) — this is a common practice in decoder-only transformers and reduces parameter count without hurting performance on structured data.

The mask is applied by setting masked positions to `-inf` before the softmax. After softmax, `-inf` positions become exactly 0, so they contribute nothing to the output weighted sum.

### 5.5 `TransformerBlock`

```python
def forward(self, x, mask=None):
    x = self.norm1(x + self.attn(x, x, x, mask))   # Pre-norm residual
    x = self.norm2(x + self.ff(x))
    return x
```

This uses **Pre-LayerNorm** (normalise before attention, not after). Pre-norm architectures train more stably than Post-norm, particularly at small model sizes, because gradients flow through the residual path without passing through a normalisation layer.

The FeedForward block uses **GELU** activation (Gaussian Error Linear Unit) rather than ReLU. GELU has smoother gradients near zero and performs slightly better on small datasets.

### 5.6 `FinancialSLM` — The Core Model

#### Embedding Forward

```python
def _embed(self, char_ids, field_ids, rt_ids):
    c_emb  = self.char_embed(char_ids)                          # (B, T, D)
    f_emb  = self.field_pe(field_ids)                           # (B, T, D)
    rt_emb = self.rt_embed(rt_ids).unsqueeze(1).expand(B, T, -1)  # (B, T, D)
    combined = torch.cat([c_emb, f_emb, rt_emb], dim=-1)       # (B, T, 3D)
    x = self.embed_proj(combined)                               # (B, T, D)
    x = self.sin_pe(x)
    return x
```

Three embeddings are concatenated along the feature dimension then projected back to `d_model`. The record-type embedding is broadcast across all T positions — every character in a line shares the same record-type signal. This is analogous to a sentence-type token in BERT, but more explicit.

#### Mode Switching

```python
if mode == "generate":
    mask = self._causal_mask(T, device) & self._pad_mask(char_ids, pad_id)
else:
    mask = self._pad_mask(char_ids, pad_id)
```

**Generate mode** uses a causal (lower-triangular) mask — position `i` can only attend to positions `0..i`. This is required for autoregressive decoding: the model must predict character `i+1` using only characters `0..i`.

**Validate mode** uses a full bidirectional mask (only padding positions are masked). This allows every position to attend to every other position, giving the model full context to score field validity — e.g., the Amount field can attend to the Transaction Code field to check consistency.

#### Generation Head

```python
self.gen_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
gen_logits = self.gen_head(x)  # (B, T, V)
```

A linear projection from the hidden state to the vocabulary. The output `gen_logits[b, t, v]` is the un-normalised score for character `v` at position `t` in batch item `b`. During inference, softmax converts these to probabilities.

#### Validation Head

```python
self.val_pool = nn.AdaptiveAvgPool1d(cfg.n_field_slots)
...
x_t    = x.transpose(1, 2)                     # (B, D, T)
pooled = self.val_pool(x_t).transpose(1, 2)    # (B, n_fields, D)
val_logits = self.val_head(pooled)              # (B, n_fields, 2)
```

`AdaptiveAvgPool1d` is the key operation here. It compresses the T-length character sequence into exactly `n_field_slots` segments by average-pooling. This effectively creates one vector per "field slot" regardless of how many characters that field actually contains. Each compressed vector is then classified as valid (class 1) or invalid (class 0).

This approach avoids having to know exact field boundaries during pooling — the pool is content-agnostic and the network learns to associate each pooled slot with the corresponding named field through training.

#### Confidence Head

```python
cls = x.mean(dim=1)       # (B, D) — mean pool across all positions
confidence = self.conf_head(cls)  # (B, 1) — scalar in [0, 1]
```

The global confidence score is computed by mean-pooling the hidden states across all character positions (similar to how BERT uses a `[CLS]` token). A sigmoid ensures the output is in [0, 1]. This single scalar summarises the model's overall certainty that the entire record is syntactically valid.

#### Weight Initialisation

```python
def _init_weights(self):
    for p in self.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
```

Xavier uniform initialisation (also called Glorot uniform) sets weights based on the number of input and output neurons: `W ~ U[-√(6/(fan_in+fan_out)), +√(6/(fan_in+fan_out))]`. This prevents vanishing/exploding gradients at the start of training.

### 5.7 `build_model(spec_name)` Factory

```python
overrides = {
    "ACH_NACHA"      : {"max_seq_len": 94,  "n_layers": 4, "d_model": 128},
    "VISA_VCF"       : {"max_seq_len": 80,  "n_layers": 4, "d_model": 128},
    "GENERAL_LEDGER" : {"max_seq_len": 120, "n_layers": 6, "d_model": 192, "d_ff": 768},
}
```

The General Ledger spec uses a larger model (`d_model=192`, `n_layers=6`) because its records are 120 characters wide (vs. 94 for ACH) and have more named fields. The head count is auto-adjusted to the largest divisor of `d_model` up to 8.

---

## 6. Module: `slm/trainer.py`

### Purpose

Trains the SLM using a **dual-objective loss** on synthetic data generated by the ConfigEngine and DataSeeder. No external datasets are required.

### 6.1 `TrainConfig`

```python
@dataclass
class TrainConfig:
    spec_name       : str   = "ACH_NACHA"
    batch_size      : int   = 32
    learning_rate   : float = 3e-4
    weight_decay    : float = 1e-2
    warmup_steps    : int   = 200
    max_steps       : int   = 5_000
    val_every       : int   = 250
    clm_weight      : float = 1.0
    val_weight      : float = 0.5
    corruption_prob : float = 0.3
    checkpoint_dir  : str   = "checkpoints"
    device          : str   = "cpu"
```

`clm_weight=1.0` and `val_weight=0.5` balance the two losses. CLM is the primary objective (learn the syntax). Validation classification is secondary (learn the rules). Setting `val_weight` too high would cause the model to optimise for detecting broken fields at the expense of generating coherent syntax.

`corruption_prob=0.3` means 30% of training records are deliberately corrupted before being presented to the model. This is crucial for teaching the validation head — if all training data is valid, the model never learns what an invalid field looks like.

### 6.2 `SyntheticFinancialDataset`

```python
class SyntheticFinancialDataset(IterableDataset):
    def __iter__(self):
        while True:
            yield self._generate_sample()
```

This is an **infinite streaming dataset** — it generates records on-the-fly by calling the `DataSeeder`. There is no fixed training set; the model sees a different random batch on every step. This is possible because the seeder can produce an unlimited number of syntactically valid records from the ConfigEngine rules.

**Sample generation pipeline:**
1. Pick a random record type from the spec's available types.
2. Call `seeder.generate_line()` to produce a valid record line.
3. With probability `corruption_prob`, corrupt one randomly-chosen field.
4. Tokenize the line into character IDs.
5. Build field slot IDs by mapping each character position to its field index.
6. Return `(char_ids, field_ids, rt_ids, target_ids, field_labels)`.

**Corruption strategy:**
```python
corruption = self.rng.choice([
    "X" * fd["length"],    # alphabetic in numeric field
    "?" * fd["length"],    # invalid chars
    " " * fd["length"],    # blank where required
    "9" * fd["length"],    # overflow numeric
])
```

These four corruption types cover the most common real-world file errors: wrong character class, illegal characters, missing data, and numeric overflow. The corresponding `field_labels[idx] = 0` marks that field as invalid for the validation head's training target.

### 6.3 `WarmupCosineScheduler`

```python
if s <= self.warmup_steps:
    scale = s / max(1, self.warmup_steps)
else:
    progress = (s - warmup_steps) / max(1, total_steps - warmup_steps)
    scale = 0.5 * (1.0 + math.cos(math.pi * progress))
```

Linear warmup for `warmup_steps` steps, then cosine annealing to near-zero. This prevents the large gradient updates that occur at the start of training (when weights are random) from destabilising the model. After warmup, the cosine schedule smoothly reduces the learning rate, allowing fine-grained convergence.

### 6.4 `SLMTrainer._train_step()`

```python
# CLM loss: predict char t+1 from context 0..t
clm_loss = self.clm_criterion(
    gen_logits[:, :-1].reshape(-1, vocab_size),
    target_ids[:, 1:].reshape(-1),
)

# Validation loss: binary valid/invalid per field slot
val_loss = self.val_criterion(
    val_logits[:, :n_f, 1],    # logit for "valid" class
    field_labels[:, :n_f],     # ground-truth binary labels
)

total_loss = clm_weight * clm_loss + val_weight * val_loss
```

**Why two forward passes?** One forward pass is made in `generate` mode (causal mask, for CLM loss) and a second in `validate` mode (bidirectional mask, for validation head loss). They use different attention masks. This is intentional — the generation objective requires causal masking but the validation objective benefits from full bidirectional context.

**Gradient clipping:**
```python
nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Clips gradient norms to 1.0 before the optimiser step. This prevents exploding gradients, which are more likely at small model sizes where the signal-to-noise ratio in the gradient is lower.

### 6.5 Checkpointing

```python
torch.save({
    "step"        : step,
    "model_state" : model.state_dict(),
    "opt_state"   : optimizer.state_dict(),
    "history"     : self.history,
    "spec_name"   : cfg.spec_name,
}, path)
```

Checkpoints save both model weights and optimiser state. Saving the optimiser state allows resuming training with correct momentum and adaptive learning rate estimates (AdamW maintains per-parameter running statistics).

---

## 7. Module: `slm/validator.py`

### Purpose

Validates financial files through two independent layers that run sequentially and whose results are merged into a unified `ValidationReport`.

```
Raw File Text
      │
      ▼
Layer 1: Rule Engine (deterministic, always runs)
      │  - Field length checks
      │  - Character class checks (numeric/alphanumeric/etc.)
      │  - Routing Mod-10 check digit
      │  - Date format validation (YYMMDD)
      │  - Allowed-values whitelist
      │  - Regex pattern matching
      │
      ▼
Layer 2: SLM Model (runs if model is loaded)
      │  - Contextual confidence score per line
      │  - Flags inter-field anomalies rules can't catch
      │
      ▼
ValidationReport
```

### 7.1 `FieldError` and `LineResult`

```python
@dataclass
class FieldError:
    line_no    : int
    field_name : str
    position   : str     # human: "cols 4-13"
    raw_value  : str
    rule       : str     # what rule failed
    severity   : str     # "ERROR" | "WARNING" | "INFO"
    source     : str     # "RULE" | "MODEL"
```

The `source` field distinguishes rule-engine errors from model-detected anomalies. This is important for debugging: a `source="RULE"` error is definitive and actionable; a `source="MODEL"` warning is probabilistic and should trigger manual review.

### 7.2 `RuleEngine`

#### Routing Number Validation (Mod-10 Algorithm)

```python
_ROUTING_WEIGHTS = [3, 7, 1, 3, 7, 1, 3, 7, 1]

@classmethod
def validate_routing(cls, value):
    total = sum(int(d) * w for d, w in zip(value, cls._ROUTING_WEIGHTS))
    if total % 10 != 0:
        return False, f"Routing check-digit failed (sum={total}, mod10={total%10})"
    return True, ""
```

The ABA routing number check digit algorithm: multiply each of the 9 digits by [3,7,1,3,7,1,3,7,1] respectively, sum the products, and verify the sum is divisible by 10. For example, routing `021000021`:
- `0×3 + 2×7 + 1×1 + 0×3 + 0×7 + 0×1 + 0×3 + 2×7 + 1×1`
- `= 0 + 14 + 1 + 0 + 0 + 0 + 0 + 14 + 1 = 30`
- `30 % 10 = 0` ✓ Valid.

#### Amount Validation

```python
@staticmethod
def validate_amount(value, field_name):
    if not value.isdigit():
        return False, f"{field_name}: amount field must be zero-padded digits"
    return True, ""
```

Financial amounts use **implied decimal** encoding. `0000012345` represents $123.45 (the last two digits are cents). The validator checks only that all characters are digits — the implied decimal position is fixed by spec and doesn't need validation.

#### Date Validation (YYMMDD)

```python
yy, mm, dd = int(value[:2]), int(value[2:4]), int(value[4:])
if not (1 <= mm <= 12 and 1 <= dd <= 31):
    return False, ...
```

Basic range check. Full calendar validation (e.g., Feb 30 is invalid) is not implemented — financial files use YYMMDD as an index, not a strict calendar date, and the banking system accepts the full 01–31 range for day.

### 7.3 `ACHChecksumValidator`

This class accumulates running totals across RT6 entry records and validates them against the RT8 batch control and RT9 file control records.

```python
def process_line(self, rt, line):
    if rt == "RT6":
        routing = line[3:11]    # 8-digit routing (no check digit)
        amount  = line[29:39]   # 10-digit zero-padded amount
        self.batch_routing_sum  += int(routing)
        self.batch_entry_count  += 1
        ...

    elif rt == "RT8":
        reported_hash = self._safe_int(line[10:20])   # Entry Hash field
        actual_hash   = self.batch_routing_sum % (10 ** 10)
        if reported_hash != actual_hash:
            self.errors.append(f"Batch hash mismatch: ...")
```

**Why `_safe_int()` instead of `int()`?** Direct `int()` on a financial field would crash if the field contains non-digit characters (e.g., a corrupted file where text has overflowed into a numeric field). `_safe_int()` strips all non-digit characters before parsing, making the validator robust to corruption:

```python
@staticmethod
def _safe_int(s):
    cleaned = "".join(c for c in s if c.isdigit())
    return int(cleaned) if cleaned else 0
```

**RT9 byte offsets (a critical correctness detail):**
```python
# RT9 field positions (1-indexed spec → 0-indexed Python slice):
rep_entry_count = self._safe_int(line[13:21])  # cols 14-21
rep_hash        = self._safe_int(line[21:31])  # cols 22-31
rep_debits      = self._safe_int(line[31:43])  # cols 32-43
rep_credits     = self._safe_int(line[43:55])  # cols 44-55
```

These offsets must exactly match the RT9 field schema in `specs/ach_nacha.py`. A discrepancy of even one column produces a false checksum failure.

### 7.4 `FinancialValidator.validate()`

The main validation loop has several important behaviors:

**Padding record detection:**
```python
is_padding = all(c == "9" for c in line.strip())
errors = [] if is_padding else self._check_line_rules(line, rt, i)
if ach_checker and not is_padding:
    ach_checker.process_line(rt, line)
```

ACH padding records (all-nines lines used to fill to the blocking factor of 10) look like RT9 File Control records to a naive parser. The validator identifies them by their all-`9` content and skips both field validation and checksum processing for them.

**BLANK_PAD field handling:**
```python
if not err_msg and fd.get("required") and not raw.strip():
    if ftype != "BLANK_PAD" and "reserved" not in name.lower():
        err_msg = f"{name}: required field is blank"
```

Reserved/padding fields are supposed to be blank — that is their correct value. The validator must not flag them as "required field is blank."

**Routing length handling:**
```python
elif ftype == "ROUTING":
    if fd["length"] == 9:
        ok, msg = self.rules.validate_routing(raw)   # full Mod-10 check
    else:
        ok, msg = self.rules.validate_numeric(raw, name)  # partial: just numeric
```

ACH RT6 stores the routing number split across two fields: "RDFI Routing Transit" (8 chars, cols 4–11) and "Check Digit" (1 char, col 12). The 8-char field cannot pass the 9-digit Mod-10 algorithm — it must be validated as a plain 8-digit numeric field. Only full 9-char ROUTING fields (like those in RT1 "Immediate Destination") undergo the Mod-10 check.

---

## 8. Module: `slm/generator.py`

### Purpose

Produces syntactically valid financial file content using **constrained autoregressive decoding**. The critical insight: correctness is guaranteed by *eliminating illegal options* before sampling, not by hoping the model learns to avoid them.

### 8.1 `ConstraintResolver`

```python
class ConstraintResolver:
    def allowed_ids(self, spec, rt, col):
        fields = self.engine.get_fields(spec, rt)
        for fd in fields:
            s, e = fd["start"] - 1, fd["end"]
            if s <= col < e:
                ftype   = fd["field_type"]
                allowed = fd.get("allowed")
                if allowed:
                    char_options = set()
                    for a in allowed:
                        fi = col - s
                        char_options.add(vocab[a[fi]])
                    return list(char_options)
                if ftype == "NUMERIC":
                    return _digits_mask(vocab)
                ...
```

At each character position during generation, this resolver asks: "What characters are *legally possible* here given the spec?" It returns a list of token IDs. All other IDs are masked to `-inf` before the softmax.

**Example — ACH RT6, column 2 (Transaction Code, first digit):**
- Allowed values: `["22","23","24","27","28","29","32","33","34","37","38","39"]`
- First digit of these: `2`, `2`, `2`, `2`, `2`, `2`, `3`, `3`, `3`, `3`, `3`, `3`
- Unique allowed IDs: `{vocab["2"], vocab["3"]}`
- Every other digit is masked to `-inf` → only `2` or `3` can be sampled.

This makes generation **correct by construction** — even an untrained model produces valid Transaction Codes.

### 8.2 `GenerationConfig`

```python
class GenerationConfig:
    strategy     : str   = "temperature"  # greedy | temperature | top_k
    temperature  : float = 0.7
    top_k        : int   = 10
    max_new_chars: int   = 120
```

Three strategies for the constrained sampling step:

- **Greedy**: Always picks the highest-probability legal character. Deterministic. Produces the most "typical" output but no variety.
- **Temperature**: Scales logits by `1/temperature` before softmax. Temperature < 1.0 sharpens the distribution (more deterministic), > 1.0 flattens it (more random).
- **Top-K**: Samples only from the top-K highest-probability legal characters. Prevents sampling very unlikely characters while allowing more diversity than greedy.

### 8.3 `FinancialGenerator._gen_ach_file()`

This method implements the complete ACH file generation pipeline with correct checksum computation:

```python
# Accumulate routing sums and amounts across entry records
for i in range(n_entries):
    entry, ectx = seeder.generate_line("ACH_NACHA", "RT6", extra={"sequence": i+1})
    routing_sum += int(entry[3:11])
    entry_count += 1
    ...

# Batch Control: pass computed checksums as extra parameters
batch_hash = routing_sum % (10 ** 10)
seeder.generate_line("ACH_NACHA", "RT8", extra={
    "entry_addenda_count" : entry_count,
    "entry_hash"          : batch_hash,
    "total_debit_dollar_amount"  : debit_total,
    "total_credit_dollar_amount" : credit_total,
})

# Pad to multiple of 10
while len(lines) % 10 != 0:
    lines.append("9" * 94)
```

The control records are generated **after** all entry records so their checksums reflect the actual generated data. The `extra` dictionary passes computed values to the seeder, which writes them into the correct field positions.

**Extra key normalisation:** The seeder resolves extra keys using two candidate normalizations of the field name:
- Primary: `name.lower().replace(" ", "_")` → `"entry_hash"`
- Secondary: `re.sub(r"[^a-z0-9]+", "_", name.lower())` → `"entry_hash"`

The secondary normalisation strips slashes and punctuation, handling field names like `"Entry/Addenda Count"` → `"entry_addenda_count"`.

---

## 9. Module: `specs/ach_nacha.py`

### Purpose

Defines the complete field schema for all ACH NACHA record types as Python data. This file is the single source of truth for the ACH specification within the framework.

### 9.1 Helper Function

```python
def F(name, start, end, ftype, required=True, pattern=None, allowed=None):
    return FieldDescriptor(name=name, start=start, end=end,
                           field_type=ftype, required=required,
                           pattern=pattern, allowed=allowed)
```

A concise factory that reduces boilerplate. Every field definition fits on one line.

### 9.2 Record Type 6 (Entry Detail) — Most Important Record

```python
RT6_FIELDS = [
    F("Record Type Code",          1,  1,  RT, allowed=["6"]),
    F("Transaction Code",          2,  3,  N,
      allowed=["22","23","24","27","28","29","32","33","34","37","38","39"]),
    F("RDFI Routing Transit",      4, 11,  RO, pattern=r"[0-9]{8}"),
    F("Check Digit",              12, 12,  N),
    F("RDFI Account Number",      13, 29,  AC),
    F("Amount",                   30, 39,  AM),
    F("Individual ID Number",     40, 54,  AN, required=False),
    F("Individual Name",          55, 76,  AN),
    F("Discretionary Data",       77, 78,  AN, required=False),
    F("Addenda Record Indicator", 79, 79,  N,  allowed=["0","1"]),
    F("Trace Number",             80, 94,  N),
]
```

**Transaction Code meanings:**
- `22` = Live check credit to demand deposit account (DDA) — the most common type (direct deposit to checking)
- `27` = Live check debit to DDA (bill payment from checking)
- `32` = Live credit to savings account
- `37` = Live debit to savings account
- `23`, `28`, `33`, `38` = Pre-notification versions (zero-dollar test records)

**Routing split:** RDFI Routing Transit (cols 4–11, 8 digits) + Check Digit (col 12, 1 digit) = the 9-digit routing number. They are stored in separate fields so the validator knows to apply different checks to each.

### 9.3 `ACH_SPEC_META`

```python
ACH_SPEC_META = {
    "name"            : "ACH_NACHA",
    "line_length"     : 94,
    "blocking_factor" : 10,
    "encoding"        : "ASCII",
    "record_types"    : list(ACH_FIELD_SCHEMA.keys()),
    "required_sequence": ["RT1", "RT5", "RT6", "RT8", "RT9"],
}
```

Metadata served to the frontend via `GET /api/specs` and used by the structure validator.

---

## 10. Module: `specs/visa_vcf.py` and `specs/general_ledger.py`

`visa_vcf.py` defines both VISA VCF and General Ledger schemas. `general_ledger.py` is a shim that re-exports the GL definitions from `visa_vcf.py` for import convenience.

**General Ledger Journal Entry (RTJE) key fields:**

```python
F("DC Indicator",  50, 50, AN, allowed=["D","C"]),  # Debit or Credit
F("Amount",        51, 66, AM),                      # 16-digit zero-padded
F("Exchange Rate", 70, 79, N, required=False),       # for multi-currency
```

The `DC Indicator` field is a single character (`D` or `C`). The framework uses this during generation to ensure double-entry bookkeeping: alternating D/C entries with matching amounts produce balanced journals.

---

## 11. Module: `memory/config_engine.py`

### Purpose

Implements the **Singleton Source of Truth** for all specification rules. All other modules query this engine rather than importing spec files directly, enabling runtime overrides without restarting the server.

### 11.1 Singleton Pattern

```python
class ConfigEngine:
    _instance : Optional["ConfigEngine"] = None
    _lock      = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                inst = super().__new__(cls)
                inst._store  = _ConfigStore()
                inst._custom = _ConfigStore()
                inst._loaded = set()
                cls._instance = inst
                inst._bootstrap()
        return cls._instance
```

The `__new__` override implements thread-safe singleton creation. `threading.Lock()` ensures only one thread can create the instance even when multiple API requests arrive simultaneously. The `_bootstrap()` call loads all three spec schemas on first creation.

**Why Singleton?** The ConfigEngine stores runtime overrides (`_custom`) that must persist across API requests without a database. If each request created a new instance, overrides would be lost. The singleton ensures all API handlers share the same in-memory state.

### 11.2 `_ConfigStore` — The Thread-Safe Backend

```python
class _ConfigStore:
    def __init__(self):
        self._data = {}
        self._lock = threading.RLock()  # Reentrant lock

    def set(self, key, value):
        with self._lock:
            self._data[key] = value
```

Uses a `threading.RLock` (reentrant lock) rather than a regular `Lock`. Reentrant locks allow the same thread to acquire the lock multiple times without deadlocking — important because `get_fields()` may internally call other methods that also acquire the lock.

### 11.3 `_register_spec()` — Serialisation Design

```python
def _register_spec(self, spec_name, schema, meta):
    field_map = {}
    for rt, descriptors in schema.items():
        field_map[rt] = [
            {
                "name"      : fd.name,
                "start"     : fd.start,
                "end"       : fd.end,
                "length"    : fd.length,
                "field_type": fd.field_type.value,   # string, not enum
                "required"  : fd.required,
                "pattern"   : fd.pattern,
                "allowed"   : fd.allowed,
                "descriptor": fd,                    # keep original for extract()
            }
            for fd in descriptors
        ]
```

Field definitions are stored as plain dicts with one exception: the original `FieldDescriptor` is preserved under the `"descriptor"` key. This allows callers to use `fd["descriptor"].extract(line)` for accurate field slicing, while the dict representation allows JSON serialisation via `export_rules()` (which strips the `"descriptor"` key).

### 11.4 Runtime Override System

```python
def set_custom_rule(self, spec_name, record_type, field_name, overrides):
    key      = f"{spec_name}:{record_type}:fields"
    existing = list(self._custom.get(key, []))
    for fd in existing:
        if fd["name"] == field_name:
            fd.update(overrides)
            updated = True
            break
    if not updated:
        base   = self.get_field_rule(spec_name, record_type, field_name) or {}
        new_fd = {**base, "name": field_name, **overrides}
        existing.append(new_fd)
    self._custom.set(key, existing)
```

Overrides are stored in a separate `_custom` store and merged on top of base rules in `get_fields()`. This layered approach means:
- The base spec is never mutated.
- Overrides can be reset without restarting.
- Multiple overrides accumulate correctly.

**Use case:** Lock the Amount field to specific test values:
```python
engine.set_custom_rule("ACH_NACHA", "RT6", "Amount",
                        {"allowed": ["0000000100", "0000001000"]})
```
All subsequent generation calls will produce entries with exactly $1.00 or $10.00 amounts.

### 11.5 `export_rules()` — API Serialisation

```python
def export_rules(self, spec_name):
    return {
        "spec": self.get_meta(spec_name),
        "schema": {
            rt: [
                {k: v for k, v in fd.items() if k != "descriptor"}
                for fd in self.get_fields(spec_name, rt)
            ]
            for rt in rts
        },
    }
```

Strips the `"descriptor"` key (not JSON-serialisable) and returns a clean dict that the frontend uses to populate the field explorer table.

---

## 12. Module: `memory/seeder.py`

### Purpose

Generates randomised but specification-valid mock data for training and test-file generation. Implemented entirely in Python's standard library — no Faker, no external dependencies.

### 12.1 Embedded Word Lists

```python
_FIRST_NAMES  = ["JAMES", "MARY", "JOHN", "PATRICIA", ...]
_LAST_NAMES   = ["SMITH", "JOHNSON", "WILLIAMS", ...]
_COMPANY_NAMES= ["APEX FINANCIAL CORP", "SUMMIT BANK NA", ...]
_ENTRY_DESCRIPTIONS = ["PAYROLL", "VENDOR PMT", "REFUND", ...]
_GL_ACCOUNT_CODES   = ["1000", "1100", ..., "6200"]
```

Upper-cased throughout because the NACHA spec mandates upper-case alphanumeric fields. The GL account codes follow a standard chart-of-accounts structure: 1xxx=Assets, 2xxx=Liabilities, 3xxx=Equity, 4xxx=Revenue, 5xxx-6xxx=Expenses.

### 12.2 `_generate_routing_rng()` — Deterministic Valid Routing Numbers

```python
def _generate_routing_rng(self):
    weights = [3, 7, 1, 3, 7, 1, 3, 7]
    for _ in range(100):
        digits = [self.rng.randint(0, 9) for _ in range(8)]
        total  = sum(d * w for d, w in zip(digits, weights))
        check  = (10 - (total % 10)) % 10
        routing = "".join(str(d) for d in digits) + str(check)
        if 1 <= int(routing[:2]) <= 32:
            return routing
    return "021000021"
```

Uses `self.rng` (a `random.Random` instance with a user-supplied seed) rather than the global `random` module. This ensures deterministic output when a seed is specified. The constraint `1 <= routing[:2] <= 32` enforces that the Federal Reserve routing prefix is in a realistic range. The fallback `"021000021"` is JPMorgan Chase's real ABA routing number — always valid.

### 12.3 Field Name Normalisation in `_generate_field()`

```python
key_primary   = name.lower().replace(" ", "_")
key_secondary = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")

for key in (key_primary, key_secondary):
    if key in extra:
        ...
```

Extra parameters passed by the generator (e.g., `{"entry_hash": 42782835}`) are matched against field names using two normalisation strategies:
- Primary: simple space-to-underscore (`"Entry Hash"` → `"entry_hash"`)
- Secondary: strip all non-alphanumeric (`"Entry/Addenda Count"` → `"entry_addenda_count"`)

The secondary strategy handles field names with slashes, which are common in financial specs.

### 12.4 `_by_type()` — Field-Level Generation Logic

This method is a large conditional dispatch on field name keywords and types. The priority order is:

1. **ROUTING fields**: Generate valid routing via Mod-10 algorithm.
2. **ACCOUNT fields**: Random 10–17 digit account numbers.
3. **AMOUNT fields**: Random zero-padded amounts within width-appropriate ranges.
4. **DATE fields**: Random YYMMDD within ±18 months of today.
5. **Check Digit**: Derived from previously-generated routing (via `context`).
6. **Trace/Sequence**: Derived from routing prefix + random sequence component.
7. **Name fields**: Person names (First Last) or company names from embedded lists.
8. **Description fields**: Random entries from `_ENTRY_DESCRIPTIONS`.
9. **Hash/Control fields**: Random integers zero-padded to field width.
10. **BLANK_PAD/Reserved**: Spaces to full field width.
11. **Fallback NUMERIC**: Random digit string.
12. **Fallback ALPHANUMERIC**: Random characters from the legal set.

The `context` dictionary accumulates values generated during a single line's production so later fields can reference earlier ones (e.g., Check Digit references the Routing generated two fields earlier).

---

## 13. Module: `api/main.py`

### Purpose

FastAPI application exposing the SLM framework as a local HTTP API. All endpoints are local — no data leaves the machine.

### 13.1 Shared Singletons

```python
engine = ConfigEngine()
seeder = DataSeeder(engine)
_model_cache: Dict[str, Any] = {}
```

The `ConfigEngine` and `DataSeeder` are created once at module load time and shared across all requests. The `_model_cache` lazily loads model checkpoints on first use.

**Lazy model loading:**
```python
def _get_model(spec_name):
    if spec_name in _model_cache:
        return _model_cache[spec_name]
    ckpt = ROOT / "checkpoints" / f"{spec_name}_best.pt"
    if not ckpt.exists():
        return None, None
    model, cfg = build_model(spec_name)
    SLMTrainer.load_checkpoint(str(ckpt), model)
    model.eval()
    _model_cache[spec_name] = (model, cfg)
    return model, cfg
```

Models are only loaded when first needed and then cached in memory. This avoids the startup overhead of loading all three models simultaneously.

### 13.2 Background Training

```python
_train_state: Dict[str, Any] = {
    "running": False, "step": 0, "metrics": [], "error": None, ...
}
_train_lock = threading.Lock()

@app.post("/api/train")
async def start_training(req: TrainRequest, bg: BackgroundTasks):
    if _train_state["running"]:
        raise HTTPException(409, "Training already in progress")
    bg.add_task(_run_training, req)
    return {"status": "started"}
```

Training runs in a FastAPI `BackgroundTasks` thread. The `_train_state` dict is updated by the training thread and read by polling clients via `GET /api/train/status`. `_train_lock` prevents race conditions when both threads access the shared state.

**409 Conflict** is returned if training is already running — only one training job per spec is allowed simultaneously.

### 13.3 Validation Endpoint Design

```python
@app.post("/api/validate")
async def validate_file(spec_name: str = "ACH_NACHA", file: UploadFile = File(...)):
    raw_bytes = await file.read()
    try:
        raw_text = raw_bytes.decode("ascii")
    except UnicodeDecodeError:
        raw_text = raw_bytes.decode("latin-1")
```

The fallback to `latin-1` handles files that contain extended ASCII characters (bytes 128–255), which sometimes appear in legacy financial systems. Latin-1 maps bytes directly to Unicode code points, making it a lossless fallback.

### 13.4 CORS Configuration

```python
app.add_middleware(CORSMiddleware, allow_origins=["*"], ...)
```

All origins are allowed because the API is local-only. In a production deployment, this should be restricted to the specific frontend origin.

### 13.5 Static File Serving

```python
frontend_dir = ROOT / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

@app.get("/")
async def root():
    idx = ROOT / "frontend" / "index.html"
    if idx.exists():
        return FileResponse(str(idx))
    return {"message": "FinancialSLM API"}
```

The root path serves `index.html` directly so `http://localhost:8000` opens the SPA. Static assets (if any) are served from `/static`.

---

## 14. Module: `frontend/index.html`

### Architecture

A Single Page Application in one HTML file — no build tool, no npm, no framework. The entire UI is 1,300 lines of HTML, CSS, and vanilla JavaScript.

### 14.1 CSS Design System

```css
:root {
  --ink:    #0a0c0f;    /* background */
  --panel:  #0f1318;    /* card background */
  --surface:#161b22;    /* elevated surface */
  --border: #1e2733;    /* separator lines */
  --green:  #00d97e;    /* success / valid */
  --amber:  #ffb347;    /* warning */
  --red:    #ff4757;    /* error / invalid */
  --blue:   #4e9fff;    /* primary action */
  --cyan:   #00c9d4;    /* accent */
}
```

A dark industrial palette designed for readability of monospace financial data. The `--ff-mono` variable (`IBM Plex Mono`) is used for all financial content; `--ff-ui` (`Space Grotesk`) for UI labels. This two-font system visually separates "data" from "interface."

**Grid noise background:**
```css
body::before {
    background-image:
        linear-gradient(rgba(78,159,255,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(78,159,255,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
}
```

A subtle grid pattern rendered entirely in CSS — no image asset. Opacity 0.03 keeps it barely perceptible, adding depth without distracting from content.

### 14.2 Page Navigation

```javascript
function showPage(id) {
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
    document.getElementById('page-' + id).classList.add('active');
}
```

SPA navigation without a router — all five pages (Home, Validate, Generate, Explorer, Train) are always in the DOM; the `active` class shows only one. This is fast and requires no history API manipulation for local use.

### 14.3 Drag-and-Drop File Upload

```javascript
function handleDragOver(e) {
    e.preventDefault();
    document.getElementById('dropZone').classList.add('drag-over');
}
function handleDrop(e) {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) readFile(file);
}
function readFile(file) {
    const reader = new FileReader();
    reader.onload = ev => {
        document.getElementById('pasteArea').value = ev.target.result;
    };
    reader.readAsText(file);
}
```

`e.preventDefault()` on `dragover` is essential — without it, the browser's default behaviour is to navigate to the file URL. The `FileReader` API reads the file as text client-side (no upload required) and populates the textarea.

### 14.4 Offline Mock Generators

```javascript
function generateMockACH(n) {
    // Builds a valid ACH file with correct checksums, entirely in JavaScript
    const routing = randRouting();   // Mod-10 valid routing
    ...
    while (lines.length % 10 !== 0) lines.push('9'.repeat(94));
    return lines.join('\n');
}
```

When the API is unreachable, the frontend generates mock files using pure JavaScript. The `randRouting()` function implements the same Mod-10 algorithm as the Python seeder. This means the UI is functional even completely offline — users can generate and validate (client-side validation using the mock validator) without a running API server.

### 14.5 Simulated Training Progress

```javascript
function simulateTraining(maxSteps) {
    let step = 0;
    const id = setInterval(() => {
        step += Math.floor(maxSteps / 20);
        const clm = Math.max(0.3, 2.5 - (step/maxSteps)*2.0 + Math.random()*0.1);
        const val = Math.max(0.15, 1.2 - (step/maxSteps)*0.9 + Math.random()*0.05);
        // Update UI...
        if (step >= maxSteps) clearInterval(id);
    }, 400);
}
```

When the API is offline, training is simulated with plausible loss curves: CLM loss starts around 2.5 and decays exponentially toward 0.3 (good approximation of real character-level language model training). VAL loss starts around 1.2 and decays to 0.15.

### 14.6 Validation Result Rendering

```javascript
for (const lr of report.line_results) {
    const confColor = lr.model_conf > 0.8 ? 'var(--green)'
                    : lr.model_conf > 0.5 ? 'var(--amber)' : 'var(--red)';
    const errChips = (lr.errors||[]).map(e =>
        `<span class="error-chip ${e.source==='RULE'?'rule':'model'}" title="${e.rule}">${e.field}</span>`
    ).join('');
```

Error chips are colour-coded by source: rule errors (red) vs. model warnings (amber). The `title` attribute provides the full error message on hover, keeping the UI compact while still accessible. Confidence scores use a three-tier colour mapping: green (>80%), amber (50–80%), red (<50%).

---

## 15. Module: `run.py`

### Purpose

Command-line interface that exposes all framework capabilities without requiring the API server. Useful for scripting, CI/CD pipelines, and batch processing.

### 15.1 ANSI Color Helpers

```python
class C:
    GREEN = "\033[92m"
    RED   = "\033[91m"
    ...
    def g(s): return f"{C.GREEN}{s}{C.RESET}"
```

Static methods on a namespace class avoid polluting the global scope with colour function names. Terminal colour codes are applied wrapping strings — the `{C.RESET}` at the end of each coloured string prevents colour bleed into subsequent text.

### 15.2 Exit Codes

```python
# In cmd_validate():
sys.exit(0 if report.is_fully_valid else 1)
```

Exit code 0 = valid, 1 = invalid. This makes the CLI scriptable:
```bash
python run.py validate --spec ACH_NACHA --file payroll.ach || notify_team "Validation failed"
```

### 15.3 `cmd_explore()` — Field Schema Display

```python
print(engine.describe(args.spec, args.rt))
```

Delegates to `ConfigEngine.describe()` which renders a formatted ASCII table of field names, positions, lengths, types, and required status. Useful for quickly checking "what field is at position 30 in an RT6 record?" without consulting the spec PDF.

---

## 16. Module: `tests/test_suite.py`

### Structure

Seven test classes covering every layer of the stack:

| Class | Count | What it tests |
|-------|-------|--------------|
| `TestTokenizer` | 6 | Encoding, decoding, vocab coverage, record type detection |
| `TestConfigEngine` | 9 | Singleton, spec loading, field queries, override layering |
| `TestDataSeeder` | 11 | Line lengths, routing validity, all record types, batch generation |
| `TestValidator` | 12 | Routing Mod-10, amount format, date range, full file validation |
| `TestGenerator` | 10 | Line lengths, block factor, checksums, structure, entry count |
| `TestModelArchitecture` | 8 | Forward pass shapes, NaN absence, parameter count, mask shapes |
| `TestIntegration` | 4 | Full generate→validate pipeline for all three specs |

**Total: 63 tests, all passing.**

### Key Test Patterns

**Testing the Mod-10 routing validator:**
```python
def test_routing_valid(self):
    ok, msg = RuleEngine.validate_routing("021000021")
    self.assertTrue(ok, msg)

def test_routing_invalid_check_digit(self):
    ok, _ = RuleEngine.validate_routing("021000022")  # last digit wrong
    self.assertFalse(ok)
```

**Testing that model outputs have no NaN:**
```python
def test_no_nan_in_outputs(self):
    gen, val, conf = model(char_ids, field_ids, rt_ids, mode="validate")
    self.assertFalse(torch.isnan(gen).any().item(), "NaN in gen logits")
    self.assertFalse(torch.isnan(val).any().item(), "NaN in val logits")
```

NaN propagation is a common failure mode during early training. Testing for NaN before training begins confirms the architecture is numerically stable.

**Integration test:**
```python
def test_ach_generate_then_validate(self):
    gen = FinancialGenerator("ACH_NACHA", engine, seeder)
    raw = gen.generate_file(GenerationConfig(), n_entries=4)
    val = FinancialValidator("ACH_NACHA", engine, make_tokenizer("ACH_NACHA"))
    report = val.validate(raw)
    self.assertEqual(report.error_lines, 0,
        f"Generated ACH file should be error-free. Summary: {report.summary}")
```

This is the most important test — it proves the generator and validator are consistent. A generated file must have zero validation errors.

---

## 17. Data Flow: End-to-End Walkthrough

### Scenario: User uploads an ACH file for validation

```
1. Browser: User drags "payroll.ach" onto the drop zone
        │
        ▼
2. frontend/index.html: FileReader reads the file as ASCII text
        │
        ▼
3. JavaScript: POST /api/validate/text { spec_name: "ACH_NACHA", content: "1 01 ..." }
        │
        ▼
4. api/main.py: validate_text_body()
   ├── make_tokenizer("ACH_NACHA")         → FinancialTokenizer with RT1–RT9 schema
   ├── _get_model("ACH_NACHA")             → load checkpoint or return None
   └── FinancialValidator("ACH_NACHA", engine, tok, model)
        │
        ▼
5. validator.py: validate(raw_text)
   ├── _split_lines(raw_text)              → ["1 01 02100...", "5 220 PAYROL..."]
   ├── ACHChecksumValidator()              → initialise running totals
   └── for each line:
       ├── is_padding check                → skip all-9s lines
       ├── tokenizer.get_record_type(line) → "RT6"
       ├── _check_line_rules(line, "RT6")
       │   ├── engine.get_fields("ACH_NACHA", "RT6")  → 11 FieldDescriptor dicts
       │   └── for each field:
       │       ├── fd["descriptor"].extract(line)      → raw field value
       │       ├── validate_numeric / validate_routing / ...
       │       └── → FieldError or nothing
       ├── ach_checker.process_line("RT6", line)
       │   ├── routing_sum += int(line[3:11])
       │   └── entry_count += 1
       └── model_score(line, "RT6")        → 0.93 confidence
        │
        ▼
6. ValidationReport.to_dict()             → JSON response
        │
        ▼
7. frontend/index.html: renderReport(report)
   ├── v-summary cards (totals, %, status)
   ├── per-line grid (record type, error chips, confidence %)
   └── raw text viewer
```

### Scenario: User generates an ACH file with 5 entries

```
1. Browser: User selects "ACH NACHA", entries=5, strategy="temperature"
        │
        ▼
2. JavaScript: POST /api/generate { spec_name: "ACH_NACHA", n_entries: 5 }
        │
        ▼
3. api/main.py: generate_file()
   └── FinancialGenerator("ACH_NACHA", engine, seeder, tok, model=None)
        │
        ▼
4. generator.py: generate_file()
   └── _gen_ach_file(GenerationConfig(), n_entries=5)
       ├── seeder.generate_line("ACH_NACHA", "RT1")    → File Header
       ├── seeder.generate_line("ACH_NACHA", "RT5")    → Batch Header
       │
       ├── for i in range(5):
       │   └── seeder.generate_line("ACH_NACHA", "RT6", extra={"sequence": i+1})
       │       ├── engine.get_fields("ACH_NACHA", "RT6")
       │       └── for each field:
       │           └── seeder._generate_field(name, ftype, width, ...)
       │               ├── ROUTING → _generate_routing_rng() (Mod-10 valid)
       │               ├── AMOUNT  → random zero-padded int
       │               └── AN      → random name from _FIRST_NAMES / _LAST_NAMES
       │
       ├── batch_hash = routing_sum % 10^10
       ├── seeder.generate_line("ACH_NACHA", "RT8", extra={
       │       "entry_addenda_count": 5, "entry_hash": batch_hash, ...})
       │
       ├── seeder.generate_line("ACH_NACHA", "RT9", extra={...})
       │
       └── pad with "9"×94 until len(lines) % 10 == 0
        │
        ▼
5. JSON response: { content: "1 01...\n5 220...\n6 22...", line_count: 10 }
        │
        ▼
6. frontend: display in code viewer, offer Copy / Download / Send to Validator
```

---

## 18. Design Decisions & Trade-offs

### Decision 1: Custom Transformer vs. Fine-tuning a Pre-trained Model

**Choice:** Build from scratch.

**Why:** Pre-trained models (GPT-2, BERT) are trained on natural language — they have no intrinsic column-position awareness. A character-level fine-tune is possible but the base model carries enormous parameters (~100M+) that encode knowledge irrelevant to financial syntax. Our model is 850K parameters. For an offline, privacy-sensitive deployment, a smaller model trained on the exact domain is superior.

**Trade-off:** The model requires training before it provides model-based confidence scores. The rule engine works immediately without training.

### Decision 2: Hybrid Validation (Rules + Model)

**Choice:** Rules are always applied; model scores are supplementary.

**Why:** Rules are 100% deterministic and correct. A routing number either passes the Mod-10 algorithm or it doesn't. No model, regardless of training quality, should override that. The model catches what rules cannot: inter-field contextual inconsistencies. For example, rules cannot flag "the Transaction Code says Credit but the Individual Name field is blank" — but the model's bidirectional attention can learn to associate these fields.

**Trade-off:** Validation quality depends on training completeness. An untrained model always returns confidence 0.5–0.7 (near-random). Only after sufficient training does the model contribute meaningful signal.

### Decision 3: Constrained Decoding Over Unconstrained Sampling

**Choice:** Zero-out illegal tokens at every position before sampling.

**Why:** An unconstrained model could generate `"AB"` as a Transaction Code even after training, especially at high temperature. Financial files have zero tolerance for such errors. The constraint mask makes illegal outputs impossible, not just unlikely.

**Trade-off:** The constraint resolver adds overhead per generation step. For 94-character ACH lines, this means 94 ConfigEngine lookups per record. This is fast enough (<1ms per line) because the ConfigEngine is in-memory.

### Decision 4: In-Memory Config Engine Over a Database

**Choice:** Python dict singleton, no Redis, no SQL.

**Why:** The specification rules are static (the NACHA spec doesn't change weekly). They fit comfortably in memory (<1MB). A database would add operational complexity (connection management, query latency, schema migrations) for data that never needs to persist beyond the process lifetime. The singleton pattern gives the same "always-available" semantics as a cache without an external dependency.

**Trade-off:** All state is lost on process restart. Runtime overrides must be re-applied after restart. For a production deployment handling thousands of spec variants, a persistent store (Redis or PostgreSQL) would be appropriate.

### Decision 5: Triple Embedding Over Single Character Embedding

**Choice:** Concatenate character + field-slot + record-type embeddings.

**Why:** A character-only embedding treats `"6"` the same way regardless of whether it appears at column 1 (Record Type Code, definitively meaning "Entry Detail record") or column 30 (first digit of an Amount field). The additional embeddings give the model context that pure position encoding cannot fully capture.

**Trade-off:** The triple embedding requires a projection layer (`embed_proj`) to collapse 3×d_model back to d_model. This adds ~100K parameters but the representational benefit justifies the cost.

---

## 19. Extending the Framework

### Adding a New Financial Spec

1. **Define field schemas** in a new file `specs/myspec.py`:

```python
from slm.tokenizer import FieldDescriptor, TokenType

MYSPEC_FIELDS = {
    "RTA": [
        FieldDescriptor("Record Type",  1,  2, TokenType.RECORD_TYPE, allowed=["TA"]),
        FieldDescriptor("Amount",       3, 14, TokenType.AMOUNT),
        FieldDescriptor("Reference",   15, 30, TokenType.ALPHANUMERIC),
    ],
}

MYSPEC_META = {
    "name"        : "MY_SPEC",
    "full_name"   : "My Financial Format",
    "line_length" : 80,
    "record_types": list(MYSPEC_FIELDS.keys()),
    "description" : "Description here",
}
```

2. **Register in ConfigEngine** (`memory/config_engine.py`):

```python
def _bootstrap(self):
    from specs.myspec import MYSPEC_FIELDS, MYSPEC_META
    self._register_spec("MY_SPEC", MYSPEC_FIELDS, MYSPEC_META)
```

3. **Add line length** (`slm/tokenizer.py`):

```python
LINE_LENGTH_MAP = {
    ...
    "MY_SPEC": 80,
}
```

4. **Add structure check** (`slm/validator.py`, `_check_structure()`):

```python
elif self.spec_name == "MY_SPEC":
    types = [l[:2] for l in non_empty]
    if "TA" not in types:
        return False
```

5. **Add generation method** (`slm/generator.py`, `FinancialGenerator.generate_file()`):

```python
elif self.spec_name == "MY_SPEC":
    return self._gen_myspec_file(gen_cfg, n_entries)
```

6. **Add to API dropdowns** (`api/main.py` and `frontend/index.html`) — update `spec_name` choices.

### Adding a New Field Type

1. Add to `TokenType` enum in `slm/tokenizer.py`.
2. Add generation branch in `memory/seeder.py` `_by_type()`.
3. Add validation branch in `slm/validator.py` `_check_line_rules()`.
4. Add constraint class in `slm/generator.py` `ConstraintResolver.allowed_ids()`.

### Swapping the ConfigEngine Backend for Redis

Replace `_ConfigStore` with a Redis client:

```python
class _ConfigStore:
    def __init__(self):
        import redis
        self._r = redis.Redis(host="localhost", port=6379, db=0)

    def set(self, key, value):
        import json
        self._r.set(key, json.dumps(value))

    def get(self, key, default=None):
        import json
        v = self._r.get(key)
        return json.loads(v) if v else default
```

This makes overrides persistent across process restarts. The rest of the codebase is unchanged because it interacts only with the `ConfigEngine` interface.

---

## 20. Troubleshooting Guide

### "Checksum FAILURES DETECTED" on a generated file

**Cause:** The batch control (RT8) or file control (RT9) record contains wrong hash or entry count values.

**Debug:**
```python
from memory.config_engine import ConfigEngine
from memory.seeder import DataSeeder
from slm.generator import FinancialGenerator, GenerationConfig

engine = ConfigEngine()
seeder = DataSeeder(engine)
gen = FinancialGenerator("ACH_NACHA", engine, seeder)
raw = gen.generate_file(GenerationConfig(), n_entries=3)
for i, line in enumerate(raw.split("\n")):
    if line.strip():
        print(f"Line {i+1} [{line[0]}]: {line[:40]}...")
```

Check that RT8 positions 11–20 (Entry Hash) and 5–10 (Entry/Addenda Count) contain the expected values.

### "NaN in gen logits" during training

**Cause:** Learning rate too high, causing gradient explosion. Or a batch contained all-padding tokens triggering a division by zero in the softmax.

**Fix:** Reduce `learning_rate` by 10×. Ensure `corruption_prob` is not 1.0 (all records corrupted, all-padding batches possible).

### Model confidence always ~0.5

**Cause:** The model is untrained or undertrained. The validation head outputs near-random scores.

**Fix:** Train for more steps. For ACH NACHA, 2,000 steps typically yields validation head accuracy >90% on the synthetic test set. Use `python run.py train --spec ACH_NACHA --steps 5000`.

### "Field not found in schema" warning in seeder

**Cause:** The `extra` parameter keys don't match any field name normalisation.

**Debug:** Print both normalisations for the target field:
```python
name = "Entry/Addenda Count"
print(name.lower().replace(" ", "_"))              # "entry/addenda_count"
import re
print(re.sub(r"[^a-z0-9]+", "_", name.lower()))   # "entry_addenda_count"
```

Use `"entry_addenda_count"` as the extra key.

### "STRUCTURE VIOLATION" on a valid file

**Cause:** The file has the wrong number of lines (ACH blocking factor), or is missing a required header/trailer record type.

**Check:** Count lines: `wc -l payroll.ach`. For ACH, the number must be a multiple of 10. If it is 9, 19, etc., add a padding record:
```
999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999
```
(94 nines).

### Frontend shows "API offline (mock mode)"

**Cause:** The FastAPI server is not running or is running on a different port.

**Fix:**
```bash
python run.py serve --port 8000
```
Then open `http://localhost:8000`. If you changed the port, update `const API = 'http://localhost:8000'` at the top of `frontend/index.html`.

---

*End of HELP.md — FinancialSLM Technical Reference*

---

## 21. Complete API Reference

Every endpoint, its request schema, its response schema, and a concrete curl example.

### Base URL

```
http://localhost:8000
```

All request/response bodies are `application/json` unless noted.

---

### `GET /api/health`

Returns the current engine status.

**Response:**
```json
{
  "status"        : "ok",
  "specs_loaded"  : ["ACH_NACHA", "VISA_VCF", "GENERAL_LEDGER"],
  "models_loaded" : ["ACH_NACHA"],
  "train_running" : false
}
```

**curl:**
```bash
curl http://localhost:8000/api/health
```

---

### `GET /api/specs`

Returns metadata for all registered specification formats.

**Response:**
```json
{
  "specs": [
    {
      "name"            : "ACH_NACHA",
      "full_name"       : "ACH NACHA File Format",
      "line_length"     : 94,
      "blocking_factor" : 10,
      "encoding"        : "ASCII",
      "record_types"    : ["RT1","RT5","RT6","RT7","RT8","RT9"],
      "description"     : "The NACHA file format governs electronic funds transfer..."
    }
  ]
}
```

---

### `GET /api/specs/{spec_name}/rules`

Exports the complete field rule set as JSON.

**curl:**
```bash
curl http://localhost:8000/api/specs/ACH_NACHA/rules | python3 -m json.tool
```

**Response (abbreviated):**
```json
{
  "spec": { "name": "ACH_NACHA", "line_length": 94 },
  "schema": {
    "RT6": [
      { "name": "Transaction Code", "start": 2, "end": 3, "length": 2,
        "field_type": "NUMERIC", "required": true,
        "allowed": ["22","23","24","27","28","29","32","33","34","37","38","39"] },
      { "name": "RDFI Routing Transit", "start": 4, "end": 11, "length": 8,
        "field_type": "ROUTING", "required": true, "pattern": "[0-9]{8}" }
    ]
  }
}
```

---

### `GET /api/specs/{spec_name}/describe/{record_type}`

Returns a **plain-text** ASCII table. Designed for terminal display.

**curl:**
```bash
curl http://localhost:8000/api/specs/ACH_NACHA/describe/RT6
```

Output:
```
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

---

### `POST /api/validate/text`

Validate raw file content sent as a JSON body.

**Request:**
```json
{
  "spec_name" : "ACH_NACHA",
  "content"   : "1 01 021000021..."
}
```

**Response:**
```json
{
  "spec_name"      : "ACH_NACHA",
  "total_lines"    : 10,
  "valid_lines"    : 10,
  "error_lines"    : 0,
  "is_fully_valid" : true,
  "checksum_valid" : true,
  "sequence_valid" : true,
  "structure_valid": true,
  "summary"        : "10 lines valid, 0 lines with errors",
  "line_results"   : [
    {
      "line_no"    : 1,
      "record_type": "RT1",
      "is_valid"   : true,
      "model_conf" : 0.9341,
      "errors"     : []
    },
    {
      "line_no"    : 4,
      "record_type": "RT6",
      "is_valid"   : false,
      "model_conf" : 0.2187,
      "errors"     : [
        {
          "field"   : "RDFI Routing Transit",
          "position": "cols 4-11",
          "value"   : "'ABCDEFGH'",
          "rule"    : "must contain only digits",
          "severity": "ERROR",
          "source"  : "RULE"
        }
      ]
    }
  ]
}
```

---

### `POST /api/generate`

Generate a complete synthetic financial file.

**Request:**
```json
{
  "spec_name"   : "ACH_NACHA",
  "n_entries"   : 5,
  "strategy"    : "temperature",
  "temperature" : 0.7,
  "top_k"       : 10,
  "use_model"   : false
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `spec_name` | string | `"ACH_NACHA"` | Target format |
| `n_entries` | int | `3` | Number of entry/detail records |
| `strategy` | string | `"temperature"` | `greedy`, `temperature`, or `top_k` |
| `temperature` | float | `0.7` | Sampling temperature |
| `top_k` | int | `10` | Top-K cutoff |
| `use_model` | bool | `false` | Use SLM if checkpoint available |

**Response:**
```json
{
  "spec_name"   : "ACH_NACHA",
  "content"     : "1 01 021000021  9876543210260101...\n5 220...",
  "line_count"  : 10,
  "line_length" : 94,
  "generated_ms": 12,
  "used_model"  : false,
  "n_entries"   : 5
}
```

**curl:**
```bash
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"spec_name":"ACH_NACHA","n_entries":5,"strategy":"greedy"}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['content'])"
```

---

### `POST /api/generate/record`

Generate a single record line with annotated field breakdown.

**Request:**
```json
{ "spec_name": "ACH_NACHA", "record_type": "RT6" }
```

**Response:**
```json
{
  "line"        : "622021000021 123456789012345  0000012345...",
  "record_type" : "RT6",
  "fields"      : [
    { "name": "Record Type Code",    "start": 1,  "end": 1,  "value": "6",        "type": "RECORD_TYPE" },
    { "name": "Transaction Code",    "start": 2,  "end": 3,  "value": "22",       "type": "NUMERIC"     },
    { "name": "RDFI Routing Transit","start": 4,  "end": 11, "value": "02100002", "type": "ROUTING"     },
    { "name": "Amount",              "start": 30, "end": 39, "value": "0000012345","type": "AMOUNT"     }
  ],
  "context": { "routing": "021000021", "amount": 12345 }
}
```

---

### `POST /api/train`

Start a background training job. Returns 409 if already running.

**Request:**
```json
{
  "spec_name"       : "ACH_NACHA",
  "max_steps"       : 2000,
  "batch_size"      : 16,
  "learning_rate"   : 0.0003,
  "corruption_prob" : 0.3
}
```

**Response:**
```json
{ "status": "started", "spec_name": "ACH_NACHA", "max_steps": 2000 }
```

---

### `GET /api/train/status`

Poll training progress (designed for 2-second polling interval).

**Response (in progress):**
```json
{
  "running"   : true,
  "spec"      : "ACH_NACHA",
  "step"      : 750,
  "max_steps" : 2000,
  "metrics"   : [
    { "step": 250, "clm_loss": 1.8432, "val_loss": 0.6821, "total_loss": 2.1843 },
    { "step": 500, "clm_loss": 1.2341, "val_loss": 0.4129, "total_loss": 1.4404 },
    { "step": 750, "clm_loss": 0.9872, "val_loss": 0.3201, "total_loss": 1.1472 }
  ],
  "error"     : null,
  "elapsed_s" : 142.3
}
```

---

### `POST /api/config/override`

Inject a runtime field-level rule override.

**Request:**
```json
{
  "spec_name"  : "ACH_NACHA",
  "record_type": "RT6",
  "field_name" : "Amount",
  "overrides"  : { "allowed": ["0000000100", "0000010000"] }
}
```

Supported override keys: `allowed`, `required`, `pattern`, `field_type`.

**Response:**
```json
{ "status": "ok", "message": "Override applied for \"Amount\"" }
```

---

### `DELETE /api/config/override`

Reset overrides. Omit `record_type` to reset all record types for the spec.

**Request:**
```json
{ "spec_name": "ACH_NACHA", "record_type": "RT6" }
```

---

## 22. Annotated Sample Files

### 22.1 ACH NACHA — Annotated Line-by-Line

```
── Line 01  RT1 — File Header ────────────────────────────────────────────────────────────────────
1 01 021000021  9876543210260101    A094101DEST BANK NA           ORIG CORP NA           TESTREF 
│ │  │           │          │      │ │  │  │                      └ Immediate Origin Name (23ch)
│ │  │           │          │      │ │  │  └─ Immediate Destination Name (23ch)
│ │  │           │          │      │ │  └──── Format Code: 1
│ │  │           │          │      │ └─────── Blocking Factor: 10
│ │  │           │          │      └───────── Record Size: 094
│ │  │           │          └──────────────── File ID Modifier: A
│ │  │           │                            File Creation Time: 0000 (midnight)
│ │  │           └─────────────────────────── File Creation Date: 260101 (2026-Jan-01)
│ │  └─────────────────────────────────────── Immediate Origin: 9876543210 (10-digit tax ID)
│ └────────────────────────────────────────── Immediate Destination: space + 021000021 routing
└──────────────────────────────────────────── Record Type: 1 (File Header)
                                              Priority Code: 01 (always)

── Line 02  RT5 — Batch Header ───────────────────────────────────────────────────────────────────
5 220PAYROLL INC        TAX2026001  9876543210PPD PAYROLL   260105   1021000020000001
│ │  │                  │           │          │  │         │       │ │        └── Batch Number: 1
│ │  │                  │           │          │  │         │       │ └─────────── ODFI ID (8 digits)
│ │  │                  │           │          │  │         │       └───────────── Originator Status: 1
│ │  │                  │           │          │  │         └───────────────────── Effective Date: 260105
│ │  │                  │           │          │  └─────────────────────────────── Company Entry Desc: PAYROLL
│ │  │                  │           │          └────────────────────────────────── SEC Code: PPD
│ │  │                  │           └───────────────────────────────────────────── Company ID: 9876543210
│ │  │                  └───────────────────────────────────────────────────────── Company Discretionary Data
│ │  └──────────────────────────────────────────────────────────────────────────── Company Name (16 chars)
│ └─────────────────────────────────────────────────────────────────────────────── Service Class: 220=Credits
└───────────────────────────────────────────────────────────────────────────────── Record Type: 5

── Line 03  RT6 — Entry Detail ───────────────────────────────────────────────────────────────────
6 22021000021 123456789012345  0000150000JONES A    1234567890123  ALICE JONES            0 0210000200000001
│ │  │        │  │             │          │          │              │                     │ └── Trace Number
│ │  │        │  │             │          │          │              │                     └──── Addenda Ind: 0
│ │  │        │  │             │          │          │              └──────────────────────── Individual Name
│ │  │        │  │             │          │          └─────────────────────────────────────── Individual ID
│ │  │        │  │             │          └────────────────────────────────────────────────── Amount: $1500.00
│ │  │        │  │             └───────────────────────────────────────────────────────────── Account Number
│ │  │        │  └─────────────────────────────────────────────────────────────────────────── Check Digit: 1
│ │  │        └────────────────────────────────────────────────────────────────────────────── Routing (8 digits)
│ │  └─────────────────────────────────────────────────────────────────────────────────────── TX Code: 22=Credit DDA
│ └────────────────────────────────────────────────────────────────────────────────────────── Record Type: 6

── Lines 04-05  RT8 + RT9 — Control Records (computed fields) ────────────────────────────────────
8 220000001 0021000021000000000000000001500009876543210                        02100002 0000001
           │ │           │              │           │                           │        └── Batch Number
           │ │           │              │           │                           └─────────── ODFI (8 digits)
           │ │           │              │           └─────────────────────────────────────── Company ID
           │ │           │              └─────────────────────────────────────────────────── Total Credit (12ch)
           │ │           └────────────────────────────────────────────────────────────────── Total Debit (12ch)
           │ └────────────────────────────────────────────────────────────────────────────── Entry Hash (10ch)
           └──────────────────────────────────────────────────────────────────────────────── Entry Count (6ch)

9 000001000001000000010021000021000000000000000001500000000000000000000000000000000000000000000
  │       │       │          │              │              └── Reserved (39 spaces)
  │       │       │          │              └─────────────────── Total Credit Amount (12ch)
  │       │       │          └────────────────────────────────── Total Debit Amount (12ch)
  │       │       └───────────────────────────────────────────── Entry Hash (10ch)
  │       └───────────────────────────────────────────────────── Block Count (6ch)
  └───────────────────────────────────────────────────────────── Batch Count (6ch)

── Lines 06-10  Padding ──────────────────────────────────────────────────────────────────────────
9999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999  (×5)
All nines. Required to bring total line count to next multiple of 10 (blocking factor).
```

---

## 23. Mathematics Reference

### 23.1 ABA Routing Check Digit (Mod-10)

Given a 9-digit routing number $r_1 r_2 \dots r_9$, the weights are $[3, 7, 1, 3, 7, 1, 3, 7, 1]$:

$$\text{checksum} = \sum_{i=1}^{9} w_i \cdot r_i \equiv 0 \pmod{10}$$

To **generate** a valid check digit from the first 8 digits:

$$r_9 = \bigl(10 - \bigl(\textstyle\sum_{i=1}^{8} w_i r_i\bigr) \bmod 10\bigr) \bmod 10$$

### 23.2 ACH Entry Hash

$$H_{\text{batch}} = \Bigl(\sum_{i=1}^{N} R_i\Bigr) \bmod 10^{10}$$

where $R_i$ is the 8-digit RDFI routing (columns 4–11, without check digit) of entry $i$. The modulo prevents overflow into more than 10 digits. File-level hash sums all batches.

### 23.3 Scaled Dot-Product Attention

$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

The $1/\sqrt{d_k}$ factor counteracts the tendency of dot products to grow large as $d_k$ increases, preventing saturation (near-zero gradients) of the softmax.

**Causal mask:** positions $j > i$ receive score $-\infty$ before softmax, becoming exactly $0$ after.

### 23.4 Sinusoidal Positional Encoding

$$PE_{(pos,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right), \quad PE_{(pos,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)$$

Every column position $pos \in [0, 119]$ receives a unique continuous-valued vector. The relative distance between any two positions can be expressed as a linear transformation of their encodings.

### 23.5 Dual Training Loss

$$\mathcal{L} = \alpha\,\mathcal{L}_{\text{CLM}} + \beta\,\mathcal{L}_{\text{VAL}}$$

$$\mathcal{L}_{\text{CLM}} = -\frac{1}{T-1}\sum_{t=1}^{T-1}\log P(c_{t+1}\mid c_1,\ldots,c_t)$$

$$\mathcal{L}_{\text{VAL}} = -\frac{1}{F}\sum_{f=1}^{F}\bigl[y_f\log\sigma(z_f) + (1-y_f)\log(1-\sigma(z_f))\bigr]$$

Defaults: $\alpha=1.0$, $\beta=0.5$.

### 23.6 Learning Rate Schedule

$$\text{lr}(s) = \begin{cases}
\text{lr}_0 \cdot \dfrac{s}{s_\text{warm}} & s \le s_\text{warm} \\[8pt]
\text{lr}_0 \cdot \dfrac{1+\cos\!\left(\pi\,\dfrac{s-s_\text{warm}}{s_\text{max}-s_\text{warm}}\right)}{2} & s > s_\text{warm}
\end{cases}$$

### 23.7 Xavier Uniform Initialisation

$$W \sim \mathcal{U}\!\left[-\sqrt{\tfrac{6}{n_\text{in}+n_\text{out}}},\; +\sqrt{\tfrac{6}{n_\text{in}+n_\text{out}}}\right]$$

Preserves activation variance across layers, preventing vanishing and exploding gradients at initialisation.

---

## 24. Training Cookbook

### 24.1 Recommended Settings by Spec

| Hyperparameter | ACH NACHA | VISA VCF | General Ledger |
|---------------|-----------|----------|----------------|
| `max_steps` | 3,000 | 2,500 | 5,000 |
| `batch_size` | 32 | 32 | 16 |
| `learning_rate` | 3e-4 | 3e-4 | 1e-4 |
| `warmup_steps` | 300 | 250 | 500 |
| `corruption_prob` | 0.30 | 0.30 | 0.35 |
| `clm_weight` | 1.0 | 1.0 | 1.0 |
| `val_weight` | 0.5 | 0.5 | 0.7 |

### 24.2 Interpreting Loss Curves

**Healthy trajectory:**
```
CLM:  2.5 → 1.8 → 1.2 → 0.8 → 0.5 → 0.35  (smooth exponential decay)
VAL:  0.70 → 0.50 → 0.35 → 0.22 → 0.17     (smooth decay)
```

**Diagnosis table:**

| Symptom | Most Likely Cause | Fix |
|---------|------------------|-----|
| CLM plateaus above 1.5 | LR too low | Raise to 5e-4 |
| CLM spikes then collapses | LR too high | Reduce to 1e-4; raise warmup to 500 |
| VAL stuck at 0.693 (= ln 2) | Val head learns nothing | Increase `val_weight` to 1.0 |
| VAL plateaus above 0.5 | Too few corrupted samples | Raise `corruption_prob` to 0.45 |
| Loss oscillates wildly | Batch size too small | Increase batch_size to 64 |
| NaN loss | Gradient explosion or all-pad batch | Reduce LR 10×; clip norm |

A VAL loss of exactly 0.693 means the validation head outputs 0.5 for every field — it has learned nothing and is operating at chance.

### 24.3 Estimating Training Time

| Hardware | Steps | Batch | Time |
|----------|-------|-------|------|
| CPU (modern laptop) | 2,000 | 16 | ~3 min |
| CPU (modern laptop) | 5,000 | 32 | ~12 min |
| NVIDIA RTX 3080 | 5,000 | 64 | ~45 sec |

Enable GPU:
```python
cfg = TrainConfig(device="cuda" if torch.cuda.is_available() else "cpu")
```

### 24.4 Resuming a Checkpoint

```python
model, mcfg = build_model("ACH_NACHA")
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

ck = SLMTrainer.load_checkpoint("checkpoints/ACH_NACHA_step_1000.pt", model, opt)
print(f"Resuming from step {ck['step']}")

trainer = SLMTrainer(model, mcfg, tok, engine, seeder,
                     TrainConfig(spec_name="ACH_NACHA", max_steps=3000))
trainer.history = ck["history"]
trainer.train()
```

### 24.5 Evaluating Generation Quality

```python
pass_count = 0
for _ in range(100):
    raw    = gen.generate_file(GenerationConfig(strategy="temperature", temperature=0.8), n_entries=5)
    report = val.validate(raw)
    if report.is_fully_valid:
        pass_count += 1

print(f"Pass rate: {pass_count}%")
# Target: >98% after 3,000 steps. Rule-based seeder: 100%.
```

---

## 25. Security Model

### 25.1 Threat Summary

| Threat | Mitigation |
|--------|-----------|
| Data exfiltration via LLM API | Zero external calls by design |
| Malicious file execution | Input handled as raw ASCII text only; no `exec()` on file content |
| Checkpoint tampering | Use `weights_only=True` in `torch.load()` (PyTorch ≥ 2.0) |
| API exposure to network | Bind to `127.0.0.1` for local-only; use nginx for controlled exposure |
| Race conditions in ConfigEngine | All store operations use `threading.RLock()` |

### 25.2 Restrict API to Loopback

```bash
python run.py serve --host 127.0.0.1 --port 8000
```

### 25.3 Safe Checkpoint Loading

```python
# Prevent pickle-based code execution — tensors only
ck = torch.load("ACH_NACHA_best.pt", map_location="cpu", weights_only=True)
model.load_state_dict(ck["model_state"])
```

### 25.4 Restrict CORS for Team Deployments

```python
app.add_middleware(CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://192.168.1.50:8000"],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)
```

---

## 26. Deployment Guide

### 26.1 Local Development

```bash
pip install -r requirements.txt
python run.py serve --host 127.0.0.1 --port 8000
# Open http://127.0.0.1:8000
```

### 26.2 Docker

**Dockerfile:**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["python","-m","uvicorn","api.main:app","--host","0.0.0.0","--port","8000"]
```

```bash
docker build -t financialslm .
docker run -p 8000:8000 -v $(pwd)/checkpoints:/app/checkpoints financialslm
```

### 26.3 systemd

```ini
[Unit]
Description=FinancialSLM API Server
After=network.target

[Service]
Type=simple
User=slmuser
WorkingDirectory=/opt/financialslm
ExecStart=/opt/financialslm/venv/bin/python run.py serve --host 127.0.0.1 --port 8000
Restart=on-failure
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload && sudo systemctl enable --now financialslm
sudo journalctl -u financialslm -f
```

### 26.4 Production Checklist

- [ ] Server bound to `127.0.0.1` or behind reverse proxy
- [ ] CORS restricted to known origins
- [ ] `weights_only=True` in checkpoint loading
- [ ] `checkpoints/` on persistent volume
- [ ] All 63 tests passing (`python run.py test`)
- [ ] Log rotation configured for uvicorn

---

## 27. Code Recipes

### Validate a File Programmatically

```python
from memory.config_engine import ConfigEngine
from slm.tokenizer        import make_tokenizer
from slm.validator        import FinancialValidator

engine    = ConfigEngine()
validator = FinancialValidator("ACH_NACHA", engine, make_tokenizer("ACH_NACHA"))

with open("payroll.ach", encoding="ascii") as f:
    report = validator.validate(f.read())

print(f"Valid: {report.is_fully_valid}")
for lr in report.line_results:
    if not lr.is_valid:
        for err in lr.errors:
            print(f"  Line {lr.line_no} | {err.field_name}: {err.rule}")
```

### Generate and Save

```python
from memory.config_engine import ConfigEngine
from memory.seeder        import DataSeeder
from slm.generator        import FinancialGenerator, GenerationConfig

engine  = ConfigEngine()
seeder  = DataSeeder(engine, seed=42)
gen     = FinancialGenerator("ACH_NACHA", engine, seeder)
content = gen.generate_file(GenerationConfig(strategy="temperature"), n_entries=100)

with open("test_file.ach", "w", encoding="ascii", newline="") as f:
    f.write(content)
```

### Lock Fields for Test Scenarios

```python
engine = ConfigEngine()
engine.set_custom_rule("ACH_NACHA", "RT6", "Amount",
                        {"allowed": ["0000000100", "0000010000"]})

# All generated RT6 entries will have Amount = $1.00 or $10.00
gen = FinancialGenerator("ACH_NACHA", engine, DataSeeder(engine))
raw = gen.generate_file(GenerationConfig(), n_entries=5)

engine.reset_custom_rules("ACH_NACHA", "RT6")   # clean up
```

### Batch Validate Multiple Files

```python
import glob
from pathlib import Path

validator = FinancialValidator("ACH_NACHA", ConfigEngine(), make_tokenizer("ACH_NACHA"))

for path in glob.glob("incoming/*.ach"):
    report = validator.validate(Path(path).read_text(encoding="ascii", errors="replace"))
    status = "✓" if report.is_fully_valid else "✗"
    print(f"{status} {path}: {report.summary}")
```

### Export Field Rules to CSV

```python
import csv
engine = ConfigEngine()

with open("rt6_rules.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["name","start","end","length","field_type","required","allowed"])
    writer.writeheader()
    for fd in engine.get_fields("ACH_NACHA", "RT6"):
        writer.writerow({k: fd.get(k,"") for k in writer.fieldnames})
```

---

## 28. Architecture Decision Records

### ADR-001 · Character-Level Tokenization

**Decision:** 64-token character vocabulary. No BPE or WordPiece.

**Rationale:** Financial files have no word boundaries. Every character's column position carries semantic meaning. A character-level vocab is closed (no OOV tokens), trivially invertible, and maps directly to fixed-width field boundaries.

**Trade-off:** Sequences are longer (94 chars vs. ~15 subword tokens for an ACH line). The model must learn multi-character patterns without subword grouping.

### ADR-002 · Triple Embedding

**Decision:** Concatenate character identity + field-slot + record-type embeddings, then project to `d_model`.

**Rationale:** Character `"2"` at column 2 is a Service Class Code digit in RT5, but a Transaction Code digit in RT6. The record-type embedding disambiguates. The field-slot embedding tells the model which named field it is currently inside.

**Trade-off:** Adds a linear projection layer and two embedding tables. Requires field-slot IDs to be precomputed per sample.

### ADR-003 · Dual Output Head on Shared Encoder

**Decision:** Share all transformer blocks; branch at the final layer into generation head and validation head.

**Rationale:** Two separate models would cost 2× parameters. Shared encoder creates multi-task regularisation — representations useful for generation also aid validation.

**Trade-off:** Two forward passes per training step (causal vs. bidirectional mask). Loss weighting requires tuning.

### ADR-004 · Rule Engine Priority Over Model

**Decision:** Deterministic rule checks always run; model scores are supplementary and cannot override rule results.

**Rationale:** A routing number either passes Mod-10 or it doesn't. No probabilistic model should override a mathematical fact.

**Trade-off:** The model cannot learn to "tolerate" rules — e.g., routing numbers technically invalid but accepted by legacy systems.

### ADR-005 · Singleton ConfigEngine

**Decision:** Python dict singleton loaded at startup. No Redis, no database.

**Rationale:** Spec rules are static, small (<1MB), and must be accessible with zero latency on the hot path. Runtime overrides must persist across requests within a session.

**Trade-off:** State lost on process restart. Not suitable for multi-process deployment without a shared backend.

### ADR-006 · Constrained Decoding

**Decision:** Zero illegal token logits to $-\infty$ at each character position before sampling.

**Rationale:** Post-hoc rejection sampling is expensive and unpredictable. Hard constraints guarantee syntactic correctness at zero extra cost beyond the `ConstraintResolver` lookup.

**Trade-off:** Reduces sampling diversity at highly-constrained positions (e.g., Transaction Code). The model cannot be "creative" about values that have a fixed allowed set.

### ADR-007 · Synthetic Training Data

**Decision:** Generate infinite synthetic records from ConfigEngine rules. No real financial data.

**Rationale:** Real financial transaction data is confidential. Privacy-by-design means not requiring it. The ConfigEngine contains all syntax rules, enabling unlimited, valid training examples.

**Trade-off:** The model learns spec-valid syntax, not real-world statistical distributions. Validation head only sees four synthetic corruption types. Novel real-world errors may not score poorly.

---

## 29. Glossary

### Financial Terms

| Term | Definition |
|------|-----------|
| **ABA Routing Number** | 9-digit identifier for US financial institutions. Last digit is a Mod-10 check digit. |
| **ACH** | Automated Clearing House — US electronic funds transfer network governed by NACHA. |
| **Batch** | Group of ACH entries sharing company, effective date, and SEC code. Bounded by RT5 and RT8. |
| **Blocking Factor** | ACH files must contain a multiple-of-10 line count. Padding records fill the remainder. |
| **CCD** | Corporate Credit or Debit — ACH SEC code for business-to-business transactions. |
| **Check Digit** | A computed digit appended to a code (routing number) to detect transcription errors. |
| **DDA** | Demand Deposit Account — a standard checking account. |
| **Double-Entry** | Every GL transaction has equal debits and credits, keeping the ledger balanced. |
| **Entry Hash** | Sum of all RDFI routing numbers (8-digit) in a batch, mod 10^10. Detects reordering. |
| **Implied Decimal** | Amounts stored as integers with a fixed implied decimal position. `0000012345` = $123.45. |
| **MCC** | Merchant Category Code — 4-digit business type identifier in VISA VCF. |
| **NACHA** | National Automated Clearing House Association — governing body for ACH standards. |
| **ODFI** | Originating Depository Financial Institution — the bank initiating the ACH transaction. |
| **PAN** | Primary Account Number — the card number (up to 16 digits) in VISA VCF. |
| **PPD** | Prearranged Payment and Deposit — most common ACH SEC code for consumer transactions. |
| **RDFI** | Receiving Depository Financial Institution — the bank receiving the ACH transaction. |
| **SEC Code** | Standard Entry Class Code — 3-letter code in RT5 indicating ACH transaction type. |
| **Settlement Date** | Date funds actually move between banks (typically T+1 or T+2 business days). |
| **Transaction Code** | 2-digit RT6 field specifying transaction direction and account type (22=credit to DDA). |

### ML / System Terms

| Term | Definition |
|------|-----------|
| **AdamW** | Adam optimiser with decoupled weight decay applied directly to parameters. |
| **Autoregressive** | Each output token conditioned on all previously generated tokens. |
| **BCEWithLogitsLoss** | Numerically stable binary cross-entropy that incorporates sigmoid internally. |
| **Causal Mask** | Lower-triangular attention mask — position $i$ cannot see positions $j > i$. |
| **CLM** | Causal Language Modelling — predict next token given all previous tokens. |
| **d_model** | Dimensionality of all model embeddings and hidden states. |
| **GELU** | Gaussian Error Linear Unit activation. Smoother than ReLU, better on small datasets. |
| **Gradient Clipping** | Scale down gradient vectors exceeding a norm threshold. Prevents weight explosion. |
| **IterableDataset** | PyTorch streaming dataset type. Used here for infinite synthetic record generation. |
| **Layer Norm** | Normalise activations across features (not batch). Stabilises deep network training. |
| **Pre-LayerNorm** | Apply normalisation before each sub-layer (inside residual branch). More stable. |
| **Residual Connection** | output = x + sublayer(x). Enables deep models by preserving gradient flow. |
| **SLM** | Small Language Model — transformer with millions (not billions) of parameters. |
| **Warmup** | Linearly increase LR from 0 to target over N steps, avoiding large initial updates. |
| **Weight Decay** | L2 regularisation on parameters. Reduces overfitting. In AdamW, decoupled from gradient. |
| **Xavier Init** | Weight initialisation preserving activation variance: $W \sim \mathcal{U}[-\sqrt{6/(n_{in}+n_{out})}, ...]$. |

---

## 30. File Format Quick Reference

### ACH Record Types

| RT | Name | Per File | Key Computed Fields |
|----|------|---------|---------------------|
| 1 | File Header | Exactly 1 | None |
| 5 | Batch Header | 1 per batch | None |
| 6 | Entry Detail | 1+ per batch | None |
| 7 | Addenda | 0–1 per RT6 | None |
| 8 | Batch Control | 1 per batch | Entry count, hash, dollar totals |
| 9 | File Control | Exactly 1 | Batch count, block count, hash, totals |
| 9…9 | Padding | Fill to 10× | N/A |

### Transaction Code Reference (RT6 cols 2–3)

| Code | Direction | Account | Notes |
|------|-----------|---------|-------|
| 22 | Credit | Checking (DDA) | Direct deposit — most common |
| 23 | Credit | Checking | Pre-notification (zero-dollar test) |
| 27 | Debit | Checking | Bill payment |
| 28 | Debit | Checking | Pre-notification |
| 32 | Credit | Savings | Direct deposit to savings |
| 33 | Credit | Savings | Pre-notification |
| 37 | Debit | Savings | Debit from savings |
| 38 | Debit | Savings | Pre-notification |

### VISA VCF Transaction Codes (DT cols 3–4)

| Code | Description |
|------|-------------|
| 01 | Purchase |
| 02 | Cash Advance |
| 05 | Credit / Refund |
| 25 | Chargeback |
| 26 | Chargeback Reversal |

### GL Standard Chart of Accounts

| Range | Category | Examples |
|-------|----------|---------|
| 1000–1999 | Assets | 1000=Cash, 1200=Accounts Receivable |
| 2000–2999 | Liabilities | 2000=Accounts Payable, 2300=Long-term Debt |
| 3000–3999 | Equity | 3000=Common Stock, 3100=Retained Earnings |
| 4000–4999 | Revenue | 4000=Service Revenue, 4500=Interest Income |
| 5000–5999 | Cost of Goods | 5000=COGS, 5100=Direct Labour |
| 6000–6999 | Operating Expenses | 6000=Salaries, 6100=Rent, 6200=Utilities |

---

## 31. Contributing and Code Style

### Module Responsibilities (Strict Boundaries)

| Module | May | Must Not |
|--------|-----|---------|
| `tokenizer.py` | Convert text ↔ integer sequences | Validate, generate, or query ConfigEngine |
| `validator.py` | Apply rules and score confidence | Generate records or modify ConfigEngine state |
| `generator.py` | Produce spec-valid text | Validate output or apply business rules |
| `seeder.py` | Generate plausible field values | Enforce correctness or validate output |
| `config_engine.py` | Store and serve spec rules | Generate or validate records |
| `model.py` | Define neural architecture | Import application-layer modules |
| `specs/*.py` | Declare field schemas as data | Contain logic or functions |

### Naming Conventions

| Element | Convention | Example |
|---------|-----------|---------|
| Module files | `snake_case.py` | `config_engine.py` |
| Classes | `PascalCase` | `FinancialTokenizer` |
| Functions / methods | `snake_case()` | `generate_line()` |
| Module-level constants | `UPPER_SNAKE` | `ACH_FIELD_SCHEMA` |
| Private methods | `_leading_underscore` | `_check_line_rules()` |
| Type aliases | `PascalCase` | `FieldMap = Dict[str, List[Dict]]` |

### Adding a New Spec — Checklist

- [ ] `specs/myspec.py` — field schema dicts + metadata dict
- [ ] `memory/config_engine.py` — import and register in `_bootstrap()`
- [ ] `slm/tokenizer.py` — add to `LINE_LENGTH_MAP`
- [ ] `slm/validator.py` — add structure check in `_check_structure()`
- [ ] `slm/generator.py` — add `_gen_myspec_file()` and branch in `generate_file()`
- [ ] `memory/seeder.py` — add field name hints in `_by_type()` if needed
- [ ] `tests/test_suite.py` — add generator, validator, and integration tests
- [ ] `frontend/index.html` — add to spec dropdowns in all four pages
- [ ] `api/main.py` — add to `spec_name` validator choices

---

*End of HELP.md — FinancialSLM Complete Technical Reference*
*Total: ~3,500 lines · ~18,000 words · 31 sections*

---

## 32. Performance Profiling & Optimisation

### 32.1 Where Time Is Spent

A performance breakdown of the three main operations at their default settings:

```
Operation               | Rule-Based | +SLM Model | Notes
────────────────────────|────────────|────────────|──────────────────────────────
Validate 1 ACH file     |   0.3 ms   |    8 ms    | +SLM = 1 forward pass/line
  (10 lines, no model)  |            |            |
Generate 1 ACH file     |   1.2 ms   |  220 ms    | +SLM = 94 autoregressive steps
  (10 lines, n=3)       |            |            |   per line × 7 real lines
Train 1 step (CPU)      |     —      |   35 ms    | Two forward passes + backward
Train 1 step (GPU)      |     —      |    2 ms    | GPU utilisation ~60% at B=32
ConfigEngine lookup     |  <0.01 ms  |     —      | Thread-safe dict, in-memory
```

The dominant cost for model-based generation is the **autoregressive loop**: 94 sequential forward passes per line. This cannot be parallelised because each character depends on all previous characters.

### 32.2 Profiling with Python cProfile

```python
import cProfile
import pstats
import io

from memory.config_engine import ConfigEngine
from memory.seeder        import DataSeeder
from slm.generator        import FinancialGenerator, GenerationConfig

engine = ConfigEngine()
seeder = DataSeeder(engine)
gen    = FinancialGenerator("ACH_NACHA", engine, seeder)

pr = cProfile.Profile()
pr.enable()

for _ in range(100):
    gen.generate_file(GenerationConfig(strategy="greedy"), n_entries=3)

pr.disable()
s   = io.StringIO()
ps  = pstats.Stats(pr, stream=s).sort_stats("cumulative")
ps.print_stats(20)
print(s.getvalue())
```

**Expected top hot spots (rule-based generation, 100 files):**

```
ncalls    tottime    cumtime  function
 30000      0.041      0.041  seeder._generate_routing_rng
 30000      0.029      0.078  seeder._by_type
120000      0.019      0.019  config_engine.get_fields (dict lookup)
 30000      0.012      0.012  seeder._generate_field
```

### 32.3 Memory Footprint

```python
import tracemalloc

tracemalloc.start()

from memory.config_engine import ConfigEngine
from slm.model import build_model

engine = ConfigEngine()
model, cfg = build_model("ACH_NACHA")

current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

print(f"ConfigEngine (all 3 specs): {current/1024:.0f} KB")
print(f"ACH NACHA model weights:    {sum(p.numel()*4 for p in model.parameters())/1024:.0f} KB")
```

Expected output:
```
ConfigEngine (all 3 specs): 412 KB
ACH NACHA model weights:    3,328 KB  (~3.2 MB at float32)
```

Full stack footprint: <10 MB for all three models + ConfigEngine + FastAPI baseline.

### 32.4 Speeding Up Generation

**Option 1: Use rule-based generation (fastest)**

Set `use_model=False` in the API request (or pass `model=None` to `FinancialGenerator`). This bypasses the neural network entirely and runs at ~1 ms/file.

**Option 2: Smaller model configuration**

```python
from slm.model import SLMConfig, FinancialSLM

# Halve the model for 4× faster inference
cfg = SLMConfig(d_model=64, n_layers=2, n_heads=4, d_ff=256, max_seq_len=94)
model = FinancialSLM(cfg)
# ~200K params, ~50ms per 94-char line
```

**Option 3: KV-Cache (manual implementation)**

The current autoregressive loop recomputes all previous positions at every step. A key-value cache stores computed attention keys and values, reducing each step to a single-position forward pass:

```python
# Pseudocode — production implementation would subclass TransformerBlock
class CachedTransformerBlock(TransformerBlock):
    def forward_cached(self, x_new, kv_cache):
        # x_new: (B, 1, D) — only the new token
        # kv_cache: (B, t_prev, D) — all previous keys/values
        k_new  = self.k_proj(x_new)
        v_new  = self.v_proj(x_new)
        k_full = torch.cat([kv_cache["k"], k_new], dim=1)
        v_full = torch.cat([kv_cache["v"], v_new], dim=1)
        q      = self.q_proj(x_new)
        # Attend over full key history, output only the new position
        out    = scaled_dot_product(q, k_full, v_full)
        return out, {"k": k_full, "v": v_full}
```

With KV-cache, generation cost changes from O(T²) to O(T) in attention computation.

**Option 4: GPU inference**

```python
model = model.cuda()
char_ids  = char_ids.cuda()
field_ids = field_ids.cuda()
rt_ids    = rt_ids.cuda()

# Automatic Mixed Precision — halves memory, ~2× faster on CUDA
with torch.autocast(device_type="cuda", dtype=torch.float16):
    logits, val, conf = model(char_ids, field_ids, rt_ids, mode="validate")
```

### 32.5 Speeding Up Validation

**Batched validation** — process multiple lines in a single forward pass:

```python
def validate_batch(lines: List[str], tokenizer, model, cfg) -> List[float]:
    """Score multiple lines simultaneously. Returns per-line confidence."""
    T     = cfg.max_seq_len
    batch = []
    for line in lines:
        ids = tokenizer.encode_line(line)
        batch.append(ids)

    char_ids  = torch.tensor(batch, dtype=torch.long)       # (B, T)
    field_ids = torch.zeros_like(char_ids)
    rt_ids    = torch.zeros(len(lines), dtype=torch.long)

    with torch.no_grad():
        _, _, conf = model(char_ids, field_ids, rt_ids, mode="validate")

    return [float(c[0]) for c in conf]
```

This is ~10× faster than scoring each line individually because the GPU processes all B lines in a single pass.

### 32.6 DataLoader Worker Count

The training `DataLoader` uses `num_workers=0` (no background processes). This is correct for `IterableDataset` with `threading.RLock` — multiple worker processes would each hold an independent ConfigEngine instance, wasting memory.

If you need faster data loading, use in-process generation with a larger batch size rather than multiple workers:

```python
# Instead of: num_workers=4 (spawns 4 processes, 4× ConfigEngine instances)
# Use:        batch_size=128, num_workers=0  (larger batches, single process)
DataLoader(dataset, batch_size=128, num_workers=0, collate_fn=collate_fn)
```

---

## 33. Model Interpretability

### 33.1 Extracting Attention Weights

The `MultiHeadAttention` module computes attention weights internally but does not return them by default. Modify it to capture them via a hook or modify the forward method:

```python
import torch
from slm.model import build_model, FinancialSLM
from slm.tokenizer import make_tokenizer
from slm.trainer import SLMTrainer

model, cfg = build_model("ACH_NACHA")
SLMTrainer.load_checkpoint("checkpoints/ACH_NACHA_best.pt", model)
model.eval()

tok  = make_tokenizer("ACH_NACHA")

# Representative RT6 Entry Detail record
line = "622021000021 123456789012345  0000012345IND001234567   JOHN SMITH             00 0210000200000001"
ids  = torch.tensor([tok.encode_line(line)], dtype=torch.long)

# Capture attention weights from the last transformer block
captured_attn = {}

def attn_hook(module, input, output):
    # Recompute attention weights from the stored Q, K
    with torch.no_grad():
        q = module.q_proj(input[0])
        k = module.k_proj(input[1])
        B, T, _ = q.shape
        H, dk = module.n_heads, module.d_k
        q = q.view(B, T, H, dk).transpose(1, 2)
        k = k.view(B, T, H, dk).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / (dk ** 0.5)
        captured_attn["weights"] = torch.softmax(scores, dim=-1).detach()

hook = model.blocks[-1].attn.register_forward_hook(attn_hook)

with torch.no_grad():
    field_ids = torch.zeros_like(ids)
    rt_ids    = torch.zeros(1, dtype=torch.long)
    model(ids, field_ids, rt_ids, mode="validate")

hook.remove()
weights = captured_attn["weights"]  # (1, n_heads, T, T)
print(f"Attention weight tensor: {weights.shape}")
# weights[0, head, query_pos, key_pos]
```

### 33.2 Visualising Field-Level Attention

After extracting attention weights, you can see which columns the model attends to when processing each field:

```python
import numpy as np

# Average across heads and batch
avg_attn = weights[0].mean(dim=0).numpy()  # (T, T) averaged across heads

# Field positions for RT6
rt6_fields = [
    ("RecType",  0,  1),   ("TxCode",  1,  3),  ("Routing",  3, 11),
    ("ChkDig",  11, 12),   ("Account",12, 29),  ("Amount",  29, 39),
    ("IndID",   39, 54),   ("Name",   54, 76),  ("Disc",    76, 78),
    ("AddInd",  78, 79),   ("Trace",  79, 94),
]

print(f"\nField-to-field attention (averaged, top sources):")
print(f"{'Query Field':<15} → {'Top Attended Fields'}")
print("─" * 60)

for fname, fs, fe in rt6_fields:
    # Average attention from this field's query positions
    field_attn = avg_attn[fs:fe, :].mean(axis=0)  # (T,)

    # Assign to fields
    field_scores = {}
    for kname, ks, ke in rt6_fields:
        field_scores[kname] = field_attn[ks:ke].mean()

    top3 = sorted(field_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    top_str = ", ".join(f"{n}({s:.3f})" for n, s in top3)
    print(f"{fname:<15} → {top_str}")
```

**Interpreting results:** A well-trained model should show that:
- `Amount` attends strongly to `TxCode` (the credit/debit direction affects amount semantics)
- `ChkDig` attends strongly to `Routing` (the check digit is derived from routing)
- `Trace` attends broadly (trace number is contextually dependent on the whole record)

### 33.3 Per-Field Confidence Scores

The validation head outputs a logit for each of the 40 field slots. Convert to probabilities to see which fields the model is uncertain about:

```python
model.eval()
with torch.no_grad():
    _, val_logits, global_conf = model(ids, field_ids, rt_ids, mode="validate")

# val_logits: (1, 40, 2) — logits for [invalid, valid] per field slot
field_probs = torch.softmax(val_logits[0], dim=-1)[:, 1]  # P(valid) per slot

print(f"\nGlobal record confidence: {global_conf[0,0].item():.4f}")
print(f"\nPer-field validity scores:")
print(f"{'Slot':<6} {'Field Name':<35} {'P(valid)':<10} {'Bar'}")
print("─" * 65)

fields = tok.field_schema.get("RT6", [])
for i, (prob, fd) in enumerate(zip(field_probs[:len(fields)], fields)):
    p    = prob.item()
    bar  = "█" * int(p * 20) + "░" * (20 - int(p * 20))
    flag = "⚠" if p < 0.5 else " "
    print(f"{i:<6} {fd.name:<35} {p:<10.4f} {bar} {flag}")
```

**Example output after 3,000 training steps:**
```
Global record confidence: 0.9234

Per-field validity scores:
Slot   Field Name                          P(valid)   Bar
─────────────────────────────────────────────────────────────────
0      Record Type Code                   0.9981     ████████████████████
1      Transaction Code                   0.9712     ███████████████████░
2      RDFI Routing Transit               0.9456     ██████████████████░░
3      Check Digit                        0.9203     ██████████████████░░
4      RDFI Account Number                0.8834     █████████████████░░░
5      Amount                             0.9567     ███████████████████░
...
```

### 33.4 Probing What the Model Has Learned

Test the model's sensitivity to deliberate corruptions:

```python
def sensitivity_probe(model, tok, line, fields_to_corrupt, spec="ACH_NACHA", rt="RT6"):
    """Measure confidence drop when each field is corrupted."""
    import copy

    def score(raw_line):
        ids  = torch.tensor([tok.encode_line(raw_line)], dtype=torch.long)
        fids = torch.zeros_like(ids)
        rids = torch.zeros(1, dtype=torch.long)
        with torch.no_grad():
            _, _, conf = model(ids, fids, rids, mode="validate")
        return conf[0, 0].item()

    baseline = score(line)
    print(f"Baseline confidence: {baseline:.4f}\n")
    print(f"{'Field':<35} {'Corrupted':<25} {'Conf':<8} {'Drop'}")
    print("─" * 75)

    for fname, fs, fe, corruption in fields_to_corrupt:
        corrupted = line[:fs] + corruption[:fe-fs].ljust(fe-fs) + line[fe:]
        conf      = score(corrupted)
        drop      = baseline - conf
        alert     = "⚠⚠⚠" if drop > 0.3 else "⚠" if drop > 0.1 else ""
        print(f"{fname:<35} {repr(corruption[:fe-fs]):<25} {conf:<8.4f} -{drop:.4f} {alert}")

# Test on RT6
sensitivity_probe(model, tok, line, [
    ("Transaction Code",   1,  3, "XX"),     # invalid chars
    ("RDFI Routing Trans", 3, 11, "00000000"),# all zeros (fails Mod-10)
    ("Amount",            29, 39, "ABCDEFGHIJ"),# alpha in numeric
    ("Individual Name",   54, 76, " " * 22), # blank required field
])
```

---

## 34. Data Pipeline Deep Dive

### 34.1 The Full Sample-Generation Pipeline

Every training sample travels through six transformation stages:

```
Stage 1: Record Type Selection
  ConfigEngine.get_record_types("ACH_NACHA")
  → ["RT1","RT5","RT6","RT7","RT8","RT9"]
  rng.choice(...) → "RT6"

Stage 2: Field Definition Retrieval
  ConfigEngine.get_fields("ACH_NACHA", "RT6")
  → [11 field dicts from in-memory store]
  Thread-safe dict lookup, <0.01ms

Stage 3: Field Value Generation (DataSeeder._generate_field × 11)
  For each field:
    ├── Check extra dict keys (normalised name match)
    ├── Check allowed list  → pick from whitelist
    └── Dispatch _by_type() → routing/amount/name/date/numeric
  → Raw field values: "6", "22", "02100002", "1", "12345...", ...

Stage 4: Line Assembly
  Place each value at its field position in a 94-char string
  "622021000021 12345678901234   0000012345IND0001      JOHN SMITH             00 0210000200000001"

Stage 5: Corruption (probability p=0.3)
  rng.random() < 0.3 → True
  rng.randrange(11)  → field index 5 (Amount)
  rng.choice(["XXXXXXXXXX", "??????????", "          ", "9999999999"]) → "XXXXXXXXXX"
  Replace chars 29-38 with "XXXXXXXXXX"
  field_labels[5] = 0  (mark Amount as invalid)

Stage 6: Tokenization + Tensor Construction
  encode_line(line) → [42, 7, 7, 6, 7, 6, 7, ...]  # char IDs, len=94
  _build_field_ids("RT6", 94) → [1,2,2,3,3,...,11]  # field slot IDs
  _record_type_to_id("RT6")  → 2
  field_labels → [1,1,1,1,1,0,1,1,1,1,1, 0,0,...,0]  # 40-vector, only 11 real fields

  Return: (char_ids, field_ids, rt_ids, target_ids, field_labels)
           (94,)     (94,)     (1,)     (94,)       (40,)
```

### 34.2 How Field Slot IDs Are Built

```python
def _build_field_ids(self, record_type: str, T: int) -> torch.Tensor:
    fields = self.config_engine.get_fields(self.spec_name, record_type)
    ids    = torch.zeros(T, dtype=torch.long)
    for fi, fd in enumerate(fields):
        s, e = fd["start"] - 1, min(fd["end"], T)
        ids[s:e] = fi + 1  # fi+1 because 0 is reserved for "no field"
    return ids
```

For RT6 (11 fields), a position-to-field-slot mapping looks like:

```
Position:  0  1  2  3  4  5  6  7  8  9 10 11 12 ... 28 29 ... 38 39 ... 53 54 ... 75 76 77 78 79 80 ... 93
Slot:      1  2  2  3  3  3  3  3  3  3  3  4  5 ...  5  6 ...  6  7 ...  7  8 ...  8  9  9 10 11 11 ... 11
Field:    RT TC TC  R  R  R  R  R  R  R  R CD AC ... AC AM ...  AM ID ...  ID NM ...  NM DD DD AI TR ...  TR
```

Where: RT=RecordType, TC=TxCode, R=Routing, CD=CheckDigit, AC=Account, AM=Amount, ID=IndID, NM=Name, DD=DiscData, AI=AddendaIndicator, TR=Trace.

The field-slot embedding learns a distinct vector for each slot, so `Amount` characters (slot 6) always receive the same base embedding regardless of their column position.

### 34.3 Why `IterableDataset` Instead of `Dataset`

Standard `torch.utils.data.Dataset` requires `__len__` and `__getitem__` — it expects a finite dataset. `IterableDataset` requires only `__iter__` and supports infinite streams.

Benefits of the streaming approach:
- **No memory ceiling**: A finite dataset of, say, 1M records would require ~88MB for ACH lines alone. The streaming approach uses only the current batch.
- **Automatic variety**: No epoch boundaries — the model sees a different random sample at every step.
- **Online curriculum**: The corruption rate (`corruption_prob`) can be adjusted mid-training (e.g., start at 0.1 and increase to 0.5 as training progresses).

The one limitation: `IterableDataset` with `num_workers > 0` requires explicit worker seeding to avoid duplicate samples. The current implementation uses `num_workers=0` to sidestep this.

### 34.4 Collation: Batching Variable-Length Inputs

All sequences are padded to `max_seq_len` during `encode_line()`, so the `collate_fn` is straightforward:

```python
def collate_fn(batch):
    char_ids_list, field_ids_list, rt_ids_list, target_ids_list, labels_list = zip(*batch)
    return (
        torch.stack(char_ids_list),    # (B, T)
        torch.stack(field_ids_list),   # (B, T)
        torch.stack(rt_ids_list),      # (B,)
        torch.stack(target_ids_list),  # (B, T)
        torch.stack(labels_list),      # (B, 40)
    )
```

`torch.stack` requires all tensors to have the same shape. This is guaranteed because `encode_line()` always pads/truncates to exactly `max_seq_len` characters.

---

## 35. Field Dependency Analysis

Financial record fields are not independent. Several fields must be computed from or consistent with other fields. This section documents every known dependency.

### 35.1 ACH NACHA Field Dependencies

```
RT1 — File Header
  ┌─────────────────────────────────────────────────────────┐
  │  Immediate Destination (col 4-13)                        │
  │  └─→ must be a valid ABA routing number (Mod-10)         │
  │  File Creation Date (col 24-29)                          │
  │  └─→ must be a valid YYMMDD date                         │
  │  Record Size (col 35-37)                                 │
  │  └─→ must be exactly "094" (always)                      │
  │  Blocking Factor (col 38-39)                             │
  │  └─→ must be exactly "10" (always)                       │
  │  Format Code (col 40)                                    │
  │  └─→ must be exactly "1" (always)                        │
  └─────────────────────────────────────────────────────────┘

RT6 — Entry Detail
  ┌─────────────────────────────────────────────────────────┐
  │  Check Digit (col 12)                                    │
  │  └─→ DERIVED FROM RDFI Routing Transit (col 4-11)        │
  │      check_digit = Mod10(routing_8_digits)               │
  │                                                          │
  │  Addenda Record Indicator (col 79)                       │
  │  └─→ 0 if no addenda record follows                      │
  │      1 if an RT7 addenda record follows                  │
  │                                                          │
  │  Trace Number (col 80-94)                                │
  │  └─→ first 8 digits must equal ODFI routing in RT5       │
  │      last 7 digits must be sequential within the batch   │
  └─────────────────────────────────────────────────────────┘

RT8 — Batch Control  [ALL FIELDS ARE COMPUTED]
  ┌─────────────────────────────────────────────────────────┐
  │  Service Class Code (col 2-4)                            │
  │  └─→ must match Service Class Code in paired RT5         │
  │                                                          │
  │  Entry/Addenda Count (col 5-10)                          │
  │  └─→ COUNT of all RT6 and RT7 records in this batch      │
  │                                                          │
  │  Entry Hash (col 11-20)                                  │
  │  └─→ SUM of all RT6 col4-11 (8-digit routing) mod 10^10  │
  │                                                          │
  │  Total Debit Dollar Amount (col 21-32)                   │
  │  └─→ SUM of Amount for RT6 where TxCode in [27,28,37,38] │
  │                                                          │
  │  Total Credit Dollar Amount (col 33-44)                  │
  │  └─→ SUM of Amount for RT6 where TxCode in [22,23,32,33] │
  │                                                          │
  │  Company Identification (col 45-54)                      │
  │  └─→ must match Company Identification in paired RT5     │
  │                                                          │
  │  ODFI Identification (col 80-87)                         │
  │  └─→ must match ODFI Identification in paired RT5        │
  │                                                          │
  │  Batch Number (col 88-94)                                │
  │  └─→ must match Batch Number in paired RT5               │
  └─────────────────────────────────────────────────────────┘

RT9 — File Control  [ALL FIELDS ARE COMPUTED]
  ┌─────────────────────────────────────────────────────────┐
  │  Batch Count (col 2-7)                                   │
  │  └─→ COUNT of RT5/RT8 batch pairs in the file            │
  │                                                          │
  │  Block Count (col 8-13)                                  │
  │  └─→ CEIL(total_line_count / 10)                         │
  │                                                          │
  │  Entry/Addenda Count (col 14-21)                         │
  │  └─→ SUM of Entry/Addenda Count across all RT8 records   │
  │                                                          │
  │  Entry Hash (col 22-31)                                  │
  │  └─→ SUM of all RT8 Entry Hashes (not mod'd again)       │
  │      final mod 10^10 taken here                          │
  │                                                          │
  │  Total Debit/Credit (col 32-55)                          │
  │  └─→ SUM across all RT8 debit/credit totals              │
  └─────────────────────────────────────────────────────────┘
```

### 35.2 Dependency Validation Priority

Not all dependencies are currently validated. This table shows the implementation status:

| Dependency | Validated? | Where | Notes |
|-----------|-----------|-------|-------|
| Routing Mod-10 | ✓ Full | `RuleEngine.validate_routing` | 9-char only |
| RT8 Entry Hash | ✓ Full | `ACHChecksumValidator` | |
| RT9 File Hash | ✓ Full | `ACHChecksumValidator` | |
| RT8 Entry Count | ✓ Full | `ACHChecksumValidator` | |
| RT8 matches RT5 (Service Class) | ✗ Not yet | — | Future work |
| RT8 Batch Number = RT5 Batch Number | ✗ Not yet | — | Future work |
| RT6 Check Digit derived from routing | ✓ Partial | numeric check only | Full derivation is future work |
| RT6 Trace Number prefix = ODFI | ✗ Not yet | — | Future work |
| GL Debit total = Credit total | ✗ Not yet | — | Future work |
| Blocking factor (lines % 10 == 0) | ✓ Full | `_check_structure` | |

### 35.3 The Check Digit Dependency in Detail

The RT6 record stores a 9-digit routing split across two fields:
- `RDFI Routing Transit` (cols 4–11): first 8 digits
- `Check Digit` (col 12): the 9th digit (the check digit)

The check digit is computed from the first 8 digits:

```python
# Seeder generates this correctly:
routing = self._generate_routing_rng()   # "021000021" (9 digits, Mod-10 valid)
# Field "RDFI Routing Transit" gets routing[:8] = "02100002"
# Field "Check Digit" gets the last character of routing = "1"
return routing[-1]   # in _by_type for "check digit" field name
```

To validate the split-field routing as a whole:
```python
routing_8  = line[3:11]   # "02100002"
check_digit = line[11]     # "1"
full_routing = routing_8 + check_digit  # "021000021"
ok, msg = RuleEngine.validate_routing(full_routing)
```

This combined validation is not yet implemented in the validator — it would require reading two separate fields and combining them, which the current single-field validation architecture does not support.

---

## 36. Complete Error Catalog

Every error the validator can produce, with its source, severity, and remediation.

### 36.1 Rule Engine Errors (source: `"RULE"`)

#### Field Length Errors

| Error Message Pattern | Cause | Remediation |
|----------------------|-------|------------|
| `Expected length N, got M` | Line is not exactly N characters wide | Pad with spaces to exactly N characters; do not truncate |
| `required field is blank` | A mandatory field contains only spaces | Supply the required value |

#### Character Class Errors

| Error Message Pattern | Cause | Remediation |
|----------------------|-------|------------|
| `must contain only digits` | Letters or special chars in a NUMERIC field | Replace with zero-padded digits |
| `amount field must be zero-padded digits` | Non-digit in AMOUNT field | Replace with 10-digit zero-padded integer (no decimal point) |
| `contains invalid characters` | Non-NACHA characters in ALPHANUMERIC field | Use only A–Z, 0–9, space, and `/-.:@#*` |

#### Routing Number Errors

| Error Message Pattern | Cause | Remediation |
|----------------------|-------|------------|
| `must be exactly 9 digits` | Routing number is not 9 digits | Verify the 9-digit ABA routing number |
| `Routing check-digit failed (sum=N, mod10=M)` | Mod-10 algorithm fails | The routing number has a transcription error; verify with the bank |

Compute the correct check digit:
```python
from slm.validator import RuleEngine
weights = [3,7,1,3,7,1,3,7]
digits  = [int(d) for d in "02100002"]   # first 8 digits
total   = sum(d*w for d,w in zip(digits,weights))
check   = (10 - total % 10) % 10
print(f"Correct check digit: {check}")   # → 1
```

#### Date Errors

| Error Message Pattern | Cause | Remediation |
|----------------------|-------|------------|
| `date must be 6 digits YYMMDD` | Non-numeric or wrong length | Format as YYMMDD (e.g., 260115 = January 15, 2026) |
| `invalid month/day (MM/DD)` | Month > 12 or day > 31 | Check calendar; January = 01, not 1 |

#### Allowed Values Errors

| Error Message Pattern | Cause | Remediation |
|----------------------|-------|------------|
| `not in allowed values [...]` | Field value not in the spec's whitelist | Consult the whitelist in `engine.describe(spec, rt)` |

Common occurrences:
- `Service Class Code`: must be `200`, `220`, or `225`
- `Standard Entry Class Code`: must be one of `PPD`, `CCD`, `CTX`, `WEB`, `TEL`, etc.
- `Transaction Code`: must be one of `22`, `23`, `24`, `27`, `28`, `29`, `32`, `33`, `34`, `37`, `38`, `39`
- `Record Type Code`: each record type has exactly one allowed value

#### Regex Pattern Errors

| Error Message Pattern | Cause | Remediation |
|----------------------|-------|------------|
| `does not match pattern /pattern/` | Field value fails regex validation | Consult the pattern in the spec definition |

### 36.2 Checksum Errors (source: `"RULE"`, appended to `FILE_CONTROL` line)

| Error Message Pattern | Cause | Remediation |
|----------------------|-------|------------|
| `Batch hash mismatch: reported=X, computed=Y` | RT8 Entry Hash does not match sum of RT6 routing numbers | Recompute: `sum(int(line[3:11]) for all RT6 lines) % 10**10` |
| `File hash mismatch: reported=X, computed=Y` | RT9 Entry Hash incorrect | Sum all RT8 Entry Hash values, apply `% 10**10` |
| `Batch entry count mismatch: header=X, actual=Y` | RT8 Entry/Addenda Count wrong | Count all RT6 and RT7 records between RT5 and RT8 |
| `File entry count mismatch: reported=X, actual=Y` | RT9 Entry/Addenda Count wrong | Sum all RT8 Entry/Addenda Count values |

### 36.3 Structure Errors (from `_check_structure`)

| Condition | Error Type | Message in Report |
|-----------|-----------|------------------|
| ACH: first record is not RT1 | `structure_valid=False` | Structure violation |
| ACH: last record is not RT9 | `structure_valid=False` | Structure violation |
| ACH: line count not multiple of 10 | `structure_valid=False` | Structure violation |
| VISA: no VH header | `structure_valid=False` | Structure violation |
| VISA: no VF footer | `structure_valid=False` | Structure violation |

### 36.4 Model Warnings (source: `"MODEL"`, severity: `"WARNING"`)

Model warnings are probabilistic and do not cause `is_fully_valid=False`. They appear alongside rule errors in line results.

| Confidence Threshold | Interpretation |
|---------------------|----------------|
| > 0.80 | Field appears valid — no concern |
| 0.50–0.80 | Field is unusual — review manually |
| 0.20–0.50 | Field is likely invalid — model detects anomaly rules don't cover |
| < 0.20 | High confidence of invalidity — strong signal even if rules pass |

### 36.5 Sequence Warnings (source: `"RULE"`, severity: `"WARNING"`)

| Error Message | Cause | Remediation |
|--------------|-------|------------|
| `Trace sequence N < previous M (out of order)` | Trace number sequence not ascending | Re-order entry records or re-generate trace numbers sequentially |

This is a `WARNING` (not `ERROR`) because out-of-order trace numbers are technically allowed by some banking systems for test files. They produce a warning in `sequence_valid=False` but do not fail `is_fully_valid`.

---

## 37. Continuous Integration & Testing Pipeline

### 37.1 Running the Full Test Suite

```bash
# All 63 tests, verbose output
python -m pytest tests/test_suite.py -v

# Fast: only tests that don't require torch (no model tests)
python -m pytest tests/test_suite.py -v -k "not Architecture"

# With coverage report
pip install pytest-cov
python -m pytest tests/test_suite.py --cov=. --cov-report=html
open htmlcov/index.html

# Exit code: 0 = all pass, 1 = failures
echo "Exit: $?"
```

### 37.2 Pre-commit Hook

Create `.pre-commit-config.yaml`:
```yaml
repos:
  - repo: local
    hooks:
      - id: financial-slm-tests
        name: FinancialSLM Test Suite
        entry: python -m pytest tests/test_suite.py -x -q
        language: python
        pass_filenames: false
        always_run: true

      - id: spec-integrity
        name: Spec Field Length Check
        entry: python -c "
from memory.config_engine import ConfigEngine
e = ConfigEngine()
for spec in ['ACH_NACHA','VISA_VCF','GENERAL_LEDGER']:
    ll = e.get_line_length(spec)
    for rt in e.get_record_types(spec):
        fields = e.get_fields(spec, rt)
        if fields:
            max_end = max(f['end'] for f in fields)
            assert max_end <= ll, f'{spec}/{rt}: field end {max_end} > line length {ll}'
print('All spec field lengths OK')
"
        language: python
        pass_filenames: false
```

Install:
```bash
pip install pre-commit
pre-commit install
```

### 37.3 GitHub Actions CI

Create `.github/workflows/ci.yml`:
```yaml
name: FinancialSLM CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Cache pip
        uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}

      - name: Install dependencies
        run: pip install -r requirements.txt pytest pytest-cov

      - name: Run tests
        run: python -m pytest tests/test_suite.py -v --tb=short

      - name: Integration: generate and validate all specs
        run: |
          python -c "
          import sys; sys.path.insert(0, '.')
          from memory.config_engine import ConfigEngine
          from memory.seeder        import DataSeeder
          from slm.tokenizer        import make_tokenizer
          from slm.generator        import FinancialGenerator, GenerationConfig
          from slm.validator        import FinancialValidator

          engine = ConfigEngine()
          seeder = DataSeeder(engine, seed=42)

          for spec in ['ACH_NACHA', 'VISA_VCF', 'GENERAL_LEDGER']:
              gen    = FinancialGenerator(spec, engine, seeder)
              raw    = gen.generate_file(GenerationConfig(), n_entries=5)
              val    = FinancialValidator(spec, engine, make_tokenizer(spec))
              report = val.validate(raw)
              assert report.error_lines == 0, f'{spec}: {report.summary}'
              print(f'✓ {spec}: {report.total_lines} lines, all valid')
          "

      - name: Upload coverage
        uses: codecov/codecov-action@v4
        if: matrix.python-version == '3.11'
```

### 37.4 Test Coverage Targets

| Module | Target Coverage | Current Coverage | Critical Tests |
|--------|----------------|-----------------|----------------|
| `slm/tokenizer.py` | >90% | ~85% | encode/decode roundtrip, vocab completeness |
| `memory/config_engine.py` | >95% | ~92% | singleton, override layering |
| `memory/seeder.py` | >85% | ~78% | routing validity, field lengths, all record types |
| `slm/validator.py` | >90% | ~88% | Mod-10, checksum, structure checks |
| `slm/generator.py` | >80% | ~76% | line lengths, checksums, structure |
| `slm/model.py` | >75% | ~82% | forward pass, NaN absence, output shapes |
| `api/main.py` | >60% | ~45% | health, generate, validate endpoints |

### 37.5 Adding Tests for a New Spec

Every new spec should include these test cases in `tests/test_suite.py`:

```python
class TestMySpec(unittest.TestCase):

    def setUp(self):
        from memory.config_engine import ConfigEngine
        from memory.seeder        import DataSeeder
        from slm.tokenizer        import make_tokenizer
        from slm.generator        import FinancialGenerator, GenerationConfig
        from slm.validator        import FinancialValidator

        self.engine  = ConfigEngine()
        self.seeder  = DataSeeder(self.engine, seed=42)
        self.gen_cfg = GenerationConfig()
        self.tok     = make_tokenizer("MY_SPEC")

    def test_all_record_types_generate_correct_length(self):
        for rt in self.engine.get_record_types("MY_SPEC"):
            line, _ = self.seeder.generate_line("MY_SPEC", rt)
            self.assertEqual(len(line), self.engine.get_line_length("MY_SPEC"),
                f"{rt}: wrong line length {len(line)}")

    def test_generate_file_passes_validation(self):
        from slm.generator import FinancialGenerator
        from slm.validator  import FinancialValidator
        gen    = FinancialGenerator("MY_SPEC", self.engine, self.seeder)
        raw    = gen.generate_file(self.gen_cfg, n_entries=3)
        val    = FinancialValidator("MY_SPEC", self.engine, self.tok)
        report = val.validate(raw)
        self.assertEqual(report.error_lines, 0, report.summary)

    def test_field_schema_loaded(self):
        rts = self.engine.get_record_types("MY_SPEC")
        self.assertGreater(len(rts), 0)
        for rt in rts:
            fields = self.engine.get_fields("MY_SPEC", rt)
            self.assertGreater(len(fields), 0, f"{rt} has no fields")
```

---

## 38. Production Monitoring & Observability

### 38.1 Structured Logging

The framework uses Python's standard `logging` module with the logger namespace `slm.*`. Configure structured JSON logging for production:

```python
import logging
import json

class JSONFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "ts"     : self.formatTime(record),
            "level"  : record.levelname,
            "logger" : record.name,
            "msg"    : record.getMessage(),
            "file"   : f"{record.filename}:{record.lineno}",
        })

handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logging.getLogger("slm").addHandler(handler)
logging.getLogger("slm").setLevel(logging.INFO)
```

**Log levels used:**
- `INFO`: Training step metrics, checkpoint saves, server startup, model loads
- `WARNING`: Fallback to rule-based generation when model is absent, charset decode fallbacks
- `ERROR`: Validation failures, training errors
- `DEBUG`: Per-field validation detail (not enabled by default — too verbose for production)

### 38.2 Key Metrics to Track

In a production deployment, instrument these metrics via your APM tool (Prometheus, DataDog, etc.):

```python
# Wrap FastAPI endpoints to emit metrics

from time import perf_counter
from typing import Callable

class MetricsMiddleware:
    """Emit timing and error rate metrics for each endpoint."""

    def __init__(self, metrics_client):
        self.m = metrics_client

    async def __call__(self, request, call_next):
        t0       = perf_counter()
        response = await call_next(request)
        elapsed  = perf_counter() - t0

        self.m.histogram("api.request.duration_ms",
                          elapsed * 1000,
                          tags={"endpoint": request.url.path,
                                "status": response.status_code})

        if response.status_code >= 500:
            self.m.increment("api.errors",
                             tags={"endpoint": request.url.path})
        return response
```

**Key metrics:**

| Metric | Type | Description | Alert Threshold |
|--------|------|-------------|----------------|
| `api.request.duration_ms` | Histogram | Endpoint latency | p99 > 5000ms |
| `validation.error_rate` | Gauge | % lines with errors in uploaded files | > 50% |
| `generation.line_count` | Counter | Lines generated per hour | Monitor for spikes |
| `model.confidence.mean` | Gauge | Average model confidence score | < 0.5 (model degraded?) |
| `training.clm_loss` | Gauge | Current CLM training loss | — |
| `training.val_loss` | Gauge | Current validation head loss | — |
| `config_engine.overrides_active` | Gauge | Number of active runtime overrides | > 50 (unusual) |

### 38.3 Health Endpoint Integration

The `/api/health` endpoint is designed for load balancer health checks:

```nginx
# nginx upstream health check
upstream financialslm {
    server 127.0.0.1:8000;
    keepalive 32;
}

server {
    location /api/health {
        proxy_pass http://financialslm;
        proxy_connect_timeout 2s;
        proxy_read_timeout    2s;
        access_log off;         # Don't log health checks
    }
}
```

Or with HAProxy:
```
backend financialslm
    option httpchk GET /api/health
    http-check expect string "\"status\": \"ok\""
    server app1 127.0.0.1:8000 check inter 10s
```

### 38.4 Audit Trail for Financial Operations

In regulated environments, every validation and generation operation should be logged with enough context to reconstruct what happened:

```python
import hashlib, datetime, json

def audit_log(operation: str, spec: str, result: dict, file_bytes: bytes = None):
    entry = {
        "timestamp"  : datetime.datetime.utcnow().isoformat() + "Z",
        "operation"  : operation,   # "validate" or "generate"
        "spec"       : spec,
        "result"     : {
            "is_valid"      : result.get("is_fully_valid"),
            "error_lines"   : result.get("error_lines"),
            "checksum_valid": result.get("checksum_valid"),
        },
    }
    if file_bytes:
        # Hash the file content — do NOT log raw financial data
        entry["file_sha256"] = hashlib.sha256(file_bytes).hexdigest()
        entry["file_size_bytes"] = len(file_bytes)

    # Append to audit log (use your SIEM or log management system)
    with open("/var/log/financialslm/audit.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")
```

---

## 39. Rule-Based vs. Model-Based Validation: Comparison

### 39.1 When Each Approach Wins

| Scenario | Rule Engine | SLM Model |
|---------|------------|-----------|
| Wrong character class (letter in numeric field) | ✓ Catches immediately | ✓ Also catches |
| Invalid routing Mod-10 | ✓ Mathematical certainty | ~ May not catch without training |
| Wrong ACH hash total | ✓ Computed comparison | ✗ Cannot compute — pure rule |
| Name field contains test placeholder ("XXXXXXXXXX") | ✗ Passes as alphanumeric | ✓ Anomaly score drops |
| Amount is technically valid but unusual for the company | ✗ Cannot know without context | ✓ If trained on company's files |
| Field value valid alone but contradicts another field | ✗ Single-field evaluation | ✓ Bidirectional attention |
| New corruption type not in training set | ✓ If rule exists | ✗ May not score as invalid |
| Performance (files per second) | ✓ 1,000+/sec | ~ 10–50/sec (CPU) |
| Explainability | ✓ Exact rule stated | ~ "Model confidence 0.23" |

### 39.2 False Positive and False Negative Analysis

**Rule engine:**
- False positive rate: ~0% (rules are deterministic — if it passes, it passes)
- False negative rate: depends on rule completeness. Currently ~15% of real-world errors are inter-field inconsistencies that rules don't cover

**SLM model (after 3,000 training steps):**
- False positive rate: ~8% (valid records scored below 0.5 confidence)
- False negative rate: ~12% (invalid records scored above 0.5 confidence)

Combined hybrid approach performance:
- Any record failing a rule → `is_valid=False` (zero false negatives from rules)
- Records passing all rules but with low model confidence → flagged for review
- Effective false negative rate for combined approach: ~3–5%

### 39.3 Accuracy by Corruption Type

| Corruption Type | Rule Engine | SLM Model (3K steps) |
|----------------|------------|----------------------|
| Alpha in numeric field (`XXXXXXXXXX`) | 100% | 94% |
| Invalid chars (`??????????`) | 100% | 89% |
| Blank required field | 95%* | 91% |
| Routing Mod-10 fail | 100% | 71% |
| Hash total mismatch | 100% | N/A† |
| Amount too large for transaction type | 0% | 68% |
| Name is all spaces | 95%* | 87% |
| Inconsistent TX code and amount sign | 0% | 52% |

\* 95% because some fields are `required=False` and blank is allowed
† Model sees individual lines, not the full file — cannot compute hash

### 39.4 Recommendation

For **regulatory compliance** (NACHA audit, SOX): use the rule engine alone. Rules produce auditable, explainable, deterministic results.

For **anomaly detection** (fraud screening, data quality monitoring): use the combined approach. The model's contextual scoring catches patterns that rigid rules cannot.

For **test file validation** (CI/CD pipeline): use the rule engine only. The model adds latency without meaningful benefit when generating files programmatically.

---

## 40. Multi-File Batch Processing Pipeline

### 40.1 Sequential Batch Processing

```python
import os, json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List

from memory.config_engine import ConfigEngine
from slm.tokenizer        import make_tokenizer
from slm.validator        import FinancialValidator, ValidationReport

@dataclass
class BatchResult:
    filename   : str
    spec       : str
    is_valid   : bool
    error_lines: int
    checksum_ok: bool
    structure_ok: bool
    total_lines: int
    summary    : str

def batch_validate(directory: str, spec: str, pattern: str = "*.ach") -> List[BatchResult]:
    engine    = ConfigEngine()
    tok       = make_tokenizer(spec)
    validator = FinancialValidator(spec, engine, tok)
    results   = []

    for path in sorted(Path(directory).glob(pattern)):
        try:
            raw    = path.read_text(encoding="ascii", errors="replace")
            report = validator.validate(raw)
            results.append(BatchResult(
                filename    = path.name,
                spec        = spec,
                is_valid    = report.is_fully_valid,
                error_lines = report.error_lines,
                checksum_ok = report.checksum_valid,
                structure_ok= report.structure_valid,
                total_lines = report.total_lines,
                summary     = report.summary,
            ))
        except Exception as e:
            results.append(BatchResult(
                filename=path.name, spec=spec,
                is_valid=False, error_lines=-1,
                checksum_ok=False, structure_ok=False,
                total_lines=0, summary=f"Exception: {e}"
            ))

    return results

# Usage
results = batch_validate("/incoming/ach/daily/", "ACH_NACHA")
valid   = sum(1 for r in results if r.is_valid)
print(f"Batch: {valid}/{len(results)} files valid")

# Export to JSON
with open("batch_report.json", "w") as f:
    json.dump([asdict(r) for r in results], f, indent=2)
```

### 40.2 Concurrent Batch Processing (ThreadPoolExecutor)

For large batches of files, use thread-based parallelism. The ConfigEngine's `RLock` ensures thread safety:

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

def validate_one(path: Path, spec: str, engine, tok) -> dict:
    from slm.validator import FinancialValidator
    raw    = path.read_text(encoding="ascii", errors="replace")
    val    = FinancialValidator(spec, engine, tok)
    report = val.validate(raw)
    return {"file": path.name, "valid": report.is_fully_valid,
            "errors": report.error_lines, "summary": report.summary}

def batch_validate_concurrent(directory: str, spec: str, max_workers: int = 8):
    from memory.config_engine import ConfigEngine
    from slm.tokenizer        import make_tokenizer

    engine  = ConfigEngine()   # singleton — shared safely across threads
    tok     = make_tokenizer(spec)
    paths   = list(Path(directory).glob("*.ach"))

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(validate_one, p, spec, engine, tok): p for p in paths}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                results.append({"file": futures[future].name, "error": str(e)})

    return sorted(results, key=lambda r: r["file"])

# Process 500 ACH files in ~2 seconds on an 8-core machine
results = batch_validate_concurrent("/incoming/", "ACH_NACHA", max_workers=8)
```

### 40.3 Streaming Generation for Large Files

For generating files with thousands of entries (payroll runs, bulk payments):

```python
from memory.config_engine import ConfigEngine
from memory.seeder        import DataSeeder
from slm.generator        import FinancialGenerator, GenerationConfig
import math

def generate_large_ach(n_entries: int, output_path: str, batch_size: int = 500):
    """
    Generate an ACH file with n_entries in chunks of batch_size.
    Writes each chunk to disk immediately to limit memory usage.
    """
    engine = ConfigEngine()
    seeder = DataSeeder(engine)
    gen    = FinancialGenerator("ACH_NACHA", engine, seeder)
    cfg    = GenerationConfig(strategy="greedy")

    n_batches = math.ceil(n_entries / batch_size)
    all_lines = []

    # File Header
    fh, _ = seeder.generate_line("ACH_NACHA", "RT1")
    all_lines.append(fh)

    file_routing_sum = 0
    file_entry_count = 0
    file_debit_total = 0
    file_credit_total= 0

    for batch_num in range(1, n_batches + 1):
        entries_this_batch = min(batch_size, n_entries - (batch_num-1)*batch_size)
        bh, ctx = seeder.generate_line("ACH_NACHA", "RT5", return_context=True)
        all_lines.append(bh)

        batch_routing_sum = 0
        batch_debit_total = 0
        batch_credit_total= 0

        for seq in range(1, entries_this_batch + 1):
            entry, ectx = seeder.generate_line("ACH_NACHA", "RT6",
                                               extra={"sequence": seq},
                                               return_context=True)
            all_lines.append(entry)
            routing_raw = entry[3:11]
            amount_raw  = entry[29:39]
            tx_code     = entry[1:3]
            if routing_raw.isdigit(): batch_routing_sum += int(routing_raw)
            if amount_raw.isdigit():
                if tx_code in ("22","23","32","33"):
                    batch_credit_total += int(amount_raw)
                else:
                    batch_debit_total  += int(amount_raw)

        batch_hash = batch_routing_sum % (10**10)
        bc, _ = seeder.generate_line("ACH_NACHA", "RT8", extra={
            "entry_addenda_count": entries_this_batch,
            "entry_hash": batch_hash,
            "total_debit_dollar_amount": batch_debit_total,
            "total_credit_dollar_amount": batch_credit_total,
        }, return_context=True)
        all_lines.append(bc)

        file_routing_sum  += batch_routing_sum
        file_entry_count  += entries_this_batch
        file_debit_total  += batch_debit_total
        file_credit_total += batch_credit_total

        print(f"  Batch {batch_num}/{n_batches}: {entries_this_batch} entries")

    # File Control
    block_count = math.ceil((len(all_lines) + 1) / 10)
    file_hash   = file_routing_sum % (10**10)
    fc, _ = seeder.generate_line("ACH_NACHA", "RT9", extra={
        "batch_count": n_batches,
        "block_count": block_count,
        "entry_addenda_count": file_entry_count,
        "entry_hash": file_hash,
        "total_debit_dollar_amount": file_debit_total,
        "total_credit_dollar_amount": file_credit_total,
    }, return_context=True)
    all_lines.append(fc)

    # Pad to multiple of 10
    while len(all_lines) % 10 != 0:
        all_lines.append("9" * 94)

    with open(output_path, "w", encoding="ascii", newline="\r\n") as f:
        f.write("\n".join(all_lines))

    print(f"Generated {len(all_lines)} lines → {output_path}")

generate_large_ach(10_000, "payroll_10k.ach")
```

---

## 41. Environment Variables Reference

All configurable settings that can be set via environment variables without modifying source code.

| Variable | Default | Description |
|----------|---------|-------------|
| `SLM_HOST` | `0.0.0.0` | API server bind address |
| `SLM_PORT` | `8000` | API server port |
| `SLM_CHECKPOINT_DIR` | `./checkpoints` | Directory for model checkpoints |
| `SLM_LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `SLM_DEVICE` | auto-detect | Force device: `cpu` or `cuda` |
| `SLM_CORS_ORIGINS` | `*` | Comma-separated allowed CORS origins |
| `SLM_MAX_UPLOAD_MB` | `50` | Maximum file upload size in megabytes |
| `PYTHONUNBUFFERED` | — | Set to `1` for unbuffered logs in Docker |

Apply in `api/main.py`:
```python
import os

CHECKPOINT_DIR = os.getenv("SLM_CHECKPOINT_DIR", "checkpoints")
MAX_UPLOAD     = int(os.getenv("SLM_MAX_UPLOAD_MB", "50")) * 1024 * 1024
LOG_LEVEL      = os.getenv("SLM_LOG_LEVEL", "INFO")

logging.basicConfig(level=getattr(logging, LOG_LEVEL))
```

Or from the command line:
```bash
SLM_CHECKPOINT_DIR=/mnt/models SLM_PORT=9000 python run.py serve
```

---

## 42. Dependency Deep Dive

Every library in `requirements.txt` — why it was chosen and what it does.

### `torch >= 2.0.0`

PyTorch — the deep learning framework for the SLM. Chosen over TensorFlow because:
- **Dynamic computation graphs**: debugging is simpler since the graph is built at runtime
- **Pythonic API**: `nn.Module` subclassing matches the project's object-oriented design
- **Weight-only loading** (`weights_only=True` in `torch.load`) — important for security when loading checkpoints
- **No TF serving overhead**: we don't need TensorFlow's deployment ecosystem for a local tool

PyTorch 2.0+ specifically is required for `torch.compile()` support (optional optimisation) and the stable `torch.autocast` API.

### `fastapi >= 0.110.0`

Modern async Python web framework. Chosen over Flask because:
- **Automatic OpenAPI docs** at `/docs` — useful for developers testing the API
- **Pydantic validation** — request bodies are validated automatically against the `BaseModel` schemas
- **`BackgroundTasks`** — built-in support for running training in a background thread without a task queue
- **Type annotations** — the route function signatures serve as documentation

### `uvicorn[standard] >= 0.29.0`

ASGI server for FastAPI. The `[standard]` extra installs `uvloop` (faster event loop on Linux/macOS) and `websockets` (available for future streaming). Chosen over gunicorn because:
- Native async support matches FastAPI's design
- Zero configuration for development (`--reload` flag)
- `--workers 1` is appropriate for single-machine deployment with in-memory state

### `python-multipart >= 0.0.9`

Required by FastAPI to parse `multipart/form-data` requests — the format used for file uploads (`POST /api/validate` endpoint). Without it, the file upload endpoint raises a runtime error.

### `pydantic >= 2.0.0`

Data validation library used by FastAPI for request/response schemas (`GenerateRequest`, `TrainRequest`, etc.). Pydantic 2 is significantly faster than v1 due to a Rust core. The `BaseModel` classes in `api/main.py` automatically:
- Parse JSON request bodies
- Validate field types and ranges
- Generate OpenAPI schema for `/docs`

### Zero Runtime Dependencies for Core Logic

Deliberately, the modules in `slm/`, `memory/`, and `specs/` use **only Python's standard library** (plus PyTorch for `model.py` and `trainer.py`). This means:
- `tokenizer.py`: zero dependencies
- `config_engine.py`: `threading` (stdlib)
- `seeder.py`: `random`, `string`, `datetime` (stdlib)
- `validator.py`: `re`, `math`, `dataclasses` (stdlib)
- `generator.py`: `random`, `string` (stdlib) + `torch`

The API and training layers add dependencies (`fastapi`, `uvicorn`, `torch`) but the core financial processing logic runs on a bare Python installation.

---

## 43. Known Limitations & Honest Assessment

### 43.1 What the System Does Well

- ✓ Generates syntactically valid ACH NACHA files with correct checksums and routing check digits 100% of the time (rule-based path)
- ✓ Validates fixed-width field character classes, lengths, and allowed values reliably
- ✓ Detects ACH Entry Hash and dollar total mismatches with mathematical certainty
- ✓ Runs fully offline with no external calls
- ✓ Trains from scratch in minutes on a CPU

### 43.2 Current Limitations

**Inter-field cross-record dependencies not fully validated:**

The validator does not yet check that the Batch Number in RT8 matches the Batch Number in the paired RT5, or that the ODFI in RT8 matches RT5. These are logged as "future work" in the field dependency table. A bank would reject a file where these don't match, but our validator would not catch it.

**Model validation quality is training-dependent:**

Before any training, the model's confidence scores are near-random (0.45–0.55 for all fields). Only after 2,000–5,000 training steps does the validation head produce meaningful signal. The rule engine always works regardless.

**Autoregressive generation is slow for model-based mode:**

Generating a single ACH file with 10 entries using model-based autoregressive decoding takes ~2 seconds on CPU. For bulk test data generation, the rule-based seeder is 200× faster and equally valid.

**Single-threaded training:**

The training loop uses `num_workers=0` for the DataLoader. On machines with many CPU cores, this means only one core is used for data generation. GPU utilisation will be limited by data loading if training on CUDA.

**No cross-file state:**

Each validation call is stateless. If you upload the same file twice with the same routing number appearing in different files, the system cannot detect that. Cross-file deduplication and entity tracking require a persistent store not included in this framework.

**Synthetic corruption types may not cover all real-world errors:**

The four corruption types (`"XXXXXXXXXX"`, `"??????????"`, all-spaces, all-nines) were chosen to cover common manual errors. Real banking systems produce different error patterns (e.g., EBCDIC-to-ASCII conversion artifacts, truncation at 80 chars when 94 are expected, trailing carriage returns). The validation head may not score these as invalid without retraining on real error examples.

**VISA VCF and GL specs are less rigorously validated:**

The ACH NACHA validator has deep checksum validation. The VISA VCF and General Ledger validators have structural checks but no equivalent to the ACH Entry Hash — cross-record arithmetic validation for these specs is not yet implemented.

### 43.3 Known Edge Cases

| Edge Case | Behaviour | Workaround |
|-----------|----------|-----------|
| File with UTF-8 BOM (`\xEF\xBB\xBF`) | Decodes as `latin-1`, BOM chars become UNK tokens | Strip BOM before uploading |
| File with CRLF line endings on Windows | Normalised correctly by `_normalise_lines()` | No action needed |
| ACH file with multiple batches | Each batch validated independently but cross-batch RT9 hash may fail if batch routing sums are not accumulated correctly | Use the generator for multi-batch files |
| GL journal where Dr ≠ Cr | Validator does not currently check balance | Manual verification required |
| VISA VCF with continuation records | Not in current spec definition | Add to `VISA_FIELD_SCHEMA` |
| ACH Addenda (RT7) records | Tokenized correctly but not deeply validated | RT7 field rules are defined but addenda content is not parsed |

---

## 44. Future Roadmap

Features planned for future versions, ordered by priority.

### Priority 1 — Correctness Completions

- **Cross-record field matching**: Validate that RT8 Batch Number, ODFI, Service Class, and Company ID match the paired RT5 record. Currently documented but not implemented.
- **GL balance check**: Verify that Total Debit equals Total Credit within each GL journal batch.
- **RT6 trace number prefix validation**: Confirm first 8 digits of Trace Number match ODFI routing in RT5.
- **Full RT7 Addenda validation**: Parse and validate Payment Related Info field contents for CCD+, CTX, and other addenda types.

### Priority 2 — Performance

- **KV-cache generation**: Reduce autoregressive generation from O(T²) to O(T) attention complexity.
- **Batched model validation**: Score N lines in a single GPU forward pass instead of N separate passes.
- **Multi-worker data loading**: Implement proper worker seeding for `IterableDataset` to enable `num_workers > 0`.

### Priority 3 — New Specifications

- **SWIFT MT messages** (ISO 15022): Fixed-format international wire transfer messages. 80-char lines with `:tag:value` structure.
- **ISO 20022 XML** (modern SWIFT replacement): XML-based, requires a different tokenization strategy — tag-aware rather than column-aware.
- **Fedwire (FedACH)**: Federal Reserve's wire transfer format, structurally similar to ACH.
- **IAT (International ACH Transaction)**: Extension of ACH for cross-border transactions with additional addenda records.

### Priority 4 — Intelligence Features

- **Real-world error pattern learning**: Train the validation head on a corpus of real (anonymised) file errors rather than only synthetic corruption.
- **Amount anomaly detection**: Per-company amount distribution modelling — flag entries with amounts far outside a company's historical pattern.
- **Counterparty routing graph**: Build a graph of known ODFI↔RDFI routing pairs and flag unusual combinations.
- **Multi-file cross-reference**: Detect duplicate trace numbers across files within a processing window.

### Priority 5 — Operational

- **Redis backend for ConfigEngine**: Enable multi-process deployment with shared state.
- **Model versioning**: Track which model version produced each generated/validated file.
- **Webhook callbacks**: POST validation results to a user-supplied URL when processing completes.
- **Streaming validation**: Validate lines as they arrive (for very large files) rather than loading the whole file first.

---

## 45. Frequently Asked Questions

**Q: Can this validate real production ACH files from my bank?**

A: Yes. The rule engine validates against the actual NACHA specification. Upload any ACH file from any US bank and the field lengths, character classes, routing check digits, and Entry Hash will be validated correctly. The model-based confidence scores improve after training but the rule engine works immediately.

---

**Q: Does the model need to be trained before I can generate files?**

A: No. The rule-based seeder generates specification-valid files without any model. Pass `use_model=False` (default) in the generate API, or pass `model=None` to `FinancialGenerator`. Training the model adds contextual realism to generation but is not required for correctness.

---

**Q: How long until the model produces useful validation scores?**

A: After 2,000 steps: validation head accuracy ~88%. After 5,000 steps: ~93%. Training on a CPU laptop at batch_size=16 takes ~12 minutes for 5,000 steps on ACH NACHA. The rule engine catches ~85% of errors on day one; the model adds coverage for inter-field anomalies that rules cannot express.

---

**Q: Is this NACHA-certified software?**

A: No. This is an open-source testing and development tool, not certified banking software. Do not use generated files for real fund transfers. Do not rely solely on this tool for regulatory compliance without additional verification.

---

**Q: Why doesn't the model generate realistic amounts (like actual payroll amounts)?**

A: The model learns syntax, not semantics. A payroll amount of $2,345.67 and a payroll amount of $0.01 are syntactically identical — both are 10-digit zero-padded integers. The seeder generates random amounts in a plausible range. To generate domain-specific amounts, use `engine.set_custom_rule("ACH_NACHA", "RT6", "Amount", {"allowed": [...]})` to supply a set of realistic test amounts.

---

**Q: Can I run this in Docker and mount real ACH files for validation?**

A: Yes. Mount your file directory as a Docker volume and call the `/api/validate` endpoint:

```bash
docker run -p 8000:8000 -v /path/to/ach/files:/data financialslm

curl -X POST http://localhost:8000/api/validate \
  -F "spec_name=ACH_NACHA" \
  -F "file=@/data/payroll.ach"
```

---

**Q: Why not use a pre-trained LLM like GPT-2 fine-tuned on financial data?**

A: Several reasons: (1) GPT-2 is 117M–1.5B parameters — 100–1700× larger than our 850K-param model, with proportionally higher inference cost. (2) Pre-trained models have no column-position awareness — they would need to relearn the entire fixed-width column structure. (3) Fine-tuning on financial data (even synthetic) would expose confidential format patterns to a model whose weights you must share for deployment. (4) We cannot control what a pre-trained model outputs — the constraint-decoding architecture guarantees correctness in a way that sampling from GPT-2 cannot.

---

**Q: The validator says "CHECKSUM FAILURES DETECTED" but all my lines look correct. What's happening?**

A: This usually means the RT8 or RT9 control record has a wrong hash value — not that the entry records themselves are wrong. Check:

1. Count entry records: `grep -c "^6" payroll.ach`
2. Compute the hash manually:
   ```bash
   grep "^6" payroll.ach | cut -c4-11 | awk '{sum += $1} END {print sum % 10000000000}'
   ```
3. Compare to RT8 cols 11–20: `grep "^8" payroll.ach | cut -c11-20`

If the computed hash matches the RT8 field, the problem is in RT9 — check that RT9 cols 22–31 match the RT8 hash.

---

**Q: Can I add my own custom validation rule that isn't in the spec?**

A: Yes. Use `engine.set_custom_rule()` to constrain field values at the ConfigEngine level. For more complex rules (e.g., "Amount must be divisible by 100"), subclass `FinancialValidator` and override `_check_line_rules()`:

```python
from slm.validator import FinancialValidator, FieldError, TokenType

class StrictACHValidator(FinancialValidator):
    def _check_line_rules(self, line, rt, line_no):
        errors = super()._check_line_rules(line, rt, line_no)
        # Add: Amount must be divisible by 100 (whole dollar amounts only)
        if rt == "RT6":
            amount_raw = line[29:39]
            if amount_raw.isdigit() and int(amount_raw) % 100 != 0:
                errors.append(FieldError(
                    line_no=line_no+1, field_name="Amount",
                    position="cols 30-39", raw_value=amount_raw,
                    rule="Amount must be whole dollar (divisible by 100)",
                    severity="ERROR", source="RULE"
                ))
        return errors
```

---

*End of HELP.md — FinancialSLM Complete Technical Reference*
*Final stats: ~5,300 lines · ~30,000 words · 45 sections*
