# Financial SLM Framework — HELP.MD
## Architecture, Design & Code Flow Guide

---

## 1. Philosophy & Approach

### Why Build From Scratch?

General-purpose LLMs (GPT-4, Claude, Llama) are trained on natural language corpora. They excel at prose, reasoning, and open-ended generation. **Financial file formats are the exact opposite**: they are rigid, position-dependent, deterministic structures where a single character in the wrong column breaks an entire payment batch.

This framework rejects the "fine-tune a general LLM" approach in favor of a **domain-native architecture**:

- **Character-level processing** instead of subword/BPE tokenization — because financial files are read by column position, not by semantic meaning.
- **Fixed-width positional awareness** baked into the model via custom positional encodings and field boundary tokens.
- **Dual-objective training** — the model learns to generate *and* validate simultaneously, creating a self-correcting loop.
- **Zero external dependencies** — no API keys, no cloud inference, no data exfiltration. Everything runs on-device.

### The "Source of Truth" Pattern

The central insight is that financial generation has two competing requirements:
1. **Creativity** — producing varied, realistic test data (names, amounts, accounts).
2. **Rigidity** — every byte must conform to a specification (length, type, padding, checksum).

The `SpecificationStore` bridges this gap. It acts as a **Source of Truth** that both the rule-based engine and the neural network consult during generation. The SLM doesn't "guess" what character comes next — it asks the store "what characters are legal at position 47?" and masks its vocabulary accordingly.

---

## 2. System Architecture

### Layer Stack

```
┌─────────────────────────────────────────────────────────────────────────┐
│ LAYER 5: PRESENTATION  (Vanilla JS SPA)                                  │
│ • Drag-and-drop upload, real-time validation console                     │
│ • Generation studio with spec dropdowns and seed controls               │
│ • Dark-theme responsive UI with collapsible record cards                │
├─────────────────────────────────────────────────────────────────────────┤
│ LAYER 4: API GATEWAY  (FastAPI)                                          │
│ • REST endpoints: /validate, /generate, /train, /model/status            │
│ • Lazy-loads model/tokenizer/validator on first request                 │
│ • Serves static frontend files                                            │
├─────────────────────────────────────────────────────────────────────────┤
│ LAYER 3: GENERATION & VALIDATION ENGINE                                   │
│ • FinancialValidator: 4-layer validation (structural → field → semantic →│
│   SLM-based)                                                              │
│ • FinancialGenerator: Rule-based + SLM-guided with constrained decoding │
│ • MockDataSeeder: Valid check digits, proper dates, realistic amounts   │
├─────────────────────────────────────────────────────────────────────────┤
│ LAYER 2: SLM CORE  (PyTorch)                                             │
│ • FinancialSLM: Custom Transformer with causal masking                  │
│ • FinancialTokenizer: Character-level with <RT_X>, <FB>, <RB> tokens    │
│ • FinancialSLMTrainer: Dual-loss training with corruption augmentation  │
├─────────────────────────────────────────────────────────────────────────┤
│ LAYER 1: CONFIG STORE  (In-Memory Source of Truth)                     │
│ • SpecificationStore: Thread-safe singleton with O(1) field lookups    │
│ • FileSpec/RecordSpec/FieldRule: Immutable specification dataclasses    │
│ • Pre-loaded: ACH NACHA, VISA VCF, General Ledger                       │
└─────────────────────────────────────────────────────────────────────────┘
```

### Data Flow Diagrams

#### Validation Flow
```
User Uploads File
       │
       ▼
┌──────────────┐
│  FastAPI     │──► Parse multipart / read text body
│  /validate   │
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│ FinancialValidator│──► 1. Split into lines (records)
│  .validate_file() │──► 2. Detect record type from first char(s)
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  _validate_record │──► 3. Look up RecordSpec from SpecificationStore
│                  │──► 4. Check record length against spec
│                  │──► 5. For each field: extract substring, validate type
│                  │──► 6. Check padding correctness
│                  │──► 7. (Optional) Run SLM validation head
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ _validate_file_level│──► 8. Check record order rules
│                  │──► 9. Verify mandatory records present
│                  │──► 10. Verify checksums (ACH hash totals, etc.)
└──────┬───────────┘
       │
       ▼
   JSON Response  ◄─── 11. Return structured validation report
```

#### Generation Flow
```
User Selects Spec + Parameters
       │
       ▼
┌──────────────┐
│  FastAPI     │──► Parse GenerateRequest (spec_id, num_records, use_slm, seed)
│  /generate   │
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│ FinancialGenerator│──► 1. Load FileSpec from store
│  .generate_file() │──► 2. Generate header record(s) via _generate_record()
└──────┬───────────┘     3. Generate N detail records
       │                  4. Generate trailer/control record(s)
       ▼
┌──────────────────┐
│ _generate_record  │──► IF use_slm=False:
│                  │      • Call MockDataSeeder for each field
│                  │      • Apply padding rules
│                  │      • Build record string via RecordSpec.build_record()
│                  │   IF use_slm=True:
│                  │      • Encode SOS + <RT_X> as prompt
│                  │      • Call model.generate() with FixedWidthConstraint
│                  │      • Constraint masks logits per field type at each position
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ _update_totals    │──► Recompute control record checksums/totals
│                  │    based on generated detail records
└──────┬───────────┘
       │
       ▼
   JSON Response  ◄─── Return content + validation status + checksum validity
```

#### Training Flow
```
POST /train {spec_id, num_samples, epochs}
       │
       ▼
┌──────────────────┐
│ Generate synthetic │──► Use rule-based generator to create valid records
│ training data      │──► 15% corruption rate: randomly mutate valid records
│                  │    (truncate, insert, replace, swap characters)
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ FinancialDataset │──► Encode each record with tokenizer
│                  │──► Pad to max_seq_len
│                  │──► Label: is_valid (1 for clean, 0 for corrupted)
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ FinancialSLMTrainer│──► For each batch:
│  .train_epoch()   │      • Forward pass → generation_logits + validation_logits
│                  │      • Generation loss: CrossEntropy on next-token prediction
│                  │      • Validation loss: CrossEntropy on validity classification
│                  │      • Combined loss = gen_loss + 0.3 * val_loss
│                  │      • Backprop + gradient clipping + AdamW step
└──────┬───────────┘
       │
       ▼
   Save checkpoint ◄─── Every N epochs + best model on validation loss
```

---

## 3. File-by-File Code Flow

### `config/store.py` — The Source of Truth Engine

**Pattern**: Singleton + Strategy + Immutable Dataclasses

**Flow**:
1. `FieldType` enum defines 11 financial data types (NUMERIC, ALPHANUMERIC, ROUTING, DATE, etc.)
2. `PaddingType` enum defines 5 padding strategies (LEFT_ZERO, RIGHT_SPACE, etc.)
3. `FieldRule` dataclass encapsulates a single field:
   - `start_pos` / `end_pos`: Absolute character positions (0-indexed, exclusive end)
   - `validate()`: Type-checks, length-checks, regex-matches, allowed-value-checks
   - `pad()`: Applies the correct padding strategy to fill to exact length
4. `RecordSpec` groups FieldRules into a record type:
   - `validate_record()`: Iterates all fields, collects errors
   - `parse_record()`: Extracts all fields into a dict
   - `build_record()`: Assembles a record from field values with correct padding
5. `FileSpec` groups RecordSpecs into a complete format:
   - `record_order_rules`: Enforces sequence (e.g., ACH: 1→5→6→8→9)
6. `SpecificationStore` (Singleton):
   - `_initialize()`: Creates empty dicts with threading.RLock
   - `register_spec()`: Stores spec + builds O(1) lookup caches:
     - `_record_type_map[spec_id][record_type_code] → record_type_id`
     - `_field_rules_cache[spec_id]["record_type_code:field_name"] → FieldRule`
   - `get_field_rule()`: Direct cache lookup — no iteration, no SQL, no Redis latency
   - Thread-safe via `RLock` (reentrant, allows nested acquires)

**Key Insight**: The cache structure `spec_id:record_type:field_name` enables the generation engine to look up "what type is position 47?" in O(1) time during constrained decoding.

---

### `config/specs.py` — Pre-Built Financial Specifications

**Pattern**: Builder + Factory

**Flow**:
1. `load_ach_spec()` constructs the complete ACH NACHA specification:
   - File Header (type "1", 94 chars, 13 fields)
   - Batch Header (type "5", 94 chars, 13 fields)
   - Entry Detail (type "6", 94 chars, 11 fields)
   - Batch Control (type "8", 94 chars, 11 fields)
   - File Control (type "9", 94 chars, 7 fields)
   - Each field meticulously mapped to NACHA 2024 rules
2. `load_visa_vcf_spec()` constructs VISA VCF:
   - Header ("H", 80 chars), Detail ("D", 80 chars), Trailer ("T", 80 chars)
3. `load_general_ledger_spec()` constructs GL:
   - HDR (120 chars), DET (120 chars), TRL (120 chars) with implied decimal amounts
4. `initialize_all_specs()` calls all three loaders and registers them with the global `spec_store`

**Key Insight**: Every field uses `default_value` for constants (e.g., ACH RecordTypeCode is always "1") and `allowed_values` for restricted enums (e.g., ServiceClassCode ∈ {"220","225","200"}). This makes the specs self-validating.

---

### `slm_core/tokenizer.py` — Domain-Specific Character Tokenizer

**Pattern**: Custom Vocabulary + Position-Aware Encoding

**Flow**:
1. `__init__()` builds vocabulary in strict order:
   - Special tokens: `<PAD>`, `<SOS>`, `<EOS>`, `<UNK>`, `<FB>`, `<RB>`, `<SPACE>`
   - Record type tokens: `<RT_0>` through `<RT_49>`
   - Base characters: digits, uppercase, lowercase, punctuation, whitespace
2. `encode()` converts text → token IDs:
   - Optionally prepends `<SOS>` + `<RT_X>`
   - Truncates/pads to fixed length
3. `encode_fixed_width_record()` is the critical method:
   - Takes a record string + list of `(start, end)` field boundaries
   - Encodes each field's characters, then inserts `<FB>` between fields
   - Ends with `<RB>` (record boundary) + `<EOS>`
   - Result: the model sees explicit structural boundaries in the token stream
4. `decode_fixed_width_record()` reverses the process:
   - Splits on `<FB>` and `<RB>` to reconstruct individual fields
   - Returns `(full_record_string, list_of_field_values)`

**Key Insight**: Unlike BPE tokenizers that merge frequent character pairs, this tokenizer preserves every character as an independent token. Position 47 in the input always corresponds to position 47 in the file — the model learns spatial structure directly.

---

### `slm_core/model.py` — FinancialSLM Transformer

**Pattern**: Custom Transformer + Multi-Task Learning

**Flow**:
1. `PositionalEncoding`:
   - Standard sinusoidal encoding, but with `max_len=5000` to handle long files
   - Added to character embeddings to give the model absolute position awareness
2. `FinancialTransformerBlock`:
   - `nn.MultiheadAttention` with `batch_first=True`
   - Pre-LayerNorm architecture: `norm1(x + attn(x))` then `norm2(x + ff(x))`
   - Causal mask prevents attending to future positions (required for generation)
3. `FinancialSLM.__init__()`:
   - `char_embedding`: Maps vocab → d_model vectors
   - `record_type_embedding`: Adds a learnable bias per record type (conditioning)
   - `transformer_blocks`: Stack of N transformer layers
   - `generation_head`: Linear(d_model → vocab_size) — predicts next character
   - `validation_head`: Linear(d_model → 3) — classifies VALID/INVALID_SYNTAX/INVALID_SEMANTIC
   - `field_boundary_head`: Linear(d_model → 1) — predicts if position is a field boundary
4. `forward()`:
   - Embeds input + adds positional encoding + record type conditioning
   - Applies causal mask and runs through transformer stack
   - Returns dict with requested logits (generation always, validation/boundary optional)
5. `generate()`:
   - Auto-regressive loop: predict one token, append to sequence, repeat
   - Supports `constraint_fn`: at each step, the function receives logits + position and can mask illegal characters
   - Supports temperature scaling and top-k sampling

**Key Insight**: The record type embedding is the conditioning mechanism. When generating an ACH Batch Header, the model receives `<RT_2>` as a prefix token, shifting its internal representations to "Batch Header mode" for the entire sequence.

---

### `slm_core/trainer.py` — Dual-Objective Training

**Pattern**: Multi-Task Loss + Data Augmentation

**Flow**:
1. `FinancialDataset.__getitem__()`:
   - 15% of valid records are randomly corrupted (`_corrupt_record()`):
     - `truncate`: Remove trailing characters → tests length validation
     - `insert`: Add random char → tests format validation
     - `replace`: Swap char with digit → tests type validation
     - `swap`: Transpose adjacent chars → tests positional validation
   - Corrupted records get `is_valid=0` label
2. `FinancialSLMTrainer.__init__()`:
   - `generation_criterion`: CrossEntropyLoss(ignore_index=PAD_ID)
   - `validation_criterion`: CrossEntropyLoss (3-class classification)
   - `validation_weight=0.3`: Balances generation vs validation learning
3. `train_epoch()`:
   - For each batch:
     - Forward pass with `return_validation=True`
     - Generation loss: compare predicted next-char vs actual next-char
     - Validation loss: compare predicted validity vs actual validity
     - `loss = gen_loss + 0.3 * val_loss`
     - Gradient clipping at norm 1.0 (prevents exploding gradients)
     - AdamW optimizer step
4. `validate()`:
   - Computes loss and accuracy on validation set
   - Tracks best model by validation loss

**Key Insight**: By corrupting 15% of training data, the model learns to recognize *what makes a record invalid* — not just memorize valid patterns. This is crucial for the validation head to generalize to real-world malformed files.

---

### `validation/validator.py` — Multi-Layer Validation Engine

**Pattern**: Pipeline + Chain of Responsibility

**Flow**:
1. `validate_file()` orchestrates the pipeline:
   - Splits content by newlines
   - Iterates records, calling `_validate_record()` for each
   - Calls `_validate_file_level()` for cross-record checks
   - Calls `_verify_checksums()` for mathematical integrity
2. `_validate_record()` performs 4 checks:
   - **Structural**: Record length matches spec total_length
   - **Field-level**: Each field's value passes `FieldRule.validate()`
   - **Padding**: Value matches expected padded form
   - **SLM**: If model available, runs neural syntax check
3. `_detect_record_type()`:
   - Tries first character (ACH: "1", "5", "6", "8", "9")
   - Falls back to first 3 characters (GL: "HDR", "DET", "TRL")
4. `_validate_file_level()`:
   - Checks record order against `FileSpec.record_order_rules`
   - Verifies mandatory records are present
5. `_verify_checksums()`:
   - ACH-specific: verifies batch count, entry count, hash total, debit/credit totals
   - Uses integer arithmetic on extracted substrings
6. `_slm_validate()`:
   - Encodes record, runs through model's validation head
   - If prediction ≠ VALID, adds a warning/error result
7. `format_report()`:
   - Produces human-readable ASCII report with per-record breakdown

**Key Insight**: The validator is designed to be **incremental** — you can run structural checks without loading the SLM, or add SLM checks after training. This makes the system usable immediately (rule-based) and progressively smarter (neural-enhanced).

---

### `generation/generator.py` — Constrained Generation Engine

**Pattern**: Strategy + Template Method + Constraint Satisfaction

**Flow**:
1. `FixedWidthConstraint` (implements `ConstraintFunction`):
   - `_build_field_map()`: Maps every character position to its governing `FieldRule`
   - `__call__()`: At generation position N, looks up the field for that position
   - `_get_allowed_chars()`: Returns permitted character set based on `FieldType`:
     - NUMERIC → "0123456789"
     - ALPHABETIC → "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
     - BLANK → " "
     - etc.
   - Masks all disallowed vocabulary positions to `-inf` in the logits
2. `MockDataSeeder`:
   - `generate_field_value()`: Routes to type-specific generator
   - `_gen_routing()`: Generates 9-digit routing number with **valid check digit** using ACH weight algorithm `[3,7,1,3,7,1,3,7]`
   - `_gen_date()`: Random date within last year, formatted `YYMMDD`
   - `_gen_currency()`: Random cents amount, zero-padded to field length
   - `_gen_decimal()`: Amount with 2 implied decimals
3. `FinancialGenerator.generate_file()`:
   - Generates header → N detail records → trailer
   - Each record via `_generate_record()`:
     - Rule-based: `MockDataSeeder` → `FieldRule.pad()` → `RecordSpec.build_record()`
     - SLM-based: `model.generate()` with `FixedWidthConstraint` applied at every step
4. `_update_totals()`:
   - ACH: Recomputes entry hash (sum of receiving DFI routing numbers), debit/credit totals, batch count
   - VCF: Recomputes record count and total amount
   - GL: Recomputes total debits, credits, and net amount

**Key Insight**: The constraint function is the bridge between the "creative" SLM and the "rigid" spec store. At position 29 (ACH Entry Detail Amount field), the SLM wants to sample from 100+ characters — the constraint forces it to choose only from "0123456789". This guarantees syntactic correctness without sacrificing the model's learned distribution over *which* digits to choose.

---

### `api/main.py` — FastAPI Backend

**Pattern**: Dependency Injection + Lazy Loading + RESTful Resources

**Flow**:
1. **Startup**:
   - `initialize_all_specs()` loads ACH, VCF, GL into global `spec_store`
   - Model, tokenizer, validator, generator are `None` initially
2. **Lazy Loading Functions**:
   - `get_model()`: Creates `FinancialSLM` with vocab_size from tokenizer
   - `get_tokenizer()`: Creates `FinancialTokenizer`
   - `get_validator()`: Wires validator to spec_store + model + tokenizer
   - `get_generator()`: Wires generator to spec_store + model + tokenizer
   - All called on first request — no memory used until needed
3. **Endpoints**:
   - `GET /api/specs`: Returns all specs with record type metadata
   - `GET /api/specs/{spec_id}`: Returns full spec JSON
   - `POST /api/validate`: Validates text content, returns structured report
   - `POST /api/validate/upload`: Multipart file upload → validation
   - `POST /api/generate`: Generates file, auto-validates output, returns content
   - `POST /api/train`: Generates synthetic data, trains model, saves checkpoints
   - `GET /api/model/status`: Returns parameter count, vocab size, device
   - `POST /api/model/load`: Loads checkpoint from disk
   - `GET /api/health`: Liveness probe
4. **Static Files**:
   - Mounts `frontend/` directory at root (`/`)
   - Serves `index.html` as SPA fallback

**Key Insight**: The lazy-loading design means you can start the server instantly, browse specs, validate files with rule-based logic, and only load the ~10MB PyTorch model when you actually need generation or SLM validation.

---

### `frontend/index.html` — SPA Shell

**Pattern**: Progressive Disclosure + Tabbed Interface

**Flow**:
1. **Header**: Brand logo + system status indicator (pulsing green dot)
2. **Navigation**: 4 tabs — Validation, Generation, Specifications, Model
3. **Validation View**:
   - Left panel: Spec selector + drag-and-drop zone + paste textarea
   - Right panel: Results area with badge (Ready/Valid/Invalid/Partial) + summary grid + collapsible record cards
4. **Generation View**:
   - Left panel: Spec selector + record count + mode radio (Rule/SLM) + seed input
   - Right panel: Code editor with line numbers + copy/download buttons + generation metadata
5. **Specifications View**:
   - Grid of spec cards with version badges and record type tags
6. **Model View**:
   - Architecture stats cards + training controls (spec, samples, epochs) + training log console

**Key Insight**: The UI uses **progressive disclosure** — record details are collapsed by default and expand on click. This prevents information overload when validating a 1000-record ACH file.

---

### `frontend/styles.css` — Dark Theme Design System

**Pattern**: CSS Custom Properties + Component-Based Styling

**Flow**:
1. `:root` defines 30+ CSS custom properties for a cohesive dark theme:
   - Background hierarchy: `--bg-primary` (#0f172a) → `--bg-secondary` (#1e293b) → `--bg-tertiary` (#334155)
   - Semantic colors: `--accent-blue`, `--accent-green`, `--accent-red`, `--accent-amber`
   - Typography: `--font-sans` (Inter) for UI, `--font-mono` (JetBrains Mono) for data
2. **Layout System**:
   - `validation-workspace`: CSS Grid `380px 1fr` for side-by-side panels
   - `panel`: Flex column with `overflow: hidden` for scrollable content areas
3. **Component Styles**:
   - `drop-zone`: Dashed border with hover state transition
   - `record-card`: Collapsible with `record-card-header` (clickable) + `record-details` (hidden/expanded)
   - `result-item`: Flex row with icon + content + meta
   - `code-editor`: Monospace with line numbers using CSS counters
4. **Animations**:
   - `pulse` on status indicator
   - `fadeIn` on view switching
   - `slideIn` on toast notifications
   - `spin` on loading spinner
5. **Responsive**:
   - `@media (max-width: 1024px)`: Stacks grid columns vertically
   - `@media (max-width: 640px)`: Reduces padding, enables horizontal nav scroll

**Key Insight**: The color system encodes meaning — green for valid, red for errors, amber for warnings, blue for actions. This allows users to scan validation results at a glance without reading text.

---

### `frontend/app.js` — Frontend Logic

**Pattern**: State Machine + Event Delegation + Async/Await

**Flow**:
1. **State Object**:
   - `currentView`: Active tab
   - `specs`: Cached specification list from API
   - `validationResult`: Last validation response
   - `generatedContent`: Last generation response
   - `isLoading`: Global loading flag
2. **Initialization** (`DOMContentLoaded`):
   - `cacheElements()`: Stores all DOM references in `elements` object
   - `bindEvents()`: Attaches listeners (navigation, drag-drop, buttons)
   - `loadSpecs()`: Fetches `/api/specs` and populates all `<select>` dropdowns
   - `loadModelStatus()`: Fetches `/api/model/status` for stats display
3. **Validation Flow**:
   - Drag/drop or file select → `processFile()`:
     - Creates `FormData`, appends file + spec_id
     - POSTs to `/api/validate/upload`
     - Calls `renderValidationResult()`:
       - Updates badge class and text
       - Populates summary grid (records, errors, warnings, checksum)
       - Generates HTML for each record card with collapsible details
       - Color-codes severity (error=red, warning=amber)
4. **Generation Flow**:
   - Click Generate → `generateFile()`:
     - Reads form values (spec, count, mode, seed)
     - POSTs to `/api/generate`
     - Calls `renderGeneratedContent()`:
       - Splits content by newline
       - Wraps each line in `<div class="line">` with line number
     - Shows metadata bar (validation status + checksum)
   - `copyToClipboard()`: Uses Navigator API
   - `downloadGenerated()`: Creates Blob, ObjectURL, triggers download
5. **Training Flow**:
   - Click Train → `startTraining()`:
     - POSTs to `/api/train`
     - Appends log entries to training console in real-time
     - Re-enables button on completion
6. **Utilities**:
   - `apiGet()` / `apiPost()` / `apiPostForm()`: Thin wrappers around fetch with error handling
   - `showToast()`: Creates ephemeral notification div with auto-dismiss
   - `toggleRecordDetails()`: Toggles `expanded` class on record detail divs

**Key Insight**: The frontend never stores the full model — it only exchanges lightweight JSON with the backend. A 1000-record ACH validation returns ~50KB of structured data, not the raw file.

---

### `demo.py` — Interactive Demonstration

**Pattern**: Script-as-Documentation

**Flow**:
1. `main()` prints banner and initializes specs
2. `demo_spec_store()`: Shows registered specs, ACH record types, field rules, O(1) lookup
3. `demo_tokenizer()`: Demonstrates vocab size, fixed-width encoding/decoding
4. `demo_validation()`: Creates a sample ACH file (with intentional errors), validates it, prints full report
5. `demo_generation()`: Generates ACH, VCF, and GL test files
6. `demo_model_training()`: Instantiates a small model (128-dim, 4 layers), shows parameter count, runs forward pass

**Key Insight**: This file serves as both a **smoke test** (verifies all imports work) and a **tutorial** (shows every major capability in sequence).

---

### `tests/test_framework.py` — Unit Tests

**Pattern**: unittest with class-based suites

**Flow**:
1. `TestSpecificationStore`: Verifies specs load, ACH header has 94 chars, field lookup works
2. `TestTokenizer`: Verifies vocab size, encode/decode roundtrip, fixed-width field extraction
3. `TestValidation`: Generates a valid file, validates it, then validates an intentionally invalid string
4. `TestGeneration`: Generates ACH/VCF/GL files and verifies expected record type markers exist
5. `TestMockDataSeeder`: Verifies numeric padding and routing check digit validity
6. `TestModel`: Instantiates small model, verifies forward pass shapes, verifies generation produces output
7. `TestFieldRule`: Tests validation logic and padding for various field types

---

## 4. Interaction Patterns

### How the SLM Consults the Spec Store During Generation

```python
# In generator.py: FinancialGenerator._generate_with_slm()

constraint = FixedWidthConstraint(tokenizer, record_spec)
# constraint._build_field_map() creates:
# {0: FieldRule("RecordTypeCode"), 
#  1: FieldRule("PriorityCode"), ...}

generated = model.generate(
    prompt,
    constraint_fn=lambda logits, pos: constraint(logits, pos, spec, record_spec)
)

# Inside model.generate() loop:
for pos in range(max_length):
    outputs = self.forward(generated, ...)
    next_token_logits = outputs['generation_logits'][0, -1, :]

    # THE BRIDGE: constraint function masks illegal characters
    next_token_logits = constraint_fn(next_token_logits, generated.size(1))

    # Now sampling can only choose valid characters for this position
    probs = F.softmax(next_token_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
```

This is the **critical architectural bridge**: the neural network provides the probability distribution, the specification store provides the legal constraints, and together they produce syntactically guaranteed output.

---

## 5. Extending the Framework

### Adding a New Financial Format

1. Define `FieldRule`s with exact start/end positions
2. Group into `RecordSpec`s with `record_type_code` and `record_type_id`
3. Group into `FileSpec` with `record_order_rules`
4. Call `spec_store.register_spec()`
5. Add checksum logic to `validator.py` `_verify_checksums()`
6. Add total-update logic to `generator.py` `_update_totals()`

### Training on Real Data

1. Parse real files into `{'text': record, 'record_type': id, 'is_valid': 1}` dicts
2. Create `FinancialDataset(records, tokenizer)`
3. Instantiate `FinancialSLMTrainer(model, tokenizer)`
4. Call `trainer.train(dataloader, epochs=N)`
5. Save checkpoint via `trainer.save_checkpoint()`

### Switching to LSTM

Replace:
```python
model = FinancialSLM(vocab_size, ...)
```
With:
```python
model = FinancialLSTM(vocab_size, ...)
```

All other code (tokenizer, trainer, validator, generator) remains unchanged — both classes implement the same interface.

---

## 6. Security & Compliance Notes

| Concern | Mitigation |
|---------|-----------|
| Data exfiltration | No HTTP calls to external APIs. All processing is local. |
| Model poisoning | Training data is synthetic (generated from specs), not user data. |
| File persistence | Uploaded files are processed in-memory; never written to disk. |
| Dependency risk | Only PyTorch + FastAPI + standard library. No ML platform SDKs. |
| Offline operation | `pip install -r requirements.txt` once, then disconnect. |

---

*Document version: 1.0.0*
*Generated for the Financial SLM Framework*
