"""
ChromaDB Schema for FinSLM
Defines the metadata contracts for every collection (analogous to DDL).

Each collection entry has:
  id        string   unique identifier
  document  string   primary searchable text (used for vector similarity)
  metadata  dict     structured filterable fields (replaces SQL columns)

All ChromaDB files are persisted to:  db/chromadb_store/
"""

# ══════════════════════════════════════════════════════════════════════════════
# Collection: ach_odfi_config
# Analogous to: Oracle ACH_ODFI_CONFIG table
# ══════════════════════════════════════════════════════════════════════════════
ODFI_CONFIG_SCHEMA = {
    "description": "ODFI bank configuration for ACH File Header records",
    "id_format":   "odfi_{sequence}",
    "document":    "bank_name  (used for semantic ODFI search)",
    "metadata_fields": {
        "routing_number":   "str  9-digit ABA routing number",
        "bank_name":        "str  Full bank name (max 23 chars)",
        "immediate_dest":   "str  Space + 9-digit routing (NACHA pos 4-13)",
        "immediate_origin": "str  10-char origin field (NACHA pos 14-23)",
        "dest_short_name":  "str  23-char destination name",
        "origin_short_name":"str  23-char origin name",
        "is_active":        "str  Y or N",
        "created_at":       "str  ISO timestamp",
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# Collection: ach_companies
# Analogous to: Oracle ACH_COMPANIES table
# ══════════════════════════════════════════════════════════════════════════════
COMPANIES_SCHEMA = {
    "description": "Originator company master — drives Batch Header records (type 5)",
    "id_format":   "co_{sequence}",
    "document":    "company_name  (semantic search by company name)",
    "metadata_fields": {
        "company_name":        "str  Max 16 chars — NACHA Batch Header pos 5-20",
        "company_id_number":   "str  Max 10 chars — originator company ID",
        "company_entry_desc":  "str  Max 10 chars — e.g. PAYROLL, INSURANCE",
        "sec_code":            "str  PPD|CCD|CTX|WEB|TEL|IAT|...",
        "service_class_code":  "str  200=Mixed 220=Credits 225=Debits",
        "odfi_id":             "str  Reference to ach_odfi_config id",
        "discretionary_data":  "str  Max 20 chars — optional",
        "is_active":           "str  Y or N",
        "created_at":          "str  ISO timestamp",
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# Collection: ach_accounts
# Analogous to: Oracle ACH_ACCOUNTS table
# ══════════════════════════════════════════════════════════════════════════════
ACCOUNTS_SCHEMA = {
    "description": "Receiver (RDFI) account master — drives Entry Detail records (type 6)",
    "id_format":   "acc_{sequence}",
    "document":    "individual_name  (semantic search by payee name)",
    "metadata_fields": {
        "company_id":        "str  Reference to ach_companies id",
        "individual_name":   "str  Max 22 chars — NACHA pos 55-76",
        "individual_id":     "str  Max 15 chars — NACHA pos 40-54",
        "rdfi_routing":      "str  8-digit RDFI routing (no check digit)",
        "rdfi_check_digit":  "str  1 ABA check digit",
        "account_number":    "str  Max 17 chars — NACHA pos 13-29",
        "account_type":      "str  C=Checking S=Savings G=GL L=Loan",
        "prenote_status":    "str  LIVE|PRENOTE|RETURN|FROZEN",
        "is_active":         "str  Y or N",
        "created_at":        "str  ISO timestamp",
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# Collection: ach_transactions
# Analogous to: Oracle ACH_TRANSACTIONS table
# ══════════════════════════════════════════════════════════════════════════════
TRANSACTIONS_SCHEMA = {
    "description": "Pending ACH transactions queued for file generation",
    "id_format":   "txn_{sequence}",
    "document":    "individual_name + amount + transaction_code  (searchable summary)",
    "metadata_fields": {
        "account_id":          "str  Reference to ach_accounts id",
        "company_id":          "str  Reference to ach_companies id",
        "transaction_code":    "str  22|27|32|37|42|47|...",
        "amount_cents":        "int  Amount in integer cents (e.g. 1999 = $19.99)",
        "effective_date":      "str  YYMMDD format",
        "individual_id":       "str  Max 15 chars — overrides account-level ID",
        "individual_name":     "str  Max 22 chars",
        "rdfi_routing":        "str  8-digit RDFI routing",
        "rdfi_check_digit":    "str  1-digit ABA check digit",
        "account_number":      "str  Max 17 chars",
        "account_type":        "str  C|S|G|L",
        "discretionary_data":  "str  2 chars",
        "addenda_info":        "str  Optional — up to 80 chars",
        "company_name":        "str  Denormalised for fast generation",
        "company_id_number":   "str  Denormalised",
        "company_entry_desc":  "str  Denormalised",
        "sec_code":            "str  Denormalised",
        "service_class_code":  "str  Denormalised",
        "odfi_routing":        "str  9-digit ODFI routing (denormalised)",
        "immediate_dest":      "str  Denormalised from ODFI config",
        "immediate_origin":    "str  Denormalised from ODFI config",
        "dest_short_name":     "str  Denormalised",
        "origin_short_name":   "str  Denormalised",
        "status":              "str  PENDING|BATCHED|SENT|RETURNED|SETTLED|VOID",
        "file_id":             "str  Reference to ach_file_log id (set after batching)",
        "batch_number":        "int  Batch number within the file",
        "return_code":         "str  R01-R99 NACHA return code",
        "created_at":          "str  ISO timestamp",
        "processed_at":        "str  ISO timestamp — set when batched",
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# Collection: ach_file_log
# Analogous to: Oracle ACH_FILE_LOG table
# ══════════════════════════════════════════════════════════════════════════════
FILE_LOG_SCHEMA = {
    "description": "Audit trail for every generated ACH NACHA file",
    "id_format":   "file_{timestamp}_{modifier}",
    "document":    "Full NACHA file content (94-char records)  — enables semantic retrieval",
    "metadata_fields": {
        "file_name":           "str  e.g. ACH_20260502_143022_A.ach",
        "file_id_modifier":    "str  A-Z cycling per NACHA spec",
        "odfi_id":             "str  Reference to ach_odfi_config id",
        "batch_count":         "int  Number of batches in file",
        "entry_count":         "int  Total entry+addenda records",
        "block_count":         "int  Total 10-record blocks",
        "total_debit_cents":   "int  Sum of all debit amounts (cents)",
        "total_credit_cents":  "int  Sum of all credit amounts (cents)",
        "generation_method":   "str  CHROMA|SYNTHETIC|SLM|HYBRID",
        "is_valid":            "str  Y or N",
        "validation_errors":   "int  Number of validation errors",
        "sec_codes":           "str  Comma-separated SEC codes in file",
        "created_at":          "str  ISO timestamp",
        "sent_at":             "str  ISO timestamp — when submitted to ODFI",
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# Collection: ach_validation_log
# Analogous to: Oracle ACH_VALIDATION_LOG table
# ══════════════════════════════════════════════════════════════════════════════
VALIDATION_LOG_SCHEMA = {
    "description": "Validation run history — one entry per file validated",
    "id_format":   "val_{timestamp}",
    "document":    "Validation summary text  (e.g. 'ACH file VALID — 3 batches 18 entries')",
    "metadata_fields": {
        "file_id":         "str  Reference to ach_file_log id",
        "file_name":       "str  Uploaded or generated file name",
        "file_type":       "str  ACH or VCF",
        "is_valid":        "str  Y or N",
        "error_count":     "int",
        "warning_count":   "int",
        "report_json":     "str  Full JSON validation report (serialised)",
        "created_at":      "str  ISO timestamp",
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# Collection: ach_training_corpus
# Analogous to: Oracle ACH_TRAINING_CORPUS table
# Purpose: stores ACH file text for SLM training; similarity search finds
#          structurally similar files for curriculum learning
# ══════════════════════════════════════════════════════════════════════════════
TRAINING_CORPUS_SCHEMA = {
    "description": "SLM training corpus — ACH file text with structural metadata",
    "id_format":   "corpus_{timestamp}_{seq}",
    "document":    "Full ACH file content  (embedded for semantic similarity search)",
    "metadata_fields": {
        "file_log_id":          "str  Reference to ach_file_log id",
        "sec_code":             "str  Primary SEC code in file",
        "service_class_code":   "str  200|220|225",
        "batch_count":          "int",
        "entry_count":          "int",
        "split_type":           "str  TRAIN|VAL|TEST",
        "is_used_for_training": "str  Y or N",
        "source":               "str  CHROMA|SYNTHETIC|SLM",
        "created_at":           "str  ISO timestamp",
    },
}

# ── Master registry ───────────────────────────────────────────────────────────
ALL_SCHEMAS = {
    "ach_odfi_config":    ODFI_CONFIG_SCHEMA,
    "ach_companies":      COMPANIES_SCHEMA,
    "ach_accounts":       ACCOUNTS_SCHEMA,
    "ach_transactions":   TRANSACTIONS_SCHEMA,
    "ach_file_log":       FILE_LOG_SCHEMA,
    "ach_validation_log": VALIDATION_LOG_SCHEMA,
    "ach_training_corpus":TRAINING_CORPUS_SCHEMA,
}


def print_schema_summary():
    """Pretty-print all collection schemas — useful for debugging."""
    for name, schema in ALL_SCHEMAS.items():
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"  {schema['description']}")
        print(f"  id format: {schema['id_format']}")
        print(f"  document:  {schema['document']}")
        print(f"  metadata fields ({len(schema['metadata_fields'])}):")
        for field, desc in schema["metadata_fields"].items():
            print(f"    {field:<25} {desc}")


if __name__ == "__main__":
    print_schema_summary()
