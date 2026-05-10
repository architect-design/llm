"""
VISA VCF (Visa Card Format) & General Ledger Specification Definitions.

VISA VCF:
  Line length: 80 characters
  Record type identified by 2-char code at positions 1-2.

General Ledger (ANSI X12 / Journal Entry flat-file convention):
  Line length: 120 characters
  Supports Journal Header, Journal Entry, and Summary/Control records.
"""

from slm.tokenizer import FieldDescriptor, TokenType

# ─────────────────────── Helpers ─────────────────────────────────────────────

def F(name, start, end, ftype, required=True, pattern=None, allowed=None):
    return FieldDescriptor(
        name=name, start=start, end=end,
        field_type=ftype, required=required,
        pattern=pattern, allowed=allowed
    )

N = TokenType.NUMERIC
AN= TokenType.ALPHANUMERIC
BL= TokenType.BLANK_PAD
RT= TokenType.RECORD_TYPE
RO= TokenType.ROUTING
AC= TokenType.ACCOUNT
AM= TokenType.AMOUNT
DT= TokenType.DATE


# ═══════════════════════════════════════════════════════════════════════════
#                         VISA VCF SPECIFICATION
# ═══════════════════════════════════════════════════════════════════════════

# VH — Volume Header
VISA_VH_FIELDS = [
    F("Record Type",            1,  2,  AN, allowed=["VH"]),
    F("File ID",                3,  8,  AN),
    F("Creation Date",          9, 14,  DT),
    F("Creation Time",         15, 18,  N),
    F("Originator ID",         19, 28,  N),
    F("File Sequence Number",  29, 34,  N),
    F("Currency Code",         35, 37,  AN),
    F("Reserved",              38, 80,  BL, required=False),
]

# DT — Detail Transaction Record
VISA_DT_FIELDS = [
    F("Record Type",            1,  2,  AN, allowed=["DT"]),
    F("Transaction Code",       3,  4,  N,
      allowed=["01","02","03","04","05","06","25","26","28","29"]),
    F("BIN Number",             5, 11,  N),
    F("Account Number",        12, 27,  AC),
    F("Transaction Date",      28, 33,  DT),
    F("Transaction Amount",    34, 45,  AM),
    F("Currency Code",         46, 48,  AN),
    F("Merchant ID",           49, 63,  AN),
    F("Merchant Category Code",64, 67,  N),
    F("Merchant Name",         68, 80,  AN),
]

# TR — Trailer / Batch Summary
VISA_TR_FIELDS = [
    F("Record Type",            1,  2,  AN, allowed=["TR"]),
    F("Record Count",           3, 10,  N),
    F("Total Debit Amount",    11, 24,  AM),
    F("Total Credit Amount",   25, 38,  AM),
    F("BIN Number",            39, 45,  N),
    F("Reserved",              46, 80,  BL, required=False),
]

# VF — Volume Footer
VISA_VF_FIELDS = [
    F("Record Type",            1,  2,  AN, allowed=["VF"]),
    F("Total Records",          3, 10,  N),
    F("Total Amount",          11, 26,  AM),
    F("File ID",               27, 32,  AN),
    F("Reserved",              33, 80,  BL, required=False),
]

VISA_FIELD_SCHEMA = {
    "RTVH": VISA_VH_FIELDS,
    "RVDT": VISA_DT_FIELDS,
    "RVTR": VISA_TR_FIELDS,
    "RTVF": VISA_VF_FIELDS,
}

VISA_SPEC_META = {
    "name"         : "VISA_VCF",
    "full_name"    : "VISA Card File Format (VCF)",
    "line_length"  : 80,
    "encoding"     : "ASCII",
    "record_types" : list(VISA_FIELD_SCHEMA.keys()),
    "description"  : (
        "VISA VCF is used for interchange transaction settlement files "
        "between acquiring banks and VISA. 80-character fixed-width records "
        "with 2-char type identifiers."
    ),
    "required_sequence": ["RTVH", "RVDT", "RTVF"],
}


# ═══════════════════════════════════════════════════════════════════════════
#                      GENERAL LEDGER SPECIFICATION
# ═══════════════════════════════════════════════════════════════════════════

# JH — Journal Header (batch-level header for a set of journal entries)
GL_JH_FIELDS = [
    F("Record Type",             1,  2,  AN, allowed=["JH"]),
    F("Journal ID",              3, 12,  AN),
    F("Journal Date",           13, 18,  DT),
    F("Ledger Code",            19, 22,  AN),
    F("Currency Code",          23, 25,  AN),
    F("Period",                 26, 31,  N,  pattern=r"[0-9]{6}"),   # YYYYMM
    F("Batch Reference",        32, 41,  AN),
    F("Description",            42, 81,  AN, required=False),
    F("Created By",             82, 91,  AN, required=False),
    F("Status",                 92, 92,  AN, allowed=["O","P","C"]),  # Open/Posted/Closed
    F("Reserved",               93,120,  BL, required=False),
]

# JE — Journal Entry Line (debit or credit)
GL_JE_FIELDS = [
    F("Record Type",             1,  2,  AN, allowed=["JE"]),
    F("Journal ID",              3, 12,  AN),
    F("Line Number",            13, 17,  N),
    F("Account Code",           18, 29,  AN),
    F("Cost Centre",            30, 39,  AN, required=False),
    F("Project Code",           40, 49,  AN, required=False),
    F("DC Indicator",           50, 50,  AN, allowed=["D","C"]),    # Debit / Credit
    F("Amount",                 51, 66,  AM),
    F("Currency Code",          67, 69,  AN),
    F("Exchange Rate",          70, 79,  N,  required=False),
    F("Description",            80,109,  AN, required=False),
    F("Tax Code",              110,113,  AN, required=False),
    F("Reserved",              114,120,  BL, required=False),
]

# GL — General Ledger Control / Trailer
GL_CT_FIELDS = [
    F("Record Type",             1,  2,  AN, allowed=["GL"]),
    F("Journal ID",              3, 12,  AN),
    F("Total Lines",            13, 18,  N),
    F("Total Debit Amount",     19, 34,  AM),
    F("Total Credit Amount",    35, 50,  AM),
    F("Control Hash",           51, 60,  N),
    F("Posted Date",            61, 66,  DT, required=False),
    F("Reserved",               67,120,  BL, required=False),
]

GL_FIELD_SCHEMA = {
    "RTJH": GL_JH_FIELDS,
    "RTJE": GL_JE_FIELDS,
    "RTGL": GL_CT_FIELDS,
}

GL_SPEC_META = {
    "name"         : "GENERAL_LEDGER",
    "full_name"    : "General Ledger Journal Entry Flat File",
    "line_length"  : 120,
    "encoding"     : "ASCII",
    "record_types" : list(GL_FIELD_SCHEMA.keys()),
    "description"  : (
        "General Ledger flat-file format used for ERP journal entry imports "
        "(SAP, Oracle, NetSuite compatible). Double-entry bookkeeping enforced: "
        "total debits must equal total credits within each journal batch."
    ),
    "required_sequence": ["RTJH", "RTJE", "RTGL"],
    "business_rules": [
        "Total debits must equal total credits per journal (balanced entry)",
        "Account codes must exist in the chart of accounts",
        "Closed periods (Status=C) cannot accept new entries",
    ],
}
