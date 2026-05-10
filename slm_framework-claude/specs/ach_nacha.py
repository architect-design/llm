"""
ACH NACHA Specification — Field Schema Definitions.

Source: NACHA Operating Rules & Guidelines (2024 edition principles).
All field positions are 1-indexed, inclusive, per the official spec.

Record Types:
  1 — File Header
  5 — Batch Header
  6 — Entry Detail
  7 — Addenda
  8 — Batch Control
  9 — File Control
  (padding records are all-9s)

Line length: exactly 94 characters.
Blocking factor: file must be a multiple of 10 lines.
"""

from slm.tokenizer import FieldDescriptor, TokenType

# ─────────────────────── Helper ────────────────────────────────────────────

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


# ─────────────────────── Record Schemas ──────────────────────────────────────

# Record Type 1 — File Header
RT1_FIELDS = [
    F("Record Type Code",            1,  1,  RT, allowed=["1"]),
    F("Priority Code",               2,  3,  N,  allowed=["01"]),
    F("Immediate Destination",       4, 13,  RO, pattern=r"[ 0-9]{10}"),
    F("Immediate Origin",           14, 23,  N),
    F("File Creation Date",         24, 29,  DT),
    F("File Creation Time",         30, 33,  N,  required=False),
    F("File ID Modifier",           34, 34,  AN, pattern=r"[A-Z0-9]"),
    F("Record Size",                35, 37,  N,  allowed=["094"]),
    F("Blocking Factor",            38, 39,  N,  allowed=["10"]),
    F("Format Code",                40, 40,  N,  allowed=["1"]),
    F("Immediate Destination Name", 41, 63,  AN),
    F("Immediate Origin Name",      64, 86,  AN),
    F("Reference Code",             87, 94,  AN, required=False),
]

# Record Type 5 — Batch Header
RT5_FIELDS = [
    F("Record Type Code",           1,  1,  RT, allowed=["5"]),
    F("Service Class Code",         2,  4,  N,  allowed=["200","220","225"]),
    F("Company Name",               5, 20,  AN),
    F("Company Discretionary Data",21, 40,  AN, required=False),
    F("Company Identification",    41, 50,  AN),
    F("Standard Entry Class Code", 51, 53,  AN,
      allowed=["PPD","CCD","CTX","WEB","TEL","POP","RCK","ARC","BOC","MTE","SHR","ACK","ATX","CIE","DNE","ENR","TRX"]),
    F("Company Entry Description", 54, 63,  AN),
    F("Company Descriptive Date",  64, 69,  AN, required=False),
    F("Effective Entry Date",      70, 75,  DT),
    F("Settlement Date",           76, 78,  N,  required=False),
    F("Originator Status Code",    79, 79,  AN, allowed=["1","2"]),
    F("ODFI Identification",       80, 87,  RO, pattern=r"[0-9]{8}"),
    F("Batch Number",              88, 94,  N),
]

# Record Type 6 — Entry Detail
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

# Record Type 7 — Addenda
RT7_FIELDS = [
    F("Record Type Code",          1,  1,  RT, allowed=["7"]),
    F("Addenda Type Code",         2,  3,  N,  allowed=["02","05","98","99"]),
    F("Payment Related Info",      4, 83,  AN, required=False),
    F("Sequence Number",          84, 87,  N),
    F("Entry Detail Sequence",    88, 94,  N),
]

# Record Type 8 — Batch Control
RT8_FIELDS = [
    F("Record Type Code",          1,  1,  RT, allowed=["8"]),
    F("Service Class Code",        2,  4,  N,  allowed=["200","220","225"]),
    F("Entry/Addenda Count",       5, 10,  N),
    F("Entry Hash",               11, 20,  N),
    F("Total Debit Dollar Amount",21, 32,  AM),
    F("Total Credit Dollar Amount",33,44,  AM),
    F("Company Identification",   45, 54,  AN),
    F("Message Auth Code",        55, 73,  AN, required=False),
    F("Reserved",                 74, 79,  BL),
    F("ODFI Identification",      80, 87,  RO, pattern=r"[0-9]{8}"),
    F("Batch Number",             88, 94,  N),
]

# Record Type 9 — File Control
RT9_FIELDS = [
    F("Record Type Code",          1,  1,  RT, allowed=["9"]),
    F("Batch Count",               2,  7,  N),
    F("Block Count",               8, 13,  N),
    F("Entry/Addenda Count",      14, 21,  N),
    F("Entry Hash",               22, 31,  N),
    F("Total Debit Dollar Amount",32, 43,  AM),
    F("Total Credit Dollar Amount",44,55,  AM),
    F("Reserved",                 56, 94,  BL),
]


# ─────────────────────── Master Schema Dict ───────────────────────────────────

ACH_FIELD_SCHEMA = {
    "RT1": RT1_FIELDS,
    "RT5": RT5_FIELDS,
    "RT6": RT6_FIELDS,
    "RT7": RT7_FIELDS,
    "RT8": RT8_FIELDS,
    "RT9": RT9_FIELDS,
}

# Spec-level metadata
ACH_SPEC_META = {
    "name"           : "ACH_NACHA",
    "full_name"      : "ACH NACHA File Format",
    "line_length"    : 94,
    "blocking_factor": 10,
    "encoding"       : "ASCII",
    "record_types"   : list(ACH_FIELD_SCHEMA.keys()),
    "description"    : (
        "The NACHA (National Automated Clearing House Association) file format "
        "governs electronic funds transfer in the US banking system. Each file "
        "is exactly 94 characters wide with a 10-line blocking factor."
    ),
    "required_sequence": ["RT1", "RT5", "RT6", "RT8", "RT9"],
}
