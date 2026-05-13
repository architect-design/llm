"""
financial_slm_framework/config/specs.py
Pre-built specifications for ACH NACHA, VISA VCF, and General Ledger formats.
Loaded into the SpecificationStore on startup.
"""

from .store import (
    FileSpec, RecordSpec, FieldRule,
    FieldType, PaddingType, spec_store
)


def load_ach_spec():
    """Load ACH NACHA specification into the store."""
    ach = FileSpec(
        spec_id="ach_nacha",
        name="ACH NACHA",
        description="Automated Clearing House NACHA file format",
        version="2024",
        record_order_rules=["1", "5", "6", "7", "8", "9"]
    )

    # File Header Record (Type Code 1)
    file_header = RecordSpec(
        record_type_code="1",
        record_type_id=1,
        name="File Header Record",
        description="Identifies the originator and file creation details",
        total_length=94
    )
    file_header.fields = [
        FieldRule("RecordTypeCode", 0, 1, FieldType.NUMERIC, 1, default_value="1"),
        FieldRule("PriorityCode", 1, 3, FieldType.NUMERIC, 2, default_value="01", padding=PaddingType.LEFT_ZERO),
        FieldRule("ImmediateDestination", 3, 13, FieldType.NUMERIC, 10, padding=PaddingType.LEFT_ZERO, description="10-digit routing number prefixed with blank"),
        FieldRule("ImmediateOrigin", 13, 23, FieldType.NUMERIC, 10, padding=PaddingType.LEFT_ZERO, description="10-digit routing number or company ID"),
        FieldRule("FileCreationDate", 23, 29, FieldType.DATE, 6, description="YYMMDD"),
        FieldRule("FileCreationTime", 29, 33, FieldType.TIME, 4, description="HHMM"),
        FieldRule("FileIDModifier", 33, 34, FieldType.ALPHANUMERIC, 1, default_value="A"),
        FieldRule("RecordSize", 34, 37, FieldType.NUMERIC, 3, default_value="094"),
        FieldRule("BlockingFactor", 37, 39, FieldType.NUMERIC, 2, default_value="10"),
        FieldRule("FormatCode", 39, 40, FieldType.NUMERIC, 1, default_value="1"),
        FieldRule("ImmediateDestinationName", 40, 63, FieldType.ALPHANUMERIC, 23, padding=PaddingType.RIGHT_SPACE),
        FieldRule("ImmediateOriginName", 63, 86, FieldType.ALPHANUMERIC, 23, padding=PaddingType.RIGHT_SPACE),
        FieldRule("ReferenceCode", 86, 94, FieldType.ALPHANUMERIC, 8, padding=PaddingType.RIGHT_SPACE, required=False),
    ]
    ach.add_record_spec(file_header)

    # Batch Header Record (Type Code 5)
    batch_header = RecordSpec(
        record_type_code="5",
        record_type_id=2,
        name="Batch Header Record",
        description="Identifies a batch of entries",
        total_length=94
    )
    batch_header.fields = [
        FieldRule("RecordTypeCode", 0, 1, FieldType.NUMERIC, 1, default_value="5"),
        FieldRule("ServiceClassCode", 1, 4, FieldType.NUMERIC, 3, allowed_values=["220", "225", "200"]),
        FieldRule("CompanyName", 4, 20, FieldType.ALPHANUMERIC, 16, padding=PaddingType.RIGHT_SPACE),
        FieldRule("CompanyDiscretionaryData", 20, 40, FieldType.ALPHANUMERIC, 20, padding=PaddingType.RIGHT_SPACE, required=False),
        FieldRule("CompanyIdentification", 40, 50, FieldType.ALPHANUMERIC, 10),
        FieldRule("CompanyEntryDescription", 50, 53, FieldType.ALPHANUMERIC, 3),
        FieldRule("CompanyDescriptiveDate", 53, 56, FieldType.ALPHANUMERIC, 3, required=False),
        FieldRule("EffectiveEntryDate", 56, 62, FieldType.DATE, 6),
        FieldRule("SettlementDate", 62, 65, FieldType.NUMERIC, 3, default_value="   ", required=False),
        FieldRule("OriginatorStatusCode", 65, 66, FieldType.NUMERIC, 1, allowed_values=["1", "2"]),
        FieldRule("OriginatingDFI", 66, 75, FieldType.NUMERIC, 9, description="Routing number without check digit"),
        FieldRule("BatchNumber", 75, 94, FieldType.NUMERIC, 19, padding=PaddingType.LEFT_ZERO),
    ]
    ach.add_record_spec(batch_header)

    # Entry Detail Record (Type Code 6)
    entry_detail = RecordSpec(
        record_type_code="6",
        record_type_id=3,
        name="Entry Detail Record",
        description="Individual transaction entry",
        total_length=94
    )
    entry_detail.fields = [
        FieldRule("RecordTypeCode", 0, 1, FieldType.NUMERIC, 1, default_value="6"),
        FieldRule("TransactionCode", 1, 3, FieldType.NUMERIC, 2, allowed_values=["22", "23", "24", "27", "28", "29", "32", "33", "34", "37", "38", "39"]),
        FieldRule("ReceivingDFI", 3, 12, FieldType.NUMERIC, 9, description="Routing number without check digit"),
        FieldRule("CheckDigit", 12, 13, FieldType.NUMERIC, 1),
        FieldRule("DFIAccountNumber", 13, 29, FieldType.ALPHANUMERIC, 17, padding=PaddingType.RIGHT_SPACE),
        FieldRule("Amount", 29, 39, FieldType.CURRENCY, 10, padding=PaddingType.LEFT_ZERO, description="Amount in cents"),
        FieldRule("IdentificationNumber", 39, 54, FieldType.ALPHANUMERIC, 15, padding=PaddingType.RIGHT_SPACE, required=False),
        FieldRule("ReceivingCompanyName", 54, 76, FieldType.ALPHANUMERIC, 22, padding=PaddingType.RIGHT_SPACE),
        FieldRule("DiscretionaryData", 76, 78, FieldType.ALPHANUMERIC, 2, padding=PaddingType.RIGHT_SPACE, required=False),
        FieldRule("AddendaRecordIndicator", 78, 79, FieldType.NUMERIC, 1, allowed_values=["0", "1"]),
        FieldRule("TraceNumber", 79, 94, FieldType.NUMERIC, 15, padding=PaddingType.LEFT_ZERO),
    ]
    ach.add_record_spec(entry_detail)

    # Batch Control Record (Type Code 8)
    batch_control = RecordSpec(
        record_type_code="8",
        record_type_id=4,
        name="Batch Control Record",
        description="Batch totals and hash",
        total_length=94
    )
    batch_control.fields = [
        FieldRule("RecordTypeCode", 0, 1, FieldType.NUMERIC, 1, default_value="8"),
        FieldRule("ServiceClassCode", 1, 4, FieldType.NUMERIC, 3),
        FieldRule("EntryAddendaCount", 4, 10, FieldType.NUMERIC, 6, padding=PaddingType.LEFT_ZERO),
        FieldRule("EntryHash", 10, 20, FieldType.NUMERIC, 10, padding=PaddingType.LEFT_ZERO),
        FieldRule("TotalDebitEntryDollarAmount", 20, 32, FieldType.CURRENCY, 12, padding=PaddingType.LEFT_ZERO),
        FieldRule("TotalCreditEntryDollarAmount", 32, 44, FieldType.CURRENCY, 12, padding=PaddingType.LEFT_ZERO),
        FieldRule("CompanyIdentification", 44, 54, FieldType.ALPHANUMERIC, 10),
        FieldRule("MessageAuthenticationCode", 54, 73, FieldType.ALPHANUMERIC, 19, padding=PaddingType.RIGHT_SPACE, required=False),
        FieldRule("Reserved", 73, 79, FieldType.BLANK, 6, padding=PaddingType.RIGHT_SPACE),
        FieldRule("OriginatingDFI", 79, 88, FieldType.NUMERIC, 9),
        FieldRule("BatchNumber", 88, 94, FieldType.NUMERIC, 6, padding=PaddingType.LEFT_ZERO),
    ]
    ach.add_record_spec(batch_control)

    # File Control Record (Type Code 9)
    file_control = RecordSpec(
        record_type_code="9",
        record_type_id=5,
        name="File Control Record",
        description="File-level totals",
        total_length=94
    )
    file_control.fields = [
        FieldRule("RecordTypeCode", 0, 1, FieldType.NUMERIC, 1, default_value="9"),
        FieldRule("BatchCount", 1, 7, FieldType.NUMERIC, 6, padding=PaddingType.LEFT_ZERO),
        FieldRule("BlockCount", 7, 13, FieldType.NUMERIC, 6, padding=PaddingType.LEFT_ZERO),
        FieldRule("EntryAddendaCount", 13, 21, FieldType.NUMERIC, 8, padding=PaddingType.LEFT_ZERO),
        FieldRule("EntryHash", 21, 31, FieldType.NUMERIC, 10, padding=PaddingType.LEFT_ZERO),
        FieldRule("TotalDebitEntryDollarAmount", 31, 43, FieldType.CURRENCY, 12, padding=PaddingType.LEFT_ZERO),
        FieldRule("TotalCreditEntryDollarAmount", 43, 55, FieldType.CURRENCY, 12, padding=PaddingType.LEFT_ZERO),
        FieldRule("Reserved", 55, 94, FieldType.BLANK, 39, padding=PaddingType.RIGHT_SPACE),
    ]
    ach.add_record_spec(file_control)

    spec_store.register_spec(ach)
    return ach


def load_visa_vcf_spec():
    """Load VISA VCF (Value Clearing Format) specification."""
    vcf = FileSpec(
        spec_id="visa_vcf",
        name="VISA VCF",
        description="VISA Value Clearing Format for interchange",
        version="2024",
        record_order_rules=["H", "D", "T"]
    )

    # Header Record
    header = RecordSpec(
        record_type_code="H",
        record_type_id=10,
        name="Header Record",
        description="VCF file header",
        total_length=80
    )
    header.fields = [
        FieldRule("RecordType", 0, 1, FieldType.ALPHABETIC, 1, default_value="H"),
        FieldRule("FileDate", 1, 9, FieldType.NUMERIC, 8, description="YYYYMMDD"),
        FieldRule("FileTime", 9, 15, FieldType.NUMERIC, 6, description="HHMMSS"),
        FieldRule("SenderID", 15, 25, FieldType.ALPHANUMERIC, 10, padding=PaddingType.RIGHT_SPACE),
        FieldRule("ReceiverID", 25, 35, FieldType.ALPHANUMERIC, 10, padding=PaddingType.RIGHT_SPACE),
        FieldRule("FileSequence", 35, 41, FieldType.NUMERIC, 6, padding=PaddingType.LEFT_ZERO),
        FieldRule("FileVersion", 41, 45, FieldType.ALPHANUMERIC, 4, default_value="0100"),
        FieldRule("Reserved", 45, 80, FieldType.BLANK, 35, padding=PaddingType.RIGHT_SPACE),
    ]
    vcf.add_record_spec(header)

    # Detail Record
    detail = RecordSpec(
        record_type_code="D",
        record_type_id=11,
        name="Detail Record",
        description="Transaction detail",
        total_length=80
    )
    detail.fields = [
        FieldRule("RecordType", 0, 1, FieldType.ALPHABETIC, 1, default_value="D"),
        FieldRule("TransactionDate", 1, 9, FieldType.NUMERIC, 8, description="YYYYMMDD"),
        FieldRule("TransactionTime", 9, 15, FieldType.NUMERIC, 6, description="HHMMSS"),
        FieldRule("CardNumber", 15, 31, FieldType.NUMERIC, 16, padding=PaddingType.LEFT_ZERO),
        FieldRule("TransactionAmount", 31, 43, FieldType.CURRENCY, 12, padding=PaddingType.LEFT_ZERO),
        FieldRule("MerchantID", 43, 53, FieldType.ALPHANUMERIC, 10, padding=PaddingType.RIGHT_SPACE),
        FieldRule("TransactionType", 53, 55, FieldType.NUMERIC, 2, allowed_values=["01", "02", "03", "04", "05"]),
        FieldRule("AuthorizationCode", 55, 61, FieldType.ALPHANUMERIC, 6, padding=PaddingType.RIGHT_SPACE),
        FieldRule("ResponseCode", 61, 63, FieldType.ALPHANUMERIC, 2, padding=PaddingType.RIGHT_SPACE),
        FieldRule("Reserved", 63, 80, FieldType.BLANK, 17, padding=PaddingType.RIGHT_SPACE),
    ]
    vcf.add_record_spec(detail)

    # Trailer Record
    trailer = RecordSpec(
        record_type_code="T",
        record_type_id=12,
        name="Trailer Record",
        description="File totals",
        total_length=80
    )
    trailer.fields = [
        FieldRule("RecordType", 0, 1, FieldType.ALPHABETIC, 1, default_value="T"),
        FieldRule("RecordCount", 1, 9, FieldType.NUMERIC, 8, padding=PaddingType.LEFT_ZERO),
        FieldRule("TotalAmount", 9, 21, FieldType.CURRENCY, 12, padding=PaddingType.LEFT_ZERO),
        FieldRule("HashTotal", 21, 37, FieldType.NUMERIC, 16, padding=PaddingType.LEFT_ZERO),
        FieldRule("Reserved", 37, 80, FieldType.BLANK, 43, padding=PaddingType.RIGHT_SPACE),
    ]
    vcf.add_record_spec(trailer)

    spec_store.register_spec(vcf)
    return vcf


def load_general_ledger_spec():
    """Load General Ledger specification."""
    gl = FileSpec(
        spec_id="general_ledger",
        name="General Ledger",
        description="Standard General Ledger entry format",
        version="2024",
        record_order_rules=["HDR", "DET", "TRL"]
    )

    # GL Header
    gl_header = RecordSpec(
        record_type_code="HDR",
        record_type_id=20,
        name="GL Header",
        description="General Ledger file header",
        total_length=120
    )
    gl_header.fields = [
        FieldRule("RecordType", 0, 3, FieldType.ALPHABETIC, 3, default_value="HDR"),
        FieldRule("CompanyCode", 3, 8, FieldType.ALPHANUMERIC, 5, padding=PaddingType.RIGHT_SPACE),
        FieldRule("FiscalYear", 8, 12, FieldType.NUMERIC, 4),
        FieldRule("Period", 12, 14, FieldType.NUMERIC, 2, padding=PaddingType.LEFT_ZERO),
        FieldRule("CurrencyCode", 14, 17, FieldType.ALPHABETIC, 3, default_value="USD"),
        FieldRule("PostingDate", 17, 27, FieldType.ALPHANUMERIC, 10, description="YYYY-MM-DD"),
        FieldRule("SourceSystem", 27, 37, FieldType.ALPHANUMERIC, 10, padding=PaddingType.RIGHT_SPACE),
        FieldRule("Reserved", 37, 120, FieldType.BLANK, 83, padding=PaddingType.RIGHT_SPACE),
    ]
    gl.add_record_spec(gl_header)

    # GL Detail
    gl_detail = RecordSpec(
        record_type_code="DET",
        record_type_id=21,
        name="GL Detail",
        description="General Ledger transaction detail",
        total_length=120
    )
    gl_detail.fields = [
        FieldRule("RecordType", 0, 3, FieldType.ALPHABETIC, 3, default_value="DET"),
        FieldRule("AccountNumber", 3, 18, FieldType.ALPHANUMERIC, 15, padding=PaddingType.RIGHT_SPACE),
        FieldRule("CostCenter", 18, 24, FieldType.ALPHANUMERIC, 6, padding=PaddingType.RIGHT_SPACE),
        FieldRule("DebitAmount", 24, 39, FieldType.DECIMAL, 15, padding=PaddingType.LEFT_ZERO, description="2 implied decimals"),
        FieldRule("CreditAmount", 39, 54, FieldType.DECIMAL, 15, padding=PaddingType.LEFT_ZERO, description="2 implied decimals"),
        FieldRule("TransactionDate", 54, 64, FieldType.ALPHANUMERIC, 10, description="YYYY-MM-DD"),
        FieldRule("JournalReference", 64, 79, FieldType.ALPHANUMERIC, 15, padding=PaddingType.RIGHT_SPACE),
        FieldRule("Description", 79, 109, FieldType.ALPHANUMERIC, 30, padding=PaddingType.RIGHT_SPACE),
        FieldRule("UserID", 109, 119, FieldType.ALPHANUMERIC, 10, padding=PaddingType.RIGHT_SPACE),
        FieldRule("Status", 119, 120, FieldType.ALPHANUMERIC, 1, allowed_values=["P", "U"]),
    ]
    gl.add_record_spec(gl_detail)

    # GL Trailer
    gl_trailer = RecordSpec(
        record_type_code="TRL",
        record_type_id=22,
        name="GL Trailer",
        description="General Ledger file trailer",
        total_length=120
    )
    gl_trailer.fields = [
        FieldRule("RecordType", 0, 3, FieldType.ALPHABETIC, 3, default_value="TRL"),
        FieldRule("TotalRecords", 3, 12, FieldType.NUMERIC, 9, padding=PaddingType.LEFT_ZERO),
        FieldRule("TotalDebits", 12, 27, FieldType.DECIMAL, 15, padding=PaddingType.LEFT_ZERO),
        FieldRule("TotalCredits", 27, 42, FieldType.DECIMAL, 15, padding=PaddingType.LEFT_ZERO),
        FieldRule("NetAmount", 42, 57, FieldType.DECIMAL, 15, padding=PaddingType.LEFT_ZERO),
        FieldRule("Reserved", 57, 120, FieldType.BLANK, 63, padding=PaddingType.RIGHT_SPACE),
    ]
    gl.add_record_spec(gl_trailer)

    spec_store.register_spec(gl)
    return gl


def initialize_all_specs():
    """Initialize all built-in specifications."""
    load_ach_spec()
    load_visa_vcf_spec()
    load_general_ledger_spec()
    print("[Config] All specifications loaded into Source of Truth store.")
