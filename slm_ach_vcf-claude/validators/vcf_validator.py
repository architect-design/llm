"""
VISA VCF (VisaNet Custom File) Validator
Validates VISA transaction files per VisaNet specifications.
Supports: TC05 (Auth), TC06 (Financial), TC15 (Reversal), TC25 (Chargeback)
Also validates ISO 8583 inspired flat-file VCF format.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


class Severity(Enum):
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass
class VCFValidationResult:
    severity: Severity
    field: str
    record_type: str
    line_number: int
    message: str
    value: Optional[str] = None
    expected: Optional[str] = None


@dataclass
class VCFValidationReport:
    is_valid: bool
    errors: List[VCFValidationResult] = field(default_factory=list)
    warnings: List[VCFValidationResult] = field(default_factory=list)
    info: List[VCFValidationResult] = field(default_factory=list)
    statistics: Dict = field(default_factory=dict)

    def add(self, result: VCFValidationResult):
        if result.severity == Severity.ERROR:
            self.errors.append(result)
        elif result.severity == Severity.WARNING:
            self.warnings.append(result)
        else:
            self.info.append(result)

    def to_dict(self):
        return {
            "is_valid": self.is_valid,
            "summary": {
                "errors": len(self.errors),
                "warnings": len(self.warnings),
                "info": len(self.info),
            },
            "errors": [self._r(r) for r in self.errors],
            "warnings": [self._r(r) for r in self.warnings],
            "info": [self._r(r) for r in self.info],
            "statistics": self.statistics,
        }

    def _r(self, r):
        return {
            "severity": r.severity.value,
            "field": r.field,
            "record_type": r.record_type,
            "line": r.line_number,
            "message": r.message,
            "value": r.value,
            "expected": r.expected,
        }


# ─── VISA VCF Specifications ──────────────────────────────────────────────────

VALID_TRANSACTION_CODES = {
    "05": "Authorization Request",
    "06": "Financial Transaction",
    "10": "Full Reversal",
    "12": "Partial Reversal",
    "15": "Reversal Advice",
    "20": "Acquirer Reconciliation",
    "25": "Chargeback",
    "26": "Second Chargeback",
    "30": "Issuer Reconciliation",
    "40": "Fee Collection",
    "42": "Fee Collection Response",
    "92": "Network Management",
    "96": "Administrative",
}

VALID_CURRENCY_CODES = {
    "840": "USD", "978": "EUR", "826": "GBP", "392": "JPY",
    "124": "CAD", "036": "AUD", "756": "CHF", "356": "INR",
    "986": "BRL", "156": "CNY", "484": "MXN", "344": "HKD",
    "702": "SGD", "208": "DKK", "752": "SEK", "578": "NOK",
    "554": "NZD", "458": "MYR", "704": "VND", "764": "THB",
}

VALID_PROCESSING_CODES = {
    "000000": "Purchase",
    "200000": "Credit/Refund",
    "010000": "Withdrawal",
    "190000": "Deposit",
    "090000": "Balance Inquiry",
    "400000": "Transfer From",
    "500000": "Transfer To",
}

VALID_POS_ENTRY_MODES = {
    "00": "Unknown",
    "01": "Manual (no terminal)",
    "02": "Magnetic stripe (no CVV)",
    "05": "Chip (ICC)",
    "07": "Contactless chip",
    "10": "Credential on file",
    "90": "Magnetic stripe",
    "91": "Contactless magnetic stripe",
}

VALID_RESPONSE_CODES = {
    "00": "Approved",
    "01": "Refer to issuer",
    "03": "Invalid merchant",
    "04": "Pick up card",
    "05": "Do not honor",
    "06": "Error",
    "07": "Pick up card (special)",
    "08": "Honor with ID",
    "10": "Partial approval",
    "11": "VIP approval",
    "12": "Invalid transaction",
    "13": "Invalid amount",
    "14": "Invalid card number",
    "15": "No such issuer",
    "19": "Re-enter transaction",
    "25": "Unable to locate record",
    "30": "Format error",
    "51": "Insufficient funds",
    "54": "Expired card",
    "55": "Incorrect PIN",
    "57": "Transaction not permitted",
    "58": "Transaction not permitted to terminal",
    "62": "Restricted card",
    "65": "Activity limit exceeded",
    "75": "PIN entry attempts exceeded",
    "76": "Ineligible account",
    "78": "Blocked",
    "91": "Issuer unavailable",
    "92": "Unable to route",
    "96": "System malfunction",
}

RECORD_LENGTHS = {
    "header": 128,
    "transaction": 256,
    "trailer": 128,
}


class VCFValidator:
    """
    VISA VCF File Validator
    Validates flat-file VCF format used in VisaNet batch processing
    """

    def validate(self, content: str) -> VCFValidationReport:
        report = VCFValidationReport(is_valid=True)
        lines = [l for l in content.splitlines() if l.strip()]

        if not lines:
            report.add(VCFValidationResult(Severity.ERROR, "file", "FILE", 0, "Empty file"))
            report.is_valid = False
            return report

        # Detect format: delimiter-based (CSV-like) or fixed-width
        is_delimited = '|' in lines[0] or ',' in lines[0] or '\t' in lines[0]

        if is_delimited:
            self._validate_delimited(lines, report)
        else:
            self._validate_fixed_width(lines, report)

        report.statistics = self._build_stats(lines, is_delimited)
        report.is_valid = len(report.errors) == 0
        return report

    def _validate_delimited(self, lines: List[str], report: VCFValidationReport):
        """Validate pipe-delimited VCF format"""
        delimiter = '|' if '|' in lines[0] else (',' if ',' in lines[0] else '\t')

        # Check header record
        header = lines[0].split(delimiter)
        self._validate_vcf_header_delimited(header, 1, report)

        # Validate transactions
        transaction_count = 0
        total_amount = 0

        for i, line in enumerate(lines[1:-1], 2):
            if not line.strip():
                continue
            fields = line.split(delimiter)

            if len(fields) < 20:
                report.add(VCFValidationResult(Severity.ERROR, "field_count", "Transaction", i,
                                               f"Transaction record has {len(fields)} fields, expected ≥20",
                                               str(len(fields)), "≥20"))
                continue

            self._validate_transaction_record(fields, i, report)
            transaction_count += 1

            try:
                total_amount += int(fields[5].replace('.', '').replace(',', ''))
            except (ValueError, IndexError):
                pass

        # Validate trailer
        if len(lines) > 1:
            trailer = lines[-1].split(delimiter)
            self._validate_vcf_trailer_delimited(trailer, len(lines), transaction_count, report)

    def _validate_fixed_width(self, lines: List[str], report: VCFValidationReport):
        """Validate fixed-width VCF format"""
        for i, line in enumerate(lines, 1):
            if len(line) < 10:
                continue
            rec_type = line[0:2]

            if rec_type == "00":
                self._validate_fixed_header(line, i, report)
            elif rec_type in VALID_TRANSACTION_CODES:
                self._validate_fixed_transaction(line, i, rec_type, report)
            elif rec_type == "99":
                self._validate_fixed_trailer(line, i, report)
            else:
                report.add(VCFValidationResult(Severity.WARNING, "record_type", "UNKNOWN", i,
                                               f"Unrecognized record type '{rec_type}'", rec_type))

    def _validate_vcf_header_delimited(self, fields: List[str], ln: int, report: VCFValidationReport):
        if len(fields) < 6:
            report.add(VCFValidationResult(Severity.ERROR, "header_fields", "Header", ln,
                                           "Header record has insufficient fields"))
            return

        # File type identifier
        file_type = fields[0].strip().upper()
        if file_type not in ("VCF", "VISA", "VCF_BATCH", "VISANET"):
            report.add(VCFValidationResult(Severity.WARNING, "file_type", "Header", ln,
                                           f"Unexpected file type identifier '{file_type}'",
                                           file_type, "VCF/VISA"))

        # Version
        version = fields[1].strip() if len(fields) > 1 else ""
        if not re.match(r'^\d+\.\d+', version):
            report.add(VCFValidationResult(Severity.INFO, "version", "Header", ln,
                                           f"File version '{version}' format may be non-standard", version))

        # Acquirer BIN (6 digits)
        if len(fields) > 2:
            acq_bin = fields[2].strip()
            if acq_bin and not re.match(r'^\d{6}$', acq_bin):
                report.add(VCFValidationResult(Severity.ERROR, "acquirer_bin", "Header", ln,
                                               "Acquirer BIN must be exactly 6 digits", acq_bin, "6 digits"))

        # File creation date
        if len(fields) > 3:
            date = fields[3].strip()
            if not re.match(r'^\d{4}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])$', date):
                report.add(VCFValidationResult(Severity.ERROR, "creation_date", "Header", ln,
                                               "Creation date must be YYYYMMDD format", date, "YYYYMMDD"))

    def _validate_transaction_record(self, fields: List[str], ln: int, report: VCFValidationReport):
        """Validate individual transaction record"""
        # Field 0: Transaction Code
        tc = fields[0].strip() if fields else ""
        if tc not in VALID_TRANSACTION_CODES:
            report.add(VCFValidationResult(Severity.ERROR, "transaction_code", "Transaction", ln,
                                           f"Invalid transaction code '{tc}'", tc,
                                           str(list(VALID_TRANSACTION_CODES.keys()))))

        # Field 1: Card Number (PAN) - 13-19 digits, may be masked
        pan = fields[1].strip() if len(fields) > 1 else ""
        pan_clean = pan.replace('*', '').replace('X', '')
        if pan_clean and not re.match(r'^\d{6,19}$', pan_clean):
            report.add(VCFValidationResult(Severity.ERROR, "pan", "Transaction", ln,
                                           "PAN must be 13-19 digits", pan))
        if pan and not ('*' in pan or 'X' in pan.upper()):
            if not self._luhn_check(pan):
                report.add(VCFValidationResult(Severity.ERROR, "pan_luhn", "Transaction", ln,
                                               "PAN failed Luhn algorithm check", pan))

        # Field 2: Processing Code
        proc_code = fields[2].strip() if len(fields) > 2 else ""
        if proc_code and proc_code not in VALID_PROCESSING_CODES:
            report.add(VCFValidationResult(Severity.WARNING, "processing_code", "Transaction", ln,
                                           f"Uncommon processing code '{proc_code}'", proc_code))

        # Field 3: Transaction Amount
        amount = fields[3].strip() if len(fields) > 3 else ""
        if amount:
            try:
                amt_val = float(amount.replace(',', ''))
                if amt_val < 0:
                    report.add(VCFValidationResult(Severity.ERROR, "amount", "Transaction", ln,
                                                   "Transaction amount cannot be negative", amount))
                if amt_val > 999999999.99:
                    report.add(VCFValidationResult(Severity.WARNING, "amount", "Transaction", ln,
                                                   "Transaction amount exceeds typical maximum", amount))
            except ValueError:
                report.add(VCFValidationResult(Severity.ERROR, "amount", "Transaction", ln,
                                               "Transaction amount must be numeric", amount))

        # Field 4: Currency Code
        currency = fields[4].strip() if len(fields) > 4 else ""
        if currency and currency not in VALID_CURRENCY_CODES:
            report.add(VCFValidationResult(Severity.ERROR, "currency_code", "Transaction", ln,
                                           f"Invalid ISO 4217 currency code '{currency}'", currency))

        # Field 5: Transaction Date/Time
        txn_dt = fields[5].strip() if len(fields) > 5 else ""
        if txn_dt and not re.match(r'^\d{12,14}$', txn_dt):
            report.add(VCFValidationResult(Severity.ERROR, "transaction_datetime", "Transaction", ln,
                                           "Transaction datetime must be YYYYMMDDHHMMSS or MMDDHHMMSS",
                                           txn_dt))

        # Field 6: Merchant Category Code (MCC) - 4 digits
        mcc = fields[6].strip() if len(fields) > 6 else ""
        if mcc and not re.match(r'^\d{4}$', mcc):
            report.add(VCFValidationResult(Severity.ERROR, "mcc", "Transaction", ln,
                                           "MCC must be exactly 4 digits", mcc, "4 digits"))

        # Field 7: POS Entry Mode
        pos = fields[7].strip() if len(fields) > 7 else ""
        if pos and pos not in VALID_POS_ENTRY_MODES:
            report.add(VCFValidationResult(Severity.WARNING, "pos_entry_mode", "Transaction", ln,
                                           f"Unrecognized POS entry mode '{pos}'", pos))

        # Field 8: Response Code
        resp = fields[8].strip() if len(fields) > 8 else ""
        if resp and resp not in VALID_RESPONSE_CODES:
            report.add(VCFValidationResult(Severity.WARNING, "response_code", "Transaction", ln,
                                           f"Unrecognized response code '{resp}'", resp))

        # Field 9: Authorization Code (6 alphanumeric)
        auth = fields[9].strip() if len(fields) > 9 else ""
        if auth and not re.match(r'^[A-Z0-9]{6}$', auth.upper()):
            report.add(VCFValidationResult(Severity.WARNING, "auth_code", "Transaction", ln,
                                           "Authorization code is typically 6 alphanumeric characters",
                                           auth, "6 alphanumeric"))

        # Field 10: Merchant ID (15 alphanumeric)
        mid = fields[10].strip() if len(fields) > 10 else ""
        if mid and not re.match(r'^[A-Z0-9 ]{1,15}$', mid.upper()):
            report.add(VCFValidationResult(Severity.INFO, "merchant_id", "Transaction", ln,
                                           "Merchant ID should be 1-15 alphanumeric characters", mid))

        # Field 11: Terminal ID (8 alphanumeric)
        tid = fields[11].strip() if len(fields) > 11 else ""
        if tid and not re.match(r'^[A-Z0-9 ]{1,8}$', tid.upper()):
            report.add(VCFValidationResult(Severity.INFO, "terminal_id", "Transaction", ln,
                                           "Terminal ID should be 1-8 alphanumeric characters", tid))

        # Field 14: ARN (Acquirer Reference Number) - 23 digits
        if len(fields) > 14:
            arn = fields[14].strip()
            if arn and not re.match(r'^\d{23}$', arn):
                report.add(VCFValidationResult(Severity.WARNING, "arn", "Transaction", ln,
                                               "ARN should be exactly 23 digits", arn, "23 digits"))

        # Field 16: Interchange Fee
        if len(fields) > 16:
            fee = fields[16].strip()
            if fee:
                try:
                    float(fee.replace(',', ''))
                except ValueError:
                    report.add(VCFValidationResult(Severity.ERROR, "interchange_fee", "Transaction", ln,
                                                   "Interchange fee must be numeric", fee))

    def _validate_vcf_trailer_delimited(self, fields: List[str], ln: int,
                                         expected_count: int, report: VCFValidationReport):
        if not fields:
            return
        rec_type = fields[0].strip()
        if rec_type not in ("TRAILER", "TRL", "99", "EOF"):
            report.add(VCFValidationResult(Severity.WARNING, "trailer_type", "Trailer", ln,
                                           f"Unexpected trailer identifier '{rec_type}'", rec_type))

        if len(fields) > 1:
            try:
                actual_count = int(fields[1].strip())
                if actual_count != expected_count:
                    report.add(VCFValidationResult(Severity.ERROR, "record_count", "Trailer", ln,
                                                   f"Record count mismatch. Counted {expected_count}, trailer says {actual_count}",
                                                   str(actual_count), str(expected_count)))
            except (ValueError, IndexError):
                pass

    def _validate_fixed_header(self, line: str, ln: int, report: VCFValidationReport):
        if len(line) < 20:
            report.add(VCFValidationResult(Severity.ERROR, "header_length", "Header", ln,
                                           "Fixed header too short"))
            return
        date = line[4:12]
        if not re.match(r'^\d{8}$', date):
            report.add(VCFValidationResult(Severity.ERROR, "file_date", "Header", ln,
                                           "File date must be YYYYMMDD", date))

    def _validate_fixed_transaction(self, line: str, ln: int, tc: str, report: VCFValidationReport):
        if len(line) < 50:
            report.add(VCFValidationResult(Severity.ERROR, "record_length", "Transaction", ln,
                                           f"Transaction record too short ({len(line)} chars)"))

    def _validate_fixed_trailer(self, line: str, ln: int, report: VCFValidationReport):
        if len(line) < 20:
            report.add(VCFValidationResult(Severity.WARNING, "trailer_length", "Trailer", ln,
                                           "Fixed trailer shorter than expected"))

    def _luhn_check(self, card_number: str) -> bool:
        """Luhn algorithm for PAN validation"""
        try:
            digits = [int(d) for d in card_number if d.isdigit()]
            if len(digits) < 13:
                return False
            odd_digits = digits[-1::-2]
            even_digits = digits[-2::-2]
            total = sum(odd_digits)
            for d in even_digits:
                total += sum(divmod(d * 2, 10))
            return total % 10 == 0
        except Exception:
            return False

    def _build_stats(self, lines: List[str], is_delimited: bool) -> Dict:
        delimiter = '|' if is_delimited and lines and '|' in lines[0] else ','
        stats = {
            "total_lines": len(lines),
            "format": "delimited" if is_delimited else "fixed-width",
            "delimiter": delimiter if is_delimited else "N/A",
        }

        if is_delimited and len(lines) > 2:
            tc_counts = {}
            for line in lines[1:-1]:
                if line.strip():
                    tc = line.split(delimiter)[0].strip()
                    tc_counts[tc] = tc_counts.get(tc, 0) + 1
            stats["transaction_code_distribution"] = tc_counts
            stats["transaction_count"] = sum(tc_counts.values())

        return stats
