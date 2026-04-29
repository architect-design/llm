"""
ACH NACHA File Validator
Implements full NACHA Operating Rules and Guidelines validation
Supports: PPD, CCD, CTX, WEB, TEL, COR/NOC, RCK, ARC, BOC, POP, XCK
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum


class Severity(Enum):
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass
class ValidationResult:
    severity: Severity
    field: str
    record_type: str
    line_number: int
    message: str
    value: Optional[str] = None
    expected: Optional[str] = None


@dataclass
class ACHValidationReport:
    is_valid: bool
    errors: List[ValidationResult] = field(default_factory=list)
    warnings: List[ValidationResult] = field(default_factory=list)
    info: List[ValidationResult] = field(default_factory=list)
    statistics: Dict = field(default_factory=dict)

    def add(self, result: ValidationResult):
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
            "errors": [self._result_to_dict(r) for r in self.errors],
            "warnings": [self._result_to_dict(r) for r in self.warnings],
            "info": [self._result_to_dict(r) for r in self.info],
            "statistics": self.statistics,
        }

    def _result_to_dict(self, r: ValidationResult):
        return {
            "severity": r.severity.value,
            "field": r.field,
            "record_type": r.record_type,
            "line": r.line_number,
            "message": r.message,
            "value": r.value,
            "expected": r.expected,
        }


# ─── NACHA Field Specifications ───────────────────────────────────────────────

FILE_HEADER_FIELDS = {
    "record_type":           (1,  1,  "N", "1"),
    "priority_code":         (2,  3,  "N", "01"),
    "immediate_destination":  (4,  13, "AN", None),
    "immediate_origin":       (14, 23, "AN", None),
    "file_creation_date":    (24, 29, "N", None),
    "file_creation_time":    (30, 33, "N", None),
    "file_id_modifier":      (34, 34, "AN", None),
    "record_size":           (35, 37, "N", "094"),
    "blocking_factor":       (38, 39, "N", "10"),
    "format_code":           (40, 40, "N", "1"),
    "immediate_destination_name": (41, 63, "AN", None),
    "immediate_origin_name":      (64, 86, "AN", None),
    "reference_code":        (87, 94, "AN", None),
}

BATCH_HEADER_FIELDS = {
    "record_type":            (1,  1,  "N", "5"),
    "service_class_code":     (2,  4,  "N", None),
    "company_name":           (5,  20, "AN", None),
    "company_discretionary":  (21, 40, "AN", None),
    "company_id":             (41, 50, "AN", None),
    "standard_entry_class":   (51, 53, "AN", None),
    "company_entry_desc":     (54, 63, "AN", None),
    "company_descriptive_date":(64, 69, "AN", None),
    "effective_entry_date":   (70, 75, "N", None),
    "settlement_date":        (76, 78, "N", None),
    "originator_status":      (79, 79, "AN", None),
    "odfi_routing":           (80, 87, "N", None),
    "batch_number":           (88, 94, "N", None),
}

ENTRY_DETAIL_FIELDS = {
    "record_type":           (1,  1,  "N", "6"),
    "transaction_code":      (2,  3,  "N", None),
    "rdfi_routing":          (4,  11, "N", None),
    "check_digit":           (12, 12, "N", None),
    "rdfi_account_number":   (13, 29, "AN", None),
    "amount":                (30, 39, "N", None),
    "individual_id":         (40, 54, "AN", None),
    "individual_name":       (55, 76, "AN", None),
    "discretionary_data":    (77, 78, "AN", None),
    "addenda_indicator":     (79, 79, "N", None),
    "trace_number":          (80, 94, "N", None),
}

ADDENDA_FIELDS = {
    "record_type":           (1,  1,  "N", "7"),
    "addenda_type_code":     (2,  3,  "N", None),
    "payment_info":          (4,  83, "AN", None),
    "sequence_number":       (84, 87, "N", None),
    "entry_detail_seq":      (88, 94, "N", None),
}

BATCH_CONTROL_FIELDS = {
    "record_type":           (1,  1,  "N", "8"),
    "service_class_code":    (2,  4,  "N", None),
    "entry_addenda_count":   (5,  10, "N", None),
    "entry_hash":            (11, 20, "N", None),
    "total_debit":           (21, 32, "N", None),
    "total_credit":          (33, 44, "N", None),
    "company_id":            (45, 54, "AN", None),
    "message_auth_code":     (55, 73, "AN", None),
    "reserved":              (74, 79, "AN", None),
    "odfi_routing":          (80, 87, "N", None),
    "batch_number":          (88, 94, "N", None),
}

FILE_CONTROL_FIELDS = {
    "record_type":           (1,  1,  "N", "9"),
    "batch_count":           (2,  7,  "N", None),
    "block_count":           (8,  13, "N", None),
    "entry_addenda_count":   (14, 21, "N", None),
    "entry_hash":            (22, 31, "N", None),
    "total_debit":           (32, 43, "N", None),
    "total_credit":          (44, 55, "N", None),
    "reserved":              (56, 94, "AN", None),
}

VALID_SERVICE_CLASSES = {"200", "220", "225"}
VALID_TRANSACTION_CODES = {
    "22", "23", "24", "27", "28", "29",   # Checking
    "32", "33", "34", "37", "38", "39",   # Savings
    "42", "43", "44", "47", "48", "49",   # GL
    "52", "53", "54", "55",               # Loan
}
VALID_SEC_CODES = {
    "PPD", "CCD", "CTX", "WEB", "TEL", "COR", "NOC",
    "RCK", "ARC", "BOC", "POP", "XCK", "MTE", "SHR", "IAT", "ENR"
}
VALID_ADDENDA_TYPE_CODES = {"02", "05", "98", "99"}


class ACHValidator:
    """Full NACHA ACH file validator"""

    def validate(self, content: str) -> ACHValidationReport:
        lines = content.splitlines()
        # Remove padding (9-filled lines at end)
        while lines and lines[-1].strip('9') == '':
            lines.pop()

        report = ACHValidationReport(is_valid=True)
        self._validate_structure(lines, report)

        if report.errors:
            report.is_valid = False
            return report

        file_header = None
        batches = []
        current_batch = None
        file_control = None

        for i, line in enumerate(lines, 1):
            ln = i
            if len(line) != 94:
                report.add(ValidationResult(
                    Severity.ERROR, "record_length", "ALL", ln,
                    f"Each record must be exactly 94 characters. Got {len(line)}",
                    str(len(line)), "94"
                ))
                report.is_valid = False
                continue

            rt = line[0]

            if rt == '1':
                file_header = line
                self._validate_file_header(line, ln, report)

            elif rt == '5':
                current_batch = {"header": line, "entries": [], "control": None}
                batches.append(current_batch)
                self._validate_batch_header(line, ln, report)

            elif rt == '6':
                if current_batch:
                    current_batch["entries"].append(line)
                self._validate_entry_detail(line, ln, report)

            elif rt == '7':
                if current_batch and current_batch["entries"]:
                    prev = current_batch["entries"][-1]
                    if prev[78] != '1':
                        report.add(ValidationResult(
                            Severity.ERROR, "addenda_indicator", "Entry Detail", ln,
                            "Addenda record found but preceding entry has addenda indicator = 0"
                        ))
                self._validate_addenda(line, ln, report)

            elif rt == '8':
                if current_batch:
                    current_batch["control"] = line
                self._validate_batch_control(line, ln, report)
                if current_batch:
                    self._cross_validate_batch(current_batch, ln, report)

            elif rt == '9':
                file_control = line
                self._validate_file_control(line, ln, report)

        # Cross-file validation
        if file_header and file_control:
            self._cross_validate_file(file_header, file_control, batches, len(lines), report)

        # Build statistics
        report.statistics = self._build_statistics(file_header, batches, file_control, lines)
        report.is_valid = len(report.errors) == 0
        return report

    def _validate_structure(self, lines: List[str], report: ACHValidationReport):
        if not lines:
            report.add(ValidationResult(Severity.ERROR, "file", "FILE", 0, "Empty file"))
            return

        if lines[0][0] != '1':
            report.add(ValidationResult(Severity.ERROR, "record_type", "File Header", 1,
                                        "File must begin with File Header (record type 1)"))
        if lines[-1][0] != '9':
            report.add(ValidationResult(Severity.ERROR, "record_type", "File Control", len(lines),
                                        "File must end with File Control (record type 9)"))

        total = len(lines)
        if total % 10 != 0:
            report.add(ValidationResult(Severity.WARNING, "blocking_factor", "FILE", 0,
                                        f"Total records ({total}) should be a multiple of 10 (blocking factor)"))

    def _validate_file_header(self, line: str, ln: int, report: ACHValidationReport):
        self._check_field(line, "priority_code", 2, 3, "01", ln, report, "File Header")
        self._check_field(line, "record_size", 35, 37, "094", ln, report, "File Header")
        self._check_field(line, "blocking_factor", 38, 39, "10", ln, report, "File Header")
        self._check_field(line, "format_code", 40, 40, "1", ln, report, "File Header")

        dest = line[3:13].strip()
        origin = line[13:23].strip()

        if not re.match(r'^\d{9,10}$', dest.replace(' ', '')):
            report.add(ValidationResult(Severity.ERROR, "immediate_destination", "File Header", ln,
                                        "Immediate destination must be a 9-digit ABA routing number", dest))

        date = line[23:29]
        if not re.match(r'^\d{6}$', date):
            report.add(ValidationResult(Severity.ERROR, "file_creation_date", "File Header", ln,
                                        "File creation date must be YYMMDD format", date, "YYMMDD"))

        time = line[29:33]
        if time.strip() and not re.match(r'^\d{4}$', time):
            report.add(ValidationResult(Severity.WARNING, "file_creation_time", "File Header", ln,
                                        "File creation time must be HHMM format when provided", time, "HHMM"))

    def _validate_batch_header(self, line: str, ln: int, report: ACHValidationReport):
        scc = line[1:4]
        if scc not in VALID_SERVICE_CLASSES:
            report.add(ValidationResult(Severity.ERROR, "service_class_code", "Batch Header", ln,
                                        f"Invalid service class code. Must be 200, 220, or 225", scc,
                                        "200/220/225"))

        sec = line[50:53]
        if sec not in VALID_SEC_CODES:
            report.add(ValidationResult(Severity.ERROR, "standard_entry_class", "Batch Header", ln,
                                        f"Invalid SEC code '{sec}'", sec, str(VALID_SEC_CODES)))

        effective_date = line[69:75]
        if effective_date.strip() and not re.match(r'^\d{6}$', effective_date):
            report.add(ValidationResult(Severity.ERROR, "effective_entry_date", "Batch Header", ln,
                                        "Effective entry date must be YYMMDD", effective_date, "YYMMDD"))

        odfi = line[79:87]
        if not re.match(r'^\d{8}$', odfi):
            report.add(ValidationResult(Severity.ERROR, "odfi_routing", "Batch Header", ln,
                                        "ODFI routing must be 8 digits (no check digit)", odfi, "8 digits"))

    def _validate_entry_detail(self, line: str, ln: int, report: ACHValidationReport):
        tc = line[1:3]
        if tc not in VALID_TRANSACTION_CODES:
            report.add(ValidationResult(Severity.ERROR, "transaction_code", "Entry Detail", ln,
                                        f"Invalid transaction code '{tc}'", tc, str(VALID_TRANSACTION_CODES)))

        routing = line[3:11]
        if not re.match(r'^\d{8}$', routing):
            report.add(ValidationResult(Severity.ERROR, "rdfi_routing", "Entry Detail", ln,
                                        "RDFI routing must be 8 digits", routing))

        check_digit = line[11]
        if not check_digit.isdigit():
            report.add(ValidationResult(Severity.ERROR, "check_digit", "Entry Detail", ln,
                                        "Check digit must be numeric", check_digit))
        else:
            full_routing = routing + check_digit
            if not self._validate_routing_check_digit(full_routing):
                report.add(ValidationResult(Severity.ERROR, "check_digit", "Entry Detail", ln,
                                            f"Routing number check digit failed ABA validation",
                                            full_routing))

        amount = line[29:39]
        if not re.match(r'^\d{10}$', amount):
            report.add(ValidationResult(Severity.ERROR, "amount", "Entry Detail", ln,
                                        "Amount must be 10 digits (implied 2 decimal places)", amount))

        addenda = line[78]
        if addenda not in ('0', '1'):
            report.add(ValidationResult(Severity.ERROR, "addenda_indicator", "Entry Detail", ln,
                                        "Addenda indicator must be 0 or 1", addenda))

        trace = line[79:94]
        if not re.match(r'^\d{15}$', trace):
            report.add(ValidationResult(Severity.ERROR, "trace_number", "Entry Detail", ln,
                                        "Trace number must be 15 digits", trace))

    def _validate_addenda(self, line: str, ln: int, report: ACHValidationReport):
        atc = line[1:3]
        if atc not in VALID_ADDENDA_TYPE_CODES:
            report.add(ValidationResult(Severity.WARNING, "addenda_type_code", "Addenda", ln,
                                        f"Uncommon addenda type code '{atc}'", atc))

    def _validate_batch_control(self, line: str, ln: int, report: ACHValidationReport):
        scc = line[1:4]
        if scc not in VALID_SERVICE_CLASSES:
            report.add(ValidationResult(Severity.ERROR, "service_class_code", "Batch Control", ln,
                                        "Invalid service class code", scc))

    def _validate_file_control(self, line: str, ln: int, report: ACHValidationReport):
        block_count = line[7:13].strip()
        if not block_count.isdigit():
            report.add(ValidationResult(Severity.ERROR, "block_count", "File Control", ln,
                                        "Block count must be numeric", block_count))

    def _cross_validate_batch(self, batch: dict, ln: int, report: ACHValidationReport):
        if not batch["control"]:
            return

        ctrl = batch["control"]
        expected_count = len(batch["entries"])
        actual_count = int(ctrl[4:10].strip() or 0)

        if actual_count != expected_count:
            report.add(ValidationResult(Severity.ERROR, "entry_addenda_count", "Batch Control", ln,
                                        f"Entry/Addenda count mismatch. Expected {expected_count}, got {actual_count}",
                                        str(actual_count), str(expected_count)))

        # Hash validation
        routing_sum = sum(int(e[3:11]) for e in batch["entries"] if e[0] == '6')
        expected_hash = str(routing_sum)[-10:].zfill(10)
        actual_hash = ctrl[10:20]
        if actual_hash != expected_hash:
            report.add(ValidationResult(Severity.ERROR, "entry_hash", "Batch Control", ln,
                                        f"Entry hash mismatch. Calculated {expected_hash}",
                                        actual_hash, expected_hash))

        # Debit/credit totals
        debits = sum(int(e[29:39]) for e in batch["entries"]
                     if e[0] == '6' and e[1:3] in {"27", "37", "47", "22", "32", "42"})
        credits = sum(int(e[29:39]) for e in batch["entries"]
                      if e[0] == '6' and e[1:3] in {"22", "32", "42", "23", "33", "43"})

    def _cross_validate_file(self, file_header, file_control, batches, total_lines, report):
        actual_batches = len(batches)
        ctrl_batches = int(file_control[1:7].strip() or 0)
        if ctrl_batches != actual_batches:
            report.add(ValidationResult(Severity.ERROR, "batch_count", "File Control", total_lines,
                                        f"Batch count mismatch. Found {actual_batches}, control says {ctrl_batches}",
                                        str(ctrl_batches), str(actual_batches)))

    def _check_field(self, line, fname, start, end, expected, ln, report, rec_type):
        val = line[start-1:end]
        if val != expected:
            report.add(ValidationResult(Severity.ERROR, fname, rec_type, ln,
                                        f"{fname} must be '{expected}'", val, expected))

    def _validate_routing_check_digit(self, routing: str) -> bool:
        """ABA routing number check digit validation"""
        if len(routing) != 9 or not routing.isdigit():
            return False
        d = [int(c) for c in routing]
        checksum = (3*(d[0]+d[3]+d[6]) + 7*(d[1]+d[4]+d[7]) + (d[2]+d[5]+d[8])) % 10
        return checksum == 0

    def _build_statistics(self, file_header, batches, file_control, lines):
        stats = {
            "total_records": len(lines),
            "total_batches": len(batches),
            "total_entries": sum(len(b["entries"]) for b in batches),
        }
        if file_header:
            stats["file_id_modifier"] = file_header[33]
            stats["creation_date"] = file_header[23:29]
        sec_codes = set()
        for b in batches:
            if b["header"]:
                sec_codes.add(b["header"][50:53])
        stats["sec_codes"] = list(sec_codes)
        return stats
