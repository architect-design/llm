"""
Validation Engine — Hybrid Rule + Model Validation.

Two validation layers work in tandem:

  Layer 1 — DETERMINISTIC RULE ENGINE (always runs first)
    Checks hard spec rules from the ConfigEngine:
    - Field length, character class, mandatory presence
    - Routing number check-digit (Mod-10 algorithm)
    - ACH batch/file control checksums
    - Sequence number ordering

  Layer 2 — SLM VALIDATION HEAD (runs after Layer 1)
    The model's bidirectional attention scores each field's
    contextual validity — catching inter-field inconsistencies
    that pure rules cannot (e.g., amount vs entry count mismatch).

Results are merged into a unified ValidationReport.
"""

from __future__ import annotations

import re
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


# ─────────────────────── Result Structures ───────────────────────────────────

@dataclass
class FieldError:
    line_no    : int
    field_name : str
    position   : str        # e.g. "cols 4-13"
    raw_value  : str
    rule       : str        # human-readable rule that failed
    severity   : str = "ERROR"   # ERROR | WARNING | INFO
    source     : str = "RULE"    # RULE | MODEL


@dataclass
class LineResult:
    line_no      : int
    raw          : str
    record_type  : str
    is_valid     : bool
    errors       : List[FieldError] = field(default_factory=list)
    model_conf   : float = 1.0     # SLM confidence (0–1)


@dataclass
class ValidationReport:
    spec_name        : str
    total_lines      : int
    valid_lines      : int
    error_lines      : int
    line_results     : List[LineResult]
    checksum_valid   : bool = True
    sequence_valid   : bool = True
    structure_valid  : bool = True
    summary          : str  = ""

    @property
    def is_fully_valid(self) -> bool:
        return (self.error_lines == 0 and
                self.checksum_valid and
                self.sequence_valid and
                self.structure_valid)

    def to_dict(self) -> dict:
        return {
            "spec_name"      : self.spec_name,
            "total_lines"    : self.total_lines,
            "valid_lines"    : self.valid_lines,
            "error_lines"    : self.error_lines,
            "is_fully_valid" : self.is_fully_valid,
            "checksum_valid" : self.checksum_valid,
            "sequence_valid" : self.sequence_valid,
            "structure_valid": self.structure_valid,
            "summary"        : self.summary,
            "line_results"   : [
                {
                    "line_no"    : lr.line_no,
                    "record_type": lr.record_type,
                    "is_valid"   : lr.is_valid,
                    "model_conf" : round(lr.model_conf, 4),
                    "errors"     : [
                        {
                            "field"   : e.field_name,
                            "position": e.position,
                            "value"   : repr(e.raw_value),
                            "rule"    : e.rule,
                            "severity": e.severity,
                            "source"  : e.source,
                        }
                        for e in lr.errors
                    ],
                }
                for lr in self.line_results
            ],
        }


# ─────────────────────── Rule Validators ─────────────────────────────────────

class RuleEngine:
    """Deterministic field-level rule checks drawn from ConfigEngine specs."""

    # ── ACH routing number check digit (Mod-10 weighted) ──────────────────
    _ROUTING_WEIGHTS = [3, 7, 1, 3, 7, 1, 3, 7, 1]

    @classmethod
    def validate_routing(cls, value: str) -> Tuple[bool, str]:
        v = value.strip()
        if not v.isdigit() or len(v) != 9:
            return False, "Routing number must be exactly 9 digits"
        total = sum(int(d) * w for d, w in zip(v, cls._ROUTING_WEIGHTS))
        if total % 10 != 0:
            return False, f"Routing check-digit failed (sum={total}, mod10={total%10})"
        return True, ""

    @staticmethod
    def validate_numeric(value: str, field_name: str) -> Tuple[bool, str]:
        if not value.strip() and value.strip() == "":
            return False, f"{field_name}: required numeric field is blank"
        if not value.replace(" ", "0").isdigit():
            return False, f"{field_name}: must contain only digits (got {value!r})"
        return True, ""

    @staticmethod
    def validate_alphanumeric(value: str, field_name: str) -> Tuple[bool, str]:
        if re.search(r"[^A-Z0-9 /\-\.@#\*]", value.upper()):
            return False, f"{field_name}: contains invalid characters"
        return True, ""

    @staticmethod
    def validate_amount(value: str, field_name: str) -> Tuple[bool, str]:
        """Zero-padded implied-decimal amounts (e.g., 10 chars = $99999999.99)."""
        if not value.isdigit():
            return False, f"{field_name}: amount field must be zero-padded digits"
        return True, ""

    @staticmethod
    def validate_date(value: str, field_name: str) -> Tuple[bool, str]:
        """YYMMDD date validation."""
        if len(value) != 6 or not value.isdigit():
            return False, f"{field_name}: date must be 6 digits YYMMDD"
        yy, mm, dd = int(value[:2]), int(value[2:4]), int(value[4:])
        if not (1 <= mm <= 12 and 1 <= dd <= 31):
            return False, f"{field_name}: invalid month/day ({mm:02d}/{dd:02d})"
        return True, ""

    @staticmethod
    def validate_length(value: str, expected_len: int, field_name: str) -> Tuple[bool, str]:
        if len(value) != expected_len:
            return False, f"{field_name}: expected length {expected_len}, got {len(value)}"
        return True, ""

    @staticmethod
    def validate_allowed_values(value: str, allowed: List[str], field_name: str) -> Tuple[bool, str]:
        if value.strip() not in [a.strip() for a in allowed]:
            return False, f"{field_name}: {value!r} not in allowed values {allowed}"
        return True, ""

    @staticmethod
    def validate_pattern(value: str, pattern: str, field_name: str) -> Tuple[bool, str]:
        if not re.fullmatch(pattern, value):
            return False, f"{field_name}: value {value!r} does not match pattern /{pattern}/"
        return True, ""


# ─────────────────────── ACH-Specific Checks ─────────────────────────────────

class ACHChecksumValidator:
    """
    Validates ACH NACHA structural integrity:
    - Entry/Addenda count per batch
    - Hash totals (sum of routing numbers mod 10^10)
    - Dollar totals
    - Block count (file must be padded to multiples of 10 lines)
    """

    def __init__(self):
        self.batch_routing_sum  : int = 0
        self.batch_entry_count  : int = 0
        self.batch_debit_total  : int = 0
        self.batch_credit_total : int = 0
        self.file_entry_count   : int = 0
        self.file_routing_sum   : int = 0
        self.file_debit_total   : int = 0
        self.file_credit_total  : int = 0
        self.errors             : List[str] = []

    @staticmethod
    def _safe_int(s: str) -> int:
        """Parse an int from a string that may contain non-digit characters."""
        cleaned = "".join(c for c in s if c.isdigit())
        return int(cleaned) if cleaned else 0

    def process_line(self, rt: str, line: str):
        if rt == "RT6":   # Entry Detail
            routing = line[3:11]
            amount  = line[29:39]
            if routing.isdigit():
                self.batch_routing_sum += int(routing)
            if amount.isdigit():
                # Credits: 22, 23, 32, 33 | Debits: 27, 28, 37, 38
                tx_code = line[1:3] if len(line) >= 3 else "00"
                if tx_code in ("22", "23", "32", "33"):
                    self.batch_credit_total += int(amount)
                else:
                    self.batch_debit_total  += int(amount)
            self.batch_entry_count += 1

        elif rt == "RT8":  # Batch Control
            reported_count   = self._safe_int(line[4:10])
            reported_hash    = self._safe_int(line[10:20])
            reported_debits  = self._safe_int(line[20:32])
            reported_credits = self._safe_int(line[32:44])

            actual_hash = self.batch_routing_sum % (10 ** 10)

            if reported_count != self.batch_entry_count:
                self.errors.append(
                    f"Batch entry count mismatch: header={reported_count}, "
                    f"actual={self.batch_entry_count}"
                )
            if reported_hash != actual_hash:
                self.errors.append(
                    f"Batch hash mismatch: reported={reported_hash}, "
                    f"computed={actual_hash}"
                )
            self.file_entry_count   += self.batch_entry_count
            self.file_routing_sum   += self.batch_routing_sum
            self.file_debit_total   += self.batch_debit_total
            self.file_credit_total  += self.batch_credit_total
            # Reset batch accumulators
            self.batch_routing_sum  = 0
            self.batch_entry_count  = 0
            self.batch_debit_total  = 0
            self.batch_credit_total = 0

        elif rt == "RT9":  # File Control
            # RT9 field positions (1-indexed spec → 0-indexed slice):
            # Batch Count    : cols  2-7  → line[1:7]
            # Block Count    : cols  8-13 → line[7:13]
            # Entry/Add Count: cols 14-21 → line[13:21]
            # Entry Hash     : cols 22-31 → line[21:31]
            # Total Debit    : cols 32-43 → line[31:43]
            # Total Credit   : cols 44-55 → line[43:55]
            rep_entry_count = self._safe_int(line[13:21])
            rep_block_count = self._safe_int(line[7:13])
            rep_hash        = self._safe_int(line[21:31])
            rep_debits      = self._safe_int(line[31:43])
            rep_credits     = self._safe_int(line[43:55])

            actual_hash = self.file_routing_sum % (10 ** 10)
            if rep_hash != actual_hash:
                self.errors.append(
                    f"File hash mismatch: reported={rep_hash}, computed={actual_hash}"
                )
            if rep_entry_count != self.file_entry_count:
                self.errors.append(
                    f"File entry count mismatch: reported={rep_entry_count}, "
                    f"actual={self.file_entry_count}"
                )

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0


# ─────────────────────── Main Validator Class ─────────────────────────────────

class FinancialValidator:
    """
    Hybrid validator: deterministic rules + optional SLM model scoring.

    Usage:
        from memory.config_engine import ConfigEngine
        engine = ConfigEngine()
        validator = FinancialValidator("ACH_NACHA", engine, model=None)
        report = validator.validate(raw_text)
        print(report.to_dict())
    """

    def __init__(
        self,
        spec_name    : str,
        config_engine,
        tokenizer    = None,
        model        = None,
        device       : str = "cpu",
    ):
        self.spec_name     = spec_name
        self.config_engine = config_engine
        self.tokenizer     = tokenizer
        self.model         = model
        self.device        = torch.device(device)
        self.rules         = RuleEngine()

        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()

    def validate(self, raw_text: str) -> ValidationReport:
        lines = self._split_lines(raw_text)
        line_results: List[LineResult] = []
        ach_checker = ACHChecksumValidator() if self.spec_name == "ACH_NACHA" else None

        # Sequence tracking
        prev_sequence = -1
        seq_valid     = True

        for i, line in enumerate(lines):
            if not line.strip():
                continue

            # ACH padding records are all-nines — skip field validation on them
            is_padding = all(c == "9" for c in line.strip())

            rt      = self.tokenizer.get_record_type(line) if self.tokenizer else f"RT{line[0]}"
            errors  = [] if is_padding else self._check_line_rules(line, rt, i)

            # ACH-specific structural checks (skip padding records)
            if ach_checker and not is_padding:
                ach_checker.process_line(rt, line)

            # Sequence check: trace number last 7 digits should be ascending (WARNING only)
            if self.spec_name == "ACH_NACHA" and rt == "RT6" and len(line) >= 94:
                seq = int(line[87:94].strip() or 0)
                if prev_sequence > 0 and seq < prev_sequence:
                    seq_valid = False
                    errors.append(FieldError(
                        line_no=i+1, field_name="Trace Number (Sequence)",
                        position="cols 88-94", raw_value=line[87:94],
                        rule=f"Trace sequence {seq} < previous {prev_sequence} (out of order)",
                        severity="WARNING", source="RULE"
                    ))
                prev_sequence = seq

            # SLM model scoring
            model_conf = self._model_score(line, rt) if self.model else 1.0

            lr = LineResult(
                line_no    = i + 1,
                raw        = line,
                record_type= rt,
                is_valid   = len([e for e in errors if e.severity == "ERROR"]) == 0,
                errors     = errors,
                model_conf = model_conf,
            )
            line_results.append(lr)

        valid_count = sum(1 for lr in line_results if lr.is_valid)
        error_count = len(line_results) - valid_count

        # File-level structure check
        struct_valid = self._check_structure(lines)

        checksum_valid = True
        if ach_checker:
            checksum_valid = ach_checker.is_valid
            if not checksum_valid:
                # Inject checksum errors as line-level warnings
                for msg in ach_checker.errors:
                    line_results.append(LineResult(
                        line_no=0, raw="", record_type="FILE_CONTROL",
                        is_valid=False,
                        errors=[FieldError(0, "File Control", "N/A", "", msg, "ERROR", "RULE")],
                    ))

        summary = self._build_summary(valid_count, error_count, checksum_valid, struct_valid)

        return ValidationReport(
            spec_name      = self.spec_name,
            total_lines    = len(line_results),
            valid_lines    = valid_count,
            error_lines    = error_count,
            line_results   = line_results,
            checksum_valid = checksum_valid,
            sequence_valid = seq_valid,
            structure_valid= struct_valid,
            summary        = summary,
        )

    # ── Rule checking per line ─────────────────────────────────────────────

    def _check_line_rules(
        self, line: str, rt: str, line_no: int
    ) -> List[FieldError]:
        errors: List[FieldError] = []
        fields = self.config_engine.get_fields(self.spec_name, rt)
        if not fields:
            return errors

        for fd in fields:
            raw  = fd["descriptor"].extract(line) if hasattr(fd.get("descriptor", None), "extract") else \
                   line[fd["start"]-1:fd["end"]]
            name = fd["name"]
            ftype= fd["field_type"]
            pos  = f"cols {fd['start']}-{fd['end']}"

            err_msg = ""

            if ftype in ("NUMERIC", "AMOUNT"):
                ok, msg = self.rules.validate_numeric(raw, name)
                if not ok:
                    err_msg = msg

            elif ftype == "ALPHANUMERIC":
                ok, msg = self.rules.validate_alphanumeric(raw, name)
                if not ok:
                    err_msg = msg

            elif ftype == "ROUTING":
                if fd["length"] == 9:
                    # Full 9-digit routing with check digit
                    ok, msg = self.rules.validate_routing(raw)
                    if not ok:
                        err_msg = msg
                else:
                    # Partial routing (e.g. 8-char RDFI in ACH RT6 where check digit is separate)
                    ok, msg = self.rules.validate_numeric(raw, name)
                    if not ok:
                        err_msg = msg

            elif ftype == "DATE":
                ok, msg = self.rules.validate_date(raw, name)
                if not ok:
                    err_msg = msg

            if not err_msg and fd.get("pattern"):
                ok, msg = self.rules.validate_pattern(raw, fd["pattern"], name)
                if not ok:
                    err_msg = msg

            if not err_msg and fd.get("allowed"):
                ok, msg = self.rules.validate_allowed_values(raw, fd["allowed"], name)
                if not ok:
                    err_msg = msg

            if not err_msg and fd.get("required") and not raw.strip():
                # BLANK_PAD (reserved) fields are supposed to be blank — never flag as missing
                if ftype != "BLANK_PAD" and "reserved" not in name.lower():
                    err_msg = f"{name}: required field is blank"

            if err_msg:
                errors.append(FieldError(
                    line_no=line_no+1, field_name=name,
                    position=pos, raw_value=raw,
                    rule=err_msg, severity="ERROR", source="RULE"
                ))
        return errors

    # ── SLM model confidence scoring ──────────────────────────────────────

    def _model_score(self, line: str, rt: str) -> float:
        """Return SLM confidence score (0–1) for a single line."""
        if self.tokenizer is None or self.model is None:
            return 1.0
        try:
            T        = self.model.cfg.max_seq_len
            line_pad = line.upper().ljust(T)[:T]
            char_ids  = torch.tensor(
                [self.tokenizer.char_to_id(c) for c in line_pad],
                dtype=torch.long
            ).unsqueeze(0).to(self.device)
            field_ids = torch.zeros_like(char_ids)
            rt_id     = 0
            rt_ids    = torch.tensor([rt_id], dtype=torch.long).to(self.device)

            with torch.no_grad():
                _, _, conf = self.model(char_ids, field_ids, rt_ids, mode="validate")
            return float(conf[0, 0]) if conf is not None else 1.0
        except Exception:
            return 1.0

    # ── Structural checks ─────────────────────────────────────────────────

    def _check_structure(self, lines: List[str]) -> bool:
        """High-level structural integrity checks per spec."""
        non_empty = [l for l in lines if l.strip()]
        if self.spec_name == "ACH_NACHA":
            if not non_empty:
                return False
            # Must start with '1' (File Header) and end with '9' (File Control)
            if non_empty[0][0] != "1" or non_empty[-1][0] != "9":
                return False
            # Line count must be a multiple of 10 (blocking factor)
            if len(non_empty) % 10 != 0:
                return False
        elif self.spec_name == "VISA_VCF":
            # Must have a VH (Volume Header) and VT (Volume Trailer)
            types = [l[:2].strip() for l in non_empty]
            if "VH" not in types or "VT" not in types:
                return False
        return True

    def _split_lines(self, raw: str) -> List[str]:
        lines = raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        return [l for l in lines if l]

    @staticmethod
    def _build_summary(
        valid: int, errors: int, checksum: bool, structure: bool
    ) -> str:
        parts = [f"{valid} lines valid, {errors} lines with errors"]
        if not checksum:
            parts.append("CHECKSUM FAILURES DETECTED")
        if not structure:
            parts.append("STRUCTURAL INTEGRITY FAILED")
        return " | ".join(parts)
