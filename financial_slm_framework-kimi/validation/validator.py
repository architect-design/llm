"""
financial_slm_framework/validation/validator.py
Syntax-Check head and rule-based validation engine.
Performs real-time validation of uploaded files against specification rules.
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import re


class ValidationSeverity(Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ValidationStatus(Enum):
    VALID = "valid"
    INVALID = "invalid"
    PARTIAL = "partial"


@dataclass
class ValidationResult:
    field_name: str
    position: Tuple[int, int]
    expected: str
    actual: str
    message: str
    severity: ValidationSeverity
    rule_type: str


@dataclass
class RecordValidation:
    record_type: str
    record_number: int
    record_text: str
    status: ValidationStatus
    results: List[ValidationResult]
    parsed_fields: Dict[str, str]


@dataclass
class FileValidation:
    spec_id: str
    filename: str
    overall_status: ValidationStatus
    records: List[RecordValidation]
    summary: Dict[str, Any]
    checksum_valid: bool = True


class FinancialValidator:
    """
    Multi-layer validation engine:
    1. Structural validation (length, record order, padding)
    2. Field-level validation (type, format, allowed values)
    3. Semantic validation (checksums, totals, cross-record consistency)
    4. SLM-based validation (neural syntax check)
    """

    def __init__(self, spec_store, model=None, tokenizer=None):
        self.spec_store = spec_store
        self.model = model
        self.tokenizer = tokenizer

    def validate_file(
        self,
        file_content: str,
        spec_id: str,
        filename: str = "unknown"
    ) -> FileValidation:
        """
        Validate an entire file against a specification.

        Args:
            file_content: Raw file content as string
            spec_id: Specification ID to validate against
            filename: Original filename for reporting

        Returns:
            FileValidation with complete validation results
        """
        spec = self.spec_store.get_spec(spec_id)
        if spec is None:
            return FileValidation(
                spec_id=spec_id,
                filename=filename,
                overall_status=ValidationStatus.INVALID,
                records=[],
                summary={"error": f"Specification '{spec_id}' not found"}
            )

        # Split into records (lines)
        lines = file_content.split('\n')
        lines = [line for line in lines if line.strip()]

        records_validation = []
        record_counts = {}
        total_errors = 0
        total_warnings = 0

        for i, line in enumerate(lines):
            record_val = self._validate_record(line, spec, i + 1)
            records_validation.append(record_val)

            record_counts[record_val.record_type] = record_counts.get(record_val.record_type, 0) + 1

            for result in record_val.results:
                if result.severity == ValidationSeverity.ERROR:
                    total_errors += 1
                elif result.severity == ValidationSeverity.WARNING:
                    total_warnings += 1

        # File-level validations
        file_level_errors = self._validate_file_level(lines, spec, record_counts)

        # Determine overall status
        if total_errors > 0:
            overall_status = ValidationStatus.INVALID
        elif total_warnings > 0:
            overall_status = ValidationStatus.PARTIAL
        else:
            overall_status = ValidationStatus.VALID

        # Checksums
        checksum_valid = self._verify_checksums(lines, spec)
        if not checksum_valid:
            overall_status = ValidationStatus.INVALID

        summary = {
            "total_records": len(lines),
            "record_counts": record_counts,
            "total_errors": total_errors,
            "total_warnings": total_warnings,
            "file_level_errors": file_level_errors,
            "checksum_valid": checksum_valid
        }

        return FileValidation(
            spec_id=spec_id,
            filename=filename,
            overall_status=overall_status,
            records=records_validation,
            summary=summary,
            checksum_valid=checksum_valid
        )

    def _validate_record(
        self,
        record: str,
        spec,
        record_number: int
    ) -> RecordValidation:
        """Validate a single record."""
        # Detect record type from first character(s)
        record_type = self._detect_record_type(record, spec)
        record_spec = spec.get_record_spec(record_type)

        if record_spec is None:
            return RecordValidation(
                record_type=record_type or "UNKNOWN",
                record_number=record_number,
                record_text=record,
                status=ValidationStatus.INVALID,
                results=[ValidationResult(
                    field_name="RecordType",
                    position=(0, 1),
                    expected="Known record type",
                    actual=record_type or "",
                    message=f"Unknown record type: '{record_type}'",
                    severity=ValidationSeverity.ERROR,
                    rule_type="structural"
                )],
                parsed_fields={}
            )

        results = []

        # Length validation
        if len(record) != record_spec.total_length:
            results.append(ValidationResult(
                field_name="RecordLength",
                position=(0, len(record)),
                expected=f"{record_spec.total_length} characters",
                actual=f"{len(record)} characters",
                message=f"Record length mismatch: Expected {record_spec.total_length}, got {len(record)}",
                severity=ValidationSeverity.ERROR,
                rule_type="structural"
            ))

        # Field-level validation
        parsed_fields = {}
        for field in record_spec.fields:
            if field.start_pos < len(record):
                value = record[field.start_pos:min(field.end_pos, len(record))]
                parsed_fields[field.name] = value

                is_valid, msg = field.validate(value)
                if not is_valid:
                    results.append(ValidationResult(
                        field_name=field.name,
                        position=(field.start_pos, field.end_pos),
                        expected=field.description or f"{field.field_type.value}",
                        actual=value,
                        message=msg,
                        severity=ValidationSeverity.ERROR,
                        rule_type="field"
                    ))

                # Padding check
                if field.padding.value != "none":
                    expected_padded = field.pad(value.strip())
                    if value != expected_padded:
                        results.append(ValidationResult(
                            field_name=field.name,
                            position=(field.start_pos, field.end_pos),
                            expected=expected_padded,
                            actual=value,
                            message=f"Incorrect padding for field '{field.name}'",
                            severity=ValidationSeverity.WARNING,
                            rule_type="padding"
                        ))

        # SLM-based validation if model is available
        if self.model is not None and self.tokenizer is not None:
            slm_results = self._slm_validate(record, record_spec)
            results.extend(slm_results)

        status = ValidationStatus.VALID if not any(
            r.severity == ValidationSeverity.ERROR for r in results
        ) else ValidationStatus.INVALID

        return RecordValidation(
            record_type=record_type,
            record_number=record_number,
            record_text=record,
            status=status,
            results=results,
            parsed_fields=parsed_fields
        )

    def _detect_record_type(self, record: str, spec) -> Optional[str]:
        """Detect record type from the record's first field."""
        if not record:
            return None

        # Try first character for single-char type codes
        first_char = record[0]
        if spec.get_record_spec(first_char):
            return first_char

        # Try first 3 characters for multi-char type codes
        if len(record) >= 3:
            first_three = record[:3]
            if spec.get_record_spec(first_three):
                return first_three

        return first_char

    def _validate_file_level(
        self,
        lines: List[str],
        spec,
        record_counts: Dict[str, int]
    ) -> List[str]:
        """Perform file-level validations."""
        errors = []

        # Check record order
        if spec.record_order_rules:
            actual_order = []
            for line in lines:
                rt = self._detect_record_type(line, spec)
                if rt:
                    actual_order.append(rt)

            # Simple check: first and last records
            if actual_order and actual_order[0] != spec.record_order_rules[0]:
                errors.append(f"First record should be '{spec.record_order_rules[0]}', got '{actual_order[0]}'")

            if len(actual_order) > 1 and actual_order[-1] != spec.record_order_rules[-1]:
                errors.append(f"Last record should be '{spec.record_order_rules[-1]}', got '{actual_order[-1]}'")

        # Check mandatory records
        for rt, rs in spec.record_specs.items():
            if rs.mandatory and record_counts.get(rt, 0) == 0:
                errors.append(f"Mandatory record type '{rt}' ({rs.name}) is missing")

        return errors

    def _verify_checksums(self, lines: List[str], spec) -> bool:
        """Verify file-level checksums and totals."""
        # ACH-specific checksum verification
        if spec.spec_id == "ach_nacha":
            return self._verify_ach_checksums(lines, spec)
        return True

    def _verify_ach_checksums(self, lines: List[str], spec) -> bool:
        """Verify ACH file control totals."""
        try:
            batch_headers = []
            batch_controls = []
            entry_details = []
            file_control = None

            for line in lines:
                if line.startswith('5'):
                    batch_headers.append(line)
                elif line.startswith('8'):
                    batch_controls.append(line)
                elif line.startswith('6'):
                    entry_details.append(line)
                elif line.startswith('9'):
                    file_control = line

            if not file_control:
                return False

            # Verify batch count
            batch_count = int(file_control[1:7])
            if batch_count != len(batch_headers):
                return False

            # Verify entry/addenda count
            entry_count = int(file_control[13:21])
            if entry_count != len(entry_details):
                return False

            return True
        except (ValueError, IndexError):
            return False

    def _slm_validate(
        self,
        record: str,
        record_spec
    ) -> List[ValidationResult]:
        """Use the SLM model for additional validation."""
        results = []

        try:
            import torch
            tokens = self.tokenizer.encode(record, record_type=record_spec.record_type_id)
            input_ids = torch.tensor([tokens], dtype=torch.long)

            with torch.no_grad():
                outputs = self.model(
                    input_ids,
                    record_type_ids=torch.tensor([record_spec.record_type_id]),
                    return_validation=True
                )

                val_logits = outputs['validation_logits']
                pred = val_logits.argmax(dim=-1).item()

                # If model predicts invalid (class 1 or 2)
                if pred != 0:
                    severity = ValidationSeverity.WARNING if pred == 2 else ValidationSeverity.ERROR
                    results.append(ValidationResult(
                        field_name="SLM_SyntaxCheck",
                        position=(0, len(record)),
                        expected="Valid",
                        actual="Invalid",
                        message="SLM model detected potential syntax anomaly",
                        severity=severity,
                        rule_type="slm"
                    ))
        except Exception:
            pass

        return results

    def format_report(self, validation: FileValidation) -> str:
        """Format validation results as a human-readable report."""
        lines = []
        lines.append("=" * 80)
        lines.append(f"VALIDATION REPORT: {validation.filename}")
        lines.append(f"Specification: {validation.spec_id}")
        lines.append(f"Overall Status: {validation.overall_status.value.upper()}")
        lines.append("=" * 80)

        summary = validation.summary
        lines.append(f"\nTotal Records: {summary['total_records']}")
        lines.append(f"Total Errors: {summary['total_errors']}")
        lines.append(f"Total Warnings: {summary['total_warnings']}")
        lines.append(f"Checksum Valid: {summary['checksum_valid']}")

        lines.append(f"\nRecord Counts: {summary['record_counts']}")

        if summary.get('file_level_errors'):
            lines.append("\n--- FILE LEVEL ERRORS ---")
            for error in summary['file_level_errors']:
                lines.append(f"  [ERROR] {error}")

        lines.append("\n--- RECORD DETAILS ---")
        for rec in validation.records:
            if rec.status != ValidationStatus.VALID:
                lines.append(f"\nRecord #{rec.record_number} [Type: {rec.record_type}] - {rec.status.value.upper()}")
                for result in rec.results:
                    icon = "❌" if result.severity == ValidationSeverity.ERROR else "⚠️"
                    lines.append(f"  {icon} [{result.rule_type}] {result.field_name}: {result.message}")
                    lines.append(f"      Expected: '{result.expected}' | Actual: '{result.actual}'")

        lines.append("\n" + "=" * 80)
        return "\n".join(lines)
