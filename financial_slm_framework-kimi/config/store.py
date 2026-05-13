"""
financial_slm_framework/config/store.py
High-performance in-memory configuration engine.
Acts as the "Source of Truth" for all financial specification rules.
Uses a singleton pattern with nested dictionary structure for O(1) lookups.
"""

from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import threading
import json
import copy


class FieldType(Enum):
    NUMERIC = "numeric"
    ALPHANUMERIC = "alphanumeric"
    ALPHABETIC = "alphabetic"
    DATE = "date"           # YYMMDD format
    TIME = "time"           # HHMM format
    DECIMAL = "decimal"     # Implied decimal
    ROUTING = "routing"     # 9-digit routing number
    ACCOUNT = "account"     # Account number
    CURRENCY = "currency"   # Amount in cents
    BLANK = "blank"         # Must be space-filled
    CONSTANT = "constant"   # Fixed value


class PaddingType(Enum):
    LEFT_ZERO = "left_zero"      # 000123
    RIGHT_ZERO = "right_zero"    # 123000
    LEFT_SPACE = "left_space"    # "   123"
    RIGHT_SPACE = "right_space"  # "123   "
    NONE = "none"


@dataclass
class FieldRule:
    """Defines a single field within a financial record."""
    name: str
    start_pos: int                    # 0-indexed start position
    end_pos: int                      # 0-indexed end position (exclusive)
    field_type: FieldType
    length: int
    padding: PaddingType = PaddingType.NONE
    required: bool = True
    allowed_values: Optional[List[str]] = None
    default_value: Optional[str] = None
    description: str = ""
    validation_regex: Optional[str] = None
    checksum_field: bool = False      # Whether this field participates in checksum

    def validate(self, value: str) -> Tuple[bool, str]:
        """Validate a field value against its rules."""
        if not isinstance(value, str):
            value = str(value)

        # Length check
        if len(value) != self.length:
            return False, f"Field '{self.name}': Expected length {self.length}, got {len(value)}"

        # Required check
        if self.required and value.strip() == "":
            return False, f"Field '{self.name}': Required field is empty"

        # Type validation
        if self.field_type == FieldType.NUMERIC:
            if not value.isdigit():
                return False, f"Field '{self.name}': Expected numeric, got '{value}'"
        elif self.field_type == FieldType.ALPHABETIC:
            if not value.isalpha():
                return False, f"Field '{self.name}': Expected alphabetic, got '{value}'"
        elif self.field_type == FieldType.ALPHANUMERIC:
            if not value.isalnum():
                return False, f"Field '{self.name}': Expected alphanumeric, got '{value}'"
        elif self.field_type == FieldType.BLANK:
            if value.strip() != "":
                return False, f"Field '{self.name}': Expected blank, got '{value}'"
        elif self.field_type == FieldType.ROUTING:
            if not (value.isdigit() and len(value) == 9):
                return False, f"Field '{self.name}': Expected 9-digit routing number"
        elif self.field_type == FieldType.DATE:
            if not (len(value) == 6 and value.isdigit()):
                return False, f"Field '{self.name}': Expected YYMMDD date format"

        # Allowed values check
        if self.allowed_values is not None and value not in self.allowed_values:
            return False, f"Field '{self.name}': Value '{value}' not in allowed values"

        # Regex check
        if self.validation_regex is not None:
            import re
            if not re.match(self.validation_regex, value):
                return False, f"Field '{self.name}': Value '{value}' failed regex validation"

        return True, "OK"

    def pad(self, value: str) -> str:
        """Pad a value according to the field's padding rules."""
        if len(value) >= self.length:
            return value[:self.length]

        if self.padding == PaddingType.LEFT_ZERO:
            return value.zfill(self.length)
        elif self.padding == PaddingType.RIGHT_ZERO:
            return value.ljust(self.length, '0')
        elif self.padding == PaddingType.LEFT_SPACE:
            return value.rjust(self.length, ' ')
        elif self.padding == PaddingType.RIGHT_SPACE:
            return value.ljust(self.length, ' ')
        else:
            return value.ljust(self.length, ' ')


@dataclass
class RecordSpec:
    """Defines a complete record specification (e.g., ACH File Header)."""
    record_type_code: str
    record_type_id: int
    name: str
    description: str
    fields: List[FieldRule] = field(default_factory=list)
    total_length: int = 0
    mandatory: bool = True
    max_occurrences: Optional[int] = None

    def __post_init__(self):
        if self.total_length == 0 and self.fields:
            self.total_length = max(f.end_pos for f in self.fields)

    def get_field(self, name: str) -> Optional[FieldRule]:
        for field in self.fields:
            if field.name == name:
                return field
        return None

    def validate_record(self, record: str) -> Tuple[bool, List[str]]:
        """Validate an entire record against this specification."""
        errors = []

        if len(record) != self.total_length:
            errors.append(f"Record length mismatch: Expected {self.total_length}, got {len(record)}")

        for field in self.fields:
            value = record[field.start_pos:field.end_pos]
            is_valid, msg = field.validate(value)
            if not is_valid:
                errors.append(msg)

        return len(errors) == 0, errors

    def parse_record(self, record: str) -> Dict[str, str]:
        """Parse a record into a dictionary of field values."""
        result = {}
        for field in self.fields:
            result[field.name] = record[field.start_pos:field.end_pos]
        return result

    def build_record(self, field_values: Dict[str, str]) -> str:
        """Build a record from field values, applying padding."""
        record_chars = [' '] * self.total_length

        for field in self.fields:
            value = field_values.get(field.name, field.default_value or '')
            padded = field.pad(value)
            record_chars[field.start_pos:field.end_pos] = list(padded)

        return "".join(record_chars)


@dataclass
class FileSpec:
    """Defines a complete file format specification."""
    spec_id: str
    name: str
    description: str
    version: str
    record_specs: Dict[str, RecordSpec] = field(default_factory=dict)
    record_order_rules: List[str] = field(default_factory=list)
    file_level_validations: List[Callable] = field(default_factory=list)

    def get_record_spec(self, record_type_code: str) -> Optional[RecordSpec]:
        return self.record_specs.get(record_type_code)

    def add_record_spec(self, spec: RecordSpec):
        self.record_specs[spec.record_type_code] = spec

    def to_dict(self) -> Dict:
        return {
            'spec_id': self.spec_id,
            'name': self.name,
            'description': self.description,
            'version': self.version,
            'record_specs': {
                k: {
                    'record_type_code': v.record_type_code,
                    'record_type_id': v.record_type_id,
                    'name': v.name,
                    'description': v.description,
                    'total_length': v.total_length,
                    'mandatory': v.mandatory,
                    'fields': [asdict(f) for f in v.fields]
                }
                for k, v in self.record_specs.items()
            },
            'record_order_rules': self.record_order_rules
        }


class SpecificationStore:
    """
    Singleton in-memory store for all financial specifications.
    Thread-safe, high-performance dictionary-based storage.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self._specs: Dict[str, FileSpec] = {}
        self._record_type_map: Dict[str, Dict[str, int]] = {}
        self._field_rules_cache: Dict[str, Dict[str, FieldRule]] = {}
        self._lock = threading.RLock()

    def register_spec(self, spec: FileSpec):
        """Register a file specification."""
        with self._lock:
            self._specs[spec.spec_id] = spec
            self._record_type_map[spec.spec_id] = {
                rs.record_type_code: rs.record_type_id
                for rs in spec.record_specs.values()
            }
            # Cache field rules for O(1) lookup
            self._field_rules_cache[spec.spec_id] = {}
            for rs in spec.record_specs.values():
                for field in rs.fields:
                    cache_key = f"{rs.record_type_code}:{field.name}"
                    self._field_rules_cache[spec.spec_id][cache_key] = field

    def get_spec(self, spec_id: str) -> Optional[FileSpec]:
        """Get a file specification by ID."""
        with self._lock:
            return self._specs.get(spec_id)

    def get_record_type_id(self, spec_id: str, record_type_code: str) -> Optional[int]:
        """Get the numeric record type ID for a specification."""
        with self._lock:
            return self._record_type_map.get(spec_id, {}).get(record_type_code)

    def get_field_rule(self, spec_id: str, record_type_code: str, field_name: str) -> Optional[FieldRule]:
        """Get a specific field rule."""
        with self._lock:
            cache_key = f"{record_type_code}:{field_name}"
            return self._field_rules_cache.get(spec_id, {}).get(cache_key)

    def list_specs(self) -> List[str]:
        """List all registered specification IDs."""
        with self._lock:
            return list(self._specs.keys())

    def get_all_specs(self) -> Dict[str, FileSpec]:
        """Get all registered specifications."""
        with self._lock:
            return copy.deepcopy(self._specs)

    def export_to_json(self, path: str):
        """Export all specifications to JSON."""
        with self._lock:
            export_data = {
                spec_id: spec.to_dict()
                for spec_id, spec in self._specs.items()
            }
        with open(path, 'w') as f:
            json.dump(export_data, f, indent=2)

    def clear(self):
        """Clear all specifications."""
        with self._lock:
            self._specs.clear()
            self._record_type_map.clear()
            self._field_rules_cache.clear()


# Global singleton instance
spec_store = SpecificationStore()
