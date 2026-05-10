"""
ConfigEngine — In-Memory Specification Source of Truth.

Implements the Singleton pattern over a high-performance Python dict
structure. No external dependencies (no Redis required; can be swapped
for Redis by replacing the _store backend).

Architecture:
  ┌─────────────────────────────────────────────────────────┐
  │                     ConfigEngine                         │
  │                     (Singleton)                          │
  │                                                          │
  │  _store: {                                               │
  │    "ACH_NACHA": {                                        │
  │      "meta": { line_length, blocking_factor, … }         │
  │      "record_types": ["RT1","RT5","RT6","RT8","RT9"]     │
  │      "fields": {                                         │
  │        "RT6": [                                          │
  │          { name, start, end, field_type,                 │
  │            length, required, pattern, allowed }          │
  │        ]                                                 │
  │      }                                                   │
  │    }                                                     │
  │  }                                                       │
  └─────────────────────────────────────────────────────────┘

The SLM queries ConfigEngine at inference time; the Validator and
Generator also use it as their canonical rule source.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional


class _ConfigStore:
    """Thread-safe in-memory key-value store (dict backend)."""

    def __init__(self):
        self._data : Dict[str, Any] = {}
        self._lock = threading.RLock()

    def set(self, key: str, value: Any):
        with self._lock:
            self._data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.get(key, default)

    def delete(self, key: str):
        with self._lock:
            self._data.pop(key, None)

    def keys(self) -> List[str]:
        with self._lock:
            return list(self._data.keys())

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._data


class ConfigEngine:
    """
    Singleton configuration engine.

    Usage:
        engine = ConfigEngine()               # always same instance
        engine.get_fields("ACH_NACHA", "RT6")
        engine.get_meta("ACH_NACHA")
        engine.set_custom_rule("ACH_NACHA", "RT6", "Amount", {"max": 9999999999})
    """

    _instance : Optional["ConfigEngine"] = None
    _lock      = threading.Lock()

    def __new__(cls) -> "ConfigEngine":
        with cls._lock:
            if cls._instance is None:
                inst = super().__new__(cls)
                inst._store    = _ConfigStore()
                inst._custom   = _ConfigStore()   # overlay for runtime overrides
                inst._loaded   : set = set()
                cls._instance  = inst
                inst._bootstrap()
        return cls._instance

    # ── Bootstrap ─────────────────────────────────────────────────────────

    def _bootstrap(self):
        """Load all three spec schemas into the store on first init."""
        from specs.ach_nacha      import ACH_FIELD_SCHEMA, ACH_SPEC_META
        from specs.visa_vcf       import VISA_FIELD_SCHEMA, VISA_SPEC_META, GL_FIELD_SCHEMA, GL_SPEC_META

        self._register_spec("ACH_NACHA",      ACH_FIELD_SCHEMA,  ACH_SPEC_META)
        self._register_spec("VISA_VCF",       VISA_FIELD_SCHEMA, VISA_SPEC_META)
        self._register_spec("GENERAL_LEDGER", GL_FIELD_SCHEMA,   GL_SPEC_META)

    def _register_spec(self, spec_name: str, schema: Dict, meta: Dict):
        """
        Convert FieldDescriptor objects into plain dicts for storage.
        Plain dicts are easier to serialise, introspect, and override.
        """
        field_map: Dict[str, List[Dict]] = {}
        for rt, descriptors in schema.items():
            field_map[rt] = [
                {
                    "name"      : fd.name,
                    "start"     : fd.start,
                    "end"       : fd.end,
                    "length"    : fd.length,
                    "field_type": fd.field_type.value,
                    "required"  : fd.required,
                    "pattern"   : fd.pattern,
                    "allowed"   : fd.allowed,
                    "descriptor": fd,   # keep original for extract()
                }
                for fd in descriptors
            ]

        self._store.set(f"{spec_name}:meta",   meta)
        self._store.set(f"{spec_name}:fields", field_map)
        self._store.set(f"{spec_name}:rts",    list(schema.keys()))
        self._loaded.add(spec_name)

    # ── Public Query API ──────────────────────────────────────────────────

    def get_meta(self, spec_name: str) -> Dict:
        return self._store.get(f"{spec_name}:meta", {})

    def get_record_types(self, spec_name: str) -> List[str]:
        return self._store.get(f"{spec_name}:rts", [])

    def get_fields(self, spec_name: str, record_type: str) -> List[Dict]:
        """
        Returns field definitions for a given spec + record type.
        Merges custom runtime overrides on top of the base spec.
        """
        base   = self._store.get(f"{spec_name}:fields", {}).get(record_type, [])
        custom = self._custom.get(f"{spec_name}:{record_type}:fields", [])
        if not custom:
            return base

        # Overlay: custom fields take precedence by field name
        custom_map = {f["name"]: f for f in custom}
        merged = []
        for fd in base:
            merged.append(custom_map.get(fd["name"], fd))
        # Append any new custom fields not in base
        base_names = {f["name"] for f in base}
        for cf in custom:
            if cf["name"] not in base_names:
                merged.append(cf)
        return merged

    def get_all_specs(self) -> List[Dict]:
        """Return metadata for all registered specs (for UI dropdown)."""
        return [self.get_meta(s) for s in self._loaded]

    def get_field_rule(self, spec: str, rt: str, field_name: str) -> Optional[Dict]:
        for fd in self.get_fields(spec, rt):
            if fd["name"] == field_name:
                return fd
        return None

    def get_line_length(self, spec_name: str) -> int:
        return self.get_meta(spec_name).get("line_length", 94)

    # ── Runtime Override API ──────────────────────────────────────────────

    def set_custom_rule(
        self,
        spec_name   : str,
        record_type : str,
        field_name  : str,
        overrides   : Dict,
    ):
        """
        Inject a runtime rule override for a specific field.

        Example:
            engine.set_custom_rule(
                "ACH_NACHA", "RT6", "Amount",
                {"allowed": ["0000000001", "0000000002"]}
            )
        This constrains the generator to only emit those amounts —
        useful for test-data scenarios.
        """
        key      = f"{spec_name}:{record_type}:fields"
        existing = list(self._custom.get(key, []))
        updated  = False
        for fd in existing:
            if fd["name"] == field_name:
                fd.update(overrides)
                updated = True
                break
        if not updated:
            base = self.get_field_rule(spec_name, record_type, field_name) or {}
            new_fd = {**base, "name": field_name, **overrides}
            existing.append(new_fd)
        self._custom.set(key, existing)

    def reset_custom_rules(self, spec_name: str, record_type: Optional[str] = None):
        """Remove runtime overrides for a spec (or specific record type)."""
        if record_type:
            self._custom.delete(f"{spec_name}:{record_type}:fields")
        else:
            for rt in self.get_record_types(spec_name):
                self._custom.delete(f"{spec_name}:{rt}:fields")

    # ── Introspection ─────────────────────────────────────────────────────

    def describe(self, spec_name: str, record_type: str) -> str:
        """Human-readable field table for a record type."""
        fields = self.get_fields(spec_name, record_type)
        if not fields:
            return f"No schema found for {spec_name}/{record_type}"
        rows = [f"{'Field Name':<35} {'Start':>5} {'End':>4} {'Len':>4} {'Type':<14} {'Req':>4}"]
        rows.append("-" * 75)
        for fd in fields:
            rows.append(
                f"{fd['name']:<35} {fd['start']:>5} {fd['end']:>4} "
                f"{fd['length']:>4} {fd['field_type']:<14} "
                f"{'Yes' if fd['required'] else 'No':>4}"
            )
        return "\n".join(rows)

    def export_rules(self, spec_name: str) -> Dict:
        """
        Export the full rule set for a spec as a JSON-serialisable dict.
        Used by the API to serve rules to the frontend.
        """
        rts = self.get_record_types(spec_name)
        return {
            "spec"  : self.get_meta(spec_name),
            "schema": {
                rt: [
                    {k: v for k, v in fd.items() if k != "descriptor"}
                    for fd in self.get_fields(spec_name, rt)
                ]
                for rt in rts
            },
        }

    def __repr__(self) -> str:
        specs = list(self._loaded)
        return f"ConfigEngine(specs={specs})"
