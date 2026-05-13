"""
financial_slm_framework/generation/generator.py
Constrained generation module for producing syntactically correct test files.
Uses auto-regressive decoding with specification-based constraints.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Callable
import random
import string
from datetime import datetime, timedelta


class ConstraintFunction:
    """Base class for generation constraints."""

    def __call__(self, logits: torch.Tensor, position: int, spec, record_spec) -> torch.Tensor:
        raise NotImplementedError


class FixedWidthConstraint(ConstraintFunction):
    """Enforce fixed-width field constraints during generation."""

    def __init__(self, tokenizer, record_spec):
        self.tokenizer = tokenizer
        self.record_spec = record_spec
        self.field_map = self._build_field_map()

    def _build_field_map(self) -> Dict[int, 'FieldRule']:
        """Map character positions to their field rules."""
        field_map = {}
        for field in self.record_spec.fields:
            for pos in range(field.start_pos, field.end_pos):
                field_map[pos] = field
        return field_map

    def __call__(self, logits: torch.Tensor, position: int, spec, record_spec) -> torch.Tensor:
        # Account for special tokens (SOS + RT)
        char_position = position - 2

        if char_position < 0:
            return logits

        field = self.field_map.get(char_position)
        if field is None:
            return logits

        # Create mask of allowed characters
        allowed_chars = self._get_allowed_chars(field)

        # Mask out disallowed characters
        mask = torch.full_like(logits, float('-inf'))
        for char in allowed_chars:
            if char in self.tokenizer.vocab:
                mask[self.tokenizer.vocab[char]] = 0

        return logits + mask

    def _get_allowed_chars(self, field) -> str:
        """Get allowed characters for a field type."""
        from config.store import FieldType

        if field.field_type == FieldType.NUMERIC or field.field_type == FieldType.CURRENCY:
            return "0123456789"
        elif field.field_type == FieldType.ALPHABETIC:
            return "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        elif field.field_type == FieldType.ALPHANUMERIC:
            return "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        elif field.field_type == FieldType.BLANK:
            return " "
        elif field.field_type == FieldType.DATE:
            return "0123456789"
        elif field.field_type == FieldType.TIME:
            return "0123456789"
        elif field.field_type == FieldType.ROUTING:
            return "0123456789"
        elif field.field_type == FieldType.DECIMAL:
            return "0123456789"
        else:
            return string.printable


class MockDataSeeder:
    """Generates randomized yet valid mock data for financial fields."""

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)

    def generate_field_value(self, field, context: Optional[Dict] = None) -> str:
        """Generate a valid mock value for a field."""
        from config.store import FieldType

        # Use default if available
        if field.default_value is not None:
            return field.default_value

        # Use allowed values if restricted
        if field.allowed_values is not None:
            return random.choice(field.allowed_values)

        generators = {
            FieldType.NUMERIC: self._gen_numeric,
            FieldType.ALPHANUMERIC: self._gen_alphanumeric,
            FieldType.ALPHABETIC: self._gen_alphabetic,
            FieldType.DATE: self._gen_date,
            FieldType.TIME: self._gen_time,
            FieldType.DECIMAL: self._gen_decimal,
            FieldType.ROUTING: self._gen_routing,
            FieldType.ACCOUNT: self._gen_account,
            FieldType.CURRENCY: self._gen_currency,
            FieldType.BLANK: self._gen_blank,
            FieldType.CONSTANT: lambda f, c: f.default_value or "",
        }

        generator = generators.get(field.field_type, self._gen_alphanumeric)
        value = generator(field, context)

        # Apply padding
        return field.pad(value)

    def _gen_numeric(self, field, context) -> str:
        length = field.length
        if field.field_type.name == "CURRENCY":
            # Generate amount in cents
            max_amount = 10 ** length - 1
            amount = random.randint(0, min(max_amount, 99999999))
            return str(amount).zfill(length)
        return str(random.randint(0, 10 ** length - 1)).zfill(length)[:length]

    def _gen_alphanumeric(self, field, context) -> str:
        chars = string.ascii_uppercase + string.digits
        return ''.join(random.choices(chars, k=field.length))

    def _gen_alphabetic(self, field, context) -> str:
        return ''.join(random.choices(string.ascii_uppercase, k=field.length))

    def _gen_date(self, field, context) -> str:
        days_back = random.randint(0, 365)
        date = datetime.now() - timedelta(days=days_back)
        return date.strftime("%y%m%d")

    def _gen_time(self, field, context) -> str:
        return f"{random.randint(0, 23):02d}{random.randint(0, 59):02d}"

    def _gen_decimal(self, field, context) -> str:
        # 2 implied decimals
        max_val = 10 ** field.length - 1
        val = random.randint(0, min(max_val, 999999999))
        return str(val).zfill(field.length)

    def _gen_routing(self, field, context) -> str:
        # 9-digit routing number with valid check digit
        first_8 = str(random.randint(10000000, 99999999))
        check = self._calculate_routing_check_digit(first_8)
        return first_8 + str(check)

    def _gen_account(self, field, context) -> str:
        length = random.randint(4, field.length)
        return str(random.randint(10 ** (length - 1), 10 ** length - 1))

    def _gen_currency(self, field, context) -> str:
        max_cents = min(10 ** field.length - 1, 99999999)
        cents = random.randint(0, max_cents)
        return str(cents).zfill(field.length)

    def _gen_blank(self, field, context) -> str:
        return " " * field.length

    @staticmethod
    def _calculate_routing_check_digit(first_8: str) -> int:
        """Calculate ACH routing number check digit."""
        weights = [3, 7, 1, 3, 7, 1, 3, 7]
        total = sum(int(d) * w for d, w in zip(first_8, weights))
        return (10 - (total % 10)) % 10


class FinancialGenerator:
    """
    Constrained generation engine for financial test files.
    Supports both rule-based generation and SLM-guided generation.
    """

    def __init__(self, spec_store, model=None, tokenizer=None, device='cpu'):
        self.spec_store = spec_store
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.seeder = MockDataSeeder()

    def generate_file(
        self,
        spec_id: str,
        num_records: int = 10,
        use_slm: bool = True,
        seed: Optional[int] = None
    ) -> str:
        """
        Generate a complete test file.

        Args:
            spec_id: Specification to generate for
            num_records: Number of detail records to generate
            use_slm: Whether to use SLM for generation (vs rule-based)
            seed: Random seed for reproducibility

        Returns:
            Generated file content as string
        """
        if seed is not None:
            self.seeder = MockDataSeeder(seed)

        spec = self.spec_store.get_spec(spec_id)
        if spec is None:
            raise ValueError(f"Specification '{spec_id}' not found")

        records = []

        # Generate header record(s)
        for rt_code in spec.record_order_rules:
            rs = spec.get_record_spec(rt_code)
            if rs and "header" in rs.name.lower():
                records.append(self._generate_record(rs, spec, use_slm))

        # Generate detail records
        detail_specs = [rs for rs in spec.record_specs.values() 
                       if "detail" in rs.name.lower() or "entry" in rs.name.lower()]

        for _ in range(num_records):
            for rs in detail_specs:
                records.append(self._generate_record(rs, spec, use_slm))

        # Generate trailer/control records
        for rt_code in spec.record_order_rules:
            rs = spec.get_record_spec(rt_code)
            if rs and ("trailer" in rs.name.lower() or "control" in rs.name.lower()):
                records.append(self._generate_record(rs, spec, use_slm))

        # Post-process: update totals and checksums
        records = self._update_totals(records, spec)

        return "\n".join(records)

    def _generate_record(
        self,
        record_spec,
        spec,
        use_slm: bool
    ) -> str:
        """Generate a single record."""
        if use_slm and self.model is not None and self.tokenizer is not None:
            return self._generate_with_slm(record_spec, spec)
        else:
            return self._generate_rule_based(record_spec, spec)

    def _generate_rule_based(self, record_spec, spec) -> str:
        """Generate record using specification rules and mock data."""
        field_values = {}

        for field in record_spec.fields:
            field_values[field.name] = self.seeder.generate_field_value(field, field_values)

        return record_spec.build_record(field_values)

    def _generate_with_slm(self, record_spec, spec) -> str:
        """Generate record using SLM with constraints."""
        import torch

        # Create constraint function
        constraint = FixedWidthConstraint(self.tokenizer, record_spec)

        # Start with SOS + record type token
        prompt = torch.tensor([[self.tokenizer.SOS_ID]], dtype=torch.long, device=self.device)

        # Generate
        generated = self.model.generate(
            prompt,
            record_type_id=record_spec.record_type_id,
            max_length=record_spec.total_length + 2,
            temperature=0.8,
            constraint_fn=lambda logits, pos: constraint(logits, pos, spec, record_spec),
            eos_token_id=self.tokenizer.EOS_ID
        )

        # Decode
        token_ids = generated[0].tolist()
        record_text, _ = self.tokenizer.decode_fixed_width_record(token_ids, [])

        # Pad/truncate to correct length
        if len(record_text) < record_spec.total_length:
            record_text = record_text.ljust(record_spec.total_length)
        elif len(record_text) > record_spec.total_length:
            record_text = record_text[:record_spec.total_length]

        return record_text

    def _update_totals(self, records: List[str], spec) -> List[str]:
        """Update control record totals based on detail records."""
        if spec.spec_id == "ach_nacha":
            return self._update_ach_totals(records, spec)
        elif spec.spec_id == "visa_vcf":
            return self._update_vcf_totals(records, spec)
        elif spec.spec_id == "general_ledger":
            return self._update_gl_totals(records, spec)
        return records

    def _update_ach_totals(self, records: List[str], spec) -> List[str]:
        """Update ACH batch and file control totals."""
        entry_count = 0
        total_debit = 0
        total_credit = 0
        entry_hash = 0

        for record in records:
            if record.startswith('6'):
                entry_count += 1
                amount = int(record[29:39])
                trans_code = record[1:3]

                if trans_code in ['27', '37']:
                    total_debit += amount
                else:
                    total_credit += amount

                receiving_dfi = record[3:12]
                entry_hash += int(receiving_dfi)

        # Update file control
        updated_records = []
        for record in records:
            if record.startswith('9'):
                # Update entry/addenda count
                record = record[:13] + str(entry_count).zfill(8) + record[21:]
                # Update entry hash (last 10 digits)
                hash_str = str(entry_hash)[-10:].zfill(10)
                record = record[:21] + hash_str + record[31:]
                # Update debit total
                record = record[:31] + str(total_debit).zfill(12) + record[43:]
                # Update credit total
                record = record[:43] + str(total_credit).zfill(12) + record[55:]
            updated_records.append(record)

        return updated_records

    def _update_vcf_totals(self, records: List[str], spec) -> List[str]:
        """Update VCF trailer totals."""
        detail_count = 0
        total_amount = 0

        for record in records:
            if record.startswith('D'):
                detail_count += 1
                amount = int(record[31:43])
                total_amount += amount

        updated_records = []
        for record in records:
            if record.startswith('T'):
                record = record[:1] + str(detail_count).zfill(8) + record[9:]
                record = record[:9] + str(total_amount).zfill(12) + record[21:]
            updated_records.append(record)

        return updated_records

    def _update_gl_totals(self, records: List[str], spec) -> List[str]:
        """Update GL trailer totals."""
        detail_count = 0
        total_debits = 0
        total_credits = 0

        for record in records:
            if record.startswith('DET'):
                detail_count += 1
                debit = int(record[24:39])
                credit = int(record[39:54])
                total_debits += debit
                total_credits += credit

        updated_records = []
        for record in records:
            if record.startswith('TRL'):
                record = record[:3] + str(detail_count).zfill(9) + record[12:]
                record = record[:12] + str(total_debits).zfill(15) + record[27:]
                record = record[:27] + str(total_credits).zfill(15) + record[42:]
                net = total_debits - total_credits
                record = record[:42] + str(abs(net)).zfill(15) + record[57:]
            updated_records.append(record)

        return updated_records
