"""
financial_slm_framework/tests/test_framework.py
Unit tests for the Financial SLM Framework.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from config import initialize_all_specs, spec_store
from config.store import FileSpec, RecordSpec, FieldRule, FieldType, PaddingType
from validation import FinancialValidator
from generation import FinancialGenerator, MockDataSeeder
from slm_core.tokenizer import FinancialTokenizer
from slm_core.model import FinancialSLM
import torch


class TestSpecificationStore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        initialize_all_specs()

    def test_specs_loaded(self):
        specs = spec_store.list_specs()
        self.assertIn("ach_nacha", specs)
        self.assertIn("visa_vcf", specs)
        self.assertIn("general_ledger", specs)

    def test_ach_file_header(self):
        ach = spec_store.get_spec("ach_nacha")
        fh = ach.get_record_spec("1")
        self.assertEqual(fh.total_length, 94)
        self.assertEqual(len(fh.fields), 13)

    def test_field_lookup(self):
        rule = spec_store.get_field_rule("ach_nacha", "1", "RecordTypeCode")
        self.assertIsNotNone(rule)
        self.assertEqual(rule.field_type, FieldType.NUMERIC)
        self.assertEqual(rule.length, 1)

    def test_record_type_mapping(self):
        rt_id = spec_store.get_record_type_id("ach_nacha", "6")
        self.assertEqual(rt_id, 3)


class TestTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FinancialTokenizer(max_record_types=50)

    def test_vocab_size(self):
        self.assertGreater(self.tokenizer.vocab_size, 200)

    def test_encode_decode(self):
        text = "101 091000019123456789"
        tokens = self.tokenizer.encode(text)
        decoded = self.tokenizer.decode(tokens)
        self.assertIn("101", decoded)

    def test_fixed_width_encoding(self):
        record = "101 091000019"
        boundaries = [(0, 1), (1, 3), (3, 13)]
        tokens = self.tokenizer.encode_fixed_width_record(record, boundaries, 1)
        self.assertGreater(len(tokens), 0)
        decoded, fields = self.tokenizer.decode_fixed_width_record(tokens, [])
        self.assertEqual(len(fields), 3)


class TestValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        initialize_all_specs()
        cls.validator = FinancialValidator(spec_store=spec_store)

    def test_valid_ach_file(self):
        # Generate a valid file first
        generator = FinancialGenerator(spec_store=spec_store)
        content = generator.generate_file("ach_nacha", num_records=1, use_slm=False, seed=42)

        result = self.validator.validate_file(content, "ach_nacha", "test.txt")
        self.assertIn(result.overall_status.value, ["valid", "partial"])

    def test_invalid_ach_file(self):
        invalid = "INVALID_RECORD_TOO_SHORT"
        result = self.validator.validate_file(invalid, "ach_nacha", "bad.txt")
        self.assertEqual(result.overall_status.value, "invalid")


class TestGeneration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        initialize_all_specs()
        cls.generator = FinancialGenerator(spec_store=spec_store)

    def test_ach_generation(self):
        content = self.generator.generate_file("ach_nacha", num_records=2, use_slm=False, seed=42)
        lines = content.strip().split("\n")
        self.assertGreater(len(lines), 0)
        # Should have header + entries + controls
        self.assertGreaterEqual(len(lines), 4)

    def test_vcf_generation(self):
        content = self.generator.generate_file("visa_vcf", num_records=2, use_slm=False, seed=42)
        self.assertIn("H", content)
        self.assertIn("D", content)
        self.assertIn("T", content)

    def test_gl_generation(self):
        content = self.generator.generate_file("general_ledger", num_records=2, use_slm=False, seed=42)
        self.assertIn("HDR", content)
        self.assertIn("DET", content)
        self.assertIn("TRL", content)


class TestMockDataSeeder(unittest.TestCase):
    def setUp(self):
        self.seeder = MockDataSeeder(seed=42)

    def test_numeric_generation(self):
        field = FieldRule("Test", 0, 5, FieldType.NUMERIC, 5, padding=PaddingType.LEFT_ZERO)
        value = self.seeder.generate_field_value(field)
        self.assertEqual(len(value), 5)
        self.assertTrue(value.isdigit())

    def test_routing_generation(self):
        field = FieldRule("Routing", 0, 9, FieldType.ROUTING, 9)
        value = self.seeder.generate_field_value(field)
        self.assertEqual(len(value), 9)
        self.assertTrue(value.isdigit())


class TestModel(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FinancialTokenizer(max_record_types=50)
        self.model = FinancialSLM(
            vocab_size=self.tokenizer.vocab_size,
            d_model=64,
            n_layers=2,
            n_heads=2,
            d_ff=128,
            max_seq_len=128
        )

    def test_forward_pass(self):
        tokens = self.tokenizer.encode("101 091000019", record_type=1)
        input_ids = torch.tensor([tokens], dtype=torch.long)

        outputs = self.model(input_ids, record_type_ids=torch.tensor([1]), return_validation=True)

        self.assertIn("generation_logits", outputs)
        self.assertIn("validation_logits", outputs)
        self.assertEqual(outputs["generation_logits"].shape[0], 1)
        self.assertEqual(outputs["validation_logits"].shape[0], 1)

    def test_generation(self):
        prompt = torch.tensor([[self.tokenizer.SOS_ID]], dtype=torch.long)
        generated = self.model.generate(prompt, record_type_id=1, max_length=10, temperature=1.0)
        self.assertGreater(generated.shape[1], 1)


class TestFieldRule(unittest.TestCase):
    def test_numeric_validation(self):
        field = FieldRule("Amount", 0, 10, FieldType.NUMERIC, 10)
        valid, msg = field.validate("0000123456")
        self.assertTrue(valid)

        invalid, msg = field.validate("ABC123")
        self.assertFalse(invalid)

    def test_padding(self):
        field = FieldRule("Code", 0, 5, FieldType.NUMERIC, 5, padding=PaddingType.LEFT_ZERO)
        padded = field.pad("123")
        self.assertEqual(padded, "00123")

    def test_allowed_values(self):
        field = FieldRule("Type", 0, 1, FieldType.NUMERIC, 1, allowed_values=["1", "5", "9"])
        valid, _ = field.validate("5")
        self.assertTrue(valid)

        invalid, _ = field.validate("3")
        self.assertFalse(invalid)


if __name__ == "__main__":
    unittest.main()
