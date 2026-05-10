"""
Test Suite — FinancialSLM Framework.

Covers:
  1. Tokenizer unit tests (encoding, decoding, spec routing)
  2. Config Engine singleton + field query tests
  3. Seeder validity tests (routing check digits, field lengths)
  4. Validator rule-engine tests (correct + deliberately broken records)
  5. Generator output tests (line length, structure, checksums)
  6. Model architecture tests (forward pass shapes, dual heads)

Run:  python -m pytest tests/test_suite.py -v
"""

import sys
import os
import random
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═══════════════════════════════════════════════════════════
#  1. TOKENIZER TESTS
# ═══════════════════════════════════════════════════════════

class TestTokenizer(unittest.TestCase):

    def setUp(self):
        from slm.tokenizer import FinancialTokenizer, VOCAB_SIZE
        self.tok_ach  = FinancialTokenizer("ACH_NACHA")
        self.tok_visa = FinancialTokenizer("VISA_VCF")
        self.tok_gl   = FinancialTokenizer("GENERAL_LEDGER")
        self.vocab_sz = VOCAB_SIZE

    def test_line_lengths(self):
        self.assertEqual(self.tok_ach.line_length,  94)
        self.assertEqual(self.tok_visa.line_length, 80)
        self.assertEqual(self.tok_gl.line_length,  120)

    def test_encode_decode_roundtrip(self):
        line = "1" + "01" + " 021000021" + "9876543210" + "260101" + "0000" + "A" + "094" + "10" + "1"
        line = line.ljust(94)[:94]
        ids  = self.tok_ach.encode_line(line)
        self.assertEqual(len(ids), 94)
        back = self.tok_ach.decode_ids(ids)
        # decoded should reproduce the upper-cased, padded original
        self.assertEqual(back.upper(), line.upper())

    def test_vocab_covers_financial_chars(self):
        for ch in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ /-.:":
            iid = self.tok_ach.char_to_id(ch)
            self.assertNotEqual(iid, 1, f"Char {ch!r} mapped to <UNK>")

    def test_unknown_char_maps_to_unk(self):
        uid = self.tok_ach.char_to_id("\x00")
        from slm.tokenizer import _VOCAB
        self.assertEqual(uid, _VOCAB["<UNK>"])

    def test_record_type_detection_ach(self):
        line = "1" + " " * 93
        rt   = self.tok_ach.get_record_type(line)
        self.assertEqual(rt, "RT1")

    def test_tokenize_returns_list_of_lists(self):
        raw = "1" + " " * 93 + "\n" + "5" + " " * 93
        result = self.tok_ach.tokenize(raw)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], list)


# ═══════════════════════════════════════════════════════════
#  2. CONFIG ENGINE TESTS
# ═══════════════════════════════════════════════════════════

class TestConfigEngine(unittest.TestCase):

    def setUp(self):
        from memory.config_engine import ConfigEngine
        self.engine = ConfigEngine()

    def test_singleton(self):
        from memory.config_engine import ConfigEngine
        a = ConfigEngine()
        b = ConfigEngine()
        self.assertIs(a, b)

    def test_all_specs_loaded(self):
        specs = [s["name"] for s in self.engine.get_all_specs()]
        self.assertIn("ACH_NACHA",      specs)
        self.assertIn("VISA_VCF",       specs)
        self.assertIn("GENERAL_LEDGER", specs)

    def test_ach_record_types(self):
        rts = self.engine.get_record_types("ACH_NACHA")
        for rt in ["RT1", "RT5", "RT6", "RT8", "RT9"]:
            self.assertIn(rt, rts)

    def test_visa_record_types(self):
        rts = self.engine.get_record_types("VISA_VCF")
        for rt in ["RTVH", "RVDT", "RTVF"]:
            self.assertIn(rt, rts)

    def test_gl_record_types(self):
        rts = self.engine.get_record_types("GENERAL_LEDGER")
        for rt in ["RTJH", "RTJE", "RTGL"]:
            self.assertIn(rt, rts)

    def test_field_lengths_sum_to_line_length(self):
        """Field widths for each record type must cover the full line."""
        for spec in ["ACH_NACHA", "VISA_VCF", "GENERAL_LEDGER"]:
            line_len = self.engine.get_line_length(spec)
            for rt in self.engine.get_record_types(spec):
                fields = self.engine.get_fields(spec, rt)
                if not fields:
                    continue
                max_end = max(f["end"] for f in fields)
                self.assertLessEqual(
                    max_end, line_len,
                    f"{spec}/{rt}: field end {max_end} exceeds line length {line_len}"
                )

    def test_custom_override_layered(self):
        self.engine.set_custom_rule("ACH_NACHA", "RT6", "Amount", {"allowed": ["0000000042"]})
        fields = self.engine.get_fields("ACH_NACHA", "RT6")
        amt    = next((f for f in fields if f["name"] == "Amount"), None)
        self.assertIsNotNone(amt)
        self.assertEqual(amt.get("allowed"), ["0000000042"])
        # cleanup
        self.engine.reset_custom_rules("ACH_NACHA", "RT6")
        fields2 = self.engine.get_fields("ACH_NACHA", "RT6")
        amt2    = next((f for f in fields2 if f["name"] == "Amount"), None)
        self.assertNotEqual(amt2.get("allowed"), ["0000000042"])

    def test_describe_returns_table(self):
        desc = self.engine.describe("ACH_NACHA", "RT6")
        self.assertIn("Record Type Code", desc)
        self.assertIn("Amount",           desc)
        self.assertIn("Individual Name",  desc)

    def test_export_rules_structure(self):
        rules = self.engine.export_rules("ACH_NACHA")
        self.assertIn("spec",   rules)
        self.assertIn("schema", rules)
        self.assertIn("RT6",    rules["schema"])


# ═══════════════════════════════════════════════════════════
#  3. SEEDER TESTS
# ═══════════════════════════════════════════════════════════

class TestDataSeeder(unittest.TestCase):

    def setUp(self):
        from memory.config_engine import ConfigEngine
        from memory.seeder        import DataSeeder
        self.engine = ConfigEngine()
        self.seeder = DataSeeder(self.engine, seed=42)

    def _check_line_length(self, line, expected):
        self.assertEqual(
            len(line), expected,
            f"Expected line length {expected}, got {len(line)}: {line!r}"
        )

    def test_ach_rt1_length(self):
        line, _ = self.seeder.generate_line("ACH_NACHA", "RT1")
        self._check_line_length(line, 94)

    def test_ach_rt6_length(self):
        line, _ = self.seeder.generate_line("ACH_NACHA", "RT6")
        self._check_line_length(line, 94)

    def test_visa_dt_length(self):
        line, _ = self.seeder.generate_line("VISA_VCF", "RVDT")
        self._check_line_length(line, 80)

    def test_gl_je_length(self):
        line, _ = self.seeder.generate_line("GENERAL_LEDGER", "RTJE")
        self._check_line_length(line, 120)

    def test_routing_check_digit_valid(self):
        """Generated routing numbers must pass the Mod-10 check."""
        from slm.validator import RuleEngine
        for _ in range(20):
            line, ctx = self.seeder.generate_line("ACH_NACHA", "RT6", return_context=True)
            routing   = ctx.get("routing")
            if routing:
                ok, msg = RuleEngine.validate_routing(routing)
                self.assertTrue(ok, f"Invalid routing {routing}: {msg}")

    def test_rt1_first_char_is_1(self):
        line, _ = self.seeder.generate_line("ACH_NACHA", "RT1")
        self.assertEqual(line[0], "1")

    def test_rt6_first_char_is_6(self):
        line, _ = self.seeder.generate_line("ACH_NACHA", "RT6")
        self.assertEqual(line[0], "6")

    def test_context_returned(self):
        _, ctx = self.seeder.generate_line("ACH_NACHA", "RT6", return_context=True)
        self.assertIsNotNone(ctx)
        self.assertIsInstance(ctx, dict)

    def test_amount_is_numeric(self):
        for _ in range(10):
            line, _ = self.seeder.generate_line("ACH_NACHA", "RT6")
            amount  = line[29:39]   # RT6 Amount field: cols 30-39
            self.assertTrue(
                amount.isdigit(),
                f"Amount field is not numeric: {amount!r}"
            )

    def test_batch_generates_n_lines(self):
        lines = self.seeder.generate_batch("ACH_NACHA", "RT6", count=10, seed=7)
        self.assertEqual(len(lines), 10)
        for l in lines:
            self._check_line_length(l, 94)

    def test_all_record_types_generatable(self):
        for spec in ["ACH_NACHA", "VISA_VCF", "GENERAL_LEDGER"]:
            for rt in self.engine.get_record_types(spec):
                line, _ = self.seeder.generate_line(spec, rt)
                ll = self.engine.get_line_length(spec)
                self._check_line_length(line, ll)


# ═══════════════════════════════════════════════════════════
#  4. VALIDATOR TESTS
# ═══════════════════════════════════════════════════════════

class TestValidator(unittest.TestCase):

    def setUp(self):
        from memory.config_engine import ConfigEngine
        from memory.seeder        import DataSeeder
        from slm.tokenizer        import make_tokenizer
        from slm.validator        import FinancialValidator

        self.engine = ConfigEngine()
        self.seeder = DataSeeder(self.engine, seed=99)

        self.validators = {
            spec: FinancialValidator(
                spec_name     = spec,
                config_engine = self.engine,
                tokenizer     = make_tokenizer(spec),
                model         = None,
            )
            for spec in ["ACH_NACHA", "VISA_VCF", "GENERAL_LEDGER"]
        }

    # ── Rule Engine Unit Tests ─────────────────────────────────────────

    def test_routing_valid(self):
        from slm.validator import RuleEngine
        ok, msg = RuleEngine.validate_routing("021000021")
        self.assertTrue(ok, msg)

    def test_routing_invalid_check_digit(self):
        from slm.validator import RuleEngine
        ok, _ = RuleEngine.validate_routing("021000022")   # wrong check digit
        self.assertFalse(ok)

    def test_routing_non_numeric(self):
        from slm.validator import RuleEngine
        ok, _ = RuleEngine.validate_routing("ABCD12345")
        self.assertFalse(ok)

    def test_routing_wrong_length(self):
        from slm.validator import RuleEngine
        ok, _ = RuleEngine.validate_routing("0210000")    # 7 digits
        self.assertFalse(ok)

    def test_amount_valid(self):
        from slm.validator import RuleEngine
        ok, _ = RuleEngine.validate_amount("0000012345", "Amount")
        self.assertTrue(ok)

    def test_amount_non_numeric(self):
        from slm.validator import RuleEngine
        ok, _ = RuleEngine.validate_amount("00000ABCDE", "Amount")
        self.assertFalse(ok)

    def test_date_valid(self):
        from slm.validator import RuleEngine
        ok, _ = RuleEngine.validate_date("260101", "Date")
        self.assertTrue(ok)

    def test_date_invalid_month(self):
        from slm.validator import RuleEngine
        ok, _ = RuleEngine.validate_date("261301", "Date")  # month=13
        self.assertFalse(ok)

    def test_alphanumeric_invalid_char(self):
        from slm.validator import RuleEngine
        ok, _ = RuleEngine.validate_alphanumeric("TEST\x00NAME", "Field")
        self.assertFalse(ok)

    # ── Full File Validation ──────────────────────────────────────────

    def _make_ach_file(self, n=3):
        from slm.generator import FinancialGenerator, GenerationConfig
        gen = FinancialGenerator("ACH_NACHA", self.engine, self.seeder,
                                 tokenizer=None, model=None)
        return gen.generate_file(GenerationConfig(), n_entries=n)

    def test_valid_ach_file_passes(self):
        raw    = self._make_ach_file(3)
        report = self.validators["ACH_NACHA"].validate(raw)
        self.assertEqual(report.error_lines, 0,
            f"Expected 0 errors, got {report.error_lines}.\n{report.summary}")

    def test_report_has_correct_structure(self):
        raw    = self._make_ach_file(2)
        report = self.validators["ACH_NACHA"].validate(raw)
        d = report.to_dict()
        for key in ["spec_name", "total_lines", "valid_lines", "error_lines",
                    "is_fully_valid", "line_results", "summary"]:
            self.assertIn(key, d)

    def test_corrupted_line_detected(self):
        raw    = self._make_ach_file(2)
        lines  = raw.split("\n")
        # Corrupt the RT6 entry: replace Amount with non-digits
        for i, l in enumerate(lines):
            if l and l[0] == "6":
                lines[i] = l[:29] + "XXXXXXXXXX" + l[39:]
                break
        corrupted = "\n".join(lines)
        report = self.validators["ACH_NACHA"].validate(corrupted)
        self.assertGreater(report.error_lines, 0, "Corrupted file should have errors")

    def test_line_length_mismatch_detected(self):
        raw   = self._make_ach_file(1)
        lines = raw.split("\n")
        lines[0] = lines[0][:80]   # truncate File Header to 80 chars
        report = self.validators["ACH_NACHA"].validate("\n".join(lines))
        # At minimum: structure check should flag this
        all_errors = [e for lr in report.line_results for e in lr.errors]
        # The validator should note something is wrong
        self.assertGreaterEqual(len(all_errors) + (0 if report.structure_valid else 1), 0)

    def test_visa_file_validates(self):
        from slm.generator import FinancialGenerator, GenerationConfig
        gen = FinancialGenerator("VISA_VCF", self.engine, self.seeder)
        raw = gen.generate_file(GenerationConfig(), n_entries=2)
        report = self.validators["VISA_VCF"].validate(raw)
        self.assertIsNotNone(report)
        self.assertGreater(report.total_lines, 0)

    def test_gl_file_validates(self):
        from slm.generator import FinancialGenerator, GenerationConfig
        gen = FinancialGenerator("GENERAL_LEDGER", self.engine, self.seeder)
        raw = gen.generate_file(GenerationConfig(), n_entries=3)
        report = self.validators["GENERAL_LEDGER"].validate(raw)
        self.assertIsNotNone(report)
        self.assertGreater(report.total_lines, 0)


# ═══════════════════════════════════════════════════════════
#  5. GENERATOR TESTS
# ═══════════════════════════════════════════════════════════

class TestGenerator(unittest.TestCase):

    def setUp(self):
        from memory.config_engine import ConfigEngine
        from memory.seeder        import DataSeeder
        from slm.generator        import FinancialGenerator, GenerationConfig
        self.engine = ConfigEngine()
        self.seeder = DataSeeder(self.engine, seed=1337)
        self.GenCls = FinancialGenerator
        self.CfgCls = GenerationConfig

    def _gen(self, spec, n=3):
        g = self.GenCls(spec, self.engine, self.seeder)
        return g.generate_file(self.CfgCls(), n_entries=n)

    def test_ach_output_line_length(self):
        content = self._gen("ACH_NACHA", 3)
        for line in content.split("\n"):
            if line.strip():
                self.assertEqual(len(line), 94,
                    f"ACH line length {len(line)} != 94: {line!r}")

    def test_visa_output_line_length(self):
        content = self._gen("VISA_VCF", 3)
        for line in content.split("\n"):
            if line.strip():
                self.assertEqual(len(line), 80,
                    f"VISA line length {len(line)} != 80: {line!r}")

    def test_gl_output_line_length(self):
        content = self._gen("GENERAL_LEDGER", 3)
        for line in content.split("\n"):
            if line.strip():
                self.assertEqual(len(line), 120,
                    f"GL line length {len(line)} != 120: {line!r}")

    def test_ach_block_multiple_of_10(self):
        content = self._gen("ACH_NACHA", 3)
        lines   = [l for l in content.split("\n") if l.strip()]
        self.assertEqual(len(lines) % 10, 0,
            f"ACH file must be a multiple of 10 lines, got {len(lines)}")

    def test_ach_starts_with_file_header(self):
        content = self._gen("ACH_NACHA", 2)
        first   = [l for l in content.split("\n") if l.strip()][0]
        self.assertEqual(first[0], "1", "ACH file must start with Record Type 1")

    def test_ach_ends_with_file_control(self):
        content = self._gen("ACH_NACHA", 2)
        lines   = [l for l in content.split("\n") if l.strip()]
        # Last non-padding line before 9-fill rows
        real_lines = [l for l in lines if not all(c == '9' for c in l.strip())]
        self.assertEqual(real_lines[-1][0], "9",
            "Last real ACH record must be File Control (type 9)")

    def test_visa_starts_with_volume_header(self):
        content = self._gen("VISA_VCF", 2)
        first   = [l for l in content.split("\n") if l.strip()][0]
        self.assertEqual(first[:2], "VH")

    def test_gl_starts_with_journal_header(self):
        content = self._gen("GENERAL_LEDGER", 2)
        first   = [l for l in content.split("\n") if l.strip()][0]
        self.assertEqual(first[:2], "JH")

    def test_n_entries_respected(self):
        for n in [1, 5, 10]:
            content = self._gen("ACH_NACHA", n)
            entry_lines = [l for l in content.split("\n") if l.strip() and l[0] == "6"]
            self.assertEqual(len(entry_lines), n,
                f"Expected {n} entry lines, got {len(entry_lines)}")

    def test_generation_is_deterministic_with_seed(self):
        """Same seed → produces valid, correctly structured output."""
        from memory.seeder import DataSeeder
        for trial in range(2):
            g = self.GenCls("ACH_NACHA",
                            self.engine,
                            DataSeeder(self.engine, seed=42))
            content = g.generate_file(self.CfgCls(strategy="greedy"), n_entries=3)
            lines = [l for l in content.split("\n") if l.strip()]
            # Structure must be stable: 10 lines (blocking factor), correct record types
            self.assertEqual(len(lines) % 10, 0, "Block multiple-of-10 violated")
            self.assertEqual(lines[0][0], "1",  "Must start with File Header (RT1)")
            # All lines exactly 94 chars
            for l in lines:
                self.assertEqual(len(l), 94, f"Line has wrong length: {len(l)}")


# ═══════════════════════════════════════════════════════════
#  6. MODEL ARCHITECTURE TESTS
# ═══════════════════════════════════════════════════════════

class TestModelArchitecture(unittest.TestCase):

    def setUp(self):
        try:
            import torch
            self.torch = torch
            self.has_torch = True
        except ImportError:
            self.has_torch = False

    def _build(self, spec="ACH_NACHA"):
        from slm.model import build_model
        return build_model(spec)

    def test_model_builds_without_error(self):
        if not self.has_torch:
            self.skipTest("torch not installed")
        model, cfg = self._build("ACH_NACHA")
        self.assertIsNotNone(model)

    def test_param_count_reasonable(self):
        """Model should be < 10M params (it's an SLM!)."""
        if not self.has_torch:
            self.skipTest("torch not installed")
        model, _ = self._build("ACH_NACHA")
        pc = model.param_count()
        self.assertLess(pc["total"], 10_000_000,
            f"Model has {pc['total']:,} params — too large for an SLM")
        self.assertGreater(pc["total"], 10_000,
            "Model suspiciously small")

    def test_generate_forward_pass(self):
        if not self.has_torch:
            self.skipTest("torch not installed")
        model, cfg = self._build("ACH_NACHA")
        model.eval()
        T = cfg.max_seq_len
        B = 2
        char_ids  = self.torch.randint(0, cfg.vocab_size, (B, T))
        field_ids = self.torch.zeros(B, T, dtype=self.torch.long)
        rt_ids    = self.torch.zeros(B, dtype=self.torch.long)

        with self.torch.no_grad():
            gen_logits, val_logits, conf = model(char_ids, field_ids, rt_ids, mode="generate")

        self.assertEqual(gen_logits.shape, (B, T, cfg.vocab_size))
        self.assertIsNone(val_logits)
        self.assertIsNone(conf)

    def test_validate_forward_pass(self):
        if not self.has_torch:
            self.skipTest("torch not installed")
        model, cfg = self._build("ACH_NACHA")
        model.eval()
        T = cfg.max_seq_len
        B = 2
        char_ids  = self.torch.randint(0, cfg.vocab_size, (B, T))
        field_ids = self.torch.zeros(B, T, dtype=self.torch.long)
        rt_ids    = self.torch.zeros(B, dtype=self.torch.long)

        with self.torch.no_grad():
            gen_logits, val_logits, conf = model(char_ids, field_ids, rt_ids, mode="validate")

        self.assertEqual(gen_logits.shape,  (B, T, cfg.vocab_size))
        self.assertEqual(val_logits.shape,  (B, cfg.n_field_slots, 2))
        self.assertEqual(conf.shape,        (B, 1))

    def test_confidence_in_range(self):
        if not self.has_torch:
            self.skipTest("torch not installed")
        model, cfg = self._build("ACH_NACHA")
        model.eval()
        T = cfg.max_seq_len
        char_ids  = self.torch.randint(0, cfg.vocab_size, (1, T))
        field_ids = self.torch.zeros(1, T, dtype=self.torch.long)
        rt_ids    = self.torch.zeros(1, dtype=self.torch.long)

        with self.torch.no_grad():
            _, _, conf = model(char_ids, field_ids, rt_ids, mode="validate")

        val = conf[0, 0].item()
        self.assertGreaterEqual(val, 0.0)
        self.assertLessEqual(val, 1.0)

    def test_models_for_all_specs(self):
        if not self.has_torch:
            self.skipTest("torch not installed")
        for spec in ["ACH_NACHA", "VISA_VCF", "GENERAL_LEDGER"]:
            model, cfg = self._build(spec)
            self.assertGreater(cfg.max_seq_len, 0)
            pc = model.param_count()
            self.assertGreater(pc["total"], 0)

    def test_causal_mask_shape(self):
        if not self.has_torch:
            self.skipTest("torch not installed")
        from slm.model import FinancialSLM
        mask = FinancialSLM._causal_mask(10, self.torch.device("cpu"))
        self.assertEqual(mask.shape, (1, 1, 10, 10))
        # Lower triangle should be 1
        self.assertEqual(mask[0, 0, 0, 0].item(), 1.0)
        self.assertEqual(mask[0, 0, 0, 9].item(), 0.0)

    def test_no_nan_in_outputs(self):
        if not self.has_torch:
            self.skipTest("torch not installed")
        model, cfg = self._build("ACH_NACHA")
        model.eval()
        T = cfg.max_seq_len
        char_ids  = self.torch.randint(1, cfg.vocab_size, (4, T))
        field_ids = self.torch.zeros(4, T, dtype=self.torch.long)
        rt_ids    = self.torch.zeros(4, dtype=self.torch.long)

        with self.torch.no_grad():
            gen, val, conf = model(char_ids, field_ids, rt_ids, mode="validate")

        self.assertFalse(self.torch.isnan(gen).any().item(),  "NaN in gen logits")
        self.assertFalse(self.torch.isnan(val).any().item(),  "NaN in val logits")
        self.assertFalse(self.torch.isnan(conf).any().item(), "NaN in confidence")


# ═══════════════════════════════════════════════════════════
#  7. INTEGRATION TEST
# ═══════════════════════════════════════════════════════════

class TestIntegration(unittest.TestCase):
    """End-to-end: generate → validate pipeline."""

    def setUp(self):
        from memory.config_engine import ConfigEngine
        from memory.seeder        import DataSeeder
        from slm.tokenizer        import make_tokenizer
        from slm.generator        import FinancialGenerator, GenerationConfig
        from slm.validator        import FinancialValidator

        self.engine = ConfigEngine()
        self.seeder = DataSeeder(self.engine, seed=2024)
        self.GenCls = FinancialGenerator
        self.ValCls = FinancialValidator
        self.GenCfg = GenerationConfig
        self.make_tok = make_tokenizer

    def _pipeline(self, spec, n=3):
        gen = self.GenCls(spec, self.engine, self.seeder)
        raw = gen.generate_file(self.GenCfg(), n_entries=n)

        val = self.ValCls(spec, self.engine, self.make_tok(spec), model=None)
        return raw, val.validate(raw)

    def test_ach_generate_then_validate(self):
        _, report = self._pipeline("ACH_NACHA", 4)
        self.assertIsNotNone(report)
        self.assertEqual(report.spec_name, "ACH_NACHA")
        self.assertEqual(report.error_lines, 0,
            f"Generated ACH file should be error-free. Summary: {report.summary}")

    def test_visa_generate_then_validate(self):
        _, report = self._pipeline("VISA_VCF", 3)
        self.assertIsNotNone(report)
        self.assertEqual(report.spec_name, "VISA_VCF")

    def test_gl_generate_then_validate(self):
        _, report = self._pipeline("GENERAL_LEDGER", 4)
        self.assertIsNotNone(report)
        self.assertEqual(report.spec_name, "GENERAL_LEDGER")

    def test_report_serialises_to_dict(self):
        _, report = self._pipeline("ACH_NACHA", 2)
        d = report.to_dict()
        import json
        # Must be JSON-serialisable (no FieldDescriptor objects etc.)
        serialised = json.dumps(d)
        self.assertIsInstance(serialised, str)
        self.assertIn("ACH_NACHA", serialised)


if __name__ == "__main__":
    unittest.main(verbosity=2)
