#!/usr/bin/env python3
"""
financial_slm_framework/demo.py
Quick demonstration of the Financial SLM Framework capabilities.
Run this to see validation, generation, and model training in action.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import initialize_all_specs, spec_store
from config.store import FieldType, PaddingType
from validation import FinancialValidator
from generation import FinancialGenerator
from slm_core.tokenizer import FinancialTokenizer
from slm_core.model import FinancialSLM
import torch


def demo_validation():
    """Demonstrate file validation against ACH specification."""
    print("=" * 80)
    print("DEMO 1: FILE VALIDATION")
    print("=" * 80)

    # Create a sample ACH file (with intentional errors)
    sample_ach = """101 0910000191234567892305241233A094101WELLS FARGO BANK   ACME CORP            
5225ACME CORP            PAYROLL         1234567890PPD230524230524   1091000010000001
627091000012123456789      0000017500               JOHN DOE            0 123456789012345
822500000100091000010000000175000000000000001234567890                         091000010000001
9000001000001000000010009100001000000017500000000000000                                       """

    validator = FinancialValidator(spec_store=spec_store)
    result = validator.validate_file(sample_ach, "ach_nacha", "sample_ach.txt")

    print(f"\nFile: {result.filename}")
    print(f"Status: {result.overall_status.value.upper()}")
    print(f"Checksum Valid: {result.checksum_valid}")
    print(f"Total Records: {result.summary['total_records']}")
    print(f"Errors: {result.summary['total_errors']}")
    print(f"Warnings: {result.summary['total_warnings']}")

    print("\n--- Detailed Report ---")
    print(validator.format_report(result))

    return result


def demo_generation():
    """Demonstrate test file generation."""
    print("\n" + "=" * 80)
    print("DEMO 2: TEST FILE GENERATION")
    print("=" * 80)

    generator = FinancialGenerator(spec_store=spec_store)

    # Generate ACH file
    print("\n--- Generating ACH NACHA File ---")
    ach_content = generator.generate_file("ach_nacha", num_records=3, use_slm=False, seed=42)
    print(ach_content)

    # Generate VISA VCF file
    print("\n--- Generating VISA VCF File ---")
    vcf_content = generator.generate_file("visa_vcf", num_records=2, use_slm=False, seed=42)
    print(vcf_content)

    # Generate General Ledger file
    print("\n--- Generating General Ledger File ---")
    gl_content = generator.generate_file("general_ledger", num_records=2, use_slm=False, seed=42)
    print(gl_content)

    return ach_content, vcf_content, gl_content


def demo_model_training():
    """Demonstrate SLM model training on synthetic data."""
    print("\n" + "=" * 80)
    print("DEMO 3: SLM MODEL TRAINING")
    print("=" * 80)

    tokenizer = FinancialTokenizer(max_record_types=50)
    model = FinancialSLM(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        n_layers=4,
        n_heads=4,
        d_ff=512,
        max_seq_len=512
    )

    print(f"\nModel Architecture:")
    print(f"  - Vocab Size: {tokenizer.vocab_size}")
    print(f"  - Model Dim: 128")
    print(f"  - Layers: 4")
    print(f"  - Heads: 4")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  - Total Parameters: {total_params:,}")

    # Quick forward pass demo
    print("\n--- Forward Pass Demo ---")
    sample_text = "101 0910000191234567892305241233A094101WELLS FARGO"
    tokens = tokenizer.encode(sample_text, record_type=1)
    input_ids = torch.tensor([tokens], dtype=torch.long)

    with torch.no_grad():
        outputs = model(input_ids, record_type_ids=torch.tensor([1]), return_validation=True)
        print(f"Generation logits shape: {outputs['generation_logits'].shape}")
        print(f"Validation logits shape: {outputs['validation_logits'].shape}")

    return model, tokenizer


def demo_tokenizer():
    """Demonstrate domain-specific tokenizer."""
    print("\n" + "=" * 80)
    print("DEMO 4: DOMAIN-SPECIFIC TOKENIZER")
    print("=" * 80)

    tokenizer = FinancialTokenizer(max_record_types=50)

    print(f"\nVocabulary Size: {tokenizer.vocab_size}")
    print(f"Special Tokens: {tokenizer.SPECIAL_TOKENS}")

    # Encode a fixed-width record
    sample_record = "101 0910000191234567892305241233A094101WELLS FARGO BANK   ACME CORP            "
    field_boundaries = [
        (0, 1), (1, 3), (3, 13), (13, 23), (23, 29), 
        (29, 33), (33, 34), (34, 37), (37, 39), (39, 40),
        (40, 63), (63, 86), (86, 94)
    ]

    tokens = tokenizer.encode_fixed_width_record(sample_record, field_boundaries, record_type=1)
    print(f"\nEncoded {len(sample_record)} chars into {len(tokens)} tokens")
    print(f"Token IDs (first 20): {tokens[:20]}")

    decoded, fields = tokenizer.decode_fixed_width_record(tokens, [])
    print(f"\nDecoded length: {len(decoded)}")
    print(f"Extracted fields: {len(fields)}")

    return tokenizer


def demo_spec_store():
    """Demonstrate the in-memory specification store."""
    print("\n" + "=" * 80)
    print("DEMO 5: SPECIFICATION STORE (SOURCE OF TRUTH)")
    print("=" * 80)

    print(f"\nRegistered Specifications: {spec_store.list_specs()}")

    ach = spec_store.get_spec("ach_nacha")
    print(f"\nACH Specification:")
    print(f"  Name: {ach.name}")
    print(f"  Version: {ach.version}")
    print(f"  Record Types: {list(ach.record_specs.keys())}")

    # Show field rules for File Header
    fh = ach.get_record_spec("1")
    print(f"\n  File Header Record ({fh.name}):")
    print(f"    Total Length: {fh.total_length}")
    print(f"    Fields: {len(fh.fields)}")
    for field in fh.fields[:5]:
        print(f"      - {field.name}: pos {field.start_pos}-{field.end_pos}, type={field.field_type.value}, len={field.length}")

    # O(1) field rule lookup
    rule = spec_store.get_field_rule("ach_nacha", "1", "RecordTypeCode")
    print(f"\n  O(1) Lookup: ach_nacha:1:RecordTypeCode -> {rule}")

    return ach


def main():
    """Run all demos."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "FINANCIAL SLM FRAMEWORK DEMO" + " " * 30 + "║")
    print("║" + " " * 15 + "Standalone Small Language Model for Financial Formats" + " " * 16 + "║")
    print("╚" + "=" * 78 + "╝")

    # Initialize specifications
    print("\n[Initializing Specifications...]")
    initialize_all_specs()

    # Run demos
    demo_spec_store()
    demo_tokenizer()
    demo_validation()
    demo_generation()
    demo_model_training()

    print("\n" + "=" * 80)
    print("ALL DEMOS COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Start the API server: uvicorn api.main:app --reload")
    print("  2. Open http://localhost:8000 in your browser")
    print("  3. Train the model via the Model Status tab")
    print("  4. Generate and validate files through the web interface")


if __name__ == "__main__":
    main()
