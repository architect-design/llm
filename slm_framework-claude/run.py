#!/usr/bin/env python3
"""
FinancialSLM — Command-Line Interface.

Usage examples:

  # Generate an ACH NACHA file with 5 entries
  python run.py generate --spec ACH_NACHA --entries 5 --out test.ach

  # Validate an existing file
  python run.py validate --spec ACH_NACHA --file test.ach

  # Validate with a compact summary only
  python run.py validate --spec ACH_NACHA --file test.ach --summary

  # Train the SLM for a spec
  python run.py train --spec ACH_NACHA --steps 2000 --batch 16

  # Explore spec field definitions
  python run.py explore --spec ACH_NACHA --rt RT6

  # Start the API server
  python run.py serve --host 0.0.0.0 --port 8000

  # Run the full test suite
  python run.py test
"""

import sys
import os
import argparse
import json

# Make sure the project root is on the path
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


# ─────────────────────── Colour helpers ──────────────────────────────────────

class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    GREEN  = "\033[92m"
    RED    = "\033[91m"
    YELLOW = "\033[93m"
    BLUE   = "\033[94m"
    CYAN   = "\033[96m"
    DIM    = "\033[2m"
    def g(s): return f"{C.GREEN}{s}{C.RESET}"
    def r(s): return f"{C.RED}{s}{C.RESET}"
    def y(s): return f"{C.YELLOW}{s}{C.RESET}"
    def b(s): return f"{C.BLUE}{s}{C.RESET}"
    def c(s): return f"{C.CYAN}{s}{C.RESET}"
    def d(s): return f"{C.DIM}{s}{C.RESET}"
    def bold(s): return f"{C.BOLD}{s}{C.RESET}"


def banner():
    print(f"""
{C.BLUE}╔══════════════════════════════════════════════════════════╗
║  {C.BOLD}FinancialSLM — Specification Intelligence Engine{C.RESET}{C.BLUE}       ║
║  {C.DIM}ACH NACHA · VISA VCF · General Ledger{C.RESET}{C.BLUE}                  ║
║  {C.DIM}Zero external deps · Fully local · Privacy-first{C.RESET}{C.BLUE}        ║
╚══════════════════════════════════════════════════════════╝{C.RESET}
""")


# ─────────────────────── Command: generate ───────────────────────────────────

def cmd_generate(args):
    from memory.config_engine import ConfigEngine
    from memory.seeder        import DataSeeder
    from slm.generator        import FinancialGenerator, GenerationConfig

    engine = ConfigEngine()
    seeder = DataSeeder(engine, seed=args.seed)
    gen    = FinancialGenerator(args.spec, engine, seeder)
    cfg    = GenerationConfig(
        strategy    = args.strategy,
        temperature = args.temperature,
        top_k       = args.topk,
    )

    print(f"{C.c('Generating')} {C.bold(args.spec)} file with {args.entries} entries…")
    content = gen.generate_file(cfg, n_entries=args.entries)
    lines   = [l for l in content.split("\n") if l.strip()]

    if args.out:
        with open(args.out, "w", encoding="ascii") as f:
            f.write(content)
        print(f"{C.g('✓')} Wrote {len(lines)} lines → {C.bold(args.out)}")
    else:
        print(f"\n{C.d('─'*70)}")
        for i, line in enumerate(lines[:20]):
            rt = f"RT{line[0]}" if args.spec == "ACH_NACHA" else line[:2]
            print(f"{C.d(f'{i+1:>3}.')} {C.c(f'[{rt}]')} {line}")
        if len(lines) > 20:
            print(f"{C.d(f'  … and {len(lines)-20} more lines')}")
        print(f"{C.d('─'*70)}")
        print(f"\n{C.g('✓')} Generated {len(lines)} lines.")


# ─────────────────────── Command: validate ───────────────────────────────────

def cmd_validate(args):
    from memory.config_engine import ConfigEngine
    from slm.tokenizer        import make_tokenizer
    from slm.validator        import FinancialValidator

    if not os.path.exists(args.file):
        print(C.r(f"✗ File not found: {args.file}"))
        sys.exit(1)

    with open(args.file, encoding="ascii", errors="replace") as f:
        raw = f.read()

    engine    = ConfigEngine()
    tok       = make_tokenizer(args.spec)
    validator = FinancialValidator(args.spec, engine, tok, model=None)

    print(f"{C.c('Validating')} {C.bold(args.file)} as {C.bold(args.spec)}…")
    report = validator.validate(raw)

    pct = (report.valid_lines / max(report.total_lines, 1) * 100)
    ok_sym  = C.g("✓") if report.is_fully_valid else C.r("✗")

    print(f"\n{ok_sym}  {C.bold('VALIDATION REPORT')}  —  {args.spec}")
    print(f"   Total lines   : {report.total_lines}")
    print(f"   Valid lines   : {C.g(report.valid_lines)}")
    print(f"   Error lines   : {C.r(report.error_lines) if report.error_lines else C.g('0')}")
    print(f"   Valid %       : {C.g(f'{pct:.1f}%') if pct==100 else C.y(f'{pct:.1f}%')}")
    print(f"   Checksum      : {C.g('PASS') if report.checksum_valid else C.r('FAIL')}")
    print(f"   Structure     : {C.g('PASS') if report.structure_valid else C.r('FAIL')}")
    print(f"   Sequence      : {C.g('PASS') if report.sequence_valid else C.r('FAIL')}")

    if not args.summary:
        err_lines = [lr for lr in report.line_results if not lr.is_valid]
        if err_lines:
            print(f"\n{C.bold('Errors:')} ({len(err_lines)} lines)")
            for lr in err_lines[:20]:
                print(f"  Line {lr.line_no:>4}  [{C.c(lr.record_type)}]  conf={lr.model_conf:.2f}")
                for e in lr.errors[:3]:
                    print(f"    {C.r('✗')} {e.field_name}: {e.rule}")
            if len(err_lines) > 20:
                print(f"  … and {len(err_lines)-20} more error lines")

    if args.json_out:
        d = report.to_dict()
        with open(args.json_out, "w") as f:
            json.dump(d, f, indent=2)
        print(f"\n{C.g('✓')} JSON report → {args.json_out}")

    sys.exit(0 if report.is_fully_valid else 1)


# ─────────────────────── Command: train ──────────────────────────────────────

def cmd_train(args):
    try:
        import torch
    except ImportError:
        print(C.r("✗ PyTorch is required for training. pip install torch"))
        sys.exit(1)

    from memory.config_engine import ConfigEngine
    from memory.seeder        import DataSeeder
    from slm.tokenizer        import make_tokenizer
    from slm.model            import build_model
    from slm.trainer          import SLMTrainer, TrainConfig

    engine  = ConfigEngine()
    seeder  = DataSeeder(engine)
    tok     = make_tokenizer(args.spec)
    model, mcfg = build_model(args.spec)
    pc = model.param_count()

    print(f"{C.c('Training')} SLM for {C.bold(args.spec)}")
    print(f"   Parameters : {pc['total']:,}")
    print(f"   Device     : {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"   Steps      : {args.steps}")
    print(f"   Batch size : {args.batch}")
    print()

    cfg = TrainConfig(
        spec_name       = args.spec,
        max_steps       = args.steps,
        batch_size      = args.batch,
        learning_rate   = args.lr,
        corruption_prob = args.corrupt,
        checkpoint_dir  = os.path.join(ROOT, "checkpoints"),
        device          = "cuda" if torch.cuda.is_available() else "cpu",
    )

    import logging
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    trainer = SLMTrainer(model, mcfg, tok, engine, seeder, cfg)
    history = trainer.train()

    if history["clm_loss"]:
        final_clm = history["clm_loss"][-1]
        final_val = history["val_loss"][-1]
        print(f"\n{C.g('✓')} Training complete.")
        print(f"   Final CLM loss : {final_clm:.4f}")
        print(f"   Final VAL loss : {final_val:.4f}")


# ─────────────────────── Command: explore ────────────────────────────────────

def cmd_explore(args):
    from memory.config_engine import ConfigEngine
    engine = ConfigEngine()

    if args.rt:
        print(f"\n{C.bold(args.spec)} / {C.c(args.rt)} — Field Schema\n")
        print(engine.describe(args.spec, args.rt))
    else:
        specs = engine.get_all_specs()
        for s in specs:
            if s["name"] == args.spec:
                print(f"\n{C.bold(s['full_name'])}")
                print(f"  Line length   : {s['line_length']} chars")
                print(f"  Record types  : {', '.join(s['record_types'])}")
                print(f"  Description   : {s['description'][:80]}…")
                print(f"\n  Record types available:")
                for rt in s["record_types"]:
                    fields = engine.get_fields(s["name"], rt)
                    print(f"    {C.c(rt):15}  {len(fields)} fields")
                break


# ─────────────────────── Command: serve ──────────────────────────────────────

def cmd_serve(args):
    try:
        import uvicorn
    except ImportError:
        print(C.r("✗ uvicorn required: pip install uvicorn[standard]"))
        sys.exit(1)

    print(f"{C.c('Starting')} FinancialSLM API server…")
    print(f"  URL : {C.bold(f'http://{args.host}:{args.port}')}")
    print(f"  Docs: {C.bold(f'http://{args.host}:{args.port}/docs')}")
    print(f"  UI  : {C.bold(f'http://{args.host}:{args.port}/')}\n")

    os.chdir(ROOT)
    uvicorn.run(
        "api.main:app",
        host      = args.host,
        port      = args.port,
        reload    = args.reload,
        log_level = "info",
    )


# ─────────────────────── Command: test ───────────────────────────────────────

def cmd_test(args):
    import unittest
    loader = unittest.TestLoader()
    suite  = loader.discover(os.path.join(ROOT, "tests"), pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


# ─────────────────────── Argument Parser ─────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(
        prog        = "run.py",
        description = "FinancialSLM — Financial Specification Intelligence Engine",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog = __doc__,
    )
    sub = p.add_subparsers(dest="command")

    # generate
    g = sub.add_parser("generate", help="Generate a synthetic financial file")
    g.add_argument("--spec",    default="ACH_NACHA",
                   choices=["ACH_NACHA","VISA_VCF","GENERAL_LEDGER"])
    g.add_argument("--entries", type=int,   default=3,           help="Number of entry records")
    g.add_argument("--out",     default=None,                    help="Output file path")
    g.add_argument("--strategy",default="temperature",
                   choices=["greedy","temperature","top_k"])
    g.add_argument("--temperature", type=float, default=0.7)
    g.add_argument("--topk",    type=int,   default=10)
    g.add_argument("--seed",    type=int,   default=None)

    # validate
    v = sub.add_parser("validate", help="Validate a financial file")
    v.add_argument("--spec",     default="ACH_NACHA",
                   choices=["ACH_NACHA","VISA_VCF","GENERAL_LEDGER"])
    v.add_argument("--file",     required=True, help="Path to file to validate")
    v.add_argument("--summary",  action="store_true", help="Print summary only")
    v.add_argument("--json-out", dest="json_out", default=None, help="Write JSON report to file")

    # train
    t = sub.add_parser("train", help="Train the SLM for a given spec")
    t.add_argument("--spec",    default="ACH_NACHA",
                   choices=["ACH_NACHA","VISA_VCF","GENERAL_LEDGER"])
    t.add_argument("--steps",   type=int,   default=2000)
    t.add_argument("--batch",   type=int,   default=16)
    t.add_argument("--lr",      type=float, default=3e-4)
    t.add_argument("--corrupt", type=float, default=0.3, help="Corruption probability")

    # explore
    e = sub.add_parser("explore", help="Explore spec field definitions")
    e.add_argument("--spec", default="ACH_NACHA",
                   choices=["ACH_NACHA","VISA_VCF","GENERAL_LEDGER"])
    e.add_argument("--rt",   default=None,  help="Record type (e.g. RT6)")

    # serve
    s = sub.add_parser("serve", help="Start the FastAPI server")
    s.add_argument("--host",   default="0.0.0.0")
    s.add_argument("--port",   type=int, default=8000)
    s.add_argument("--reload", action="store_true")

    # test
    sub.add_parser("test", help="Run the test suite")

    return p


# ─────────────────────── Main ─────────────────────────────────────────────────

def main():
    banner()
    parser = build_parser()
    args   = parser.parse_args()

    dispatch = {
        "generate": cmd_generate,
        "validate": cmd_validate,
        "train"   : cmd_train,
        "explore" : cmd_explore,
        "serve"   : cmd_serve,
        "test"    : cmd_test,
    }

    if args.command not in dispatch:
        parser.print_help()
        sys.exit(0)

    dispatch[args.command](args)


if __name__ == "__main__":
    main()
