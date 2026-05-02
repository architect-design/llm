"""
Oracle-Aware SLM Training Pipeline
Merges real Oracle transaction files with synthetic data for richer training.

Priority:
  1. Real ACH files from Oracle corpus table  (highest quality)
  2. ACH files generated from Oracle live data
  3. Purely synthetic files (fallback)
"""

import os
import sys
import random
import logging

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from db.oracle_connector import OracleConfig
from data.oracle_ach_generator import OracleACHGenerator
from data.generator import ACHGenerator

log = logging.getLogger(__name__)

# ── Try to load ML backend ────────────────────────────────────────────────────
try:
    import torch
    BACKEND = "transformer"
except ImportError:
    BACKEND = "bigram"


DEFAULT_CONFIG = {
    "model_config":      "small",
    "n_oracle_files":    300,   # from Oracle live + corpus
    "n_synthetic_files": 100,   # always include some synthetic for diversity
    "n_val_files":       50,
    "oracle_weight":     0.75,  # fraction of training data from Oracle
    "block_size":        512,
    "batch_size":        4,
    "max_epochs":        10,
    "learning_rate":     3e-4,
    "weight_decay":      0.1,
    "grad_clip":         1.0,
    "warmup_steps":      50,
    "save_dir":          os.path.join(BASE, "trained_models"),
}


class OracleAwareTrainer:
    """
    Training pipeline that combines Oracle-sourced and synthetic ACH files.
    Falls back gracefully to pure synthetic when Oracle is unavailable.
    """

    def __init__(
        self,
        oracle_config: OracleConfig = None,
        train_config:  dict = None,
    ):
        self.oracle_config = oracle_config or OracleConfig()
        self.cfg = {**DEFAULT_CONFIG, **(train_config or {})}
        self.backend = BACKEND
        self.oracle_gen = OracleACHGenerator(self.oracle_config)
        self.synth_gen  = ACHGenerator()
        os.makedirs(self.cfg["save_dir"], exist_ok=True)

        source = "mock" if self.oracle_gen.repo.pool.is_mock else "oracle"
        log.info("OracleAwareTrainer: backend=%s, oracle=%s", BACKEND, source)

    # ─── Data collection ──────────────────────────────────────────────────────

    def collect_training_data(self, callback=None) -> dict:
        """
        Collect ACH files from all available sources.
        Returns {"train": [...], "val": [...], "source_breakdown": {...}}
        """
        def rpt(msg, p=None):
            log.info(msg)
            if callback: callback(msg, p)

        train_files, val_files = [], []
        source_breakdown = {"oracle_corpus": 0, "oracle_live": 0, "synthetic": 0}

        # ── 1. Oracle corpus table (highest quality — previously saved files) ──
        rpt("Fetching files from Oracle corpus table...", 5)
        corpus_files = self.oracle_gen.fetch_corpus_files(
            split="TRAIN", limit=self.cfg["n_oracle_files"]
        )
        corpus_val   = self.oracle_gen.fetch_corpus_files(
            split="VAL",   limit=self.cfg["n_val_files"] // 2
        )
        train_files.extend(corpus_files)
        val_files.extend(corpus_val)
        source_breakdown["oracle_corpus"] = len(corpus_files) + len(corpus_val)
        rpt(f"Corpus: {len(corpus_files)} train, {len(corpus_val)} val", 10)

        # ── 2. Generate fresh files from Oracle live data ─────────────────────
        need_live = max(0, self.cfg["n_oracle_files"] - len(train_files))
        if need_live > 0:
            rpt(f"Generating {need_live} files from Oracle live transactions...", 12)
            live_files = self.oracle_gen.generate_training_batch(
                n_files=need_live, callback=callback
            )
            train_files.extend(live_files)
            source_breakdown["oracle_live"] = len(live_files)

        # ── 3. Synthetic files for diversity ──────────────────────────────────
        rpt(f"Generating {self.cfg['n_synthetic_files']} synthetic files...", 65)
        for i in range(self.cfg["n_synthetic_files"]):
            f = self.synth_gen.generate_file(
                num_batches=random.randint(1, 4),
                entries_per_batch=random.randint(3, 15),
            )
            train_files.append(f)
        source_breakdown["synthetic"] = self.cfg["n_synthetic_files"]

        # ── 4. Validation from synthetic if DB had none ────────────────────────
        need_val = max(0, self.cfg["n_val_files"] - len(val_files))
        for _ in range(need_val):
            val_files.append(self.synth_gen.generate_file(
                num_batches=random.randint(1, 2),
                entries_per_batch=random.randint(3, 8),
            ))

        # Shuffle
        random.shuffle(train_files)
        random.shuffle(val_files)

        rpt(
            f"Data ready: {len(train_files)} train / {len(val_files)} val "
            f"({source_breakdown})", 70
        )
        return {
            "train": train_files,
            "val":   val_files,
            "source_breakdown": source_breakdown,
        }

    # ─── Training ─────────────────────────────────────────────────────────────

    def train(self, callback=None) -> dict:
        def rpt(msg, p=None):
            log.info(msg)
            if callback: callback(msg, p)

        rpt("Collecting training data...", 0)
        data = self.collect_training_data(callback)
        train_files = data["train"]
        val_files   = data["val"]

        rpt(
            f"Training on {len(train_files)} files "
            f"(oracle_corpus={data['source_breakdown']['oracle_corpus']}, "
            f"oracle_live={data['source_breakdown']['oracle_live']}, "
            f"synthetic={data['source_breakdown']['synthetic']})", 70
        )

        if self.backend == "transformer":
            return self._train_transformer(train_files, val_files, rpt)
        else:
            return self._train_bigram(train_files, val_files, rpt)

    def _train_bigram(self, train_files, val_files, rpt) -> dict:
        from model.bigram_fallback import BigramSLM

        rpt("Training Bigram SLM on Oracle + synthetic data...", 72)
        model = BigramSLM(order=5)
        model.file_type = "ACH"

        model.train(train_files, callback=rpt)
        val_score = self._bigram_eval(model, val_files[:30])
        rpt(f"Validation perplexity: {val_score:.3f}", 95)

        save_path = os.path.join(self.cfg["save_dir"], "ach_model.pkl")
        model.save(save_path)
        rpt("Model saved!", 100)

        return {
            "model_path":      save_path,
            "backend":         "bigram",
            "best_val_loss":   val_score,
            "train_files":     len(train_files),
            "source_breakdown": {},
        }

    def _bigram_eval(self, model, val_files) -> float:
        import numpy as np
        total_log, n = 0.0, 0
        for text in val_files:
            for j in range(len(text) - model.order):
                ctx = text[j: j + model.order]
                nxt = text[j + model.order]
                if ctx in model.counts and nxt in model.counts[ctx]:
                    p = model.counts[ctx][nxt] / sum(model.counts[ctx].values())
                    total_log += np.log(max(p, 1e-10))
                    n += 1
        return float(np.exp(-total_log / max(n, 1)))

    def _train_transformer(self, train_files, val_files, rpt) -> dict:
        import torch
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR
        from model.transformer import FinancialSLM
        from model.tokenizer import FinancialTokenizer

        device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = FinancialTokenizer()

        rpt("Tokenizing...", 72)
        train_in, train_tgt = tokenizer.create_training_batch(
            train_files, block_size=self.cfg["block_size"], file_type="ACH"
        )
        val_in, val_tgt = tokenizer.create_training_batch(
            val_files, block_size=self.cfg["block_size"], file_type="ACH"
        )

        model = FinancialSLM(
            vocab_size=tokenizer.vocab_size,
            max_seq_len=self.cfg["block_size"],
            config=self.cfg["model_config"],
        ).to(device)

        optimizer = AdamW(
            model.parameters(),
            lr=self.cfg["learning_rate"],
            weight_decay=self.cfg["weight_decay"],
            betas=(0.9, 0.95),
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(len(train_in) * self.cfg["max_epochs"], 1),
            eta_min=1e-5,
        )

        bs = self.cfg["batch_size"]
        best_val, train_losses = float("inf"), []

        for epoch in range(self.cfg["max_epochs"]):
            model.train()
            perm = torch.randperm(len(train_in))
            train_in  = train_in[perm]
            train_tgt = train_tgt[perm]
            loss_sum, n = 0.0, 0

            for i in range(0, len(train_in), bs):
                xb, yb = train_in[i:i+bs].to(device), train_tgt[i:i+bs].to(device)
                optimizer.zero_grad()
                _, loss = model(xb, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.cfg["grad_clip"]
                )
                optimizer.step()
                scheduler.step()
                loss_sum += loss.item(); n += 1

            avg = loss_sum / max(n, 1)
            train_losses.append(avg)

            model.eval()
            vl = 0.0
            with torch.no_grad():
                for i in range(0, min(len(val_in), 16), bs):
                    xv, yv = val_in[i:i+bs].to(device), val_tgt[i:i+bs].to(device)
                    _, l = model(xv, yv); vl += l.item()
            vl /= max(min(16, len(val_in)) // bs, 1)

            p = 72 + int(((epoch + 1) / self.cfg["max_epochs"]) * 25)
            rpt(f"Epoch {epoch+1}/{self.cfg['max_epochs']} | train={avg:.4f} val={vl:.4f}", p)

            if vl < best_val:
                best_val = vl
                save_path = os.path.join(self.cfg["save_dir"], "ach_model.pt")
                model.save(save_path, tokenizer, "ACH")

        tok_path = os.path.join(self.cfg["save_dir"], "ach_tokenizer.json")
        tokenizer.save(tok_path)
        rpt("Training complete!", 100)

        return {
            "model_path":   os.path.join(self.cfg["save_dir"], "ach_model.pt"),
            "backend":      "transformer",
            "best_val_loss": best_val,
            "train_losses": train_losses,
            "train_files":  len(train_files),
        }
