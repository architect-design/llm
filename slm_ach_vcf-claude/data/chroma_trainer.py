"""
ChromaDB-Aware SLM Training Pipeline
Replaces: data/oracle_trainer.py

Data priority (highest quality first):
  1. ach_training_corpus  ← ACH files previously saved in ChromaDB
  2. ChromaACHGenerator   ← fresh files built from ChromaDB transactions
  3. ACHGenerator         ← purely synthetic (diversity + fallback)
"""

import os
import sys
import random
import logging
from typing import Optional

BASE = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, BASE)

from db.ach_repository import ACHRepository
from data.chroma_ach_generator import ChromaACHGenerator
from data.generator import ACHGenerator

log = logging.getLogger(__name__)

try:
    import torch
    BACKEND = "transformer"
except ImportError:
    BACKEND = "bigram"

DEFAULT_CONFIG = {
    "model_config":       "small",
    "n_corpus_files":     300,   # from ChromaDB corpus collection
    "n_chroma_files":     100,   # generated fresh from ChromaDB transactions
    "n_synthetic_files":  100,   # always include some synthetic for diversity
    "n_val_files":        50,
    "corpus_weight":      0.6,   # fraction from ChromaDB corpus
    "block_size":         512,
    "batch_size":         4,
    "max_epochs":         10,
    "learning_rate":      3e-4,
    "weight_decay":       0.1,
    "grad_clip":          1.0,
    "warmup_steps":       50,
    "save_dir":           os.path.join(BASE, "trained_models"),
}


class ChromaAwareTrainer:
    """
    Training pipeline that blends ChromaDB corpus files, freshly generated
    ChromaDB-backed files, and synthetic files.
    """

    def __init__(self, store_path: Optional[str] = None, train_config: dict = None):
        from typing import Optional as Opt
        self.store_path   = store_path
        self.cfg          = {**DEFAULT_CONFIG, **(train_config or {})}
        self.backend      = BACKEND
        self.chroma_gen   = ChromaACHGenerator(store_path)
        self.synth_gen    = ACHGenerator()
        self.repo         = ACHRepository(store_path)
        os.makedirs(self.cfg["save_dir"], exist_ok=True)
        log.info("ChromaAwareTrainer — backend=%s", BACKEND)

    # ─── Data collection ──────────────────────────────────────────────────────

    def collect_training_data(self, callback=None) -> dict:
        def rpt(msg, p=None):
            log.info(msg)
            if callback: callback(msg, p)

        train_files: list = []
        val_files:   list = []
        breakdown = {"corpus": 0, "chroma_live": 0, "synthetic": 0}

        # 1. Pull from ach_training_corpus collection
        rpt("Fetching files from ChromaDB corpus collection...", 5)
        corpus_train = self.repo.fetch_corpus(split="TRAIN", limit=self.cfg["n_corpus_files"])
        corpus_val   = self.repo.fetch_corpus(split="VAL",   limit=self.cfg["n_val_files"] // 2)
        train_files.extend(corpus_train)
        val_files.extend(corpus_val)
        breakdown["corpus"] = len(corpus_train) + len(corpus_val)
        rpt(f"Corpus: {len(corpus_train)} train, {len(corpus_val)} val", 15)

        # 2. Generate fresh files from ChromaDB transactions
        need_live = max(0, self.cfg["n_chroma_files"] - len(train_files))
        if need_live > 0:
            rpt(f"Generating {need_live} files from ChromaDB transactions...", 20)
            live = self.chroma_gen.generate_training_batch(n_files=need_live, callback=callback)
            train_files.extend(live)
            breakdown["chroma_live"] = len(live)

        # 3. Synthetic augmentation
        rpt(f"Generating {self.cfg['n_synthetic_files']} synthetic files...", 65)
        for i in range(self.cfg["n_synthetic_files"]):
            f = self.synth_gen.generate_file(
                num_batches=random.randint(1, 4),
                entries_per_batch=random.randint(3, 15),
            )
            train_files.append(f)
        breakdown["synthetic"] = self.cfg["n_synthetic_files"]

        # 4. Pad validation set with synthetic if needed
        pad_val = max(0, self.cfg["n_val_files"] - len(val_files))
        for _ in range(pad_val):
            val_files.append(self.synth_gen.generate_file(
                num_batches=random.randint(1, 2),
                entries_per_batch=random.randint(3, 8),
            ))

        random.shuffle(train_files)
        random.shuffle(val_files)

        stats = self.repo.corpus_stats()
        rpt(
            f"Data ready — {len(train_files)} train / {len(val_files)} val | "
            f"breakdown={breakdown} | corpus_total={stats.get('total', 0)}", 70
        )
        return {"train": train_files, "val": val_files, "breakdown": breakdown}

    # ─── Training entry point ─────────────────────────────────────────────────

    def train(self, callback=None) -> dict:
        def rpt(msg, p=None):
            log.info(msg)
            if callback: callback(msg, p)

        rpt("Collecting training data from ChromaDB...", 0)
        data = self.collect_training_data(callback)

        rpt(
            f"Training on {len(data['train'])} files "
            f"(corpus={data['breakdown']['corpus']}, "
            f"chroma_live={data['breakdown']['chroma_live']}, "
            f"synthetic={data['breakdown']['synthetic']})", 70
        )

        if self.backend == "transformer":
            return self._train_transformer(data["train"], data["val"], rpt)
        return self._train_bigram(data["train"], data["val"], rpt)

    # ─── Backends ─────────────────────────────────────────────────────────────

    def _train_bigram(self, train_files, val_files, rpt) -> dict:
        from model.bigram_fallback import BigramSLM

        rpt("Training Bigram SLM on ChromaDB + synthetic data...", 72)
        model = BigramSLM(order=5)
        model.file_type = "ACH"
        model.train(train_files, callback=rpt)

        val_score = self._bigram_eval(model, val_files[:30])
        rpt(f"Val perplexity: {val_score:.3f}", 95)

        save_path = os.path.join(self.cfg["save_dir"], "ach_model.pkl")
        model.save(save_path)
        rpt("Model saved!", 100)

        return {
            "model_path":    save_path,
            "backend":       "bigram",
            "best_val_loss": val_score,
            "train_files":   len(train_files),
        }

    def _bigram_eval(self, model, val_files) -> float:
        import numpy as np
        total_log = n = 0
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

        optimizer = AdamW(model.parameters(), lr=self.cfg["learning_rate"],
                          weight_decay=self.cfg["weight_decay"], betas=(0.9, 0.95))
        scheduler = CosineAnnealingLR(
            optimizer, T_max=max(len(train_in) * self.cfg["max_epochs"], 1), eta_min=1e-5
        )

        bs       = self.cfg["batch_size"]
        best_val = float("inf")
        losses   = []

        for epoch in range(self.cfg["max_epochs"]):
            model.train()
            perm = torch.randperm(len(train_in))
            ti, tt = train_in[perm], train_tgt[perm]
            lsum = nb = 0

            for i in range(0, len(ti), bs):
                xb, yb = ti[i:i+bs].to(device), tt[i:i+bs].to(device)
                optimizer.zero_grad()
                _, loss = model(xb, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg["grad_clip"])
                optimizer.step(); scheduler.step()
                lsum += loss.item(); nb += 1

            avg = lsum / max(nb, 1); losses.append(avg)

            model.eval()
            vl = 0.0
            with torch.no_grad():
                for i in range(0, min(len(val_in), 16), bs):
                    xv = val_in[i:i+bs].to(device)
                    yv = val_tgt[i:i+bs].to(device)
                    _, l = model(xv, yv); vl += l.item()
            vl /= max(min(16, len(val_in)) // bs, 1)

            p = 72 + int(((epoch + 1) / self.cfg["max_epochs"]) * 25)
            rpt(f"Epoch {epoch+1}/{self.cfg['max_epochs']} | train={avg:.4f} val={vl:.4f}", p)

            if vl < best_val:
                best_val = vl
                sp = os.path.join(self.cfg["save_dir"], "ach_model.pt")
                model.save(sp, tokenizer, "ACH")

        tok_path = os.path.join(self.cfg["save_dir"], "ach_tokenizer.json")
        tokenizer.save(tok_path)
        rpt("Training complete!", 100)

        return {
            "model_path":    os.path.join(self.cfg["save_dir"], "ach_model.pt"),
            "backend":       "transformer",
            "best_val_loss": best_val,
            "train_losses":  losses,
            "train_files":   len(train_files),
        }


# Allow Optional import at module level without circular import
