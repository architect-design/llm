"""
Training Pipeline for FinancialSLM.

Two training objectives run simultaneously:
  1. Causal Language Modelling (CLM) — next-character prediction.
     Loss: CrossEntropy over char vocab.
     This teaches the model the *syntax* of financial records.

  2. Field Validation Classification — binary valid/invalid per field.
     Loss: BCEWithLogits over labeled field correctness.
     This teaches the model the *rules* of financial formats.

Training data is synthesised from the in-memory config engine (no external
datasets required) producing an effectively infinite stream of valid +
deliberately corrupted records.

Checkpoint strategy: saves best val-loss model to `checkpoints/`.
"""

from __future__ import annotations

import os
import time
import random
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset

log = logging.getLogger("slm.trainer")


# ─────────────────────── Training Config ─────────────────────────────────────

@dataclass
class TrainConfig:
    spec_name       : str   = "ACH_NACHA"
    batch_size      : int   = 32
    learning_rate   : float = 3e-4
    weight_decay    : float = 1e-2
    warmup_steps    : int   = 200
    max_steps       : int   = 5_000
    val_every       : int   = 250
    save_every      : int   = 500
    clm_weight      : float = 1.0    # weight for CLM loss
    val_weight      : float = 0.5    # weight for validation-head loss
    corruption_prob : float = 0.3    # fraction of training lines deliberately corrupted
    checkpoint_dir  : str   = "checkpoints"
    device          : str   = "cpu"  # "cuda" if GPU available


# ─────────────────────── Synthetic Dataset ───────────────────────────────────

class SyntheticFinancialDataset(IterableDataset):
    """
    Infinite stream of synthetic financial records drawn from the in-memory
    config engine. Each item is a tuple of:
        (char_ids, field_ids, rt_ids, target_ids, field_labels)

    `field_labels` is a binary vector (n_fields,) indicating field validity.
    When corruption_prob > 0, some fields are deliberately broken so the
    validation head learns to detect errors.
    """

    def __init__(
        self,
        spec_name      : str,
        config_engine  ,            # memory.config_engine.ConfigEngine
        seeder         ,            # memory.seeder.DataSeeder
        tokenizer      ,            # slm.tokenizer.FinancialTokenizer
        model_cfg      ,            # slm.model.SLMConfig
        corruption_prob: float = 0.3,
        seed           : int   = 42,
    ):
        super().__init__()
        self.spec_name       = spec_name
        self.config_engine   = config_engine
        self.seeder          = seeder
        self.tokenizer       = tokenizer
        self.model_cfg       = model_cfg
        self.corruption_prob = corruption_prob
        self.rng             = random.Random(seed)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, ...]]:
        while True:
            yield self._generate_sample()

    def _generate_sample(self) -> Tuple[torch.Tensor, ...]:
        # 1. Generate a valid synthetic record line
        record_type = self.rng.choice(
            self.config_engine.get_record_types(self.spec_name)
        )
        raw_line, field_labels = self.seeder.generate_line(
            self.spec_name, record_type
        )

        # 2. Optionally corrupt some fields
        if self.rng.random() < self.corruption_prob:
            raw_line, field_labels = self._corrupt(raw_line, field_labels, record_type)

        # 3. Tokenize
        T    = self.model_cfg.max_seq_len
        line = raw_line.upper().ljust(T)[:T]
        char_ids   = torch.tensor([self.tokenizer.char_to_id(c) for c in line], dtype=torch.long)
        field_ids  = self._build_field_ids(record_type, T)
        rt_id      = self._record_type_to_id(record_type)
        rt_ids     = torch.full((1,), rt_id, dtype=torch.long)

        # 4. Targets for CLM: shift right by 1
        target_ids = char_ids.clone()

        # 5. Field labels as float tensor
        n_fields = self.model_cfg.n_field_slots
        labels   = torch.zeros(n_fields, dtype=torch.float)
        for i, v in enumerate(field_labels[:n_fields]):
            labels[i] = float(v)

        return char_ids, field_ids, rt_ids.squeeze(0), target_ids, labels

    def _corrupt(
        self,
        line: str,
        labels: List[int],
        record_type: str,
    ) -> Tuple[str, List[int]]:
        """Introduce a random field-level error to teach the validation head."""
        fields = self.config_engine.get_fields(self.spec_name, record_type)
        if not fields:
            return line, labels

        idx = self.rng.randrange(len(fields))
        fd  = fields[idx]
        corrupt_labels = list(labels)
        corrupt_labels[idx] = 0

        s, e = fd["start"] - 1, fd["end"]
        corruption = self.rng.choice([
            "X" * fd["length"],             # alphabetic in numeric field
            "?" * fd["length"],             # invalid chars
            " " * fd["length"],             # blank where required
            "9" * fd["length"],             # overflow numeric
        ])
        line_chars = list(line)
        line_chars[s:e] = list(corruption[:fd["length"]])
        return "".join(line_chars), corrupt_labels

    def _build_field_ids(self, record_type: str, T: int) -> torch.Tensor:
        fields = self.config_engine.get_fields(self.spec_name, record_type)
        ids    = torch.zeros(T, dtype=torch.long)
        for fi, fd in enumerate(fields):
            s, e = fd["start"] - 1, min(fd["end"], T)
            ids[s:e] = fi + 1  # 0 reserved for "no field"
        return ids

    def _record_type_to_id(self, rt: str) -> int:
        mapping = {rt: i for i, rt in enumerate(
            self.config_engine.get_record_types(self.spec_name)
        )}
        return mapping.get(rt, 0)


def collate_fn(batch):
    char_ids_list, field_ids_list, rt_ids_list, target_ids_list, labels_list = zip(*batch)
    return (
        torch.stack(char_ids_list),
        torch.stack(field_ids_list),
        torch.stack(rt_ids_list),
        torch.stack(target_ids_list),
        torch.stack(labels_list),
    )


# ─────────────────────── Learning-Rate Scheduler ─────────────────────────────

class WarmupCosineScheduler:
    """Linear warmup then cosine annealing."""

    def __init__(self, optimizer, warmup_steps: int, total_steps: int):
        self.opt          = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps  = total_steps
        self.step_count   = 0
        self._base_lrs    = [pg["lr"] for pg in optimizer.param_groups]

    def step(self):
        self.step_count += 1
        s = self.step_count
        if s <= self.warmup_steps:
            scale = s / max(1, self.warmup_steps)
        else:
            progress = (s - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            import math
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        for pg, base_lr in zip(self.opt.param_groups, self._base_lrs):
            pg["lr"] = base_lr * scale


# ─────────────────────────── Trainer ─────────────────────────────────────────

class SLMTrainer:
    """
    Orchestrates the dual-objective training loop.

    Example usage:
        from slm.model      import build_model
        from slm.trainer    import SLMTrainer, TrainConfig
        from memory.config_engine import ConfigEngine
        from memory.seeder  import DataSeeder
        from slm.tokenizer  import make_tokenizer

        cfg     = TrainConfig(spec_name="ACH_NACHA", max_steps=2000)
        engine  = ConfigEngine()
        seeder  = DataSeeder(engine)
        tok     = make_tokenizer("ACH_NACHA")
        model, mcfg = build_model("ACH_NACHA")
        trainer = SLMTrainer(model, mcfg, tok, engine, seeder, cfg)
        trainer.train()
    """

    def __init__(
        self,
        model      ,
        model_cfg  ,
        tokenizer  ,
        config_engine,
        seeder     ,
        train_cfg  : TrainConfig,
    ):
        self.model        = model
        self.model_cfg    = model_cfg
        self.tokenizer    = tokenizer
        self.config_engine = config_engine
        self.seeder       = seeder
        self.cfg          = train_cfg
        self.device       = torch.device(train_cfg.device)
        self.model.to(self.device)

        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=train_cfg.learning_rate,
            weight_decay=train_cfg.weight_decay,
        )
        self.scheduler = WarmupCosineScheduler(
            self.optimizer, train_cfg.warmup_steps, train_cfg.max_steps
        )
        self.clm_criterion = nn.CrossEntropyLoss(
            ignore_index=model_cfg.pad_token_id
        )
        self.val_criterion = nn.BCEWithLogitsLoss()

        # Metrics history
        self.history: Dict[str, List[float]] = {
            "clm_loss": [], "val_loss": [], "total_loss": [], "step": []
        }

        Path(train_cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def _build_dataloader(self) -> DataLoader:
        dataset = SyntheticFinancialDataset(
            spec_name       = self.cfg.spec_name,
            config_engine   = self.config_engine,
            seeder          = self.seeder,
            tokenizer       = self.tokenizer,
            model_cfg       = self.model_cfg,
            corruption_prob = self.cfg.corruption_prob,
        )
        return DataLoader(
            dataset,
            batch_size  = self.cfg.batch_size,
            collate_fn  = collate_fn,
            num_workers = 0,
        )

    def _train_step(self, batch) -> Dict[str, float]:
        char_ids, field_ids, rt_ids, target_ids, field_labels = [
            t.to(self.device) for t in batch
        ]

        self.optimizer.zero_grad()

        # — Generation pass (causal mode)
        gen_logits, _, _ = self.model(char_ids, field_ids, rt_ids, mode="generate")
        # Shift: predict token at t+1 from context 0..t
        clm_loss = self.clm_criterion(
            gen_logits[:, :-1].reshape(-1, self.model_cfg.vocab_size),
            target_ids[:, 1:].reshape(-1),
        )

        # — Validation pass (bidirectional mode)
        _, val_logits, _ = self.model(char_ids, field_ids, rt_ids, mode="validate")
        n_f = min(self.model_cfg.n_field_slots, field_labels.size(1))
        # val_logits: (B, n_fields, 2) — take logit for class "valid" (index 1)
        val_loss = self.val_criterion(
            val_logits[:, :n_f, 1],
            field_labels[:, :n_f],
        )

        total_loss = self.cfg.clm_weight * clm_loss + self.cfg.val_weight * val_loss
        total_loss.backward()

        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()

        return {
            "clm_loss"  : clm_loss.item(),
            "val_loss"  : val_loss.item(),
            "total_loss": total_loss.item(),
        }

    def train(self, progress_callback=None) -> Dict[str, List[float]]:
        """
        Main training loop.
        progress_callback(step, metrics) is called every val_every steps
        if provided (useful for API streaming).
        """
        log.info(f"Starting training: {self.cfg.spec_name} | "
                 f"{self.cfg.max_steps} steps | device={self.device}")

        loader   = iter(self._build_dataloader())
        best_val = float("inf")
        t0       = time.time()

        for step in range(1, self.cfg.max_steps + 1):
            self.model.train()
            batch   = next(loader)
            metrics = self._train_step(batch)

            if step % self.cfg.val_every == 0:
                elapsed = time.time() - t0
                lr      = self.optimizer.param_groups[0]["lr"]
                log.info(
                    f"Step {step:>5}/{self.cfg.max_steps} | "
                    f"CLM={metrics['clm_loss']:.4f} | "
                    f"VAL={metrics['val_loss']:.4f} | "
                    f"LR={lr:.2e} | {elapsed:.0f}s"
                )
                for k, v in metrics.items():
                    self.history[k].append(v)
                self.history["step"].append(step)

                if progress_callback:
                    progress_callback(step, metrics)

                if metrics["total_loss"] < best_val:
                    best_val = metrics["total_loss"]
                    self._save_checkpoint(step, "best")

            if step % self.cfg.save_every == 0:
                self._save_checkpoint(step, f"step_{step}")

        log.info(f"Training complete. Best loss: {best_val:.4f}")
        return self.history

    def _save_checkpoint(self, step: int, tag: str):
        path = Path(self.cfg.checkpoint_dir) / f"{self.cfg.spec_name}_{tag}.pt"
        torch.save({
            "step"      : step,
            "model_state": self.model.state_dict(),
            "opt_state" : self.optimizer.state_dict(),
            "history"   : self.history,
            "spec_name" : self.cfg.spec_name,
        }, path)
        log.info(f"Saved checkpoint → {path}")

    @classmethod
    def load_checkpoint(cls, path: str, model, optimizer=None):
        ck = torch.load(path, map_location="cpu")
        model.load_state_dict(ck["model_state"])
        if optimizer and "opt_state" in ck:
            optimizer.load_state_dict(ck["opt_state"])
        return ck
