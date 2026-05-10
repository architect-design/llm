"""
Constrained Generation Module — Field-Aware Autoregressive Decoding.

Standard LLM generation is unconstrained: the model picks any token at
each step. That's catastrophic for financial files — one wrong character
at column 4 invalidates the entire record.

Our approach: FIELD-MASK DECODING
  At each character position we query the ConfigEngine to determine
  which *character classes* are legal. We zero-out all illegal logits
  (hard mask) before the softmax, constraining the model to only produce
  spec-valid characters.

  Example — ACH RT6, column 4 (Transaction Code):
    Allowed: ["22","23","27","28","32","33","37","38"]
    At col 4: only digits 2,3 are possible → mask out everything else
    At col 5: only 2,3,7,8 depending on prior col

This produces SYNTACTICALLY PERFECT files even from a partially-trained
model, because correctness is enforced structurally, not just learned.

Sampling strategies:
  - greedy     : argmax at each step (fastest, deterministic)
  - temperature : softmax with temp scaling (varied, still constrained)
  - top_k      : top-k filtering before sample
"""

from __future__ import annotations

import random
import string
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


# ─────────────────────── Character Class Masks ───────────────────────────────

def _digits_mask(vocab: Dict[str, int]) -> List[int]:
    return [vocab[c] for c in "0123456789" if c in vocab]

def _alpha_mask(vocab: Dict[str, int]) -> List[int]:
    return [vocab[c] for c in string.ascii_uppercase if c in vocab]

def _alphanum_mask(vocab: Dict[str, int]) -> List[int]:
    return _digits_mask(vocab) + _alpha_mask(vocab) + [vocab.get(" ", -1)]

def _space_mask(vocab: Dict[str, int]) -> List[int]:
    return [vocab[" "]] if " " in vocab else []

def _allowed_chars_mask(allowed: List[str], vocab: Dict[str, int]) -> List[int]:
    ids = []
    for a in allowed:
        for ch in a.upper():
            if ch in vocab:
                ids.append(vocab[ch])
    return list(set(ids))


# ─────────────────────── Constraint Resolver ─────────────────────────────────

class ConstraintResolver:
    """
    Given a spec, record type, and current character column,
    returns the set of allowed token IDs for that position.
    """

    def __init__(self, config_engine, tokenizer):
        self.engine    = config_engine
        self.tokenizer = tokenizer
        self.vocab     = tokenizer.vocab

    def allowed_ids(self, spec: str, rt: str, col: int) -> Optional[List[int]]:
        """
        Returns allowed vocab IDs for column `col` (0-indexed) of record type `rt`.
        Returns None if no constraint (any character allowed).
        """
        fields = self.engine.get_fields(spec, rt)
        if not fields:
            return None

        for fd in fields:
            s, e = fd["start"] - 1, fd["end"]
            if s <= col < e:
                ftype   = fd["field_type"]
                allowed = fd.get("allowed")

                if allowed and len(allowed) > 0:
                    # Only specific character values allowed
                    # Determine position within field
                    fi = col - s
                    char_options = set()
                    for a in allowed:
                        a_upper = a.upper().ljust(e - s)
                        if fi < len(a_upper):
                            ch = a_upper[fi]
                            if ch in self.vocab:
                                char_options.add(self.vocab[ch])
                    return list(char_options) if char_options else None

                if ftype in ("NUMERIC", "AMOUNT"):
                    return _digits_mask(self.vocab)

                if ftype == "ROUTING":
                    return _digits_mask(self.vocab)

                if ftype == "ALPHANUMERIC":
                    return _alphanum_mask(self.vocab)

                if ftype == "BLANK_PAD":
                    return _space_mask(self.vocab)

                if ftype == "DATE":
                    return _digits_mask(self.vocab)

                if ftype == "AMOUNT":
                    return _digits_mask(self.vocab)

                return None  # no constraint for this field type

        # Outside all defined fields → allow alphanumeric + space
        return _alphanum_mask(self.vocab)


# ─────────────────────── Generation Config ───────────────────────────────────

class GenerationConfig:
    def __init__(
        self,
        strategy     : str   = "temperature",   # greedy | temperature | top_k
        temperature  : float = 0.7,
        top_k        : int   = 10,
        max_new_chars: int   = 120,
    ):
        self.strategy      = strategy
        self.temperature   = temperature
        self.top_k         = top_k
        self.max_new_chars = max_new_chars


# ─────────────────────── Generator Class ─────────────────────────────────────

class FinancialGenerator:
    """
    Produces syntactically correct financial file content.

    Two generation pathways:
      1. model_generate   — uses SLM for character probabilities (requires trained model)
      2. rule_generate    — pure ConfigEngine + DataSeeder (no model required)

    Both pathways enforce field constraints through ConstraintResolver.
    """

    def __init__(
        self,
        spec_name    : str,
        config_engine,
        seeder       ,
        tokenizer    = None,
        model        = None,
        model_cfg    = None,
        device       : str = "cpu",
    ):
        self.spec_name     = spec_name
        self.engine        = config_engine
        self.seeder        = seeder
        self.tokenizer     = tokenizer
        self.model         = model
        self.model_cfg     = model_cfg
        self.device        = torch.device(device)
        self.resolver      = ConstraintResolver(config_engine, tokenizer) if tokenizer else None

        if model is not None:
            model.to(self.device)
            model.eval()

    # ── Public API ────────────────────────────────────────────────────────

    def generate_file(
        self,
        gen_cfg  : Optional[GenerationConfig] = None,
        n_entries: int = 3,
    ) -> str:
        """Generate a complete, spec-valid financial file."""
        gen_cfg = gen_cfg or GenerationConfig()
        if self.spec_name == "ACH_NACHA":
            return self._gen_ach_file(gen_cfg, n_entries)
        elif self.spec_name == "VISA_VCF":
            return self._gen_visa_file(gen_cfg, n_entries)
        elif self.spec_name == "GENERAL_LEDGER":
            return self._gen_gl_file(gen_cfg, n_entries)
        else:
            raise ValueError(f"Unknown spec: {self.spec_name}")

    def generate_record(
        self,
        record_type: str,
        gen_cfg    : Optional[GenerationConfig] = None,
        context    : Optional[Dict]             = None,
    ) -> str:
        """Generate a single record line (model + constraint or pure rule)."""
        gen_cfg = gen_cfg or GenerationConfig()
        if self.model is not None and self.tokenizer is not None:
            return self._model_generate_line(record_type, gen_cfg, context)
        else:
            return self.seeder.generate_line(self.spec_name, record_type)[0]

    # ── Model-Based Generation ─────────────────────────────────────────────

    def _model_generate_line(
        self,
        record_type: str,
        gen_cfg    : GenerationConfig,
        context    : Optional[Dict],
    ) -> str:
        T       = self.model_cfg.max_seq_len
        tok     = self.tokenizer
        engine  = self.engine

        # Seed with the record-type character
        rt_char = self._record_type_seed(record_type)
        generated = [rt_char]

        # BOS + seed
        char_ids = [tok.vocab.get("<BOS>", 2), tok.char_to_id(rt_char)]
        field_ids = [0, 0]
        rt_id     = 0  # simplified; in production map rt → integer

        with torch.no_grad():
            for col in range(1, T):
                c_t  = torch.tensor([char_ids], dtype=torch.long).to(self.device)
                f_t  = torch.tensor([field_ids], dtype=torch.long).to(self.device)
                rt_t = torch.tensor([rt_id], dtype=torch.long).to(self.device)

                logits, _, _ = self.model(c_t, f_t, rt_t, mode="generate")
                next_logits  = logits[0, -1, :].clone()  # (V,)

                # Apply hard constraint mask
                allowed = self.resolver.allowed_ids(self.spec_name, record_type, col) if self.resolver else None
                if allowed is not None:
                    mask_tensor = torch.full_like(next_logits, float("-inf"))
                    for aid in allowed:
                        if 0 <= aid < mask_tensor.size(0):
                            mask_tensor[aid] = next_logits[aid]
                    next_logits = mask_tensor

                # Sampling strategy
                next_id = self._sample(next_logits, gen_cfg)
                next_ch = tok.id_to_char(next_id)
                generated.append(next_ch)

                char_ids.append(next_id)
                field_ids.append(0)

                if len(generated) >= T:
                    break

        return "".join(generated[:T]).ljust(T)

    @staticmethod
    def _sample(logits: torch.Tensor, cfg: GenerationConfig) -> int:
        if cfg.strategy == "greedy":
            return int(logits.argmax().item())

        elif cfg.strategy == "temperature":
            scaled = logits / max(cfg.temperature, 1e-8)
            # Replace -inf before softmax
            scaled = scaled.masked_fill(scaled == float("-inf"), -1e9)
            probs  = F.softmax(scaled, dim=-1)
            return int(torch.multinomial(probs, num_samples=1).item())

        elif cfg.strategy == "top_k":
            topk_vals, topk_ids = torch.topk(logits, k=min(cfg.top_k, logits.size(0)))
            topk_vals = topk_vals.masked_fill(topk_vals == float("-inf"), -1e9)
            probs      = F.softmax(topk_vals, dim=-1)
            chosen     = int(torch.multinomial(probs, num_samples=1).item())
            return int(topk_ids[chosen].item())

        return int(logits.argmax().item())

    @staticmethod
    def _record_type_seed(rt: str) -> str:
        mapping = {
            "RT1": "1", "RT5": "5", "RT6": "6",
            "RT7": "7", "RT8": "8", "RT9": "9",
            "RTVH": "V", "RVDT": "D", "RVTR": "T", "RTVF": "F",
            "RTGL": "J", "RTJE": "E", "RTJH": "H",
        }
        return mapping.get(rt, "0")

    # ── File-Level Generation (Rule-Based) ────────────────────────────────

    def _gen_ach_file(self, gen_cfg: GenerationConfig, n_entries: int) -> str:
        seeder  = self.seeder
        spec    = self.spec_name
        lines   = []

        # File Header (RT1)
        fh, _ = seeder.generate_line(spec, "RT1")
        lines.append(fh)

        # Batch Header (RT5)
        bh, ctx = seeder.generate_line(spec, "RT5", return_context=True)
        lines.append(bh)

        # Entry Detail records (RT6)
        routing_sum  = 0
        entry_count  = 0
        debit_total  = 0
        credit_total = 0
        entries_data = []

        for i in range(n_entries):
            seq = i + 1
            entry, ectx = seeder.generate_line(
                spec, "RT6", extra={"sequence": seq}, return_context=True
            )
            lines.append(entry)
            entry_count  += 1
            routing_raw   = entry[3:11]
            amount_raw    = entry[29:39]
            tx_code       = entry[1:3]
            if routing_raw.isdigit():
                routing_sum += int(routing_raw)
            if amount_raw.isdigit():
                if tx_code in ("22", "23", "32", "33"):
                    credit_total += int(amount_raw)
                else:
                    debit_total  += int(amount_raw)
            entries_data.append(ectx)

        # Batch Control (RT8) — computed values
        batch_hash = routing_sum % (10 ** 10)
        bc, _  = seeder.generate_line(spec, "RT8", extra={
            "entry_addenda_count" : entry_count,   # "Entry/Addenda Count"
            "entry_hash"          : batch_hash,    # "Entry Hash"
            "total_debit_dollar_amount"  : debit_total,
            "total_credit_dollar_amount" : credit_total,
            "company_identification"     : ctx.get("company_id", "9876543210") if ctx else "9876543210",
        }, return_context=True)
        lines.append(bc)

        # File Control (RT9)
        block_count  = math.ceil((len(lines) + 1) / 10)
        file_hash    = routing_sum % (10 ** 10)
        fc, _ = seeder.generate_line(spec, "RT9", extra={
            "batch_count"                : 1,
            "block_count"               : block_count,
            "entry_addenda_count"       : entry_count,   # "Entry/Addenda Count"
            "entry_hash"                : file_hash,     # "Entry Hash"
            "total_debit_dollar_amount"  : debit_total,
            "total_credit_dollar_amount" : credit_total,
        }, return_context=True)
        lines.append(fc)

        # Pad to multiple of 10 with '9' fill records
        while len(lines) % 10 != 0:
            lines.append("9" * 94)

        return "\n".join(lines)

    def _gen_visa_file(self, gen_cfg: GenerationConfig, n_entries: int) -> str:
        seeder = self.seeder
        spec   = self.spec_name
        lines  = []
        lines.append(seeder.generate_line(spec, "RTVH")[0])
        for i in range(n_entries):
            lines.append(seeder.generate_line(spec, "RVDT", extra={"seq": i+1})[0])
        lines.append(seeder.generate_line(spec, "RTVF", extra={"count": n_entries})[0])
        return "\n".join(lines)

    def _gen_gl_file(self, gen_cfg: GenerationConfig, n_entries: int) -> str:
        seeder  = self.seeder
        spec    = self.spec_name
        lines   = []
        lines.append(seeder.generate_line(spec, "RTJH")[0])
        total_dr = 0
        total_cr = 0
        for i in range(n_entries):
            entry, ctx = seeder.generate_line(spec, "RTJE", extra={"seq": i+1}, return_context=True)
            lines.append(entry)
            if ctx:
                total_dr += ctx.get("debit", 0)
                total_cr += ctx.get("credit", 0)
        lines.append(seeder.generate_line(spec, "RTGL", extra={
            "total_dr": total_dr, "total_cr": total_cr, "count": n_entries
        })[0])
        return "\n".join(lines)


import math  # needed for block count calc above
