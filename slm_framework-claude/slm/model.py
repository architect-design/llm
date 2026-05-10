"""
FinancialSLM — Custom Transformer Architecture for Financial File Formats.

Architecture choices driven by the unique structure of financial files:

1. POSITIONAL ENCODING — Sinusoidal + learnable FIELD-POSITION encoding.
   A standard NLP model encodes token order; we additionally encode
   *character column position* (0–119) so the model learns that
   column 4 in an ACH file is always "Service Class Code."

2. MULTI-HEAD ATTENTION — Standard but with a causal mask for generation
   and a full mask for validation (bidirectional context).

3. DUAL OUTPUT HEADS:
   a) Generation Head  → next-character probability distribution (autoregressive)
   b) Validation Head  → per-field binary valid/invalid classification

4. FIELD-AWARE EMBEDDING — Record-type tokens receive a dedicated embedding
   table separate from character embeddings, letting the model specialise
   attention patterns per record type.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────── Configuration ───────────────────────────────────────

@dataclass
class SLMConfig:
    vocab_size     : int   = 64      # character vocab (digits + alpha + specials)
    d_model        : int   = 128     # embedding dimension
    n_heads        : int   = 4       # attention heads  (d_model must be divisible)
    n_layers       : int   = 4       # transformer blocks
    d_ff           : int   = 512     # feed-forward hidden dim
    max_seq_len    : int   = 120     # longest supported line (GL = 120 chars)
    dropout        : float = 0.1
    n_record_types : int   = 32      # distinct record-type sentinel tokens
    n_field_slots  : int   = 40      # max named fields per record
    pad_token_id   : int   = 0


# ──────────────────────── Positional Encodings ────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal PE — encodes absolute character column."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class FieldSlotEncoding(nn.Module):
    """
    Learnable encoding for FIELD SLOT position (which named field this
    character belongs to within a record, e.g., field 3 of 9 in ACH RT6).
    This is separate from character-column PE.
    """

    def __init__(self, n_field_slots: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(n_field_slots, d_model)

    def forward(self, field_ids: torch.Tensor) -> torch.Tensor:
        # field_ids: (batch, seq_len) — integer field slot index per character
        return self.embedding(field_ids)


# ─────────────────────── Attention & Transformer Blocks ─────────────────────

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_k     = d_model // n_heads

        self.q_proj  = nn.Linear(d_model, d_model, bias=False)
        self.k_proj  = nn.Linear(d_model, d_model, bias=False)
        self.v_proj  = nn.Linear(d_model, d_model, bias=False)
        self.out     = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, _ = q.shape

        def split_heads(x: torch.Tensor) -> torch.Tensor:
            return x.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        Q, K, V = split_heads(self.q_proj(q)), split_heads(self.k_proj(k)), split_heads(self.v_proj(v))

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn   = self.dropout(F.softmax(scores, dim=-1))
        out    = (attn @ V).transpose(1, 2).contiguous().view(B, T, -1)
        return self.out(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: SLMConfig):
        super().__init__()
        self.attn    = MultiHeadAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.ff      = FeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)
        self.norm1   = nn.LayerNorm(cfg.d_model)
        self.norm2   = nn.LayerNorm(cfg.d_model)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.norm1(x + self.attn(x, x, x, mask))
        x = self.norm2(x + self.ff(x))
        return x


# ───────────────────────── Dual-Head SLM ─────────────────────────────────────

class FinancialSLM(nn.Module):
    """
    The core Small Language Model.

    Inputs per forward pass:
      char_ids   : (B, T) — character token IDs
      field_ids  : (B, T) — field-slot IDs per character (0 if unknown)
      rt_ids     : (B,)   — record-type ID for each line in batch
      mode       : 'generate' (causal) | 'validate' (bidirectional)

    Outputs:
      GenerationOutput  — logits (B, T, vocab_size)
      ValidationOutput  — field_logits (B, n_fields, 2), confidence (B,)
    """

    def __init__(self, cfg: SLMConfig):
        super().__init__()
        self.cfg = cfg

        # ── Embedding Stack ───────────────────────────────────────────────
        self.char_embed   = nn.Embedding(cfg.vocab_size, cfg.d_model,
                                         padding_idx=cfg.pad_token_id)
        self.rt_embed     = nn.Embedding(cfg.n_record_types, cfg.d_model)
        self.field_pe     = FieldSlotEncoding(cfg.n_field_slots, cfg.d_model)
        self.sin_pe       = SinusoidalPositionalEncoding(cfg.d_model,
                                                          cfg.max_seq_len,
                                                          cfg.dropout)
        self.embed_proj   = nn.Linear(cfg.d_model * 3, cfg.d_model)

        # ── Transformer Blocks ────────────────────────────────────────────
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm   = nn.LayerNorm(cfg.d_model)

        # ── Head A: Autoregressive Generation ────────────────────────────
        self.gen_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # ── Head B: Per-Field Validation ─────────────────────────────────
        # Pools over field boundaries and classifies each as valid/invalid
        self.val_pool = nn.AdaptiveAvgPool1d(cfg.n_field_slots)
        self.val_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Linear(cfg.d_model // 2, 2),  # binary: valid / invalid
        )
        # Global confidence score for the entire record
        self.conf_head = nn.Sequential(
            nn.Linear(cfg.d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ── Mask Builders ─────────────────────────────────────────────────────

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular mask for autoregressive decoding (bool tensor)."""
        return torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)

    @staticmethod
    def _pad_mask(char_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
        """Mask padding positions (bool tensor)."""
        return (char_ids != pad_id).unsqueeze(1).unsqueeze(2)

    # ── Embedding Forward ─────────────────────────────────────────────────

    def _embed(
        self,
        char_ids : torch.Tensor,   # (B, T)
        field_ids: torch.Tensor,   # (B, T)
        rt_ids   : torch.Tensor,   # (B,)
    ) -> torch.Tensor:
        B, T = char_ids.shape

        c_emb  = self.char_embed(char_ids)                       # (B, T, D)
        f_emb  = self.field_pe(field_ids)                        # (B, T, D)
        rt_emb = self.rt_embed(rt_ids).unsqueeze(1).expand(B, T, -1)  # (B, T, D)

        combined = torch.cat([c_emb, f_emb, rt_emb], dim=-1)    # (B, T, 3D)
        x = self.embed_proj(combined)                             # (B, T, D)
        x = self.sin_pe(x)
        return x

    # ── Main Forward ──────────────────────────────────────────────────────

    def forward(
        self,
        char_ids  : torch.Tensor,
        field_ids : torch.Tensor,
        rt_ids    : torch.Tensor,
        mode      : str = "generate",
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Returns:
            gen_logits  : (B, T, V) always returned
            val_logits  : (B, n_fields, 2) in validate mode, else None
            confidence  : (B, 1) in validate mode, else None
        """
        B, T = char_ids.shape
        x = self._embed(char_ids, field_ids, rt_ids)

        if mode == "generate":
            mask = self._causal_mask(T, char_ids.device) & self._pad_mask(char_ids, self.cfg.pad_token_id)
        else:
            mask = self._pad_mask(char_ids, self.cfg.pad_token_id)

        for block in self.blocks:
            x = block(x, mask)
        x = self.norm(x)  # (B, T, D)

        gen_logits = self.gen_head(x)  # (B, T, V)

        val_logits = None
        confidence = None
        if mode == "validate":
            # Pool hidden states across character sequence → field slots
            x_t = x.transpose(1, 2)                    # (B, D, T)
            pooled = self.val_pool(x_t).transpose(1, 2) # (B, n_fields, D)
            val_logits = self.val_head(pooled)          # (B, n_fields, 2)

            # Global record confidence from [CLS]-like mean pool
            cls = x.mean(dim=1)                         # (B, D)
            confidence = self.conf_head(cls)            # (B, 1)

        return gen_logits, val_logits, confidence

    # ── Convenience: count parameters ─────────────────────────────────────

    def param_count(self) -> Dict[str, int]:
        total   = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

    def __repr__(self) -> str:
        pc = self.param_count()
        return (
            f"FinancialSLM(d={self.cfg.d_model}, "
            f"layers={self.cfg.n_layers}, "
            f"heads={self.cfg.n_heads}, "
            f"params={pc['total']:,})"
        )


# ──────────────────────── Model Factory ──────────────────────────────────────

def build_model(spec_name: str) -> Tuple[FinancialSLM, SLMConfig]:
    """
    Returns a pre-configured model for a given financial spec.
    Larger specs (GL = 120 cols) get a slightly wider model.
    """
    from slm.tokenizer import VOCAB_SIZE

    base = SLMConfig(vocab_size=VOCAB_SIZE)

    overrides = {
        "ACH_NACHA"      : {"max_seq_len": 94,  "n_layers": 4, "d_model": 128},
        "VISA_VCF"       : {"max_seq_len": 80,  "n_layers": 4, "d_model": 128},
        "GENERAL_LEDGER" : {"max_seq_len": 120, "n_layers": 6, "d_model": 192, "d_ff": 768},
    }

    cfg_dict = {
        "vocab_size"     : base.vocab_size,
        "d_model"        : base.d_model,
        "n_heads"        : base.n_heads,
        "n_layers"       : base.n_layers,
        "d_ff"           : base.d_ff,
        "max_seq_len"    : base.max_seq_len,
        "dropout"        : base.dropout,
        "n_record_types" : base.n_record_types,
        "n_field_slots"  : base.n_field_slots,
        "pad_token_id"   : base.pad_token_id,
    }
    cfg_dict.update(overrides.get(spec_name, {}))

    # n_heads must divide d_model
    d = cfg_dict["d_model"]
    for h in [8, 6, 4, 3, 2, 1]:
        if d % h == 0:
            cfg_dict["n_heads"] = h
            break

    cfg = SLMConfig(**cfg_dict)
    model = FinancialSLM(cfg)
    return model, cfg
