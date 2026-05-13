"""
financial_slm_framework/slm_core/model.py
Custom Transformer architecture optimized for fixed-width financial file parsing.
No external LLM APIs. Pure PyTorch implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, List, Tuple


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding adapted for character-level financial data."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class FinancialTransformerBlock(nn.Module):
    """Transformer block with masked attention for auto-regressive generation."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out, _ = self.attention(x, x, x, attn_mask=mask, need_weights=False)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x


class FinancialSLM(nn.Module):
    """
    Small Language Model for financial file format processing.
    Character-level with fixed-width field awareness.
    Dual heads: Generation + Validation.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 8,
        n_heads: int = 8,
        d_ff: int = 1024,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        num_record_types: int = 20,
        num_validation_classes: int = 3
    ):
        super().__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size

        self.char_embedding = nn.Embedding(vocab_size, d_model)
        self.record_type_embedding = nn.Embedding(num_record_types, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        self.transformer_blocks = nn.ModuleList([
            FinancialTransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.generation_head = nn.Linear(d_model, vocab_size)

        self.validation_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_validation_classes)
        )

        self.field_boundary_head = nn.Linear(d_model, 1)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(
        self,
        input_ids: torch.Tensor,
        record_type_ids: Optional[torch.Tensor] = None,
        return_validation: bool = False,
        return_boundaries: bool = False
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        x = self.char_embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        if record_type_ids is not None:
            record_emb = self.record_type_embedding(record_type_ids).unsqueeze(1)
            x = x + record_emb

        causal_mask = self._generate_causal_mask(seq_len, device)

        for block in self.transformer_blocks:
            x = block(x, causal_mask)

        x = self.norm(x)

        outputs = {'generation_logits': self.generation_head(x)}

        if return_validation:
            pooled = x.mean(dim=1)
            outputs['validation_logits'] = self.validation_head(pooled)

        if return_boundaries:
            outputs['boundary_logits'] = self.field_boundary_head(x).squeeze(-1)

        return outputs

    def generate(
        self,
        prompt: torch.Tensor,
        record_type_id: Optional[int] = None,
        max_length: int = 1000,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        constraint_fn: Optional[callable] = None,
        eos_token_id: Optional[int] = None
    ) -> torch.Tensor:
        self.eval()
        device = prompt.device
        generated = prompt.clone()

        record_type_tensor = None
        if record_type_id is not None:
            record_type_tensor = torch.tensor([record_type_id], device=device)

        with torch.no_grad():
            for pos in range(max_length):
                outputs = self.forward(generated, record_type_ids=record_type_tensor)
                next_token_logits = outputs['generation_logits'][0, -1, :] / temperature

                if constraint_fn is not None:
                    next_token_logits = constraint_fn(next_token_logits, generated.size(1))

                if top_k is not None:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')

                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

                if eos_token_id is not None and next_token.item() == eos_token_id:
                    break
                if generated.size(1) >= self.max_seq_len:
                    break

        return generated
