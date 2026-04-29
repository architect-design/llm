"""
Small Language Model (SLM) - Character-level Transformer
Designed for structured financial file generation (ACH NACHA & VISA VCF)
Architecture: GPT-style decoder-only transformer (~2-5M params)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import os


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, C)
        return self.W_o(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ff(self.ln2(x))
        return x


class FinancialSLM(nn.Module):
    """
    Small Language Model for Financial File Generation
    ~2-5M parameters depending on config
    Supports: ACH NACHA format, VISA VCF format
    """

    CONFIGS = {
        "nano": dict(d_model=128, n_heads=4, n_layers=4, d_ff=512),    # ~1M params
        "small": dict(d_model=256, n_heads=8, n_layers=6, d_ff=1024),  # ~4M params
        "medium": dict(d_model=384, n_heads=6, n_layers=8, d_ff=1536), # ~10M params
    }

    def __init__(self, vocab_size, max_seq_len=2048, config="small", dropout=0.1):
        super().__init__()
        cfg = self.CONFIGS[config]
        self.d_model = cfg["d_model"]
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size

        self.token_emb = nn.Embedding(vocab_size, cfg["d_model"])
        self.pos_emb = nn.Embedding(max_seq_len, cfg["d_model"])
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(cfg["d_model"], cfg["n_heads"], cfg["d_ff"], dropout)
            for _ in range(cfg["n_layers"])
        ])

        self.ln_f = nn.LayerNorm(cfg["d_model"])
        self.lm_head = nn.Linear(cfg["d_model"], vocab_size, bias=False)

        # Weight tying
        self.token_emb.weight = self.lm_head.weight

        self._init_weights()
        print(f"FinancialSLM ({config}) initialized with {self._count_params():,} parameters")

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _count_params(self):
        return sum(p.numel() for p in self.parameters())

    def _causal_mask(self, T, device):
        mask = torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(0)
        return mask

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds max {self.max_seq_len}"

        tok = self.token_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = self.dropout(tok + pos)

        mask = self._causal_mask(T, idx.device)
        for block in self.blocks:
            x = block(x, mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=50, top_p=0.9,
                 stop_token=None):
        """Generate tokens autoregressively with top-k/top-p sampling"""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_idx_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[sorted_idx_to_remove] = float('-inf')
                logits.scatter_(1, sorted_idx, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            if stop_token is not None and idx_next.item() == stop_token:
                break

            idx = torch.cat([idx, idx_next], dim=1)

        return idx

    def save(self, path, tokenizer=None, file_type=None):
        """Save model checkpoint"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            "model_state": self.state_dict(),
            "vocab_size": self.vocab_size,
            "max_seq_len": self.max_seq_len,
            "file_type": file_type,
        }
        if tokenizer:
            checkpoint["vocab"] = tokenizer.vocab
            checkpoint["char2idx"] = tokenizer.char2idx
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path, config="small"):
        """Load model from checkpoint"""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(
            vocab_size=checkpoint["vocab_size"],
            max_seq_len=checkpoint["max_seq_len"],
            config=config
        )
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        return model, checkpoint
