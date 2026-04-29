"""
Numpy-based Bigram / N-gram Language Model
Fallback when PyTorch is not available.
Less powerful than Transformer, but trains fast and produces valid-structured files.
"""

import numpy as np
import json
import os
import pickle
from collections import defaultdict
from typing import List, Dict, Tuple


class BigramSLM:
    """
    Character-level bigram model with n-gram smoothing.
    Serves as a lightweight fallback when PyTorch is unavailable.
    Not as powerful as the Transformer but produces recognizable ACH/VCF structure.
    """

    def __init__(self, order: int = 4):
        self.order = order  # n-gram order
        self.counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.vocab: set = set()
        self.file_type = None

    def train(self, texts: List[str], callback=None):
        """Train on a list of file strings"""
        total = len(texts)
        for i, text in enumerate(texts):
            self.vocab.update(text)
            for j in range(len(text) - self.order):
                ctx = text[j: j + self.order]
                nxt = text[j + self.order]
                self.counts[ctx][nxt] += 1

            if callback and i % 50 == 0:
                progress = 45 + int((i / total) * 50)
                callback(f"Training on file {i}/{total}...", progress)

        print(f"BigramSLM trained on {total} files, vocab={len(self.vocab)}")

    def generate(self, seed: str = None, max_chars: int = 2000,
                 temperature: float = 1.0) -> str:
        if seed is None:
            seed = "1" if self.file_type == "ACH" else "VCF|"

        result = list(seed)
        ctx = seed[-self.order:] if len(seed) >= self.order else seed.ljust(self.order)

        for _ in range(max_chars):
            key = ctx[-self.order:]
            if key in self.counts and self.counts[key]:
                chars = list(self.counts[key].keys())
                raw_counts = np.array([self.counts[key][c] for c in chars], dtype=float)
                # Temperature scaling
                log_probs = np.log(raw_counts + 1e-8) / temperature
                log_probs -= log_probs.max()
                probs = np.exp(log_probs)
                probs /= probs.sum()
                nxt = np.random.choice(chars, p=probs)
            else:
                # Backoff: use shorter context
                found = False
                for back in range(self.order - 1, 0, -1):
                    short_key = ctx[-back:]
                    if short_key in self.counts and self.counts[short_key]:
                        chars = list(self.counts[short_key].keys())
                        raw_counts = np.array([self.counts[short_key][c] for c in chars], dtype=float)
                        probs = raw_counts / raw_counts.sum()
                        nxt = np.random.choice(chars, p=probs)
                        found = True
                        break
                if not found:
                    nxt = np.random.choice(list(self.vocab)) if self.vocab else ' '

            result.append(nxt)
            ctx = ctx[1:] + nxt

            if nxt == '\x00':  # EOS
                break

        return ''.join(result)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'order': self.order,
                'counts': dict(self.counts),
                'vocab': self.vocab,
                'file_type': self.file_type,
            }, f)
        print(f"BigramSLM saved to {path}")

    @classmethod
    def load(cls, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls(order=data['order'])
        model.counts = defaultdict(lambda: defaultdict(int), {
            k: defaultdict(int, v) for k, v in data['counts'].items()
        })
        model.vocab = data['vocab']
        model.file_type = data.get('file_type')
        return model
