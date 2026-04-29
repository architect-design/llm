"""
Character-level tokenizer optimized for financial file formats (ACH/VCF)
Includes special tokens for record boundaries and field separators
"""

import json
from typing import List, Optional


class FinancialTokenizer:
    """
    Character-level tokenizer with special tokens for structured financial files.
    Vocabulary includes all printable ASCII chars + special control tokens.
    """

    SPECIAL_TOKENS = {
        "<PAD>": 0,
        "<BOS>": 1,   # Begin of sequence
        "<EOS>": 2,   # End of sequence
        "<UNK>": 3,   # Unknown character
        "<REC>": 4,   # Record separator
        "<ACH>": 5,   # ACH file type marker
        "<VCF>": 6,   # VCF file type marker
    }

    def __init__(self):
        self.char2idx = dict(self.SPECIAL_TOKENS)
        self.idx2char = {v: k for k, v in self.char2idx.items()}
        self._build_base_vocab()

    def _build_base_vocab(self):
        """Build vocab from printable ASCII characters"""
        next_idx = max(self.char2idx.values()) + 1
        for i in range(32, 127):  # Printable ASCII
            ch = chr(i)
            if ch not in self.char2idx:
                self.char2idx[ch] = next_idx
                self.idx2char[next_idx] = ch
                next_idx += 1
        # Add newline and tab
        for ch in ['\n', '\r', '\t']:
            if ch not in self.char2idx:
                self.char2idx[ch] = next_idx
                self.idx2char[next_idx] = ch
                next_idx += 1

    @property
    def vocab(self):
        return self.char2idx

    @property
    def vocab_size(self):
        return len(self.char2idx)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs"""
        tokens = []
        if add_special_tokens:
            tokens.append(self.char2idx["<BOS>"])
        for ch in text:
            tokens.append(self.char2idx.get(ch, self.char2idx["<UNK>"]))
        if add_special_tokens:
            tokens.append(self.char2idx["<EOS>"])
        return tokens

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """Decode token IDs to text"""
        special_ids = set(self.SPECIAL_TOKENS.values())
        result = []
        for idx in ids:
            if idx in special_ids:
                if not skip_special:
                    result.append(self.idx2char.get(idx, ""))
                continue
            result.append(self.idx2char.get(idx, "?"))
        return "".join(result)

    def encode_file(self, content: str, file_type: str = "ACH") -> List[int]:
        """Encode entire file with file type marker"""
        marker = self.char2idx.get(f"<{file_type}>", self.char2idx["<BOS>"])
        tokens = [marker] + self.encode(content, add_special_tokens=False)
        tokens.append(self.char2idx["<EOS>"])
        return tokens

    def save(self, path: str):
        data = {"char2idx": self.char2idx, "idx2char": {str(k): v for k, v in self.idx2char.items()}}
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str):
        tok = cls.__new__(cls)
        with open(path) as f:
            data = json.load(f)
        tok.char2idx = data["char2idx"]
        tok.idx2char = {int(k): v for k, v in data["idx2char"].items()}
        return tok

    def create_training_batch(self, texts: List[str], block_size: int, file_type: str = "ACH"):
        """Create (inputs, targets) pairs for training"""
        import torch
        all_tokens = []
        for text in texts:
            all_tokens.extend(self.encode_file(text, file_type))

        inputs, targets = [], []
        for i in range(0, len(all_tokens) - block_size - 1, block_size):
            chunk = all_tokens[i: i + block_size + 1]
            if len(chunk) == block_size + 1:
                inputs.append(chunk[:-1])
                targets.append(chunk[1:])

        return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)
