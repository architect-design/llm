"""
financial_slm_framework/slm_core/tokenizer.py
Domain-specific tokenizer handling fixed-width fields, offsets, and delimiters.
"""

from typing import List, Dict, Optional, Tuple
import json


class FinancialTokenizer:
    BASE_VOCAB = (
        "0123456789"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
        " !\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
        "\t\n\r"
    )

    SPECIAL_TOKENS = ["<PAD>", "<SOS>", "<EOS>", "<UNK>", "<FB>", "<RB>", "<SPACE>"]

    def __init__(self, max_record_types: int = 50):
        self.max_record_types = max_record_types
        self.vocab = {}
        self.idx_to_char = {}

        for i, token in enumerate(self.SPECIAL_TOKENS):
            self.vocab[token] = i
            self.idx_to_char[i] = token

        for i in range(max_record_types):
            token = f"<RT_{i}>"
            idx = len(self.vocab)
            self.vocab[token] = idx
            self.idx_to_char[idx] = token

        for char in self.BASE_VOCAB:
            if char not in self.vocab:
                idx = len(self.vocab)
                self.vocab[char] = idx
                self.idx_to_char[idx] = char

        self.vocab_size = len(self.vocab)
        self.PAD_ID = self.vocab["<PAD>"]
        self.SOS_ID = self.vocab["<SOS>"]
        self.EOS_ID = self.vocab["<EOS>"]
        self.UNK_ID = self.vocab["<UNK>"]
        self.FB_ID = self.vocab["<FB>"]
        self.RB_ID = self.vocab["<RB>"]
        self.SPACE_ID = self.vocab.get("<SPACE>", self.vocab.get(" ", self.UNK_ID))

    def encode(self, text: str, record_type: Optional[int] = None,
               add_special_tokens: bool = True, max_length: Optional[int] = None,
               pad_to_max_length: bool = False) -> List[int]:
        tokens = []
        if add_special_tokens:
            tokens.append(self.SOS_ID)
        if record_type is not None:
            rt_token = f"<RT_{record_type}>"
            tokens.append(self.vocab.get(rt_token, self.UNK_ID))
        for char in text:
            if char in self.vocab:
                tokens.append(self.vocab[char])
            elif char == ' ':
                tokens.append(self.SPACE_ID)
            else:
                tokens.append(self.UNK_ID)
        if add_special_tokens:
            tokens.append(self.EOS_ID)
        if max_length is not None:
            tokens = tokens[:max_length]
        if pad_to_max_length and max_length is not None:
            while len(tokens) < max_length:
                tokens.append(self.PAD_ID)
        return tokens

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        chars = []
        for idx in token_ids:
            char = self.idx_to_char.get(idx, "<UNK>")
            if skip_special_tokens and char in self.SPECIAL_TOKENS:
                continue
            if char.startswith("<RT_"):
                continue
            chars.append(char)
        return "".join(chars)

    def encode_fixed_width_record(self, record: str,
                                   field_boundaries: List[Tuple[int, int]],
                                   record_type: int) -> List[int]:
        tokens = [self.SOS_ID, self.vocab.get(f"<RT_{record_type}>", self.UNK_ID)]
        for i, (start, end) in enumerate(field_boundaries):
            field_value = record[start:end]
            for char in field_value:
                tokens.append(self.vocab.get(char, self.UNK_ID))
            if i < len(field_boundaries) - 1:
                tokens.append(self.FB_ID)
        tokens.append(self.RB_ID)
        tokens.append(self.EOS_ID)
        return tokens

    def decode_fixed_width_record(self, token_ids: List[int],
                                   field_lengths: List[int]) -> Tuple[str, List[str]]:
        chars = []
        fields = []
        current_field = []
        for idx in token_ids:
            char = self.idx_to_char.get(idx, "")
            if char in ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]:
                continue
            if char.startswith("<RT_"):
                continue
            if char == "<FB>":
                fields.append("".join(current_field))
                current_field = []
            elif char == "<RB>":
                fields.append("".join(current_field))
                current_field = []
            else:
                current_field.append(char)
                chars.append(char)
        if current_field:
            fields.append("".join(current_field))
        return "".join(chars), fields

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump({'vocab': self.vocab, 'max_record_types': self.max_record_types}, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "FinancialTokenizer":
        with open(path, 'r') as f:
            data = json.load(f)
        tokenizer = cls(max_record_types=data['max_record_types'])
        tokenizer.vocab = {k: int(v) for k, v in data['vocab'].items()}
        tokenizer.idx_to_char = {int(v): k for k, v in tokenizer.vocab.items()}
        tokenizer.vocab_size = len(tokenizer.vocab)
        return tokenizer
