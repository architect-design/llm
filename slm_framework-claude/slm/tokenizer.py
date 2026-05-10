"""
FinancialTokenizer — Domain-Specific Tokenizer for Fixed-Width Financial Formats.

Unlike NLP tokenizers that split on words or subwords, financial files are
structured by CHARACTER POSITION. This tokenizer understands:
  - Fixed-width field boundaries (position-based slicing)
  - Record-type prefixes (e.g., '1', '5', '6', '8', '9' in ACH NACHA)
  - Numeric, alphanumeric, and blank-padded field semantics
  - Mandatory offsets and file-level alignment rules
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

# ─────────────────────────── Token Types ────────────────────────────────────

class TokenType(Enum):
    RECORD_TYPE   = "RECORD_TYPE"    # single-char record identifier
    NUMERIC       = "NUMERIC"        # pure digit field
    ALPHANUMERIC  = "ALPHANUMERIC"   # mixed chars, often left-justified
    BLANK_PAD     = "BLANK_PAD"      # mandatory space padding
    DELIMITER     = "DELIMITER"      # structural separator (e.g., newline)
    CHECKSUM      = "CHECKSUM"       # computed checksum / hash field
    AMOUNT        = "AMOUNT"         # zero-padded monetary amount (implied decimal)
    ROUTING       = "ROUTING"        # ABA routing / transit number
    ACCOUNT       = "ACCOUNT"        # account number field
    DATE          = "DATE"           # YYMMDD or similar date token
    UNKNOWN       = "UNKNOWN"        # fallback for unrecognised content


@dataclass
class FinancialToken:
    token_type : TokenType
    raw_value  : str          # exact characters from file
    position   : int          # absolute character offset in line
    length     : int          # field width
    field_name : str = ""     # human label from spec (e.g. "Routing Number")
    line_no    : int = 0      # source line index

    @property
    def token_id(self) -> int:
        """Deterministic integer ID for embedding lookup."""
        return _VOCAB.get(self.raw_value.strip(), _VOCAB.get("<UNK>", 0))

    def __repr__(self) -> str:
        return (
            f"FT({self.token_type.value}|{self.field_name!r}|"
            f"pos={self.position}|len={self.length}|{self.raw_value!r})"
        )


# ─────────────────────── Vocabulary Construction ────────────────────────────

def _build_vocab() -> Dict[str, int]:
    """
    Builds a position-aware character vocabulary.

    Financial files work with a closed, deterministic character set:
      - Digits 0-9
      - Upper-case A-Z  (NACHA spec: alphanumeric fields are upper-cased)
      - Space ' '
      - Special chars: / - . @ # *
    We add reserved tokens for structural roles.
    """
    vocab: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3,
                              "<SEP>": 4, "<MASK>": 5}
    idx = len(vocab)
    for ch in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ /-.:@#*":
        vocab[ch] = idx
        idx += 1
    # Record-type sentinel tokens
    for rt in ["RT1", "RT5", "RT6", "RT7", "RT8", "RT9",
               "RTVH", "RVDT", "RVTR", "RTVF",  # VISA VCF
               "RTGL", "RTJE", "RTJH"]:          # General Ledger
        vocab[rt] = idx
        idx += 1
    return vocab


_VOCAB: Dict[str, int] = _build_vocab()
VOCAB_SIZE: int = len(_VOCAB)
_ID2TOKEN: Dict[int, str] = {v: k for k, v in _VOCAB.items()}


# ──────────────────────── Field Descriptor ──────────────────────────────────

@dataclass
class FieldDescriptor:
    """Describes a single fixed-width field within a record."""
    name       : str
    start      : int          # 1-based, matches NACHA spec column refs
    end        : int          # inclusive
    field_type : TokenType
    required   : bool = True
    pattern    : Optional[str] = None   # regex the raw value must match
    allowed    : Optional[List[str]] = None  # whitelist of literal values

    @property
    def length(self) -> int:
        return self.end - self.start + 1

    def extract(self, line: str) -> str:
        """Slice the raw value from a fixed-width line (0-indexed internally)."""
        s, e = self.start - 1, self.end
        return line[s:e] if len(line) >= e else line[s:].ljust(self.length)


# ─────────────────────── Core Tokenizer Class ────────────────────────────────

class FinancialTokenizer:
    """
    Converts raw financial file text into sequences of FinancialTokens.

    Usage:
        tok = FinancialTokenizer(spec_name="ACH_NACHA")
        tokens = tok.tokenize(raw_file_text)
        ids    = tok.tokens_to_ids(tokens)
        back   = tok.ids_to_tokens(ids)
    """

    LINE_LENGTH_MAP: Dict[str, int] = {
        "ACH_NACHA"      : 94,
        "VISA_VCF"       : 80,
        "GENERAL_LEDGER" : 120,
    }

    def __init__(self, spec_name: str, field_schema: Optional[Dict] = None):
        if spec_name not in self.LINE_LENGTH_MAP:
            raise ValueError(f"Unknown spec: {spec_name}. "
                             f"Choose from {list(self.LINE_LENGTH_MAP)}")
        self.spec_name   = spec_name
        self.line_length = self.LINE_LENGTH_MAP[spec_name]
        self.field_schema: Dict[str, List[FieldDescriptor]] = field_schema or {}
        self.vocab       = _VOCAB
        self.vocab_size  = VOCAB_SIZE

    # ── Public API ──────────────────────────────────────────────────────────

    def tokenize(self, raw_text: str) -> List[List[FinancialToken]]:
        """
        Tokenize a full financial file.
        Returns a list-of-lists: outer = lines, inner = field tokens.
        """
        lines = self._normalise_lines(raw_text)
        tokenised: List[List[FinancialToken]] = []
        for line_no, line in enumerate(lines):
            if not line.strip():
                continue
            record_type = self._detect_record_type(line)
            fields      = self.field_schema.get(record_type, [])
            row_tokens  = self._tokenize_line(line, fields, line_no, record_type)
            tokenised.append(row_tokens)
        return tokenised

    def tokens_to_ids(
        self, tokenised: List[List[FinancialToken]]
    ) -> List[List[int]]:
        """Convert token lists to integer ID lists (for model input)."""
        return [
            [_VOCAB.get(ch, _VOCAB["<UNK>"])
             for tok in row
             for ch in tok.raw_value]
            for row in tokenised
        ]

    def ids_to_text(self, ids: List[int]) -> str:
        """Decode a flat ID sequence back to raw text."""
        return "".join(_ID2TOKEN.get(i, "?") for i in ids)

    def char_to_id(self, ch: str) -> int:
        return _VOCAB.get(ch.upper(), _VOCAB["<UNK>"])

    def id_to_char(self, idx: int) -> str:
        return _ID2TOKEN.get(idx, "?")

    def encode_line(self, line: str) -> List[int]:
        """Encode a single raw line to a padded integer sequence."""
        line = line.upper().ljust(self.line_length)[:self.line_length]
        return [self.char_to_id(ch) for ch in line]

    def decode_ids(self, ids: List[int]) -> str:
        return "".join(self.id_to_char(i) for i in ids)

    def get_record_type(self, line: str) -> str:
        return self._detect_record_type(line)

    # ── Private Helpers ──────────────────────────────────────────────────────

    def _normalise_lines(self, raw: str) -> List[str]:
        """Strip CRLF, enforce line length, upper-case per spec."""
        lines = raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        normed = []
        for ln in lines:
            ln = ln.upper()
            if len(ln) < self.line_length:
                ln = ln.ljust(self.line_length)
            normed.append(ln[:self.line_length])
        return normed

    def _detect_record_type(self, line: str) -> str:
        if self.spec_name == "ACH_NACHA":
            return f"RT{line[0]}" if line else "RTUNK"
        elif self.spec_name == "VISA_VCF":
            code = line[:2].strip()
            return f"RV{code}" if code else "RVUNK"
        elif self.spec_name == "GENERAL_LEDGER":
            code = line[:2].strip()
            return f"GL{code}" if code else "GLUNK"
        return "RTUNK"

    def _tokenize_line(
        self,
        line: str,
        fields: List[FieldDescriptor],
        line_no: int,
        record_type: str,
    ) -> List[FinancialToken]:
        tokens: List[FinancialToken] = []

        # Always emit the record-type sentinel first
        tokens.append(FinancialToken(
            token_type = TokenType.RECORD_TYPE,
            raw_value  = record_type,
            position   = 0,
            length     = 1,
            field_name = "Record Type Code",
            line_no    = line_no,
        ))

        if fields:
            for fd in fields:
                raw = fd.extract(line)
                tokens.append(FinancialToken(
                    token_type = fd.field_type,
                    raw_value  = raw,
                    position   = fd.start - 1,
                    length     = fd.length,
                    field_name = fd.name,
                    line_no    = line_no,
                ))
        else:
            # Fallback: character-by-character tokenisation
            for pos, ch in enumerate(line):
                tokens.append(FinancialToken(
                    token_type = self._infer_type(ch),
                    raw_value  = ch,
                    position   = pos,
                    length     = 1,
                    field_name = f"col_{pos+1}",
                    line_no    = line_no,
                ))
        return tokens

    @staticmethod
    def _infer_type(ch: str) -> TokenType:
        if ch.isdigit():
            return TokenType.NUMERIC
        if ch == " ":
            return TokenType.BLANK_PAD
        if ch.isalpha():
            return TokenType.ALPHANUMERIC
        return TokenType.UNKNOWN


# ─────────────────────── Convenience Factory ─────────────────────────────────

def make_tokenizer(spec_name: str, field_schema: Optional[Dict] = None) -> FinancialTokenizer:
    """Factory that wires the tokenizer to the appropriate spec schema."""
    from specs.ach_nacha      import ACH_FIELD_SCHEMA
    from specs.visa_vcf       import VISA_FIELD_SCHEMA
    from specs.general_ledger import GL_FIELD_SCHEMA

    schemas = {
        "ACH_NACHA"      : ACH_FIELD_SCHEMA,
        "VISA_VCF"       : VISA_FIELD_SCHEMA,
        "GENERAL_LEDGER" : GL_FIELD_SCHEMA,
    }
    schema = field_schema or schemas.get(spec_name, {})
    return FinancialTokenizer(spec_name=spec_name, field_schema=schema)
