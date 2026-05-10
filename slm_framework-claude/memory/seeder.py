"""
DataSeeder — Randomised yet valid mock data injection pipeline.

Generates spec-correct synthetic financial records using only
Python's standard library (no Faker, no external deps).

Seeding strategy per field type:
  ROUTING    → valid ABA routing numbers with correct check digit
  ACCOUNT    → random 10-17 digit account numbers
  AMOUNT     → zero-padded amounts in plausible financial ranges
  ALPHANUMERIC → realistic names/descriptions from embedded word lists
  DATE       → valid YYMMDD dates within recent ±2 year window
  NUMERIC    → range-aware random integers, zero-padded
  BLANK_PAD  → spaces to spec width

All values are seeded deterministically when a seed is provided,
enabling reproducible test file generation.
"""

from __future__ import annotations

import random
import string
import datetime
from typing import Any, Dict, List, Optional, Tuple


# ─────────────────────── Embedded Word Lists ─────────────────────────────────

_FIRST_NAMES = [
    "JAMES", "MARY", "JOHN", "PATRICIA", "ROBERT", "JENNIFER", "MICHAEL",
    "LINDA", "WILLIAM", "BARBARA", "DAVID", "ELIZABETH", "RICHARD", "SUSAN",
    "JOSEPH", "JESSICA", "THOMAS", "SARAH", "CHARLES", "KAREN", "CHRISTOPHER",
    "LISA", "DANIEL", "NANCY", "MATTHEW", "BETTY", "ANTHONY", "MARGARET",
    "MARK", "SANDRA", "DONALD", "ASHLEY", "STEVEN", "KIMBERLY", "PAUL",
    "EMILY", "ANDREW", "DONNA", "KENNETH", "MICHELLE", "GEORGE", "CAROL",
]

_LAST_NAMES = [
    "SMITH", "JOHNSON", "WILLIAMS", "BROWN", "JONES", "GARCIA", "MILLER",
    "DAVIS", "RODRIGUEZ", "MARTINEZ", "HERNANDEZ", "LOPEZ", "GONZALEZ",
    "WILSON", "ANDERSON", "THOMAS", "TAYLOR", "MOORE", "JACKSON", "MARTIN",
    "LEE", "PEREZ", "THOMPSON", "WHITE", "HARRIS", "SANCHEZ", "CLARK",
    "RAMIREZ", "LEWIS", "ROBINSON", "WALKER", "YOUNG", "ALLEN", "KING",
    "WRIGHT", "SCOTT", "TORRES", "NGUYEN", "HILL", "FLORES", "GREEN",
]

_COMPANY_NAMES = [
    "APEX FINANCIAL CORP", "SUMMIT BANK NA", "LAKESIDE CREDIT UNION",
    "PINNACLE TRUST CO", "RIVERSIDE SAVINGS BANK", "KEYSTONE FEDERAL",
    "HARBOR NATIONAL BANK", "HIGHLAND SECURITIES LLC", "PARKVIEW FINANCE",
    "MEADOW BROOK BANK", "CRESTLINE FINANCIAL", "STERLING CAPITAL GROUP",
    "WESTFIELD TRUST", "NORTHGATE LENDING", "CLEARWATER BANK",
    "BLUE RIDGE FINANCIAL", "CORNERSTONE BANK", "GRANITE TRUST CO",
    "IRONBRIDGE CAPITAL", "CEDARWOOD SAVINGS",
]

_ENTRY_DESCRIPTIONS = [
    "PAYROLL", "VENDOR PMT", "REFUND", "TRANSFER", "DIRECT DEP",
    "UTILITIES", "RENT PMT", "INSURANCE", "DIVIDEND", "INTEREST",
    "TAX REFUND", "LOAN PMT", "BONUS PMT", "COMMISSION", "SALARY",
    "REIMBURSE", "HEALTH INS", "PENSION", "ANNUITY", "ROYALTIES",
]

_GL_ACCOUNT_CODES = [
    "1000", "1100", "1200", "1300", "1400", "1500",  # Assets
    "2000", "2100", "2200", "2300",                   # Liabilities
    "3000", "3100",                                    # Equity
    "4000", "4100", "4200", "4500",                   # Revenue
    "5000", "5100", "5200", "5300", "5400", "5500",   # Expenses
    "6000", "6100", "6200",                            # Other
]

_GL_DESCRIPTIONS = [
    "SALARY EXPENSE", "OFFICE SUPPLIES", "RENT EXPENSE", "UTILITIES EXP",
    "ACCOUNTS RECEIVABLE", "CASH AND EQUIV", "PREPAID INSURANCE",
    "ACCOUNTS PAYABLE", "ACCRUED LIABILITIES", "RETAINED EARNINGS",
    "SERVICE REVENUE", "INTEREST INCOME", "DEPRECIATION EXP",
    "INSURANCE EXPENSE", "ADVERTISING EXP", "TRAVEL EXPENSE",
    "PROFESSIONAL FEES", "BANK CHARGES", "MISCELLANEOUS EXP",
]

_LEDGER_CODES = ["MAIN", "CORP", "CONS", "MGMT", "SUBL", "INTL"]
_CURRENCY_CODES = ["USD", "EUR", "GBP", "CAD", "AUD", "JPY", "CHF"]
_MCC_CODES = ["5411", "5812", "7011", "4111", "5912", "5311", "7372", "5734"]


# ─────────────────────── ABA Routing Number Generator ────────────────────────

def _generate_routing() -> str:
    """
    Generate a valid 9-digit ABA routing number with correct check digit.
    Uses the Mod-10 weighted algorithm: 3·d1 + 7·d2 + 1·d3 + ... must be divisible by 10.
    """
    weights = [3, 7, 1, 3, 7, 1, 3, 7]
    while True:
        digits = [random.randint(0, 9) for _ in range(8)]
        total  = sum(d * w for d, w in zip(digits, weights))
        check  = (10 - (total % 10)) % 10
        routing = "".join(str(d) for d in digits) + str(check)
        # First two digits should be 01-12 (Federal Reserve routing prefix)
        if 1 <= int(routing[:2]) <= 12:
            return routing
        # or 21-32 (thrift institution prefix)


# ─────────────────────── Amount Generator ────────────────────────────────────

def _generate_amount(width: int = 10, max_dollars: int = 99_999) -> str:
    """Zero-padded implied-decimal amount. $0.01 to $max_dollars.99"""
    cents = random.randint(1, max_dollars * 100)
    return str(cents).zfill(width)


# ─────────────────────── Date Generator ──────────────────────────────────────

def _generate_date_yymmdd() -> str:
    """Random date within ±18 months of today."""
    today   = datetime.date.today()
    delta   = random.randint(-540, 540)
    target  = today + datetime.timedelta(days=delta)
    return target.strftime("%y%m%d")


# ─────────────────────── Main Seeder Class ───────────────────────────────────

class DataSeeder:
    """
    Generates randomised yet specification-valid field values.

    Usage:
        from memory.config_engine import ConfigEngine
        engine = ConfigEngine()
        seeder = DataSeeder(engine, seed=42)

        line, ctx = seeder.generate_line("ACH_NACHA", "RT6", return_context=True)
        print(line)  # 94-char ACH Entry Detail record
    """

    def __init__(self, config_engine, seed: Optional[int] = None):
        self.engine = config_engine
        self.rng    = random.Random(seed)
        if seed is not None:
            random.seed(seed)

    # ── Public API ─────────────────────────────────────────────────────────

    def generate_line(
        self,
        spec_name   : str,
        record_type : str,
        extra       : Optional[Dict] = None,
        return_context: bool = False,
    ) -> Tuple[str, Optional[Dict]]:
        """
        Generate a single specification-valid record line.

        Returns:
            (line_str, context_dict)  if return_context=True
            (line_str, None)          otherwise

        `context_dict` contains the generated field values for use by
        the caller (e.g., to compute checksums for control records).
        """
        extra   = extra or {}
        fields  = self.engine.get_fields(spec_name, record_type)
        length  = self.engine.get_line_length(spec_name)

        line    = list(" " * length)
        context : Dict[str, Any] = {}

        for fd in fields:
            s, e    = fd["start"] - 1, fd["end"]
            width   = fd["length"]
            fname   = fd["name"]
            ftype   = fd["field_type"]
            allowed = fd.get("allowed")
            pattern = fd.get("pattern")

            value = self._generate_field(
                fname, ftype, width, allowed, extra, context, spec_name
            )
            value = value[:width].ljust(width)[:width]  # enforce exact width
            line[s:e] = list(value)

        line_str = "".join(line)
        return (line_str, context) if return_context else (line_str, None)

    def generate_batch(
        self,
        spec_name   : str,
        record_type : str,
        count       : int,
        seed        : Optional[int] = None,
    ) -> List[str]:
        """Generate `count` records of the given type."""
        if seed is not None:
            self.rng = random.Random(seed)
        return [self.generate_line(spec_name, record_type)[0] for _ in range(count)]

    # ── Field Value Generation ─────────────────────────────────────────────

    def _generate_field(
        self,
        name    : str,
        ftype   : str,
        width   : int,
        allowed : Optional[List[str]],
        extra   : Dict,
        context : Dict,
        spec_name: str,
    ) -> str:
        # 1. Build multiple candidate lookup keys for the extra dict
        # Primary: name.lower().replace(" ", "_")
        # Secondary: strip punctuation (handles "Entry/Addenda Count" → "entry_addenda_count")
        import re as _re
        key_primary   = name.lower().replace(" ", "_")
        key_secondary = _re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")

        for key in (key_primary, key_secondary):
            if key in extra:
                v = str(extra[key])
                if ftype in ("NUMERIC", "AMOUNT"):
                    v = v.zfill(width)
                return v.ljust(width)[:width]

        # 2. If the field has a fixed allowed list, pick one
        if allowed and len(allowed) > 0:
            chosen = self.rng.choice(allowed)
            return chosen.ljust(width)[:width]

        # 3. Type-based generation with name hints
        return self._by_type(name, ftype, width, context, spec_name)

    def _generate_routing_rng(self) -> str:
        """Valid 9-digit ABA routing number using self.rng (deterministic)."""
        weights = [3, 7, 1, 3, 7, 1, 3, 7]
        for _ in range(100):
            digits = [self.rng.randint(0, 9) for _ in range(8)]
            total  = sum(d * w for d, w in zip(digits, weights))
            check  = (10 - (total % 10)) % 10
            routing = "".join(str(d) for d in digits) + str(check)
            if 1 <= int(routing[:2]) <= 32:
                return routing
        return "021000021"  # fallback: Chase ABA (always valid)

    def _by_type(
        self, name: str, ftype: str, width: int, context: Dict, spec_name: str
    ) -> str:
        name_lower = name.lower()

        # ── Routing Numbers ────────────────────────────────────────────
        if ftype == "ROUTING" or "routing" in name_lower or "transit" in name_lower:
            r = self._generate_routing_rng()
            context["routing"] = r
            return r[:width].ljust(width)[:width]

        # ── Account Numbers ────────────────────────────────────────────
        if ftype == "ACCOUNT" or "account" in name_lower:
            acc = "".join(str(self.rng.randint(0, 9)) for _ in range(self.rng.randint(10, 17)))
            context["account"] = acc
            return acc.ljust(width)[:width]

        # ── Monetary Amounts ───────────────────────────────────────────
        if ftype == "AMOUNT" or "amount" in name_lower or "dollar" in name_lower:
            max_d  = 99_999 if width >= 10 else 9_999
            amount = _generate_amount(width, max_d)
            cents  = int(amount)
            if "debit" in name_lower:
                context["debit"] = cents
            elif "credit" in name_lower:
                context["credit"] = cents
            else:
                context["amount"] = cents
            return amount[:width].zfill(width)

        # ── Dates ──────────────────────────────────────────────────────
        if ftype == "DATE" or "date" in name_lower:
            today   = datetime.date.today()
            delta   = self.rng.randint(-540, 540)
            target  = today + datetime.timedelta(days=delta)
            if width == 6:
                d = target.strftime("%y%m%d")
            elif width == 4:
                d = target.strftime("%m%d")
            else:
                d = target.strftime("%y%m%d")
            context["date"] = d
            return d[:width]

        # ── Check Digit (single char, derived from preceding routing) ──
        if "check digit" in name_lower:
            routing = context.get("routing", _generate_routing())
            return routing[-1]

        # ── Trace Number ───────────────────────────────────────────────
        if "trace" in name_lower or "sequence" in name_lower:
            r = context.get("routing", "021000021")
            seq = self.rng.randint(1, 999_999)
            trace = r[:8] + str(seq).zfill(7)
            return trace[:width].zfill(width)

        # ── Individual / Company Names ─────────────────────────────────
        if "name" in name_lower:
            if "company" in name_lower or "origin" in name_lower or "dest" in name_lower:
                v = self.rng.choice(_COMPANY_NAMES)
            else:
                fn = self.rng.choice(_FIRST_NAMES)
                ln = self.rng.choice(_LAST_NAMES)
                v  = f"{ln} {fn}"
            context["name"] = v
            return v[:width].ljust(width)

        # ── Entry / Batch Descriptions ─────────────────────────────────
        if "description" in name_lower or "entry" in name_lower:
            v = self.rng.choice(_ENTRY_DESCRIPTIONS)
            return v[:width].ljust(width)

        # ── Company ID / Identification ────────────────────────────────
        if "identification" in name_lower or "company id" in name_lower:
            cid = "".join(str(self.rng.randint(0, 9)) for _ in range(10))
            context["company_id"] = cid
            return cid[:width]

        # ── Batch / Sequence Numbers ───────────────────────────────────
        if "batch number" in name_lower or "batch count" in name_lower:
            return str(self.rng.randint(1, 9999)).zfill(width)

        # ── Block Count ────────────────────────────────────────────────
        if "block count" in name_lower:
            return str(self.rng.randint(1, 99)).zfill(width)

        # ── Priority / Format / Record Size (fixed) ───────────────────
        if "priority" in name_lower:
            return "01"[:width].ljust(width)

        # ── GL-Specific: Account Code ──────────────────────────────────
        if "account code" in name_lower:
            v = self.rng.choice(_GL_ACCOUNT_CODES)
            return v.ljust(width)[:width]

        if "ledger code" in name_lower:
            v = self.rng.choice(_LEDGER_CODES)
            return v.ljust(width)[:width]

        if "currency" in name_lower:
            v = "USD" if spec_name != "GENERAL_LEDGER" else self.rng.choice(_CURRENCY_CODES)
            return v[:width].ljust(width)

        if "merchant category" in name_lower or "mcc" in name_lower:
            return self.rng.choice(_MCC_CODES)[:width]

        if "merchant" in name_lower:
            return self.rng.choice(_COMPANY_NAMES)[:width].ljust(width)

        if "period" in name_lower:
            today = datetime.date.today()
            return today.strftime("%Y%m")[:width]

        if "journal id" in name_lower:
            jid = "JNL" + "".join(str(self.rng.randint(0, 9)) for _ in range(7))
            context["journal_id"] = jid
            return jid[:width].ljust(width)

        # ── Hash / Control Totals ──────────────────────────────────────
        if "hash" in name_lower or "control hash" in name_lower:
            h = str(self.rng.randint(0, 9_999_999_999)).zfill(width)
            return h[:width]

        # ── Pure Numeric fallback ──────────────────────────────────────
        if ftype in ("NUMERIC", "AMOUNT"):
            return "".join(str(self.rng.randint(0, 9)) for _ in range(width))

        # ── Blank Pad ──────────────────────────────────────────────────
        if ftype == "BLANK_PAD" or "reserved" in name_lower:
            return " " * width

        # ── Alphanumeric fallback ──────────────────────────────────────
        chars = string.ascii_uppercase + string.digits + " "
        return "".join(self.rng.choice(chars) for _ in range(width))
