"""
Oracle ACH Data Repository
All SQL queries and data-access logic for ACH file generation.
Provides typed, validated objects that the generator can consume directly.

Query strategy:
  • All queries are parameterised (no f-string SQL — prevents injection)
  • DATE types are returned as Python datetime objects by oracledb
  • CLOB columns are read() before use
  • Falls back to synthetic data if in MOCK mode
"""

import os
import sys
import logging
from datetime import datetime, date, timedelta
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field

BASE = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, BASE)

from db.oracle_connector import get_pool, execute_query, OracleConfig

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Domain objects returned by the repository
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ODFIConfig:
    routing_number:    str
    bank_name:         str
    immediate_dest:    str
    immediate_origin:  str
    dest_short_name:   str
    origin_short_name: str


@dataclass
class CompanyRecord:
    company_id:        int
    company_name:      str          # max 16 chars
    company_id_number: str          # max 10 chars
    company_entry_desc: str         # max 10 chars
    sec_code:          str          # 3 chars
    service_class_code: str         # 200/220/225
    discretionary_data: str = "  "


@dataclass
class AccountRecord:
    account_id:        int
    individual_name:   str          # max 22 chars
    individual_id:     str          # max 15 chars
    rdfi_routing:      str          # 8 digits
    rdfi_check_digit:  str          # 1 digit
    account_number:    str          # max 17 chars
    account_type:      str          # C/S/G/L


@dataclass
class TransactionRecord:
    transaction_id:    int
    account_id:        int
    company_id:        int
    transaction_code:  str          # 2 digits
    amount_cents:      int
    effective_date_str: str         # YYMMDD
    individual_id:     str
    individual_name:   str
    rdfi_routing:      str          # 8 digits
    rdfi_check_digit:  str
    account_number:    str
    account_type:      str
    discretionary_data: str
    addenda_info:      Optional[str]
    company_name:      str
    company_id_number: str
    company_entry_desc: str
    sec_code:          str
    service_class_code: str
    company_disc_data: str
    odfi_routing:      str          # 9 digits
    immediate_dest:    str
    immediate_origin:  str
    dest_short_name:   str
    origin_short_name: str

    @property
    def amount_str(self) -> str:
        """Format cents as 10-digit NACHA amount field."""
        return str(self.amount_cents).zfill(10)

    @property
    def full_rdfi_routing(self) -> str:
        return self.rdfi_routing + self.rdfi_check_digit


# ─────────────────────────────────────────────────────────────────────────────
# SQL Queries
# ─────────────────────────────────────────────────────────────────────────────

_SQL_PENDING_TRANSACTIONS = """
SELECT
    TRANSACTION_ID,
    COMPANY_ID,
    ACCOUNT_ID,
    TRANSACTION_CODE,
    AMOUNT_CENTS,
    EFFECTIVE_DATE_STR,
    INDIVIDUAL_ID,
    INDIVIDUAL_NAME,
    RDFI_ROUTING,
    RDFI_CHECK_DIGIT,
    ACCOUNT_NUMBER,
    ACCOUNT_TYPE,
    DISCRETIONARY_DATA,
    ADDENDA_INFO,
    COMPANY_NAME,
    COMPANY_ID_NUMBER,
    COMPANY_ENTRY_DESC,
    SEC_CODE,
    SERVICE_CLASS_CODE,
    COMPANY_DISC_DATA,
    NVL(ODFI_ROUTING, '021000021')    AS ODFI_ROUTING,
    NVL(IMMEDIATE_DEST, ' 021000021') AS IMMEDIATE_DEST,
    NVL(IMMEDIATE_ORIGIN,'021000021 ') AS IMMEDIATE_ORIGIN,
    NVL(DEST_SHORT_NAME, 'JPMORGAN CHASE NA  ') AS DEST_SHORT_NAME,
    NVL(ORIGIN_SHORT_NAME,'FINSLM PAYMENTS    ') AS ORIGIN_SHORT_NAME,
    STATUS
FROM {schema}.V_ACH_PENDING_TRANSACTIONS
WHERE 1=1
  {company_filter}
  {sec_filter}
  {date_filter}
ORDER BY COMPANY_ID, TRANSACTION_ID
FETCH FIRST :max_rows ROWS ONLY
"""

_SQL_COMPANIES = """
SELECT
    c.COMPANY_ID,
    c.COMPANY_NAME,
    c.COMPANY_ID_NUMBER,
    c.COMPANY_ENTRY_DESC,
    c.SEC_CODE,
    c.SERVICE_CLASS_CODE,
    NVL(c.DISCRETIONARY_DATA,'  ') AS DISCRETIONARY_DATA
FROM {schema}.ACH_COMPANIES c
WHERE c.IS_ACTIVE = 'Y'
  {sec_filter}
ORDER BY c.COMPANY_ID
"""

_SQL_ODFI = """
SELECT
    ROUTING_NUMBER,
    BANK_NAME,
    IMMEDIATE_DEST,
    IMMEDIATE_ORIGIN,
    DEST_SHORT_NAME,
    ORIGIN_SHORT_NAME
FROM {schema}.ACH_ODFI_CONFIG
WHERE IS_ACTIVE = 'Y'
  AND ODFI_ID = :odfi_id
"""

_SQL_LOG_FILE = """
INSERT INTO {schema}.ACH_FILE_LOG (
    FILE_NAME, FILE_ID_MODIFIER, ODFI_ID,
    BATCH_COUNT, ENTRY_COUNT, BLOCK_COUNT,
    TOTAL_DEBIT_CENTS, TOTAL_CREDIT_CENTS,
    GENERATION_METHOD, IS_VALID, FILE_CONTENT
) VALUES (
    :file_name, :modifier, :odfi_id,
    :batch_count, :entry_count, :block_count,
    :debit, :credit,
    'ORACLE', 'Y', :content
) RETURNING FILE_ID INTO :file_id
"""

_SQL_SAVE_CORPUS = """
INSERT INTO {schema}.ACH_TRAINING_CORPUS (
    FILE_LOG_ID, FILE_CONTENT, SEC_CODE,
    SERVICE_CLASS_CODE, BATCH_COUNT, ENTRY_COUNT, SPLIT_TYPE
) VALUES (
    :file_log_id, :content, :sec_code,
    :scc, :batches, :entries, :split
)
"""

_SQL_FETCH_CORPUS = """
SELECT FILE_CONTENT
FROM {schema}.ACH_TRAINING_CORPUS
WHERE SPLIT_TYPE = :split
  AND IS_USED_FOR_TRAINING = 'N'
  {sec_filter}
ORDER BY DBMS_RANDOM.VALUE
FETCH FIRST :limit ROWS ONLY
"""


# ─────────────────────────────────────────────────────────────────────────────
# Repository class
# ─────────────────────────────────────────────────────────────────────────────

class ACHRepository:
    """
    All database interactions for ACH generation.
    Automatically uses MOCK data when the DB is unavailable.
    """

    def __init__(self, config: Optional[OracleConfig] = None):
        self.pool = get_pool(config)
        self.schema = (config or OracleConfig()).schema
        self._mock = self.pool.is_mock

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _sql(self, template: str, **filters) -> str:
        """Inject schema and optional WHERE fragments."""
        return template.format(schema=self.schema, **filters)

    # ── Transaction queries ───────────────────────────────────────────────────

    def get_pending_transactions(
        self,
        company_id: Optional[int] = None,
        sec_code:   Optional[str] = None,
        effective_date: Optional[str] = None,   # YYMMDD string
        max_rows:   int = 5000,
    ) -> List[TransactionRecord]:

        if self._mock:
            return self._mock_transactions(max_rows, sec_code)

        company_filter = "AND COMPANY_ID = :company_id" if company_id else ""
        sec_filter     = "AND SEC_CODE = :sec_code"     if sec_code    else ""
        date_filter    = "AND EFFECTIVE_DATE_STR = :eff_date" if effective_date else ""

        sql = self._sql(
            _SQL_PENDING_TRANSACTIONS,
            company_filter=company_filter,
            sec_filter=sec_filter,
            date_filter=date_filter,
        )

        params: Dict[str, Any] = {"max_rows": max_rows}
        if company_id:     params["company_id"] = company_id
        if sec_code:       params["sec_code"]   = sec_code
        if effective_date: params["eff_date"]   = effective_date

        try:
            rows = execute_query(sql, params)
            return [self._row_to_txn(r) for r in rows]
        except Exception as e:
            log.error("get_pending_transactions failed: %s — using mock data", e)
            return self._mock_transactions(max_rows, sec_code)

    def get_companies(self, sec_code: Optional[str] = None) -> List[CompanyRecord]:
        if self._mock:
            return self._mock_companies(sec_code)

        sec_filter = "AND c.SEC_CODE = :sec_code" if sec_code else ""
        sql = self._sql(_SQL_COMPANIES, sec_filter=sec_filter)
        params = {"sec_code": sec_code} if sec_code else {}

        try:
            rows = execute_query(sql, params)
            return [CompanyRecord(**{k: r[k] for k in CompanyRecord.__dataclass_fields__}) for r in rows]
        except Exception as e:
            log.error("get_companies failed: %s", e)
            return self._mock_companies(sec_code)

    def log_file(self, file_name: str, modifier: str, odfi_id: int,
                 batch_count: int, entry_count: int, block_count: int,
                 total_debit: int, total_credit: int,
                 content: str) -> Optional[int]:
        """Write generated file to audit log, return FILE_ID."""
        if self._mock:
            log.info("MOCK log_file: %s (%d entries)", file_name, entry_count)
            return None

        sql = self._sql(_SQL_LOG_FILE)
        params = dict(
            file_name=file_name, modifier=modifier, odfi_id=odfi_id,
            batch_count=batch_count, entry_count=entry_count, block_count=block_count,
            debit=total_debit, credit=total_credit, content=content,
            file_id=None,
        )
        try:
            with self.pool.connection() as conn:
                cur = conn.cursor()
                out = cur.var(int)
                params["file_id"] = out
                cur.execute(sql, params)
                conn.commit()
                return out.getvalue()
        except Exception as e:
            log.error("log_file failed: %s", e)
            return None

    def save_corpus_entry(self, content: str, sec_code: str, scc: str,
                          batches: int, entries: int,
                          split: str = "TRAIN",
                          file_log_id: Optional[int] = None):
        """Persist a generated ACH file to the training corpus table."""
        if self._mock:
            return

        sql = self._sql(_SQL_SAVE_CORPUS)
        params = dict(
            file_log_id=file_log_id, content=content, sec_code=sec_code,
            scc=scc, batches=batches, entries=entries, split=split,
        )
        try:
            with self.pool.connection() as conn:
                cur = conn.cursor()
                cur.execute(sql, params)
                conn.commit()
        except Exception as e:
            log.warning("save_corpus_entry failed: %s", e)

    def fetch_corpus(
        self,
        split: str = "TRAIN",
        sec_code: Optional[str] = None,
        limit: int = 500,
    ) -> List[str]:
        """Retrieve stored training files from the corpus table."""
        if self._mock:
            return []

        sec_filter = "AND SEC_CODE = :sec_code" if sec_code else ""
        sql = self._sql(_SQL_FETCH_CORPUS, sec_filter=sec_filter)
        params: Dict[str, Any] = {"split": split, "limit": limit}
        if sec_code:
            params["sec_code"] = sec_code

        try:
            rows = execute_query(sql, params)
            results = []
            for r in rows:
                content = r.get("file_content")
                if hasattr(content, "read"):
                    content = content.read()
                if content:
                    results.append(content)
            return results
        except Exception as e:
            log.error("fetch_corpus failed: %s", e)
            return []

    # ── Row → dataclass mapping ───────────────────────────────────────────────

    @staticmethod
    def _row_to_txn(r: dict) -> TransactionRecord:
        def s(key, default="", maxlen=None):
            v = str(r.get(key) or default).strip()
            return v[:maxlen] if maxlen else v

        return TransactionRecord(
            transaction_id    = int(r.get("transaction_id", 0)),
            account_id        = int(r.get("account_id", 0)),
            company_id        = int(r.get("company_id", 0)),
            transaction_code  = s("transaction_code", "22", 2),
            amount_cents      = int(r.get("amount_cents", 0)),
            effective_date_str= s("effective_date_str", datetime.now().strftime("%y%m%d"), 6),
            individual_id     = s("individual_id", "          ", 15).ljust(15)[:15],
            individual_name   = s("individual_name", "UNKNOWN", 22).ljust(22)[:22],
            rdfi_routing      = s("rdfi_routing", "02100002", 8).ljust(8)[:8],
            rdfi_check_digit  = s("rdfi_check_digit", "1", 1),
            account_number    = s("account_number", "000000000", 17).ljust(17)[:17],
            account_type      = s("account_type", "C", 1),
            discretionary_data= s("discretionary_data", "  ", 2).ljust(2)[:2],
            addenda_info      = s("addenda_info") or None,
            company_name      = s("company_name", "COMPANY", 16).ljust(16)[:16],
            company_id_number = s("company_id_number", "1000000000", 10).ljust(10)[:10],
            company_entry_desc= s("company_entry_desc", "PAYMENT", 10).ljust(10)[:10],
            sec_code          = s("sec_code", "PPD", 3),
            service_class_code= s("service_class_code", "200", 3),
            company_disc_data = s("company_disc_data", "                    ", 20).ljust(20)[:20],
            odfi_routing      = s("odfi_routing", "021000021", 9),
            immediate_dest    = s("immediate_dest", " 021000021", 10).ljust(10)[:10],
            immediate_origin  = s("immediate_origin", "021000021 ", 10).ljust(10)[:10],
            dest_short_name   = s("dest_short_name", "JPMORGAN CHASE NA  ", 23).ljust(23)[:23],
            origin_short_name = s("origin_short_name", "FINSLM PAYMENTS    ", 23).ljust(23)[:23],
        )

    # ── Mock data generators (used when DB unavailable) ───────────────────────

    def _mock_transactions(self, n: int, sec_code: Optional[str]) -> List[TransactionRecord]:
        import random
        from data.generator import ACHGenerator, VALID_ROUTING_NUMBERS, INDIVIDUAL_NAMES, COMPANY_NAMES
        rng = VALID_ROUTING_NUMBERS
        sec = sec_code or random.choice(["PPD", "CCD", "WEB", "TEL"])
        scc = "200"
        co_name = random.choice(COMPANY_NAMES)
        co_id   = f"1{random.randint(10**8, 10**9-1)}"
        eff     = (datetime.now() + timedelta(days=2)).strftime("%y%m%d")

        txns = []
        for i in range(min(n, 20)):
            routing = random.choice(rng)
            tc = random.choice(["22", "27", "32", "37"])
            amt = random.randint(100, 500000)
            name = random.choice(INDIVIDUAL_NAMES)
            txns.append(TransactionRecord(
                transaction_id   = i + 1,
                account_id       = i + 1,
                company_id       = 1,
                transaction_code = tc,
                amount_cents     = amt,
                effective_date_str = eff,
                individual_id    = f"ID{i:013d}",
                individual_name  = name[:22].ljust(22),
                rdfi_routing     = routing[:8],
                rdfi_check_digit = routing[8],
                account_number   = f"{random.randint(10**6,10**9-1)}".ljust(17)[:17],
                account_type     = "C",
                discretionary_data = "  ",
                addenda_info     = None,
                company_name     = co_name[:16].ljust(16),
                company_id_number= co_id[:10].ljust(10),
                company_entry_desc = "PAYROLL   ",
                sec_code         = sec,
                service_class_code = scc,
                company_disc_data= "                    ",
                odfi_routing     = "021000021",
                immediate_dest   = " 021000021",
                immediate_origin = "021000021 ",
                dest_short_name  = "JPMORGAN CHASE NA  ",
                origin_short_name= "FINSLM PAYMENTS    ",
            ))
        return txns

    def _mock_companies(self, sec_code: Optional[str]) -> List[CompanyRecord]:
        from data.generator import COMPANY_NAMES
        import random
        names = COMPANY_NAMES[:5]
        secs = [sec_code or s for s in ["PPD", "CCD", "WEB", "TEL", "PPD"]]
        return [
            CompanyRecord(
                company_id        = i + 1,
                company_name      = n[:16],
                company_id_number = f"1{i:09d}",
                company_entry_desc= "PAYROLL   ",
                sec_code          = secs[i % len(secs)],
                service_class_code= "200",
                discretionary_data= "  ",
            )
            for i, n in enumerate(names)
        ]
