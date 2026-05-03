"""
ChromaDB ACH Repository
All data-access operations for ACH generation using ChromaDB.

Key design decisions vs Oracle:
  - No SQL JOINs: transactions are stored denormalised (company + ODFI fields
    copied in at insert time) so the generator reads one collection only.
  - ChromaDB `where` filters replace SQL WHERE clauses.
  - `get()` replaces SELECT by PK; `query()` replaces full-table scans.
  - Amounts stay as integer cents in metadata; ChromaDB metadata supports int.
  - Document text is the primary searchable body; metadata drives filtering.
"""

import os
import sys
import json
import uuid
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import random

BASE = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, BASE)

from db.chroma_client import get_client, get_collection, Collections

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Domain objects (identical interface to the old Oracle repository)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ODFIConfig:
    id: str
    routing_number: str
    bank_name: str
    immediate_dest: str
    immediate_origin: str
    dest_short_name: str
    origin_short_name: str


@dataclass
class CompanyRecord:
    company_id: str
    company_name: str
    company_id_number: str
    company_entry_desc: str
    sec_code: str
    service_class_code: str
    odfi_id: str = ""
    discretionary_data: str = "  "


@dataclass
class TransactionRecord:
    transaction_id: str
    account_id: str
    company_id: str
    transaction_code: str
    amount_cents: int
    effective_date_str: str
    individual_id: str
    individual_name: str
    rdfi_routing: str
    rdfi_check_digit: str
    account_number: str
    account_type: str
    discretionary_data: str
    addenda_info: Optional[str]
    company_name: str
    company_id_number: str
    company_entry_desc: str
    sec_code: str
    service_class_code: str
    company_disc_data: str
    odfi_routing: str
    immediate_dest: str
    immediate_origin: str
    dest_short_name: str
    origin_short_name: str
    status: str = "PENDING"

    @property
    def amount_str(self) -> str:
        return str(self.amount_cents).zfill(10)

    @property
    def full_rdfi_routing(self) -> str:
        return self.rdfi_routing + self.rdfi_check_digit


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now().isoformat()

def _uid(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:12]}"

def _pad(s: str, n: int, char: str = " ") -> str:
    return str(s or "").ljust(n)[:n]

def _safe_int(v, default: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


# ─────────────────────────────────────────────────────────────────────────────
# Repository
# ─────────────────────────────────────────────────────────────────────────────

class ACHRepository:
    """
    ChromaDB-backed data repository for ACH generation.
    Provides the same public interface as the former Oracle repository
    so the generator and trainer require minimal changes.
    """

    def __init__(self, store_path: Optional[str] = None):
        self._db = get_client(store_path)
        self._db.seed_defaults()

    # ── ODFI ─────────────────────────────────────────────────────────────────

    def get_odfi(self, odfi_id: Optional[str] = None) -> ODFIConfig:
        """Return one ODFI config (default: first active record)."""
        col = self._db.collection(Collections.ODFI_CONFIG)
        if odfi_id:
            res = col.get(ids=[odfi_id])
        else:
            res = col.get(where={"is_active": {"$eq": "Y"}}, limit=1)

        if not res or not res.get("ids"):
            return self._default_odfi()

        meta = res["metadatas"][0]
        return ODFIConfig(
            id=res["ids"][0],
            routing_number=meta.get("routing_number", "021000021"),
            bank_name=meta.get("bank_name", "DEFAULT BANK"),
            immediate_dest=meta.get("immediate_dest", " 021000021"),
            immediate_origin=meta.get("immediate_origin", "021000021 "),
            dest_short_name=meta.get("dest_short_name", "DEFAULT BANK       "),
            origin_short_name=meta.get("origin_short_name", "FINSLM PAYMENTS    "),
        )

    def list_odfi(self) -> List[ODFIConfig]:
        col = self._db.collection(Collections.ODFI_CONFIG)
        res = col.get(where={"is_active": {"$eq": "Y"}})
        if not res or not res.get("ids"):
            return [self._default_odfi()]
        return [
            ODFIConfig(
                id=rid,
                routing_number=m.get("routing_number", ""),
                bank_name=m.get("bank_name", ""),
                immediate_dest=m.get("immediate_dest", ""),
                immediate_origin=m.get("immediate_origin", ""),
                dest_short_name=m.get("dest_short_name", ""),
                origin_short_name=m.get("origin_short_name", ""),
            )
            for rid, m in zip(res["ids"], res["metadatas"])
        ]

    @staticmethod
    def _default_odfi() -> ODFIConfig:
        return ODFIConfig(
            id="odfi_default",
            routing_number="021000021",
            bank_name="JPMORGAN CHASE BANK NA",
            immediate_dest=" 021000021",
            immediate_origin="021000021 ",
            dest_short_name="JPMORGAN CHASE NA  ",
            origin_short_name="FINSLM PAYMENTS    ",
        )

    # ── Companies ─────────────────────────────────────────────────────────────

    def get_companies(self, sec_code: Optional[str] = None) -> List[CompanyRecord]:
        col = self._db.collection(Collections.COMPANIES)
        where = {"is_active": {"$eq": "Y"}}
        if sec_code:
            where = {"$and": [{"is_active": {"$eq": "Y"}}, {"sec_code": {"$eq": sec_code}}]}

        res = col.get(where=where, limit=200)
        if not res or not res.get("ids"):
            return self._mock_companies(sec_code)

        return [
            CompanyRecord(
                company_id=rid,
                company_name=_pad(m.get("company_name", "COMPANY"), 16),
                company_id_number=_pad(m.get("company_id_number", "1000000000"), 10),
                company_entry_desc=_pad(m.get("company_entry_desc", "PAYMENT"), 10),
                sec_code=m.get("sec_code", "PPD"),
                service_class_code=m.get("service_class_code", "200"),
                odfi_id=m.get("odfi_id", "odfi_001"),
                discretionary_data=_pad(m.get("discretionary_data", "  "), 20),
            )
            for rid, m in zip(res["ids"], res["metadatas"])
        ]

    def add_company(self, company: CompanyRecord) -> str:
        col = self._db.collection(Collections.COMPANIES)
        cid = company.company_id or _uid("co_")
        col.add(
            ids=[cid],
            documents=[company.company_name.strip()],
            metadatas=[{
                "company_name":        company.company_name,
                "company_id_number":   company.company_id_number,
                "company_entry_desc":  company.company_entry_desc,
                "sec_code":            company.sec_code,
                "service_class_code":  company.service_class_code,
                "odfi_id":             company.odfi_id,
                "discretionary_data":  company.discretionary_data,
                "is_active":           "Y",
                "created_at":          _now(),
            }],
        )
        return cid

    # ── Accounts ──────────────────────────────────────────────────────────────

    def add_account(self, meta: dict) -> str:
        col = self._db.collection(Collections.ACCOUNTS)
        aid = _uid("acc_")
        col.add(
            ids=[aid],
            documents=[meta.get("individual_name", "UNKNOWN")],
            metadatas=[{**meta, "is_active": "Y", "created_at": _now()}],
        )
        return aid

    # ── Transactions ──────────────────────────────────────────────────────────

    def add_transaction(self, txn: TransactionRecord) -> str:
        col = self._db.collection(Collections.TRANSACTIONS)
        tid = txn.transaction_id or _uid("txn_")
        doc = f"{txn.individual_name.strip()} {txn.transaction_code} {txn.amount_cents}"
        col.add(
            ids=[tid],
            documents=[doc],
            metadatas=[{
                "account_id":          txn.account_id,
                "company_id":          txn.company_id,
                "transaction_code":    txn.transaction_code,
                "amount_cents":        txn.amount_cents,
                "effective_date":      txn.effective_date_str,
                "individual_id":       txn.individual_id,
                "individual_name":     txn.individual_name,
                "rdfi_routing":        txn.rdfi_routing,
                "rdfi_check_digit":    txn.rdfi_check_digit,
                "account_number":      txn.account_number,
                "account_type":        txn.account_type,
                "discretionary_data":  txn.discretionary_data,
                "addenda_info":        txn.addenda_info or "",
                "company_name":        txn.company_name,
                "company_id_number":   txn.company_id_number,
                "company_entry_desc":  txn.company_entry_desc,
                "sec_code":            txn.sec_code,
                "service_class_code":  txn.service_class_code,
                "company_disc_data":   txn.company_disc_data,
                "odfi_routing":        txn.odfi_routing,
                "immediate_dest":      txn.immediate_dest,
                "immediate_origin":    txn.immediate_origin,
                "dest_short_name":     txn.dest_short_name,
                "origin_short_name":   txn.origin_short_name,
                "status":              "PENDING",
                "created_at":          _now(),
                "processed_at":        "",
                "file_id":             "",
                "batch_number":        0,
                "return_code":         "",
            }],
        )
        return tid

    def get_pending_transactions(
        self,
        company_id: Optional[str] = None,
        sec_code:   Optional[str] = None,
        effective_date: Optional[str] = None,
        max_rows:   int = 5000,
    ) -> List[TransactionRecord]:
        col = self._db.collection(Collections.TRANSACTIONS)

        # Build ChromaDB `where` filter
        filters = [{"status": {"$eq": "PENDING"}}]
        if company_id:
            filters.append({"company_id": {"$eq": str(company_id)}})
        if sec_code:
            filters.append({"sec_code": {"$eq": sec_code}})
        if effective_date:
            filters.append({"effective_date": {"$eq": effective_date}})

        where = {"$and": filters} if len(filters) > 1 else filters[0]

        try:
            res = col.get(where=where, limit=max_rows)
        except Exception as e:
            log.warning("get_pending_transactions fallback: %s", e)
            res = col.get(limit=max_rows)

        if not res or not res.get("ids"):
            log.info("No ChromaDB transactions found — using synthetic fallback")
            return self._mock_transactions(min(max_rows, 20), sec_code)

        return [self._meta_to_txn(rid, m)
                for rid, m in zip(res["ids"], res["metadatas"])
                if m.get("status") == "PENDING"]

    def mark_batched(self, transaction_ids: List[str], file_id: str, batch_number: int):
        col = self._db.collection(Collections.TRANSACTIONS)
        for tid in transaction_ids:
            try:
                existing = col.get(ids=[tid])
                if existing and existing.get("metadatas"):
                    meta = existing["metadatas"][0]
                    meta.update(status="BATCHED", file_id=file_id,
                                batch_number=batch_number, processed_at=_now())
                    col.update(ids=[tid], metadatas=[meta],
                               documents=existing.get("documents", [tid]))
            except Exception as e:
                log.warning("mark_batched %s: %s", tid, e)

    # ── File log ──────────────────────────────────────────────────────────────

    def log_file(
        self, file_name: str, modifier: str, odfi_id: str,
        batch_count: int, entry_count: int, block_count: int,
        total_debit: int, total_credit: int,
        content: str, sec_codes: str = "",
    ) -> str:
        col = self._db.collection(Collections.FILE_LOG)
        fid = f"file_{datetime.now().strftime('%Y%m%d%H%M%S')}_{modifier}"
        col.add(
            ids=[fid],
            documents=[content],          # full NACHA text — queryable by similarity
            metadatas=[{
                "file_name":           file_name,
                "file_id_modifier":    modifier,
                "odfi_id":             odfi_id,
                "batch_count":         batch_count,
                "entry_count":         entry_count,
                "block_count":         block_count,
                "total_debit_cents":   total_debit,
                "total_credit_cents":  total_credit,
                "generation_method":   "CHROMA",
                "is_valid":            "Y",
                "validation_errors":   0,
                "sec_codes":           sec_codes,
                "created_at":          _now(),
                "sent_at":             "",
            }],
        )
        log.info("Logged file %s (%d entries)", fid, entry_count)
        return fid

    def get_file_log(self, limit: int = 50) -> List[Dict]:
        col = self._db.collection(Collections.FILE_LOG)
        res = col.get(limit=limit)
        if not res or not res.get("ids"):
            return []
        return [
            {"file_id": rid, **m}
            for rid, m in zip(res["ids"], res["metadatas"])
        ]

    # ── Validation log ────────────────────────────────────────────────────────

    def log_validation(
        self, file_name: str, file_type: str, is_valid: bool,
        error_count: int, warning_count: int,
        report: dict, file_id: str = "",
    ) -> str:
        col = self._db.collection(Collections.VALIDATION_LOG)
        vid = _uid("val_")
        summary = (
            f"{'VALID' if is_valid else 'INVALID'} {file_type} file: "
            f"{file_name} — {error_count} errors, {warning_count} warnings"
        )
        col.add(
            ids=[vid],
            documents=[summary],
            metadatas=[{
                "file_id":       file_id,
                "file_name":     file_name,
                "file_type":     file_type,
                "is_valid":      "Y" if is_valid else "N",
                "error_count":   error_count,
                "warning_count": warning_count,
                "report_json":   json.dumps(report)[:2000],  # truncate for metadata limit
                "created_at":    _now(),
            }],
        )
        return vid

    # ── Training corpus ───────────────────────────────────────────────────────

    def save_corpus_entry(
        self, content: str, sec_code: str, scc: str,
        batches: int, entries: int, split: str = "TRAIN",
        file_log_id: str = "", source: str = "CHROMA",
    ) -> str:
        col = self._db.collection(Collections.TRAINING_CORPUS)
        cid = _uid("corpus_")
        col.add(
            ids=[cid],
            documents=[content],           # full ACH text — embedded for similarity search
            metadatas=[{
                "file_log_id":          file_log_id,
                "sec_code":             sec_code,
                "service_class_code":   scc,
                "batch_count":          batches,
                "entry_count":          entries,
                "split_type":           split,
                "is_used_for_training": "N",
                "source":               source,
                "created_at":           _now(),
            }],
        )
        return cid

    def fetch_corpus(
        self,
        split: str = "TRAIN",
        sec_code: Optional[str] = None,
        limit: int = 500,
        similar_to: Optional[str] = None,
    ) -> List[str]:
        """
        Retrieve training files from the corpus collection.
        If `similar_to` is provided, uses ChromaDB vector similarity search
        to find structurally similar ACH files — powerful for curriculum learning.
        """
        col = self._db.collection(Collections.TRAINING_CORPUS)
        where = {"split_type": {"$eq": split}}
        if sec_code:
            where = {"$and": [{"split_type": {"$eq": split}},
                               {"sec_code":   {"$eq": sec_code}}]}

        try:
            if similar_to:
                # Semantic similarity search — find ACH files similar to a seed
                res = col.query(
                    query_texts=[similar_to],
                    n_results=min(limit, col.count() or 1),
                    where=where,
                )
                docs = res.get("documents", [[]])[0]
            else:
                res = col.get(where=where, limit=limit)
                docs = res.get("documents", [])

            return [d for d in docs if d]

        except Exception as e:
            log.warning("fetch_corpus failed: %s", e)
            return []

    def corpus_stats(self) -> Dict[str, Any]:
        col = self._db.collection(Collections.TRAINING_CORPUS)
        total = col.count()
        if total == 0:
            return {"total": 0}

        stats: Dict[str, int] = {"total": total}
        for split in ("TRAIN", "VAL", "TEST"):
            try:
                res = col.get(where={"split_type": {"$eq": split}}, limit=10000)
                stats[split.lower()] = len(res.get("ids", []))
            except Exception:
                stats[split.lower()] = 0
        return stats

    # ── Meta → domain object ──────────────────────────────────────────────────

    @staticmethod
    def _meta_to_txn(rid: str, m: dict) -> TransactionRecord:
        def s(k, default="", n=None):
            v = str(m.get(k) or default).strip()
            return v[:n] if n else v

        return TransactionRecord(
            transaction_id     = rid,
            account_id         = s("account_id"),
            company_id         = s("company_id"),
            transaction_code   = s("transaction_code", "22", 2),
            amount_cents       = _safe_int(m.get("amount_cents")),
            effective_date_str = s("effective_date", datetime.now().strftime("%y%m%d"), 6),
            individual_id      = _pad(s("individual_id"), 15),
            individual_name    = _pad(s("individual_name", "UNKNOWN"), 22),
            rdfi_routing       = _pad(s("rdfi_routing", "02100002"), 8),
            rdfi_check_digit   = s("rdfi_check_digit", "1", 1),
            account_number     = _pad(s("account_number", "000000000"), 17),
            account_type       = s("account_type", "C", 1),
            discretionary_data = _pad(s("discretionary_data", "  "), 2),
            addenda_info       = s("addenda_info") or None,
            company_name       = _pad(s("company_name", "COMPANY"), 16),
            company_id_number  = _pad(s("company_id_number", "1000000000"), 10),
            company_entry_desc = _pad(s("company_entry_desc", "PAYMENT"), 10),
            sec_code           = s("sec_code", "PPD", 3),
            service_class_code = s("service_class_code", "200", 3),
            company_disc_data  = _pad(s("company_disc_data"), 20),
            odfi_routing       = s("odfi_routing", "021000021", 9),
            immediate_dest     = _pad(s("immediate_dest", " 021000021"), 10),
            immediate_origin   = _pad(s("immediate_origin", "021000021 "), 10),
            dest_short_name    = _pad(s("dest_short_name", "JPMORGAN CHASE NA  "), 23),
            origin_short_name  = _pad(s("origin_short_name", "FINSLM PAYMENTS    "), 23),
            status             = s("status", "PENDING"),
        )

    # ── Synthetic fallbacks (when corpus is empty) ────────────────────────────

    def _mock_transactions(self, n: int, sec_code: Optional[str]) -> List[TransactionRecord]:
        from data.generator import (
            VALID_ROUTING_NUMBERS, INDIVIDUAL_NAMES, COMPANY_NAMES
        )
        sec   = sec_code or random.choice(["PPD", "CCD", "WEB", "TEL"])
        co    = random.choice(COMPANY_NAMES)
        co_id = f"1{random.randint(10**8, 10**9 - 1)}"
        eff   = (datetime.now() + timedelta(days=2)).strftime("%y%m%d")
        odfi  = self._default_odfi()

        return [
            TransactionRecord(
                transaction_id     = f"mock_{i:05d}",
                account_id         = f"acc_mock_{i}",
                company_id         = "co_mock_001",
                transaction_code   = random.choice(["22", "27", "32", "37"]),
                amount_cents       = random.randint(100, 500_000),
                effective_date_str = eff,
                individual_id      = _pad(f"ID{i:013d}", 15),
                individual_name    = _pad(random.choice(INDIVIDUAL_NAMES), 22),
                rdfi_routing       = random.choice(VALID_ROUTING_NUMBERS)[:8],
                rdfi_check_digit   = random.choice(VALID_ROUTING_NUMBERS)[8],
                account_number     = _pad(str(random.randint(10**6, 10**9)), 17),
                account_type       = "C",
                discretionary_data = "  ",
                addenda_info       = None,
                company_name       = _pad(co, 16),
                company_id_number  = _pad(co_id, 10),
                company_entry_desc = "PAYROLL   ",
                sec_code           = sec,
                service_class_code = "200",
                company_disc_data  = _pad("", 20),
                odfi_routing       = odfi.routing_number,
                immediate_dest     = odfi.immediate_dest,
                immediate_origin   = odfi.immediate_origin,
                dest_short_name    = odfi.dest_short_name,
                origin_short_name  = odfi.origin_short_name,
            )
            for i in range(n)
        ]

    def _mock_companies(self, sec_code: Optional[str]) -> List[CompanyRecord]:
        from data.generator import COMPANY_NAMES
        secs = [sec_code or s for s in ["PPD", "CCD", "WEB", "TEL", "PPD"]]
        return [
            CompanyRecord(
                company_id         = f"co_mock_{i + 1:03d}",
                company_name       = _pad(n, 16),
                company_id_number  = _pad(f"1{i:09d}", 10),
                company_entry_desc = "PAYROLL   ",
                sec_code           = secs[i % len(secs)],
                service_class_code = "200",
                odfi_id            = "odfi_001",
            )
            for i, n in enumerate(COMPANY_NAMES[:5])
        ]
