"""
ChromaDB VCF Repository
Stores and retrieves VCF transaction data across all 10 categories.
Collections mirror the VCFCategory enum values.
"""

import os
import sys
import json
import uuid
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any

BASE = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, BASE)

from db.chroma_client import get_client, ChromaDBClient

log = logging.getLogger(__name__)

VCF_COLLECTIONS = [
    "vcf_card_mgmt",
    "vcf_purchase",
    "vcf_cash",
    "vcf_hotel",
    "vcf_car_rental",
    "vcf_airline",
    "vcf_refund",
    "vcf_chargeback",
    "vcf_transfer",
    "vcf_recurring",
    "vcf_file_log",
    "vcf_training_corpus",
]

_now = lambda: datetime.now().isoformat()
_uid = lambda p="": f"{p}{uuid.uuid4().hex[:12]}"


class VCFRepository:
    """ChromaDB-backed VCF data store for all transaction categories."""

    def __init__(self, store_path: Optional[str] = None):
        self._db = get_client(store_path)
        self._ensure_vcf_collections()

    def _ensure_vcf_collections(self):
        from db.chroma_client import _EMBEDDING_FN
        for name in VCF_COLLECTIONS:
            try:
                self._db._client.get_or_create_collection(
                    name=name,
                    embedding_function=_EMBEDDING_FN,
                    metadata={"hnsw:space": "cosine"},
                )
            except Exception as e:
                log.warning("VCF collection %s: %s", name, e)

    def _col(self, name: str):
        from db.chroma_client import _EMBEDDING_FN
        return self._db._client.get_or_create_collection(
            name=name,
            embedding_function=_EMBEDDING_FN,
            metadata={"hnsw:space": "cosine"},
        )

    # ── File log ───────────────────────────────────────────────────────────────

    def log_vcf_file(self, file_name: str, content: str, metadata: dict) -> str:
        col  = self._col("vcf_file_log")
        fid  = f"vcf_file_{datetime.now().strftime('%Y%m%d%H%M%S')}_{metadata.get('modifier','A')}"
        meta = {
            "file_name":       file_name,
            "transaction_count": metadata.get("transaction_count", 0),
            "total_amount":    metadata.get("total_amount", 0.0),
            "categories":      metadata.get("categories", ""),
            "acquirer_bin":    metadata.get("acquirer_bin", ""),
            "generation_method": metadata.get("generation_method", "CHROMA"),
            "created_at":      _now(),
        }
        col.add(ids=[fid], documents=[content], metadatas=[meta])
        return fid

    def get_file_log(self, limit: int = 20) -> List[dict]:
        col = self._col("vcf_file_log")
        res = col.get(limit=limit)
        if not res or not res.get("ids"):
            return []
        return [{"file_id": rid, **m}
                for rid, m in zip(res["ids"], res["metadatas"])]

    # ── Training corpus ────────────────────────────────────────────────────────

    def save_vcf_corpus(self, content: str, category: str, split: str = "TRAIN",
                        file_log_id: str = "") -> str:
        col = self._col("vcf_training_corpus")
        cid = _uid("vcf_corpus_")
        col.add(
            ids=[cid],
            documents=[content],
            metadatas=[{
                "category":             category,
                "split_type":           split,
                "file_log_id":          file_log_id,
                "is_used_for_training": "N",
                "created_at":           _now(),
            }],
        )
        return cid

    def fetch_vcf_corpus(self, split: str = "TRAIN", category: Optional[str] = None,
                         limit: int = 500, similar_to: Optional[str] = None) -> List[str]:
        col   = self._col("vcf_training_corpus")
        total = col.count()
        if total == 0:
            return []
        where = {"split_type": {"$eq": split}}
        if category:
            where = {"$and": [{"split_type": {"$eq": split}},
                               {"category":   {"$eq": category}}]}
        try:
            if similar_to:
                res  = col.query(query_texts=[similar_to],
                                 n_results=min(limit, max(total,1)), where=where)
                return [d for d in res.get("documents",[[]])[0] if d]
            else:
                res  = col.get(where=where, limit=limit)
                return [d for d in res.get("documents",[]) if d]
        except Exception as e:
            log.warning("fetch_vcf_corpus: %s", e)
            return []

    def vcf_corpus_stats(self) -> dict:
        col   = self._col("vcf_training_corpus")
        total = col.count()
        stats = {"total": total}
        for split in ("TRAIN","VAL","TEST"):
            try:
                res = col.get(where={"split_type": {"$eq": split}}, limit=100000)
                stats[split.lower()] = len(res.get("ids",[]))
            except Exception:
                stats[split.lower()] = 0
        return stats

    # ── Store individual transaction records ───────────────────────────────────

    def store_transactions(self, records: List[dict], category: str) -> int:
        """Store VCF transaction metadata into category collection."""
        col_name = f"vcf_{category.lower()}"
        if col_name not in VCF_COLLECTIONS:
            log.warning("Unknown VCF category collection: %s", col_name)
            return 0
        try:
            col = self._col(col_name)
        except Exception:
            return 0

        ids, docs, metas = [], [], []
        for rec in records:
            tid = _uid(f"vcf_{category[:3]}_")
            ids.append(tid)
            docs.append(rec.get("document", rec.get("merchant_name", "VCF TRANSACTION")))
            metas.append({k: str(v)[:500] for k, v in rec.items()
                          if isinstance(v, (str, int, float, bool)) and k != "document"})

        if ids:
            col.add(ids=ids, documents=docs, metadatas=metas)
        return len(ids)

    # ── Health ─────────────────────────────────────────────────────────────────

    def health(self) -> dict:
        counts = {}
        for name in VCF_COLLECTIONS:
            try:
                counts[name] = self._db._client.get_collection(name).count()
            except Exception:
                counts[name] = -1
        return {"vcf_collections": counts, "total": sum(v for v in counts.values() if v >= 0)}
