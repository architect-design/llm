"""
ChromaDB Client for FinSLM
Replaces Oracle — uses ChromaDB as the persistent vector + metadata store.

All ChromaDB data files are stored on-disk inside the project at:
  db/chromadb_store/

Collections (analogous to Oracle tables):
  ach_odfi_config        ODFI bank routing configuration
  ach_companies          Originator company master
  ach_accounts           Receiver (RDFI) account master
  ach_transactions       Pending payment transactions
  ach_file_log           Generated file audit trail
  ach_validation_log     Validation run results
  ach_training_corpus    SLM training examples (ACH file text)

ChromaDB concepts used here:
  document   - the primary text body (ACH file content, account number, etc.)
  metadata   - structured dict of searchable/filterable fields
  id         - unique string identifier per record
  embedding  - auto-generated (default) or omitted (metadata-only collections)
"""

import os
import json
import logging
import threading
import time
from typing import Optional, Dict, List, Any
from pathlib import Path

import hashlib
import chromadb
from chromadb.config import Settings
from chromadb import EmbeddingFunction, Documents, Embeddings


class FinancialHashEmbedding(EmbeddingFunction):
    """
    Lightweight offline embedding function for financial text.
    Uses character n-gram hashing — no model download required.
    Works entirely offline; semantic similarity is approximate but sufficient
    for corpus retrieval. Replace with sentence-transformers in production.
    """
    DIM = 256  # embedding dimension

    def __call__(self, input: Documents) -> Embeddings:
        result = []
        for text in input:
            vec = [0.0] * self.DIM
            text = str(text or "")
            # Character n-gram hashing across multiple window sizes
            for n in (2, 3, 4):
                for i in range(max(1, len(text) - n + 1)):
                    gram  = text[i:i + n]
                    h     = int(hashlib.md5(gram.encode()).hexdigest(), 16)
                    idx   = h % self.DIM
                    vec[idx] += 1.0
            # L2-normalise
            norm = (sum(v * v for v in vec) ** 0.5) or 1.0
            vec  = [v / norm for v in vec]
            result.append(vec)
        return result


_EMBEDDING_FN = FinancialHashEmbedding()

log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE      = Path(__file__).parent          # db/
_STORE_DIR = _HERE / "chromadb_store"       # db/chromadb_store/
_STORE_DIR.mkdir(parents=True, exist_ok=True)


# ── Collection names ──────────────────────────────────────────────────────────
class Collections:
    ODFI_CONFIG       = "ach_odfi_config"
    COMPANIES         = "ach_companies"
    ACCOUNTS          = "ach_accounts"
    TRANSACTIONS      = "ach_transactions"
    FILE_LOG          = "ach_file_log"
    VALIDATION_LOG    = "ach_validation_log"
    TRAINING_CORPUS   = "ach_training_corpus"

    ALL = [
        ODFI_CONFIG, COMPANIES, ACCOUNTS, TRANSACTIONS,
        FILE_LOG, VALIDATION_LOG, TRAINING_CORPUS,
    ]


# ── Singleton client ──────────────────────────────────────────────────────────
class ChromaDBClient:
    """
    Thread-safe persistent ChromaDB client.
    All data is saved to  db/chromadb_store/  inside the project folder.
    """

    _instance: Optional["ChromaDBClient"] = None
    _lock = threading.Lock()

    def __init__(self, store_path: Optional[str] = None):
        self._path = str(store_path or _STORE_DIR)
        self._client: Optional[chromadb.PersistentClient] = None
        self._collections: Dict[str, Any] = {}
        self._connect()

    # ── Singleton factory ────────────────────────────────────────────────────
    @classmethod
    def get_instance(cls, store_path: Optional[str] = None) -> "ChromaDBClient":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(store_path)
            return cls._instance

    @classmethod
    def reset(cls):
        with cls._lock:
            cls._instance = None

    # ── Connection ───────────────────────────────────────────────────────────
    def _connect(self):
        try:
            self._client = chromadb.PersistentClient(
                path=self._path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                ),
            )
            self._ensure_collections()
            log.info("ChromaDB connected — store: %s", self._path)
        except Exception as e:
            log.error("ChromaDB connection failed: %s", e)
            raise

    # ── Collection bootstrap ──────────────────────────────────────────────────
    def _ensure_collections(self):
        """Create all collections if they don't already exist."""
        for name in Collections.ALL:
            col = self._client.get_or_create_collection(
                name=name,
                embedding_function=_EMBEDDING_FN,
                metadata={"hnsw:space": "cosine"},
            )
            self._collections[name] = col
            log.debug("Collection ready: %s (%d docs)", name, col.count())

    # ── Public accessors ──────────────────────────────────────────────────────
    def collection(self, name: str):
        if name not in self._collections:
            self._collections[name] = self._client.get_or_create_collection(name)
        return self._collections[name]

    def get_client(self):
        return self._client

    # ── Health ────────────────────────────────────────────────────────────────
    def health(self) -> Dict[str, Any]:
        counts = {}
        for name in Collections.ALL:
            try:
                counts[name] = self.collection(name).count()
            except Exception:
                counts[name] = -1
        return {
            "status":     "ok",
            "store_path": self._path,
            "store_size_mb": round(
                sum(f.stat().st_size for f in Path(self._path).rglob("*") if f.is_file())
                / 1_048_576, 2
            ),
            "collections": counts,
        }

    def test_connection(self) -> tuple[bool, str]:
        try:
            total = sum(self.collection(n).count() for n in Collections.ALL)
            return True, f"ChromaDB OK — {total} total documents across {len(Collections.ALL)} collections"
        except Exception as e:
            return False, f"ChromaDB error: {e}"

    # ── Seed default ODFI if empty ────────────────────────────────────────────
    def seed_defaults(self):
        """Populate default ODFI config if collection is empty."""
        col = self.collection(Collections.ODFI_CONFIG)
        if col.count() > 0:
            return

        default_odfis = [
            {
                "id": "odfi_001",
                "routing_number": "021000021",
                "bank_name": "JPMORGAN CHASE BANK NA",
                "immediate_dest": " 021000021",
                "immediate_origin": "021000021 ",
                "dest_short_name": "JPMORGAN CHASE NA  ",
                "origin_short_name": "FINSLM PAYMENTS    ",
                "is_active": "Y",
            },
            {
                "id": "odfi_002",
                "routing_number": "121000248",
                "bank_name": "WELLS FARGO BANK NA",
                "immediate_dest": " 121000248",
                "immediate_origin": "121000248 ",
                "dest_short_name": "WELLS FARGO BANK   ",
                "origin_short_name": "FINSLM PAYMENTS    ",
                "is_active": "Y",
            },
            {
                "id": "odfi_003",
                "routing_number": "111000025",
                "bank_name": "BANK OF AMERICA NA",
                "immediate_dest": " 111000025",
                "immediate_origin": "111000025 ",
                "dest_short_name": "BANK OF AMERICA NA ",
                "origin_short_name": "FINSLM PAYMENTS    ",
                "is_active": "Y",
            },
        ]

        col.add(
            ids=[o["id"] for o in default_odfis],
            documents=[o["bank_name"] for o in default_odfis],
            metadatas=[{k: v for k, v in o.items() if k != "id"} for o in default_odfis],
        )
        log.info("Seeded %d default ODFI records", len(default_odfis))


# ── Module-level convenience ──────────────────────────────────────────────────

def get_client(store_path: Optional[str] = None) -> ChromaDBClient:
    """Return the singleton ChromaDB client (creates + seeds on first call)."""
    client = ChromaDBClient.get_instance(store_path)
    client.seed_defaults()
    return client


def get_collection(name: str, store_path: Optional[str] = None):
    """Shortcut — return a named collection."""
    return get_client(store_path).collection(name)
