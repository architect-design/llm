"""
Oracle-backed ACH NACHA File Generator
Reads real transaction data from Oracle and produces spec-compliant
94-character-per-record NACHA files.

Flow:
  OracleACHGenerator.generate()
      ├── ACHRepository.get_pending_transactions()  ← Oracle SELECT
      ├── _group_by_company_and_batch()
      ├── _build_records()                          ← assemble 94-char lines
      │     ├── file_header (type 1)
      │     ├── for each batch:
      │     │     ├── batch_header  (type 5)
      │     │     ├── entry_detail  (type 6)  ← one per transaction
      │     │     ├── addenda       (type 7)  ← if addenda_info present
      │     │     └── batch_control (type 8)
      │     ├── file_control  (type 9)
      │     └── padding       (9-filled lines to reach next block of 10)
      ├── ACHRepository.log_file()                  ← Oracle INSERT audit
      └── ACHRepository.save_corpus_entry()         ← Oracle INSERT training data
"""

import os
import sys
import random
import string
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from itertools import groupby

BASE = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, BASE)

from db.oracle_connector import OracleConfig
from db.ach_repository   import ACHRepository, TransactionRecord

log = logging.getLogger(__name__)

# File ID modifier cycles A–Z across files generated in a single day
_MODIFIER_CYCLE = list(string.ascii_uppercase)


class OracleACHGenerator:
    """
    Generates NACHA ACH files by reading transaction data from Oracle.
    Falls back to synthetic data automatically when Oracle is unavailable.
    """

    def __init__(self, config: Optional[OracleConfig] = None):
        self.repo   = ACHRepository(config)
        self.config = config or OracleConfig()
        self._file_seq = 0  # tracks modifier within a session

    # ─── Public entry points ─────────────────────────────────────────────────

    def generate(
        self,
        company_id:     Optional[int] = None,
        sec_code:       Optional[str] = None,
        effective_date: Optional[str] = None,
        max_transactions: int = 5000,
        save_audit:     bool = True,
        save_corpus:    bool = True,
    ) -> Dict[str, Any]:
        """
        Generate a NACHA file from Oracle pending transactions.

        Returns:
            {
              "content":       str,   # full NACHA file text
              "file_name":     str,
              "batch_count":   int,
              "entry_count":   int,
              "total_debit":   int,   # cents
              "total_credit":  int,   # cents
              "source":        str,   # "oracle" | "mock"
              "transactions":  list,  # IDs of included transactions
              "file_id":       int | None,
            }
        """
        # 1. Fetch pending transactions from Oracle
        transactions = self.repo.get_pending_transactions(
            company_id=company_id,
            sec_code=sec_code,
            effective_date=effective_date,
            max_rows=max_transactions,
        )

        source = "mock" if self.repo.pool.is_mock else "oracle"
        log.info("Fetched %d transactions from %s", len(transactions), source)

        if not transactions:
            log.warning("No pending transactions found — generating synthetic file")
            from data.generator import ACHGenerator
            content = ACHGenerator().generate_file(
                num_batches=1, entries_per_batch=5, sec_code=sec_code or "PPD"
            )
            return {
                "content": content, "file_name": "synthetic_fallback.ach",
                "batch_count": 1, "entry_count": 5,
                "total_debit": 0, "total_credit": 0,
                "source": "synthetic_fallback", "transactions": [], "file_id": None,
            }

        # 2. Build NACHA records
        file_modifier = _MODIFIER_CYCLE[self._file_seq % 26]
        self._file_seq += 1
        file_date = datetime.now().strftime("%y%m%d")
        file_time = datetime.now().strftime("%H%M")

        records, stats = self._build_records(transactions, file_modifier, file_date, file_time)
        content = "\n".join(records)

        # 3. Audit log
        file_name = f"ACH_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file_modifier}.ach"
        file_id = None

        if save_audit:
            odfi_id = 1  # default; ideally resolved from transactions[0].odfi_routing
            file_id = self.repo.log_file(
                file_name   = file_name,
                modifier    = file_modifier,
                odfi_id     = odfi_id,
                batch_count = stats["batch_count"],
                entry_count = stats["entry_count"],
                block_count = stats["block_count"],
                total_debit = stats["total_debit"],
                total_credit= stats["total_credit"],
                content     = content,
            )

        # 4. Persist to training corpus
        if save_corpus and transactions:
            first = transactions[0]
            self.repo.save_corpus_entry(
                content    = content,
                sec_code   = first.sec_code,
                scc        = first.service_class_code,
                batches    = stats["batch_count"],
                entries    = stats["entry_count"],
                split      = "TRAIN" if random.random() < 0.85 else "VAL",
                file_log_id= file_id,
            )

        return {
            "content":     content,
            "file_name":   file_name,
            "batch_count": stats["batch_count"],
            "entry_count": stats["entry_count"],
            "total_debit": stats["total_debit"],
            "total_credit":stats["total_credit"],
            "source":      source,
            "transactions":[t.transaction_id for t in transactions],
            "file_id":     file_id,
        }

    def generate_training_batch(
        self,
        n_files: int = 200,
        sec_code: Optional[str] = None,
        callback=None,
    ) -> List[str]:
        """
        Generate multiple ACH files for SLM training.
        Uses Oracle data when available, synthetic otherwise.
        """
        files = []
        for i in range(n_files):
            try:
                result = self.generate(
                    sec_code  = sec_code,
                    max_transactions = random.randint(5, 50),
                    save_audit  = False,
                    save_corpus = False,
                )
                files.append(result["content"])
            except Exception as e:
                log.warning("Training file %d failed: %s", i, e)

            if callback and i % 20 == 0:
                pct = int((i / n_files) * 50)
                callback(f"Generated {i}/{n_files} training files from Oracle...", 20 + pct)

        return files

    def fetch_corpus_files(self, split: str = "TRAIN", limit: int = 500) -> List[str]:
        """Retrieve previously generated files from the corpus table."""
        return self.repo.fetch_corpus(split=split, limit=limit)

    # ─── NACHA record builders ────────────────────────────────────────────────

    def _build_records(
        self,
        transactions: List[TransactionRecord],
        modifier:   str,
        file_date:  str,
        file_time:  str,
    ) -> Tuple[List[str], Dict]:

        # Group transactions by (company_id, sec_code, service_class_code)
        batches: List[List[TransactionRecord]] = []
        key_fn = lambda t: (t.company_id, t.sec_code, t.service_class_code)
        for _, grp in groupby(sorted(transactions, key=key_fn), key=key_fn):
            batches.append(list(grp))

        records: List[str] = []
        total_debit  = 0
        total_credit = 0
        total_entries = 0
        all_routing_sum = 0
        batch_num = 0

        # File Header placeholder (built last once we know ODFI from first txn)
        first_txn = transactions[0]
        records.append(self._file_header(first_txn, file_date, file_time, modifier))

        for batch_txns in batches:
            batch_num += 1
            t0 = batch_txns[0]
            eff_date = t0.effective_date_str

            batch_records = [self._batch_header(t0, batch_num, eff_date)]
            debit = credit = 0
            routing_sum = 0
            entry_seq = 0

            for txn in batch_txns:
                entry_seq += 1
                entry, addenda = self._entry_detail(txn, batch_num, entry_seq, t0.odfi_routing[:8])
                batch_records.append(entry)
                if addenda:
                    batch_records.append(addenda)
                    entry_seq += 1  # addenda counts in entry+addenda count

                amt = txn.amount_cents
                if txn.transaction_code in {"27","28","29","37","38","39","47","48","49"}:
                    debit += amt
                else:
                    credit += amt
                routing_sum += int(txn.rdfi_routing)

            n_entries = len(batch_txns) + sum(1 for t in batch_txns if t.addenda_info)
            entry_hash = str(routing_sum)[-10:].zfill(10)
            batch_records.append(
                self._batch_control(t0, batch_num, n_entries, entry_hash, debit, credit)
            )

            records.extend(batch_records)
            total_debit   += debit
            total_credit  += credit
            total_entries += n_entries
            all_routing_sum += routing_sum

        # File Control
        file_hash = str(all_routing_sum)[-10:].zfill(10)
        block_count = (len(records) + 1) // 10 + 1
        records.append(
            self._file_control(batch_num, block_count, total_entries,
                               file_hash, total_debit, total_credit)
        )

        # Padding to next multiple of 10
        total = len(records)
        pad = (10 - (total % 10)) % 10
        records.extend(["9" * 94] * pad)

        stats = dict(
            batch_count  = batch_num,
            entry_count  = total_entries,
            block_count  = len(records) // 10,
            total_debit  = total_debit,
            total_credit = total_credit,
        )
        return records, stats

    # ── Individual record formatters ─────────────────────────────────────────

    def _file_header(self, t: TransactionRecord, date: str, time: str, modifier: str) -> str:
        rec = (
            "1"
            "01"
            f"{t.immediate_dest[:10].ljust(10)}"
            f"{t.immediate_origin[:10].ljust(10)}"
            f"{date}"
            f"{time}"
            f"{modifier}"
            "094"
            "10"
            "1"
            f"{t.dest_short_name[:23].ljust(23)}"
            f"{t.origin_short_name[:23].ljust(23)}"
            "        "
        )
        return rec[:94].ljust(94)

    def _batch_header(self, t: TransactionRecord, batch_num: int, eff_date: str) -> str:
        rec = (
            "5"
            f"{t.service_class_code}"
            f"{t.company_name[:16].ljust(16)}"
            f"{t.company_disc_data[:20].ljust(20)}"
            f"{t.company_id_number[:10].ljust(10)}"
            f"{t.sec_code}"
            f"{t.company_entry_desc[:10].ljust(10)}"
            f"{'      '}"                    # Company descriptive date (6 spaces)
            f"{eff_date}"
            f"   "                           # Settlement date (bank fills)
            "1"                              # Originator status = ODFI
            f"{t.odfi_routing[:8]}"
            f"{str(batch_num).zfill(7)}"
        )
        return rec[:94].ljust(94)

    def _entry_detail(
        self, t: TransactionRecord, batch_num: int, seq: int, odfi8: str
    ) -> Tuple[str, Optional[str]]:
        addenda_indicator = "1" if t.addenda_info else "0"
        trace = f"{odfi8}{str(seq).zfill(7)}"

        entry = (
            "6"
            f"{t.transaction_code}"
            f"{t.rdfi_routing[:8]}"
            f"{t.rdfi_check_digit}"
            f"{t.account_number[:17].ljust(17)}"
            f"{t.amount_str}"
            f"{t.individual_id[:15].ljust(15)}"
            f"{t.individual_name[:22].ljust(22)}"
            f"{t.discretionary_data[:2]}"
            f"{addenda_indicator}"
            f"{trace}"
        )

        addenda = None
        if t.addenda_info:
            payment_info = t.addenda_info[:80].ljust(80)
            addenda = (
                "7"
                "05"
                f"{payment_info}"
                f"{str(seq).zfill(4)}"
                f"{trace}"
            )[:94].ljust(94)

        return entry[:94].ljust(94), addenda

    def _batch_control(
        self, t: TransactionRecord, batch_num: int, entry_count: int,
        entry_hash: str, debit: int, credit: int
    ) -> str:
        rec = (
            "8"
            f"{t.service_class_code}"
            f"{str(entry_count).zfill(6)}"
            f"{entry_hash}"
            f"{str(debit).zfill(12)}"
            f"{str(credit).zfill(12)}"
            f"{t.company_id_number[:10].ljust(10)}"
            f"{''.ljust(19)}"               # Message auth code
            f"{''.ljust(6)}"                # Reserved
            f"{t.odfi_routing[:8]}"
            f"{str(batch_num).zfill(7)}"
        )
        return rec[:94].ljust(94)

    def _file_control(
        self, batch_count: int, block_count: int, entry_count: int,
        entry_hash: str, total_debit: int, total_credit: int
    ) -> str:
        rec = (
            "9"
            f"{str(batch_count).zfill(6)}"
            f"{str(block_count).zfill(6)}"
            f"{str(entry_count).zfill(8)}"
            f"{entry_hash}"
            f"{str(total_debit).zfill(12)}"
            f"{str(total_credit).zfill(12)}"
            f"{''.ljust(39)}"
        )
        return rec[:94].ljust(94)
