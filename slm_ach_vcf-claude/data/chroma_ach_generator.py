"""
ChromaDB-backed ACH NACHA File Generator
Reads transaction data from ChromaDB and produces spec-compliant
94-character-per-record NACHA files.

Replaces: data/oracle_ach_generator.py

Key differences from the Oracle version:
  - Data source is ChromaDB collections (db/chromadb_store/) instead of SQL
  - Repository calls use ChromaDB `get`/`where` filters instead of SQL WHERE
  - Generated files are stored back into ChromaDB (ach_file_log, ach_training_corpus)
  - Falls back to synthetic data when the ChromaDB corpus is empty
  - Corpus similarity search (via ChromaDB embeddings) can seed training
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

from db.ach_repository import ACHRepository, TransactionRecord

log = logging.getLogger(__name__)

_MODIFIER_CYCLE = list(string.ascii_uppercase)


class ChromaACHGenerator:
    """
    Generates NACHA ACH files from ChromaDB transaction data.
    Falls back to synthetic data automatically when ChromaDB is empty.
    """

    def __init__(self, store_path: Optional[str] = None):
        self.repo = ACHRepository(store_path)
        self._file_seq = 0

    # ─── Public API ──────────────────────────────────────────────────────────

    def generate(
        self,
        company_id:       Optional[str] = None,
        sec_code:         Optional[str] = None,
        effective_date:   Optional[str] = None,
        max_transactions: int = 5000,
        save_audit:       bool = True,
        save_corpus:      bool = True,
    ) -> Dict[str, Any]:
        """
        Generate a complete NACHA file from ChromaDB PENDING transactions.

        Returns dict with keys:
            content, file_name, batch_count, entry_count,
            total_debit, total_credit, source, transactions, file_id
        """
        transactions = self.repo.get_pending_transactions(
            company_id=company_id,
            sec_code=sec_code,
            effective_date=effective_date,
            max_rows=max_transactions,
        )

        source = "chromadb" if transactions and not transactions[0].transaction_id.startswith("mock_") \
                 else "synthetic"
        log.info("Fetched %d transactions (source=%s)", len(transactions), source)

        if not transactions:
            from data.generator import ACHGenerator
            content = ACHGenerator().generate_file(
                num_batches=1, entries_per_batch=5,
                sec_code=sec_code or "PPD",
            )
            return {
                "content": content, "file_name": "synthetic_fallback.ach",
                "batch_count": 1, "entry_count": 5, "total_debit": 0,
                "total_credit": 0, "source": "synthetic_fallback",
                "transactions": [], "file_id": None,
            }

        modifier = _MODIFIER_CYCLE[self._file_seq % 26]
        self._file_seq += 1
        file_date = datetime.now().strftime("%y%m%d")
        file_time = datetime.now().strftime("%H%M")

        records, stats = self._build_records(transactions, modifier, file_date, file_time)
        content = "\n".join(records)

        file_name = f"ACH_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{modifier}.ach"
        file_id: Optional[str] = None

        # Audit log → ach_file_log collection
        if save_audit:
            sec_set = {t.sec_code for t in transactions}
            file_id = self.repo.log_file(
                file_name    = file_name,
                modifier     = modifier,
                odfi_id      = "odfi_001",
                batch_count  = stats["batch_count"],
                entry_count  = stats["entry_count"],
                block_count  = stats["block_count"],
                total_debit  = stats["total_debit"],
                total_credit = stats["total_credit"],
                content      = content,
                sec_codes    = ",".join(sorted(sec_set)),
            )

        # Training corpus → ach_training_corpus collection
        if save_corpus and transactions:
            t0 = transactions[0]
            self.repo.save_corpus_entry(
                content     = content,
                sec_code    = t0.sec_code,
                scc         = t0.service_class_code,
                batches     = stats["batch_count"],
                entries     = stats["entry_count"],
                split       = "TRAIN" if random.random() < 0.85 else "VAL",
                file_log_id = file_id or "",
                source      = source,
            )

        return {
            "content":      content,
            "file_name":    file_name,
            "batch_count":  stats["batch_count"],
            "entry_count":  stats["entry_count"],
            "total_debit":  stats["total_debit"],
            "total_credit": stats["total_credit"],
            "source":       source,
            "transactions": [t.transaction_id for t in transactions],
            "file_id":      file_id,
        }

    def seed_transactions(
        self,
        n: int = 50,
        sec_code: Optional[str] = None,
        company_id: Optional[str] = None,
    ) -> int:
        """
        Populate ChromaDB with synthetic transactions for demonstration.
        Returns the number of transactions added.
        """
        from data.generator import (
            ACHGenerator, VALID_ROUTING_NUMBERS, INDIVIDUAL_NAMES,
            COMPANY_NAMES, SEC_CODES,
        )
        repo = self.repo
        odfi = repo.get_odfi()
        sec  = sec_code or random.choice(SEC_CODES)
        co_name = random.choice(COMPANY_NAMES)
        co_id = company_id or f"co_{random.randint(1000, 9999)}"
        co_id_num = f"1{random.randint(10**8, 10**9 - 1)}"
        eff_date = (datetime.now() + timedelta(days=2)).strftime("%y%m%d")

        added = 0
        for i in range(n):
            routing = random.choice(VALID_ROUTING_NUMBERS)
            tc = random.choice(["22", "27", "32", "37"])
            from db.ach_repository import TransactionRecord, _pad
            txn = TransactionRecord(
                transaction_id     = "",
                account_id         = f"acc_seed_{i:05d}",
                company_id         = co_id,
                transaction_code   = tc,
                amount_cents       = random.randint(100, 500_000),
                effective_date_str = eff_date,
                individual_id      = _pad(f"ID{i:013d}", 15),
                individual_name    = _pad(random.choice(INDIVIDUAL_NAMES), 22),
                rdfi_routing       = routing[:8],
                rdfi_check_digit   = routing[8],
                account_number     = _pad(str(random.randint(10**6, 10**9)), 17),
                account_type       = random.choice(["C", "S"]),
                discretionary_data = "  ",
                addenda_info       = None,
                company_name       = _pad(co_name, 16),
                company_id_number  = _pad(co_id_num, 10),
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
            repo.add_transaction(txn)
            added += 1

        log.info("Seeded %d transactions into ChromaDB", added)
        return added

    def generate_training_batch(
        self,
        n_files: int = 200,
        sec_code: Optional[str] = None,
        callback=None,
    ) -> List[str]:
        """
        Generate multiple ACH files for SLM training.
        First checks ChromaDB corpus; augments with synthetic as needed.
        """
        # 1. Pull from existing corpus (similarity-diverse sample)
        corpus = self.repo.fetch_corpus(split="TRAIN", sec_code=sec_code, limit=n_files)
        files = list(corpus)

        # 2. Generate fresh files from ChromaDB transactions
        need = n_files - len(files)
        for i in range(max(need, 10)):
            try:
                result = self.generate(
                    sec_code         = sec_code,
                    max_transactions = random.randint(5, 30),
                    save_audit       = False,
                    save_corpus      = False,
                )
                files.append(result["content"])
            except Exception as e:
                log.warning("Training file %d error: %s", i, e)

            if callback and i % 20 == 0:
                pct = 20 + int((i / max(need, 1)) * 45)
                callback(f"Generated {i}/{need} training files from ChromaDB...", pct)

        random.shuffle(files)
        return files[:n_files]

    # ─── NACHA record builders ────────────────────────────────────────────────

    def _build_records(
        self,
        transactions: List[TransactionRecord],
        modifier:     str,
        file_date:    str,
        file_time:    str,
    ) -> Tuple[List[str], Dict]:

        key_fn = lambda t: (t.company_id, t.sec_code, t.service_class_code)
        batches: List[List[TransactionRecord]] = [
            list(grp)
            for _, grp in groupby(sorted(transactions, key=key_fn), key=key_fn)
        ]

        records: List[str] = []
        total_debit = total_credit = total_entries = all_routing_sum = 0
        batch_num = 0

        first = transactions[0]
        records.append(self._file_header(first, file_date, file_time, modifier))

        for batch_txns in batches:
            batch_num += 1
            t0       = batch_txns[0]
            eff_date = t0.effective_date_str

            b_records = [self._batch_header(t0, batch_num, eff_date)]
            debit = credit = routing_sum = 0
            entry_seq = 0

            for txn in batch_txns:
                entry_seq += 1
                entry, addenda = self._entry_detail(
                    txn, batch_num, entry_seq, t0.odfi_routing[:8]
                )
                b_records.append(entry)
                if addenda:
                    b_records.append(addenda)

                amt = txn.amount_cents
                if txn.transaction_code in {"27","28","29","37","38","39","47","48","49"}:
                    debit  += amt
                else:
                    credit += amt
                routing_sum += int(txn.rdfi_routing)

            n_ent = len(batch_txns) + sum(1 for t in batch_txns if t.addenda_info)
            b_records.append(self._batch_control(
                t0, batch_num, n_ent,
                str(routing_sum)[-10:].zfill(10),
                debit, credit,
            ))

            records.extend(b_records)
            total_debit       += debit
            total_credit      += credit
            total_entries     += n_ent
            all_routing_sum   += routing_sum

        file_hash   = str(all_routing_sum)[-10:].zfill(10)
        block_count = (len(records) + 1) // 10 + 1
        records.append(self._file_control(
            batch_num, block_count, total_entries,
            file_hash, total_debit, total_credit,
        ))

        pad = (10 - (len(records) % 10)) % 10
        records.extend(["9" * 94] * pad)

        return records, dict(
            batch_count  = batch_num,
            entry_count  = total_entries,
            block_count  = len(records) // 10,
            total_debit  = total_debit,
            total_credit = total_credit,
        )

    def _file_header(self, t: TransactionRecord, date, time, modifier) -> str:
        rec = (
            "1" "01"
            f"{t.immediate_dest[:10].ljust(10)}"
            f"{t.immediate_origin[:10].ljust(10)}"
            f"{date}{time}{modifier}"
            "094" "10" "1"
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
            "      "                        # Company descriptive date
            f"{eff_date}"
            "   "                           # Settlement date (bank fills)
            "1"                             # Originator status
            f"{t.odfi_routing[:8]}"
            f"{str(batch_num).zfill(7)}"
        )
        return rec[:94].ljust(94)

    def _entry_detail(
        self, t: TransactionRecord, batch_num: int, seq: int, odfi8: str
    ) -> Tuple[str, Optional[str]]:
        has_addenda = bool(t.addenda_info)
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
            f"{'1' if has_addenda else '0'}"
            f"{trace}"
        )
        addenda = None
        if has_addenda:
            addenda = (
                "7" "05"
                f"{(t.addenda_info or '')[:80].ljust(80)}"
                f"{str(seq).zfill(4)}"
                f"{trace}"
            )[:94].ljust(94)

        return entry[:94].ljust(94), addenda

    def _batch_control(
        self, t: TransactionRecord, batch_num: int, entry_count: int,
        entry_hash: str, debit: int, credit: int,
    ) -> str:
        rec = (
            "8"
            f"{t.service_class_code}"
            f"{str(entry_count).zfill(6)}"
            f"{entry_hash}"
            f"{str(debit).zfill(12)}"
            f"{str(credit).zfill(12)}"
            f"{t.company_id_number[:10].ljust(10)}"
            f"{''.ljust(19)}"
            f"{''.ljust(6)}"
            f"{t.odfi_routing[:8]}"
            f"{str(batch_num).zfill(7)}"
        )
        return rec[:94].ljust(94)

    def _file_control(
        self, batch_count, block_count, entry_count,
        entry_hash, total_debit, total_credit,
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
