"""
Synthetic Data Generator for ACH NACHA and VISA VCF files
Generates realistic training data for the SLM
"""

import random
import string
from datetime import datetime, timedelta
from typing import List, Optional


# ─── Realistic Reference Data ──────────────────────────────────────────────────

COMPANY_NAMES = [
    "ACME CORP", "PAYROLL SOLUTIONS", "TECH INDUSTRIES", "GLOBAL FINANCE",
    "METRO HEALTH PLAN", "CITY WATER DEPT", "PREMIER AUTO PAY", "NEXGEN PAYROLL",
    "SUNRISE MORTGAGE", "BLUE RIDGE UTILS", "ALPINE INSURANCE", "CEDAR BENEFITS",
    "HORIZON STAFFING", "CRESTVIEW BANK", "LAKESIDE LENDING", "SUMMIT CREDIT UN",
    "PINECREST PAYMT", "RIVERSIDE ENERGY", "OCEANVIEW SVCS", "WESTFIELD PAY CO",
]

INDIVIDUAL_NAMES = [
    "JOHN SMITH", "JANE DOE", "ROBERT JOHNSON", "MARY WILLIAMS", "DAVID BROWN",
    "PATRICIA JONES", "MICHAEL DAVIS", "LINDA MILLER", "WILLIAM WILSON", "BARBARA MOORE",
    "JAMES TAYLOR", "ELIZABETH ANDERSON", "RICHARD THOMAS", "JENNIFER JACKSON",
    "CHARLES WHITE", "MARGARET HARRIS", "JOSEPH MARTIN", "DOROTHY THOMPSON",
    "THOMAS GARCIA", "JESSICA MARTINEZ", "CHRISTOPHER ROBINSON", "SARAH CLARK",
    "DANIEL RODRIGUEZ", "KAREN LEWIS", "MATTHEW LEE", "NANCY WALKER", "ANTHONY HALL",
    "BETTY ALLEN", "MARK YOUNG", "HELEN HERNANDEZ",
]

BANK_NAMES = [
    "FIRST NATIONAL BANK", "CHASE BANK NA", "WELLS FARGO BANK", "BANK OF AMERICA",
    "CITIBANK NA", "US BANK NA", "TRUIST BANK", "PNC BANK NA", "CAPITAL ONE NA",
    "TD BANK NA", "REGIONS BANK", "BB&T CORP", "SUNTRUST BANK", "FIFTH THIRD BANK",
    "KEYBANK NA", "HUNTINGTON NATL", "CITIZENS BANK NA", "M&T BANK CORP",
    "COMERICA BANK", "ZIONS BANCORP", "CULLEN FROST BK", "OLD NATIONAL BNK",
]

# Valid ABA routing numbers (real bank routing numbers for testing)
VALID_ROUTING_NUMBERS = [
    "021000021",  # JPMorgan Chase
    "021200339",  # Citibank
    "021300077",  # HSBC Bank
    "021301115",  # Bank of America
    "022300173",  # NYCB
    "031201360",  # Wells Fargo
    "031207607",  # TD Bank
    "036001808",  # Barclays
    "044000037",  # Fifth Third
    "051000017",  # Bank of America
    "054001204",  # Capital One
    "055002707",  # PNC Bank
    "061000052",  # Regions Bank
    "061000104",  # SunTrust
    "071000013",  # US Bank
    "071000505",  # Chase
    "071921891",  # Fifth Third
    "073000545",  # Veridian CU
    "082000073",  # Simmons
    "091000019",  # US Bank
    "101000187",  # Commerce Bank
    "102000076",  # Wells Fargo
    "103000648",  # BancFirst
    "104000016",  # First National
    "107001481",  # BBVA
    "111000614",  # Frost Bank (corrected)
    "111000025",  # JPMorgan Chase
    "113000023",  # Bank of America
    "121000248",  # Wells Fargo
    "122000247",  # Bank of America
]

SEC_CODES = ["PPD", "CCD", "CTX", "WEB", "TEL"]

TRANSACTION_CODES_CREDIT = ["22", "32"]  # Checking/Savings credit
TRANSACTION_CODES_DEBIT = ["27", "37"]   # Checking/Savings debit

MERCHANT_NAMES = [
    "AMAZON.COM", "WALMART STORE 4521", "TARGET CORP", "COSTCO WHSE 0123",
    "HOME DEPOT 0456", "BEST BUY 00789", "KROGER #5521", "WALGREENS #1234",
    "CVS PHARMACY 456", "MCDONALDS #12345", "STARBUCKS #67890", "NETFLIX.COM",
    "APPLE.COM/BILL", "GOOGLE *GSUITE", "UBER TRIP", "LYFT RIDE",
    "AIRBNB.COM", "DELTA AIR LINES", "UNITED AIRLINES", "AMERICAN AIRLINES",
]

MCC_CODES = [
    "5411",  # Grocery Stores
    "5912",  # Drug Stores
    "5732",  # Electronics
    "5812",  # Eating Places
    "5541",  # Service Stations
    "5311",  # Department Stores
    "4816",  # Computer Networks
    "7011",  # Hotels
    "4511",  # Airlines
    "5999",  # Miscellaneous
]


# ─── ACH Generator ─────────────────────────────────────────────────────────────

class ACHGenerator:
    """Generates realistic synthetic ACH NACHA files for training"""

    def generate_file(self,
                      num_batches: int = None,
                      entries_per_batch: int = None,
                      sec_code: str = None,
                      file_date: str = None) -> str:
        """Generate a complete NACHA ACH file"""
        num_batches = num_batches or random.randint(1, 3)
        entries_per_batch = entries_per_batch or random.randint(2, 8)
        sec_code = sec_code or random.choice(SEC_CODES)
        file_date = file_date or self._random_date()

        odfi_routing = random.choice(VALID_ROUTING_NUMBERS)
        file_id = random.choice(string.ascii_uppercase)

        records = []
        records.append(self._file_header(odfi_routing, file_date, file_id))

        total_entry_count = 0
        total_debit = 0
        total_credit = 0
        all_routing_sums = 0
        batch_count = 0

        for b in range(num_batches):
            batch_num = b + 1
            n_entries = entries_per_batch or random.randint(2, 10)
            scc = self._choose_service_class()

            batch_records, entry_count, debit_sum, credit_sum, routing_sum = \
                self._generate_batch(batch_num, odfi_routing[:8], sec_code, scc, n_entries, file_date)

            records.extend(batch_records)
            total_entry_count += entry_count
            total_debit += debit_sum
            total_credit += credit_sum
            all_routing_sums += routing_sum
            batch_count += 1

        records.append(self._file_control(batch_count, records, total_entry_count,
                                          all_routing_sums, total_debit, total_credit))

        # Pad to block size (multiple of 10 records)
        record_count = len(records)
        padding_needed = (10 - (record_count % 10)) % 10
        for _ in range(padding_needed):
            records.append('9' * 94)

        return '\n'.join(records)

    def generate_dataset(self, n: int = 100) -> List[str]:
        """Generate n ACH files for training"""
        files = []
        for _ in range(n):
            n_batches = random.randint(1, 4)
            n_entries = random.randint(2, 15)
            sec = random.choice(SEC_CODES)
            files.append(self.generate_file(n_batches, n_entries, sec))
        return files

    def _file_header(self, odfi: str, date: str, file_id: str) -> str:
        dest_name = random.choice(BANK_NAMES)[:23].ljust(23)
        origin_name = random.choice(COMPANY_NAMES)[:23].ljust(23)
        time = f"{random.randint(0,23):02d}{random.randint(0,59):02d}"
        rec = (
            "1"               # Record type
            "01"              # Priority code
            f" {odfi[:9]}"    # Immediate destination (space + 9 digits)
            f"{odfi[:9]} "    # Immediate origin
            f"{date}"         # File creation date YYMMDD
            f"{time}"         # File creation time HHMM
            f"{file_id}"      # File ID modifier
            "094"             # Record size
            "10"              # Blocking factor
            "1"               # Format code
            f"{dest_name}"    # Immediate destination name
            f"{origin_name}"  # Immediate origin name
            "        "        # Reference code (8 spaces)
        )
        return rec[:94].ljust(94)

    def _generate_batch(self, batch_num, odfi8, sec, scc, n_entries, file_date):
        company = random.choice(COMPANY_NAMES)
        company_id = f"1{random.randint(10**8, 10**9-1)}"
        eff_date = self._effective_date(file_date)

        header = self._batch_header(batch_num, odfi8, company, company_id, sec, scc, eff_date)
        entries = []
        debit_sum = 0
        credit_sum = 0
        routing_sum = 0

        for e in range(n_entries):
            routing = random.choice(VALID_ROUTING_NUMBERS)
            tc = random.choice(TRANSACTION_CODES_CREDIT if scc == "220"
                               else TRANSACTION_CODES_DEBIT if scc == "225"
                               else TRANSACTION_CODES_CREDIT + TRANSACTION_CODES_DEBIT)
            amount = random.randint(100, 500000)  # 1.00 to 5000.00
            entry, trace = self._entry_detail(e + 1, routing, tc, amount, batch_num, odfi8)
            entries.append(entry)
            routing_sum += int(routing[:8])
            if tc in TRANSACTION_CODES_DEBIT:
                debit_sum += amount
            else:
                credit_sum += amount

        entry_hash = str(routing_sum)[-10:].zfill(10)
        ctrl = self._batch_control(batch_num, odfi8, company_id, scc, n_entries,
                                   entry_hash, debit_sum, credit_sum)

        records = [header] + entries + [ctrl]
        return records, n_entries, debit_sum, credit_sum, routing_sum

    def _batch_header(self, batch_num, odfi8, company, company_id, sec, scc, eff_date):
        rec = (
            "5"
            f"{scc}"
            f"{company[:16].ljust(16)}"
            f"{''.ljust(20)}"            # Discretionary data
            f"{company_id[:10].ljust(10)}"
            f"{sec}"
            f"{'PAYROLL'.ljust(10)}"     # Company entry description
            f"{''.ljust(6)}"             # Descriptive date
            f"{eff_date}"                # Effective entry date
            f"{'   '}"                   # Settlement date (bank fills)
            "1"                          # Originator status
            f"{odfi8}"
            f"{str(batch_num).zfill(7)}"
        )
        return rec[:94].ljust(94)

    def _entry_detail(self, seq, routing, tc, amount, batch_num, odfi8):
        rdfi = routing[:8]
        check = routing[8] if len(routing) > 8 else '0'
        account = f"{random.randint(10**6, 10**9-1)}".ljust(17)[:17]
        ind_id = f"ID{random.randint(10000, 99999)}".ljust(15)[:15]
        name = random.choice(INDIVIDUAL_NAMES)[:22].ljust(22)
        disc = "  "
        trace = f"{odfi8}{str(seq).zfill(7)}"

        rec = (
            "6"
            f"{tc}"
            f"{rdfi}"
            f"{check}"
            f"{account}"
            f"{str(amount).zfill(10)}"
            f"{ind_id}"
            f"{name}"
            f"{disc}"
            "0"              # No addenda
            f"{trace}"
        )
        return rec[:94].ljust(94), trace

    def _batch_control(self, batch_num, odfi8, company_id, scc, entry_count,
                       entry_hash, debit, credit):
        rec = (
            "8"
            f"{scc}"
            f"{str(entry_count).zfill(6)}"
            f"{entry_hash}"
            f"{str(debit).zfill(12)}"
            f"{str(credit).zfill(12)}"
            f"{company_id[:10].ljust(10)}"
            f"{''.ljust(19)}"            # Message auth code
            f"{''.ljust(6)}"             # Reserved
            f"{odfi8}"
            f"{str(batch_num).zfill(7)}"
        )
        return rec[:94].ljust(94)

    def _file_control(self, batch_count, records, entry_count, routing_sum,
                      total_debit, total_credit):
        block_count = len(records) // 10 + 1
        entry_hash = str(routing_sum)[-10:].zfill(10)
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

    def _choose_service_class(self):
        return random.choice(["200", "220", "225"])

    def _random_date(self):
        dt = datetime.now() - timedelta(days=random.randint(0, 365))
        return dt.strftime("%y%m%d")

    def _effective_date(self, file_date):
        year = int("20" + file_date[:2])
        month = int(file_date[2:4])
        day = int(file_date[4:6])
        try:
            dt = datetime(year, month, day) + timedelta(days=random.randint(1, 3))
            return dt.strftime("%y%m%d")
        except Exception:
            return file_date


# ─── VCF Generator ─────────────────────────────────────────────────────────────

class VCFGenerator:
    """Generates realistic synthetic VISA VCF files for training"""

    def generate_file(self,
                      num_transactions: int = None,
                      file_date: str = None,
                      acquirer_bin: str = None) -> str:
        """Generate a complete VISA VCF file in pipe-delimited format"""
        num_transactions = num_transactions or random.randint(5, 50)
        file_date = file_date or datetime.now().strftime("%Y%m%d")
        acquirer_bin = acquirer_bin or f"{random.randint(400000, 499999)}"

        lines = []
        lines.append(self._vcf_header(file_date, acquirer_bin))

        total_amount = 0.0
        for _ in range(num_transactions):
            txn, amt = self._transaction_record()
            lines.append(txn)
            total_amount += amt

        lines.append(self._vcf_trailer(num_transactions, total_amount, file_date))
        return '\n'.join(lines)

    def generate_dataset(self, n: int = 100) -> List[str]:
        files = []
        for _ in range(n):
            n_txns = random.randint(5, 100)
            files.append(self.generate_file(n_txns))
        return files

    def _vcf_header(self, date: str, acq_bin: str) -> str:
        version = "2.0"
        proc_date = date
        proc_time = f"{random.randint(0,23):02d}{random.randint(0,59):02d}{random.randint(0,59):02d}"
        bank_name = random.choice(BANK_NAMES)[:30]
        return f"VCF|{version}|{acq_bin}|{date}|{proc_time}|{bank_name}|PRODUCTION|001"

    def _transaction_record(self):
        tc = random.choice(["06", "05", "10", "25"])
        pan = self._generate_pan()
        proc_code = random.choice(list(VALID_PROCESSING_CODES.keys()))
        amount = round(random.uniform(1.0, 9999.99), 2)
        currency = random.choice(["840", "978", "826", "392", "124"])
        txn_dt = (datetime.now() - timedelta(minutes=random.randint(0, 1440))).strftime("%Y%m%d%H%M%S")
        mcc = random.choice(MCC_CODES)
        pos = random.choice(["05", "02", "07", "90"])
        resp = "00" if tc in ("05", "06") else random.choice(["00", "05", "51"])
        auth_code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
        mid = ''.join(random.choices(string.ascii_uppercase + string.digits, k=15))
        tid = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
        merchant_name = random.choice(MERCHANT_NAMES)[:25]
        merchant_city = random.choice(["NEW YORK", "LOS ANGELES", "CHICAGO", "HOUSTON", "PHOENIX"])
        arn = f"{random.randint(10**22, 10**23-1)}"
        rref = ''.join(random.choices(string.digits, k=12))
        interchange = round(amount * random.uniform(0.015, 0.025), 2)
        stan = ''.join(random.choices(string.digits, k=6))
        country = random.choice(["840", "826", "484", "124", "036"])

        line = "|".join([
            tc, pan, proc_code, f"{amount:.2f}", currency, txn_dt,
            mcc, pos, resp, auth_code, mid, tid,
            merchant_name, merchant_city, arn, rref, f"{interchange:.2f}",
            stan, country, "00", "0"
        ])
        return line, amount

    def _vcf_trailer(self, count: int, total_amount: float, date: str) -> str:
        hash_total = str(int(total_amount * 100))[-16:].zfill(16)
        return f"TRAILER|{count}|{total_amount:.2f}|{hash_total}|{date}|00"

    def _generate_pan(self):
        """Generate a valid Visa PAN with correct Luhn checksum"""
        prefix = "4"
        length = random.choice([16, 16, 16, 13])
        # Generate all digits except last
        pan = [int(c) for c in (prefix + ''.join(random.choices(string.digits, k=length-2)))]
        # Calculate Luhn check digit
        total = 0
        for i, d in enumerate(reversed(pan)):
            if i % 2 == 0:
                d *= 2
                if d > 9:
                    d -= 9
            total += d
        check = (10 - (total % 10)) % 10
        return ''.join(str(d) for d in pan) + str(check)

    # Make valid_processing_codes and valid_pos available
    @property
    def valid_processing_codes(self):
        return list(_VALID_PROCESSING_CODES.keys())

# Module-level lookup (used by VCFGenerator._transaction_record)
VALID_PROCESSING_CODES = {
    "000000": "Purchase",
    "200000": "Credit/Refund",
    "010000": "Withdrawal",
    "190000": "Deposit",
    "090000": "Balance Inquiry",
    "400000": "Transfer From",
    "500000": "Transfer To",
}
