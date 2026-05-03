"""
VISA VCF File Generator — All Transaction Types
Produces pipe-delimited VCF files covering:
  Cards, Purchases, Cash, Hotels, Car Rentals, Airlines, Refunds,
  Chargebacks, Transfers, Recurring subscriptions
"""

import os
import sys
import random
import string
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple

BASE = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, BASE)

from data.vcf.vcf_models import (
    VCFCategory, VCFTransaction, TransactionStatus,
    CardManagementRecord, HotelRecord, CarRentalRecord,
    AirlineRecord, RecurringRecord,
    TRANSACTION_CODES, MCC_BY_CATEGORY, RESPONSE_CODES,
    POS_ENTRY_MODES, CURRENCIES, COUNTRIES,
)
from data.vcf.vcf_reference_data import (
    CARDHOLDER_NAMES, CARD_TYPES, CARD_TYPE_WEIGHTS,
    RETAIL_MERCHANTS, HOTELS, CAR_RENTAL_COMPANIES,
    VEHICLE_CLASSES, RENTAL_LOCATIONS, AIRLINES,
    AIRPORTS, CABIN_CLASSES, CABIN_WEIGHTS, FARE_BASIS_CODES,
    ACQUIRER_BINS, CARD_TYPES,
    gen_pan, gen_auth_code, gen_merchant_id, gen_terminal_id,
    gen_arn, gen_retrieval_ref, gen_stan, gen_folio, gen_agreement,
    gen_ticket_number, gen_pnr, gen_flight_number, gen_seat,
    gen_loyalty, gen_subscription_id, gen_confirmation,
    future_date, past_date, random_datetime, random_currency, random_country,
    weighted_choice, processing_code_for, pos_entry_for,
)

log = logging.getLogger(__name__)

# Category mix weights for realistic VCF file generation
DEFAULT_CATEGORY_MIX = {
    VCFCategory.PURCHASE:    40,
    VCFCategory.HOTEL:       15,
    VCFCategory.CAR_RENTAL:  12,
    VCFCategory.AIRLINE:     12,
    VCFCategory.CASH:         6,
    VCFCategory.RECURRING:    6,
    VCFCategory.REFUND:       4,
    VCFCategory.TRANSFER:     3,
    VCFCategory.CHARGEBACK:   1,
    VCFCategory.CARD_MGMT:    1,
}


class VCFGenerator:
    """
    Generates complete VISA VCF pipe-delimited files.
    Can produce single-category or mixed-category files.
    """

    def __init__(self):
        self._seq = 0

    # ── Public API ─────────────────────────────────────────────────────────────

    def generate_file(
        self,
        num_transactions: int = 20,
        categories: Optional[List[VCFCategory]] = None,
        category_mix: Optional[Dict[VCFCategory, int]] = None,
        file_date: Optional[str] = None,
        acquirer_bin: Optional[str] = None,
        force_approved: bool = False,
    ) -> str:
        """Generate a complete VCF file with header, transactions, and trailer."""
        file_date    = file_date    or datetime.now().strftime("%Y%m%d")
        acquirer_bin = acquirer_bin or random.choice(ACQUIRER_BINS)

        # Determine category distribution
        if categories:
            mix = {c: 1 for c in categories}
        else:
            mix = category_mix or DEFAULT_CATEGORY_MIX

        records = self._generate_transactions(num_transactions, mix, force_approved)

        lines = [self._header(file_date, acquirer_bin)]
        total_amount = 0.0
        for rec in records:
            lines.append(rec.to_pipe_record())
            total_amount += rec.amount
        lines.append(self._trailer(len(records), total_amount, file_date))

        return "\n".join(lines)

    def generate_dataset(self, n: int = 100) -> List[str]:
        """Generate n VCF files for SLM training."""
        files = []
        for _ in range(n):
            n_txns = random.randint(5, 100)
            files.append(self.generate_file(num_transactions=n_txns))
        return files

    def generate_by_category(self, category: VCFCategory, n: int = 20) -> str:
        """Generate a VCF file containing only one transaction category."""
        return self.generate_file(num_transactions=n, categories=[category])

    # ── Transaction builders ───────────────────────────────────────────────────

    def _generate_transactions(
        self,
        n: int,
        mix: Dict[VCFCategory, int],
        force_approved: bool,
    ) -> List[VCFTransaction]:
        cats  = list(mix.keys())
        wts   = [mix[c] for c in cats]
        recs  = []
        # Pick a shared card for ~30% of transactions (same cardholder)
        shared_pan  = gen_pan()
        shared_name = random.choice(CARDHOLDER_NAMES)
        shared_type = weighted_choice(CARD_TYPES, CARD_TYPE_WEIGHTS)

        for _ in range(n):
            cat = random.choices(cats, weights=wts, k=1)[0]
            use_shared = random.random() < 0.3
            pan  = shared_pan  if use_shared else gen_pan()
            name = shared_name if use_shared else random.choice(CARDHOLDER_NAMES)
            ctype= shared_type if use_shared else weighted_choice(CARD_TYPES, CARD_TYPE_WEIGHTS)

            builder = {
                VCFCategory.PURCHASE:    self._purchase,
                VCFCategory.CASH:        self._cash,
                VCFCategory.HOTEL:       self._hotel,
                VCFCategory.CAR_RENTAL:  self._car_rental,
                VCFCategory.AIRLINE:     self._airline,
                VCFCategory.REFUND:      self._refund,
                VCFCategory.CHARGEBACK:  self._chargeback,
                VCFCategory.TRANSFER:    self._transfer,
                VCFCategory.RECURRING:   self._recurring,
                VCFCategory.CARD_MGMT:   self._card_mgmt,
            }.get(cat, self._purchase)

            rec = builder(pan, name, ctype, force_approved)
            recs.append(rec)

        return recs

    # ── Base fields helper ─────────────────────────────────────────────────────

    def _base(
        self,
        category: VCFCategory,
        tc: str,
        pan: str,
        name: str,
        ctype: str,
        amount: float,
        mcc_entry: Tuple[str, str],
        merchant_name: str,
        merchant_city: str,
        merchant_state: str,
        merchant_country: str,
        force_approved: bool = False,
    ) -> VCFTransaction:
        self._seq += 1
        resp = "00" if (force_approved or random.random() < 0.92) else \
               random.choice(["05", "51", "54", "12", "61", "65"])
        status = TransactionStatus.APPROVED if resp == "00" else TransactionStatus.DECLINED
        currency = random_currency()

        return VCFTransaction(
            transaction_id   = f"VCF{self._seq:010d}",
            category         = category,
            transaction_code = tc,
            pan              = pan,
            processing_code  = processing_code_for(category),
            amount           = round(amount, 2),
            currency_code    = currency,
            transaction_dt   = random_datetime(30),
            mcc              = mcc_entry[0],
            mcc_description  = mcc_entry[1],
            pos_entry_mode   = pos_entry_for(category),
            response_code    = resp,
            auth_code        = gen_auth_code() if resp == "00" else "000000",
            merchant_id      = gen_merchant_id(),
            terminal_id      = gen_terminal_id(),
            merchant_name    = merchant_name[:25],
            merchant_city    = merchant_city[:13],
            merchant_state   = merchant_state,
            merchant_country = merchant_country,
            acquirer_bin     = random.choice(ACQUIRER_BINS),
            arn              = gen_arn(),
            retrieval_ref    = gen_retrieval_ref(),
            interchange_fee  = round(amount * random.uniform(0.015, 0.028), 2),
            stan             = gen_stan(),
            cardholder_name  = name,
            card_type        = ctype,
            status           = status,
        )

    # ── Category builders ──────────────────────────────────────────────────────

    def _purchase(self, pan, name, ctype, fa=False) -> VCFTransaction:
        merchant = random.choice(RETAIL_MERCHANTS)
        mcc_pool = MCC_BY_CATEGORY[VCFCategory.PURCHASE]
        mcc      = random.choice(mcc_pool)
        tc       = random.choice(list(TRANSACTION_CODES[VCFCategory.PURCHASE].keys()))
        amount   = round(random.uniform(1.5, 2500.0), 2)
        return self._base(
            VCFCategory.PURCHASE, tc, pan, name, ctype, amount, mcc,
            merchant[0], merchant[1], merchant[2], merchant[3], fa,
        )

    def _cash(self, pan, name, ctype, fa=False) -> VCFTransaction:
        mcc   = random.choice(MCC_BY_CATEGORY[VCFCategory.CASH])
        tc    = random.choice(list(TRANSACTION_CODES[VCFCategory.CASH].keys()))
        amt   = round(random.choice([20,40,50,60,80,100,200,300,500]) * random.uniform(0.9,1.1), 2)
        loc   = random.choice(RENTAL_LOCATIONS)
        return self._base(
            VCFCategory.CASH, tc, pan, name, ctype, amt, mcc,
            f"ATM {loc[0]} BRANCH", loc[1], "", random_country(), fa,
        )

    def _hotel(self, pan, name, ctype, fa=False) -> VCFTransaction:
        hotel  = random.choice(HOTELS)
        nights = random.randint(1, 14)
        rate   = round(random.uniform(*hotel[5]), 2)
        total  = round(rate * nights * random.uniform(1.05, 1.18), 2)  # incl taxes
        mcc    = random.choice(MCC_BY_CATEGORY[VCFCategory.HOTEL])
        tc     = random.choice(list(TRANSACTION_CODES[VCFCategory.HOTEL].keys()))

        base = self._base(
            VCFCategory.HOTEL, tc, pan, name, ctype, total, mcc,
            hotel[0], hotel[2], hotel[3], hotel[4], fa,
        )

        check_in  = past_date(1, 30)
        check_out = (datetime.strptime(check_in, "%Y%m%d") + timedelta(days=nights)).strftime("%Y%m%d")
        extra     = round(random.uniform(0, total * 0.15), 2)
        tax       = round(total * random.uniform(0.08, 0.18), 2)

        return HotelRecord(
            **base.__dict__,
            hotel_name      = hotel[0],
            hotel_code      = hotel[1],
            check_in_date   = check_in,
            check_out_date  = check_out,
            nights          = nights,
            room_type       = random.choice(["STD","DLX","STE","PNT","DLX","STD"]),
            room_rate       = rate,
            folio_number    = gen_folio(),
            guest_name      = name,
            no_show         = random.random() < 0.04,
            extra_charges   = extra,
            tax_amount      = tax,
            property_phone  = f"+1{random.randint(2000000000,9999999999)}",
            confirmation_no = gen_confirmation(),
            loyalty_number  = gen_loyalty() if random.random() < 0.6 else "",
        )

    def _car_rental(self, pan, name, ctype, fa=False) -> VCFTransaction:
        company   = random.choice(CAR_RENTAL_COMPANIES)
        v_class, rate_range = random.choice(VEHICLE_CLASSES)
        days      = random.randint(1, 21)
        daily     = round(random.uniform(*rate_range), 2)
        damage    = round(random.uniform(0, 400), 2) if random.random() < 0.06 else 0.0
        fuel      = round(random.uniform(20, 80), 2) if random.random() < 0.15 else 0.0
        ins_type  = random.choice(["CDW","LDW","PAI","NONE","CDW","CDW"])
        ins_amt   = round(daily * days * random.uniform(0.12, 0.22), 2) if ins_type != "NONE" else 0.0
        total     = round(daily * days + damage + fuel + ins_amt, 2)
        pickup    = random.choice(RENTAL_LOCATIONS)
        ret_loc   = random.choice(RENTAL_LOCATIONS)
        mcc       = random.choice(MCC_BY_CATEGORY[VCFCategory.CAR_RENTAL])
        tc        = random.choice(list(TRANSACTION_CODES[VCFCategory.CAR_RENTAL].keys()))
        pick_date = past_date(days + 1, days + 5)
        ret_date  = (datetime.strptime(pick_date, "%Y%m%d") + timedelta(days=days)).strftime("%Y%m%d")

        base = self._base(
            VCFCategory.CAR_RENTAL, tc, pan, name, ctype, total, mcc,
            company[:25], pickup[1], "", random_country(), fa,
        )

        return CarRentalRecord(
            **base.__dict__,
            rental_company   = company,
            location_code    = pickup[0],
            pickup_date      = pick_date,
            return_date      = ret_date,
            rental_days      = days,
            vehicle_class    = v_class,
            daily_rate       = daily,
            agreement_number = gen_agreement(),
            renter_name      = name,
            pickup_location  = f"{pickup[0]} - {pickup[1]}",
            return_location  = f"{ret_loc[0]} - {ret_loc[1]}",
            mileage_allowed  = random.choice([0, 150, 200, 250, 300]),
            mileage_used     = random.randint(50, 3000),
            fuel_policy      = random.choice(["FULL_TO_FULL","FULL_TO_FULL","PREPAID","INCLUDED"]),
            damage_amount    = damage,
            fuel_charge      = fuel,
            insurance_type   = ins_type,
            insurance_amount = ins_amt,
            loyalty_number   = gen_loyalty() if random.random() < 0.5 else "",
        )

    def _airline(self, pan, name, ctype, fa=False) -> VCFTransaction:
        airline_code, airline_name = random.choice(AIRLINES)
        cabin    = weighted_choice(CABIN_CLASSES, CABIN_WEIGHTS)
        origin   = random.choice(AIRPORTS)
        dest     = random.choice([a for a in AIRPORTS if a != origin])
        base_fares = {
            "ECONOMY":     (150, 900),
            "PREMIUM_ECO": (400, 1800),
            "BUSINESS":    (1200, 8000),
            "FIRST":       (3000, 18000),
        }
        base_fare = round(random.uniform(*base_fares[cabin]), 2)
        taxes     = round(base_fare * random.uniform(0.12, 0.28), 2)
        baggage   = round(random.uniform(0, 80), 2) if cabin == "ECONOMY" else 0.0
        ancillary = round(random.uniform(0, 120), 2)
        total     = round(base_fare + taxes + baggage + ancillary, 2)
        mcc       = random.choice(MCC_BY_CATEGORY[VCFCategory.AIRLINE])
        tc        = random.choice(list(TRANSACTION_CODES[VCFCategory.AIRLINE].keys()))

        dep_dt = (datetime.now() + timedelta(days=random.randint(1, 120))).strftime("%Y%m%d%H%M")
        arr_dt = (datetime.strptime(dep_dt, "%Y%m%d%H%M") + timedelta(hours=random.randint(1,16))).strftime("%Y%m%d%H%M")

        base = self._base(
            VCFCategory.AIRLINE, tc, pan, name, ctype, total, mcc,
            airline_name[:25], origin, "", random_country(), fa,
        )

        return AirlineRecord(
            **base.__dict__,
            airline_code      = airline_code,
            airline_name      = airline_name,
            ticket_number     = gen_ticket_number(),
            passenger_name    = name,
            origin            = origin,
            destination       = dest,
            departure_dt      = dep_dt,
            arrival_dt        = arr_dt,
            flight_number     = gen_flight_number(airline_code),
            cabin_class       = cabin,
            fare_basis        = random.choice(FARE_BASIS_CODES),
            ticket_type       = random.choice(["ONE_WAY","ROUND_TRIP","ROUND_TRIP","MULTI_CITY"]),
            base_fare         = base_fare,
            taxes_fees        = taxes,
            ancillary_fee     = ancillary,
            baggage_fee       = baggage,
            seat_number       = gen_seat(),
            frequent_flyer_no = gen_loyalty() if random.random() < 0.55 else "",
            pnr               = gen_pnr(),
            stops             = random.choice([0, 0, 0, 1, 1, 2]),
        )

    def _refund(self, pan, name, ctype, fa=False) -> VCFTransaction:
        merchant = random.choice(RETAIL_MERCHANTS)
        mcc      = random.choice(MCC_BY_CATEGORY[VCFCategory.REFUND])
        tc       = random.choice(list(TRANSACTION_CODES[VCFCategory.REFUND].keys()))
        amount   = round(random.uniform(5.0, 800.0), 2)
        base     = self._base(
            VCFCategory.REFUND, tc, pan, name, ctype, amount, mcc,
            merchant[0], merchant[1], merchant[2], merchant[3], True,
        )
        base.response_code = "00"
        base.status        = TransactionStatus.APPROVED
        return base

    def _chargeback(self, pan, name, ctype, fa=False) -> VCFTransaction:
        merchant = random.choice(RETAIL_MERCHANTS)
        mcc      = random.choice(MCC_BY_CATEGORY[VCFCategory.CHARGEBACK])
        tc       = random.choice(list(TRANSACTION_CODES[VCFCategory.CHARGEBACK].keys()))
        amount   = round(random.uniform(10.0, 1200.0), 2)
        return self._base(
            VCFCategory.CHARGEBACK, tc, pan, name, ctype, amount, mcc,
            merchant[0], merchant[1], merchant[2], merchant[3], True,
        )

    def _transfer(self, pan, name, ctype, fa=False) -> VCFTransaction:
        mcc    = random.choice(MCC_BY_CATEGORY[VCFCategory.TRANSFER])
        tc     = random.choice(list(TRANSACTION_CODES[VCFCategory.TRANSFER].keys()))
        amount = round(random.uniform(10.0, 5000.0), 2)
        loc    = random.choice(RENTAL_LOCATIONS)
        return self._base(
            VCFCategory.TRANSFER, tc, pan, name, ctype, amount, mcc,
            "VISA DIRECT TRANSFER", loc[1], "", random_country(), fa,
        )

    def _recurring(self, pan, name, ctype, fa=False) -> VCFTransaction:
        services = [
            ("NETFLIX", "4899", "STREAMING"),     ("SPOTIFY", "4899", "STREAMING"),
            ("AMAZON PRIME", "5968", "SHOPPING"),  ("APPLE ONE", "7372", "TECH"),
            ("GOOGLE WORKSPACE", "7372", "TECH"),   ("MICROSOFT 365", "7372", "TECH"),
            ("GYM MEMBERSHIP", "7941", "FITNESS"),  ("NYT DIGITAL", "5968", "NEWS"),
            ("HULU PLUS", "4899", "STREAMING"),     ("DISNEY PLUS", "4899", "STREAMING"),
            ("ADOBE CC", "7372", "TECH"),           ("DROPBOX PRO", "7372", "TECH"),
        ]
        svc, scc, _ = random.choice(services)
        mcc_entry    = (scc, svc)
        tc           = random.choice(list(TRANSACTION_CODES[VCFCategory.RECURRING].keys()))
        cycle        = random.choice(["MONTHLY","MONTHLY","MONTHLY","ANNUAL","QUARTERLY"])
        amounts      = {"MONTHLY": (5,25), "ANNUAL": (50,200), "QUARTERLY": (15,75), "WEEKLY": (5,15)}
        amount       = round(random.uniform(*amounts[cycle]), 2)
        svc_start    = past_date(30, 365)
        svc_end      = future_date(1, 365)

        base = self._base(
            VCFCategory.RECURRING, tc, pan, name, ctype, amount, mcc_entry,
            svc[:25], "ONLINE", "", "840", fa,
        )

        return RecurringRecord(
            **base.__dict__,
            subscription_id   = gen_subscription_id(),
            billing_cycle     = cycle,
            cycle_number      = random.randint(1, 36),
            total_cycles      = random.choice([0, 12, 24, 36]),
            service_name      = svc,
            service_start     = svc_start,
            service_end       = svc_end,
            instalment_plan   = "",
            is_trial          = random.random() < 0.05,
            next_billing_date = future_date(1, 35),
        )

    def _card_mgmt(self, pan, name, ctype, fa=False) -> VCFTransaction:
        actions = ["ACTIVATE","BLOCK","REPLACE","ISSUE","LIMIT_CHANGE","PIN_CHANGE"]
        action  = random.choice(actions)
        reasons = {"BLOCK": ["LOST","STOLEN","FRAUD","REQUEST"]}
        tc_map  = {
            "ACTIVATE":     "01", "BLOCK":       "02", "REPLACE":     "03",
            "LIMIT_CHANGE": "04", "PIN_CHANGE":  "05", "ISSUE":       "06",
        }
        mcc     = ("6012", "Financial Institutions")
        tc      = tc_map.get(action, "01")
        amount  = 0.0

        base = self._base(
            VCFCategory.CARD_MGMT, tc, pan, name, ctype, amount, mcc,
            "CARD CENTER", "ISSUER BANK", "", "840", True,
        )

        limit     = round(random.uniform(1000, 50000), 2)
        new_limit = round(limit * random.uniform(1.1, 2.0), 2) if action == "LIMIT_CHANGE" else limit

        return CardManagementRecord(
            **base.__dict__,
            card_action      = action,
            old_pan          = gen_pan() if action == "REPLACE" else "",
            new_expiry       = f"{random.randint(1,12):02d}{random.randint(26,30)}",
            block_reason     = random.choice(reasons.get(action, [""])),
            credit_limit     = limit,
            new_credit_limit = new_limit,
            card_delivery    = random.choice(["BRANCH","MAIL","DIGITAL"]),
        )

    # ── File structure ─────────────────────────────────────────────────────────

    def _header(self, file_date: str, acquirer_bin: str) -> str:
        proc_time = datetime.now().strftime("%H%M%S")
        bank      = random.choice(["JPMORGAN CHASE", "WELLS FARGO", "BANK OF AMERICA",
                                   "CITIBANK NA", "US BANK"])
        return "|".join([
            "VCF", "2.1", acquirer_bin, file_date, proc_time,
            bank, "PRODUCTION", f"{random.randint(1,999):03d}",
            "VISA", "ALL",
        ])

    def _trailer(self, count: int, total: float, date: str) -> str:
        hash_total = str(int(total * 100))[-16:].zfill(16)
        return "|".join([
            "TRAILER", str(count), f"{total:.2f}",
            hash_total, date, "00", "VISA", "END",
        ])
