"""
VISA VCF Domain Models
Complete dataclass definitions for every VCF transaction type.

Supported categories:
  CARD_MGMT     Card issuance, activation, block, replacement, limit change
  PURCHASE      Retail, e-commerce, contactless, chip purchases
  CASH          ATM withdrawal, cash advance, balance inquiry
  HOTEL         Check-in/out, no-show, extra charges, deposit hold
  CAR_RENTAL    Rental open/close, damage charge, fuel surcharge
  AIRLINE       Ticket purchase, upgrade, refund, ancillary fees
  REFUND        Merchandise return, service credit, partial refund
  CHARGEBACK    First/second chargeback, reversal, re-presentment
  TRANSFER      P2P, wallet load/unload, currency conversion
  RECURRING     Subscription billing, instalment payment
"""

import random
import string
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any
from enum import Enum


# ─── Enumerations ─────────────────────────────────────────────────────────────

class VCFCategory(Enum):
    CARD_MGMT   = "CARD_MGMT"
    PURCHASE    = "PURCHASE"
    CASH        = "CASH"
    HOTEL       = "HOTEL"
    CAR_RENTAL  = "CAR_RENTAL"
    AIRLINE     = "AIRLINE"
    REFUND      = "REFUND"
    CHARGEBACK  = "CHARGEBACK"
    TRANSFER    = "TRANSFER"
    RECURRING   = "RECURRING"


class CardStatus(Enum):
    ACTIVE    = "ACTIVE"
    BLOCKED   = "BLOCKED"
    EXPIRED   = "EXPIRED"
    REPLACED  = "REPLACED"
    PENDING   = "PENDING"


class TransactionStatus(Enum):
    APPROVED  = "APPROVED"
    DECLINED  = "DECLINED"
    PENDING   = "PENDING"
    REVERSED  = "REVERSED"
    SETTLED   = "SETTLED"


# ─── VCF Transaction Codes by Category ────────────────────────────────────────

TRANSACTION_CODES = {
    VCFCategory.CARD_MGMT:  {
        "01": "Card Activation",
        "02": "Card Block",
        "03": "Card Replacement",
        "04": "Limit Change",
        "05": "PIN Change",
        "06": "Card Issuance",
        "07": "Card Closure",
        "08": "Supplementary Card Add",
    },
    VCFCategory.PURCHASE:   {
        "06": "Financial Transaction",
        "60": "Mail/Phone Order",
        "61": "E-Commerce Purchase",
        "62": "Contactless Purchase",
        "63": "Chip Purchase",
        "64": "Recurring Purchase",
        "65": "Instalment Purchase",
    },
    VCFCategory.CASH:       {
        "10": "ATM Cash Withdrawal",
        "11": "Cash Advance",
        "12": "Balance Inquiry",
        "13": "Mini Statement",
        "14": "Cash Deposit",
    },
    VCFCategory.HOTEL:      {
        "20": "Hotel Check-In Auth",
        "21": "Hotel Check-Out Settlement",
        "22": "Hotel No-Show Charge",
        "23": "Hotel Extra Charges",
        "24": "Hotel Deposit Hold",
        "25": "Hotel Deposit Release",
        "26": "Hotel Room Upgrade",
        "27": "Hotel Early Departure",
    },
    VCFCategory.CAR_RENTAL: {
        "30": "Car Rental Open Auth",
        "31": "Car Rental Close Settlement",
        "32": "Car Rental Damage Charge",
        "33": "Car Rental Fuel Surcharge",
        "34": "Car Rental Mileage Overage",
        "35": "Car Rental Extension",
        "36": "Car Rental Cancellation",
        "37": "Car Rental Insurance Charge",
    },
    VCFCategory.AIRLINE:    {
        "40": "Airline Ticket Purchase",
        "41": "Airline Upgrade",
        "42": "Airline Baggage Fee",
        "43": "Airline Seat Selection",
        "44": "Airline Refund",
        "45": "Airline Change Fee",
        "46": "Airline Cancellation",
        "47": "Lounge Access",
    },
    VCFCategory.REFUND:     {
        "50": "Merchandise Return",
        "51": "Service Credit",
        "52": "Partial Refund",
        "53": "Chargeback Credit",
        "54": "Goodwill Credit",
    },
    VCFCategory.CHARGEBACK: {
        "25": "First Chargeback",
        "26": "Second Chargeback",
        "28": "Chargeback Reversal",
        "29": "Re-Presentment",
    },
    VCFCategory.TRANSFER:   {
        "70": "P2P Transfer",
        "71": "Wallet Load",
        "72": "Wallet Unload",
        "73": "Currency Conversion",
        "74": "Cross-Border Transfer",
        "75": "Interbank Transfer",
    },
    VCFCategory.RECURRING:  {
        "64": "Subscription Billing",
        "65": "Instalment Payment",
        "66": "Utility Payment",
        "67": "Insurance Premium",
        "68": "Loan EMI",
    },
}

# MCC codes by category
MCC_BY_CATEGORY = {
    VCFCategory.PURCHASE:   [
        ("5411", "Grocery Stores"),      ("5912", "Drug Stores"),
        ("5732", "Electronics Stores"),  ("5812", "Restaurants"),
        ("5541", "Gas Stations"),        ("5311", "Department Stores"),
        ("5999", "Miscellaneous Retail"),("5651", "Clothing Stores"),
        ("5944", "Jewelry Stores"),      ("5945", "Toy/Hobby Shops"),
        ("7372", "Computer Software"),   ("5969", "Direct Marketing"),
    ],
    VCFCategory.CASH:       [
        ("6011", "Automated Cash Disbursements"),
        ("6010", "Manual Cash Disbursements"),
        ("6012", "Financial Institutions"),
    ],
    VCFCategory.HOTEL:      [
        ("7011", "Hotels/Motels"),   ("7012", "Timeshares"),
        ("7013", "Vacation Rentals"),("7014", "Serviced Apartments"),
        ("7015", "Bed & Breakfast"), ("3501", "Holiday Inn"),
        ("3502", "Hilton Hotels"),   ("3503", "Marriott Hotels"),
        ("3504", "Hyatt Hotels"),    ("3505", "Sheraton Hotels"),
        ("3506", "InterContinental"),("3507", "Westin Hotels"),
    ],
    VCFCategory.CAR_RENTAL: [
        ("7512", "Car Rental Agencies"),  ("7513", "Truck Rental"),
        ("3351", "Hertz Car Rental"),     ("3352", "Avis Car Rental"),
        ("3353", "Budget Car Rental"),    ("3354", "Enterprise Rent-A-Car"),
        ("3355", "National Car Rental"),  ("3356", "Dollar Rent A Car"),
        ("3357", "Thrifty Car Rental"),   ("3358", "Alamo Rent A Car"),
    ],
    VCFCategory.AIRLINE:    [
        ("4511", "Airlines"),             ("3000", "United Airlines"),
        ("3001", "American Airlines"),    ("3002", "Delta Air Lines"),
        ("3003", "Southwest Airlines"),   ("3004", "British Airways"),
        ("3005", "Air France"),           ("3006", "Lufthansa"),
        ("3007", "Emirates"),             ("3008", "Singapore Airlines"),
        ("4582", "Airports"),             ("7011", "Airport Hotels"),
    ],
    VCFCategory.TRANSFER:   [
        ("6540", "Non-Financial Institutions"),
        ("6010", "Financial Institutions"),
        ("6099", "Financial Services"),
        ("4829", "Money Transfer"),
    ],
    VCFCategory.RECURRING:  [
        ("4899", "Cable/Satellite TV"),  ("4814", "Telephone Services"),
        ("4816", "Computer Networks"),   ("5968", "Direct Marketing Sub"),
        ("6300", "Insurance Services"),  ("8011", "Doctors/Physicians"),
        ("8099", "Health Services"),     ("7372", "Computer Software"),
        ("7929", "Bands/Orchestras"),    ("7941", "Sports Clubs"),
    ],
    VCFCategory.CARD_MGMT:  [("6012", "Financial Institutions")],
    VCFCategory.REFUND:     [("5999", "Miscellaneous Retail")],
    VCFCategory.CHARGEBACK: [("5999", "Miscellaneous Retail")],
}

RESPONSE_CODES = {
    "00": "Approved",               "01": "Refer to Issuer",
    "03": "Invalid Merchant",       "04": "Pick Up Card",
    "05": "Do Not Honor",           "06": "General Error",
    "08": "Honor With ID",          "10": "Partial Approval",
    "12": "Invalid Transaction",    "13": "Invalid Amount",
    "14": "Invalid Card Number",    "15": "No Such Issuer",
    "19": "Re-Enter Transaction",   "25": "Unable to Locate Record",
    "30": "Format Error",           "41": "Lost Card",
    "43": "Stolen Card",            "51": "Insufficient Funds",
    "54": "Expired Card",           "55": "Incorrect PIN",
    "57": "Txn Not Permitted",      "58": "Txn Not Permitted Terminal",
    "61": "Exceeds Limit",          "62": "Restricted Card",
    "65": "Activity Limit Exceeded","75": "PIN Attempts Exceeded",
    "76": "Ineligible Account",     "78": "Blocked",
    "91": "Issuer Unavailable",     "92": "Unable to Route",
    "96": "System Malfunction",     "N7": "CVV2 Mismatch",
}

POS_ENTRY_MODES = {
    "00": "Unknown",               "01": "Manual",
    "02": "Magnetic Stripe",       "05": "Chip",
    "07": "Contactless Chip",      "10": "Credential on File",
    "71": "Contactless Magnetic",  "79": "Chip Fallback",
    "81": "E-Commerce",            "90": "Magnetic Stripe Auto",
    "91": "Contactless Auto",      "95": "Integrated Circuit",
}

CURRENCIES = {
    "840": "USD", "978": "EUR", "826": "GBP", "392": "JPY",
    "124": "CAD", "036": "AUD", "756": "CHF", "356": "INR",
    "986": "BRL", "156": "CNY", "484": "MXN", "344": "HKD",
    "702": "SGD", "208": "DKK", "752": "SEK", "578": "NOK",
    "554": "NZD", "458": "MYR", "764": "THB", "410": "KRW",
    "682": "SAR", "784": "AED", "566": "NGN", "710": "ZAR",
}

COUNTRIES = {
    "840": "USA",    "826": "GBR",    "978": "DEU",    "124": "CAN",
    "036": "AUS",    "392": "JPN",    "356": "IND",    "702": "SGP",
    "344": "HKG",    "484": "MEX",    "276": "DEU",    "250": "FRA",
    "380": "ITA",    "724": "ESP",    "528": "NLD",    "756": "CHE",
    "784": "ARE",    "682": "SAU",    "410": "KOR",    "156": "CHN",
}

# ─── Base transaction dataclass ────────────────────────────────────────────────

@dataclass
class VCFTransaction:
    """Base fields present in every VCF record."""
    transaction_id:    str
    category:          VCFCategory
    transaction_code:  str
    pan:               str           # 13-19 digit card number
    processing_code:   str           # 6-digit ISO processing code
    amount:            float         # transaction amount
    currency_code:     str           # ISO 4217 3-digit
    transaction_dt:    str           # YYYYMMDDHHmmSS
    mcc:               str           # 4-digit merchant category code
    mcc_description:   str
    pos_entry_mode:    str
    response_code:     str
    auth_code:         str           # 6 alphanumeric
    merchant_id:       str           # 15 alphanumeric
    terminal_id:       str           # 8 alphanumeric
    merchant_name:     str
    merchant_city:     str
    merchant_state:    str
    merchant_country:  str
    acquirer_bin:      str           # 6-digit acquirer BIN
    arn:               str           # 23-digit Acquirer Reference Number
    retrieval_ref:     str           # 12-digit retrieval reference
    interchange_fee:   float
    stan:              str           # 6-digit System Trace Audit Number
    cardholder_name:   str
    card_type:         str           # VISA_CLASSIC / VISA_GOLD / VISA_PLATINUM / VISA_INFINITE
    status:            TransactionStatus
    network_id:        str = "VISA"

    def to_pipe_record(self) -> str:
        """Produce a pipe-delimited VCF line."""
        return "|".join([
            self.transaction_code,
            self.pan,
            self.processing_code,
            f"{self.amount:.2f}",
            self.currency_code,
            self.transaction_dt,
            self.mcc,
            self.pos_entry_mode,
            self.response_code,
            self.auth_code,
            self.merchant_id.ljust(15)[:15],
            self.terminal_id.ljust(8)[:8],
            self.merchant_name[:25],
            self.merchant_city[:13],
            self.arn,
            self.retrieval_ref,
            f"{self.interchange_fee:.2f}",
            self.stan,
            self.merchant_country,
            "00",                        # settlement indicator
            "0",                         # partial approval indicator
            self.cardholder_name[:26],
            self.card_type,
            self.category.value,
            self.transaction_id,
        ])


# ─── Category-specific extension dataclasses ──────────────────────────────────

@dataclass
class CardManagementRecord(VCFTransaction):
    card_action:       str = ""      # ACTIVATE / BLOCK / REPLACE / ISSUE
    old_pan:           str = ""      # for replacements
    new_expiry:        str = ""      # MMYY
    block_reason:      str = ""      # LOST / STOLEN / FRAUD / REQUEST
    credit_limit:      float = 0.0
    new_credit_limit:  float = 0.0
    card_delivery:     str = ""      # BRANCH / MAIL / DIGITAL

    def to_pipe_record(self) -> str:
        base = super().to_pipe_record()
        return base + "|" + "|".join([
            self.card_action,
            self.old_pan,
            self.new_expiry,
            self.block_reason,
            f"{self.credit_limit:.2f}",
            f"{self.new_credit_limit:.2f}",
            self.card_delivery,
        ])


@dataclass
class HotelRecord(VCFTransaction):
    hotel_name:        str = ""
    hotel_code:        str = ""      # 8-char property code
    check_in_date:     str = ""      # YYYYMMDD
    check_out_date:    str = ""      # YYYYMMDD
    nights:            int = 1
    room_type:         str = ""      # STD / DLX / STE / PNT
    room_rate:         float = 0.0   # nightly rate
    folio_number:      str = ""
    guest_name:        str = ""
    no_show:           bool = False
    extra_charges:     float = 0.0
    tax_amount:        float = 0.0
    property_phone:    str = ""
    confirmation_no:   str = ""
    loyalty_number:    str = ""

    def to_pipe_record(self) -> str:
        base = super().to_pipe_record()
        return base + "|" + "|".join([
            self.hotel_name[:30],
            self.hotel_code,
            self.check_in_date,
            self.check_out_date,
            str(self.nights),
            self.room_type,
            f"{self.room_rate:.2f}",
            self.folio_number,
            self.guest_name[:26],
            "Y" if self.no_show else "N",
            f"{self.extra_charges:.2f}",
            f"{self.tax_amount:.2f}",
            self.confirmation_no,
            self.loyalty_number,
        ])


@dataclass
class CarRentalRecord(VCFTransaction):
    rental_company:    str = ""
    location_code:     str = ""      # IATA airport or city code
    pickup_date:       str = ""      # YYYYMMDD
    return_date:       str = ""      # YYYYMMDD
    rental_days:       int = 1
    vehicle_class:     str = ""      # ECONOMY / COMPACT / MIDSIZE / FULLSIZE / SUV / LUXURY / VAN
    daily_rate:        float = 0.0
    agreement_number:  str = ""
    renter_name:       str = ""
    pickup_location:   str = ""
    return_location:   str = ""
    mileage_allowed:   int = 0       # 0 = unlimited
    mileage_used:      int = 0
    fuel_policy:       str = ""      # FULL_TO_FULL / PREPAID / INCLUDED
    damage_amount:     float = 0.0
    fuel_charge:       float = 0.0
    insurance_type:    str = ""      # CDW / LDW / PAI / NONE
    insurance_amount:  float = 0.0
    loyalty_number:    str = ""

    def to_pipe_record(self) -> str:
        base = super().to_pipe_record()
        return base + "|" + "|".join([
            self.rental_company[:20],
            self.location_code,
            self.pickup_date,
            self.return_date,
            str(self.rental_days),
            self.vehicle_class,
            f"{self.daily_rate:.2f}",
            self.agreement_number,
            self.renter_name[:26],
            self.pickup_location[:20],
            self.return_location[:20],
            str(self.mileage_allowed),
            str(self.mileage_used),
            self.fuel_policy,
            f"{self.damage_amount:.2f}",
            f"{self.fuel_charge:.2f}",
            self.insurance_type,
            f"{self.insurance_amount:.2f}",
            self.loyalty_number,
        ])


@dataclass
class AirlineRecord(VCFTransaction):
    airline_code:      str = ""      # IATA 2-char carrier code
    airline_name:      str = ""
    ticket_number:     str = ""      # 13-digit ticket number
    passenger_name:    str = ""
    origin:            str = ""      # IATA airport code
    destination:       str = ""      # IATA airport code
    departure_dt:      str = ""      # YYYYMMDDHHmm
    arrival_dt:        str = ""
    flight_number:     str = ""
    cabin_class:       str = ""      # ECONOMY / BUSINESS / FIRST / PREMIUM_ECO
    fare_basis:        str = ""
    ticket_type:       str = ""      # ONE_WAY / ROUND_TRIP / MULTI_CITY
    base_fare:         float = 0.0
    taxes_fees:        float = 0.0
    ancillary_fee:     float = 0.0
    baggage_fee:       float = 0.0
    seat_number:       str = ""
    frequent_flyer_no: str = ""
    pnr:               str = ""      # 6-char booking reference
    stops:             int = 0

    def to_pipe_record(self) -> str:
        base = super().to_pipe_record()
        return base + "|" + "|".join([
            self.airline_code,
            self.airline_name[:25],
            self.ticket_number,
            self.passenger_name[:26],
            self.origin,
            self.destination,
            self.departure_dt,
            self.arrival_dt,
            self.flight_number,
            self.cabin_class,
            self.fare_basis,
            self.ticket_type,
            f"{self.base_fare:.2f}",
            f"{self.taxes_fees:.2f}",
            f"{self.ancillary_fee:.2f}",
            f"{self.baggage_fee:.2f}",
            self.seat_number,
            self.pnr,
            str(self.stops),
        ])


@dataclass
class RecurringRecord(VCFTransaction):
    subscription_id:   str = ""
    billing_cycle:     str = ""      # MONTHLY / QUARTERLY / ANNUAL / WEEKLY
    cycle_number:      int = 1
    total_cycles:      int = 0       # 0 = indefinite
    service_name:      str = ""
    service_start:     str = ""      # YYYYMMDD
    service_end:       str = ""      # YYYYMMDD
    instalment_plan:   str = ""
    is_trial:          bool = False
    next_billing_date: str = ""

    def to_pipe_record(self) -> str:
        base = super().to_pipe_record()
        return base + "|" + "|".join([
            self.subscription_id,
            self.billing_cycle,
            str(self.cycle_number),
            str(self.total_cycles),
            self.service_name[:30],
            self.service_start,
            self.service_end,
            self.instalment_plan,
            "Y" if self.is_trial else "N",
            self.next_billing_date,
        ])
