"""
VCF Reference Data
Realistic lookup tables for merchants, hotels, car companies, airlines,
cardholder names, and card types used by the VCF generator.
"""

import random
import string
from datetime import datetime, timedelta

# ─── Cardholder Names ─────────────────────────────────────────────────────────
CARDHOLDER_NAMES = [
    "JOHN A SMITH", "JANE M DOE", "ROBERT J JOHNSON", "MARY L WILLIAMS",
    "DAVID K BROWN", "PATRICIA A JONES", "MICHAEL T DAVIS", "LINDA S MILLER",
    "WILLIAM R WILSON", "BARBARA E MOORE", "JAMES F TAYLOR", "ELIZABETH A ANDERSON",
    "RICHARD H THOMAS", "JENNIFER L JACKSON", "CHARLES W WHITE", "MARGARET D HARRIS",
    "JOSEPH M MARTIN", "DOROTHY R THOMPSON", "THOMAS B GARCIA", "JESSICA P MARTINEZ",
    "CHRISTOPHER J ROBINSON", "SARAH K CLARK", "DANIEL R RODRIGUEZ", "KAREN L LEWIS",
    "MATTHEW J LEE", "NANCY M WALKER", "ANTHONY C HALL", "BETTY J ALLEN",
    "MARK D YOUNG", "HELEN S HERNANDEZ", "RAJESH K SHARMA", "PRIYA V PATEL",
    "AHMED AL-RASHID", "FATIMA HASSAN", "WANG LEI", "LI MING",
    "JEAN-PIERRE DUBOIS", "MARIE CLAIRE MARTIN", "HANS MUELLER", "ANNA SCHMIDT",
    "CARLOS RODRIGUES", "SOFIA FERNANDEZ", "YUKI TANAKA", "KENJI YAMAMOTO",
    "AISHA OKONKWO", "CHIDI ADEYEMI", "ISABELLA ROSSI", "MARCO BIANCHI",
]

CARD_TYPES = [
    "VISA_CLASSIC", "VISA_GOLD", "VISA_PLATINUM",
    "VISA_INFINITE", "VISA_SIGNATURE", "VISA_BUSINESS",
    "VISA_CORPORATE", "VISA_PURCHASING",
]

CARD_TYPE_WEIGHTS = [30, 25, 20, 10, 8, 4, 2, 1]  # higher weight = more common

# ─── Retail Merchants ─────────────────────────────────────────────────────────
RETAIL_MERCHANTS = [
    ("AMAZON.COM", "SEATTLE", "WA", "840"),
    ("WALMART STORE 4521", "BENTONVILLE", "AR", "840"),
    ("TARGET STORE 0782", "MINNEAPOLIS", "MN", "840"),
    ("COSTCO WHSE 0123", "ISSAQUAH", "WA", "840"),
    ("HOME DEPOT 0456", "ATLANTA", "GA", "840"),
    ("BEST BUY 00789", "RICHFIELD", "MN", "840"),
    ("KROGER #5521", "CINCINNATI", "OH", "840"),
    ("WALGREENS #1234", "DEERFIELD", "IL", "840"),
    ("CVS PHARMACY 456", "WOONSOCKET", "RI", "840"),
    ("MCDONALDS #12345", "OAK BROOK", "IL", "840"),
    ("STARBUCKS #67890", "SEATTLE", "WA", "840"),
    ("WHOLE FOODS MKT", "AUSTIN", "TX", "840"),
    ("TRADER JOES 0099", "MONROVIA", "CA", "840"),
    ("APPLE STORE NYC", "NEW YORK", "NY", "840"),
    ("MACYS DEPT STORE", "NEW YORK", "NY", "840"),
    ("NORDSTROM #0088", "SEATTLE", "WA", "840"),
    ("SEPHORA 0145", "SAN FRANCISCO", "CA", "840"),
    ("ZARA USA 0034", "NEW YORK", "NY", "840"),
    ("IKEA ELIZABETH", "ELIZABETH", "NJ", "840"),
    ("CHEESECAKE FACTORY", "CALABASAS", "CA", "840"),
    ("SHELL OIL #2244", "HOUSTON", "TX", "840"),
    ("EXXON MOBIL 9901", "IRVING", "TX", "840"),
    ("NETFLIX.COM", "LOS GATOS", "CA", "840"),
    ("SPOTIFY AB", "STOCKHOLM", "SWE", "752"),
    ("GOOGLE *GSUITE", "MOUNTAIN VIEW", "CA", "840"),
    ("MICROSOFT STORE", "REDMOND", "WA", "840"),
    ("UBER TRIP", "SAN FRANCISCO", "CA", "840"),
    ("LYFT RIDE", "SAN FRANCISCO", "CA", "840"),
    ("DOORDASH INC", "SAN FRANCISCO", "CA", "840"),
    ("AIRBNB PAYMENT", "SAN FRANCISCO", "CA", "840"),
]

# ─── Hotels ───────────────────────────────────────────────────────────────────
HOTELS = [
    # (name, code, city, state, country, nightly_rate_range)
    ("MARRIOTT MARQUIS NYC",     "MRMQNYC", "NEW YORK",        "NY",  "840", (350, 850)),
    ("HILTON MIDTOWN",           "HILMIDNY", "NEW YORK",       "NY",  "840", (280, 680)),
    ("HYATT REGENCY CHICAGO",    "HYAREGCH", "CHICAGO",        "IL",  "840", (210, 520)),
    ("WESTIN BEVERLY HILLS",     "WESTBVH",  "BEVERLY HILLS",  "CA",  "840", (420, 980)),
    ("SHERATON GRAND LONDON",    "SHERGLN",  "LONDON",         "",    "826", (280, 650)),
    ("INTERCONTINENTAL PARIS",   "ICPARIS",  "PARIS",          "",    "978", (390, 950)),
    ("FOUR SEASONS DUBAI",       "FSDUBAI",  "DUBAI",          "",    "784", (450, 1200)),
    ("RITZ CARLTON TOKYO",       "RCTOKYO",  "TOKYO",          "",    "392", (480, 1100)),
    ("PENINSULA HONG KONG",      "PENHKG",   "HONG KONG",      "",    "344", (420, 980)),
    ("MANDARIN ORIENTAL SG",     "MOSIN",    "SINGAPORE",      "",    "702", (380, 850)),
    ("W HOTEL BARCELONA",        "WBCN",     "BARCELONA",      "",    "978", (280, 720)),
    ("GRAND HYATT BERLIN",       "GHYBER",   "BERLIN",         "",    "978", (220, 580)),
    ("SHANGRI-LA SYDNEY",        "SLASYD",   "SYDNEY",         "",    "036", (310, 780)),
    ("JW MARRIOTT MUMBAI",       "JWMMUM",   "MUMBAI",         "",    "356", (180, 450)),
    ("FAIRMONT BANFF SPRINGS",   "FMTBNF",   "BANFF",          "AB",  "124", (290, 750)),
    ("WALDORF ASTORIA NYC",      "WANYC",    "NEW YORK",       "NY",  "840", (580, 1400)),
    ("BELLAGIO LAS VEGAS",       "BELLV",    "LAS VEGAS",      "NV",  "840", (250, 980)),
    ("ATLANTIS PARADISE ISL",    "ATLATL",   "NASSAU",         "",    "044", (380, 1200)),
    ("BURJ AL ARAB",             "BURALR",   "DUBAI",          "",    "784", (950, 2800)),
    ("PARK HYATT SYDNEY",        "PHYSYD",   "SYDNEY",         "",    "036", (450, 1100)),
]

# ─── Car Rental Companies ─────────────────────────────────────────────────────
CAR_RENTAL_COMPANIES = [
    "HERTZ CORPORATION",
    "AVIS RENT A CAR",
    "BUDGET CAR RENTAL",
    "ENTERPRISE RENT-A-CAR",
    "NATIONAL CAR RENTAL",
    "DOLLAR RENT A CAR",
    "THRIFTY CAR RENTAL",
    "ALAMO RENT A CAR",
    "SIXT RENT A CAR",
    "EUROPCAR",
]

VEHICLE_CLASSES = [
    ("ECONOMY",  (35, 65)),
    ("COMPACT",  (45, 80)),
    ("MIDSIZE",  (55, 100)),
    ("FULLSIZE", (65, 120)),
    ("SUV",      (80, 180)),
    ("LUXURY",   (150, 400)),
    ("VAN",      (90, 200)),
    ("ELECTRIC", (70, 150)),
]

RENTAL_LOCATIONS = [
    ("JFK", "NEW YORK"),    ("LAX", "LOS ANGELES"), ("ORD", "CHICAGO"),
    ("DFW", "DALLAS"),      ("ATL", "ATLANTA"),     ("SFO", "SAN FRANCISCO"),
    ("MIA", "MIAMI"),       ("LHR", "LONDON"),      ("CDG", "PARIS"),
    ("FRA", "FRANKFURT"),   ("DXB", "DUBAI"),       ("NRT", "TOKYO"),
    ("SIN", "SINGAPORE"),   ("SYD", "SYDNEY"),      ("YYZ", "TORONTO"),
    ("GRU", "SAO PAULO"),   ("MEX", "MEXICO CITY"), ("AMS", "AMSTERDAM"),
    ("BCN", "BARCELONA"),   ("BOM", "MUMBAI"),
]

# ─── Airlines ─────────────────────────────────────────────────────────────────
AIRLINES = [
    ("UA", "UNITED AIRLINES"),         ("AA", "AMERICAN AIRLINES"),
    ("DL", "DELTA AIR LINES"),         ("WN", "SOUTHWEST AIRLINES"),
    ("BA", "BRITISH AIRWAYS"),         ("AF", "AIR FRANCE"),
    ("LH", "LUFTHANSA"),               ("EK", "EMIRATES"),
    ("SQ", "SINGAPORE AIRLINES"),      ("QF", "QANTAS AIRWAYS"),
    ("CX", "CATHAY PACIFIC"),          ("JL", "JAPAN AIRLINES"),
    ("NH", "ANA ALL NIPPON AIRWAYS"),  ("KE", "KOREAN AIR"),
    ("TK", "TURKISH AIRLINES"),        ("QR", "QATAR AIRWAYS"),
    ("EY", "ETIHAD AIRWAYS"),          ("AC", "AIR CANADA"),
    ("VS", "VIRGIN ATLANTIC"),         ("IB", "IBERIA"),
]

AIRPORTS = [
    "JFK", "LAX", "ORD", "DFW", "ATL", "SFO", "MIA", "SEA", "BOS", "IAH",
    "LHR", "CDG", "FRA", "AMS", "MAD", "FCO", "ZRH", "MUC", "VIE", "CPH",
    "DXB", "DOH", "AUH", "SIN", "HKG", "NRT", "PEK", "PVG", "ICN", "BOM",
    "SYD", "MEL", "YYZ", "YVR", "GRU", "EZE", "MEX", "BOG", "SCL", "LIM",
]

CABIN_CLASSES = ["ECONOMY", "PREMIUM_ECO", "BUSINESS", "FIRST"]
CABIN_WEIGHTS  = [60, 15, 20, 5]

FARE_BASIS_CODES = [
    "YOWUS", "QOWUS", "MOWUS", "HOWUS", "KOWUS",
    "YRTUS", "QRTUS", "BRTUS", "VRTUS", "LRTUS",
    "J1OWS", "C1OWS", "D1OWS", "P1RTG", "F1OWS",
]

# ─── Acquirer BINs ────────────────────────────────────────────────────────────
ACQUIRER_BINS = [
    "400001", "400002", "411111", "411112", "424242",
    "431274", "444444", "451234", "465432", "400566",
    "414720", "418685", "426617", "432610", "437826",
]

# ─── Helper functions ──────────────────────────────────────────────────────────

def gen_pan(card_type: str = "VISA_CLASSIC") -> str:
    """Generate a Luhn-valid Visa PAN."""
    length = 16
    prefix = "4"
    digits = [int(c) for c in (prefix + "".join(random.choices(string.digits, k=length - 2)))]
    total  = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    check  = (10 - (total % 10)) % 10
    return "".join(str(d) for d in digits) + str(check)


def gen_auth_code() -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=6))


def gen_merchant_id() -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=15))


def gen_terminal_id() -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=8))


def gen_arn() -> str:
    return "".join(random.choices(string.digits, k=23))


def gen_retrieval_ref() -> str:
    return "".join(random.choices(string.digits, k=12))


def gen_stan() -> str:
    return "".join(random.choices(string.digits, k=6))


def gen_folio() -> str:
    return "F" + "".join(random.choices(string.digits, k=7))


def gen_agreement() -> str:
    return "AGR" + "".join(random.choices(string.ascii_uppercase + string.digits, k=9))


def gen_ticket_number() -> str:
    """13-digit airline ticket number."""
    return "".join(random.choices(string.digits, k=13))


def gen_pnr() -> str:
    """6-char airline PNR."""
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=6))


def gen_flight_number(airline_code: str) -> str:
    return f"{airline_code}{random.randint(100, 9999)}"


def gen_seat() -> str:
    row = random.randint(1, 50)
    col = random.choice(list("ABCDEF"))
    return f"{row}{col}"


def gen_loyalty() -> str:
    return "".join(random.choices(string.ascii_uppercase, k=2)) + \
           "".join(random.choices(string.digits, k=9))


def gen_subscription_id() -> str:
    return "SUB-" + "".join(random.choices(string.ascii_uppercase + string.digits, k=12))


def gen_confirmation() -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=8))


def future_date(days_min: int = 1, days_max: int = 30) -> str:
    dt = datetime.now() + timedelta(days=random.randint(days_min, days_max))
    return dt.strftime("%Y%m%d")


def past_date(days_min: int = 1, days_max: int = 90) -> str:
    dt = datetime.now() - timedelta(days=random.randint(days_min, days_max))
    return dt.strftime("%Y%m%d")


def random_datetime(days_back: int = 30) -> str:
    dt = datetime.now() - timedelta(
        days=random.randint(0, days_back),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59),
        seconds=random.randint(0, 59),
    )
    return dt.strftime("%Y%m%d%H%M%S")


def random_currency() -> str:
    from data.vcf.vcf_models import CURRENCIES as _CURR
    return random.choice(list(_CURR.keys()))


def random_country() -> str:
    from data.vcf.vcf_models import COUNTRIES as _CTR
    return random.choice(list(_CTR.keys()))


def weighted_choice(items: list, weights: list):
    return random.choices(items, weights=weights, k=1)[0]


def processing_code_for(category) -> str:
    from data.vcf.vcf_models import VCFCategory
    mapping = {
        VCFCategory.PURCHASE:   ["000000", "000010", "000020"],
        VCFCategory.CASH:       ["010000", "011000", "012000"],
        VCFCategory.REFUND:     ["200000", "200010"],
        VCFCategory.TRANSFER:   ["400000", "401000", "500000"],
        VCFCategory.HOTEL:      ["000000", "000010"],
        VCFCategory.CAR_RENTAL: ["000000", "000010"],
        VCFCategory.AIRLINE:    ["000000", "000010"],
        VCFCategory.RECURRING:  ["000000", "000020"],
        VCFCategory.CHARGEBACK: ["200000"],
        VCFCategory.CARD_MGMT:  ["900000"],
    }
    choices = mapping.get(category, ["000000"])
    return random.choice(choices)


def pos_entry_for(category) -> str:
    from data.vcf.vcf_models import VCFCategory
    mapping = {
        VCFCategory.PURCHASE:   ["05", "07", "02", "81", "10"],
        VCFCategory.CASH:       ["01", "02", "05"],
        VCFCategory.HOTEL:      ["01", "10"],
        VCFCategory.CAR_RENTAL: ["01", "10"],
        VCFCategory.AIRLINE:    ["01", "81", "10"],
        VCFCategory.RECURRING:  ["10"],
        VCFCategory.TRANSFER:   ["00"],
        VCFCategory.CARD_MGMT:  ["00"],
        VCFCategory.REFUND:     ["05", "02"],
        VCFCategory.CHARGEBACK: ["00"],
    }
    return random.choice(mapping.get(category, ["00"]))
