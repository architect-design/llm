"""
VISA VCF Validator — Full Specification Coverage
Validates all 10 VCF transaction categories:
  CARD_MGMT, PURCHASE, CASH, HOTEL, CAR_RENTAL, AIRLINE,
  REFUND, CHARGEBACK, TRANSFER, RECURRING
"""

import re
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


class Severity(Enum):
    ERROR   = "ERROR"
    WARNING = "WARNING"
    INFO    = "INFO"


@dataclass
class VCFIssue:
    severity:    Severity
    field:       str
    record_type: str
    line_number: int
    message:     str
    value:       Optional[str] = None
    expected:    Optional[str] = None


@dataclass
class VCFValidationReport:
    is_valid:   bool
    errors:     List[VCFIssue] = field(default_factory=list)
    warnings:   List[VCFIssue] = field(default_factory=list)
    info:       List[VCFIssue] = field(default_factory=list)
    statistics: Dict           = field(default_factory=dict)

    def add(self, issue: VCFIssue):
        {Severity.ERROR: self.errors, Severity.WARNING: self.warnings,
         Severity.INFO: self.info}[issue.severity].append(issue)

    def to_dict(self) -> dict:
        def _r(i):
            return {"severity": i.severity.value, "field": i.field,
                    "record_type": i.record_type, "line": i.line_number,
                    "message": i.message, "value": i.value, "expected": i.expected}
        return {
            "is_valid": self.is_valid,
            "summary":  {"errors": len(self.errors), "warnings": len(self.warnings),
                         "info": len(self.info)},
            "errors":   [_r(i) for i in self.errors],
            "warnings": [_r(i) for i in self.warnings],
            "info":     [_r(i) for i in self.info],
            "statistics": self.statistics,
        }


VALID_CATEGORIES    = {"CARD_MGMT","PURCHASE","CASH","HOTEL","CAR_RENTAL",
                        "AIRLINE","REFUND","CHARGEBACK","TRANSFER","RECURRING"}
VALID_TXN_CODES     = {"01","02","03","04","05","06","07","08","09","10","11","12","13","14",
                        "15","20","21","22","23","24","25","26","27","28","29","30","31","32",
                        "33","34","35","36","37","40","41","42","43","44","45","46","47","50",
                        "51","52","53","54","60","61","62","63","64","65","66","67","68","70",
                        "71","72","73","74","75","92","96"}
VALID_CURRENCIES    = {"840","978","826","392","124","036","756","356","986","156","484","344",
                        "702","208","752","578","554","458","764","410","682","784","566","710"}
VALID_RESPONSE_CODES= {"00","01","03","04","05","06","08","10","12","13","14","15","19","25",
                        "30","41","43","51","54","55","57","58","61","62","65","75","76","78",
                        "91","92","96","N7","CV","CW"}
VALID_POS_MODES     = {"00","01","02","05","07","10","71","79","81","90","91","95"}
VALID_CARD_TYPES    = {"VISA_CLASSIC","VISA_GOLD","VISA_PLATINUM","VISA_INFINITE",
                        "VISA_SIGNATURE","VISA_BUSINESS","VISA_CORPORATE","VISA_PURCHASING"}
VALID_VEHICLE_CLASSES = {"ECONOMY","COMPACT","MIDSIZE","FULLSIZE","SUV","LUXURY","VAN","ELECTRIC"}
VALID_CABIN_CLASSES = {"ECONOMY","PREMIUM_ECO","BUSINESS","FIRST"}
VALID_TICKET_TYPES  = {"ONE_WAY","ROUND_TRIP","MULTI_CITY"}
VALID_BILLING_CYCLES= {"MONTHLY","QUARTERLY","ANNUAL","WEEKLY","BIANNUAL"}
VALID_FUEL_POLICIES = {"FULL_TO_FULL","PREPAID","INCLUDED","SIMILAR_TO_SIMILAR"}
VALID_INSURANCE_TYPES={"CDW","LDW","PAI","NONE","SLI","PEC"}
VALID_ROOM_TYPES    = {"STD","DLX","STE","PNT","JR_STE","TWIN","FAMILY"}
VALID_CARD_ACTIONS  = {"ACTIVATE","BLOCK","REPLACE","ISSUE","CLOSURE",
                        "LIMIT_CHANGE","PIN_CHANGE","SUPPLEMENTARY_ADD"}
VALID_BLOCK_REASONS = {"LOST","STOLEN","FRAUD","REQUEST","EXPIRED","DAMAGED",""}
VALID_CARD_DELIVERY = {"BRANCH","MAIL","DIGITAL","COURIER"}

EXT = 25   # extended fields start after base field index 24


class VCFValidator:
    """VISA VCF file validator — pipe-delimited and fixed-width support."""

    def validate(self, content: str) -> VCFValidationReport:
        report = VCFValidationReport(is_valid=True)
        lines  = [l for l in content.splitlines() if l.strip()]
        if not lines:
            report.add(VCFIssue(Severity.ERROR,"file","FILE",0,"Empty file"))
            report.is_valid = False
            return report

        delim = next((d for d in ("|",",","\t") if lines[0].count(d) >= 5), None)
        if delim:
            self._validate_delimited(lines, delim, report)
        else:
            self._validate_fixed(lines, report)

        report.statistics = self._build_stats(lines, delim)
        report.is_valid   = len(report.errors) == 0
        return report

    # ── Delimited ─────────────────────────────────────────────────────────────

    def _validate_delimited(self, lines, delim, report):
        self._check_header(lines[0].split(delim), 1, report)
        self._check_trailer(lines[-1].split(delim), len(lines), report)
        for i, line in enumerate(lines[1:-1], 2):
            if not line.strip(): continue
            p = line.split(delim)
            self._check_base(p, i, report)
            cat = p[23].strip().upper() if len(p) > 23 else ""
            {
                "HOTEL":      self._check_hotel,
                "CAR_RENTAL": self._check_car_rental,
                "AIRLINE":    self._check_airline,
                "RECURRING":  self._check_recurring,
                "CARD_MGMT":  self._check_card_mgmt,
                "CASH":       self._check_cash,
                "CHARGEBACK": self._check_chargeback,
            }.get(cat, lambda p,i,r: None)(p, i, report)

    def _check_header(self, p, ln, report):
        if len(p) < 4:
            report.add(VCFIssue(Severity.ERROR,"header_fields","HEADER",ln,
                                f"Header needs ≥4 fields, got {len(p)}",str(len(p)),"≥4"))
            return
        ft = p[0].strip().upper()
        if ft not in ("VCF","VISA","VISANET","VCF_BATCH"):
            report.add(VCFIssue(Severity.WARNING,"file_type","HEADER",ln,
                                f"Unexpected file type '{ft}'",ft,"VCF"))
        if len(p) > 2 and p[2].strip() and not re.match(r"^\d{6}$", p[2].strip()):
            report.add(VCFIssue(Severity.ERROR,"acquirer_bin","HEADER",ln,
                                "Acquirer BIN must be 6 digits",p[2].strip(),"6N"))
        if len(p) > 3 and p[3].strip() and not re.match(r"^\d{8}$", p[3].strip()):
            report.add(VCFIssue(Severity.ERROR,"file_date","HEADER",ln,
                                "File date must be YYYYMMDD",p[3].strip(),"YYYYMMDD"))

    def _check_trailer(self, p, ln, report):
        if p[0].strip().upper() not in ("TRAILER","TRL","99","EOF","END"):
            report.add(VCFIssue(Severity.WARNING,"trailer_id","TRAILER",ln,
                                f"Unexpected trailer id '{p[0].strip()}'"))
        if len(p) > 2:
            try: float(p[2].strip())
            except ValueError:
                report.add(VCFIssue(Severity.ERROR,"total_amount","TRAILER",ln,
                                    "Trailer total must be numeric",p[2].strip()))

    # ── Base fields ────────────────────────────────────────────────────────────

    def _check_base(self, p, ln, report):
        if len(p) < 20:
            report.add(VCFIssue(Severity.ERROR,"field_count","TRANSACTION",ln,
                                f"Got {len(p)} fields, need ≥20",str(len(p)),"≥20"))
            return

        def f(i): return p[i].strip() if i < len(p) else ""

        tc = f(0)
        if tc and tc not in VALID_TXN_CODES:
            report.add(VCFIssue(Severity.WARNING,"transaction_code","TRANSACTION",ln,
                                f"Unrecognised TC '{tc}'",tc))

        pan = f(1)
        if pan:
            clean = pan.replace("*","").replace("X","")
            if not re.match(r"^\d{13,19}$", clean):
                report.add(VCFIssue(Severity.ERROR,"pan","TRANSACTION",ln,
                                    "PAN must be 13-19 digits",pan))
            elif "*" not in pan and "X" not in pan.upper():
                if not self._luhn(clean):
                    report.add(VCFIssue(Severity.ERROR,"pan_luhn","TRANSACTION",ln,
                                        "PAN failed Luhn check",pan))

        amt = f(3)
        if amt:
            try:
                v = float(amt.replace(",",""))
                if v < 0:
                    report.add(VCFIssue(Severity.ERROR,"amount","TRANSACTION",ln,"Amount < 0",amt))
                if v > 1_000_000:
                    report.add(VCFIssue(Severity.WARNING,"amount","TRANSACTION",ln,
                                        "Amount > $1M — verify",amt))
            except ValueError:
                report.add(VCFIssue(Severity.ERROR,"amount","TRANSACTION",ln,
                                    "Amount must be numeric",amt))

        curr = f(4)
        if curr and curr not in VALID_CURRENCIES:
            report.add(VCFIssue(Severity.ERROR,"currency_code","TRANSACTION",ln,
                                f"Invalid ISO 4217 code '{curr}'",curr))

        dt = f(5)
        if dt and not re.match(r"^\d{8}(\d{4,6})?$", dt):
            report.add(VCFIssue(Severity.ERROR,"transaction_dt","TRANSACTION",ln,
                                "Datetime must be YYYYMMDD[HHmmSS]",dt))

        mcc = f(6)
        if mcc and not re.match(r"^\d{4}$", mcc):
            report.add(VCFIssue(Severity.ERROR,"mcc","TRANSACTION",ln,
                                "MCC must be 4 digits",mcc,"4N"))

        pos = f(7)
        if pos and pos not in VALID_POS_MODES:
            report.add(VCFIssue(Severity.WARNING,"pos_entry_mode","TRANSACTION",ln,
                                f"Unrecognised POS mode '{pos}'",pos))

        resp = f(8)
        if resp and resp not in VALID_RESPONSE_CODES:
            report.add(VCFIssue(Severity.WARNING,"response_code","TRANSACTION",ln,
                                f"Unrecognised response '{resp}'",resp))

        arn = f(14)
        if arn and not re.match(r"^\d{23}$", arn):
            report.add(VCFIssue(Severity.WARNING,"arn","TRANSACTION",ln,
                                "ARN should be 23 digits",arn,"23N"))

        ctype = f(22)
        if ctype and ctype not in VALID_CARD_TYPES:
            report.add(VCFIssue(Severity.INFO,"card_type","TRANSACTION",ln,
                                f"Unrecognised card type '{ctype}'",ctype))

        cat = f(23).upper()
        if cat and cat not in VALID_CATEGORIES:
            report.add(VCFIssue(Severity.WARNING,"category","TRANSACTION",ln,
                                f"Unrecognised category '{cat}'",cat))

    # ── Category validators ────────────────────────────────────────────────────

    def _ef(self, p, offset):
        idx = EXT + offset
        return p[idx].strip() if idx < len(p) else ""

    def _check_hotel(self, p, ln, report):
        ef = lambda o: self._ef(p, o)
        for name, o in (("check_in_date",2),("check_out_date",3)):
            v = ef(o)
            if v and not re.match(r"^\d{8}$", v):
                report.add(VCFIssue(Severity.ERROR,name,"HOTEL",ln,f"{name} must be YYYYMMDD",v,"YYYYMMDD"))
        nights = ef(4)
        if nights:
            try:
                n = int(nights)
                if n < 1 or n > 365:
                    report.add(VCFIssue(Severity.WARNING,"nights","HOTEL",ln,f"Unusual nights {n}",nights))
            except ValueError:
                report.add(VCFIssue(Severity.ERROR,"nights","HOTEL",ln,"Nights must be integer",nights))
        rt = ef(5)
        if rt and rt not in VALID_ROOM_TYPES:
            report.add(VCFIssue(Severity.INFO,"room_type","HOTEL",ln,f"Uncommon room type '{rt}'",rt))
        for fname, o in (("room_rate",6),("extra_charges",10),("tax_amount",11)):
            v = ef(o)
            if v:
                try:
                    if float(v) < 0:
                        report.add(VCFIssue(Severity.ERROR,fname,"HOTEL",ln,f"{fname} < 0",v))
                except ValueError:
                    report.add(VCFIssue(Severity.ERROR,fname,"HOTEL",ln,f"{fname} must be numeric",v))
        ns = ef(9)
        if ns and ns not in ("Y","N","YES","NO",""):
            report.add(VCFIssue(Severity.WARNING,"no_show","HOTEL",ln,"Must be Y or N",ns,"Y/N"))

    def _check_car_rental(self, p, ln, report):
        ef = lambda o: self._ef(p, o)
        for name, o in (("pickup_date",2),("return_date",3)):
            v = ef(o)
            if v and not re.match(r"^\d{8}$", v):
                report.add(VCFIssue(Severity.ERROR,name,"CAR_RENTAL",ln,f"{name} must be YYYYMMDD",v,"YYYYMMDD"))
        days = ef(4)
        if days:
            try:
                d = int(days)
                if d < 1 or d > 366:
                    report.add(VCFIssue(Severity.WARNING,"rental_days","CAR_RENTAL",ln,f"Unusual days {d}",days))
            except ValueError:
                report.add(VCFIssue(Severity.ERROR,"rental_days","CAR_RENTAL",ln,"Days must be integer",days))
        vc = ef(5)
        if vc and vc not in VALID_VEHICLE_CLASSES:
            report.add(VCFIssue(Severity.INFO,"vehicle_class","CAR_RENTAL",ln,f"Uncommon class '{vc}'",vc))
        fp = ef(13)
        if fp and fp not in VALID_FUEL_POLICIES:
            report.add(VCFIssue(Severity.INFO,"fuel_policy","CAR_RENTAL",ln,f"Uncommon fuel policy '{fp}'",fp))
        it = ef(16)
        if it and it not in VALID_INSURANCE_TYPES:
            report.add(VCFIssue(Severity.INFO,"insurance_type","CAR_RENTAL",ln,f"Uncommon ins type '{it}'",it))
        for fname, o in (("daily_rate",6),("damage_amount",14),("fuel_charge",15),("insurance_amount",17)):
            v = ef(o)
            if v:
                try:
                    if float(v) < 0:
                        report.add(VCFIssue(Severity.ERROR,fname,"CAR_RENTAL",ln,f"{fname} < 0",v))
                except ValueError:
                    report.add(VCFIssue(Severity.ERROR,fname,"CAR_RENTAL",ln,f"{fname} must be numeric",v))

    def _check_airline(self, p, ln, report):
        ef = lambda o: self._ef(p, o)
        code = ef(0)
        if code and not re.match(r"^[A-Z0-9]{2}$", code.upper()):
            report.add(VCFIssue(Severity.WARNING,"airline_code","AIRLINE",ln,"IATA code must be 2 chars",code,"2AN"))
        ticket = ef(2)
        if ticket and not re.match(r"^\d{13}$", ticket):
            report.add(VCFIssue(Severity.WARNING,"ticket_number","AIRLINE",ln,"Ticket must be 13 digits",ticket,"13N"))
        for fname, o in (("origin",4),("destination",5)):
            ap = ef(o)
            if ap and not re.match(r"^[A-Z]{3}$", ap.upper()):
                report.add(VCFIssue(Severity.WARNING,fname,"AIRLINE",ln,"Airport code must be 3 letters",ap,"3A"))
        cabin = ef(9)
        if cabin and cabin not in VALID_CABIN_CLASSES:
            report.add(VCFIssue(Severity.WARNING,"cabin_class","AIRLINE",ln,f"Unrecognised cabin '{cabin}'",cabin))
        tt = ef(11)
        if tt and tt not in VALID_TICKET_TYPES:
            report.add(VCFIssue(Severity.INFO,"ticket_type","AIRLINE",ln,f"Uncommon ticket type '{tt}'",tt))
        pnr = ef(17)
        if pnr and not re.match(r"^[A-Z0-9]{6}$", pnr.upper()):
            report.add(VCFIssue(Severity.INFO,"pnr","AIRLINE",ln,"PNR should be 6 alphanumeric",pnr,"6AN"))
        for fname, o in (("base_fare",12),("taxes_fees",13),("ancillary_fee",14),("baggage_fee",15)):
            v = ef(o)
            if v:
                try:
                    if float(v) < 0:
                        report.add(VCFIssue(Severity.ERROR,fname,"AIRLINE",ln,f"{fname} < 0",v))
                except ValueError:
                    report.add(VCFIssue(Severity.ERROR,fname,"AIRLINE",ln,f"{fname} must be numeric",v))

    def _check_recurring(self, p, ln, report):
        ef = lambda o: self._ef(p, o)
        cycle = ef(1)
        if cycle and cycle not in VALID_BILLING_CYCLES:
            report.add(VCFIssue(Severity.WARNING,"billing_cycle","RECURRING",ln,f"Unrecognised cycle '{cycle}'",cycle))
        for fname, o in (("cycle_number",2),("total_cycles",3)):
            v = ef(o)
            if v:
                try: int(v)
                except ValueError:
                    report.add(VCFIssue(Severity.ERROR,fname,"RECURRING",ln,f"{fname} must be integer",v))
        for fname, o in (("service_start",5),("service_end",6),("next_billing_date",9)):
            v = ef(o)
            if v and not re.match(r"^\d{8}$", v):
                report.add(VCFIssue(Severity.ERROR,fname,"RECURRING",ln,f"{fname} must be YYYYMMDD",v,"YYYYMMDD"))

    def _check_card_mgmt(self, p, ln, report):
        ef = lambda o: self._ef(p, o)
        action = ef(0)
        if action and action not in VALID_CARD_ACTIONS:
            report.add(VCFIssue(Severity.WARNING,"card_action","CARD_MGMT",ln,f"Unrecognised action '{action}'",action))
        expiry = ef(2)
        if expiry and not re.match(r"^\d{4}$", expiry):
            report.add(VCFIssue(Severity.WARNING,"new_expiry","CARD_MGMT",ln,"Expiry should be MMYY",expiry,"MMYY"))
        br = ef(3)
        if br and br not in VALID_BLOCK_REASONS:
            report.add(VCFIssue(Severity.INFO,"block_reason","CARD_MGMT",ln,f"Uncommon block reason '{br}'",br))
        delivery = ef(6)
        if delivery and delivery not in VALID_CARD_DELIVERY:
            report.add(VCFIssue(Severity.INFO,"card_delivery","CARD_MGMT",ln,f"Uncommon delivery '{delivery}'",delivery))

    def _check_cash(self, p, ln, report):
        amt = p[3].strip() if len(p) > 3 else ""
        if amt:
            try:
                if float(amt) > 10000:
                    report.add(VCFIssue(Severity.WARNING,"amount","CASH",ln,
                                        f"Large cash transaction ${float(amt):.2f}",amt))
            except ValueError:
                pass

    def _check_chargeback(self, p, ln, report):
        tc = p[0].strip() if p else ""
        if tc not in ("25","26","28","29"):
            report.add(VCFIssue(Severity.WARNING,"transaction_code","CHARGEBACK",ln,
                                f"Unexpected TC '{tc}' for chargeback",tc,"25/26/28/29"))

    # ── Fixed-width ────────────────────────────────────────────────────────────

    def _validate_fixed(self, lines, report):
        for i, line in enumerate(lines, 1):
            if len(line) < 10: continue
            rt = line[:2]
            if rt == "00":
                if not re.match(r"^\d{8}$", line[4:12]):
                    report.add(VCFIssue(Severity.ERROR,"file_date","HEADER",i,"Date must be YYYYMMDD",line[4:12]))
            elif rt in VALID_TXN_CODES:
                if len(line) < 50:
                    report.add(VCFIssue(Severity.ERROR,"record_length","TRANSACTION",i,
                                        f"Fixed record too short ({len(line)} chars)"))
            elif rt not in ("99","  "):
                report.add(VCFIssue(Severity.INFO,"record_type","UNKNOWN",i,
                                    f"Unrecognised record type '{rt}'",rt))

    # ── Luhn ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _luhn(pan: str) -> bool:
        try:
            digits = [int(d) for d in pan if d.isdigit()]
            if len(digits) < 13: return False
            odd   = digits[-1::-2]
            even  = digits[-2::-2]
            total = sum(odd) + sum(sum(divmod(d*2,10)) for d in even)
            return total % 10 == 0
        except Exception:
            return False

    # ── Statistics ─────────────────────────────────────────────────────────────

    def _build_stats(self, lines, delim) -> dict:
        stats = {"total_lines": len(lines), "format": "delimited" if delim else "fixed-width",
                 "delimiter": delim or "N/A"}
        if delim and len(lines) > 2:
            txns = lines[1:-1]
            cats: Dict[str,int] = {}
            approved = declined = 0
            total_amt = 0.0
            for line in txns:
                if not line.strip(): continue
                p = line.split(delim)
                cat  = p[23].strip() if len(p) > 23 else ""
                resp = p[8].strip()  if len(p) > 8  else ""
                amt  = p[3].strip()  if len(p) > 3  else "0"
                cats[cat] = cats.get(cat, 0) + 1
                if resp == "00": approved += 1
                else:            declined += 1
                try: total_amt += float(amt.replace(",",""))
                except ValueError: pass
            n = len([l for l in txns if l.strip()])
            stats.update({
                "transaction_count":    n,
                "approved_count":       approved,
                "declined_count":       declined,
                "approval_rate_pct":    round(approved/max(n,1)*100, 1),
                "total_amount":         round(total_amt, 2),
                "category_distribution":cats,
            })
        return stats
