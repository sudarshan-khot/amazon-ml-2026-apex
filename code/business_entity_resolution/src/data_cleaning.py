"""Data cleaning for the Business Entity Resolution challenge -- single entry point.

Run from code/business_entity_resolution/:

    python src/data_cleaning.py --data ../../dataset

What it does
  1. builds the learned resources (artifacts/resources.pkl) if they do not exist:
       * Source-1 name vocabulary (train + test Source 1, no labels)
       * phonetic index of Indian Source-1 words (native-script fallback)
       * native-script -> English word lexicon learned from TRAIN ground-truth pairs
  2. normalises every source file (train + test, Source 1/2/3) in parallel
  3. fills state_inf (state implied by the city, learned from Source 1, purity >= 0.95)
  4. writes tab-separated files with the same layout as the input dataset:
       artifacts/cleaned_dataset/train/train_source1.tsv ... test/test_source3.tsv

Read the output like the original data:
    pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
(keep_default_na=False matters: a business may genuinely be called 'Nan' or 'Null'.)

Sections: A dictionaries | B normalisation | C learned resources | D pipeline
"""
import argparse
import csv
import math
import os
import pickle
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from functools import lru_cache
from multiprocessing import Pool

import pandas as pd
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
from rapidfuzz.distance import Levenshtein

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


# ######################################################################################
# SECTION A - DICTIONARIES (mined from the training pairs / standard postal & legal terms)
# ######################################################################################
# --------------------------------------------------------------------------------------
# Missing-value markers (whole address part, after folding)
# --------------------------------------------------------------------------------------
NULL_PARTS = {"null", "n a", "na", "none", "nan", "nil", "not available", "unknown", "-", ""}

# --------------------------------------------------------------------------------------
# States / regions  (whole address part  ->  canonical code)
# --------------------------------------------------------------------------------------
US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california",
    "co": "colorado", "ct": "connecticut", "de": "delaware", "dc": "district of columbia",
    "fl": "florida", "ga": "georgia", "hi": "hawaii", "id": "idaho", "il": "illinois",
    "in": "indiana", "ia": "iowa", "ks": "kansas", "ky": "kentucky", "la": "louisiana",
    "me": "maine", "md": "maryland", "ma": "massachusetts", "mi": "michigan", "mn": "minnesota",
    "ms": "mississippi", "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada",
    "nh": "new hampshire", "nj": "new jersey", "nm": "new mexico", "ny": "new york",
    "nc": "north carolina", "nd": "north dakota", "oh": "ohio", "ok": "oklahoma", "or": "oregon",
    "pa": "pennsylvania", "ri": "rhode island", "sc": "south carolina", "sd": "south dakota",
    "tn": "tennessee", "tx": "texas", "ut": "utah", "vt": "vermont", "va": "virginia",
    "wa": "washington", "wv": "west virginia", "wi": "wisconsin", "wy": "wyoming",
    "pr": "puerto rico",
}
INDIA_STATES = {
    "mh": "maharashtra", "dl": "delhi", "up": "uttar pradesh", "ka": "karnataka",
    "tn": "tamil nadu", "wb": "west bengal", "gj": "gujarat", "tg": "telangana",
    "hr": "haryana", "rj": "rajasthan", "kl": "kerala", "br": "bihar", "mp": "madhya pradesh",
    "ap": "andhra pradesh", "od": "orissa", "pb": "punjab",
}
# states with no code in the data: matched by full name only (their 2-letter codes such as
# 'ch', 'as', 'ga' collide with ordinary address words)
INDIA_STATE_NAMES_ONLY = {
    "goa": "GA", "assam": "AS", "jharkhand": "JH", "chhattisgarh": "CG", "uttarakhand": "UK",
    "himachal pradesh": "HP", "jammu and kashmir": "JK", "chandigarh": "CH", "puducherry": "PY",
    "pondicherry": "PY", "uttaranchal": "UK",
}
INDIA_STATE_ALIASES = {  # alternative spellings -> code
    "odisha": "od", "keralam": "kl", "nct of delhi": "dl", "orissa": "od",
}
# native-script state names (the only native text that occurs in addresses; mined, purity 1.00)
INDIA_NATIVE_STATES = {
    "महाराष्ट्र": "mh", "दिल्ली": "dl", "उत्तर प्रदेश": "up", "ಕರ್ನಾಟಕ": "ka", "தமிழ்நாடு": "tn",
    "ગુજરાત": "gj", "পশ্চিমবঙ্গ": "wb", "తెలంగాణ": "tg", "हरियाणा": "hr", "राजस्थान": "rj",
    "കേരളം": "kl", "बिहार": "br", "मध्य प्रदेश": "mp", "ఆంధ్రప్రదేశ్": "ap", "ਪੰਜਾਬ": "pb",
    "ଓଡ଼ିଶା": "od",
}
FRANCE_REGIONS = {
    "hauts de france": "hdf", "nouvelle aquitaine": "naq", "pays de la loire": "pdl",
    "ile de france": "idf", "bretagne": "bre", "normandie": "nor", "occitanie": "occ",
    "grand est": "ges", "auvergne rhone alpes": "ara", "bourgogne franche comte": "bfc",
    "centre val de loire": "cvl", "provence alpes cote d azur": "pac", "corse": "cor",
}
FRANCE_DEPARTMENTS = {  # department -> region (all departments seen in the data + neighbours)
    "nord": "hdf", "pas de calais": "hdf", "somme": "hdf", "aisne": "hdf", "oise": "hdf",
    "gironde": "naq", "landes": "naq", "charente maritime": "naq", "dordogne": "naq",
    "loire atlantique": "pdl", "vendee": "pdl", "maine et loire": "pdl", "sarthe": "pdl",
    "mayenne": "pdl",
}


# Source 2 still labels Hyderabad-area addresses 'Andhra Pradesh' (pre-2014 state) while
# Source 1 says 'Telangana': 570 of 604 state disagreements in true pairs.  One group code.
STATE_GROUPS = {"TG": "APTG", "AP": "APTG"}


def build_state_lookup():
    """country -> {folded part text -> canonical state code}"""
    us = {}
    for code, name in US_STATES.items():
        us[code] = code.upper(); us[name] = code.upper()
    ind = {}
    for code, name in INDIA_STATES.items():
        ind[code] = code.upper(); ind[name] = code.upper()
    for alias, code in INDIA_STATE_ALIASES.items():
        ind[alias] = code.upper()
    ind.update(INDIA_STATE_NAMES_ONLY)
    fr = {}
    for name, code in FRANCE_REGIONS.items():
        fr[name] = code.upper()
    for name, code in FRANCE_DEPARTMENTS.items():
        fr[name] = code.upper()
    return {"US": us, "India": ind, "France": fr}


# --------------------------------------------------------------------------------------
# Address token canonicalisation (token -> canonical token), per country
# --------------------------------------------------------------------------------------
_US_STREET = {
    "street": "st", "str": "st", "saint": "st", "road": "rd", "drive": "dr", "drv": "dr",
    "avenue": "ave", "av": "ave", "aven": "ave", "lane": "ln", "court": "ct", "crt": "ct",
    "place": "pl", "circle": "cir", "circ": "cir", "trail": "trl", "boulevard": "blvd",
    "boul": "blvd", "highway": "hwy", "hiway": "hwy", "parkway": "pkwy", "pkway": "pkwy",
    "terrace": "ter", "terr": "ter", "cove": "cv", "square": "sq", "point": "pt",
    "turnpike": "tpke", "ridge": "rdg", "fort": "ft", "mount": "mt", "mountain": "mtn",
    "expressway": "expy", "freeway": "fwy", "crossing": "xing", "heights": "hts",
    "junction": "jct", "center": "ctr", "centre": "ctr", "creek": "crk", "grove": "grv",
    "harbor": "hbr", "island": "is", "lake": "lk", "meadows": "mdws", "park": "park",
    "plaza": "plz", "route": "rte", "station": "sta", "valley": "vly", "view": "vw",
    "village": "vlg", "way": "way", "loop": "loop", "run": "run", "pike": "pike",
    "north": "n", "south": "s", "east": "e", "west": "w", "northeast": "ne",
    "northwest": "nw", "southeast": "se", "southwest": "sw", "atl": "atlanta",
    "township": "twp", "county": "county",
}
_INDIA_STREET = {
    "road": "rd", "street": "st", "saint": "st", "marg": "marg", "nagar": "nagar", "ngr": "nagar",
    "sector": "sector", "sec": "sector", "phase": "phase", "ph": "phase", "colony": "colony",
    "col": "colony", "apartment": "apt", "apartments": "apt", "apts": "apt", "appt": "apt",
    "building": "bldg", "bldg": "bldg", "bld": "bldg", "complex": "complex", "cmplx": "complex",
    "society": "soc", "soc": "soc", "village": "vill", "vill": "vill", "vil": "vill",
    "vpo": "vill", "post": "po", "district": "dist", "distt": "dist", "dist": "dist",
    "taluk": "tal", "taluka": "tal", "tq": "tal", "tehsil": "tal", "teh": "tal",
    "main": "main", "cross": "cross", "layout": "layout", "block": "block", "blk": "block",
    "extension": "extn", "extn": "extn", "ext": "extn", "industrial": "indl", "indl": "indl",
    "estate": "estate", "market": "mkt", "mkt": "mkt", "chowk": "chowk", "north": "n",
    "south": "s", "east": "e", "west": "w", "highway": "hwy", "hwy": "hwy", "near": "",
    "nr": "", "opp": "", "opposite": "", "behind": "", "beside": "", "besides": "", "bh": "",
    "next": "", "to": "", "adjacent": "", "adj": "",
    # city aliases (old/new names and frequent misspellings)
    "bombay": "mumbai", "calcutta": "kolkata", "culcutta": "kolkata", "bengaluru": "bangalore",
    "banglore": "bangalore", "bangaluru": "bangalore", "gurugram": "gurgaon", "poona": "pune",
    "madras": "chennai", "ahmadabad": "ahmedabad", "baroda": "vadodara",
    "trivandrum": "thiruvananthapuram", "cochin": "kochi", "mysuru": "mysore",
    "mangaluru": "mangalore", "calicut": "kozhikode", "allahabad": "prayagraj",
    "benaras": "varanasi", "banaras": "varanasi", "vizag": "visakhapatnam",
    "gauhati": "guwahati", "simla": "shimla", "pondicherry": "puducherry",
    "keralam": "kerala", "odisha": "orissa", "hubballi": "hubli", "belagavi": "belgaum",
}
_FRANCE_STREET = {
    "rue": "r", "avenue": "av", "ave": "av", "boulevard": "bd", "blvd": "bd", "allee": "all",
    "alle": "all", "impasse": "imp", "chemin": "ch", "chem": "ch", "place": "pl",
    "route": "rte", "quai": "q", "cours": "crs", "passage": "psg", "pass": "psg",
    "residence": "res", "square": "sq", "faubourg": "fg", "saint": "st", "sainte": "ste",
    "bis": "", "ter": "", "numero": "",
    "chez": "", "de": "", "du": "", "des": "", "la": "", "le": "", "les": "", "l": "", "d": "",
    "et": "", "aux": "", "au": "", "en": "", "sur": "",
}
ADDR_CANON = {"US": _US_STREET, "India": _INDIA_STREET, "France": _FRANCE_STREET}
# words dropped from address tokens in every country (pure connectors)
ADDR_STOP = {"of", "the", "and", "at", "in", "no", "number", "nos", "floor", "flr"}

# Number-prefix words that the generator swaps freely (plot->h, flat->door, ...). When one
# of these directly precedes a number it is dropped and only the number is kept.
NUMBER_PREFIX_RE = (
    r"(?:h\s*no|hno|hn|house\s*no|door\s*no|d\s*no|dno|plot\s*no|plot|flat\s*no|flat|"
    r"shop\s*no|shop|office\s*no|room\s*no|unit\s*no|building\s*no|bldg\s*no|site\s*no|"
    r"no|nos|n[o°º]|numero|num|#+)"
)
# Floor expressions (noisy, low value, and leak fake numbers) are removed completely.
FLOOR_RE = (
    r"(?:\b(?:(?:\d+)\s*(?:st|nd|rd|th)?|ground|first|second|third|fourth|fifth|sixth|"
    r"seventh|eighth|ninth|tenth|top|upper\s+ground|lower\s+ground|upper|lower|basement|mezzanine)"
    r"\s*(?:floor|flr|fl)\b(?:\s*no)?|\b(?:floor|flr|fl)\s*(?:no\s*)?\d+\b|\b(?:u?l?g\s*f|f\s*f|s\s*f|t\s*f|lgf|ugf|gf|ff|sf|tf)\b)"
)
ORDINAL_WORDS = {
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5", "sixth": "6",
    "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10", "eleventh": "11",
    "twelfth": "12", "thirteenth": "13", "fourteenth": "14", "fifteenth": "15",
    "sixteenth": "16", "seventeenth": "17", "eighteenth": "18", "nineteenth": "19",
    "twentieth": "20",
}
UNIT_WORDS = {"unit", "apt", "apartment", "suite", "ste", "room", "rm", "#"}

# --------------------------------------------------------------------------------------
# Business-name vocabularies
# --------------------------------------------------------------------------------------
# legal / entity-type words -> canonical legal token.  Which of them count as legal depends
# on the country (LEGAL_BY_COUNTRY); an unknown country uses all of them.
LEGAL = {
    "inc": "inc", "incorporated": "inc", "incorporation": "inc",
    "corp": "corp", "corporation": "corp",
    "co": "co", "company": "co", "cie": "co", "compagnie": "co",
    "ltd": "ltd", "limited": "ltd", "ltda": "ltd",
    "pvt": "pvt", "private": "pvt", "pte": "pvt",
    "llc": "llc", "lc": "llc", "llp": "llp", "lp": "lp", "pllc": "pllc", "pc": "pc", "plc": "plc",
    "pa": "pa", "opc": "opc",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "eurl": "eurl", "sa": "sa", "sci": "sci",
    "ei": "ei", "snc": "snc", "scp": "scp", "scm": "scm", "selarl": "selarl", "scop": "scop",
    "gmbh": "gmbh",
}
_COMMON_LEGAL = {"inc", "incorporated", "incorporation", "corp", "corporation", "co", "company",
                 "ltd", "limited", "llc", "llp", "lp", "plc", "gmbh"}
LEGAL_BY_COUNTRY = {
    "US": _COMMON_LEGAL | {"lc", "pllc", "pc", "pa", "ltda", "pvt", "private"},
    "India": _COMMON_LEGAL | {"pvt", "private", "pte", "opc", "pllc", "pc"},
    "France": _COMMON_LEGAL | {"cie", "compagnie", "sarl", "sas", "sasu", "eurl", "sa", "sci",
                               "ei", "snc", "scp", "scm", "selarl", "scop"},
}
LEGAL_ALL = set(LEGAL)
# professional credentials appended to practitioner names (dropped by the generator like legal words)
CREDENTIALS = {"md", "dds", "dmd", "od", "dpm", "phd", "cpa", "dvm", "np", "rn", "esq", "mba",
               "lcsw", "lpc", "pt", "dpt", "ms", "facs", "facp", "faap", "msw", "cfp"}
# typo-tolerant targets for the long legal words (the generator scrambles 'private' a lot)
LEGAL_FUZZY_TARGETS = {"private": "pvt", "limited": "ltd", "incorporated": "inc",
                       "corporation": "corp", "company": "co", "compagnie": "co"}
# honorifics the generator injects as the FIRST word (India). 'shree/sree/ms' are real name
# words in Source 1 and are deliberately NOT here.
HONORIFICS = {"dr", "mr", "mrs", "smt", "sri", "shri", "m/s", "ms/", "messrs", "m s"}
NAME_STOP = {"the", "and", "of", "a", "an", "for", "de", "du", "des", "la", "le", "les", "l",
             "d", "et", "au", "aux", "en"}
# spelling variants of the same word (mined from low-purity native-lexicon entries)
NAME_VARIANTS = {"sree": "shree", "laxmi": "lakshmi", "jay": "jai", "shiv": "shiva",
                 "centre": "center", "frs": "freres", "ets": "etablissements",
                 "intl": "international", "mfg": "manufacturing", "svc": "services",
                 "svcs": "services", "assoc": "associates", "bros": "brothers",
                 "mgmt": "management"}
# alias markers: the real name is always the RIGHT-hand side (3352 / 3352 pairs).
# Unambiguous markers always split; 'aka'/'nee' (also ordinary name words) split only when
# the left-hand side is a generated brand.
ALIAS_STRONG_RE = (r"\s+(?:d\s*[/.]\s*b\s*[/.]\s*a\.?|dba|doing business as|formerly known as|"
                   r"formerly|f\s*[/.]\s*k\s*[/.]\s*a\.?|fka|a\s*[/.]\s*k\s*[/.]\s*a\.?|"
                   r"also known as|t\s*/\s*a|trading as)\s*:?\s+")
ALIAS_WEAK_RE = r"\s+(?:aka|n[eé]e)\s*:?\s+"
# random-brand generator syllables (validated: 1 false positive in 115k real S1 words)
BRAND_SYLLABLES = ("aria avi belo brix calo cira delta dova drex ecto evo faye flux gild halo iri "
                   "jax kelo kor lum lyra mira nex novi nyla onyx orbi pyra quo riza sol syn tavo "
                   "umbra vanta vantage arc veo vera vio wex xylo yuma zeph zeta").split()
BRAND_SUFFIX_WORDS = {"one", "labs", "sys", "co", "group", "io", "x"}
DOMAIN_TLD_RE = r"\.(?:c[o0]m|net|org|in|co\.in|co|fr|biz|info|io|us)\b"


# ######################################################################################
# SECTION B - NORMALISATION FUNCTIONS
# ######################################################################################
# ======================================================================================
# 0. Text repair and folding
# ======================================================================================
_MOJIBAKE = re.compile(r"[ÂâÃ]?[\x80-\x9f]+")           # e.g. 'Â\x80\x93' (broken en-dash)
_INVISIBLE = re.compile(r"[​-‏⁠﻿\x00-\x08\x0b-\x1f\x7f]")
_LIGATURES = str.maketrans({"œ": "oe", "Œ": "oe", "æ": "ae", "Æ": "ae", "ø": "o", "Ø": "o",
                            "ł": "l", "Ł": "l", "đ": "d", "ß": "ss", "’": "'", "‘": "'",
                            "`": "'", "´": "'", "–": "-", "—": "-", "¿": " ", '"': " "})
_INDIC = re.compile(r"[ऀ-෿]")
_SPACES = re.compile(r"\s+")


def repair(s):
    """Fix encoding damage; keeps case, accents and punctuation."""
    if not s:
        return ""
    s = _MOJIBAKE.sub(" ", s)
    s = _INVISIBLE.sub("", s)
    s = unicodedata.normalize("NFKC", s)          # also turns 'º' into 'o'
    s = s.translate(_LIGATURES)
    return _SPACES.sub(" ", s).strip()


def fold(s):
    """lowercase + strip accents (Latin text only)."""
    s = unicodedata.normalize("NFKD", s.lower())
    return "".join(c for c in s if not unicodedata.combining(c))


# ======================================================================================
# 1. Indic script handling
# ======================================================================================
_SCRIPT_RANGES = [(0x0900, sanscript.DEVANAGARI, "devanagari"), (0x0980, sanscript.BENGALI, "bengali"),
                  (0x0A00, sanscript.GURMUKHI, "gurmukhi"), (0x0A80, sanscript.GUJARATI, "gujarati"),
                  (0x0B00, sanscript.ORIYA, "oriya"), (0x0B80, sanscript.TAMIL, "tamil"),
                  (0x0C00, sanscript.TELUGU, "telugu"), (0x0C80, sanscript.KANNADA, "kannada"),
                  (0x0D00, sanscript.MALAYALAM, "malayalam")]


def detect_script(s):
    for ch in s:
        o = ord(ch)
        if 0x0900 <= o < 0x0E00:
            for lo, sc, name in _SCRIPT_RANGES:
                if lo <= o < lo + 0x80:
                    return sc, name
    return None, "latin"


@lru_cache(maxsize=200_000)
def transliterate_token(tok):
    """Indic token -> Latin approximation tuned on the training pairs."""
    sc, _ = detect_script(tok)
    if sc is None:
        return fold(tok)
    # candra-o (vowel sign and independent vowel) is not handled by IAST
    t = tok.replace("ॉ", "ो").replace("ऑ", "ओ").replace("ૉ", "ો").replace("ઑ", "ઓ")
    t = fold(transliterate(t, sc, sanscript.IAST))
    t = re.sub(r"m(?=[kgcjtdnpbszh])", "n", t)          # anusvara before a consonant
    t = re.sub(r"[^a-z0-9]", "", t)
    return t


_PHON_PAIRS = [("ph", "f"), ("sh", "s"), ("kh", "k"), ("gh", "g"), ("th", "t"), ("dh", "d"),
               ("bh", "b"), ("ch", "c"), ("x", "ks"), ("ck", "k"), ("w", "v"), ("z", "j"), ("q", "k")]
_VOICE = str.maketrans("bdgj", "ptks")


@lru_cache(maxsize=500_000)
def phon(word):
    """Consonant skeleton used to compare transliterated and English words (AUC 0.990)."""
    w = re.sub(r"[^a-z]", "", word)
    if not w:
        return word
    for a, b in _PHON_PAIRS:
        w = w.replace(a, b)
    w = re.sub(r"c(?=[eiy])", "s", w).replace("c", "k")
    w = w.translate(_VOICE)
    w = w[0] + re.sub(r"[aeiouy]", "", w[1:])
    return re.sub(r"(.)\1+", r"\1", w)


# ======================================================================================
# 2. Business names
# ======================================================================================
_PIPE = re.compile(r"\s\|\s.*$")
_ID = re.compile(r"\(?\s*\bid\s*[:#]?\s*\d+\s*\)?", re.I)
_PHONE = re.compile(r"(?:\s[-]\s*)?\+?\d[\d -]{7,}\d")
_HASHNUM = re.compile(r"#\s*\d{3,}")
_ALIAS_STRONG = re.compile(ALIAS_STRONG_RE, re.I)
_ALIAS_WEAK = re.compile(ALIAS_WEAK_RE, re.I)
_DOMAIN = re.compile(r"[a-z0-9]{2,}" + DOMAIN_TLD_RE)
_TLD = re.compile(DOMAIN_TLD_RE)
_WWW = re.compile(r"\bwww\s*\.?")
_HANDLE = re.compile(r"(?:^|\s)[@#][a-z]")
_HONORIFIC = re.compile(r"^\s*(?:m\s*/\s*s|messrs|dr|mr|mrs|smt|sri|shri)\s*\.?\s+(?=\S)")
_FR_ELISION = re.compile(r"\b(?:l|d|j|m|n|s|t|qu)'(?=[a-z])")
_NON_ALNUM = re.compile(r"[^a-z0-9\s]")
_LEET = {"0": "o", "1": "l", "5": "s", "6": "g", "8": "b", "3": "e", "4": "a", "7": "t"}


class NameResources:
    """Data-driven resources (vocabulary, phonetic index, native lexicon)."""

    def __init__(self, vocab, native_lex, eng_phon):
        self.vocab = vocab                      # token -> count (S1 names, train + test)
        self.native_lex = native_lex            # native token -> english token
        self.eng_phon = eng_phon                # phonetic key -> most frequent english token
        self.known = {t for t, c in vocab.items() if c >= 20}
        self.total = sum(vocab.values())
        seg_vocab = {t: c for t, c in vocab.items() if c >= 5 and t.isalpha() and len(t) >= 2}
        tot = sum(seg_vocab.values())
        self.seg_cost = {t: -math.log(c / tot) for t, c in seg_vocab.items()}
        self.seg_maxlen = max(map(len, seg_vocab))


def _brand_segment(t, syl=tuple(sorted(BRAND_SYLLABLES, key=len, reverse=True))):
    @lru_cache(maxsize=None)
    def seg(u):
        if u == "":
            return 0
        if u in ("io", "x"):
            return 0
        for s in syl:
            if u.startswith(s):
                r = seg(u[len(s):])
                if r is not None:
                    return r + 1
        return None
    return seg(t)


@lru_cache(maxsize=300_000)
def is_brand_token(t):
    if not t.isalpha() or len(t) < 5:
        return False
    n = _brand_segment(t)
    return n is not None and n >= 2


class NameNormalizer:
    def __init__(self, res: NameResources):
        self.res = res
        self._fix = lru_cache(maxsize=500_000)(self._fix_token)
        self._segment = lru_cache(maxsize=200_000)(self._segment_concat)
        self._native = lru_cache(maxsize=200_000)(self._native_token)

    # ---------------- native script ----------------
    def _native_token(self, tok):
        w = self.res.native_lex.get(tok)
        if w:
            return w
        lat = transliterate_token(tok)
        lat = re.sub(r"a$", "", lat) if len(lat) > 3 else lat     # final inherent vowel
        # map to an English word only when the phonetic key is long enough to be reliable
        # (>= 5 consonants: 97.6 % precision on held-out lexicon words); shorter keys keep the
        # transliteration, which still matches through name_phon
        key = phon(lat)
        if len(key) >= 5:
            return self.res.eng_phon.get(key, lat)
        return lat

    def native_to_latin(self, s):
        out = []
        for tok in s.split():
            core = tok.strip(".,()[]{}:;-'")
            if not core:
                continue
            out.append(self._native(core) if _INDIC.search(core) else core)
        return " ".join(out)

    # ---------------- token repair (vocabulary guided) ----------------
    def _fix_token(self, t):
        known = self.res.known
        if self.res.vocab.get(t, 0) > 0 or len(t) < 3:     # real Source-1 word: never touch
            return t
        cands = []
        if any(c.isdigit() for c in t) and any(c.isalpha() for c in t):
            base = "".join(_LEET.get(c, c) for c in t)
            cands.append(base)
            if "1" in t:
                cands.append("".join("i" if c == "1" else _LEET.get(c, c) for c in t))
        if "l" in t:
            for i, c in enumerate(t):
                if c == "l":
                    cands.append(t[:i] + "i" + t[i + 1:])
        if "m" in t:
            for i, c in enumerate(t):
                if c == "m":
                    cands.append(t[:i] + "rn" + t[i + 1:])
        best = None
        for c in cands:
            if c in known and (best is None or self.res.vocab[c] > self.res.vocab[best]):
                best = c
        if best:
            return best
        if t.isalpha() and len(t) >= 5:                   # scrambled legal words (pirvate, llmited)
            for target, canon in LEGAL_FUZZY_TARGETS.items():
                if Levenshtein.normalized_similarity(t, target) >= 0.7 and t[0] == target[0]:
                    return target
        return t

    # ---------------- concatenated domain / handle splitting ----------------
    def _segment_concat(self, s):
        cost, mx = self.res.seg_cost, self.res.seg_maxlen
        n = len(s)
        best = [0.0] + [math.inf] * n
        back = [0] * (n + 1)
        for i in range(1, n + 1):
            for j in range(max(0, i - mx), i):
                w = s[j:i]
                c = cost.get(w)
                if c is not None and best[j] + c < best[i]:
                    best[i] = best[j] + c
                    back[i] = j
        if math.isinf(best[n]):
            return None
        out, i = [], n
        while i > 0:
            out.append(s[back[i]:i])
            i = back[i]
        out.reverse()
        return out

    # ---------------- main entry ----------------
    def normalize(self, raw, country):
        s = repair(raw)
        _, script = detect_script(s)
        if script != "latin":
            s = self.native_to_latin(s)
        s = _PIPE.sub(" ", s)
        s = _ID.sub(" ", s)
        s = _PHONE.sub(" ", s)
        s = _HASHNUM.sub(" ", s)

        # alias split: 'Brand DBA Real Name' -> real name on the right
        brand_part = ""
        m = _ALIAS_STRONG.search(s)
        if m and s[:m.start()].strip():
            brand_part, s = s[:m.start()], s[m.end():]
        else:
            m = _ALIAS_WEAK.search(s)
            if m:
                left = re.findall(r"[a-z]+", fold(s[:m.start()]))
                if left and any(is_brand_token(x) for x in left):
                    brand_part, s = s[:m.start()], s[m.end():]

        raw_upper_short = s.strip().isupper() and len(s.strip()) <= 6
        s = fold(s)
        ntype = "normal"
        if _WWW.search(s) or _DOMAIN.search(s):
            ntype = "domain"
        elif _HANDLE.search(s):
            ntype = "handle"
        s = _WWW.sub(" ", s)
        s = _TLD.sub(" ", s)
        if country == "India":
            s = _HONORIFIC.sub("", s)
        s = s.replace("&", " and ").replace("+", " and ")
        if country == "France":
            s = _FR_ELISION.sub(" ", s)
        s = s.replace("'", "")
        s = _NON_ALNUM.sub(" ", s)
        toks = s.split()

        toks = [self._fix(t) for t in toks]
        toks = _merge_single_letters(toks)
        toks = [NAME_VARIANTS.get(t, t) for t in toks]
        toks = [t for i, t in enumerate(toks) if i == 0 or t != toks[i - 1]]   # 'american american'

        legal_set = LEGAL_BY_COUNTRY.get(country, LEGAL_ALL)
        legal, core = [], []
        for i, t in enumerate(toks):
            nxt = toks[i + 1] if i + 1 < len(toks) else ""
            if t in legal_set:
                legal.append(LEGAL[t])
            elif t == "public" and nxt in ("ltd", "limited"):
                legal.append("public")
            elif country == "US" and i > 0 and t in CREDENTIALS:
                legal.append(t)
            elif t in NAME_STOP:
                continue
            else:
                core.append(t)

        if ntype in ("domain", "handle") and len(core) == 1 and len(core[0]) >= 6:
            pieces = []
            tok = core[0]
            if any(c.isdigit() for c in tok):              # leet inside a concatenation
                leet = "".join(_LEET.get(c, c) for c in tok)
                if self._segment(leet):
                    tok = leet
            for run in _ALNUM_RUN.findall(tok):
                seg = self._segment(run) if run.isalpha() and len(run) >= 4 else None
                pieces += seg if seg else [run]
            if len(pieces) >= 2:
                segl = [NAME_VARIANTS.get(t, t) for t in pieces]
                legal += [LEGAL[t] for t in segl if t in legal_set]
                core = [t for t in segl if t not in legal_set and t not in NAME_STOP]

        brand = []
        if brand_part:
            brand = [t for t in _NON_ALNUM.sub(" ", fold(brand_part)).split()
                     if t not in BRAND_SUFFIX_WORDS]
            ntype = "alias"
        elif core and any(is_brand_token(t) for t in core) and \
                all(is_brand_token(t) or t in BRAND_SUFFIX_WORDS for t in core):
            ntype = "brand"
            brand = [t for t in core if t not in BRAND_SUFFIX_WORDS]
        elif ntype == "normal" and len(core) == 1 and raw_upper_short and core[0] not in self.res.known \
                and core[0].isalpha() and 2 <= len(core[0]) <= 5:
            ntype = "acronym"

        return {
            "name_core": " ".join(core),
            "name_legal": " ".join(sorted(set(legal))),
            "name_brand": " ".join(brand),
            "name_type": ntype,
            "name_script": script,
            "name_concat": "".join(core),
            "name_acronym": "".join(t[0] for t in core) if len(core) >= 2 else "",
            "name_phon": " ".join(phon(t) if t.isalpha() else t for t in core),
        }


def _merge_single_letters(toks):
    """'p c' -> 'pc', 'l l c' -> 'llc', 'a k industries' -> 'ak industries'."""
    out, run = [], []
    for t in toks:
        if len(t) == 1 and t.isalpha():
            run.append(t)
            continue
        if run:
            out.append("".join(run))
            run = []
        out.append(t)
    if run:
        out.append("".join(run))
    return out


# ======================================================================================
# 3. Addresses
# ======================================================================================
_STATE_LOOKUP = build_state_lookup()
_STATE_ALL = {k: v for d in _STATE_LOOKUP.values() for k, v in d.items()}
_ADDR_KEEP = re.compile(r"[^a-z0-9#/°\s]")
_FLOOR = re.compile(FLOOR_RE)
_CARE_OF = re.compile(r"\b[cswd]\s*/\s*o\b")
# US unit designators ('Unit K', '# K', 'Unit APARTMENT G', 'Suite 200').  Only a single
# letter or a number (optionally + letter) is accepted as the unit value.
_UNIT_KW = r"(?:unit|apt|apartment|suite|ste|room|rm)"
_UNIT_VAL = r"(\d+(?:/\d+)?[a-z]?|[a-z]\d*)"
_UNIT_PART = re.compile(r"^(?:(?:" + _UNIT_KW + r"|#)\s*)+#?\s*" + _UNIT_VAL + r"$")
_UNIT_INLINE = re.compile(r"\b" + _UNIT_KW + r"(?:\s+" + _UNIT_KW + r")*\s*#?\s*" + _UNIT_VAL + r"\b")
_NUM_PREFIX = re.compile(r"(?:(?<=\s)|^)" + NUMBER_PREFIX_RE + r"\s*[.:#-]*\s*(?=[a-z]?\s*-?\s*\d)")
_HASH = re.compile(r"#+")
_POBOX = re.compile(r"\b(?:pmb|p\s*o\s*box|po\s*box|post\s*box|box|bp)\s*#?\s*\d+[a-z]?\b")
_ORD_GLUE = re.compile(r"\b(\d+)(?:st|nd|rd|th)(?=[a-z]{3,})")
_ORD_SUFFIX = re.compile(r"\b(\d+)\s*(?:st|nd|rd|th)\b")
_LEAD_ZERO = re.compile(r"\b0+(?=\d)")
_ALNUM_RUN = re.compile(r"[a-z]+|\d+")


def _fold_part(p):
    p = fold(p)
    p = p.replace("n°", " no ").replace("°", " ")
    p = _ADDR_KEEP.sub(" ", p.replace("-", " ").replace(".", " ").replace("'", " "))
    return _SPACES.sub(" ", p).strip()


def normalize_address(raw, country):
    s = repair(raw)
    lookup = _STATE_LOOKUP.get(country, _STATE_ALL)
    canon = ADDR_CANON.get(country, {})
    parts = [p.strip() for p in s.split(",")]

    states, kept = [], []      # kept: folded part strings; state-like parts as ("__STATE__", text, code)
    for p in parts:
        if not p:
            continue
        if _INDIC.search(p):
            code = INDIA_NATIVE_STATES.get(p.strip(" '\""))
            if code:
                states.append((len(kept), code.upper(), True))
                kept.append(("__STATE__", "", code.upper()))
            else:
                kept.append(" ".join(transliterate_token(w) for w in p.split()))
            continue
        f = _fold_part(p)
        f_plain = " ".join(f.replace("#", " ").replace("/", " ").split())
        if f_plain in NULL_PARTS:
            continue
        code = lookup.get(f_plain)
        if code:
            states.append((len(kept), code, len(f_plain) <= 3))
            kept.append(("__STATE__", f_plain, code))
            continue
        kept.append(f)

    # choose the state: an explicit code wins, otherwise the last state-like part.  Other
    # state-like parts (e.g. 'New York' next to 'NY', or the city 'Delhi') stay as text.
    state, chosen = "", -1
    if states:
        codes = [x for x in states if x[2]]
        chosen, state, _ = (codes or states)[-1]
        state = STATE_GROUPS.get(state, state)
    parts2 = []
    for i, k in enumerate(kept):
        if isinstance(k, tuple):
            if i != chosen and k[1]:
                parts2.append(k[1])
            continue
        parts2.append(k)

    unit = ""
    words, nums, city, house_no = [], [], "", ""
    for f in parts2:
        f = _CARE_OF.sub(" ", f)                    # 'c/o', 's/o' need the slash
        f = _POBOX.sub(" ", f)                      # mailbox numbers are not house numbers
        if country == "US":                         # units may be written '1/2'
            um = _UNIT_PART.match(f.strip())
            if um:
                unit = unit or um.group(1)
                continue
            um = _UNIT_INLINE.search(f)
            if um:
                unit = unit or um.group(1)
                f = f[:um.start()] + " " + f[um.end():]
        f = _SPACES.sub(" ", f.replace("/", " "))
        f = _ORD_GLUE.sub(r"\1 ", f)               # '2ndfloor' -> '2 floor', '5thcross' -> '5 cross'
        f = _SPACES.sub(" ", _FLOOR.sub(" ", f)).strip()
        f = _NUM_PREFIX.sub(" ", " " + f)
        f = _HASH.sub(" ", f)
        f = _ORD_SUFFIX.sub(r"\1", f)
        f = _LEAD_ZERO.sub("", f)
        toks = []
        for t in _ALNUM_RUN.findall(f):
            t = ORDINAL_WORDS.get(t, t)
            if not t.isdigit():
                t = canon.get(t, t)
                if not t or t in ADDR_STOP:
                    continue
            if toks and toks[-1] == t:            # 'city city', 'unit unit'
                continue
            toks.append(t)
        part_nums = [t for t in toks if t.isdigit()]
        part_words = [t for t in toks if not t.isdigit()]
        if part_nums and not house_no:
            house_no = part_nums[0]
        nums += part_nums
        words += part_words
        if part_words and not part_nums:
            city = " ".join(part_words)
    seen = set()
    nums = [n for n in nums if not (n in seen or seen.add(n))]
    seen = set()
    words = [w for w in words if not (w in seen or seen.add(w))]     # 'Mumbai, Mumbai, Mumbai City'
    return {
        "addr_words": " ".join(words),
        "addr_nums": " ".join(nums),
        "house_no": house_no,
        "unit": unit,
        "city": city,
        "state": state,
        "addr_missing": int(not words and not nums),
    }


# ######################################################################################
# SECTION C - LEARNED RESOURCES (vocabulary, phonetic index, native lexicon)
# ######################################################################################
def basic_tokens(name):
    s = fold(repair(name)).replace("&", " and ").replace("+", " and ").replace("'", "")
    return [NAME_VARIANTS.get(t, t) for t in re.sub(r"[^a-z0-9\s]", " ", s).split()]


def native_tokens(name):
    out = []
    for tok in repair(name).split():
        core = tok.strip(".,()[]{}:;-'")
        if core:
            out.append(core)
    return out


def build_learned_resources(data_dir, out_path, min_count=3, min_purity=0.8):
    """Vocabulary + phonetic index (Source-1 names, no labels) and native lexicon (TRAIN pairs).
    The native lexicon position-aligns native-script names with their Source-1 name (99.998 %
    of native names have the same token count) and keeps words with purity >= min_purity."""
    # ---- vocabulary from Source-1 names (train + test) ----
    vocab, vocab_india = Counter(), Counter()
    for split in ("train", "test"):
        s1 = read_tsv(os.path.join(data_dir, split, f"{split}_source1.tsv"), ["business_name", "country"])
        for n, c in zip(s1.business_name, s1.country):
            toks = set(basic_tokens(n))
            vocab.update(toks)
            if c == "India":
                vocab_india.update(toks)
    print("vocab size", len(vocab), "india vocab size", len(vocab_india))

    # phonetic index for native-script fallback: built from INDIAN Source-1 names so that the
    # Indian-English spelling is recovered ('jewellers', not 'jewelers')
    eng_best = {}
    for t, c in vocab_india.items():
        if c >= 20 and t.isalpha():
            k = phon(t)
            if k not in eng_best or c > vocab_india[eng_best[k]]:
                eng_best[k] = t
    print("phonetic index size", len(eng_best))

    # ---- native lexicon from TRAIN pairs ----
    d = os.path.join(data_dir, "train")
    gt = read_tsv(os.path.join(d, "train_ground_truth.tsv"))
    s1 = read_tsv(os.path.join(d, "train_source1.tsv"), ["entity_id", "business_name", "country"])
    s1 = s1[s1.country != "US"].set_index("entity_id").business_name
    link = {}
    for s1id, ms in zip(gt.source1_entity_id, gt.matched_entity_ids):
        if s1id in s1.index:
            for m in ms.split(","):
                if m:
                    link[m] = s1id
    co = defaultdict(Counter)
    n_pairs = n_eq = 0
    for k in "23":
        src = read_tsv(os.path.join(d, f"train_source{k}.tsv"), ["entity_id", "business_name"])
        src = src[src.business_name.str.contains(_INDIC)]
        for eid, name in zip(src.entity_id, src.business_name):
            s1id = link.get(eid)
            if s1id is None:
                continue
            n_pairs += 1
            nt, et = native_tokens(name), basic_tokens(s1[s1id])
            if len(nt) != len(et):
                continue
            n_eq += 1
            for x, y in zip(nt, et):
                if _INDIC.search(x):
                    co[x][y] += 1
    lex = {}
    for x, c in co.items():
        y, n = c.most_common(1)[0]
        tot = sum(c.values())
        # legal words written in full vs abbreviated are the same canonical legal token
        canon = Counter()
        for yy, nn in c.items():
            canon[LEGAL.get(yy, yy)] += nn
        y2, n2 = canon.most_common(1)[0]
        if tot >= min_count and n2 / tot >= min_purity:
            lex[x] = y if LEGAL.get(y, y) == y2 else y2
    print(f"native pairs {n_pairs}, equal length {n_eq}, lexicon entries {len(lex)} / {len(co)}")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({"vocab": dict(vocab), "eng_phon": eng_best, "native_lex": lex}, f)
    print("saved", out_path)


# ######################################################################################
# SECTION D - PIPELINE: clean every file and write TSV
# ######################################################################################
OUT_COLUMNS = ["entity_id", "country",
               "name_core", "name_legal", "name_brand", "name_type", "name_script",
               "name_concat", "name_acronym", "name_phon",
               "addr_words", "addr_nums", "house_no", "unit", "city", "state", "state_inf", "addr_missing"]
_BAD_CHARS = re.compile(r"[\t\r\n]")
_NN = None


def load_name_normalizer(res_path):
    res = pickle.load(open(res_path, "rb"))
    return NameNormalizer(NameResources(res["vocab"], res["native_lex"], res["eng_phon"]))


def _init_worker(res_path):
    global _NN
    _NN = load_name_normalizer(res_path)


def clean_frame(chunk, nn=None):
    """Raw source rows (entity_id, business_name, business_address, country) -> cleaned rows."""
    nn = nn or _NN
    names = [nn.normalize(n, c) for n, c in zip(chunk.business_name, chunk.country)]
    addrs = [normalize_address(a, c) for a, c in zip(chunk.business_address, chunk.country)]
    out = pd.concat([chunk[["entity_id", "country"]].reset_index(drop=True),
                     pd.DataFrame(names), pd.DataFrame(addrs)], axis=1)
    out["state_inf"] = out["state"]
    out["addr_missing"] = out["addr_missing"].astype(int).astype(str)
    return out[OUT_COLUMNS]


def read_tsv(path, usecols=None, chunksize=None):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE,
                       usecols=usecols, chunksize=chunksize)


def write_tsv(df, path, header):
    for col in OUT_COLUMNS:                                   # a stray tab/newline would corrupt the file
        if df[col].str.contains(_BAD_CHARS).any():
            raise ValueError(f"tab/newline in column {col}")
    df.to_csv(path, sep="\t", index=False, header=header, mode="w" if header else "a",
              quoting=csv.QUOTE_NONE, lineterminator="\n")


def clean_file(pool, src, dst, chunk, city_state=None):
    t0, n = time.time(), 0
    for i, df in enumerate(pool.imap(clean_frame, read_tsv(src, chunksize=chunk))):
        if city_state is not None:
            df["state_inf"] = [st or city_state.get((c, city), "")
                               for c, city, st in zip(df.country, df.city, df.state)]
        write_tsv(df, dst, header=(i == 0))
        n += len(df)
    return n, time.time() - t0


def learn_city_state(paths, min_support=20, min_purity=0.95):
    """(country, city) -> state, from cleaned Source-1 files."""
    cnt = defaultdict(Counter)
    for p in paths:
        s1 = read_tsv(p, usecols=["country", "city", "state"])
        s1 = s1[(s1.city != "") & (s1.state != "")]
        for c, city, st in zip(s1.country, s1.city, s1.state):
            cnt[(c, city)][st] += 1
    out = {}
    for key, c in cnt.items():
        st, n = c.most_common(1)[0]
        tot = sum(c.values())
        if tot >= min_support and n / tot >= min_purity:
            out[key] = st
    return out


def main():
    ap = argparse.ArgumentParser(description="Clean all source files into TSV.")
    ap.add_argument("--data", required=True, help="dataset folder containing train/ and test/")
    ap.add_argument("--out", default=os.path.join(ROOT, "artifacts", "cleaned_dataset"))
    ap.add_argument("--resources", default=os.path.join(ROOT, "artifacts", "resources.pkl"))
    ap.add_argument("--rebuild_resources", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--chunk", type=int, default=25000)
    a = ap.parse_args()

    if a.rebuild_resources or not os.path.exists(a.resources):
        build_learned_resources(a.data, a.resources)

    for split in ("train", "test"):
        os.makedirs(os.path.join(a.out, split), exist_ok=True)
    path = lambda split, k: os.path.join(a.out, split, f"{split}_source{k}.tsv")
    src = lambda split, k: os.path.join(a.data, split, f"{split}_source{k}.tsv")

    with Pool(a.workers, initializer=_init_worker, initargs=(a.resources,)) as pool:
        # 1) Source 1 (needed first: it teaches the city -> state links)
        for split in ("train", "test"):
            n, dt = clean_file(pool, src(split, 1), path(split, 1), a.chunk)
            print(f"{split} source1: {n:,} rows in {dt:,.0f}s", flush=True)
        city_state = learn_city_state([path("train", 1), path("test", 1)])
        pickle.dump(city_state, open(os.path.join(a.out, "city_state.pkl"), "wb"))
        print(f"city -> state links learned from Source 1: {len(city_state):,}", flush=True)
        for split in ("train", "test"):                     # fill Source 1's own missing states
            df = read_tsv(path(split, 1))
            df["state_inf"] = [st or city_state.get((c, city), "")
                               for c, city, st in zip(df.country, df.city, df.state)]
            write_tsv(df, path(split, 1), header=True)
        # 2) Sources 2 and 3
        for split in ("train", "test"):
            for k in (2, 3):
                n, dt = clean_file(pool, src(split, k), path(split, k), a.chunk, city_state)
                print(f"{split} source{k}: {n:,} rows in {dt:,.0f}s", flush=True)

    # 3) integrity report
    for split in ("train", "test"):
        for k in (1, 2, 3):
            ids_in = read_tsv(src(split, k), usecols=["entity_id"]).entity_id
            df = read_tsv(path(split, k), usecols=["entity_id", "state", "state_inf"])
            assert len(df) == len(ids_in) and (df.entity_id.values == ids_in.values).all(), \
                f"{split} source{k}: row order / count mismatch"
            print(f"{split} source{k}: {len(df):,} rows OK | state missing {(df.state == '').mean():.4f} "
                  f"-> after city inference {(df.state_inf == '').mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
