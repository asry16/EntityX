"""
Text and Address Preprocessing & Normalization.
Handles multilingual accents, Indic script transliteration (Devanagari, Tamil, Kannada, Telugu, Bengali),
legal suffixes, address abbreviations, phone numbers, and entity token extraction.
"""

import re
import unicodedata
from typing import List, Set, Tuple, Optional


# US State to 2-letter abbreviation mapping
US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv", "new hampshire": "nh",
    "new jersey": "nj", "new mexico": "nm", "new york": "ny", "north carolina": "nc",
    "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa",
    "rhode island": "ri", "south carolina": "sc", "south dakota": "sd", "tennessee": "tn",
    "texas": "tx", "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa",
    "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy", "district of columbia": "dc"
}

# India State to abbreviation mapping
INDIA_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr",
    "himachal pradesh": "hp", "jharkhand": "jh", "karnataka": "ka", "kerala": "kl",
    "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml",
    "mizoram": "mz", "nagaland": "nl", "odisha": "or", "punjab": "pb", "rajasthan": "rj",
    "sikkim": "sk", "tamil nadu": "tn", "telangana": "tg", "tripura": "tr",
    "uttar pradesh": "up", "uttarakhand": "uk", "west bengal": "wb", "delhi": "dl"
}

# Common street type abbreviations
STREET_MAP = {
    "st": "street", "saint": "street", "rd": "road", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "dr": "drive", "ln": "lane", "ct": "court", "ter": "terrace",
    "hwy": "highway", "pkwy": "parkway", "pl": "place", "sq": "square", "cir": "circle",
    "fl": "floor", "apt": "apartment", "ste": "suite", "bldg": "building", "dept": "department",
    "r": "rue", "bd": "boulevard", "bld": "boulevard"
}

# Legal suffixes to normalize or strip for core name matching
LEGAL_TERMS = {
    "inc", "incorporated", "corp", "corporation", "llc", "pllc", "llp", "ltd", "limited",
    "pvt", "private", "co", "company", "enterprises", "solutions", "services",
    "group", "holdings", "partners", "center", "consultants", "associates",
    "industries", "ventures", "management", "international", "technologies",
    "elelpi", "sarl", "sas", "sa", "sci", "eurl", "snc"
}

# Name prefixes to strip (articles and honorifics)
NAME_PREFIX_STOPWORDS = {"the", "mr", "mrs", "ms", "shri", "smt", "dr", "m/s", "messrs"}

# Indic transliteration tables
_DEVANAGARI_MAP = {
    0x0905: 'a', 0x0906: 'a', 0x0907: 'i', 0x0908: 'i', 0x0909: 'u', 0x090A: 'u',
    0x090D: 'e', 0x090E: 'e', 0x090F: 'e', 0x0910: 'ai', 0x0911: 'o', 0x0912: 'o', 0x0913: 'o', 0x0914: 'au',
    0x0915: 'k', 0x0916: 'kh', 0x0917: 'g', 0x0918: 'gh', 0x0919: 'n',
    0x091A: 'ch', 0x091B: 'chh', 0x091C: 'j', 0x091D: 'jh', 0x091E: 'n',
    0x091F: 't', 0x0920: 'th', 0x0921: 'd', 0x0922: 'dh', 0x0923: 'n',
    0x0924: 't', 0x0925: 'th', 0x0926: 'd', 0x0927: 'dh', 0x0928: 'n',
    0x092A: 'p', 0x092B: 'ph', 0x092C: 'b', 0x092D: 'bh', 0x092E: 'm',
    0x092F: 'y', 0x0930: 'r', 0x0932: 'l', 0x0935: 'v', 0x0936: 'sh',
    0x0937: 'sh', 0x0938: 's', 0x0939: 'h',
    0x093E: 'a', 0x093F: 'i', 0x0940: 'i', 0x0941: 'u', 0x0942: 'u',
    0x0945: 'e', 0x0946: 'e', 0x0947: 'e', 0x0948: 'ai', 0x0949: 'o', 0x094A: 'o', 0x094B: 'o', 0x094C: 'au',
    0x0902: 'n', 0x0901: 'n', 0x094D: ''
}

_TAMIL_MAP = {
    0x0B85: 'a', 0x0B86: 'a', 0x0B87: 'i', 0x0B88: 'i', 0x0B89: 'u', 0x0B8A: 'u',
    0x0B8E: 'e', 0x0B8F: 'e', 0x0B90: 'ai', 0x0B92: 'o', 0x0B93: 'o', 0x0B94: 'au',
    0x0B95: 'k', 0x0B99: 'ng', 0x0B9A: 's', 0x0B9C: 'j', 0x0B9E: 'n', 0x0B9F: 't',
    0x0BA3: 'n', 0x0BA4: 'th', 0x0BA8: 'n', 0x0BA9: 'n', 0x0BAA: 'p', 0x0BAE: 'm',
    0x0BAF: 'y', 0x0BB0: 'r', 0x0BB1: 'r', 0x0BB2: 'l', 0x0BB3: 'l', 0x0BB4: 'zh',
    0x0BB5: 'v', 0x0BB6: 'sh', 0x0BB7: 'sh', 0x0BB8: 's', 0x0BB9: 'h',
    0x0BBE: 'a', 0x0BBF: 'i', 0x0BC0: 'i', 0x0BC1: 'u', 0x0BC2: 'u', 0x0BC6: 'e',
    0x0BC7: 'e', 0x0BC8: 'ai', 0x0BCA: 'o', 0x0BCB: 'o', 0x0BCC: 'au', 0x0BCD: ''
}

_KANNADA_MAP = {
    0x0C85: 'a', 0x0C86: 'a', 0x0C87: 'i', 0x0C88: 'i', 0x0C89: 'u', 0x0C8A: 'u',
    0x0C8E: 'e', 0x0C8F: 'e', 0x0C90: 'ai', 0x0C92: 'o', 0x0C93: 'o', 0x0C94: 'au',
    0x0C95: 'k', 0x0C96: 'kh', 0x0C97: 'g', 0x0C98: 'gh', 0x0C99: 'n',
    0x0C9A: 'ch', 0x0C9B: 'chh', 0x0C9C: 'j', 0x0C9D: 'jh', 0x0C9E: 'n',
    0x0C9F: 't', 0x0CA0: 'th', 0x0CA1: 'd', 0x0CA2: 'dh', 0x0CA3: 'n',
    0x0CA4: 't', 0x0CA5: 'th', 0x0CA6: 'd', 0x0CA7: 'dh', 0x0CA8: 'n',
    0x0CAA: 'p', 0x0CAB: 'ph', 0x0CAC: 'b', 0x0CAD: 'bh', 0x0CAE: 'm',
    0x0CAF: 'y', 0x0CB0: 'r', 0x0CB2: 'l', 0x0CB5: 'v', 0x0CB6: 'sh',
    0x0CB7: 'sh', 0x0CB8: 's', 0x0CB9: 'h', 0x0CBE: 'a', 0x0CBF: 'i',
    0x0CC0: 'i', 0x0CC1: 'u', 0x0CC2: 'u', 0x0CC6: 'e', 0x0CC7: 'e',
    0x0CC8: 'ai', 0x0CCA: 'o', 0x0CCB: 'o', 0x0CCC: 'au', 0x0CCD: ''
}

_TELUGU_MAP = {
    0x0C05: 'a', 0x0C06: 'a', 0x0C07: 'i', 0x0C08: 'i', 0x0C09: 'u', 0x0C0A: 'u',
    0x0C0E: 'e', 0x0C0F: 'e', 0x0C10: 'ai', 0x0C12: 'o', 0x0C13: 'o', 0x0C14: 'au',
    0x0C15: 'k', 0x0C16: 'kh', 0x0C17: 'g', 0x0C18: 'gh', 0x0C19: 'n',
    0x0C1A: 'ch', 0x0C1B: 'chh', 0x0C1C: 'j', 0x0C1D: 'jh', 0x0C1E: 'n',
    0x0C1F: 't', 0x0C20: 'th', 0x0C21: 'd', 0x0C22: 'dh', 0x0C23: 'n',
    0x0C24: 't', 0x0C25: 'th', 0x0C26: 'd', 0x0C27: 'dh', 0x0C28: 'n',
    0x0C2A: 'p', 0x0C2B: 'ph', 0x0C2C: 'b', 0x0C2D: 'bh', 0x0C2E: 'm',
    0x0C2F: 'y', 0x0C30: 'r', 0x0C32: 'l', 0x0C35: 'v', 0x0C36: 'sh',
    0x0C37: 'sh', 0x0C38: 's', 0x0C39: 'h', 0x0C3E: 'a', 0x0C3F: 'i',
    0x0C40: 'i', 0x0C41: 'u', 0x0C42: 'u', 0x0C46: 'e', 0x0C47: 'e',
    0x0C48: 'ai', 0x0C4A: 'o', 0x0C4B: 'o', 0x0C4C: 'au', 0x0C4D: ''
}

_BENGALI_MAP = {
    0x0985: 'a', 0x0986: 'a', 0x0987: 'i', 0x0988: 'i', 0x0989: 'u', 0x098A: 'u',
    0x098F: 'e', 0x0990: 'ai', 0x0993: 'o', 0x0994: 'au',
    0x0995: 'k', 0x0996: 'kh', 0x0997: 'g', 0x0998: 'gh', 0x0999: 'n',
    0x099A: 'ch', 0x099B: 'chh', 0x099C: 'j', 0x099D: 'jh', 0x099E: 'n',
    0x099F: 't', 0x09A0: 'th', 0x09A1: 'd', 0x09A2: 'dh', 0x09A3: 'n',
    0x09A4: 't', 0x09A5: 'th', 0x09A6: 'd', 0x09A7: 'dh', 0x09A8: 'n',
    0x09AA: 'p', 0x09AB: 'ph', 0x09AC: 'b', 0x09AD: 'bh', 0x09AE: 'm',
    0x09AF: 'y', 0x09B0: 'r', 0x09B2: 'l', 0x09B6: 'sh', 0x09B7: 'sh',
    0x09B8: 's', 0x09B9: 'h', 0x09BE: 'a', 0x09BF: 'i', 0x09C0: 'i',
    0x09C1: 'u', 0x09C2: 'u', 0x09C7: 'e', 0x09C8: 'ai', 0x09CB: 'o',
    0x09CC: 'au', 0x09CD: ''
}

# Merge all Indic transliteration maps
_ALL_INDIC_MAP = {**_DEVANAGARI_MAP, **_TAMIL_MAP, **_KANNADA_MAP, **_TELUGU_MAP, **_BENGALI_MAP}


def transliterate_indic(text: str) -> str:
    """Transliterate Devanagari, Tamil, Kannada, Telugu, Bengali unicode to Latin phonetics."""
    if not text:
        return ""
    return "".join(_ALL_INDIC_MAP.get(ord(c), c) for c in text)


def remove_accents(text: str) -> str:
    """Normalize unicode and strip diacritical accents (e.g., é -> e, à -> a)."""
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def clean_text(text: str) -> str:
    """Basic cleaning, lowercasing, transliterating Indic scripts, and whitespace collapse."""
    if not text or not isinstance(text, str):
        return ""
    text = transliterate_indic(text)
    text = remove_accents(text).lower()
    text = re.sub(r"\bnull\b|\b<null>\b", " ", text)
    text = re.sub(r"[<>[\]()\{\}\"\']", " ", text)
    text = re.sub(r"[-_/.,:;+*&#@!?~`|\\]+", " ", text)
    return " ".join(text.split())


def clean_business_name(name: str) -> str:
    """Clean business name, remove URL domains, phone numbers, DBA, and leading stop prefixes."""
    if not name or not isinstance(name, str):
        return ""
    # Remove M/s, C/o, S/o, D/o prefixes
    name = re.sub(r"\b[mM]\s*/?\s*[sS]\b|\b[cC]\s*/?\s*[oO]\b|\b[sS]\s*/?\s*[oO]\b|\b[dD]\s*/?\s*[oO]\b", " ", name)
    text = clean_text(name)
    text = re.sub(r"\bwww\b|\.com\b|\.org\b|\.in\b|\.net\b|\.co\b|\.fr\b", " ", text)
    text = re.sub(r"\b\d{7,12}\b", " ", text)
    text = re.sub(r"\bd\s*b\s*a\b", " ", text)
    
    tokens = text.split()
    # Strip leading honorifics / articles (e.g., The Scott -> Scott, Mr Prabhav -> Prabhav)
    while tokens and tokens[0] in NAME_PREFIX_STOPWORDS:
        tokens = tokens[1:]
    return " ".join(tokens)


def get_core_name_tokens(clean_name: str) -> List[str]:
    """Extract significant core name tokens (omitting generic corporate suffixes)."""
    tokens = clean_name.split()
    core = [t for t in tokens if t not in LEGAL_TERMS]
    return core if core else tokens


def get_name_blocking_keys(clean_name: str) -> List[str]:
    """Generate high-yield blocking keys for business name."""
    core = get_core_name_tokens(clean_name)
    keys = []
    if not core:
        return keys
    
    first = core[0]
    # 1. First core token
    keys.append(f"n1:{first}")
    if len(first) >= 4:
        keys.append(f"np4:{first[:4]}")
        keys.append(f"np3:{first[:3]}")
            
    # 2. First two core tokens joined
    if len(core) >= 2:
        keys.append(f"n2:{core[0]}_{core[1]}")
        
    # 3. If there is a second distinct unique word in core (length >= 4)
    for tok in core[1:3]:
        if len(tok) >= 4 and tok != first:
            keys.append(f"nt:{tok}")
            break
            
    return keys


def extract_address_numbers(address: str) -> List[str]:
    """Extract numbers from address with normalized leading zeros and prefix handling."""
    if not address or not isinstance(address, str):
        return []
    raw_nums = re.findall(r"\b(?:[a-zA-Z]{1,2}-)?\d+[a-zA-Z]?(?:-\d+)?\b", address.lower())
    cleaned_nums = []
    for num in raw_nums:
        s = num.replace("-", "")
        # Strip leading zeros even with prefix letters e.g. af-0684 -> af684, 0026 -> 26
        norm_num = re.sub(r"(^[a-z]*)0+(\d+)", r"\1\2", s)
        if len(norm_num) >= 1 and norm_num not in {"null", "00", "0"}:
            cleaned_nums.append(norm_num)
    return list(dict.fromkeys(cleaned_nums))


def clean_address(address: str, country: str = "") -> str:
    """Normalize address components, street types, and state codes."""
    if not address or not isinstance(address, str):
        return ""
    text = clean_text(address)
    tokens = text.split()
    normalized_tokens = []
    for t in tokens:
        if t in STREET_MAP:
            normalized_tokens.append(STREET_MAP[t])
        elif country == "US" and t in US_STATES:
            normalized_tokens.append(US_STATES[t])
        elif country == "India" and t in INDIA_STATES:
            normalized_tokens.append(INDIA_STATES[t])
        else:
            normalized_tokens.append(t)
    return " ".join(normalized_tokens)


def get_address_blocking_keys(clean_addr: str, raw_addr: str, country: str = "") -> List[str]:
    """Generate high-precision address blocking keys for top address numbers + tokens."""
    keys = []
    nums = extract_address_numbers(raw_addr)
    tokens = [t for t in clean_addr.split() if len(t) >= 4 and t not in STREET_MAP.values()]
    
    if nums:
        # Check top 3 numbers in address (handles apartment/unit first vs street number first)
        for num in nums[:3]:
            # Number + top significant address words (e.g., 85_wayne, 1400_greatwolf)
            for tok in tokens[:2]:
                if not tok.isdigit():
                    keys.append(f"an:{num}_{tok}")
            # Standalone distinct 5 or 6 digit postal/PIN code
            if len(num) in (5, 6) and num.isdigit():
                keys.append(f"zip:{num}")
                
    return keys
