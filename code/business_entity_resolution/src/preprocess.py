"""High-performance text preprocessing and normalization for entity resolution."""

import re
import unicodedata
from typing import List, Set, Tuple

# Domain suffixes to strip from business names that are website URLs
DOMAIN_RE = re.compile(r'^(?:https?:\/\/)?(?:www\.)?([a-zA-Z0-9\-]+)(?:\.[a-zA-Z]{2,})+(?:\/.*)?$')

# Common legal suffixes across US, India, France
LEGAL_SUFFIXES = {
    'inc', 'incorporated', 'corp', 'corporation', 'llc', 'ltd', 'limited',
    'pvt', 'private', 'llp', 'co', 'company', 'enterprises', 'enterprise',
    'services', 'solutions', 'technologies', 'holdings', 'group',
    # French legal forms
    'sarl', 'sas', 'sasu', 'eurl', 'sa', 'snc', 'gie', 'association',
    # Common words that add noise
    'the', 'and', 'of'
}

# Common street/address abbreviations
STREET_ABBREVIATIONS = {
    'rd': 'road',
    'st': 'street',
    'ave': 'avenue',
    'av': 'avenue',
    'blvd': 'boulevard',
    'bd': 'boulevard',
    'dr': 'drive',
    'ln': 'lane',
    'hwy': 'highway',
    'sq': 'square',
    'ct': 'court',
    'pl': 'place',
    'ste': 'suite',
    'apt': 'apartment',
    'fl': 'floor',
    'bldg': 'building',
    'pkwy': 'parkway',
    'rue': 'rue',
    'chem': 'chemin'
}

PUNCT_RE = re.compile(r'[^\w\s]')
MULTISPACE_RE = re.compile(r'\s+')
DIGITS_RE = re.compile(r'\b\d+\b')


def normalize_text(text: str) -> str:
    """Normalize text: NFKD decomposition (strips accents), lowercase, clean punctuation."""
    if not text:
        return ""
    # Strip accents / diacritics (e.g. é -> e, ü -> u)
    text = unicodedata.normalize('NFKD', text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    # If it's a domain name (e.g., maurewilliamscolombier.com), extract domain part
    m = DOMAIN_RE.match(text)
    if m:
        text = m.group(1).replace('-', ' ')
    text = PUNCT_RE.sub(' ', text)
    text = MULTISPACE_RE.sub(' ', text).strip()
    return text


def get_core_name(normalized_name: str) -> str:
    """Strip common legal suffixes from normalized business name."""
    tokens = normalized_name.split()
    filtered = [t for t in tokens if t not in LEGAL_SUFFIXES]
    return " ".join(filtered) if filtered else normalized_name


def get_tokens(text: str, min_len: int = 2) -> Set[str]:
    """Extract word tokens from text with minimum length threshold."""
    if not text:
        return set()
    return {w for w in text.split() if len(w) >= min_len}


def get_char_ngrams(text: str, n: int = 3) -> Set[str]:
    """Extract character n-grams from text."""
    if not text or len(text) < n:
        return {text} if text else set()
    cleaned = text.replace(' ', '')
    return {cleaned[i:i+n] for i in range(len(cleaned) - n + 1)}


def get_numbers(address: str) -> Set[str]:
    """Extract distinct numeric tokens from address (building #, pin code, etc)."""
    if not address:
        return set()
    return set(DIGITS_RE.findall(address))
