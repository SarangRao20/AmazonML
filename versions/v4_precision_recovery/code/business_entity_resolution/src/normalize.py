"""
Text normalization for multilingual entity resolution.

Handles:
- Unicode NFKC normalization (preserves Indic, handles French diacritics)
- Legal suffix expansion and stripping
- Transliteration (Devanagari -> Latin phonetic bridge)
- Address component extraction (street number, postal code)
- Accent stripping via AnyASCII
"""

import re
import unicodedata
from typing import Dict, Tuple

try:
    from anyascii import anyascii
except ImportError:
    print("⚠️  anyascii not installed. Using fallback.")
    def anyascii(text):
        """Fallback if anyascii not available."""
        return text.encode('ascii', 'ignore').decode('ascii')


# ==================== LEGAL SUFFIX MAPPINGS ====================
LEGAL_SUFFIX_EXPANSION = {
    "corp": "corporation",
    "pvt": "private",
    "ltd": "limited",
    "inc": "incorporated",
    "llc": "limited liability company",
    "llp": "limited liability partnership",
    "pllc": "professional limited liability company",
    "plc": "public limited company",
    "co": "company",
    "co.": "company",
}

LEGAL_SUFFIXES_REGEX = r"\b(" + "|".join(LEGAL_SUFFIX_EXPANSION.keys()) + r")\b"

# ==================== DEVANAGARI TRANSLITERATION ====================
# Phonetic mapping from Devanagari to Latin characters
DEVANAGARI_TO_LATIN = {
    # Vowels
    'अ': 'a', 'आ': 'aa', 'इ': 'i', 'ई': 'ee',
    'उ': 'u', 'ऊ': 'uu', 'ऋ': 'ri', 'ॠ': 'ree',
    'ऌ': 'li', 'ॡ': 'lee', 'ए': 'e', 'ऐ': 'ai',
    'ओ': 'o', 'औ': 'au', 'ऑ': 'o',
    
    # Consonants
    'क': 'ka', 'ख': 'kha', 'ग': 'ga', 'घ': 'gha', 'ङ': 'nga',
    'च': 'cha', 'छ': 'chha', 'ज': 'ja', 'झ': 'jha', 'ञ': 'nya',
    'ट': 'ta', 'ठ': 'tha', 'ड': 'da', 'ढ': 'dha', 'ण': 'na',
    'त': 'ta', 'थ': 'tha', 'द': 'da', 'ध': 'dha', 'न': 'na',
    'प': 'pa', 'फ': 'pha', 'ब': 'ba', 'भ': 'bha', 'म': 'ma',
    'य': 'ya', 'र': 'ra', 'ल': 'la', 'व': 'va',
    'श': 'sha', 'ष': 'sha', 'स': 'sa', 'ह': 'ha',
    
    # Nukta variants
    'क़': 'qa', 'ख़': 'kha', 'ग़': 'ga', 'ज़': 'za', 'ड़': 'da', 'ढ़': 'dha', 'फ़': 'fa',
    
    # Special characters
    'ः': 'h', 'ं': 'n',
    
    # Numbers
    '०': '0', '१': '1', '२': '2', '३': '3', '४': '4',
    '५': '5', '६': '6', '७': '7', '८': '8', '९': '9',
}


def normalize_unicode(text: str) -> str:
    """
    Apply NFKC Unicode normalization.
    
    NFKC:
    - Decomposition: converts composed characters to base + combining marks
    - Compatibility: handles variants (e.g., ﬁ -> fi)
    - Mark-safe: preserves Indic vowel signs
    
    Args:
        text: Raw text
    
    Returns:
        NFKC normalized text
    """
    # Handle None/NaN
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize("NFKC", text)


def transliterate_devanagari(text: str) -> str:
    """
    Phonetically transliterate Devanagari to Latin characters.
    
    Creates a bridge for matching Hindi names to English variants:
    "मुंबई" -> "mumbai"
    
    Args:
        text: Text possibly containing Devanagari
    
    Returns:
        Text with Devanagari converted to Latin phonetics
    """
    result = []
    for char in text:
        if char in DEVANAGARI_TO_LATIN:
            result.append(DEVANAGARI_TO_LATIN[char])
        else:
            result.append(char)
    return ''.join(result)


def expand_legal_suffixes(text: str) -> str:
    """
    Expand abbreviated legal suffixes to full forms.
    
    "ACME Corp" -> "ACME corporation"
    "XYZ Pvt Ltd" -> "XYZ private limited"
    
    Args:
        text: Text with abbreviated suffixes
    
    Returns:
        Text with expanded suffixes
    """
    def replace_suffix(match):
        abbr = match.group(1).lower()
        return LEGAL_SUFFIX_EXPANSION.get(abbr, abbr)
    
    return re.sub(LEGAL_SUFFIXES_REGEX, replace_suffix, text, flags=re.IGNORECASE)


def strip_legal_suffix(text: str) -> str:
    """
    Remove legal suffixes from text.
    
    "ACME Corporation" -> "ACME"
    "XYZ Private Limited" -> "XYZ"
    
    Args:
        text: Text with legal suffix
    
    Returns:
        Text with suffix removed
    """
    # Remove expanded forms first
    expanded_forms = [
        "corporation", "private", "limited", "incorporated",
        "limited liability company", "limited liability partnership",
        "professional limited liability company", "public limited company",
        "company",
    ]
    
    for form in expanded_forms:
        # Case-insensitive, word boundary removal
        text = re.sub(rf"\s+{re.escape(form)}\s*$", "", text, flags=re.IGNORECASE)
        text = re.sub(rf"^{re.escape(form)}\s+", "", text, flags=re.IGNORECASE)
    
    return text.strip()


def extract_numeric_signature(text: str) -> str:
    """
    Extract numeric patterns from text (e.g., street numbers, suite numbers).
    
    "123 Main Street, Suite 456" -> "123 456"
    Used as a separate feature for matching precision.
    
    Args:
        text: Address or name text
    
    Returns:
        Space-separated numeric values found
    """
    if not isinstance(text, str):
        return ""
    numbers = re.findall(r'\d+', text)
    return ' '.join(numbers)


def extract_street_number(text: str) -> str:
    """
    Extract leading street number from address.
    
    "456 Oak Lane" -> "456"
    "Suite 789" -> "789"
    
    Args:
        text: Address text
    
    Returns:
        Street number or empty string
    """
    if not isinstance(text, str):
        return ""
    match = re.search(r'^(\d+)', text.strip())
    return match.group(1) if match else ""


def extract_postal_code(text: str) -> str:
    """
    Extract postal code from address.
    
    Supports multiple formats:
    - US: 5 digits (12345)
    - India: 6 digits (123456)
    - France: 5 digits (75001)
    
    Args:
        text: Address text
    
    Returns:
        Postal code or empty string
    """
    if not isinstance(text, str):
        return ""
    # Try 5-6 digit postal codes
    match = re.search(r'\b(\d{5,6})\b', text)
    return match.group(1) if match else ""


def normalize_name(raw_name: str, return_all_variants: bool = False) -> Dict[str, str]:
    """
    Comprehensive name normalization.
    
    Pipeline:
    1. Unicode NFKC normalization
    2. Lowercase
    3. Expand legal suffixes
    4. Transliterate Devanagari -> Latin
    5. AnyASCII transliteration (accent stripping)
    6. Normalize punctuation and whitespace
    7. Strip legal suffixes (create root name)
    
    Args:
        raw_name: Raw business name
        return_all_variants: If True, return all variants; else just normalized
    
    Returns:
        Dict with normalized variants or just normalized name string
    """
    # Step 1-2: Unicode normalization + lowercase
    norm = normalize_unicode(raw_name).lower().strip()
    
    # Step 3: Expand legal suffixes (for better matching before stripping)
    norm = expand_legal_suffixes(norm)
    
    # Step 4: Transliterate Devanagari
    norm = transliterate_devanagari(norm)
    
    # Step 5: AnyASCII for accent stripping (French, Spanish, etc.)
    norm = anyascii(norm)
    
    # Step 6: Normalize punctuation and whitespace
    norm = norm.replace("&", "and")
    norm = norm.replace("'", "")
    norm = norm.replace('"', "")
    norm = re.sub(r'\s+', ' ', norm).strip()
    
    if return_all_variants:
        # Step 7: Create root name (strip suffix)
        root = strip_legal_suffix(norm)
        
        return {
            "normalized": norm,
            "root": root,
            "raw": raw_name,
            "ascii": anyascii(raw_name),
            "ascii_root": strip_legal_suffix(anyascii(raw_name)),
        }
    else:
        return norm


def normalize_address(raw_address: str, return_components: bool = False) -> Dict[str, str]:
    """
    Comprehensive address normalization.
    
    Pipeline:
    1. Unicode NFKC normalization
    2. Lowercase
    3. Extract street number (structured feature)
    4. Extract postal code (structured feature)
    5. Transliterate Devanagari
    6. AnyASCII for accent stripping
    7. Expand street abbreviations
    8. Normalize punctuation and whitespace
    
    Args:
        raw_address: Raw address string
        return_components: If True, return street number, postal code separately
    
    Returns:
        Dict with normalized address and optionally components
    """
    # Step 1-2: Unicode normalization + lowercase
    norm = normalize_unicode(raw_address).lower().strip()
    
    # Step 3: Extract street number
    street_num = extract_street_number(norm)
    
    # Step 4: Extract postal code
    postal = extract_postal_code(norm)
    
    # Step 5: Transliterate Devanagari
    norm = transliterate_devanagari(norm)
    
    # Step 6: AnyASCII for accent stripping
    norm = anyascii(norm)
    
    # Step 7: Expand street abbreviations
    street_abbr = {
        "rd": "road",
        "st": "street",
        "ave": "avenue",
        "blvd": "boulevard",
        "ln": "lane",
        "dr": "drive",
        "ct": "court",
        "pl": "place",
        "ter": "terrace",
        "pkwy": "parkway",
    }
    
    for abbr, full in street_abbr.items():
        norm = re.sub(rf"\b{abbr}\b", full, norm, flags=re.IGNORECASE)
    
    # Step 8: Normalize punctuation and whitespace
    norm = norm.replace("&", "and")
    norm = re.sub(r'\s+', ' ', norm).strip()
    
    if return_components:
        return {
            "normalized": norm,
            "street_num": street_num,
            "postal_code": postal,
            "raw": raw_address,
        }
    else:
        return norm


def normalize_text_pair(s1_name: str, s1_addr: str, s2_or_s3_name: str, 
                        s2_or_s3_addr: str) -> Dict[str, Dict]:
    """
    Normalize both sides of a candidate pair for feature extraction.
    
    Args:
        s1_name: Source 1 business name
        s1_addr: Source 1 business address
        s2_or_s3_name: Source 2/3 business name
        s2_or_s3_addr: Source 2/3 business address
    
    Returns:
        Dict with normalized components for both sides
    """
    s1_norm = {
        "name": normalize_name(s1_name),
        "address": normalize_address(s1_addr),
    }
    
    s2_s3_norm = {
        "name": normalize_name(s2_or_s3_name),
        "address": normalize_address(s2_or_s3_addr),
    }
    
    return {
        "s1": s1_norm,
        "s2_s3": s2_s3_norm,
    }


if __name__ == "__main__":
    # Test normalization
    print("=" * 80)
    print("Testing Text Normalization")
    print("=" * 80)
    
    # Test English
    print("\n[ENGLISH]")
    test_name = "ACME Corporation"
    print(f"Input: {test_name}")
    print(f"Output: {normalize_name(test_name)}")
    
    # Test with legal suffix
    print("\n[LEGAL SUFFIX]")
    test_name = "Rocky Mountain Industries, Inc."
    print(f"Input: {test_name}")
    print(f"Output: {normalize_name(test_name)}")
    
    # Test Hindi (Devanagari) - simulated
    print("\n[DEVANAGARI TRANSLITERATION]")
    test_name = "मुंबई विद्यालय"  # Mumbai Vidyalaya
    print(f"Input: {test_name}")
    print(f"Output: {normalize_name(test_name)}")
    
    # Test French
    print("\n[FRENCH DIACRITICS]")
    test_name = "Crème Brûlée Café"
    print(f"Input: {test_name}")
    print(f"Output: {normalize_name(test_name)}")
    
    # Test address
    print("\n[ADDRESS]")
    test_addr = "456 Oak Lane, Suite 789, New York, NY 10001"
    print(f"Input: {test_addr}")
    result = normalize_address(test_addr, return_components=True)
    print(f"Normalized: {result['normalized']}")
    print(f"Street Number: {result['street_num']}")
    print(f"Postal Code: {result['postal_code']}")
    
    print("\n✅ Normalization tests completed!")
