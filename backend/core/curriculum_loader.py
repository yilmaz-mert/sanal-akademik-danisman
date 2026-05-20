"""
Dumb data-loading helpers for curriculum and intibak JSON files.
No matching logic here — raw data only.
"""
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
BACKEND_DIR = Path(__file__).parent.parent

YEAR_CURRICULUM_MAP: Dict[str, Path] = {
    "2021-2022": BACKEND_DIR / "mufredat_21_22.json",
    "2022-2023": BACKEND_DIR / "mufredat_22_23.json",
    "2023-2024": BACKEND_DIR / "mufredat_23_24.json",
    "2024-2025": BACKEND_DIR / "mufredat_24_25.json",
    "2025-2026": BACKEND_DIR / "mufredat_25_26.json",
}

INTIBAK_PATH = BACKEND_DIR / "intibak_kurallari.json"

_year_cache: Dict[str, dict] = {}
_intibak_cache: Optional[dict] = None
_latest_lookup_cache: Optional[Tuple] = None  # (by_code, by_base, by_name)

# ─────────────────────────────────────────────────────────────
# GRADE TABLES  (shared with transcript parser)
# ─────────────────────────────────────────────────────────────
GRADE_POINTS: Dict[str, float] = {
    "AA": 4.00, "BA": 3.50, "BB": 3.00, "CB": 2.50, "CC": 2.00,
    "DC": 1.50, "DD": 1.00, "F": 0.00, "FD": 0.00, "VZ": 0.00, "DZ": 0.00,
}
PASSING_GRADES = {"AA", "BA", "BB", "CB", "CC", "DC", "DD", "BL", "MU", "G", "P"}
FAILED_GRADES = {"F", "FD", "VZ", "DZ", "K", "U"}
ALL_GRADES = PASSING_GRADES | FAILED_GRADES


def is_passing(grade: str) -> bool:
    return grade.upper() in PASSING_GRADES


def is_failed(grade: str) -> bool:
    return grade.upper() in FAILED_GRADES


# ─────────────────────────────────────────────────────────────
# STRING NORMALIZATION (used for fuzzy curriculum matching)
# ─────────────────────────────────────────────────────────────
def normalize_str(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFC", s)
    s = s.replace("İ", "I").replace("ı", "i")
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s.translate(str.maketrans("ğüöşç", "guosc"))


_TR_STOP_WORDS = {
    "icin", "ve", "veya", "bir", "ile", "olan", "ya", "de", "da",
    "mi", "mu", "mü", "gibi", "bu", "o", "en", "her",
}

_SUFFIX_RE = re.compile(r"-[A-Z]$")


def strip_suffix(code: str) -> str:
    return _SUFFIX_RE.sub("", code.strip())


def _token_similarity(name_a: str, name_b: str) -> float:
    def _tokens(s: str) -> set:
        cleaned = re.sub(r"[^\w\s]", " ", normalize_str(s))
        return {t for t in cleaned.split() if t and t not in _TR_STOP_WORDS and len(t) > 1}
    ta, tb = _tokens(name_a), _tokens(name_b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# ─────────────────────────────────────────────────────────────
# LOADERS
# ─────────────────────────────────────────────────────────────
def load_year_curriculum(year_key: str) -> dict:
    if year_key in _year_cache:
        return _year_cache[year_key]
    path = YEAR_CURRICULUM_MAP.get(year_key)
    if path is None or not path.exists():
        path = YEAR_CURRICULUM_MAP.get("2025-2026", BACKEND_DIR / "mufredat_25_26.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    _year_cache[year_key] = data
    return data


def load_latest_curriculum() -> dict:
    try:
        return load_year_curriculum("2025-2026")
    except Exception:
        # Fall back to the first available curriculum
        for year in reversed(list(YEAR_CURRICULUM_MAP.keys())):
            try:
                return load_year_curriculum(year)
            except Exception:
                continue
        return {}


def load_intibak_rules() -> dict:
    global _intibak_cache
    if _intibak_cache is None:
        if INTIBAK_PATH.exists():
            with open(INTIBAK_PATH, encoding="utf-8") as f:
                _intibak_cache = json.load(f)
            print(
                f"[INTIBAK] Loaded {len(_intibak_cache.get('intibak_kurallari', []))} rules "
                f"({len(_intibak_cache.get('eski_yeni_eslestirme', {}))} quick-lookup entries)"
            )
        else:
            _intibak_cache = {}
            print("[INTIBAK] intibak_kurallari.json not found")
    return _intibak_cache


# ─────────────────────────────────────────────────────────────
# LOOKUP HELPERS  (used to map LLM-returned codes to full course objects)
# ─────────────────────────────────────────────────────────────
def build_lookup(dersler: List[dict]) -> Tuple[dict, dict, dict]:
    by_code: Dict[str, dict] = {}
    by_base: Dict[str, dict] = {}
    by_name: Dict[str, dict] = {}
    for c in dersler:
        by_code[c["kod"]] = c
        base = strip_suffix(c["kod"])
        if base not in by_base:
            by_base[base] = c
        norm = normalize_str(c["ad"])
        if norm not in by_name:
            by_name[norm] = c
    return by_code, by_base, by_name


def get_latest_curriculum_lookup() -> Tuple[dict, dict, dict]:
    global _latest_lookup_cache
    if _latest_lookup_cache is None:
        data = load_latest_curriculum()
        _latest_lookup_cache = build_lookup(data.get("dersler", []))
        print(f"[FALLBACK-LOOKUP] Built lookup tables ({len(_latest_lookup_cache[0])} entries)")
    return _latest_lookup_cache


def lookup_course(raw_code: str, name: str,
                  by_code: dict, by_base: dict, by_name: dict) -> Optional[dict]:
    """5-step cascade: exact → stripped → base → exact name → token similarity."""
    stripped = strip_suffix(raw_code)
    curr = by_code.get(raw_code) or by_code.get(stripped) or by_base.get(stripped)
    if curr is None and name:
        curr = by_name.get(normalize_str(name))
    if curr is None and name and by_name:
        for cname, candidate in by_name.items():
            if _token_similarity(name, cname) >= 0.85:
                curr = candidate
                break
    return curr
