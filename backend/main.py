import asyncio
import operator
import os
import json
import re
import uuid
import unicodedata
from pathlib import Path
from typing import Annotated, Literal, Optional, Dict, List, TypedDict

from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langgraph.types import Send

# ─────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────
app = FastAPI(title="Sanal Akademik Danışman API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CURRICULUM_PATH = os.path.join("..", "mufredat_tam_chunked.json")

# ─────────────────────────────────────────────────────────────
# HISTORICAL CURRICULUM PATHS
# ─────────────────────────────────────────────────────────────
BACKEND_DIR = Path(__file__).parent
YEAR_CURRICULUM_MAP: Dict[str, Path] = {
    "2021-2022": BACKEND_DIR / "mufredat_21_22.json",
    "2022-2023": BACKEND_DIR / "mufredat_22_23.json",
    "2023-2024": BACKEND_DIR / "mufredat_23_24.json",
    "2024-2025": BACKEND_DIR / "mufredat_24_25.json",
    "2025-2026": BACKEND_DIR / "mufredat_25_26.json",
}
_year_cache: Dict[str, dict] = {}
# Lazy-loaded lookup tables for the latest curriculum (global fallback)
_latest_lookup_cache: Optional[tuple] = None  # (by_code, by_base, by_name)

# ─────────────────────────────────────────────────────────────
# INTIBAK (EQUIVALENCY) RULES  backend/intibak_kurallari.json
# ─────────────────────────────────────────────────────────────
INTIBAK_PATH = BACKEND_DIR / "intibak_kurallari.json"
_intibak_cache: Optional[dict] = None


def load_intibak_rules() -> dict:
    """Load and cache intibak equivalency rules from JSON."""
    global _intibak_cache
    if _intibak_cache is None:
        if INTIBAK_PATH.exists():
            with open(INTIBAK_PATH, encoding="utf-8") as f:
                _intibak_cache = json.load(f)
            print(f"[INTIBAK] Loaded {len(_intibak_cache.get('intibak_kurallari', []))} rules "
                  f"({len(_intibak_cache.get('eski_yeni_eslestirme', {}))} quick-lookup entries)")
        else:
            _intibak_cache = {}
            print("[INTIBAK] intibak_kurallari.json not found — equivalency review disabled")
    return _intibak_cache

# ─────────────────────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────────────────────
chat_model = ChatOllama(model="llama3.1:8b", temperature=0.1, num_ctx=4096)

# ─────────────────────────────────────────────────────────────
# SESSIONS & CHECKPOINTER
# ─────────────────────────────────────────────────────────────
sessions: Dict[str, Dict] = {}
checkpointer = MemorySaver()


def get_session(sid: str) -> Dict:
    if sid not in sessions:
        sessions[sid] = {"transcript": None}
    return sessions[sid]


# ─────────────────────────────────────────────────────────────
# CURRICULUM (cached)
# ─────────────────────────────────────────────────────────────
_curriculum_cache: Optional[dict] = None


def load_curriculum() -> dict:
    global _curriculum_cache
    if _curriculum_cache is None:
        with open(CURRICULUM_PATH, "r", encoding="utf-8") as f:
            _curriculum_cache = json.load(f)
    return _curriculum_cache


def load_year_curriculum(year_key: str) -> dict:
    """Load a year-specific curriculum; falls back to 2025-2026 if file missing."""
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
    """Load the most recent historical curriculum (2025-2026)."""
    try:
        return load_year_curriculum("2025-2026")
    except Exception:
        return load_curriculum()


# ─────────────────────────────────────────────────────────────
# STRING NORMALIZATION
# ─────────────────────────────────────────────────────────────
def normalize_str(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFC", s)
    # Handle Turkish I/İ BEFORE lower() — Python's "İ".lower() yields "i̇" (combining dot)
    s = s.replace("İ", "I").replace("ı", "i")
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    # Map remaining Turkish lowercase letters to ASCII equivalents
    return s.translate(str.maketrans("ğüöşç", "guosc"))


# ─────────────────────────────────────────────────────────────
# SUFFIX NORMALIZATION
# ─────────────────────────────────────────────────────────────
_SUFFIX_RE = re.compile(r"-[A-Z]$")


def strip_suffix(code: str) -> str:
    """Strip trailing section suffix (-A, -B, …) from a course code."""
    return _SUFFIX_RE.sub("", code.strip())


# ─────────────────────────────────────────────────────────────
# GRADE SYSTEM
# ─────────────────────────────────────────────────────────────
GRADE_POINTS: Dict[str, float] = {
    "AA": 4.00, "BA": 3.50, "BB": 3.00, "CB": 2.50, "CC": 2.00,
    "DC": 1.50, "DD": 1.00, "F": 0.00, "FD": 0.00, "VZ": 0.00, "DZ": 0.00,
}
# Grades that satisfy a course requirement (DC/DD technically pass at course level)
PASSING_GRADES = {"AA", "BA", "BB", "CB", "CC", "DC", "DD", "BL", "MU", "G", "P"}
# Grades that require retake
FAILED_GRADES = {"F", "FD", "VZ", "DZ", "K", "U"}
# All valid letter grades
ALL_GRADES = PASSING_GRADES | FAILED_GRADES


def is_passing(grade: str) -> bool:
    return grade.upper() in PASSING_GRADES


def is_failed(grade: str) -> bool:
    return grade.upper() in FAILED_GRADES


# ─────────────────────────────────────────────────────────────
# TXT ENCODING HELPER
# ─────────────────────────────────────────────────────────────
def _decode_txt(raw: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "iso-8859-9", "cp1254", "windows-1254", "latin-1"):
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", errors="replace")


# ─────────────────────────────────────────────────────────────
# TXT TRANSCRIPT PARSER
# ─────────────────────────────────────────────────────────────
# Semester header — matches many common variants:
#   "2021 - 2022 Güz Yarıyılı"          (with Yarıyılı)
#   "2021 - 2022 Güz Dönemi"            (with Dönemi)
#   "2021 - 2022 Güz"                   (bare)
#   "2021-2022 Eğitim-Öğretim Yılı Güz Yarıyılı"
SEM_HEADER_RE = re.compile(
    r"(\d{4})\s*[-–—]\s*(\d{4})"
    r"(?:\s+E[gğ]itim[- ]?[ÖO][gğ]retim\s+Y[ıi]l[ıi])?"
    r"\s+(Güz|Bahar|Yaz|GÜZ|BAHAR|YAZ|Guz|SPRING|FALL|SUMMER)"
    r"(?:\s+(?:Yarıyılı?|Dönemi?|D[öo]nemi?|Semester|Term|Y[ıi]l[ıi]))?",
    re.IGNORECASE | re.UNICODE,
)

TERM_GPA_RE = re.compile(
    r"(?:ANO|Yar[iı]y[iı]l\s+Ortalamas[iı]|D[öo]nem\s+Ortalamas[iı]|D[öo]nem\s+GNO|Yar[iı]y[iı]l\s+GNO)"
    r"\s*[:\s]?\s*([\d.,]+)",
    re.IGNORECASE,
)
# Bare "Yarıyıl  2.47" pattern — only catches values in valid GPA range (0–4)
_TERM_GPA_BARE_RE = re.compile(r"Yar[iı]y[iı]l\s+([\d.,]+)", re.IGNORECASE)

CUMULATIVE_GPA_RE = re.compile(
    r"(?:Genel\s+GNO|CGPA|Genel\s+Ortalama|Kümülatif\s+Ortalama|^Genel)"
    r"\s*[:\s]\s*([\d.,]+)",
    re.IGNORECASE | re.MULTILINE,
)

# ─────────────────────────────────────────────────────────────
# NEURO-SYMBOLIC SCHEMA DISCOVERY
# Stage 1: LLM (+ deterministic heuristic) infers column layout ONCE per file.
# Stage 2: Python uses the cached schema to parse every course line — fast.
# ─────────────────────────────────────────────────────────────

# Course code: 2–6 uppercase letters, optional space, 3–4 digits, optional trailing letter.
# Handles "BLM101", "BLM 101", "YAZGE102", "ING106", "CS3001A", etc.
COURSE_CODE_RE = re.compile(r"^([A-Z]{2,6})\s?(\d{3,4}[A-Z]?)\b")

_STATUS_WORDS = {"gecti", "kaldi", "kalti", "basarili", "basarisiz", "passed", "failed"}
_TYPE_TOKENS  = {"Z", "S", "Y", "SE", "ZE", "ME", "SE1", "SE2", "T"}

_SKIP_LINE_RE = re.compile(
    r"Kodu|Ders\s*Ad|Not\s*u|Kredi|AKTS|Ortalama|ANO|T\s+U\s+L|Yarıyıl|D[öo]nem\s+No",
    re.IGNORECASE | re.UNICODE,
)


class TranscriptSchemaConfig(BaseModel):
    """
    Structural schema describing the column layout of a transcript's course lines.

      after_numbers  (default — most Turkish universities):
          CODE  NAME  [TYPE]  [T U L]  CREDIT  ECTS  GRADE  [STATUS]

      before_numbers  (grade appears in the middle, before CREDIT/ECTS):
          CODE  NAME  [TYPE]  [T U L]  GRADE  CREDIT  ECTS  [STATUS]
    """

    grade_position: Literal["after_numbers", "before_numbers"] = Field(
        default="after_numbers",
        description=(
            "'after_numbers': grade token comes AFTER the numeric credit/ECTS columns. "
            "'before_numbers': grade token comes BEFORE credit/ECTS (in the middle of the row)."
        ),
    )
    has_status_word: bool = Field(
        default=True,
        description="True if a status word (Geçti/Kaldı) follows the grade token.",
    )
    has_type_column: bool = Field(
        default=True,
        description="True if a single-letter type code (Z/S) precedes the numeric columns.",
    )
    has_row_number: bool = Field(
        default=False,
        description="True if each line begins with a sequential row number.",
    )


# Singleton default — used when schema discovery has nothing to work with
_FALLBACK_SCHEMA = TranscriptSchemaConfig()


def _detect_schema_heuristic(sample_lines: List[str]) -> TranscriptSchemaConfig:
    """
    Deterministic structural detector: vote on whether numeric tokens appear
    LEFT of the grade (→ after_numbers) or RIGHT of the grade (→ before_numbers).
    This is the ground-truth decider; the LLM call is a non-blocking cross-check.
    """
    after_votes = 0    # grade is last; numbers are to the left
    before_votes = 0   # grade is in the middle; numbers follow it
    has_status = False
    has_type = False
    has_row_num = False

    for raw in sample_lines:
        line = raw.strip()
        if not line:
            continue
        if re.match(r"^\d{1,3}[\s.]+", line):
            has_row_num = True
        if re.search(r"\b[ZS]\b", line):
            has_type = True

        tokens = line.split()
        for i, tok in enumerate(tokens):
            if tok.upper().rstrip(".") in ALL_GRADES:
                nums_right = 0
                for t in tokens[i + 1:]:
                    if normalize_str(t) in _STATUS_WORDS:
                        has_status = True
                        break
                    try:
                        float(t.replace(",", "."))
                        nums_right += 1
                    except ValueError:
                        break
                if nums_right >= 1:
                    before_votes += 1
                else:
                    after_votes += 1
                break  # one vote per line

    grade_pos: Literal["after_numbers", "before_numbers"] = (
        "before_numbers" if before_votes > after_votes else "after_numbers"
    )
    print(
        f"[SCHEMA] Heuristic → grade_pos={grade_pos!r} "
        f"(after={after_votes}, before={before_votes}, "
        f"status={has_status}, type_col={has_type})"
    )
    return TranscriptSchemaConfig(
        grade_position=grade_pos,
        has_status_word=has_status,
        has_type_column=has_type,
        has_row_number=has_row_num,
    )


def discover_transcript_schema(sample_text: str) -> TranscriptSchemaConfig:
    """
    Neuro-Symbolic Stage 1 — called ONCE per file upload.

    Pipeline:
      1. Extract up to 4 candidate course lines from the first ~4 000 chars.
      2. Run the deterministic heuristic (always fast, always reliable).
      3. Cross-check with the LLM via with_structured_output (best-effort).
      4. Merge: heuristic owns structural/positional fields;
         LLM may refine simple boolean fields.
      5. Any LLM failure → return heuristic result directly (never blocks upload).
    """
    # Gather up to 4 candidate course lines
    candidates: List[str] = []
    for raw in sample_text.splitlines():
        line = raw.strip()
        if _SKIP_LINE_RE.search(line):
            continue
        clean = re.sub(r"^\d{1,3}[\s.]+", "", line).strip()
        if re.match(r"^[A-Z]{2,6}\s?\d{3,4}", clean):
            candidates.append(clean)
        if len(candidates) >= 4:
            break

    if not candidates:
        print("[SCHEMA] No candidate course lines — using fallback schema")
        return _FALLBACK_SCHEMA

    # Primary ground-truth: Python heuristic
    heuristic = _detect_schema_heuristic(candidates)

    # Optional cross-check: LLM structured output
    try:
        sample_str = "\n".join(f"  {i+1}. {ln}" for i, ln in enumerate(candidates))
        prompt = (
            "Aşağıdaki Türk üniversite transkriptinden alınmış ders satırlarını incele:\n\n"
            f"{sample_str}\n\n"
            "Her satır şu sütunları içerebilir: KOD  AD  [TİP]  [T U L]  KREDİ  AKTS  NOT  [DURUM]\n"
            "Kritik soru: NOT değeri (AA/BB/CC/F gibi), Kredi ve AKTS sayılarından ÖNCE mi gelir "
            "yoksa SONRA mi?\n"
            "  'after_numbers'  → NOT sayıların SONUNDA (en yaygın)\n"
            "  'before_numbers' → NOT sayıların ÖNÜNDE (ortada)\n"
            "JSON formatında yanıt ver."
        )
        llm_schema: TranscriptSchemaConfig = (
            chat_model.with_structured_output(TranscriptSchemaConfig).invoke(prompt)
        )
        print(
            f"[SCHEMA] LLM → grade_pos={llm_schema.grade_position!r}, "
            f"status={llm_schema.has_status_word}, type={llm_schema.has_type_column}"
        )
        # Trust heuristic for positional/structural fields; accept LLM booleans
        return TranscriptSchemaConfig(
            grade_position=heuristic.grade_position,   # heuristic is authoritative
            has_status_word=llm_schema.has_status_word,
            has_type_column=llm_schema.has_type_column,
            has_row_number=heuristic.has_row_number,   # heuristic is authoritative
        )
    except Exception as exc:
        print(f"[SCHEMA] LLM failed ({exc!r}) — using heuristic only")
        return heuristic


def _parse_course_line(
    line: str,
    schema: Optional[TranscriptSchemaConfig] = None,
) -> Optional[dict]:
    """
    Neuro-Symbolic Stage 2: schema-guided Python parser, called for every line.

    grade_position="after_numbers" (default):
        CODE  NAME  [TYPE]  [T U L]  CREDIT  ECTS  GRADE  [STATUS]

    grade_position="before_numbers":
        CODE  NAME  [TYPE]  [T U L]  GRADE  CREDIT  ECTS  [STATUS]

    Automatic per-line fallback: if the primary strategy yields no numbers,
    the alternative strategy is attempted before giving up.
    """
    if schema is None:
        schema = _FALLBACK_SCHEMA

    line = line.strip()
    if not line:
        return None
    line = re.sub(r"^\d{1,3}[\s.]+", "", line).strip()

    m = COURSE_CODE_RE.match(line)
    if not m:
        return None

    code = (m.group(1) + m.group(2)).upper()
    rest = line[m.end():].strip()
    tokens = rest.split()
    if not tokens:
        return None

    # ── Locate the grade token (right-to-left; works for both layouts) ───────
    grade = None
    grade_idx = None
    for i in range(len(tokens) - 1, -1, -1):
        t = tokens[i].upper().rstrip(".")
        if t in ALL_GRADES:
            grade = t
            grade_idx = i
            break

    if grade is None or grade_idx is None:
        print(f"    [PARSE-SKIP] No grade token in: {line[:80]!r}")
        return None

    before = tokens[:grade_idx]
    after_grade = tokens[grade_idx + 1:]

    # Pre-compute numbers immediately after the grade (needed for before_numbers)
    nums_after: List[float] = []
    for tok in after_grade:
        if normalize_str(tok) in _STATUS_WORDS:
            break
        try:
            nums_after.append(float(tok.replace(",", ".")))
        except ValueError:
            break

    # ── Decide which extraction strategy to use for this line ────────────────
    use_before = schema.grade_position == "before_numbers" and len(nums_after) >= 1
    if schema.grade_position == "before_numbers" and not use_before:
        print(f"    [PARSE-INFO] before_numbers: no nums after grade — retrying as after_numbers: {line[:60]!r}")

    # ── before_numbers: CREDIT/ECTS live after the grade ─────────────────────
    if use_before:
        akts = nums_after[-1]
        credit = nums_after[-2] if len(nums_after) >= 2 else akts

        # Course name from `before`; strip trailing T/U/L numbers first
        ne = len(before) - 1
        while ne >= 0:
            try:
                float(before[ne].replace(",", "."))
                ne -= 1
            except ValueError:
                break
        tip = ""
        if ne >= 0 and before[ne].upper() in _TYPE_TOKENS:
            tip = before[ne].upper()
            ne -= 1
        name = " ".join(before[:ne + 1]).strip()

    # ── after_numbers: CREDIT/ECTS live before the grade (default) ───────────
    else:
        numbers: List[float] = []
        ne = len(before) - 1
        while ne >= 0:
            try:
                numbers.insert(0, float(before[ne].replace(",", ".")))
                ne -= 1
            except ValueError:
                break

        if not numbers:
            print(f"    [PARSE-SKIP] No numeric columns for {code}: {line[:80]!r}")
            return None

        akts = numbers[-1]
        credit = numbers[-2] if len(numbers) >= 2 else akts
        tip = ""
        if ne >= 0 and before[ne].upper() in _TYPE_TOKENS:
            tip = before[ne].upper()
            ne -= 1
        name = " ".join(before[:ne + 1]).strip()

    if not name:
        print(f"    [PARSE-SKIP] Empty name for code={code}")
        return None

    result = {
        "kod": code,
        "ad": name,
        "tip": tip,
        "kredi": round(credit, 1),
        "akts": round(akts, 1),
        "not": grade,
        "puan": GRADE_POINTS.get(grade, None),
        "durum": "Geçti" if is_passing(grade) else "Kaldı",
    }
    print(
        f"    [PARSE-OK ] {code} | {name!r} | {grade} | "
        f"akts={akts} | layout={schema.grade_position}"
    )
    return result


def _extract_term_gpa(block: str) -> Optional[float]:
    m = TERM_GPA_RE.search(block)
    if m:
        try:
            return float(m.group(1).replace(",", "."))
        except ValueError:
            pass
    # Fallback: bare "Yarıyıl  2.47" — guard with GPA range sanity check
    m2 = _TERM_GPA_BARE_RE.search(block)
    if m2:
        try:
            val = float(m2.group(1).replace(",", "."))
            if 0.0 <= val <= 4.0:
                return val
        except ValueError:
            pass
    return None


def _collect_courses_from_block(
    block: str,
    schema: Optional[TranscriptSchemaConfig] = None,
) -> List[dict]:
    """Parse all course lines from a text block using the given schema."""
    courses: List[dict] = []
    for raw_line in block.splitlines():
        if _SKIP_LINE_RE.search(raw_line):
            continue
        parsed = _parse_course_line(raw_line, schema)
        if parsed:
            courses.append(parsed)
    return courses


def parse_transcript_txt(raw_bytes: bytes) -> dict:
    text = _decode_txt(raw_bytes)
    print(f"\n[PARSER] Transcript length: {len(text)} chars")

    # ── Stage 1: Schema discovery — called ONCE, result reused for all lines ─
    schema = discover_transcript_schema(text[:4000])
    print(
        f"[PARSER] Active schema: grade_pos={schema.grade_position!r}, "
        f"status={schema.has_status_word}, type_col={schema.has_type_column}"
    )

    # Extended code scan for all_codes fallback
    raw_code_matches = re.findall(r"\b([A-Z]{2,6})\s?(\d{3,4}[A-Z]?)\b", text)
    all_codes = list(set((g[0] + g[1]).upper() for g in raw_code_matches))
    print(f"[PARSER] all_codes in raw text: {len(all_codes)}")

    sem_matches = list(SEM_HEADER_RE.finditer(text))
    print(f"[PARSER] Semester headers: {len(sem_matches)}")
    semesters: List[dict] = []

    # ── Stage 2: Parse each semester block with the discovered schema ─────────
    if sem_matches:
        for i, sm in enumerate(sem_matches):
            year_s, year_e = sm.group(1), sm.group(2)
            term = sm.group(3).capitalize()
            label = f"{year_s} - {year_e} {term}"
            start = sm.end()
            end = sem_matches[i + 1].start() if i + 1 < len(sem_matches) else len(text)
            block = text[start:end]

            print(f"\n[PARSER] --- {label} ({block.count(chr(10))} lines) ---")
            courses = _collect_courses_from_block(block, schema)
            print(f"[PARSER] Parsed {len(courses)} course(s) from {label}")

            term_gpa = _extract_term_gpa(block)
            term_ects = sum(c["akts"] for c in courses if c["durum"] == "Geçti")

            semesters.append(
                {
                    "semester_name": label,
                    "courses": courses,
                    "term_gpa": term_gpa,
                    "term_ects": term_ects,
                }
            )

    # ── Flat-parse fallback (same schema, whole document) ────────────────────
    total_parsed = sum(len(s["courses"]) for s in semesters)
    if total_parsed == 0:
        print("\n[PARSER] Semester blocks empty — trying flat scan with same schema")
        flat = _collect_courses_from_block(text, schema)
        print(f"[PARSER] Flat scan: {len(flat)} courses")
        if flat:
            label = "Transkript (Düz Tarama)" if semesters else "Genel Transkript"
            semesters = [
                {
                    "semester_name": label,
                    "courses": flat,
                    "term_gpa": None,
                    "term_ects": sum(c["akts"] for c in flat if c["durum"] == "Geçti"),
                }
            ]

    if not semesters:
        print("[PARSER] No semesters and no courses found — file may be unreadable")

    # ── Cumulative GPA — use findall + [-1] to capture the FINAL value ─────────
    cumulative_gpa: Optional[float] = None
    all_gpa_hits = CUMULATIVE_GPA_RE.findall(text)
    if all_gpa_hits:
        try:
            cumulative_gpa = float(all_gpa_hits[-1].replace(",", "."))
            print(f"[PARSER] Cumulative GPA hits: {all_gpa_hits} → using last: {cumulative_gpa}")
        except ValueError:
            pass

    if cumulative_gpa is None and semesters:
        gpas = [s["term_gpa"] for s in semesters if s["term_gpa"] is not None]
        if gpas:
            cumulative_gpa = round(sum(gpas) / len(gpas), 2)

    print(f"[PARSER] Final: {len(semesters)} semester(s), gpa={cumulative_gpa}\n")

    return {
        "semesters": semesters,
        "all_codes": all_codes,
        "cumulative_gpa": cumulative_gpa,
        "schema_used": schema.grade_position,
    }


# ═════════════════════════════════════════════════════════════
# MULTI-AGENT CHRONOLOGICAL STATE MACHINE (LangGraph)
#
# Node 1 — Historian (Tarihçi)
#   Walks through semesters chronologically.  For every semester it loads
#   the exact curriculum JSON for that academic year, then records each
#   course appearance with its year-specific curriculum entry.
#
# Node 2 — Mapper (Haritacı)
#   Builds a "Retake Graph" from all chronological records.
#   Sticky-pass rule: a "Geçti" is NEVER overwritten by a later fail.
#   ECTS counted once per canonical base code (deduplication).
#
# Node 3 — Judge (Hakem)
#   Checks the resolved map against the LATEST graduation curriculum.
#   Produces final passed/failed/missing lists and total ECTS.
#   If total_ects == 0 with a non-empty transcript, routes back to
#   Mapper once (ReAct retry) before giving up.
#
# status_queue is Annotated with operator.add so each node appends
# its own messages; the endpoint emits them via SSE as they arrive.
# ═════════════════════════════════════════════════════════════

class PipelineState(TypedDict):
    # ── Input ──────────────────────────────────────────────────
    semesters:             List[Dict]
    all_codes:             List[str]
    # ── Historian (accumulated from parallel semester agents) ──
    chronological_records: Annotated[List[Dict], operator.add]
    # ── Mapper ─────────────────────────────────────────────────
    resolved_courses:      Dict[str, Dict]   # canonical → entry
    unrecognized:          Dict[str, Dict]   # base_code → entry
    # ── Judge ──────────────────────────────────────────────────
    passed_codes:          List[str]
    failed_courses:        List[Dict]
    missing_courses:       List[Dict]
    unrecognized_electives: List[Dict]
    total_ects:            float
    matched_log:           List[str]
    judge_verdict:         str               # "approved" | "needs_remapping"
    mapper_retries:        int
    # ── Intibak (equivalency) review ──────────────────────────
    intibak_cleared_codes: List[str]   # codes cleared by intibak equivalency rules
    # ── SSE messages (accumulated across nodes) ────────────────
    status_queue:          Annotated[List[str], operator.add]


# ─────────────────────────────────────────────────────────────
# PYDANTIC SCHEMAS FOR STRUCTURED LLM OUTPUT
# Used with chat_model.with_structured_output(...).ainvoke()
# Enforces schema at inference time — prevents hanging / rambling.
# ─────────────────────────────────────────────────────────────
class ReviewerNote(BaseModel):
    kod: str = Field(description="Ders kodu (örn. DERS101)")
    aciklama: str = Field(description="Kısa Türkçe onay notu")


class ReviewerAIResponse(BaseModel):
    notlar: List[ReviewerNote] = Field(
        description="Her tanınmayan ders için bir onay notu"
    )


class IntibakClearedItem(BaseModel):
    kod: str = Field(description="Temizlenen ders kodu (örn. ING111)")
    sebep: str = Field(description="Neden eksik sayılmaması gerektiğinin Türkçe açıklaması")


class IntibakAIResponse(BaseModel):
    temizlenen: List[IntibakClearedItem] = Field(
        description="Eksik listesinden çıkarılması gereken dersler; yoksa boş liste"
    )


def _format_intibak_rules(kurallari: list) -> str:
    """Format intibak_kurallari list as compact readable text for LLM prompt."""
    lines = []
    for rule in kurallari[:50]:  # cap at 50 to stay within context window
        eski = rule.get("eski_ders", {}).get("kod", "")
        yeni_kodlar = [y.get("kod", "") for y in rule.get("yeni_dersler", []) if y.get("kod")]
        kosul = rule.get("kosul_yili_oncesi", "")
        if eski and yeni_kodlar:
            kosul_str = f" [kosul: {kosul} ve oncesi]" if kosul else ""
            lines.append(f"{eski} -> {', '.join(yeni_kodlar)}{kosul_str}")
    return "\n".join(lines) if lines else "(kural bulunamadi)"


def _build_lookup(dersler: List[dict]):
    """Build by_code, by_base_code, by_name dicts for a curriculum list."""
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


def _get_latest_curriculum_lookup() -> tuple:
    """Return (by_code, by_base, by_name) for the latest curriculum, built once and cached."""
    global _latest_lookup_cache
    if _latest_lookup_cache is None:
        data = load_latest_curriculum()
        _latest_lookup_cache = _build_lookup(data.get("dersler", []))
        print(f"[FALLBACK-LOOKUP] Latest curriculum lookup tables built "
              f"({len(_latest_lookup_cache[0])} entries)")
    return _latest_lookup_cache


def _lookup_course(raw_code: str, name: str,
                   by_code: dict, by_base: dict, by_name: dict):
    """4-step cascade: exact → stripped → base_dict → name."""
    stripped = strip_suffix(raw_code)
    curr = by_code.get(raw_code)
    if curr is None:
        curr = by_code.get(stripped)
    if curr is None:
        curr = by_base.get(stripped)
    if curr is None and name:
        curr = by_name.get(normalize_str(name))
    return curr


# ── Node 1a: Historian Init (fan-out trigger) ─────────────────
def historian_init_node(state: PipelineState) -> dict:
    """Emits initial status and triggers parallel semester agent dispatch."""
    n = len(state.get("semesters", []))
    msgs = [f"Paralel Dönem Ajanları devreye alınıyor ({n} dönem)..."]
    return {"status_queue": msgs}


def _route_to_semester_agents(state: PipelineState):
    """Conditional edge: fan-out each semester to its own parallel agent via Send."""
    sems = state.get("semesters", [])
    if not sems:
        return "mapper"
    return [Send("semester_agent", {"semester": sem}) for sem in sems]


# ── Node 1b: Semester Agent (runs in parallel, one per semester) ──
def semester_agent_node(state: dict) -> dict:
    """Processes a single semester block and returns its chronological records."""
    sem = state["semester"]
    sem_name = sem.get("semester_name", "")
    records: List[Dict] = []

    m = re.search(r"(\d{4})\s*[-–—]\s*(\d{4})", sem_name)
    year_key = f"{m.group(1)}-{m.group(2)}" if m else "2025-2026"

    msgs = [f"Dönem Ajanı [{sem_name}] işleniyor ({year_key} müfredatı)..."]

    try:
        curr_data = load_year_curriculum(year_key)
    except Exception:
        curr_data = load_latest_curriculum()

    by_code, by_base, by_name = _build_lookup(curr_data.get("dersler", []))

    # Global fallback lookup tables (latest curriculum) — built once, cached globally
    latest_by_code, latest_by_base, latest_by_name = _get_latest_curriculum_lookup()

    for course in sem.get("courses", []):
        raw_code = course.get("kod", "")
        name     = course.get("ad", "")
        grade    = course.get("not", "")
        status   = course.get("durum", "")
        akts_t   = course.get("akts", 0)

        # Primary: look up in this semester's historical curriculum
        curr = _lookup_course(raw_code, name, by_code, by_base, by_name)

        # Global fallback: if not found in historical JSON, try the latest curriculum
        if curr is None:
            curr = _lookup_course(raw_code, name, latest_by_code, latest_by_base, latest_by_name)
            if curr:
                print(f"    [FALLBACK] {raw_code} not in {year_key} → found in latest curriculum")

        canonical = strip_suffix(curr["kod"]) if curr else None

        records.append({
            "semester_name":    sem_name,
            "year_key":         year_key,
            "raw_code":         raw_code,
            "ad":               name,
            "canonical":        canonical,
            "curriculum_entry": curr,
            "grade":            grade,
            "status":           status,
            "akts_transcript":  akts_t,
        })

    print(f"[SEMESTER-AGENT] {sem_name}: {len(records)} records")
    return {"chronological_records": records, "status_queue": msgs}


# ── Node 2: Mapper ────────────────────────────────────────────
def mapper_node(state: PipelineState) -> dict:
    msgs = [
        "Haritacı başarısız derslerin tekrar alınma (Retake) haritasını çıkarıyor...",
    ]

    resolved: Dict[str, Dict] = {}
    unrecognized: Dict[str, Dict] = {}

    for rec in state["chronological_records"]:
        canonical = rec.get("canonical")
        curr      = rec.get("curriculum_entry")
        grade     = rec.get("grade", "")
        status    = rec.get("status", "")
        raw_code  = rec.get("raw_code", "")
        name      = rec.get("ad", "")
        akts_t    = rec.get("akts_transcript", 0)

        if canonical and curr:
            prev = resolved.get(canonical)
            # Sticky-pass: never overwrite a pass with a later fail
            if prev is not None and prev["status"] == "Geçti" and status != "Geçti":
                pass  # preserve existing pass
            else:
                resolved[canonical] = {
                    "status":           status,
                    "grade":            grade,
                    "curriculum_entry": curr,
                    "raw_code":         raw_code,
                }
        else:
            # Not in any year's curriculum → unrecognized
            base_key = strip_suffix(raw_code)
            if status == "Geçti":
                # Latest pass wins; overwrite if retaken
                unrecognized[base_key] = {
                    "kod":  raw_code,
                    "ad":   name,
                    "not":  grade,
                    "akts": akts_t,
                }

    msgs.append("Haritacı AKTS çakışmalarını temizliyor...")
    print(f"[MAPPER] resolved={len(resolved)}, unrecognized={len(unrecognized)}")
    return {"resolved_courses": resolved, "unrecognized": unrecognized, "status_queue": msgs}


# ── Node 2b: Reviewer AI ──────────────────────────────────────
async def reviewer_ai_node(state: PipelineState) -> dict:
    """Async: uses structured LLM output to annotate each unrecognized elective."""
    unrecognized = state.get("unrecognized", {})
    if not unrecognized:
        return {"status_queue": ["Gözden Geçiren AI: Tanınmayan seçmeli ders bulunamadı."]}

    msgs = [f"Gözden Geçiren AI {len(unrecognized)} tanınmayan dersi değerlendiriyor..."]

    courses_list = "\n".join(
        f"- Kod: {v['kod']}, Ad: {v['ad']}" for v in unrecognized.values()
    )
    prompt = (
        "Aşağıdaki dersler üniversite müfredat veri tabanında bulunamadı. "
        "Bunların geçerli üniversite seçmeli dersleri olup olmadığını değerlendir. "
        "Örneğin 'Kariyer Planlama', 'İngilizce', 'Türk Dili', teknik/mesleki seçmeli dersler geçerlidir. "
        "Her ders için kısa Türkçe bir onay notu yaz.\n\n"
        f"Dersler:\n{courses_list}"
    )

    enriched = dict(unrecognized)
    _default_note = "Gözden geçirildi — geçerli seçmeli ders"
    try:
        llm = chat_model.with_structured_output(ReviewerAIResponse)
        result: ReviewerAIResponse = await llm.ainvoke([HumanMessage(content=prompt)])
        notes_map = {n.kod: n.aciklama for n in result.notlar}
        for k, v in enriched.items():
            enriched[k] = {**v, "reviewer_note": notes_map.get(v["kod"], _default_note)}
    except Exception as exc:
        print(f"[REVIEWER-AI] LLM error ({exc!r}) — using default notes")
        for k, v in enriched.items():
            if "reviewer_note" not in v:
                enriched[k] = {**v, "reviewer_note": _default_note}

    msgs.append("Gözden Geçiren AI değerlendirmesini tamamladı.")
    print(f"[REVIEWER-AI] Enriched {len(enriched)} unrecognized courses")
    return {"unrecognized": enriched, "status_queue": msgs}


# ── Node 2c: Missing Course Reviewer (Intibak) ───────────────
async def review_missing_courses_node(state: PipelineState) -> dict:
    """
    Async: reduces false-positive missing courses using intibak equivalency rules.

    Pass 1 — deterministic: expand the effective "passed" set via eski_yeni_eslestirme.
    Pass 2 — async AI (structured output): check unrecognized courses for base-code
      similarity and entry-year exemptions, using the full intibak_kurallari rules text.
    """
    msgs = ["İntibak Gözden Geçiren: müfredat dönüşüm kuralları kontrol ediliyor..."]

    resolved     = state.get("resolved_courses", {})
    unrecognized = state.get("unrecognized", {})

    # ── Build effective passed set ────────────────────────────
    passed_set: set = {
        canonical
        for canonical, entry in resolved.items()
        if entry.get("status") == "Geçti"
    }
    raw_passed: set = set()
    for canonical, entry in resolved.items():
        if entry.get("status") == "Geçti":
            rc = entry.get("raw_code", "")
            raw_passed.add(rc)
            raw_passed.add(strip_suffix(rc))
    unrec_passed = {strip_suffix(v["kod"]) for v in unrecognized.values()}
    all_passed   = passed_set | raw_passed | unrec_passed

    # ── Entry year ────────────────────────────────────────────
    semesters  = state.get("semesters", [])
    entry_year = "unknown"
    if semesters:
        m = re.search(r"(\d{4})\s*[-–—]\s*(\d{4})",
                      semesters[0].get("semester_name", ""))
        if m:
            entry_year = f"{m.group(1)}-{m.group(2)}"

    # ── Load intibak rules ────────────────────────────────────
    intibak        = load_intibak_rules()
    quick_lookup   = intibak.get("eski_yeni_eslestirme", {})   # old → [new, ...]
    intibak_rules  = intibak.get("intibak_kurallari", [])       # full rule objects

    # ── Pass 1: Deterministic expansion ──────────────────────
    cleared: set  = set()
    for old_code in list(all_passed):
        for new_code in quick_lookup.get(old_code, []):
            base_new = strip_suffix(new_code)
            if base_new not in passed_set:
                cleared.add(base_new)

    print(f"[INTIBAK] Pass-1 cleared {len(cleared)}: {cleared}")

    # ── Pass 2: Async AI review (structured output) ──────────
    ai_cleared: list = []
    if unrecognized and intibak_rules:
        latest_curr  = load_latest_curriculum()
        curr_dersler = latest_curr.get("dersler", [])
        failed_can   = {
            strip_suffix(e.get("curriculum_entry", {}).get("kod", ""))
            for e in resolved.values()
            if e.get("status") != "Geçti"
        }
        seen_bases: set = set()
        prelim_missing  = []
        for c in curr_dersler:
            if c.get("tip") != "Zorunlu":
                continue
            base = strip_suffix(c["kod"])
            if base in passed_set or base in cleared or base in failed_can or base in seen_bases:
                continue
            seen_bases.add(base)
            prelim_missing.append(f"{c['kod']}: {c['ad']}")

        if prelim_missing:
            unrec_list   = "\n".join(f"- {v['kod']}: {v['ad']}"
                                     for v in unrecognized.values())
            missing_list = "\n".join(prelim_missing[:20])
            # Full structured rules text — gives the LLM the actual academic rules,
            # not just a compressed quick-lookup dict.
            rules_text   = _format_intibak_rules(intibak_rules)

            prompt = (
                f"Sen bir müfredat eşdeğerlik uzmanısın. Öğrencinin kayıt yılı: {entry_year}\n\n"
                f"Tanınmayan (müfredat dışı) geçilen dersler:\n{unrec_list}\n\n"
                f"Ön hesaplamada eksik görünen zorunlu dersler (ilk 20):\n{missing_list}\n\n"
                "Akademik Takvim Kurallari (eski ders -> yeni ders esdeğerlikleri):\n"
                f"{rules_text}\n\n"
                "Gorev:\n"
                "1. Taninamayan derslerden herhangi biri eksik derslerden birinin ESDEGERI mi? "
                "(ornek: CNT350 aldiysa CNT250 eksik sayilmamali)\n"
                "2. Ogrencinin kayit yilina gore gecerli olmayan yeni dersler var mi? "
                "(ornek: ING111 2025-2026'da eklendi; intibak kuralina gore eski ders alanlar muaf)\n\n"
                "Temizlenecek ders yoksa bos liste dondur."
            )
            try:
                llm = chat_model.with_structured_output(IntibakAIResponse)
                result: IntibakAIResponse = await llm.ainvoke([HumanMessage(content=prompt)])
                for item in result.temizlenen:
                    kod = strip_suffix(item.kod)
                    if kod:
                        ai_cleared.append(kod)
                        print(f"[INTIBAK-AI] Cleared {kod}: {item.sebep}")
            except Exception as exc:
                print(f"[INTIBAK] AI pass failed ({exc!r}) — deterministic only")

    all_cleared = list(cleared | set(ai_cleared))
    if all_cleared:
        msgs.append(
            f"İntibak Gözden Geçiren: {len(all_cleared)} ders eşdeğerlik kuralıyla temizlendi "
            f"({', '.join(all_cleared[:5])}{'...' if len(all_cleared) > 5 else ''})"
        )
    else:
        msgs.append("İntibak Gözden Geçiren: Temizlenecek ders bulunamadı.")

    print(f"[INTIBAK] Total cleared: {all_cleared}")
    return {"intibak_cleared_codes": all_cleared, "status_queue": msgs}


# ── Node 3: Judge ─────────────────────────────────────────────
def judge_node(state: PipelineState) -> dict:
    msgs = ["Hakem ajan mezuniyet şartlarını denetliyor..."]

    resolved       = state.get("resolved_courses", {})
    unrecognized   = state.get("unrecognized", {})
    mapper_retries = state.get("mapper_retries", 0)
    semesters      = state.get("semesters", [])

    latest_curr  = load_latest_curriculum()
    curr_dersler = latest_curr.get("dersler", [])

    # ── Resolve final pass / fail ──────────────────────────────
    passed: set       = set()
    failed_courses:  List[Dict] = []
    matched_log:     List[str]  = []
    total_ects: float = 0.0

    for canonical, entry in resolved.items():
        curr   = entry["curriculum_entry"]
        status = entry["status"]
        grade  = entry["grade"]

        if status == "Geçti":
            passed.add(canonical)
            total_ects += curr.get("akts", 0)
            matched_log.append(f"{curr['kod']} - {curr['ad']} (kod)")
        elif is_failed(grade):
            failed_courses.append({
                "kod":   curr["kod"],
                "ad":    curr["ad"],
                "not":   grade,
                "akts":  curr.get("akts", 0),
                "donem": curr.get("donem", ""),
            })

    # ── Add unrecognized electives ─────────────────────────────
    recognized_bases = set(resolved.keys())
    unrecognized_electives: List[Dict] = []
    for base_key, entry in unrecognized.items():
        if base_key not in recognized_bases:
            unrecognized_electives.append(entry)
            total_ects += entry.get("akts", 0)

    # ── All-codes fallback (nothing matched) ───────────────────
    if not matched_log and not failed_courses and not unrecognized_electives:
        by_code_l, by_base_l, _ = _build_lookup(curr_dersler)
        for code in state.get("all_codes", []):
            target = _lookup_course(code, "", by_code_l, by_base_l, {})
            if target:
                can = strip_suffix(target["kod"])
                if can not in passed:
                    passed.add(can)
                    total_ects += target.get("akts", 0)
                    matched_log.append(f"{target['kod']} - {target['ad']} (fallback)")

    # ── ReAct retry: route back to Mapper if ECTS = 0 ─────────
    has_courses = any(len(s.get("courses", [])) > 0 for s in semesters)
    if total_ects == 0 and has_courses and mapper_retries < 1:
        msgs.append("Hakem mantıksal tutarsızlık tespit etti — Haritacı'ya yeniden yönlendiriliyor...")
        print("[JUDGE] ReAct: zero ECTS with non-empty transcript — retrying Mapper")
        return {
            "judge_verdict":   "needs_remapping",
            "mapper_retries":  mapper_retries + 1,
            "status_queue":    msgs,
        }

    # ── Missing mandatory courses (deduped by base code) ───────
    failed_canonicals  = {strip_suffix(f["kod"]) for f in failed_courses}
    intibak_cleared    = set(state.get("intibak_cleared_codes", []))
    seen_bases: set    = set()
    missing:    List[Dict] = []
    for c in curr_dersler:
        if c.get("tip") != "Zorunlu":
            continue
        base = strip_suffix(c["kod"])
        if (base in passed or base in failed_canonicals
                or base in seen_bases or base in intibak_cleared):
            continue
        seen_bases.add(base)
        missing.append({
            "kod":   c["kod"],
            "ad":    c["ad"],
            "akts":  c.get("akts", 0),
            "donem": c.get("donem", ""),
            "tip":   c.get("tip", ""),
        })

    msgs.append("Hakem sonuçları onayladı, rapor hazırlanıyor...")
    print(f"[JUDGE] passed={len(passed)}, failed={len(failed_courses)}, "
          f"missing={len(missing)}, ects={round(total_ects,1)}, "
          f"unrecognized={len(unrecognized_electives)}")

    return {
        "passed_codes":          list(passed),
        "failed_courses":        failed_courses,
        "missing_courses":       missing,
        "unrecognized_electives": unrecognized_electives,
        "total_ects":            round(total_ects, 1),
        "matched_log":           matched_log,
        "judge_verdict":         "approved",
        "mapper_retries":        mapper_retries,
        "status_queue":          msgs,
    }


def _judge_router(state: PipelineState) -> str:
    return "mapper" if state.get("judge_verdict") == "needs_remapping" else END


# ── Compile pipeline (Map-Reduce: parallel semester agents) ───
_g = StateGraph(PipelineState)
_g.add_node("historian_init",        historian_init_node)
_g.add_node("semester_agent",        semester_agent_node)
_g.add_node("mapper",                mapper_node)
_g.add_node("reviewer_ai",           reviewer_ai_node)
_g.add_node("review_missing_courses", review_missing_courses_node)
_g.add_node("judge",                 judge_node)
_g.set_entry_point("historian_init")
# Fan-out: each semester → its own parallel agent; empty → skip straight to mapper
_g.add_conditional_edges("historian_init", _route_to_semester_agents, ["semester_agent", "mapper"])
# Fan-in: all semester agents converge to mapper
_g.add_edge("semester_agent",        "mapper")
_g.add_edge("mapper",                "reviewer_ai")
_g.add_edge("reviewer_ai",           "review_missing_courses")
_g.add_edge("review_missing_courses", "judge")
_g.add_conditional_edges("judge", _judge_router, {"mapper": "mapper", END: END})
analysis_pipeline = _g.compile()



# ─────────────────────────────────────────────────────────────
# AGENT TOOLS (closures per-request for session isolation)
# ─────────────────────────────────────────────────────────────
TOOL_STATUS = {
    "GetTranscriptData":      "Tarihçi transkript kayıtlarını tarıyor...",
    "GetCurriculumData":      "Haritacı müfredat verilerini yüklüyor...",
    "CalculateGraduationPath": "Hakem mezuniyet durumunu hesaplıyor...",
}


def make_tools(session_id: str) -> list:
    def get_transcript_data(_: str = "") -> str:
        """Öğrencinin dönem dönem transkript verilerini getirir."""
        data = get_session(session_id).get("transcript")
        return (
            "Henüz transkript yüklenmemiş."
            if not data
            else json.dumps(data, ensure_ascii=False, indent=2)
        )

    def get_curriculum_data(_: str = "") -> str:
        """Bilgisayar Mühendisliği müfredatını getirir."""
        try:
            return json.dumps(load_curriculum(), ensure_ascii=False, indent=2)
        except Exception as e:
            return f"Müfredat yüklenemedi: {e}"

    def calculate_graduation_path(_: str = "") -> str:
        """Eksik ve başarısız dersler + mezuniyet hesabı + tanınmayan seçmeliler."""
        transcript = get_session(session_id).get("transcript")
        if not transcript:
            return "Transkript verisi bulunamadı."
        try:
            curr = load_curriculum()
            sartlar = curr.get("mezuniyet_sartlari", {})
            hedef = sartlar.get("toplam_akts_hedefi", 240)
            total = transcript.get("total_ects", 0)
            missing = transcript.get("missing_courses", [])
            failed = transcript.get("failed_courses", [])
            unrecognized = transcript.get("unrecognized_electives", [])
            unrecognized_ects = round(sum(e.get("akts", 0) for e in unrecognized), 1)

            # Build prioritised next-semester list:
            # 1) Failed courses first (mandatory retake)
            # 2) Missing mandatory courses ordered by semester number
            priorities: List[dict] = []
            seen = set()
            for f in failed:
                priorities.append(
                    {
                        "kod": f["kod"],
                        "ad": f["ad"],
                        "akts": f.get("akts", 0),
                        "sebep": f"Önceki dönemde {f.get('not','F')} aldı — TEKRAR ZORUNLU",
                        "oncelik": "YÜKSEK",
                    }
                )
                seen.add(f["kod"])

            for m in sorted(missing, key=lambda x: str(x.get("donem", "99"))):
                if m["kod"] not in seen:
                    priorities.append(
                        {
                            "kod": m["kod"],
                            "ad": m["ad"],
                            "akts": m.get("akts", 0),
                            "sebep": "Henüz alınmamış zorunlu ders",
                            "oncelik": "NORMAL",
                        }
                    )
                    seen.add(m["kod"])

            return json.dumps(
                {
                    "tamamlanan_akts": total,
                    "kalan_akts": round(hedef - total, 1),
                    "hedef_akts": hedef,
                    "minimum_gno": sartlar.get("minimum_gno", 2.0),
                    "eksik_zorunlu_sayisi": len(missing),
                    "basarisiz_ders_sayisi": len(failed),
                    "taninmayan_secmeli_sayisi": len(unrecognized),
                    "taninmayan_secmeli_akts": unrecognized_ects,
                    "taninmayan_secmeliler": [
                        {
                            "kod": e["kod"],
                            "ad": e["ad"],
                            "akts": e.get("akts", 0),
                            "aciklama": (
                                "Müfredat veri tabanında kaydı bulunmuyor — "
                                "AKTS mezuniyet toplamına dahil edildi"
                            ),
                        }
                        for e in unrecognized
                    ],
                    "tekrar_alinacaklar": failed,
                    "eksik_zorunlular": missing,
                    "oneri_gelecek_donem": priorities[:8],
                },
                ensure_ascii=False,
                indent=2,
            )
        except Exception as e:
            return f"Hesaplama hatası: {e}"

    return [
        StructuredTool.from_function(
            func=get_transcript_data,
            name="GetTranscriptData",
            description=(
                "Öğrencinin dönem dönem transkript verilerini getirir. "
                "Dönem GPA'ları, AKTS değerleri ve ders listesi için kullan. "
                "NOT: total_ects ve cumulative_gpa Python tarafından önceden hesaplanmıştır; "
                "bu sayıları yeniden hesaplama, doğrudan kullan."
            ),
        ),
        StructuredTool.from_function(
            func=get_curriculum_data,
            name="GetCurriculumData",
            description=(
                "Bilgisayar Mühendisliği müfredatını getirir: zorunlu/seçmeli dersler, "
                "AKTS değerleri ve mezuniyet koşulları."
            ),
        ),
        StructuredTool.from_function(
            func=calculate_graduation_path,
            name="CalculateGraduationPath",
            description=(
                "Eksik ve başarısız zorunlu dersleri, tanınmayan seçmeli dersleri ve "
                "mezuniyet durumunu önceden hesaplanmış Python değerleriyle döndürür. "
                "'Mezun olabilir miyim?', 'Kaç dersim eksik?', 'Gelecek dönem ne almalıyım?', "
                "'Başarısız derslerim var mı?' sorularında kullan. "
                "NOT: akts_tamamlanan ve mezuniyet_akts_yeterli gibi sayısal alanlar "
                "Python tarafından hesaplanmıştır; yeniden hesaplama yapma."
            ),
        ),
    ]


# ─────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """Sen Galatasaray Üniversitesi Bilgisayar Mühendisliği bölümünün kıdemli, \
samimi ve %100 dürüst Akademik Danışmanısın. Öğrenciye yalnızca Türkçe yanıt ver.

## DÜŞÜNCE SÜRECİ (Her yanıttan önce şu adımları zihinsel olarak uygula)
ADIM 1 — Soruyu Analiz Et: Öğrenci tam olarak ne istiyor? \
(Mezuniyet kontrolü mü? Eksik dersler mi? GNO durumu mu? Gelecek dönem planı mı?)
ADIM 2 — Araç Seçimi: Hangi araçlara ihtiyacın var? \
Transkript verisini mi, müfredatı mı, yoksa mezuniyet hesabını mı kullanmalısın?
ADIM 3 — Veri Doğrulama: Araçlardan gelen veriyi özetle. \
Tutarsızlık var mı? Eksik veri var mı? Sayılar mantıklı mı?
ADIM 4 — Yanıtı Yaz: Yalnızca bu düşünce sürecinden sonra öğrenciye nihai yanıtı ver.

## MATEMATİK KURALI (KESİNLİKLE UYULMASI ZORUNLU)
Araçlardan dönen `total_ects`, `cumulative_gpa`, `term_gpa`, `term_ects` değerleri \
Python işçisi tarafından hesaplanmış kesin sayılardır. Bu sayıları ASLA yeniden hesaplama, \
toplama, çıkarma ya da sorgulama. Rakamları olduğu gibi kullan; yorumla, tekrar hesaplama.
`failed_courses` ve `missing_courses` listeleri Python tarafından yeniden alınan dersler \
de dahil edilerek çakışmalar çözülmüş nihai listelerdir. Ham transkript verisinden kendi \
başına başarısız ya da eksik ders keşfetmeye ÇALIŞMA; bu listeler kesindir ve güvenilirdir.

## YANIT KURALLARI
- Öğrenciye doğrudan "sen" diye hitap et.
- Sadece gerçek verilere dayan; bilmiyorsan açıkça belirt.
- Yanıtlarında gerektiğinde markdown tablo ve madde işareti kullan.
- Yanıtın kısa ve odaklı olsun; gereksiz tekrar yapma.

## TANINMAYAN SEÇMELİ DERSLER
Transkriptte müfredat veri tabanında kaydı bulunmayan dersler \
(unrecognized_electives) olabilir. Bunlar büyük olasılıkla güncel olmayan \
teknik/seçmeli derslerdir. AKTS değerleri mezuniyet toplamına BAŞARIYLA dahil \
edilmiştir. "Bu derslerin AKTS'i sayılmadı" diye ASLA yanlış bilgi verme; \
aksine katkılarını olumlu biçimde belirt.

## KRİTİK: ASLA KURS KODU UYDURMA
Araç çıktılarından gelen gerçek ders kodu ve isimlerini birebir kullan. \
'DERS1', 'DERS3', 'COURSE1' gibi sahte, uydurma ders kodları ASLA yazma. \
Araç sana eksik derslerin listesini döndürüyorsa, o listedeki EXACT kodları \
ve isimleri kullan — hiçbir şey ekleme, hiçbir şey çıkarma, hiçbir şey uydurma.

## KRİTİK: HAM JSON ARAÇ ÇAĞRISI YASAK
Araçları NATIVE API üzerinden çağır. Yanıt metninde asla \
`{"name": "ToolName", "parameters": {...}}` gibi ham JSON blokları yazma. \
Eğer bir araç çağırmak istiyorsan, onu doğrudan faaliyete geçir — metin olarak \
JSON çıktısı verme. Araç zaten çalıştı ve veri döndürdüyse, o veriyi Türkçe \
olarak açıkla; aracı tekrar çağırma."""


# ─────────────────────────────────────────────────────────────
# REQUEST MODEL
# ─────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    session_id: str


# ─────────────────────────────────────────────────────────────
# UPLOAD ENDPOINT  (TXT input)
# ─────────────────────────────────────────────────────────────
@app.post("/upload")
async def upload_transcript(
    file: UploadFile = File(...),
    session_id: str = Query(default=None),
):
    if not session_id:
        session_id = str(uuid.uuid4())

    print(f"\n--- TRANSKRİPT ANALİZİ | session: {session_id[:8]} ---")

    raw_bytes = await file.read()

    async def generate():
        yield _sse({"type": "status", "data": "Transkript dosyası okunuyor ve ayrıştırılıyor..."})

        try:
            parsed = await asyncio.to_thread(parse_transcript_txt, raw_bytes)
        except Exception as e:
            yield _sse({"type": "error", "data": f"Transkript ayrıştırılamadı: {e}"})
            return

        initial: PipelineState = {
            "semesters":              parsed.get("semesters", []),
            "all_codes":              parsed.get("all_codes", []),
            "chronological_records":  [],
            "resolved_courses":       {},
            "unrecognized":           {},
            "passed_codes":           [],
            "failed_courses":         [],
            "missing_courses":        [],
            "unrecognized_electives": [],
            "total_ects":             0.0,
            "matched_log":            [],
            "judge_verdict":          "",
            "mapper_retries":         0,
            "intibak_cleared_codes":  [],
            "status_queue":           [],
        }

        emitted = 0
        final_state = None
        try:
            async for chunk in analysis_pipeline.astream(initial, stream_mode="values"):
                final_state = chunk
                q = chunk.get("status_queue", [])
                while emitted < len(q):
                    yield _sse({"type": "status", "data": q[emitted]})
                    emitted += 1
        except Exception as e:
            yield _sse({"type": "error", "data": f"Analiz hatası: {e}"})
            return

        if final_state is None:
            yield _sse({"type": "error", "data": "Pipeline çalıştırılamadı."})
            return

        transcript_data = {
            "session_id":             session_id,
            "semesters":              parsed.get("semesters", []),
            "cumulative_gpa":         parsed.get("cumulative_gpa"),
            "passed_codes":           final_state.get("passed_codes", []),
            "matched_log":            final_state.get("matched_log", []),
            "total_ects":             final_state.get("total_ects", 0.0),
            "missing_courses":        final_state.get("missing_courses", []),
            "failed_courses":         final_state.get("failed_courses", []),
            "unrecognized_electives": final_state.get("unrecognized_electives", []),
        }

        get_session(session_id)["transcript"] = transcript_data

        try:
            with open("son_ayiklanan_transcript.json", "w", encoding="utf-8") as f:
                json.dump(transcript_data, f, ensure_ascii=False, indent=4)
        except Exception:
            pass

        semester_summaries = [
            {
                "name":         s["semester_name"],
                "gpa":          s.get("term_gpa"),
                "ects":         s.get("term_ects", 0),
                "course_count": len(s.get("courses", [])),
                "failed_count": sum(1 for c in s.get("courses", []) if c.get("durum") == "Kaldı"),
                "courses":      s.get("courses", []),
            }
            for s in parsed.get("semesters", [])
        ]

        try:
            latest_curr = load_latest_curriculum()
            target_ects = latest_curr.get("mezuniyet_sartlari", {}).get("toplam_akts_hedefi", 240)
        except Exception:
            target_ects = 240

        unrecognized    = final_state.get("unrecognized_electives", [])
        total_ects      = final_state.get("total_ects", 0.0)
        missing_courses = final_state.get("missing_courses", [])
        failed_courses  = final_state.get("failed_courses", [])

        summary = {
            "total_ects":           total_ects,
            "target_ects":          target_ects,
            "missing_count":        len(missing_courses),
            "failed_count":         len(failed_courses),
            "unrecognized_count":   len(unrecognized),
            "unrecognized_ects":    round(sum(e.get("akts", 0) for e in unrecognized), 1),
            "semester_count":       len(parsed.get("semesters", [])),
            "cumulative_gpa":       parsed.get("cumulative_gpa"),
            "semesters":            semester_summaries,
            # Full arrays for modal display in the frontend
            "missing_courses":      missing_courses,
            "failed_courses":       failed_courses,
            "unrecognized_electives": unrecognized,
        }

        yield _sse({"type": "summary", "data": {"session_id": session_id, "summary": summary}})

        # LLM initial assessment
        missing_names = [f"{c['kod']} - {c['ad']}" for c in missing_courses]
        failed_names  = [f"{c['kod']} - {c['ad']} ({c.get('not','F')})" for c in failed_courses]
        unrec_ects    = round(sum(e.get("akts", 0) for e in unrecognized), 1)

        prompt = (
            f"Öğrencinin transkripti analiz edildi:\n"
            f"- Tamamlanan AKTS: {total_ects} / {target_ects}\n"
            f"- Genel GNO: {parsed.get('cumulative_gpa') or 'Tespit edilemedi'}\n"
            f"- Dönem sayısı: {len(semester_summaries)}\n"
            f"- Eksik zorunlu ders: {len(missing_courses)}\n"
            f"- Başarısız (F) ders: {len(failed_courses)}\n"
            + (f"- Başarısız: {', '.join(failed_names)}\n" if failed_names else "")
            + (
                f"- Tanınmayan seçmeli: {len(unrecognized)} ders "
                f"({unrec_ects} AKTS — mezuniyet toplamına dahil edildi)\n"
                if unrecognized else ""
            )
            + (
                f"- Eksikler (ilk 5): {', '.join(missing_names[:5])}"
                + ("..." if len(missing_names) > 5 else "")
                + "\n"
                if missing_names else ""
            )
            + "\nBu sonuçları öğrenciye samimi ve profesyonel Türkçeyle 2-3 cümlede özetle. "
            "İlk cümlede genel durumu belirt, ikincide kritik eksikler/başarısızlıkları vurgula, "
            "üçüncüde yönlendirici tavsiye ver. Tanınmayan seçmeli ders varsa AKTS'lerinin "
            "başarıyla sayıldığını olumlu bir şekilde belirt."
        )

        yield _sse({"type": "status", "data": "Akademik danışman ilk değerlendirmeyi hazırlıyor..."})

        try:
            result = await asyncio.to_thread(
                chat_model.invoke,
                [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)],
            )
            reply = result.content
        except Exception:
            reply = (
                f"Transkriptiniz analiz edildi. {total_ects} AKTS tamamlandı"
                + (f", GNO: {parsed.get('cumulative_gpa'):.2f}" if parsed.get("cumulative_gpa") else "")
                + f". {len(missing_courses)} zorunlu ders eksik"
                + (f", {len(failed_courses)} ders başarısız" if failed_courses else "")
                + ". Detaylar için soru sorabilirsiniz."
            )

        yield _sse({"type": "done", "data": reply})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "Connection":        "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ─────────────────────────────────────────────────────────────
# CHAT STREAM  (SSE via astream_events)
# ─────────────────────────────────────────────────────────────
@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    tools = make_tools(request.session_id)
    agent = create_react_agent(
        model=chat_model,
        tools=tools,
        prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )
    config = {"configurable": {"thread_id": request.session_id}}

    async def generate():
        yield _sse({"type": "status", "data": "⚡ Akademik danışman devreye giriyor..."})

        got_done = False
        try:
            async for event in agent.astream_events(
                {"messages": [HumanMessage(content=request.message)]},
                config=config,
                version="v2",
            ):
                etype = event["event"]
                ename = event.get("name", "")

                if etype == "on_tool_start":
                    yield _sse(
                        {
                            "type": "status",
                            "data": TOOL_STATUS.get(ename, f"{ename} çalıştırılıyor..."),
                        }
                    )
                elif etype == "on_tool_end":
                    yield _sse({"type": "status", "data": "🔍 Veriler analiz ediliyor..."})

                elif etype == "on_chain_end" and ename == "LangGraph":
                    # The outermost graph finished — extract the final AI text.
                    # ONLY consider AIMessage / AIMessageChunk objects; skip ToolMessage,
                    # HumanMessage, and anything else that would produce Python-object repr.
                    output = event.get("data", {}).get("output", {})
                    all_msgs = output.get("messages", [])
                    final = ""
                    for msg in reversed(all_msgs):
                        msg_type = type(msg).__name__
                        if msg_type not in ("AIMessage", "AIMessageChunk"):
                            continue
                        content = getattr(msg, "content", None)
                        if not content:
                            continue
                        if isinstance(content, list):
                            # Content blocks: keep only plain-text parts
                            text_parts = [
                                b.get("text", "")
                                for b in content
                                if isinstance(b, dict) and b.get("type") == "text"
                            ]
                            text = " ".join(p for p in text_parts if p).strip()
                        elif isinstance(content, str):
                            text = content.strip()
                        else:
                            text = ""
                        if text:
                            final = text
                            break

                    if not final:
                        final = "Yanıt alınamadı. Lütfen tekrar deneyin."

                    yield _sse({"type": "done", "data": final})
                    got_done = True
                    break

        except Exception as e:
            print(f"Agent error: {e}")
            if not got_done:
                yield _sse(
                    {
                        "type": "error",
                        "data": "Danışman şu an yanıt veremiyor. Lütfen tekrar deneyin.",
                    }
                )

        if not got_done:
            yield _sse({"type": "error", "data": "Yanıt alınamadı. Lütfen tekrar deneyin."})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


# ─────────────────────────────────────────────────────────────
# SESSION RESET
# ─────────────────────────────────────────────────────────────
@app.delete("/session/{session_id}")
async def reset_session(session_id: str):
    sessions.pop(session_id, None)
    try:
        checkpointer.storage.pop(session_id, None)
    except Exception:
        pass
    return {"status": "ok"}


@app.get("/health")
async def health():
    return {"status": "ok", "model": "llama3.1:8b"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
