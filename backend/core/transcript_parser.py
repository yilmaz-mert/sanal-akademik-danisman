"""
Deterministic TXT transcript parser.
Python-only — no LLM calls here. The heuristic schema detector is reliable enough.
"""
import re
import unicodedata
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from core.curriculum_loader import (
    ALL_GRADES, GRADE_POINTS, PASSING_GRADES, FAILED_GRADES,
    is_passing, is_failed, normalize_str,
)

# ─────────────────────────────────────────────────────────────
# REGEX PATTERNS
# ─────────────────────────────────────────────────────────────
SEM_HEADER_RE = re.compile(
    r"(\d{4})\s*[-–—]\s*(\d{4})"
    r"(?:\s+E[gğ]itim[- ]?[ÖO][gğ]retim\s+Y[ıi]l[ıi])?"
    r"\s+(Güz|Bahar|Yaz|GÜZ|BAHAR|YAZ|Guz|SPRING|FALL|SUMMER)"
    r"(?:\s+(?:Yarıyılı?|Dönemi?|D[öo]nemi?|Semester|Term|Y[ıi]l[ıi]))?",
    re.IGNORECASE | re.UNICODE,
)

TERM_GPA_RE = re.compile(
    r"(?:ANO|Yar[iı]y[iı]l\s+Ortalamas[iı]|D[öo]nem\s+Ortalamas[iı]"
    r"|D[öo]nem\s+GNO|Yar[iı]y[iı]l\s+GNO)"
    r"\s*[:\s]?\s*([\d.,]+)",
    re.IGNORECASE,
)
_TERM_GPA_BARE_RE = re.compile(r"Yar[iı]y[iı]l\s+([\d.,]+)", re.IGNORECASE)

CUMULATIVE_GPA_RE = re.compile(
    r"(?:Genel\s+GNO|CGPA|Genel\s+Ortalama|Kümülatif\s+Ortalama|^Genel)"
    r"\s*[:\s]\s*([\d.,]+)",
    re.IGNORECASE | re.MULTILINE,
)

COURSE_CODE_RE = re.compile(r"^([A-Z]{2,6})\s?(\d{3,4}[A-Z]?)\b")

_STATUS_WORDS = {"gecti", "kaldi", "kalti", "basarili", "basarisiz", "passed", "failed"}
_TYPE_TOKENS = {"Z", "S", "Y", "SE", "ZE", "ME", "SE1", "SE2", "T"}

_SKIP_LINE_RE = re.compile(
    r"Kodu|Ders\s*Ad|Not\s*u|Kredi|AKTS|Ortalama|ANO|T\s+U\s+L|Yarıyıl|D[öo]nem\s+No",
    re.IGNORECASE | re.UNICODE,
)

_SUFFIX_RE = re.compile(r"-[A-Z]$")


def strip_suffix(code: str) -> str:
    return _SUFFIX_RE.sub("", code.strip())


# ─────────────────────────────────────────────────────────────
# SCHEMA DETECTION  (heuristic only — reliable and fast)
# ─────────────────────────────────────────────────────────────
class TranscriptSchemaConfig(BaseModel):
    grade_position: Literal["after_numbers", "before_numbers"] = Field(
        default="after_numbers",
    )
    has_status_word: bool = Field(default=True)
    has_type_column: bool = Field(default=True)
    has_row_number: bool = Field(default=False)


_FALLBACK_SCHEMA = TranscriptSchemaConfig()


def _detect_schema_heuristic(sample_lines: List[str]) -> TranscriptSchemaConfig:
    after_votes = 0
    before_votes = 0
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
                break

    grade_pos: Literal["after_numbers", "before_numbers"] = (
        "before_numbers" if before_votes > after_votes else "after_numbers"
    )
    print(
        f"[SCHEMA] grade_pos={grade_pos!r} "
        f"(after={after_votes}, before={before_votes})"
    )
    return TranscriptSchemaConfig(
        grade_position=grade_pos,
        has_status_word=has_status,
        has_type_column=has_type,
        has_row_number=has_row_num,
    )


def discover_transcript_schema(sample_text: str) -> TranscriptSchemaConfig:
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
        return _FALLBACK_SCHEMA
    return _detect_schema_heuristic(candidates)


# ─────────────────────────────────────────────────────────────
# LINE PARSER
# ─────────────────────────────────────────────────────────────
def _parse_course_line(
    line: str,
    schema: Optional[TranscriptSchemaConfig] = None,
) -> Optional[dict]:
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

    grade = None
    grade_idx = None
    for i in range(len(tokens) - 1, -1, -1):
        t = tokens[i].upper().rstrip(".")
        if t in ALL_GRADES:
            grade = t
            grade_idx = i
            break

    if grade is None or grade_idx is None:
        return None

    before = tokens[:grade_idx]
    after_grade = tokens[grade_idx + 1:]

    nums_after: List[float] = []
    for tok in after_grade:
        if normalize_str(tok) in _STATUS_WORDS:
            break
        try:
            nums_after.append(float(tok.replace(",", ".")))
        except ValueError:
            break

    use_before = schema.grade_position == "before_numbers" and len(nums_after) >= 1

    if use_before:
        akts = nums_after[-1]
        credit = nums_after[-2] if len(nums_after) >= 2 else akts

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
            return None

        akts = numbers[-1]
        credit = numbers[-2] if len(numbers) >= 2 else akts
        tip = ""
        if ne >= 0 and before[ne].upper() in _TYPE_TOKENS:
            tip = before[ne].upper()
            ne -= 1
        name = " ".join(before[:ne + 1]).strip()

    if not name:
        return None

    return {
        "kod": code,
        "ad": name,
        "tip": tip,
        "kredi": round(credit, 1),
        "akts": round(akts, 1),
        "not": grade,
        "puan": GRADE_POINTS.get(grade, None),
        "durum": "Geçti" if is_passing(grade) else "Kaldı",
    }


def _extract_term_gpa(block: str) -> Optional[float]:
    m = TERM_GPA_RE.search(block)
    if m:
        try:
            return float(m.group(1).replace(",", "."))
        except ValueError:
            pass
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
    courses: List[dict] = []
    for raw_line in block.splitlines():
        if _SKIP_LINE_RE.search(raw_line):
            continue
        parsed = _parse_course_line(raw_line, schema)
        if parsed:
            courses.append(parsed)
    return courses


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
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────
def parse_transcript_txt(raw_bytes: bytes) -> dict:
    text = _decode_txt(raw_bytes)
    print(f"\n[PARSER] Transcript length: {len(text)} chars")

    schema = discover_transcript_schema(text[:4000])
    print(f"[PARSER] Schema: grade_pos={schema.grade_position!r}")

    raw_code_matches = re.findall(r"\b([A-Z]{2,6})\s?(\d{3,4}[A-Z]?)\b", text)
    all_codes = list(set((g[0] + g[1]).upper() for g in raw_code_matches))

    sem_matches = list(SEM_HEADER_RE.finditer(text))
    print(f"[PARSER] Semester headers found: {len(sem_matches)}")
    semesters: List[dict] = []

    if sem_matches:
        for i, sm in enumerate(sem_matches):
            year_s, year_e = sm.group(1), sm.group(2)
            term = sm.group(3).capitalize()
            label = f"{year_s} - {year_e} {term}"
            start = sm.end()
            end = sem_matches[i + 1].start() if i + 1 < len(sem_matches) else len(text)
            block = text[start:end]

            courses = _collect_courses_from_block(block, schema)
            term_gpa = _extract_term_gpa(block)
            term_ects = sum(c["akts"] for c in courses if c["durum"] == "Geçti")

            semesters.append({
                "semester_name": label,
                "courses": courses,
                "term_gpa": term_gpa,
                "term_ects": term_ects,
            })

    # Flat-parse fallback
    total_parsed = sum(len(s["courses"]) for s in semesters)
    if total_parsed == 0:
        flat = _collect_courses_from_block(text, schema)
        if flat:
            semesters = [{
                "semester_name": "Transkript (Düz Tarama)",
                "courses": flat,
                "term_gpa": None,
                "term_ects": sum(c["akts"] for c in flat if c["durum"] == "Geçti"),
            }]

    # Cumulative GPA — take the LAST match to get the final value
    cumulative_gpa: Optional[float] = None
    all_gpa_hits = CUMULATIVE_GPA_RE.findall(text)
    if all_gpa_hits:
        try:
            cumulative_gpa = float(all_gpa_hits[-1].replace(",", "."))
        except ValueError:
            pass

    if cumulative_gpa is None and semesters:
        gpas = [s["term_gpa"] for s in semesters if s["term_gpa"] is not None]
        if gpas:
            cumulative_gpa = round(sum(gpas) / len(gpas), 2)

    print(f"[PARSER] Done: {len(semesters)} semester(s), gpa={cumulative_gpa}\n")
    return {
        "semesters": semesters,
        "all_codes": all_codes,
        "cumulative_gpa": cumulative_gpa,
        "schema_used": schema.grade_position,
    }


def detect_entry_year(semesters: list) -> str:
    """Return 'YYYY-YYYY' string for the student's first enrolled academic year."""
    start_years: List[int] = []
    for sem in semesters:
        m = re.search(r"(\d{4})\s*[-–—]\s*(\d{4})", sem.get("semester_name", ""))
        if m:
            start_years.append(int(m.group(1)))
    if start_years:
        min_year = min(start_years)
        return f"{min_year}-{min_year + 1}"
    return "2025-2026"
