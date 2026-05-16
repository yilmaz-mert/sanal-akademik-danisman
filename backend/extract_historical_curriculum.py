#!/usr/bin/env python3
"""
extract_historical_curriculum.py

Parses 5 historical curriculum PDFs and writes one JSON per academic year.

Output files (backend/):
  mufredat_21_22.json
  mufredat_22_23.json
  mufredat_23_24.json
  mufredat_24_25.json
  mufredat_25_26.json
"""
import json
import re
import unicodedata
from pathlib import Path

import pdfplumber

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

# Matches: INF113, ING116-A, INF112-B, CNT120, ATA001, TUR002, etc.
_COURSE_CODE_RE = re.compile(r"^([A-Z]{2,6})\s?(\d{3,4}[A-Z]?(?:-[A-Z])?)\b")


def clean(value) -> str:
    """Collapse all whitespace (including newlines in merged cells) to single spaces."""
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def norm(text: str) -> str:
    """Lowercase + ASCII-fold Turkish characters for type comparison."""
    s = unicodedata.normalize("NFC", str(text or ""))
    s = s.replace("İ", "I").replace("ı", "i")
    s = s.lower()
    return s.translate(str.maketrans("ğüöşç", "guosc"))


def safe_num(text: str):
    """Parse a credit/ECTS string to float; '' on failure."""
    s = clean(text).replace(",", ".")
    try:
        return float(s)
    except (ValueError, TypeError):
        return 0.0


def resolve_type(cell_text: str):
    """Return canonical 'Zorunlu', 'Seçmeli', or None."""
    n = norm(cell_text)
    if n == "zorunlu":
        return "Zorunlu"
    if n == "secmeli":
        return "Seçmeli"
    return None


# ─────────────────────────────────────────────────────────────
# PDF Parser
# ─────────────────────────────────────────────────────────────

def parse_curriculum_pdf(pdf_path: str) -> list:
    """
    Extracts course rows from a per-course-syllabus PDF.

    Each course occupies several pages:
      - course-header row : CODE  |  NAME  |  SEMESTER  |  T U L  |  CREDIT  |  AKTS
      - Türü row           : Türü  |  Zorunlu/Seçmeli  (appears within same course block)

    Strategy:
      1. Collect all ("course", data) and ("type", value) events in page order.
      2. Assign each type event to the most-recent preceding course that has no type yet
         (sticky-pass: once a course has a type, later events are ignored for it).
      3. Deduplicate by course code — first occurrence wins.
    """
    events = []  # list of ("course", dict) or ("type", str)

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for tbl in page.extract_tables():
                if not tbl:
                    continue
                for row in tbl:
                    if not row:
                        continue

                    cells = [clean(c) for c in row]

                    c0  = cells[0]  if len(cells) > 0  else ""
                    c2  = cells[2]  if len(cells) > 2  else ""
                    c4  = cells[4]  if len(cells) > 4  else ""
                    c8  = cells[8]  if len(cells) > 8  else ""
                    c9  = cells[9]  if len(cells) > 9  else ""

                    # ── Course-data row? ────────────────────────────────────
                    # Heuristic: code matches, name in col-2, AKTS (col-9) starts with digit
                    m = _COURSE_CODE_RE.match(c0)
                    if m and c2 and re.match(r"^\d", c9):
                        code = (m.group(1) + m.group(2)).upper()
                        try:
                            donem: int | str = int(c4)
                        except (ValueError, TypeError):
                            donem = c4 if c4 else ""
                        events.append(("course", {
                            "kod":   code,
                            "ad":    c2,
                            "donem": donem,
                            "akts":  safe_num(c9),
                            "tip":   None,
                        }))
                        continue  # skip type-row check for course rows

                    # ── Type row? ───────────────────────────────────────────
                    # Type rows have ≤ 4 non-empty cells; one of them is "Zorunlu"/"Seçmeli".
                    non_empty = [c for c in cells if c]
                    if len(non_empty) <= 4:
                        for cell in non_empty:
                            t = resolve_type(cell)
                            if t:
                                events.append(("type", t))
                                break

    # ── Assign types to courses ────────────────────────────────────────────
    courses = []
    pending = None

    for kind, data in events:
        if kind == "course":
            if pending is not None:
                courses.append(pending)
            pending = dict(data)          # copy so we can mutate safely
        elif kind == "type" and pending is not None and pending["tip"] is None:
            pending["tip"] = data

    if pending is not None:
        courses.append(pending)

    # ── Deduplicate: first occurrence of each code ─────────────────────────
    seen: set = set()
    unique = []
    for c in courses:
        if c["kod"] in seen:
            continue
        seen.add(c["kod"])
        if c["tip"] is None:
            c["tip"] = "Zorunlu"   # safe fallback; review manually if needed
        unique.append(c)

    return unique


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

PDF_MAP = {
    "21-22.pdf": ("mufredat_21_22.json", "2021-2022"),
    "22-23.pdf": ("mufredat_22_23.json", "2022-2023"),
    "23-24.pdf": ("mufredat_23_24.json", "2023-2024"),
    "24-25.pdf": ("mufredat_24_25.json", "2024-2025"),
    "25-26.pdf": ("mufredat_25_26.json", "2025-2026"),
}

GRADUATION_REQUIREMENTS = {
    "toplam_akts_hedefi": 240,
    "minimum_gno": 2.00,
    "staj_zorunlulugu_gun": 60,
    "hazirlik_sarti": "Tamamlanmış olmalı",
}

# PDFs live at the project root (one level above this script's backend/ dir)
ROOT    = Path(__file__).parent.parent
BACKEND = Path(__file__).parent


def main() -> None:
    print("=" * 60)
    print("Historical Curriculum Extractor")
    print("=" * 60)

    all_ok = True
    for pdf_name, (json_name, yil) in PDF_MAP.items():
        pdf_path  = ROOT / pdf_name
        json_path = BACKEND / json_name

        print(f"\n[{yil}]  {pdf_name}")

        if not pdf_path.exists():
            print(f"  ERROR: {pdf_path} not found — skipping")
            all_ok = False
            continue

        try:
            courses = parse_curriculum_pdf(str(pdf_path))
        except Exception as exc:
            print(f"  ERROR during parsing: {exc}")
            all_ok = False
            continue

        zorunlu = [c for c in courses if c["tip"] == "Zorunlu"]
        secmeli = [c for c in courses if c["tip"] == "Seçmeli"]
        print(f"  Courses: {len(courses)} total — {len(zorunlu)} Zorunlu, {len(secmeli)} Seçmeli")

        dersler = [
            {
                "kod":   c["kod"],
                "ad":    c["ad"],
                "akts":  c["akts"],
                "tip":   c["tip"],
                "donem": c["donem"],
            }
            for c in courses
        ]

        output = {
            "universite":         "Galatasaray",
            "bolum":              "Bilgisayar Mühendisliği",
            "yil":                yil,
            "mezuniyet_sartlari": GRADUATION_REQUIREMENTS,
            "dersler":            dersler,
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=4)

        print(f"  Saved  : {json_path.name}")

        # Preview first 4 and last 2
        preview = dersler[:4] + (["..."] if len(dersler) > 6 else []) + dersler[-2:]
        for c in preview:
            if c == "...":
                print("    ...")
                continue
            print(f"    {c['kod']:15s} sem={str(c['donem']):3s}  akts={c['akts']:<5}  "
                  f"{c['tip']:<10s}  {c['ad'][:40]}")

    print("\n" + "=" * 60)
    if all_ok:
        print("All 5 files created successfully.")
    else:
        print("Some files could not be created (see errors above).")
    print("=" * 60)


if __name__ == "__main__":
    main()
