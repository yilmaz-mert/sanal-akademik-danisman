"""
extract_curriculum_rules.py
────────────────────────────
Extracts equivalency/substitution rules (Intibak Kurallari), general rules,
and graduation requirements from all 5 Akademik Takvim PDFs.

Output: backend/intibak_kurallari.json

Notes on encoding:
  pdfplumber returns proper Unicode strings. Turkish chars (ö, ü, ş, etc.)
  are present as real code-points; PowerShell's display renders them as '?'
  due to its code-page, but Python string comparisons and regex work correctly.
"""

import json
import re
from pathlib import Path

import pdfplumber

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
PDF_FILES = {
    "2021-2022": ROOT / "Akademik_Takvim_2021_2022.pdf",
    "2022-2023": ROOT / "Akademik_Takvim_2022_2023.pdf",
    "2023-2024": ROOT / "Akademik_Takvim_2023-2024.pdf",
    "2024-2025": ROOT / "Akademik_Takvim_2024_2025.pdf",
    "2025-2026": ROOT / "Akademik_Takvim_2025_2026.pdf",
}
OUTPUT = ROOT / "backend" / "intibak_kurallari.json"


# ── Core regexes ───────────────────────────────────────────────────────────────
# D[öo]nem: ö is U+00F6; also matches ASCII 'o' as fallback for any garbled case
SEMESTER_RE = re.compile(r"D[öo]nem\s+(\d+)", re.IGNORECASE)

# Matches "Not 1:", "Not 2:", "Not 10", "Not 1 :" etc. at start of any line.
# Trailing colon is optional (2024-2025 has no colon).
NOTE_RE = re.compile(r"^Not\s+(\d+)\s*:?", re.IGNORECASE | re.MULTILINE)

# "Genel Kural:" / "Genel Kurallar:" / "GENEL KURALLAR"
GEN_RULE_RE = re.compile(r"(?:GENEL\s+KURALLAR?|Genel\s+Kurallar?)\s*:?", re.IGNORECASE)

# Graduation section header
GRAD_RE = re.compile(r"MEZUN.{0,3}YET|D[öo]nem\s+8", re.IGNORECASE)

# Year condition: "2020-2021 öğretim yılı ve öncesinde"
YEAR_COND_RE = re.compile(r"(\d{4}-\d{4})")

# Course code: ING104, INF 102, CNT120, MAT 301, ING106, etc.
COURSE_CODE_PAT = r"[A-Z]{2,5}\s?\d{3}[A-Z]?"
COURSE_CODE_RE  = re.compile(r"\b(" + COURSE_CODE_PAT + r")\b")

# Bullet markers used in 2025-2026 format
BULLET_RE = re.compile(r"[\x95•●‣\xb7]\s*|\n\s*(?=[A-Z]{2,5}\s*\d{3})")


def clean_code(code: str) -> str:
    return re.sub(r"\s+", "", code).upper()


# ── Text extraction ────────────────────────────────────────────────────────────
def extract_full_text(pdf_path: Path) -> str:
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            pages.append(t)
    return "\n".join(pages)


# ── Mapping extraction (old_course → new_courses) ────────────────────────────
def _extract_mappings_from_text(text: str) -> list:
    """
    Extract all (old_code, old_name, [new_codes]) triples from a text block.
    Handles three formats:
      1. Sentence: «ING104 Matematik I» dersinden kalan ... yerine «ING106...» dersini alırlar.
      2. Colon: ING104 Matematik I dersinden kalan ... yerine ING106 dersini alırlar.
      3. Bullet (2025-2026): • ING104 Matematik I dersinden kalanlar, yerine ING106...,
    """
    mappings = []
    seen_old = set()

    # Find every "dersinden kalan" occurrence and work outward from each
    for m_ders in re.finditer(r"\bdersinden\s+kalan", text, re.IGNORECASE):
        # ── Find old course code (look 200 chars back) ──
        lookback = text[max(0, m_ders.start() - 200): m_ders.start()]
        # Within lookback, only search after the last newline (avoid crossing into
        # a previous note's content if notes aren't well-separated)
        last_nl = lookback.rfind("\n")
        search_segment = lookback[last_nl + 1:]

        old_matches = list(COURSE_CODE_RE.finditer(search_segment))
        if not old_matches:
            continue
        old_m = old_matches[-1]  # last code before "dersinden kalan" on this line
        old_kod = clean_code(old_m.group(1))
        if old_kod in seen_old:
            continue

        # Extract name between code end and "dersinden" start
        name_raw = search_segment[old_m.end():]
        # Strip non-word characters (quotes, guillemets, parentheses…)
        name_clean = re.sub(r"[^\w\s]", " ", name_raw).strip()
        old_ad = re.sub(r"\s+", " ", name_clean).strip()

        # ── Find "yerine" after "dersinden kalan" (look 400 chars forward) ──
        look_fwd = text[m_ders.end(): m_ders.end() + 400]
        yerine_m = re.search(r"\byerine\b", look_fwd, re.IGNORECASE)
        if not yerine_m:
            continue

        after_yerine = look_fwd[yerine_m.end():]

        # ── Find end of new-course list ──
        # Terminal: "derslerini" or "dersini" followed by comma, period, or "al..."
        stop_m = re.search(
            r"derslerini\s*[,.\n]|dersini\s*[,.\n]|derslerini\s+al|dersini\s+al",
            after_yerine,
            re.IGNORECASE,
        )
        new_block = after_yerine[: stop_m.end() if stop_m else min(300, len(after_yerine))]

        new_codes = [clean_code(cm.group(1)) for cm in COURSE_CODE_RE.finditer(new_block)]
        if not new_codes:
            continue

        seen_old.add(old_kod)
        mappings.append({
            "eski": {"kod": old_kod, "ad": old_ad},
            "yeni": [{"kod": c, "ad": ""} for c in new_codes],
        })

    return mappings


# ── Note parsing ──────────────────────────────────────────────────────────────
def parse_notes(text: str, academic_year: str, semester_no: int) -> list:
    """Parse all 'Not N' entries in a semester text block."""
    notes = []
    positions = [
        (m.start(), m.end(), int(m.group(1))) for m in NOTE_RE.finditer(text)
    ]

    for i, (start, body_start, note_no) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        body = text[body_start:end].strip()

        # Clean description
        description = re.sub(r"\s{2,}", " ", " ".join(body.splitlines()))

        # Year condition
        cond_m = YEAR_COND_RE.search(body)
        kosul_yili = cond_m.group(1) if cond_m else None

        mappings = _extract_mappings_from_text(body)

        notes.append({
            "akademik_yil":      academic_year,
            "donem":             semester_no,
            "not_no":            note_no,
            "kosul_yili_oncesi": kosul_yili,
            "eslestirmeler":     mappings,
            "aciklama":          description[:500],
        })

    return notes


# ── General rules extraction ──────────────────────────────────────────────────
def extract_general_rules(text: str) -> list:
    """Extract numbered/bullet general rules from GENEL KURALLAR sections."""
    rules = []
    in_block = False
    for line in text.splitlines():
        s = line.strip()
        if GEN_RULE_RE.match(s):
            in_block = True
            continue
        if in_block:
            if SEMESTER_RE.match(s) or NOTE_RE.match(s) or s.startswith("Sayfa"):
                in_block = False
                continue
            m = re.match(r"^(\d+)[.)]\s+(.+)", s)
            b = re.match(r"^[*\-\xb7\x95•]\s+(.+)", s)
            if m:
                rule = m.group(2).strip()
                if rule and rule not in rules:
                    rules.append(rule)
            elif b:
                rule = b.group(1).strip()
                if rule and rule not in rules:
                    rules.append(rule)
    return rules


# ── Graduation requirements ───────────────────────────────────────────────────
def extract_graduation_reqs(text: str) -> list:
    """Extract graduation requirements from the MEZUNIYET / Dönem 8 section."""
    reqs = []
    in_block = False
    for line in text.splitlines():
        s = line.strip()
        if GRAD_RE.search(s):
            in_block = True
            continue
        if in_block:
            if not s or s.startswith("Sayfa"):
                continue
            m = re.match(r"^(\d+)[.)]\s+(.+)", s)
            b = re.match(r"^[*\-\xb7\x95•]\s+(.+)", s)
            akts = re.match(r"^AKTS\s+toplam", s, re.IGNORECASE)
            if m:
                req = m.group(2).strip()
                if req and req not in reqs:
                    reqs.append(req)
            elif b:
                req = b.group(1).strip()
                if req and req not in reqs:
                    reqs.append(req)
            elif akts:
                if s not in reqs:
                    reqs.append(s)
    return reqs


# ── Per-PDF processing ────────────────────────────────────────────────────────
def process_pdf(academic_year: str, pdf_path: Path) -> dict:
    print(f"  [{academic_year}] {pdf_path.name}")
    full_text = extract_full_text(pdf_path)

    all_notes = []

    # Split into semester blocks
    sem_positions = [
        (m.start(), int(m.group(1))) for m in SEMESTER_RE.finditer(full_text)
    ]
    print(f"    Semester blocks found: {len(sem_positions)}")

    for s_idx, (s_start, sem_no) in enumerate(sem_positions):
        s_end = (
            sem_positions[s_idx + 1][0]
            if s_idx + 1 < len(sem_positions)
            else len(full_text)
        )
        sem_block = full_text[s_start:s_end]
        notes = parse_notes(sem_block, academic_year, sem_no)
        all_notes.extend(notes)

    general_rules = extract_general_rules(full_text)
    graduation    = extract_graduation_reqs(full_text)

    notes_with_mapping = sum(1 for n in all_notes if n["eslestirmeler"])
    print(
        f"    -> {len(all_notes)} not, "
        f"{notes_with_mapping} eski->yeni eslestirme, "
        f"{len(general_rules)} genel kural, "
        f"{len(graduation)} mezuniyet sarti"
    )

    return {
        "akademik_yil":       academic_year,
        "kaynak_dosya":       pdf_path.name,
        "notlar":             all_notes,
        "genel_kurallar":     general_rules,
        "mezuniyet_sartlari": graduation,
    }


# ── Deduplication across years ────────────────────────────────────────────────
def build_equivalency_map(all_years: list) -> list:
    """
    Deduplicate intibak rules across all years.
    Key = (eski_kod, tuple of sorted yeni_kodlar).
    """
    seen = {}
    for year_data in all_years:
        yil = year_data["akademik_yil"]
        for note in year_data["notlar"]:
            for mapping in note["eslestirmeler"]:
                eski = mapping["eski"]
                yeni_kodlar = [y["kod"] for y in mapping["yeni"]]
                key = (eski["kod"], tuple(sorted(yeni_kodlar)))
                if key not in seen:
                    seen[key] = {
                        "eski_ders":         eski,
                        "yeni_dersler":      mapping["yeni"],
                        "kosul_yili_oncesi": note["kosul_yili_oncesi"],
                        "ilk_gorulen_yil":   yil,
                        "son_gorulen_yil":   yil,
                        "aciklama":          note["aciklama"],
                    }
                else:
                    seen[key]["son_gorulen_yil"] = yil

    return sorted(
        seen.values(),
        key=lambda x: (x.get("kosul_yili_oncesi") or "9999", x["eski_ders"]["kod"]),
    )


def build_quick_lookup(equivalency_map: list) -> dict:
    """eski_kod -> [yeni_kod, ...] for O(1) lookups by the advisor AI."""
    lookup: dict = {}
    for rule in equivalency_map:
        eski_kod  = rule["eski_ders"]["kod"]
        yeni_list = [y["kod"] for y in rule["yeni_dersler"]]
        if eski_kod not in lookup:
            lookup[eski_kod] = yeni_list
        else:
            for k in yeni_list:
                if k not in lookup[eski_kod]:
                    lookup[eski_kod].append(k)
    return lookup


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    print("Akademik Takvim Intibak Kural Cikartici")
    print("=" * 50)

    all_years = []
    for academic_year, pdf_path in sorted(PDF_FILES.items()):
        if not pdf_path.exists():
            print(f"  [ATLA] {pdf_path.name} bulunamadi")
            continue
        all_years.append(process_pdf(academic_year, pdf_path))

    equivalency_map = build_equivalency_map(all_years)
    quick_lookup    = build_quick_lookup(equivalency_map)

    # Unique general rules across all years
    all_general_rules: list = []
    seen_rules: set = set()
    for y in all_years:
        for r in y["genel_kurallar"]:
            if r not in seen_rules:
                seen_rules.add(r)
                all_general_rules.append(r)

    # Graduation requirements: use latest non-empty
    mezuniyet: list = []
    for y in reversed(all_years):
        if y["mezuniyet_sartlari"]:
            mezuniyet = y["mezuniyet_sartlari"]
            break

    output = {
        "meta": {
            "kaynak": (
                "Galatasaray Universitesi Bilgisayar Muhendisligi Bolumu "
                "Akademik Takvim & Uygulama Esaslari"
            ),
            "islenen_yillar": [y["akademik_yil"] for y in all_years],
            "toplam_intibak_kurali": len(equivalency_map),
            "aciklama": (
                "Bu dosya, mufredat degisikliklerinde hangi eski dersin hangi yeni ders(ler)in "
                "yerine gectigi (intibak/esdegerlik kurallari), genel kayit kurallari ve "
                "mezuniyet sartlarini icerir."
            ),
        },
        # Primary quick-lookup: eski_kod -> [yeni_kod, ...]
        "eski_yeni_eslestirme": quick_lookup,
        # Full deduplicated rules with context
        "intibak_kurallari": equivalency_map,
        # Unique general rules across all years
        "genel_kurallar": all_general_rules,
        # Latest graduation requirements
        "mezuniyet_sartlari": mezuniyet,
        # Per-year raw detail for deep queries
        "yil_bazli_detay": all_years,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    size_kb = OUTPUT.stat().st_size / 1024
    print(f"\nCikti: {OUTPUT}")
    print(f"  {len(equivalency_map):3d} esdegerlik kurali (deduplikasyon sonrasi)")
    print(f"  {len(all_general_rules):3d} genel kural")
    print(f"  {len(mezuniyet):3d} mezuniyet sarti")
    print(f"  {size_kb:.1f} KB")

    print("\nIlk 15 eski->yeni kodu eslestirmesi:")
    for eski_kod, yeni_list in list(quick_lookup.items())[:15]:
        print(f"  {eski_kod:12s} -> {', '.join(yeni_list)}")

    print("\nTamamlandi.")


if __name__ == "__main__":
    main()
