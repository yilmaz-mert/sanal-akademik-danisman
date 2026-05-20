"""
Dumb data-fetching tools for the MAS agents.
Python logic lives here ONLY to load raw data — all reasoning is done by the LLMs.

Tool output design:
- All outputs are compact plain text (~300-600 tokens) — never raw JSON dumps.
- Transcript tool deduplicates suffix variants before the LLM sees the data:
  if a student passed ING116-A, any earlier ING116 failure is hidden entirely.
"""
from typing import List

from langchain_core.tools import StructuredTool

from core.curriculum_loader import load_intibak_rules, load_year_curriculum, strip_suffix


# ─────────────────────────────────────────────────────────────
# STATIC TOOLS  (no session binding required)
# ─────────────────────────────────────────────────────────────

def _get_curriculum_by_year(year: str = "2025-2026") -> str:
    """Verilen akademik yıla ait zorunlu dersleri kompakt metin olarak döndür.

    Args:
        year: Akademik yıl (örn: '2021-2022', '2022-2023', '2025-2026')
    """
    try:
        data = load_year_curriculum(year)
        sartlar = data.get("mezuniyet_sartlari", {})
        mandatory = [c for c in data.get("dersler", []) if c.get("tip") == "Zorunlu"]
        parts = [f"{c['kod']} ({c.get('akts', 0)} AKTS)" for c in mandatory]
        return (
            f"Yıl: {data.get('yil', year)} | "
            f"AKTS Hedefi: {sartlar.get('toplam_akts_hedefi', 240)} | "
            f"Min GNO: {sartlar.get('minimum_gno', 2.0)}\n"
            f"Zorunlu: {', '.join(parts)}"
        )
    except Exception as e:
        return f"Hata: {e}"


def _get_intibak_rules(_dummy: str = "") -> str:
    """Tüm intibak eşdeğerlik kurallarını kompakt metin olarak döndür."""
    try:
        data = load_intibak_rules()
        seen: set = set()
        mappings: List[str] = []

        for old_code, new_list in data.get("eski_yeni_eslestirme", {}).items():
            if old_code not in seen:
                seen.add(old_code)
                mappings.append(f"{old_code} -> {', '.join(new_list)}")

        for r in data.get("intibak_kurallari", []):
            eski = r.get("eski_ders", {}).get("kod", "")
            yeni_list = [y.get("kod", "") for y in r.get("yeni_dersler", [])]
            if eski and yeni_list and eski not in seen:
                seen.add(eski)
                kosul = r.get("kosul_yili_oncesi", "")
                suffix = f" [{kosul} ve öncesi]" if kosul else ""
                mappings.append(f"{eski} -> {', '.join(yeni_list)}{suffix}")

        if not mappings:
            return "İntibak kuralı bulunamadı."
        return "İntibak Kuralları (Eski -> Yeni):\n" + "\n".join(mappings)
    except Exception as e:
        return f"Hata: {e}"


get_curriculum_by_year = StructuredTool.from_function(
    func=_get_curriculum_by_year,
    name="get_curriculum_by_year",
    description=(
        "Verilen akademik yıla ait zorunlu ders listesini ve mezuniyet koşullarını döndür. "
        "Parametre: year (örn: '2021-2022')"
    ),
)

get_intibak_rules = StructuredTool.from_function(
    func=_get_intibak_rules,
    name="get_intibak_rules",
    description=(
        "Tüm intibak ve eşdeğerlik kurallarını kompakt metin olarak döndür. "
        "Hangi eski dersin hangi yeni dersin yerine geçtiğini gösterir."
    ),
)


# ─────────────────────────────────────────────────────────────
# SESSION-BOUND TRANSCRIPT TOOL FACTORY
# ─────────────────────────────────────────────────────────────

def make_transcript_tool(transcript_raw: dict) -> StructuredTool:
    """Return a StructuredTool bound to the session's parsed transcript."""

    def _get_transcript_data(_dummy: str = "") -> str:
        """Öğrencinin ders geçmişini kompakt metin olarak döndür.

        Suffix varyantları tekilleştirildi: aynı temel koda sahip bir geçme
        varsa önceki başarısızlar tamamen gizlenir.
        Örnek: ING116 → FF, ING116-A → CC  ⟹  ING116 GEÇILDI (FF görünmez).
        """
        # ── Pass 1: group every course record by base code ─────
        # passed[base] = best (most recent) passing entry
        # failed[base] = most recent failing entry
        passed: dict = {}
        failed: dict = {}

        for sem in transcript_raw.get("semesters", []):
            for c in sem.get("courses", []):
                raw_kod = c.get("kod", "").upper().strip()
                base = strip_suffix(raw_kod)
                entry = {
                    "ad":   c.get("ad", ""),
                    "not":  c.get("not", ""),
                    "akts": c.get("akts", 0),
                }
                if c.get("durum") == "Geçti":
                    passed[base] = entry   # later passes overwrite earlier ones
                else:
                    failed[base] = entry   # keep the latest failure per base

        # ── Pass 2: build compact output ───────────────────────
        gpa = transcript_raw.get("cumulative_gpa")
        lines: List[str] = []
        lines.append(f"GNO: {gpa:.2f}" if isinstance(gpa, (int, float)) else "GNO: Bilinmiyor")

        if passed:
            parts = [
                f"{base} ({v['ad']}, {v['akts']} AKTS)"
                for base, v in passed.items()
            ]
            lines.append(f"Geçilen Dersler: {', '.join(parts)}")
        else:
            lines.append("Geçilen Dersler: Yok")

        # Only expose failures for courses the student never subsequently passed
        truly_failed = {base: e for base, e in failed.items() if base not in passed}
        if truly_failed:
            parts = [
                f"{base} ({e['ad']}, not: {e['not']})"
                for base, e in truly_failed.items()
            ]
            lines.append(f"Başarısız/Eksik Dersler (Güncel): {', '.join(parts)}")
        else:
            lines.append("Başarısız/Eksik Dersler (Güncel): Yok")

        return "\n".join(lines)

    return StructuredTool.from_function(
        func=_get_transcript_data,
        name="get_transcript_data",
        description=(
            "Öğrencinin ders geçmişini kompakt metin olarak döndür. "
            "Suffix varyantları tekilleştirildi — ING116-A geçildiyse ING116 geçilmiş sayılır, "
            "önceki FF görünmez."
        ),
    )


# ─────────────────────────────────────────────────────────────
# REPORTING TOOL FACTORIES  (capture structured output from ReAct loop)
# ─────────────────────────────────────────────────────────────

def make_curriculum_reporting_tool(captured: dict) -> StructuredTool:
    """Curriculum_Agent's submit tool — saves found missing/unrecognized codes."""

    def _report(
        missing_mandatory_codes: List[str],
        unrecognized_student_codes: List[str],
        summary: str,
    ) -> str:
        """Müfredat analizini tamamladıktan sonra bulgularını bu araçla kaydet.

        Args:
            missing_mandatory_codes: Öğrencinin hiç geçmediği zorunlu ders kodları listesi
            unrecognized_student_codes: Öğrencinin aldığı fakat müfredatta bulunmayan ders kodları
            summary: Türkçe analiz özeti (1-2 cümle)
        """
        captured["missing_codes"] = missing_mandatory_codes
        captured["unrecognized_codes"] = unrecognized_student_codes
        captured["summary"] = summary
        return "Bulgular başarıyla kaydedildi. Analiz tamamlandı."

    return StructuredTool.from_function(
        func=_report,
        name="report_curriculum_findings",
        description=(
            "ZORUNLU: Müfredat analizi bitince bu aracı çağır ve bulgularını kaydet. "
            "missing_mandatory_codes: zorunlu müfredattan eksik ders kodları. "
            "unrecognized_student_codes: öğrencinin aldığı ama müfredatta olmayan kodlar."
        ),
    )


def make_intibak_reporting_tool(captured: dict) -> StructuredTool:
    """Intibak_Agent's submit tool — saves which codes were cleared."""

    def _report(
        cleared_codes: List[str],
        summary: str,
    ) -> str:
        """İntibak analizi tamamlandıktan sonra muafiyet kararlarını kaydet.

        Args:
            cleared_codes: Eşdeğerlik kuralı gereği eksik listesinden çıkarılacak ders kodları
            summary: Türkçe açıklama
        """
        captured["cleared_codes"] = cleared_codes
        captured["summary"] = summary
        return "İntibak bulguları kaydedildi."

    return StructuredTool.from_function(
        func=_report,
        name="report_intibak_findings",
        description=(
            "ZORUNLU: İntibak analizi bitince bu aracı çağır. "
            "cleared_codes: eşdeğerlik kuralıyla temizlenen ders kodları listesi."
        ),
    )


def make_audit_reporting_tool(captured: dict) -> StructuredTool:
    """Audit_Agent's submit tool — saves the graduation decision."""

    def _report(
        graduation_status: str,
        assessment: str,
    ) -> str:
        """Mezuniyet denetimi kararını kaydet.

        Args:
            graduation_status: 'Mezun Olabilir' VEYA 'Mezun Olamaz (sebep)'
            assessment: Türkçe detaylı değerlendirme
        """
        captured["graduation_status"] = graduation_status
        captured["assessment"] = assessment
        return "Mezuniyet kararı kaydedildi."

    return StructuredTool.from_function(
        func=_report,
        name="report_audit_decision",
        description=(
            "ZORUNLU: Denetim bitince bu aracı çağır ve kararı kaydet. "
            "graduation_status: 'Mezun Olabilir' veya 'Mezun Olamaz (sebep)'."
        ),
    )
