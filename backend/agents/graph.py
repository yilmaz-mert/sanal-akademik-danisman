"""
LangGraph Supervisor Multi-Agent System.

Graph topology:
    START → supervisor → curriculum_agent → supervisor
                       → intibak_agent    → supervisor
                       → audit_agent      → supervisor → END

Each specialist is a create_react_agent that calls its dumb data tools and then
calls a reporting tool to store structured findings back into the state.
The Supervisor is a Groq LLM with structured output that decides routing.
"""
import json
from typing import Dict, List, Literal, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel

from agents.prompts import (
    AUDIT_AGENT_PROMPT,
    CURRICULUM_AGENT_PROMPT,
    INTIBAK_AGENT_PROMPT,
    SUPERVISOR_PROMPT,
)
from agents.state import AdvisorState
from core.config import get_llm
from core.curriculum_loader import (
    build_lookup,
    load_latest_curriculum,
    load_year_curriculum,
    strip_suffix,
)
from tools.academic_tools import (
    get_curriculum_by_year,
    get_intibak_rules,
    make_audit_reporting_tool,
    make_curriculum_reporting_tool,
    make_intibak_reporting_tool,
    make_transcript_tool,
)

# ─────────────────────────────────────────────────────────────
# STRUCTURED OUTPUT SCHEMA FOR SUPERVISOR
# ─────────────────────────────────────────────────────────────
class RoutingDecision(BaseModel):
    next: Literal["curriculum_agent", "intibak_agent", "audit_agent", "__end__"]
    mesaj: str  # Turkish status message shown in SSE


# ─────────────────────────────────────────────────────────────
# NODE 4: SUPERVISOR  (the LLM router)
# ─────────────────────────────────────────────────────────────
async def supervisor_node(state: AdvisorState) -> dict:
    iteration = state.get("iteration", 0)

    # Safety guard — prevent runaway loops
    if iteration >= 8:
        return {
            "next_agent": "__end__",
            "iteration": iteration + 1,
            "status_queue": ["[🧠 Süpervizör] Güvenlik sınırına ulaşıldı, analiz sonlandırılıyor."],
        }

    missing_done = state.get("missing_courses") is not None
    intibak_done = state.get("intibak_cleared") is not None
    audit_done = bool(state.get("graduation_status"))

    context = (
        f"Müfredat analizi: {'TAMAMLANDI' if missing_done else 'YAPILMADI'}\n"
        f"İntibak kontrolü: {'TAMAMLANDI' if intibak_done else 'YAPILMADI'}\n"
        f"Mezuniyet kararı: {'VERİLDİ' if audit_done else 'VERİLMEDİ'}\n"
    )

    # Try LLM routing (as requested by the architecture spec)
    try:
        model = get_llm(temperature=0).with_structured_output(RoutingDecision)
        decision: RoutingDecision = await model.ainvoke([
            SystemMessage(content=SUPERVISOR_PROMPT),
            HumanMessage(content=context + "\nSonraki ajanı belirle."),
        ])
        next_agent = decision.next
        status_msg = f"[🧠 Süpervizör] {decision.mesaj}"
    except Exception as e:
        print(f"[SUPERVISOR] LLM routing failed ({e!r}), using deterministic fallback")
        # Deterministic fallback — identical to what the LLM would decide
        if not missing_done:
            next_agent = "curriculum_agent"
            status_msg = "[🧠 Süpervizör] Transkript analizi başlatıldı, Müfredat Uzmanına yönlendiriliyor..."
        elif not intibak_done:
            next_agent = "intibak_agent"
            status_msg = "[🧠 Süpervizör] Müfredat analizi tamamlandı, İntibak Avukatına yönlendiriliyor..."
        elif not audit_done:
            next_agent = "audit_agent"
            status_msg = "[🧠 Süpervizör] İntibak kontrolü tamamlandı, Mezuniyet Denetçisine yönlendiriliyor..."
        else:
            next_agent = "__end__"
            status_msg = "[🧠 Süpervizör] Tüm analizler tamamlandı, sonuçlar hazırlanıyor."

    return {
        "next_agent": next_agent,
        "iteration": iteration + 1,
        "status_queue": [status_msg],
    }


def _route_by_supervisor(state: AdvisorState) -> str:
    return state.get("next_agent", "__end__")


# ─────────────────────────────────────────────────────────────
# NODE 1: CURRICULUM AGENT
# ─────────────────────────────────────────────────────────────
async def curriculum_agent_node(state: AdvisorState) -> dict:
    transcript_raw = state["transcript_raw"]
    entry_year = state.get("entry_year", "2025-2026")

    captured: Dict = {}

    tools = [
        make_transcript_tool(transcript_raw),
        get_curriculum_by_year,
        make_curriculum_reporting_tool(captured),
    ]

    agent = create_react_agent(
        model=get_llm(temperature=0),
        tools=tools,
        prompt=CURRICULUM_AGENT_PROMPT,
    )

    input_msg = (
        f"Öğrenci giriş yılı: {entry_year}\n"
        f"1. get_transcript_data çağır → geçilen dersleri gör\n"
        f"2. get_curriculum_by_year(year='{entry_year}') çağır → zorunlu müfredatı al\n"
        f"3. Eksik ve tanınmayan ders kodlarını belirle\n"
        f"4. HEMEN report_curriculum_findings çağır — başka metin yazma, sadece aracı çağır."
    )

    try:
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=input_msg)]},
            config={"recursion_limit": 20},
        )
        agent_summary = result["messages"][-1].content if result.get("messages") else ""
    except Exception as e:
        print(f"[CURRICULUM-AGENT] ReAct error: {e!r}")
        agent_summary = f"Müfredat analizi tamamlanamadı: {e}"
        captured.setdefault("missing_codes", [])
        captured.setdefault("unrecognized_codes", [])

    # ── Map LLM-returned codes → full course dicts ────────────
    missing_courses: List[Dict] = _resolve_missing_courses(
        codes=captured.get("missing_codes", []),
        entry_year=entry_year,
    )
    unrecognized_courses: List[Dict] = _resolve_unrecognized_courses(
        codes=captured.get("unrecognized_codes", []),
        transcript_raw=transcript_raw,
    )

    n_miss = len(missing_courses)
    n_unrec = len(unrecognized_courses)
    print(f"[CURRICULUM-AGENT] missing={n_miss}, unrecognized={n_unrec}")

    return {
        "missing_courses": missing_courses,
        "unrecognized_courses": unrecognized_courses,
        "messages": [AIMessage(content=(
            f"[Müfredat Uzmanı] {n_miss} eksik zorunlu ders, "
            f"{n_unrec} tanınmayan ders bulundu. "
            f"{captured.get('summary', agent_summary)}"
        ))],
        "status_queue": [
            f"[🔍 Müfredat Uzmanı] {entry_year} müfredat anayasası ile öğrenci geçmişi eşleştiriliyor...",
            f"[🔍 Müfredat Uzmanı] {n_miss} eksik zorunlu ders, {n_unrec} tanınmayan ders tespit edildi.",
        ],
    }


def _resolve_missing_courses(codes: List[str], entry_year: str) -> List[Dict]:
    """Map code strings from LLM → full course dicts from curriculum JSON."""
    if not codes:
        return []
    try:
        curr = load_year_curriculum(entry_year)
        latest = load_latest_curriculum()
        all_courses = curr.get("dersler", []) + latest.get("dersler", [])
        by_code, by_base, _ = build_lookup(all_courses)
        seen: set = set()
        result: List[Dict] = []
        for code in codes:
            base = strip_suffix(code.strip().upper())
            if base in seen:
                continue
            seen.add(base)
            c = by_code.get(code.strip().upper()) or by_base.get(base)
            if c:
                result.append({
                    "kod": c["kod"],
                    "ad": c["ad"],
                    "akts": c.get("akts", 0),
                    "donem": c.get("donem", ""),
                    "tip": c.get("tip", "Zorunlu"),
                })
        return result
    except Exception as e:
        print(f"[CURRICULUM-AGENT] _resolve_missing_courses error: {e!r}")
        return []


def _resolve_unrecognized_courses(codes: List[str], transcript_raw: dict) -> List[Dict]:
    """Build unrecognized course objects from transcript using codes found by LLM."""
    if not codes:
        return []
    # Build quick lookup from transcript
    code_map: Dict[str, Dict] = {}
    for sem in transcript_raw.get("semesters", []):
        for c in sem.get("courses", []):
            if c.get("durum") == "Geçti":
                code_map[c["kod"].upper()] = c
                code_map[strip_suffix(c["kod"].upper())] = c

    seen: set = set()
    result: List[Dict] = []
    for code in codes:
        base = strip_suffix(code.strip().upper())
        if base in seen:
            continue
        seen.add(base)
        c = code_map.get(code.strip().upper()) or code_map.get(base)
        if c:
            result.append({
                "kod": c["kod"],
                "ad": c.get("ad", ""),
                "not": c.get("not", ""),
                "akts": c.get("akts", 0),
            })
    return result


# ─────────────────────────────────────────────────────────────
# NODE 2: INTIBAK AGENT
# ─────────────────────────────────────────────────────────────
async def intibak_agent_node(state: AdvisorState) -> dict:
    transcript_raw = state["transcript_raw"]
    missing_courses = state.get("missing_courses") or []
    unrecognized_courses = state.get("unrecognized_courses") or []
    entry_year = state.get("entry_year", "2025-2026")

    if not missing_courses:
        return {
            "intibak_cleared": [],
            "messages": [AIMessage(content="[İntibak Avukatı] Temizlenecek eksik ders bulunamadı.")],
            "status_queue": ["[⚖️ İntibak Avukatı] Eksik ders listesi boş, bu aşama atlanıyor."],
        }

    captured: Dict = {}

    # Inject current missing/unrecognized context into agent input message
    missing_list = "\n".join(f"- {c['kod']}: {c['ad']}" for c in missing_courses[:30])
    unrec_list = "\n".join(
        f"- {c['kod']}: {c.get('ad', '')}" for c in unrecognized_courses[:20]
    )

    tools = [
        make_transcript_tool(transcript_raw),
        get_intibak_rules,
        make_intibak_reporting_tool(captured),
    ]

    agent = create_react_agent(
        model=get_llm(temperature=0),
        tools=tools,
        prompt=INTIBAK_AGENT_PROMPT,
    )

    input_msg = (
        f"Öğrenci giriş yılı: {entry_year}\n\n"
        f"EKSİK ZORUNLU DERSLER:\n{missing_list}\n\n"
        f"TANINMAYAN geçilen dersler:\n{unrec_list if unrec_list else '(yok)'}\n\n"
        f"1. get_intibak_rules çağır → muafiyet kurallarını al\n"
        f"2. Hangi eksik dersler eşdeğerlik kuralıyla temizlenebilir? Cleared_codes listesini oluştur.\n"
        f"3. HEMEN report_intibak_findings çağır — başka metin yazma, sadece aracı çağır."
    )

    try:
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=input_msg)]},
            config={"recursion_limit": 20, "callbacks":[lambda event: print(f"[AUDIT-AGENT] ReAct event: {event}")]},
        )
        agent_summary = result["messages"][-1].content if result.get("messages") else ""
    except Exception as e:
        print(f"[INTIBAK-AGENT] ReAct error: {e!r}")
        agent_summary = f"İntibak analizi tamamlanamadı: {e}"
        captured.setdefault("cleared_codes", [])

    cleared = [c.strip().upper() for c in captured.get("cleared_codes", [])]
    # Remove cleared codes from missing_courses
    cleared_bases = {strip_suffix(c) for c in cleared}
    final_missing = [
        c for c in missing_courses
        if strip_suffix(c["kod"].upper()) not in cleared_bases
    ]

    n_cleared = len(cleared)
    print(f"[INTIBAK-AGENT] cleared={n_cleared}, remaining_missing={len(final_missing)}")

    return {
        "intibak_cleared": cleared,
        "missing_courses": final_missing,  # overwrite with cleaned list
        "messages": [AIMessage(content=(
            f"[İntibak Avukatı] {n_cleared} ders muafiyet kuralıyla temizlendi. "
            f"Kalan eksik: {len(final_missing)}. "
            f"{captured.get('summary', agent_summary)}"
        ))],
        "status_queue": [
            "[⚖️ İntibak Avukatı] Eksik dersler için bölüm muafiyet kuralları taranıyor...",
            f"[⚖️ İntibak Avukatı] {n_cleared} ders eşdeğerlik kuralıyla temizlendi, "
            f"{len(final_missing)} eksik kaldı.",
        ],
    }


# ─────────────────────────────────────────────────────────────
# NODE 3: AUDIT AGENT
# ─────────────────────────────────────────────────────────────
async def audit_agent_node(state: AdvisorState) -> dict:
    transcript_raw = state["transcript_raw"]
    missing_courses = state.get("missing_courses") or []
    entry_year = state.get("entry_year", "2025-2026")

    captured: Dict = {}

    # Compute total ECTS in Python (simple arithmetic, no LLM needed)
    total_ects = _compute_total_ects(transcript_raw, state.get("unrecognized_courses") or [])

    missing_list = "\n".join(f"- {c['kod']}: {c['ad']}" for c in missing_courses[:30])

    # Get ECTS target from curriculum
    try:
        curr = load_latest_curriculum()
        target_ects = curr.get("mezuniyet_sartlari", {}).get("toplam_akts_hedefi", 240)
    except Exception:
        target_ects = 240

    tools = [
        make_transcript_tool(transcript_raw),
        make_audit_reporting_tool(captured),
    ]

    agent = create_react_agent(
        model=get_llm(temperature=0),
        tools=tools,
        prompt=AUDIT_AGENT_PROMPT,
    )

    input_msg = (
        f"Tamamlanan AKTS: {round(total_ects, 1)} / {target_ects}\n"
        f"Kalan EKSİK ZORUNLU DERSLER ({len(missing_courses)} adet):\n"
        f"{missing_list if missing_list else '(Eksik zorunlu ders yok)'}\n\n"
        f"Karar ver:\n"
        f"- Eksik ders varsa → graduation_status = 'Mezun Olamaz (Eksik Zorunlu Dersler Mevcut)'\n"
        f"- Eksik ders yok ama AKTS < {target_ects} → graduation_status = 'Mezun Olamaz (Yetersiz AKTS: {round(total_ects,1)}/{target_ects})'\n"
        f"- Eksik yok, AKTS yeterli → graduation_status = 'Mezun Olabilir'\n\n"
        f"HEMEN report_audit_decision çağır — başka metin yazma, sadece aracı çağır."
    )

    try:
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=input_msg)]},
            config={"recursion_limit": 15},
        )
        agent_summary = result["messages"][-1].content if result.get("messages") else ""
    except Exception as e:
        print(f"[AUDIT-AGENT] ReAct error: {e!r}")
        agent_summary = ""
        captured.setdefault("graduation_status", "")

    # Fallback if agent didn't call reporting tool
    if not captured.get("graduation_status"):
        if missing_courses:
            captured["graduation_status"] = "Mezun Olamaz (Eksik Zorunlu Dersler Mevcut)"
        elif round(total_ects, 1) < target_ects:
            captured["graduation_status"] = (
                f"Mezun Olamaz (Yetersiz AKTS: {round(total_ects, 1)} / {target_ects})"
            )
        else:
            captured["graduation_status"] = "Mezun Olabilir"

    grad_status = captured["graduation_status"]
    assessment = captured.get("assessment", agent_summary)

    print(f"[AUDIT-AGENT] graduation_status={grad_status!r}, ects={round(total_ects, 1)}")

    return {
        "graduation_status": grad_status,
        "final_report": assessment,
        "messages": [AIMessage(content=f"[Mezuniyet Denetçisi] Karar: {grad_status}. {assessment}")],
        "status_queue": [
            "[🎓 Mezuniyet Denetçisi] Toplam AKTS hesaplanıyor ve nihai karar veriliyor...",
            f"[🎓 Mezuniyet Denetçisi] Karar: {grad_status}",
        ],
    }


def _compute_total_ects(transcript_raw: dict, unrecognized_courses: List[Dict]) -> float:
    """Sum ECTS for all passed courses in the transcript."""
    total = 0.0
    seen_codes: set = set()

    # Recognized passed courses (from semesters)
    for sem in transcript_raw.get("semesters", []):
        for c in sem.get("courses", []):
            if c.get("durum") == "Geçti":
                base = strip_suffix(c.get("kod", "").upper())
                if base not in seen_codes:
                    seen_codes.add(base)
                    total += float(c.get("akts", 0))

    # Add unrecognized electives (they count toward total)
    for c in unrecognized_courses:
        base = strip_suffix(c.get("kod", "").upper())
        if base not in seen_codes:
            seen_codes.add(base)
            total += float(c.get("akts", 0))

    return round(total, 1)


# ─────────────────────────────────────────────────────────────
# GRAPH ASSEMBLY
# ─────────────────────────────────────────────────────────────
def build_analysis_graph():
    """Compile and return the Supervisor MAS graph."""
    workflow = StateGraph(AdvisorState)

    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("curriculum_agent", curriculum_agent_node)
    workflow.add_node("intibak_agent", intibak_agent_node)
    workflow.add_node("audit_agent", audit_agent_node)

    workflow.set_entry_point("supervisor")

    workflow.add_conditional_edges(
        "supervisor",
        _route_by_supervisor,
        {
            "curriculum_agent": "curriculum_agent",
            "intibak_agent":    "intibak_agent",
            "audit_agent":      "audit_agent",
            "__end__":          END,
        },
    )
    # Each specialist returns to the supervisor after finishing
    workflow.add_edge("curriculum_agent", "supervisor")
    workflow.add_edge("intibak_agent",    "supervisor")
    workflow.add_edge("audit_agent",      "supervisor")

    return workflow.compile()


# Singleton — compiled once, reused for every upload request
_ANALYSIS_GRAPH = None


def get_analysis_graph():
    global _ANALYSIS_GRAPH
    if _ANALYSIS_GRAPH is None:
        _ANALYSIS_GRAPH = build_analysis_graph()
    return _ANALYSIS_GRAPH
