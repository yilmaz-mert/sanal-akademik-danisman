"""
Sanal Akademik Danışman — FastAPI entry point.

Architecture:
  /upload  → parse TXT → run Supervisor MAS (Groq/Llama agents) → SSE stream
  /chat/stream → Groq ReAct chat agent with transcript context
"""
import asyncio
import json
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")
from typing import Dict, List, Optional

from fastapi import FastAPI, File, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel

from agents.graph import get_analysis_graph
from agents.prompts import CHAT_SYSTEM_PROMPT
from agents.state import AdvisorState
from core.config import get_llm
from core.curriculum_loader import load_intibak_rules, load_latest_curriculum, load_year_curriculum
from core.transcript_parser import detect_entry_year, parse_transcript_txt, strip_suffix

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

# ─────────────────────────────────────────────────────────────
# SESSION STORE
# ─────────────────────────────────────────────────────────────
sessions: Dict[str, Dict] = {}
checkpointer = MemorySaver()


def get_session(sid: str) -> Dict:
    if sid not in sessions:
        sessions[sid] = {"transcript": None}
    return sessions[sid]


# ─────────────────────────────────────────────────────────────
# SSE HELPER
# ─────────────────────────────────────────────────────────────
def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


# ─────────────────────────────────────────────────────────────
# CHAT AGENT CONTEXT BUILDER
# Injects pre-computed transcript facts into the system prompt so the LLM
# never needs a tool call to answer "what grade did I get in X?"
# ─────────────────────────────────────────────────────────────
def _build_dynamic_context(transcript: dict) -> str:
    lines = [
        "\n\n=== ÖĞRENCİ AKADEMİK GEÇMİŞİ VE GÜNCEL DURUMU ===",
        "_(Kesin veriler — araç çağırmadan bu bölümü kullan.)_",
    ]

    gpa = transcript.get("cumulative_gpa")
    total = transcript.get("total_ects", 0)
    status = transcript.get("graduation_status", "")

    lines.append(f"- **Tamamlanan AKTS:** {total}")
    if gpa is not None:
        lines.append(f"- **Genel GNO:** {gpa:.2f}")
    if status:
        lines.append(f"- **Mezuniyet Durumu:** {status}")

    failed = transcript.get("failed_courses", [])
    if failed:
        failed_str = ", ".join(
            f"{c.get('kod','')} {c.get('ad','')} ({c.get('not','')})" for c in failed
        )
        lines.append(f"- **Başarısız Dersler ({len(failed)} adet):** {failed_str}")
    else:
        lines.append("- **Başarısız Dersler:** Yok")

    missing = transcript.get("missing_courses", [])
    if missing:
        miss_str = ", ".join(f"{c.get('kod','')} {c.get('ad','')}" for c in missing[:20])
        if len(missing) > 20:
            miss_str += f" (+{len(missing) - 20} daha)"
        lines.append(f"- **Eksik Zorunlu Dersler ({len(missing)} adet):** {miss_str}")
    else:
        lines.append("- **Eksik Zorunlu Dersler:** Yok")

    unrecognized = transcript.get("unrecognized_electives", [])
    if unrecognized:
        unrec_ects = round(sum(e.get("akts", 0) for e in unrecognized), 1)
        unrec_str = ", ".join(f"{e.get('kod','')}" for e in unrecognized)
        lines.append(
            f"- **Tanınmayan Seçmeliler ({len(unrecognized)} adet, "
            f"{unrec_ects} AKTS — mezuniyet toplamına sayıldı):** {unrec_str}"
        )

    semesters = transcript.get("semesters", [])
    if semesters:
        lines.append("\n**Dönem Dönem Not Geçmişi:**")
        for sem in semesters:
            sem_name = sem.get("semester_name", "")
            term_gpa = sem.get("term_gpa")
            gpa_str = f" (GNO: {term_gpa})" if term_gpa is not None else ""
            courses = sem.get("courses", [])
            if courses:
                course_str = ", ".join(
                    f"{c.get('kod','')} ({c.get('not','')})" for c in courses
                )
                lines.append(f"- **{sem_name}{gpa_str}:** {course_str}")
            else:
                lines.append(f"- **{sem_name}:** kayıt yok")

    lines.append(
        "\n**NOT: Bu veriler kesindir. 'Başarısız' listesi doluysa ASLA 'başarısız ders yok' deme.**"
    )
    lines.append("=== GEÇMİŞ BİTİŞİ ===")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# CHAT AGENT TOOLS  (per-request closures)
# ─────────────────────────────────────────────────────────────
TOOL_STATUS = {
    "GetCurriculumData": "Müfredat verileri yükleniyor...",
    "GetIntibakRules":   "İntibak ve eşdeğerlik kuralları yükleniyor...",
}


def make_chat_tools(session_id: str) -> list:
    def get_curriculum_data(year_key: str = "") -> str:
        """Bölümün GENEL MÜFREDATINı getirir. Yıla özgü için year_key geç ('2023-2024' gibi).
        SADECE müfredat genel bilgisi sorulduğunda kullan — not sorgularında kullanma."""
        try:
            data = load_year_curriculum(year_key.strip()) if year_key.strip() else load_latest_curriculum()
            dersler = data.get("dersler", [])
            sartlar = data.get("mezuniyet_sartlari", {})
            lines = [
                f"## Müfredat: {data.get('yil', year_key or 'güncel')}",
                f"- AKTS Hedefi: {sartlar.get('toplam_akts_hedefi', 240)}",
                f"- Min GNO: {sartlar.get('minimum_gno', 2.0)}",
            ]
            zorunlu = [c for c in dersler if c.get("tip") == "Zorunlu"]
            by_donem: Dict[int, List] = {}
            for c in zorunlu:
                by_donem.setdefault(int(c.get("donem", 0)), []).append(c)
            if by_donem:
                lines.append("\n### Zorunlu Dersler")
                for donem in sorted(by_donem):
                    parts = ", ".join(
                        f"{c['kod']} {c['ad']} ({c.get('akts', 0)} AKTS)"
                        for c in by_donem[donem]
                    )
                    lines.append(f"- Dönem {donem}: {parts}")
            return "\n".join(lines)
        except Exception as e:
            return f"Müfredat yüklenemedi: {e}"

    def get_intibak_rules_chat(_: str = "") -> str:
        """Eski-yeni ders eşleşmelerini ve intibak kurallarını getirir.
        Yalnızca muafiyet, intibak, eşdeğerlik soruları için kullan."""
        try:
            data = load_intibak_rules()
            quick = data.get("eski_yeni_eslestirme", {})
            rules = data.get("intibak_kurallari", [])
            lines = ["## Eski → Yeni Ders Eşleşmeleri"]
            for old, new_list in quick.items():
                lines.append(f"- {old} → {', '.join(new_list)}")
            if rules:
                lines.append("\n## Detaylı İntibak Kuralları")
                for i, r in enumerate(rules[:25], 1):
                    eski = r.get("eski_ders", {}).get("kod", "")
                    yeni = ", ".join(y.get("kod", "") for y in r.get("yeni_dersler", []))
                    kosul = r.get("kosul_yili_oncesi", "")
                    acik = r.get("aciklama", "")
                    kosul_str = f" [{kosul} ve öncesi]" if kosul else ""
                    lines.append(f"{i}. {eski} → {yeni}{kosul_str}: {acik}")
            return "\n".join(lines)
        except Exception as e:
            return f"İntibak kuralları yüklenemedi: {e}"

    return [
        StructuredTool.from_function(
            func=get_curriculum_data,
            name="GetCurriculumData",
            description=(
                "Bölümün genel müfredatını (ders listesi, AKTS, mezuniyet koşulları) getirir. "
                "Belirli bir yıl için year_key parametresini geç. "
                "KESİNLİKLE öğrencinin kendi notlarını sorgulamak için kullanma."
            ),
        ),
        StructuredTool.from_function(
            func=get_intibak_rules_chat,
            name="GetIntibakRules",
            description=(
                "Eski-yeni ders eşleşmelerini ve intibak kurallarını getirir. "
                "Yalnızca muafiyet/intibak/eşdeğerlik soruları için kullan."
            ),
        ),
    ]


# ─────────────────────────────────────────────────────────────
# REQUEST MODEL
# ─────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    session_id: str


# ─────────────────────────────────────────────────────────────
# UPLOAD ENDPOINT  (SSE streaming)
# ─────────────────────────────────────────────────────────────
_NODE_STATUS = {
    "supervisor":        "Süpervizör yönlendirme kararı veriyor...",
    "curriculum_agent":  "Müfredat Uzmanı transkripti analiz ediyor...",
    "intibak_agent":     "İntibak Avukatı muafiyet kurallarını araştırıyor...",
    "audit_agent":       "Mezuniyet Denetçisi nihai kararı veriyor...",
}


@app.post("/upload")
async def upload_transcript(
    file: UploadFile = File(...),
    session_id: str = Query(default=None),
):
    if not session_id:
        session_id = str(uuid.uuid4())

    raw_bytes = await file.read()

    async def generate():
        yield _sse({"type": "status", "data": "Transkript dosyası okunuyor ve ayrıştırılıyor..."})

        try:
            parsed = await asyncio.to_thread(parse_transcript_txt, raw_bytes)
        except Exception as e:
            yield _sse({"type": "error", "data": f"Transkript ayrıştırılamadı: {e}"})
            return

        entry_year = detect_entry_year(parsed.get("semesters", []))
        yield _sse({"type": "status", "data": f"Giriş yılı tespit edildi: {entry_year}"})

        # Build initial graph state
        initial_state: AdvisorState = {
            "transcript_raw":      parsed,
            "entry_year":          entry_year,
            "session_id":          session_id,
            "messages":            [],
            "missing_courses":     None,
            "unrecognized_courses": None,
            "intibak_cleared":     None,
            "graduation_status":   "",
            "final_report":        "",
            "next_agent":          "",
            "iteration":           0,
            "status_queue":        [],
        }

        yield _sse({"type": "status", "data": "Çok-Ajanlı Danışman Sistemi başlatılıyor..."})

        # ── Stream the MAS graph execution via astream_events ────
        graph = get_analysis_graph()
        final_state = initial_state.copy()

        try:
            async for event in graph.astream_events(
                initial_state,
                version="v2",
                config={"recursion_limit": 30},
            ):
                etype = event["event"]
                ename = event.get("name", "")

                if etype == "on_chain_start" and ename in _NODE_STATUS:
                    yield _sse({"type": "status", "data": _NODE_STATUS[ename]})

                elif etype == "on_chain_end" and ename in _NODE_STATUS:
                    output = event.get("data", {}).get("output", {})
                    for msg in output.get("status_queue", []):
                        yield _sse({"type": "status", "data": msg})

                elif etype == "on_chain_end" and ename == "LangGraph":
                    final_output = event.get("data", {}).get("output", {})
                    if isinstance(final_output, dict):
                        final_state = final_output

        except Exception as e:
            print(f"[GRAPH] Streaming error: {e!r}")
            yield _sse({"type": "status", "data": "Analiz tamamlanıyor (fallback)..."})
            try:
                result = await graph.ainvoke(
                    initial_state,
                    config={"recursion_limit": 30},
                )
                final_state = result
            except Exception as e2:
                yield _sse({"type": "error", "data": f"Analiz hatası: {e2}"})
                return

        # ── Compute failed_courses: only courses never subsequently passed ──
        # Build the set of base codes the student ultimately passed anywhere in
        # the transcript — these are NOT shown in the active-failures red box.
        _passed_bases: set = {
            strip_suffix(c.get("kod", "").upper())
            for sem in parsed.get("semesters", [])
            for c in sem.get("courses", [])
            if c.get("durum") == "Geçti"
        }
        failed_courses: List[Dict] = []
        seen_failed: set = set()
        for sem in parsed.get("semesters", []):
            for c in sem.get("courses", []):
                if c.get("durum") == "Kaldı":
                    base = strip_suffix(c.get("kod", "").upper())
                    # Skip if student later passed the same base code
                    if base in _passed_bases:
                        continue
                    if base not in seen_failed:
                        seen_failed.add(base)
                        failed_courses.append({
                            "kod":   c["kod"],
                            "ad":    c.get("ad", ""),
                            "not":   c.get("not", ""),
                            "akts":  c.get("akts", 0),
                            "donem": sem.get("semester_name", ""),
                        })

        # ── Build transcript_data for session store ────────────
        missing_courses = final_state.get("missing_courses") or []
        unrecognized = final_state.get("unrecognized_courses") or []
        graduation_status = final_state.get("graduation_status", "")

        # ECTS from graph (computed by audit node) — fall back to Python calc
        total_ects = _sum_ects_from_transcript(parsed, unrecognized)

        transcript_data = {
            "session_id":             session_id,
            "semesters":              parsed.get("semesters", []),
            "cumulative_gpa":         parsed.get("cumulative_gpa"),
            "total_ects":             total_ects,
            "missing_courses":        missing_courses,
            "failed_courses":         failed_courses,
            "unrecognized_electives": unrecognized,
            "graduation_status":      graduation_status,
        }
        get_session(session_id)["transcript"] = transcript_data

        try:
            with open("son_ayiklanan_transcript.json", "w", encoding="utf-8") as f:
                json.dump(transcript_data, f, ensure_ascii=False, indent=4)
        except Exception:
            pass

        # ── Build summary for frontend ──────────────────────────
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
            target_ects = (
                load_latest_curriculum()
                .get("mezuniyet_sartlari", {})
                .get("toplam_akts_hedefi", 240)
            )
        except Exception:
            target_ects = 240

        summary = {
            "total_ects":             total_ects,
            "target_ects":            target_ects,
            "missing_count":          len(missing_courses),
            "failed_count":           len(failed_courses),
            "unrecognized_count":     len(unrecognized),
            "unrecognized_ects":      round(sum(e.get("akts", 0) for e in unrecognized), 1),
            "semester_count":         len(semester_summaries),
            "cumulative_gpa":         parsed.get("cumulative_gpa"),
            "graduation_status":      graduation_status,
            "semesters":              semester_summaries,
            "missing_courses":        missing_courses,
            "failed_courses":         failed_courses,
            "unrecognized_electives": unrecognized,
        }

        yield _sse({"type": "summary", "data": {"session_id": session_id, "summary": summary}})

        # ── Initial LLM assessment ─────────────────────────────
        yield _sse({"type": "status", "data": "Akademik danışman ilk değerlendirmeyi hazırlıyor..."})

        missing_names = [f"{c['kod']} - {c['ad']}" for c in missing_courses]
        failed_names  = [f"{c['kod']} ({c.get('not','F')})" for c in failed_courses]
        unrec_ects    = round(sum(e.get("akts", 0) for e in unrecognized), 1)

        prompt = (
            f"Öğrencinin transkripti çok-ajanlı sistem tarafından analiz edildi:\n"
            f"- Tamamlanan AKTS: {total_ects} / {target_ects}\n"
            f"- Genel GNO: {parsed.get('cumulative_gpa') or 'Tespit edilemedi'}\n"
            f"- Dönem sayısı: {len(semester_summaries)}\n"
            f"- Eksik zorunlu ders: {len(missing_courses)}\n"
            f"- Başarısız ders: {len(failed_courses)}"
            + (f" ({', '.join(failed_names)})" if failed_names else "")
            + "\n"
            + (f"- Tanınmayan seçmeli: {len(unrecognized)} ders ({unrec_ects} AKTS — dahil edildi)\n"
               if unrecognized else "")
            + (f"- Eksikler (ilk 5): {', '.join(missing_names[:5])}"
               + ("..." if len(missing_names) > 5 else "")
               + "\n"
               if missing_names else "")
            + "\nBu sonuçları öğrenciye samimi ve profesyonel Türkçeyle 2-3 cümlede özetle."
        )

        try:
            model = get_llm(temperature=0.2)
            llm_result = await model.ainvoke(
                [SystemMessage(content=CHAT_SYSTEM_PROMPT), HumanMessage(content=prompt)]
            )
            reply = llm_result.content
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


def _sum_ects_from_transcript(parsed: dict, unrecognized: List[Dict]) -> float:
    total = 0.0
    seen: set = set()
    for sem in parsed.get("semesters", []):
        for c in sem.get("courses", []):
            if c.get("durum") == "Geçti":
                base = strip_suffix(c.get("kod", "").upper())
                if base not in seen:
                    seen.add(base)
                    total += float(c.get("akts", 0))
    for c in unrecognized:
        base = strip_suffix(c.get("kod", "").upper())
        if base not in seen:
            seen.add(base)
            total += float(c.get("akts", 0))
    return round(total, 1)


# ─────────────────────────────────────────────────────────────
# CHAT STREAM  (SSE via astream_events)
# ─────────────────────────────────────────────────────────────
@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    transcript = get_session(request.session_id).get("transcript")
    active_prompt = (
        CHAT_SYSTEM_PROMPT + _build_dynamic_context(transcript)
        if transcript
        else CHAT_SYSTEM_PROMPT
    )

    tools = make_chat_tools(request.session_id)
    agent = create_react_agent(
        model=get_llm(temperature=0.2),
        tools=tools,
        prompt=active_prompt,
        checkpointer=checkpointer,
    )
    config = {"configurable": {"thread_id": request.session_id}}

    async def generate():
        yield _sse({"type": "status", "data": "Akademik danışman devreye giriyor..."})

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
                    yield _sse({
                        "type": "status",
                        "data": TOOL_STATUS.get(ename, f"{ename} çalıştırılıyor..."),
                    })
                elif etype == "on_tool_end":
                    yield _sse({"type": "status", "data": "Veriler analiz ediliyor..."})

                elif etype == "on_chain_end" and ename == "LangGraph":
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
            print(f"[CHAT] Agent error: {e!r}")
            if not got_done:
                yield _sse({
                    "type": "error",
                    "data": "Danışman şu an yanıt veremiyor. Lütfen tekrar deneyin.",
                })
                return

        if not got_done:
            yield _sse({"type": "error", "data": "Yanıt alınamadı. Lütfen tekrar deneyin."})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection":    "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


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
    api_key_set = bool(os.environ.get("GROQ_API_KEY"))
    return {"status": "ok", "model": "llama-3.3-70b-versatile", "api_key_set": api_key_set}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
