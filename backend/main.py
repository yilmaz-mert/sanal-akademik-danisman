import os
import json
import re
import uuid
import unicodedata
from typing import Optional, Dict, List

from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

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


# ─────────────────────────────────────────────────────────────
# STRING NORMALIZATION
# ─────────────────────────────────────────────────────────────
def normalize_str(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFC", s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s.translate(str.maketrans("ıİğĞüÜöÖşŞçÇ", "iigguuoosscc"))


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
# Semester header: "2021 - 2022 Güz Yarıyılı"
SEM_HEADER_RE = re.compile(
    r"(\d{4})\s*[-–—]\s*(\d{4})\s+(Güz|Bahar|Yaz|GÜZ|BAHAR|YAZ)\s+Yarıyılı?",
    re.IGNORECASE,
)

TERM_GPA_RE = re.compile(
    r"(?:ANO|Yar[iı]y[iı]l\s+Ortalamas[iı]|D[öo]nem\s+Ortalamas[iı]|D[öo]nem\s+GNO)"
    r"\s*[:\s]\s*([\d.,]+)",
    re.IGNORECASE,
)

CUMULATIVE_GPA_RE = re.compile(
    r"(?:Genel\s+GNO|CGPA|Genel\s+Ortalama|Kümülatif\s+Ortalama)"
    r"\s*[:\s]\s*([\d.,]+)",
    re.IGNORECASE,
)

COURSE_CODE_RE = re.compile(r"^([A-Z]{2,4}\d{3}(?:-[A-Z])?)\b")


def _parse_course_line(line: str) -> Optional[dict]:
    """
    Parse a single transcript line into a course dict.
    Handles variable column layouts: Code Name [Type] [T U L] Credit AKTS Grade [Status]
    """
    line = line.strip()
    # Strip leading row-number prefix (e.g. "1  INF101 ...")
    line = re.sub(r"^\d+[\s.]+", "", line).strip()

    m = COURSE_CODE_RE.match(line)
    if not m:
        return None

    code = m.group(1)
    rest = line[m.end():].strip()
    tokens = rest.split()

    if not tokens:
        return None

    # Scan from the right to find the grade token
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

    # Everything before the grade (excluding trailing status words like Geçti/Kaldı)
    before = tokens[:grade_idx]

    # Strip trailing Type token (Z or S) placed between name and numbers
    # and collect trailing number tokens for credit/AKTS
    numbers: List[float] = []
    name_end = len(before) - 1

    while name_end >= 0:
        t = before[name_end].replace(",", ".")
        try:
            numbers.insert(0, float(t))
            name_end -= 1
        except ValueError:
            break

    # Check if the token just before numbers is Z or S (type)
    tip = ""
    if name_end >= 0 and before[name_end].upper() in ("Z", "S"):
        tip = before[name_end].upper()
        name_end -= 1

    # Remaining tokens form the course name
    name = " ".join(before[: name_end + 1]).strip()
    if not name:
        return None

    # We need at least one number (AKTS); credit defaults to AKTS if only one number
    if len(numbers) == 0:
        return None

    akts = numbers[-1]
    credit = numbers[-2] if len(numbers) >= 2 else akts

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
    return None


def parse_transcript_txt(raw_bytes: bytes) -> dict:
    text = _decode_txt(raw_bytes)
    all_codes = list(set(re.findall(r"\b([A-Z]{2,4}\d{3}(?:-[A-Z])?)\b", text)))

    sem_matches = list(SEM_HEADER_RE.finditer(text))
    semesters: List[dict] = []

    if sem_matches:
        for i, sm in enumerate(sem_matches):
            year_s, year_e = sm.group(1), sm.group(2)
            term = sm.group(3).capitalize()
            label = f"{year_s} - {year_e} {term}"
            start = sm.end()
            end = sem_matches[i + 1].start() if i + 1 < len(sem_matches) else len(text)
            block = text[start:end]

            courses: List[dict] = []
            for raw_line in block.splitlines():
                # Skip obvious header/summary lines
                if re.search(
                    r"Kodu|Ders Ad|Notu|Kredi|AKTS|Ortalama|ANO|T\s+U\s+L",
                    raw_line,
                    re.IGNORECASE,
                ):
                    continue
                parsed = _parse_course_line(raw_line)
                if parsed:
                    courses.append(parsed)

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

    # Cumulative GPA
    cumulative_gpa: Optional[float] = None
    cm = CUMULATIVE_GPA_RE.search(text)
    if cm:
        try:
            cumulative_gpa = float(cm.group(1).replace(",", "."))
        except ValueError:
            pass

    # Estimate cumulative GPA from semester GPAs if not found
    if cumulative_gpa is None and semesters:
        gpas = [s["term_gpa"] for s in semesters if s["term_gpa"] is not None]
        if gpas:
            cumulative_gpa = round(sum(gpas) / len(gpas), 2)

    return {
        "semesters": semesters,
        "all_codes": all_codes,
        "cumulative_gpa": cumulative_gpa,
    }


# ─────────────────────────────────────────────────────────────
# STRICT CURRICULUM MATCHING
# Code must match exactly. Name is a secondary normalized check.
# Name-only (fuzzy) matching is intentionally absent.
# ─────────────────────────────────────────────────────────────
def match_with_curriculum(parsed: dict, curriculum: List[dict]) -> dict:
    by_code = {c["kod"]: c for c in curriculum}
    passed: set = set()
    failed_courses: List[dict] = []
    total_ects = 0
    matched_log: List[str] = []

    for sem in parsed.get("semesters", []):
        for c in sem.get("courses", []):
            code = c.get("kod", "")
            name = c.get("ad", "")
            grade = c.get("not", "")
            status = c.get("durum", "")

            if code not in by_code:
                continue

            curr = by_code[code]
            curr_n = normalize_str(curr["ad"])
            trans_n = normalize_str(name)

            name_ok = (
                not trans_n
                or trans_n == curr_n
                or curr_n in trans_n
                or trans_n in curr_n
            )

            if not name_ok:
                continue

            if status == "Geçti":
                passed.add(code)
                total_ects += curr.get("akts", 0)
                matched_log.append(f"{code} - {curr['ad']} [{grade}]")
            elif is_failed(grade):
                failed_courses.append(
                    {
                        "kod": code,
                        "ad": curr["ad"],
                        "not": grade,
                        "akts": curr.get("akts", 0),
                        "donem": curr.get("donem", ""),
                    }
                )

    # Fallback: if semester parser found no course rows (e.g. catalog TXT uploaded)
    if not matched_log and not failed_courses:
        for code in parsed.get("all_codes", []):
            if code in by_code and code not in passed:
                curr = by_code[code]
                passed.add(code)
                total_ects += curr.get("akts", 0)
                matched_log.append(f"{code} - {curr['ad']} (kod)")

    missing = [
        {
            "kod": c["kod"],
            "ad": c["ad"],
            "akts": c.get("akts", 0),
            "donem": c.get("donem", ""),
            "tip": c.get("tip", ""),
        }
        for c in curriculum
        if c["kod"] not in passed
        and c["kod"] not in {f["kod"] for f in failed_courses}
        and c.get("tip") == "Zorunlu"
    ]

    return {
        "passed_codes": list(passed),
        "matched_log": matched_log,
        "total_ects": total_ects,
        "missing_courses": missing,
        "failed_courses": failed_courses,
    }


# ─────────────────────────────────────────────────────────────
# AGENT TOOLS (closures per-request for session isolation)
# ─────────────────────────────────────────────────────────────
TOOL_STATUS = {
    "GetTranscriptData": "📋 Transkript verileri taranıyor...",
    "GetCurriculumData": "📚 Müfredat kuralları kontrol ediliyor...",
    "CalculateGraduationPath": "🎓 Mezuniyet yolu hesaplanıyor...",
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
        """Eksik ve başarısız dersler + mezuniyet hesabı."""
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
                    "kalan_akts": hedef - total,
                    "hedef_akts": hedef,
                    "minimum_gno": sartlar.get("minimum_gno", 2.0),
                    "eksik_zorunlu_sayisi": len(missing),
                    "basarisiz_ders_sayisi": len(failed),
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
                "Geçilen/başarısız dersler, dönem GPA'ları ve AKTS değerleri için kullan."
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
                "Eksik ve başarısız zorunlu dersleri hesaplar, gelecek dönem önerileri sunar. "
                "'Mezun olabilir miyim?', 'Kaç dersim eksik?', 'Gelecek dönem ne almalıyım?', "
                "'Başarısız derslerim var mı?' sorularında kullan."
            ),
        ),
    ]


# ─────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "Sen Galatasaray Üniversitesi Bilgisayar Mühendisliği bölümünün kıdemli, samimi ve "
    "%100 dürüst Akademik Danışmanısın. Öğrenciye yalnızca Türkçe yanıt ver. "
    "Kısa, net ve profesyonel ol. Öğrenciye doğrudan 'sen' diye hitap et. "
    "Sadece gerçek verilere dayan; bilmiyorsan açıkça belirt. "
    "Yanıtlarında gerektiğinde markdown tablo ve madde işareti kullan."
)


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

    try:
        curriculum_data = load_curriculum()
        curriculum = curriculum_data.get("dersler", [])
    except Exception as e:
        return {"error": f"Müfredat yüklenemedi: {e}", "session_id": session_id}

    parsed = parse_transcript_txt(raw_bytes)
    matched = match_with_curriculum(parsed, curriculum)

    transcript_data = {
        "session_id": session_id,
        "semesters": parsed.get("semesters", []),
        "cumulative_gpa": parsed.get("cumulative_gpa"),
        "passed_codes": matched["passed_codes"],
        "matched_log": matched["matched_log"],
        "total_ects": matched["total_ects"],
        "missing_courses": matched["missing_courses"],
        "failed_courses": matched["failed_courses"],
    }

    get_session(session_id)["transcript"] = transcript_data

    # Persist (overwrite son_ayiklanan_transcript.json as required)
    try:
        with open("son_ayiklanan_transcript.json", "w", encoding="utf-8") as f:
            json.dump(transcript_data, f, ensure_ascii=False, indent=4)
    except Exception:
        pass

    # Build rich frontend summary
    semester_summaries = [
        {
            "name": s["semester_name"],
            "gpa": s.get("term_gpa"),
            "ects": s.get("term_ects", 0),
            "course_count": len(s.get("courses", [])),
            "failed_count": sum(1 for c in s.get("courses", []) if c.get("durum") == "Kaldı"),
        }
        for s in parsed.get("semesters", [])
    ]

    target_ects = curriculum_data.get("mezuniyet_sartlari", {}).get("toplam_akts_hedefi", 240)

    summary = {
        "total_ects": matched["total_ects"],
        "target_ects": target_ects,
        "missing_count": len(matched["missing_courses"]),
        "failed_count": len(matched["failed_courses"]),
        "semester_count": len(parsed.get("semesters", [])),
        "cumulative_gpa": parsed.get("cumulative_gpa"),
        "semesters": semester_summaries,
    }

    # LLM initial summary
    missing_names = [f"{c['kod']} - {c['ad']}" for c in matched["missing_courses"]]
    failed_names = [
        f"{c['kod']} - {c['ad']} ({c.get('not','F')})"
        for c in matched["failed_courses"]
    ]

    prompt = (
        f"Öğrencinin transkripti analiz edildi:\n"
        f"- Tamamlanan AKTS: {matched['total_ects']} / {target_ects}\n"
        f"- Genel GNO: {parsed.get('cumulative_gpa') or 'Tespit edilemedi'}\n"
        f"- Dönem sayısı: {len(semester_summaries)}\n"
        f"- Eksik zorunlu ders: {len(matched['missing_courses'])}\n"
        f"- Başarısız (F) ders: {len(matched['failed_courses'])}\n"
        + (f"- Başarısız: {', '.join(failed_names)}\n" if failed_names else "")
        + (
            f"- Eksikler (ilk 5): {', '.join(missing_names[:5])}"
            + ("..." if len(missing_names) > 5 else "")
            + "\n"
            if missing_names
            else ""
        )
        + "\nBu sonuçları öğrenciye samimi ve profesyonel Türkçeyle 2-3 cümlede özetle. "
        "İlk cümlede genel durumu belirt, ikincide kritik eksikler/başarısızlıkları vurgula, "
        "üçüncüde yönlendirici tavsiye ver."
    )

    try:
        result = chat_model.invoke(
            [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
        )
        reply = result.content
    except Exception:
        reply = (
            f"Transkriptiniz analiz edildi. {matched['total_ects']} AKTS tamamlandı"
            + (
                f", GNO: {parsed.get('cumulative_gpa'):.2f}"
                if parsed.get("cumulative_gpa")
                else ""
            )
            + f". {len(matched['missing_courses'])} zorunlu ders eksik"
            + (f", {len(matched['failed_courses'])} ders başarısız" if matched["failed_courses"] else "")
            + ". Detaylar için soru sorabilirsiniz."
        )

    return {"session_id": session_id, "reply": reply, "summary": summary}


# ─────────────────────────────────────────────────────────────
# CHAT STREAM  (SSE via astream_events)
# ─────────────────────────────────────────────────────────────
@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    tools = make_tools(request.session_id)
    agent = create_agent(
        model=chat_model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
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
                    output = event.get("data", {}).get("output", {})
                    msgs = output.get("messages", [])
                    final = ""
                    if msgs:
                        last = msgs[-1]
                        final = last.content if hasattr(last, "content") else str(last)
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
