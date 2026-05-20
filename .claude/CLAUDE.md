# Sanal Akademik Danışman - Project Context

## 🎯 Project Overview
AI-powered Academic Advisor using a **LangGraph Supervisor Multi-Agent System (MAS)** with Google Gemini 1.5 Flash. It parses transcripts (TXT with Python), then routes through a 4-agent Gemini swarm (Supervisor → Curriculum_Agent → Intibak_Agent → Audit_Agent) that reasons about graduation eligibility. Python only fetches raw data; all logical matching and equivalency checks are done by the LLMs.

## 📁 Module Layout
```
backend/core/        ← config.py, curriculum_loader.py, transcript_parser.py
backend/tools/       ← academic_tools.py (dumb @tools + reporting tool factories)
backend/agents/      ← state.py, prompts.py, graph.py (the LangGraph MAS)
backend/main.py      ← FastAPI app (/upload, /chat/stream, /session, /health)
```

## ⚠️ Coding Rules & Guidelines (CRITICAL)
1. **PYTHON = DUMB DATA ONLY:** Python tools ONLY fetch raw JSON data (`get_curriculum_by_year`, `get_intibak_rules`, `get_transcript_data`). ALL logical reasoning — course matching, equivalency checks, graduation decisions — is done by Gemini LLM agents.
2. **GPA EXTRACTION:** Always use `re.findall` and take the `[-1]` element to find the final Cumulative GPA. Semester GPAs extracted using `Yarıyıl\s+([\d.,]+)`.
3. **TOOL CALLING ENFORCEMENT:** Chat Agent System Prompt must explicitly FORBID outputting raw JSON tool calls. Use native tool invocation only.
4. **UI INTERACTIVITY:** Summary cards on the right panel MUST be clickable, opening a Tailwind Modal that displays detailed tables (Course Code, Name, ECTS, Grades).
5. **CHAT STREAMING — NO RAW OBJECT LEAKS:** `/chat/stream` `on_chain_end` handler MUST only react to `ename == "LangGraph"`. Filter messages to `AIMessage`/`AIMessageChunk` ONLY. Never call `str(msg)` on a LangChain message object.
6. **INTIBAK VIA AGENT:** Intibak equivalency checking is now done by `Intibak_Agent` (Gemini ReAct). It calls `get_intibak_rules()` to get the full JSON (35 rules, 30 quick-lookup entries) and reasons about equivalencies autonomously. After the ReAct loop it calls `report_intibak_findings(cleared_codes, summary)`.
7. **ALL LLM NODES MUST BE ASYNC:** Every LangGraph node that calls the LLM MUST be `async def`. Use `await model.ainvoke(...)`. NEVER use synchronous `invoke()` inside a LangGraph node.
8. **REPORTING TOOL PATTERN:** Each specialist agent (Curriculum, Intibak, Audit) has a `report_*` StructuredTool factory (`make_*_reporting_tool(captured: dict)`). The tool writes structured findings into `captured`. After the ReAct loop the node reads from `captured` and maps codes to full objects with Python lookups.
9. **SUPERVISOR ROUTING SENTINELS:** `missing_courses is None` = curriculum not done; `intibak_cleared is None` = intibak not done; `graduation_status == ""` = audit not done. Supervisor LLM uses `RoutingDecision(next, mesaj)` structured output with deterministic Python fallback.