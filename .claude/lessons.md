# Lessons Learned

## Session 3 — Intibak PDF Extraction (2026-05-13)

**Problem:** `extract_curriculum_rules.py` produced 0 notes from all 5 PDFs.

**Root causes:**
1. `SEMESTER_RE = r"D[?o][?e]?nem\s+(\d+)"` used literal `?` instead of Unicode `ö` (U+00F6). PowerShell displays `ö` as `?` due to CP1254 codepage, but Python strings contain real Unicode — regex must use the actual character.
2. `NOTE_RE = r"Not\s+(\d+)\s*:?\s*\n"` required a newline after the header, failing for 2021-2022 inline format (`Not 2: content on same line`) and 2024-2025 no-colon format (`Not 2\ncontent`).
3. Old `OLD_COURSE_RE` used `[A-Z]` to start course name capture — failed against `«`/`»` French quote characters that precede course codes in the PDFs.

**Fix:** Rewrote extraction logic using a position-anchored approach: scan backward from each `dersinden kalan` occurrence for the last course code, scan forward for `yerine` → terminal (`derslerini/dersini al`). Also added bullet-char handling for 2025-2026 format.

**Result:** 35 deduplicated equivalency rules, 16 general rules across all 5 years.

---

## Session 4 — Streaming Fix + Intibak Reviewer Node (2026-05-13)

### Bug 1: Chat Streaming Leaked Raw Python Objects

**Symptom:** Frontend received `content='' additional_kwargs={} tool_calls=[...]` as chat text.

**Root cause:** The `on_chain_end` handler watched `ename in ("LangGraph", "agent")`. The "agent" event fires at every ReAct step (including mid-tool-call steps). When no AI text was found, `str(msgs[-1])` stringified a raw LangChain `AIMessage` or `ToolMessage` object and sent it as `done`.

**Fix:**
- Changed condition to `ename == "LangGraph"` only (outermost terminal event, fires once).
- Added `msg_type not in ("AIMessage", "AIMessageChunk")` guard to skip ToolMessages, HumanMessages.
- Removed `str(msgs[-1])` fallback entirely; replaced with plain Turkish error string.
- Only yield `done` when actual text is found; continue listening otherwise.

**Key rule:** Never call `str()` on a LangChain message object in a streaming SSE context.

### Bug 2: False-Positive Missing Courses (ING111, CNT250 etc.)

**Symptom:** Students who enrolled in 2021 were shown ING111 (Ekonominin Temelleri) as missing, even though it was added in 2025-2026 and intibak rules grant it automatically when a student passed ING104/ING105.

**Root cause:** The Judge compared the student's passed codes against the LATEST (2025-2026) curriculum. New courses that didn't exist when the student enrolled appeared as "missing". The intibak equivalency file had the rule but was never consulted.

**Fix:** Added `review_missing_courses_node` between `reviewer_ai` and `judge`.
- **Pass 1 (deterministic):** For every old code the student passed, look it up in `eski_yeni_eslestirme` from `intibak_kurallari.json`. Add all mapped new codes to `intibak_cleared_codes`.
- **Pass 2 (AI):** If there are unrecognized courses, ask the LLM whether any match missing courses by base-code similarity or entry-year exemption.
- Judge reads `intibak_cleared_codes` and skips those from the missing list.

**Pipeline order:** `historian_init → semester_agent(×N) → mapper → reviewer_ai → review_missing_courses → judge`

**Key rule:** `intibak_kurallari.json` MUST be loaded at startup and consulted before finalizing `missing_courses`.

---

## Session 5 — Async Refactor + Structured Output (2026-05-13)

### Bug: Backend Hangs After `[INTIBAK] Pass-1 cleared...`

**Symptom:** Server froze exactly after the deterministic pass logged its output. The AI pass never returned.

**Root cause (double):**
1. **Sync blocking:** Both `reviewer_ai_node` and `review_missing_courses_node` were `def` (synchronous). LangGraph runs async nodes natively; sync nodes are run via `asyncio.to_thread`. A slow local Ollama call on a thread-pool thread can still block the event loop if the pool is exhausted, and can hang indefinitely if the model rambles without producing valid JSON.
2. **Wrong context:** `review_missing_courses_node` passed only the `quick_lookup` (a compact `{old: [new]}` dict) to the LLM. This deprives it of the actual rule conditions (`kosul_yili_oncesi`, rule narratives), so the model gets confused and over-generates.
3. **Fragile output parsing:** Both nodes used `re.search(r'\{.*\}', result.content, re.DOTALL)` + `json.loads`. If the LLM prefixes its output with explanation text or generates trailing tokens, the regex either mismatches or `json.loads` raises, and the default fallback silently hides the issue.

**Fix:**
1. Changed both nodes to `async def`.
2. Replaced all `chat_model.invoke(...)` with `await chat_model.with_structured_output(PydanticModel).ainvoke(...)`. `with_structured_output` uses Ollama's native JSON mode — the model MUST terminate its output as valid JSON matching the schema.
3. Added four Pydantic schemas: `ReviewerNote`, `ReviewerAIResponse`, `IntibakClearedItem`, `IntibakAIResponse`.
4. Replaced `rules_sample` (quick-lookup dict) with `_format_intibak_rules(intibak_rules)` — the full textual rules like `ING104 -> ING106, INF115, INF116, ING111 [kosul: 2020-2021 ve oncesi]`.
5. Removed all `re.search` regex-based LLM output parsing.

**Architectural rule (permanent):** Any LangGraph node that calls an LLM MUST be `async def` and MUST use `with_structured_output(Pydantic).ainvoke()`. No exceptions.
