"""
Shared state TypedDict for the Supervisor Multi-Agent System.
"""
import operator
from typing import Annotated, Dict, List, Optional, TypedDict

from langgraph.graph.message import add_messages


class AdvisorState(TypedDict):
    # ── Input (set once at graph entry) ──────────────────────
    transcript_raw: Dict          # full parsed output from transcript_parser
    entry_year: str               # e.g. "2021-2022"
    session_id: str

    # ── Agent message bus ─────────────────────────────────────
    messages: Annotated[list, add_messages]

    # ── Structured findings (updated by specialist agents) ────
    missing_courses: Optional[List[Dict]]      # None = not yet computed
    unrecognized_courses: Optional[List[Dict]]
    intibak_cleared: Optional[List[str]]       # codes cleared by Intibak_Agent
    graduation_status: str                     # filled by Audit_Agent
    final_report: str                          # Turkish summary from Audit_Agent

    # ── Routing control ───────────────────────────────────────
    next_agent: str    # "curriculum_agent" | "intibak_agent" | "audit_agent" | "__end__"
    iteration: int     # safety counter

    # ── SSE progress queue (accumulates across all nodes) ─────
    status_queue: Annotated[List[str], operator.add]
