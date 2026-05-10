# Sanal Akademik Danışman (Virtual Academic Advisor) - Project Context

## 🎯 Project Overview
This project is an AI-powered Academic Advisor for university students. It takes a student's transcript (PDF), parses it, compares it against the university curriculum (JSON), calculates ECTS (AKTS) credits, and uses an LLM to provide advisory feedback (e.g., missing courses, graduation status).

## 🛠️ Tech Stack
- **Backend:** Python, FastAPI, pdfplumber (for PDF extraction), LangChain, LangGraph (Target), Ollama (Local LLMs)
- **Frontend:** React, Vite, TailwindCSS, Axios, Lucide React
- **LLM:** Llama 3.1 / Mistral / Gemma series via local Ollama.

## 📂 Project Structure
- `/backend/`: Contains FastAPI server (`main.py`), parsers, and local JSON databases.
  - `mufredat_tam_chunked.json`: The source of truth for the curriculum.
  - `son_ayiklanan_transcript.json`: Temporary state file for the current session's parsed transcript.
- `/frontend/`: React application. Main logic is in `src/App.jsx`.

## 🏗️ Target Architecture (Agentic System)
We are currently migrating from a simple prompt-injection flow to a **Tool-based Agentic Architecture**. 
- The Agent must be capable of deciding which tool to use (e.g., Check Curriculum, Check Passed Semesters).
- The system must stream its thought process (e.g., "Agent is thinking...", "Scanning semester 3...") to the UI via Server-Sent Events (SSE) or WebSockets.
- Transcript data must be grouped and saved strictly by **Semesters** (e.g., "2016-2017 Fall"), not as a flat list.

## ⚠️ Coding Rules & Guidelines
1. **Strict Matching:** Never use semantic/fuzzy matching for course approvals. A course is only passed if BOTH the Course Code and Course Name strongly match the curriculum.
2. **State Management:** Backend must use `session_id` strictly. No data bleed between different chat sessions.
3. **No Destructive Overwrites:** When refactoring, do not delete existing functional UI components unless replacing them with enhanced equivalents.
4. **Language:** The codebase logic (variables, functions) can be in English, but the LLM Prompts, UI Texts, and Agent responses MUST be in strictly professional, sincere **Turkish**.
5. **Robustness:** Always include `try-except` blocks for file reading, PDF parsing, and LLM API calls.