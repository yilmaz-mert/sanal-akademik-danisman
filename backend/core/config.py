import itertools
import os

from langchain_groq import ChatGroq

MODEL = "llama-3.3-70b-versatile"

_model_cache: dict = {}
_key_cycle = None


def _init_key_cycle() -> None:
    global _key_cycle
    if _key_cycle is not None:
        return
    keys: list = []
    # Prefer numbered keys GROQ_API_KEY_1 … _N (stop at first gap)
    for i in range(1, 10):
        k = os.environ.get(f"GROQ_API_KEY_{i}", "").strip()
        if k:
            keys.append(k)
        else:
            break
    # Fall back to bare GROQ_API_KEY
    if not keys:
        k = os.environ.get("GROQ_API_KEY", "").strip()
        if k:
            keys.append(k)
    if not keys:
        raise ValueError(
            "No Groq API key found. Set GROQ_API_KEY (or GROQ_API_KEY_1 … _3) in backend/.env"
        )
    _key_cycle = itertools.cycle(keys)


def get_llm(temperature: float = 0.0) -> ChatGroq:
    _init_key_cycle()
    api_key = next(_key_cycle)  # round-robin across available keys
    cache_key = (api_key, round(temperature, 3))
    if cache_key not in _model_cache:
        _model_cache[cache_key] = ChatGroq(
            model=MODEL,
            groq_api_key=api_key,
            temperature=temperature,
        )
    return _model_cache[cache_key]
