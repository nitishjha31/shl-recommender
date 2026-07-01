"""
Thin, optional LLM wrapper.

The agent's *decisions* (ask vs retrieve vs refuse vs compare) are all made by
deterministic code in agent.py -- never by asking an LLM "what should I do
next", which is exactly the kind of thing that makes a graded, 8-turn-capped
conversation flaky. The LLM, when configured, is only used to phrase the final
text of a reply and to write a comparison grounded in catalog fields we
already retrieved. If no API key is present, every call falls back to a
deterministic template so the service is fully functional with zero external
dependencies.
"""
import os
from typing import Optional

_client = None
_provider = None

ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")

if ANTHROPIC_KEY:
    try:
        import anthropic

        _client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
        _provider = "anthropic"
    except Exception:
        _client = None
elif OPENAI_KEY:
    try:
        import openai

        _client = openai.OpenAI(api_key=OPENAI_KEY)
        _provider = "openai"
    except Exception:
        _client = None


def llm_available() -> bool:
    return _client is not None


def generate(system: str, user: str, max_tokens: int = 300) -> Optional[str]:
    """Returns None on any failure/absence so callers can fall back to templates."""
    if not _client:
        return None
    try:
        if _provider == "anthropic":
            resp = _client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                timeout=8.0,
            )
            return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
        elif _provider == "openai":
            resp = _client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=max_tokens,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                timeout=8.0,
            )
            return resp.choices[0].message.content.strip()
    except Exception:
        return None
    return None
