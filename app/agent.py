"""
Deterministic conversation state machine.

Because the endpoint is stateless (full history resent every call) and capped
at 8 turns, the agent re-derives "what do we know so far" from the whole
transcript on every request rather than trying to persist state anywhere.
That single choice is what makes refinement ("actually, add personality
tests") and correction ("actually make that senior, not mid") work for free:
there's no stored state to get out of sync with what the user actually said.

Flow per request:
  1. Guardrail check on the latest user turn -> refuse if out of scope.
  2. Comparison check -> if the user is asking "X vs Y", answer from catalog
     data directly, regardless of what stage the conversation is otherwise in.
  3. Otherwise extract slots (role/skill keywords, seniority, requested test
     types, explicit "give me recommendations now") from the ENTIRE user-side
     transcript.
  4. Decide: ask a clarifying question, or retrieve + return a shortlist.
"""
import re
from typing import Optional

from app.retrieval import get_index
from app.guardrails import check_scope, REFUSAL_MESSAGES
from app.llm import generate, llm_available

SENIORITY_MAP = {
    "entry": "Entry-Level",
    "entry-level": "Entry-Level",
    "junior": "Entry-Level",
    "grad": "Graduate",
    "graduate": "Graduate",
    "new grad": "Graduate",
    "mid": "Mid-Professional",
    "mid-level": "Mid-Professional",
    "intermediate": "Mid-Professional",
    "senior": "Professional Individual Contributor",
    "sr": "Professional Individual Contributor",
    "manager": "Manager",
    "managerial": "Manager",
    "supervisor": "Supervisor",
    "lead": "Manager",
    "director": "Executive",
    "executive": "Executive",
    "vp": "Executive",
    "c-level": "Executive",
}

TEST_TYPE_PHRASES = [
    (r"\bpersonalit(y|ies)\b", "P"),
    (r"\bbehav(ior|ioural|ioral)\b", "P"),
    (r"\bcognitiv(e|e ability)\b", "A"),
    (r"\breasoning\b", "A"),
    (r"\baptitude\b", "A"),
    (r"\b(coding|programming|technical|knowledge)\s*(test|assessment)?\b", "K"),
    (r"\bskills?\s*test\b", "K"),
    (r"\bsimulation\b", "S"),
    (r"\bsituational judg(e)?ment\b", "S"),
    (r"\bsjt\b", "S"),
]

EXPLICIT_RECOMMEND_PHRASES = [
    r"\bjust (give|show|recommend)\b",
    r"\bgo ahead\b",
    r"\bthat'?s (all|enough|it)\b",
    r"\bno more (info|information|details)\b",
    r"\bwhatever you (think|recommend)\b",
    r"\brecommend (something|anything)\b",
]

CLOSURE_PHRASES = [
    r"^\s*thanks?\s*!?\.?\s*$",
    r"\bthank you\b",
    r"\bthat'?s all\b.*\bneed",
    r"\bsounds good\b",
    r"\bperfect\b",
    r"\bgreat,?\s*that'?s all\b",
    r"\bno further questions\b",
    r"\bthat works\b",
]

COMPARE_PATTERNS = [
    r"difference between\s+(.+?)\s+and\s+(.+?)(?:\?|$)",
    r"compare\s+(.+?)\s+(?:and|with|to|vs\.?|versus)\s+(.+?)(?:\?|$)",
    r"(.+?)\s+vs\.?\s+(.+?)(?:\?|$)",
    r"(.+?)\s+versus\s+(.+?)(?:\?|$)",
]


def _user_texts(messages: list[dict]) -> list[str]:
    return [m["content"] for m in messages if m["role"] == "user"]


def _find_seniority(all_text: str) -> Optional[str]:
    for phrase, level in SENIORITY_MAP.items():
        if re.search(rf"\b{re.escape(phrase)}\b", all_text):
            return level
    return None


def _find_requested_test_types(all_text: str) -> list[str]:
    found = []
    for pattern, letter in TEST_TYPE_PHRASES:
        if re.search(pattern, all_text) and letter not in found:
            found.append(letter)
    return found


def _has_role_or_skill_signal(all_text: str, index) -> bool:
    vocab = set()
    for item in index.items:
        vocab.update(k.lower() for k in item.get("keywords", []))
    tokens = re.findall(r"[a-zA-Z+.#]+", all_text.lower())
    return any(tok in vocab for tok in tokens) or any(
        v in all_text for v in vocab if " " in v
    )


def _wants_recommendation_now(latest_user_text: str) -> bool:
    t = latest_user_text.lower()
    return any(re.search(p, t) for p in EXPLICIT_RECOMMEND_PHRASES)


def _is_closure(latest_user_text: str) -> bool:
    t = latest_user_text.lower().strip()
    return any(re.search(p, t) for p in CLOSURE_PHRASES)


def _try_comparison(latest_user_text: str, index) -> Optional[tuple[dict, dict]]:
    t = latest_user_text.lower()
    if not any(w in t for w in ["difference", "compare", " vs", "versus", "vs."]):
        return None
    for pattern in COMPARE_PATTERNS:
        m = re.search(pattern, t)
        if m:
            a_raw, b_raw = m.group(1).strip(" ?."), m.group(2).strip(" ?.")
            a_item = index.fuzzy_get(a_raw)
            b_item = index.fuzzy_get(b_raw)
            if a_item and b_item and a_item["name"] != b_item["name"]:
                return a_item, b_item
    return None


def _format_comparison(a: dict, b: dict) -> str:
    template = (
        f"**{a['name']}** ({a['test_type']}) vs **{b['name']}** ({b['test_type']})\n\n"
        f"- {a['name']}: {a['description']} Typical duration: {a.get('duration_minutes', 'n/a')} min. "
        f"Levels: {', '.join(a.get('job_levels', [])) or 'n/a'}.\n"
        f"- {b['name']}: {b['description']} Typical duration: {b.get('duration_minutes', 'n/a')} min. "
        f"Levels: {', '.join(b.get('job_levels', [])) or 'n/a'}.\n\n"
        f"In short: {a['name']} is a {a['test_type']}-type assessment while {b['name']} is a "
        f"{b['test_type']}-type assessment -- pick based on whether you're screening for "
        f"{'workplace behavior/traits' if a['test_type']=='P' else 'demonstrable knowledge or ability'} "
        f"versus {'workplace behavior/traits' if b['test_type']=='P' else 'demonstrable knowledge or ability'}."
    )
    if llm_available():
        polished = generate(
            system=(
                "You explain the difference between two SHL assessments to a recruiter. "
                "Use ONLY the facts given in the user message -- do not add any detail, name, "
                "number, or claim that isn't explicitly stated there. 3-5 sentences, plain text."
            ),
            user=(
                f"Assessment A: {a['name']}, type {a['test_type']}, description: {a['description']} "
                f"duration {a.get('duration_minutes')} min, levels {a.get('job_levels')}.\n"
                f"Assessment B: {b['name']}, type {b['test_type']}, description: {b['description']} "
                f"duration {b.get('duration_minutes')} min, levels {b.get('job_levels')}.\n"
                "Explain the key difference between them for someone choosing which to use."
            ),
        )
        if polished:
            return polished
    return template


def _clarifying_question(has_skill: bool, has_seniority: bool) -> str:
    if not has_skill:
        return (
            "Happy to help! What role or skills are you hiring for "
            "(e.g. a specific programming language, customer service, a management role)?"
        )
    if not has_seniority:
        return "Got it. What seniority level is this for (entry-level, mid, senior, manager)?"
    return "Any specific type of assessment you want included -- e.g. personality, cognitive, or a coding test?"


def handle_chat(messages: list[dict]) -> dict:
    index = get_index()
    user_msgs = _user_texts(messages)
    if not user_msgs:
        return {
            "reply": "Hi! I can help you find SHL assessments. What role or skills are you hiring for?",
            "recommendations": [],
            "end_of_conversation": False,
        }

    latest = user_msgs[-1]
    all_text = " ".join(user_msgs).lower()
    prior_assistant_turns = sum(1 for m in messages if m["role"] == "assistant")

    # 1. Guardrails on the latest turn only.
    scope_issue = check_scope(latest)
    if scope_issue:
        return {
            "reply": REFUSAL_MESSAGES[scope_issue],
            "recommendations": [],
            "end_of_conversation": False,
        }

    # 2. Closure detection (after we've already been through at least one turn).
    if prior_assistant_turns > 0 and _is_closure(latest) and len(latest.split()) <= 6:
        return {
            "reply": "Glad that helps! Come back anytime you need more SHL assessment recommendations.",
            "recommendations": [],
            "end_of_conversation": True,
        }

    # 3. Comparison intent, checked before the recommend/clarify flow.
    comparison = _try_comparison(latest, index)
    if comparison:
        a, b = comparison
        return {
            "reply": _format_comparison(a, b),
            "recommendations": [],
            "end_of_conversation": False,
        }

    # 4. Slot extraction across the whole conversation.
    has_skill = _has_role_or_skill_signal(all_text, index)
    seniority = _find_seniority(all_text)
    requested_types = _find_requested_test_types(all_text)
    force_recommend = _wants_recommendation_now(latest)

    dimensions = sum([has_skill, bool(seniority), bool(requested_types)])

    # Never recommend on turn 1 for a genuinely vague query.
    if prior_assistant_turns == 0 and not has_skill and not force_recommend:
        return {
            "reply": _clarifying_question(has_skill, bool(seniority)),
            "recommendations": [],
            "end_of_conversation": False,
        }

    # Ask at most one more clarifying question if we have a skill but nothing else,
    # and we haven't already asked twice (keep well inside the 8-turn cap).
    if (
        not force_recommend
        and dimensions < 2
        and prior_assistant_turns < 2
        and not requested_types
    ):
        return {
            "reply": _clarifying_question(has_skill, bool(seniority)),
            "recommendations": [],
            "end_of_conversation": False,
        }

    # 5. Retrieve.
    query = " ".join(user_msgs)
    results = index.search(
        query,
        top_k=8,
        test_types=requested_types or None,
        job_level=seniority,
    )
    if not results:
        # broaden: drop filters rather than return nothing
        results = index.search(query, top_k=5)

    if not results:
        return {
            "reply": (
                "I wasn't able to match that to anything in the SHL catalog. "
                "Could you tell me more about the specific role, skill, or assessment type you need?"
            ),
            "recommendations": [],
            "end_of_conversation": False,
        }

    recs = [
        {"name": r["name"], "url": r["url"], "test_type": r["test_type"]} for r in results
    ]
    names = ", ".join(r["name"] for r in recs[:5])
    reply = (
        f"Based on what you've shared, here are {len(recs)} SHL assessments that fit: {names}"
        f"{' and more' if len(recs) > 5 else ''}. Let me know if you'd like to narrow this down "
        f"further (e.g. add a personality assessment, change seniority, or drop one of these)."
    )
    return {"reply": reply, "recommendations": recs, "end_of_conversation": False}
