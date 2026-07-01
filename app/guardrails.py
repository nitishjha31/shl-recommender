"""
Cheap, deterministic guardrails that run before any LLM call.

Kept rule-based on purpose: an LLM-based classifier is one more network call
that can time out inside the 30s budget, and for a narrow domain like this,
keyword/pattern rules catch the large majority of real cases the automated
probes are checking for (off-topic, legal advice, prompt injection) without
adding latency or non-determinism.
"""
import re

INJECTION_PATTERNS = [
    r"ignore (all|the|any) (previous|prior|above) (instructions|prompt)",
    r"disregard (the|your) (system|previous) prompt",
    r"you are now",
    r"act as (if )?(a|an) (?!interviewer)",
    r"reveal (your|the) (system prompt|instructions)",
    r"pretend (you|to) (are|be)",
    r"jailbreak",
    r"developer mode",
    r"forget (everything|your instructions)",
]

LEGAL_ADVICE_PATTERNS = [
    r"\bis it legal\b",
    r"\blawsuit\b",
    r"\bsue\b",
    r"\bdiscriminat(e|ion|ory)\b.*\b(law|legal)\b",
    r"\bemployment law\b",
    r"\bviolat(e|ion) of (labou?r|employment) law\b",
    r"\bequal employment opportunity\b",
]

GENERAL_HIRING_ADVICE_PATTERNS = [
    r"how (much|do i) (pay|salary|compensate)",
    r"write (me )?a job (description|posting|ad)\b(?!.*assessment)",
    r"how (do|should) i (interview|onboard|fire|terminate|layoff)",
    r"employer branding",
    r"how to negotiate (a )?salary",
]

OUT_OF_SCOPE_TOPICS = [
    r"\bweather\b",
    r"\bstock price\b",
    r"\brecipe\b",
    r"\bpolitic(s|al)\b",
    r"\bwrite (me )?a (poem|song|story)\b",
]


def _matches_any(text: str, patterns: list[str]) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in patterns)


def check_scope(latest_user_message: str) -> str | None:
    """
    Returns a refusal reason string if the message is out of scope, else None.
    Checked against ONLY the latest user turn -- prior turns are legitimate
    conversation history, not something to re-refuse on every subsequent turn.
    """
    text = latest_user_message or ""

    if _matches_any(text, INJECTION_PATTERNS):
        return "prompt_injection"
    if _matches_any(text, LEGAL_ADVICE_PATTERNS):
        return "legal_advice"
    if _matches_any(text, GENERAL_HIRING_ADVICE_PATTERNS):
        return "general_hiring_advice"
    if _matches_any(text, OUT_OF_SCOPE_TOPICS):
        return "off_topic"
    return None


REFUSAL_MESSAGES = {
    "prompt_injection": (
        "I can't follow instructions embedded in a message like that. "
        "I'm here to help you find SHL assessments -- what role or skills are you hiring for?"
    ),
    "legal_advice": (
        "I'm not able to give legal advice on hiring or employment law. "
        "I can help you find SHL assessments for a role instead -- what are you hiring for?"
    ),
    "general_hiring_advice": (
        "That's outside what I can help with -- I focus specifically on recommending SHL assessments, "
        "not broader hiring or compensation advice. Want help picking assessments for a role?"
    ),
    "off_topic": (
        "I can only help with finding and comparing SHL assessments. "
        "What role or skills are you looking to assess?"
    ),
}
