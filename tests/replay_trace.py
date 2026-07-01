"""
Minimal local replay harness -- NOT the real evaluator, just something to sanity
check a trace end-to-end before you submit.

It uses an LLM (Anthropic, if ANTHROPIC_API_KEY is set) to role-play the persona
truthfully from its facts, same as the real harness is described to do, and
talks to your locally running POST /chat until a shortlist comes back or the
8-turn cap is hit.

Usage:
    uvicorn app.main:app --port 8000 &
    ANTHROPIC_API_KEY=... python tests/replay_trace.py tests/sample_traces/example_trace.json
"""
import json
import sys

import requests

API_URL = "http://127.0.0.1:8000/chat"
MAX_TURNS = 8


def simulated_user_reply(persona: str, facts: dict, history: list[dict]) -> str:
    from app.llm import generate, llm_available

    if not llm_available():
        # Deterministic fallback: just dump the facts on turn 1, "no preference" after.
        if len(history) == 0:
            return f"I'm {persona}. " + "; ".join(f"{k}: {v}" for k, v in facts.items())
        return "No particular preference, just recommend what fits."

    transcript = "\n".join(f"{m['role']}: {m['content']}" for m in history)
    reply = generate(
        system=(
            f"You are role-playing this persona in a conversation with an assessment "
            f"recommender: {persona}. Known facts: {json.dumps(facts)}. Answer the "
            f"agent's questions truthfully from these facts. If asked something outside "
            f"these facts, say you have no preference. Keep replies short and natural. "
            f"If the agent has already given you a shortlist, say thanks and stop."
        ),
        user=f"Conversation so far:\n{transcript}\n\nYour next message:",
        max_tokens=100,
    )
    return reply or "No particular preference, just recommend what fits."


def main():
    trace_path = sys.argv[1] if len(sys.argv) > 1 else "tests/sample_traces/example_trace.json"
    trace = json.loads(open(trace_path).read())

    history: list[dict] = []
    for turn in range(MAX_TURNS):
        user_msg = simulated_user_reply(trace["persona"], trace["facts"], history)
        history.append({"role": "user", "content": user_msg})
        print(f"USER: {user_msg}")

        resp = requests.post(API_URL, json={"messages": history}, timeout=30).json()
        print(f"AGENT: {resp['reply']}")
        if resp["recommendations"]:
            print("RECS:", [r["name"] for r in resp["recommendations"]])
        history.append({"role": "assistant", "content": resp["reply"]})

        if resp["recommendations"] or resp.get("end_of_conversation"):
            break

    print("\n--- expected shortlist should contain one of ---")
    print(trace.get("expected_shortlist_contains_any_of"))


if __name__ == "__main__":
    main()
