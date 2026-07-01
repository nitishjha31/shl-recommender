# Approach Document

*Fill in the bracketed sections with your own decisions and results before submitting.
Target: 2 pages max. Delete this instruction line and the bracket prompts when done.*

## 1. Problem framing

Built a stateless FastAPI service (`POST /chat`) that moves a user from a vague
hiring intent to a grounded shortlist of SHL Individual Test Solutions through
multi-turn dialogue, with support for clarification, refinement, and
comparison, while staying in scope.

## 2. Catalog & retrieval

- **Source**: scraped SHL's product catalog, restricted to Individual Test
  Solutions (`type=1`), via `data/scrape_catalog.py`. [State how many items you
  ended up with, and whether you hand-curated `keywords` per item or derived
  them automatically -- e.g. "N=XXX items; keywords derived by TF-IDF term
  extraction over name+description, then hand-corrected for the top 50 most
  common job families."]
- **Index**: TF-IDF (`scikit-learn`) over `name (x2) + keywords (x2) +
  description + job_levels`, cosine similarity at query time. Chosen over
  embeddings for zero external dependency inside a 30s-per-call budget and a
  catalog small enough that TF-IDF's vocabulary-overlap weakness doesn't bite
  much. [If you swapped to embeddings, describe the model and why.]
- **Filtering**: requested test types (personality/cognitive/coding/
  simulation) and seniority level soft-multiply score rather than hard-filter,
  so a strong topical match isn't discarded for a near-miss metadata field.

## 3. Conversation design (agent.py)

The agent is a deterministic state machine, not an LLM making
ask/retrieve/refuse decisions -- that choice was made specifically to avoid
non-deterministic conversational collapse across an 8-turn cap:

1. Guardrail the latest turn (off-topic / legal advice / prompt injection).
2. Check for comparison intent ("X vs Y") -- answered directly from stored
   catalog fields, never from the model's prior knowledge.
3. Extract slots (role/skill keyword match, seniority, requested test types)
   from the *entire* transcript on every call, since the endpoint is
   stateless.
4. Never recommend on turn 1 for a genuinely vague query; ask at most ~2
   clarifying questions total, then commit to a shortlist regardless of
   completeness, to respect the turn cap.
5. Refinement and correction fall out of step 3 for free -- there's no stored
   state to desync from what the user actually said.

An LLM (Anthropic/OpenAI, optional) is used only to *phrase* comparison
answers and clarifying questions more naturally; if no API key is configured,
deterministic templates are used instead, so the service is fully functional
with zero external calls. [State which provider/model you actually used, if any,
and whether you found the LLM-polished phrasing measurably better on the
behavior probes.]

## 4. Evaluation approach

[Describe what you actually ran: e.g. "Ran all 10 public traces via
tests/replay_trace.py against the deployed endpoint, logging turn count,
final shortlist, and whether recommendations matched
expected_shortlist_contains_any_of. Also wrote N additional behavior probes
for: off-topic refusal, injection resistance, turn-1-no-recommend, and
refinement honoring, in tests/test_api.py."]

**Public trace results**: [table or summary -- e.g. "8/10 traces produced a
shortlist containing at least one expected item within 8 turns; the 2 misses
were both due to catalog vocabulary gaps for X and Y skill areas, fixed by
adding keywords."]

**Recall@10 (public traces)**: [your number]

## 5. What didn't work / iteration notes

[Be concrete and specific -- this is one of the four things they're grading.
Examples of the kind of thing to report:]
- [e.g. "First version hard-filtered on requested test_type, which zeroed out
  relevant K-type results when the user only mentioned 'personality' for a
  secondary criterion -- switched to score multiplication instead."]
- [e.g. "Tried asking the LLM to decide the next action (ask/retrieve/refuse)
  each turn; on trace #4 it looped asking seniority twice despite already
  having it in the fact set, because it wasn't grounded in the transcript --
  moved fully to rule-based slot extraction."]
- [e.g. "Embeddings vs TF-IDF: tried X, saw Y% recall lift/no meaningful
  difference at this catalog size, kept TF-IDF for the dependency-free
  win."]

## 6. AI-tool usage disclosure

[State plainly what you used and for what -- e.g. "Used Claude/ChatGPT for:
scaffolding the FastAPI structure, drafting the scraper's CSS selectors,
generating the initial test list. Wrote/verified the agent state machine and
retrieval scoring logic myself; can walk through every branch in
agent.py."]

## 7. Known limitations

- [e.g. catalog coverage depends on scrape completeness/date]
- [e.g. regex guardrails will miss cleverly obfuscated injection attempts]
- [anything else you'd fix with more time]
