# SHL Assessment Recommender -- Take-Home Starter

A stateless FastAPI service that turns a conversation ("I'm hiring a Java developer")
into a grounded shortlist of SHL Individual Test Solutions.

This repo is a **complete, working starting point** -- it runs, passes its own
tests, and satisfies the API contract. Read the whole README before you touch
code: several steps below are things *you* need to do (get the real catalog,
get the real 10 traces, pick your LLM key) that I could not do for you.

```
shl-recommender/
├── app/
│   ├── main.py         # FastAPI app: GET /health, POST /chat
│   ├── schemas.py       # Pydantic request/response models (matches spec exactly)
│   ├── agent.py         # The conversation state machine (the "brain")
│   ├── retrieval.py      # TF-IDF search over the catalog
│   ├── guardrails.py     # Off-topic / legal-advice / prompt-injection detection
│   └── llm.py            # Optional LLM wrapper (Anthropic/OpenAI), safe no-op if unset
├── data/
│   ├── catalog.json       # Seed catalog: ~40 real SHL assessments (see note below)
│   └── scrape_catalog.py  # Run this to pull the FULL, current catalog yourself
├── tests/
│   ├── test_api.py         # pytest suite covering the contract + behaviors
│   ├── replay_trace.py     # Local multi-turn simulator for one persona trace
│   └── sample_traces/      # ONE illustrative example -- not the real 10 traces
├── requirements.txt
├── Dockerfile
├── render.yaml            # One-click-ish deploy to Render's free tier
├── .env.example
└── APPROACH.md            # Fill this in -- it's your submission document
```

## Important: what you still need to do

1. **Get the real catalog.** `data/catalog.json` here is a curated seed set of
   ~40 real assessments (names, URLs, and descriptions I pulled from public
   search results), not a full crawl. Before you submit, run:
   ```
   pip install requests beautifulsoup4
   python data/scrape_catalog.py
   ```
   This walks the live SHL catalog (`type=1`, Individual Test Solutions only)
   and overwrites `data/catalog.json`. Inspect the output -- SHL's HTML
   structure changes occasionally, so check that `name`, `url`, `description`,
   and `test_type` are populating correctly for a sample of rows, and patch
   the CSS selectors in the script if not. Add a `keywords` list per item by
   hand or with a quick script -- retrieval quality depends heavily on this.

2. **Get the real 10 conversation traces.** The assignment doc references a
   zip of persona/fact/expected-shortlist traces that wasn't attached to what
   I was given. Download it from the link in your assignment email, unzip it
   into `tests/sample_traces/`, and use `tests/replay_trace.py` (or your own
   script) to run each one against your local server before submitting.

3. **Decide on an LLM key (optional but recommended).** The agent's *logic*
   (ask vs. retrieve vs. refuse vs. compare) is 100% rule-based in
   `app/agent.py` -- it never asks an LLM to decide what to do next, so it's
   fast and won't go incoherent mid-conversation. An LLM is only used, if
   configured, to (a) phrase comparison answers more naturally and (b) will
   fall back to clean deterministic templates if no key is set. Get a free
   key from Anthropic, OpenAI, or a free tier (Groq, OpenRouter -- point
   `app/llm.py` at whichever base URL you like) and put it in `.env`.

## Step-by-step: local setup

```bash
# 1. Clone / unzip this project, then:
cd shl-recommender
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) add your LLM key
cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY or OPENAI_API_KEY

# 4. Pull the real catalog (see section above) -- do this before relying on results
python data/scrape_catalog.py

# 5. Run the server
uvicorn app.main:app --reload --port 8000

# 6. Sanity check
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hiring a Java developer who works with stakeholders"}]}'

# 7. Run the test suite
pytest tests/ -v

# 8. Replay a full persona conversation locally (needs an LLM key to role-play the user)
python tests/replay_trace.py tests/sample_traces/example_trace.json
```

## Step-by-step: how the agent decides what to do (read this before editing agent.py)

Every `POST /chat` call re-derives everything from the full message history --
there is no database, session, or cache. On each call:

1. **Guardrail check** on the *latest* user message only (prior turns are
   legitimate history, not something to keep re-refusing). Catches prompt
   injection, legal-advice questions, and generic off-topic requests.
2. **Comparison check** -- regex for "X vs Y" / "difference between X and Y" /
   "compare X and Y". If both sides fuzzy-match a catalog item, answer
   directly from that item's stored fields (never invented).
3. **Slot extraction** over the whole conversation: does any user turn contain
   a role/skill keyword that matches catalog vocabulary? A seniority word? An
   explicit test-type request (personality / cognitive / coding / simulation)?
4. **Decide**:
   - Turn 1, no skill signal at all → ask a clarifying question, return an
     empty shortlist. (This is a hard-required behavior probe: never
     recommend on turn 1 for a vague query.)
   - Skill signal present but fewer than 2 total "dimensions" of context and
     we haven't already asked twice → ask one more clarifying question.
   - Otherwise → run TF-IDF retrieval over the catalog, filtered/boosted by
     any seniority level and test types mentioned, return 1-10 results.
5. **Refinement** falls out of this for free: "actually, add personality
   tests" just adds a new detected test-type filter next time slots are
   extracted from the (now longer) history, and retrieval re-runs from
   scratch with the updated filter set.

## Step-by-step: deploying

**Render (recommended, free tier):**
1. Push this repo to GitHub.
2. On [render.com](https://render.com), New → Web Service → connect the repo.
   It should auto-detect `render.yaml`. Otherwise set:
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
3. Add `ANTHROPIC_API_KEY` (or `OPENAI_API_KEY`) as an environment variable if
   you're using one.
4. Deploy. First `/health` call after a cold start can take up to the 2
   minutes the assignment allows for -- Render's free tier sleeps after
   inactivity.

**Fly.io / Railway / Modal / HF Spaces** all work the same way with the
included `Dockerfile` -- `docker build -t shl-recommender . && docker run -p
8000:8000 shl-recommender` locally to confirm the image works before pushing.

## Step-by-step: what to submit

1. Confirm both endpoints are reachable on your deployed URL:
   ```
   curl https://<your-app>/health
   curl -X POST https://<your-app>/chat -H "Content-Type: application/json" -d '{...}'
   ```
2. Fill in `APPROACH.md` (already scaffolded -- see below) and export/keep it
   as your 2-page approach document.
3. Submit the deployed URL + approach doc via the assignment form.

## Known limitations / where to spend more of your own time

- `data/catalog.json` needs the real scrape (step 1 above) -- recall@10 will
  be capped by catalog coverage otherwise.
- Retrieval is TF-IDF, not embeddings -- good enough for a small catalog and
  zero external dependencies, but if you have time, swapping in a sentence-
  embedding model (e.g. `sentence-transformers/all-MiniLM-L6-v2` run locally,
  no API needed) will likely lift recall on paraphrased/indirect queries.
- Guardrails are regex-based on purpose (fast, deterministic, no extra network
  call inside the 30s budget) -- expect some false negatives on cleverly
  phrased injection attempts. If you have budget left, an LLM-based
  classifier as a second-pass check on top of the regex layer would harden this.
- `_has_role_or_skill_signal` and the clarifying-question logic are simple
  heuristics -- read them, they're short, and tune the "how many clarifying
  turns before we just recommend" knob once you see how the real 10 traces
  behave.
