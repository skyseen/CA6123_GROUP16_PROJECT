# HDB Resale Market Intelligence and Eligibility Advisor

This CA6123 prototype searches registered HDB resale transactions, summarizes historical prices, retrieves evidence from dated HDB and CPF rules, and estimates public-transport commutes when verified OneMap data is present. It sends buyers to the HDB Flat Portal for active listings and the HFE letter for an official grant and loan assessment.

Start with the [agent architecture](docs/AGENT_ARCHITECTURE.md), the [reading guide](docs/READING_GUIDE.md) for studying the code in order, the [phase summary](PHASE_SUMMARY.md) or the [runnable walkthrough](walkthrough.ipynb).

## Run everything with Docker (recommended)

Install Docker Desktop, copy `.env.example` to `.env` and fill in the keys you have, then run:

```bash
docker compose up --build
```

| Service | URL | What it is |
| --- | --- | --- |
| `app` | http://localhost:8501 | Streamlit UI for the advisor |
| `phoenix` | http://localhost:6006 | Arize Phoenix traces for every run |

`data/` and `logs/` are mounted from this folder, so the transaction snapshot, OneMap cache, checkpoints and preferences persist between restarts. Keys stay in `.env` and are never copied into the image. Run the CLI, tests or evaluation inside the same image:

```bash
docker compose run --rm app python -m pytest -q
docker compose run --rm app python -m evals.run_evaluation
docker compose run --rm app python -m hdb_advisor --offline --auto '4-room under 600k, commute to NTU'
docker compose down
```

### Using the UI

1. Choose the engine in the sidebar: Gemini agents (live) or offline (rule-based, no key needed).
   Sign in with a name (no password in this prototype) to keep **Your conversations** and **My memory** across sessions, or continue as a guest.
2. Describe the search in your own words, then select **Run research**. One brief can combine a budget, a commute limit, grant questions and a loan estimate. The **Try an example** chips only fill the box, so you can edit them first.
3. Read **How the assistant is handling your request**. It shows the interpreted request (with the facts still needed for grants), the research plan as step cards, the six agents and whether each has run, and a Perceive → Reason → Human → Action → Report → Learn timeline.
4. Answer the human-in-the-loop gates as they appear. At the plan gate, **Run with these filters** runs exactly the values on screen. Clarification asks only for the facts still missing and shows what was already understood.
5. Read the **Answer** first (for a commute brief, which sampled routes met your limit), then use **Refine this search** to ask a follow-up such as "What about Woodlands instead?". Your approved filters and household details carry forward, and the plan gate states what changed and what was kept. The Grants tab has **Add household details** to re-run the screening, and a search with no matches offers to raise the budget or widen the date range.
6. Read the result tabs: **Report** (cited Markdown, downloadable), **Market** (price bands in shortlist order, price against commute), **Grants and loan** (the agents' screening plus a calculator that shows the grant results, every step of the loan working and how the price is funded) and **Agent trace** (intent, plan, model calls, tokens, guard events, evidence).

## Run locally without Docker

Use Python 3.12. Run these commands from the project directory.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m hdb_advisor --offline --auto '4-room under S$600k, last 12 months, weekday commute to NTU'
python -m pip install -r requirements-app.txt
streamlit run streamlit_app.py
```

The submission ZIP includes a SQLite snapshot of 245,351 official transactions through November 2026. The GitHub repository does not include `data/`, so after cloning run `python -m scripts.ingest_hdb` first (and `python -m scripts.onemap_cache` with OneMap credentials for commute times). The latest month may be partial. The Chroma index builds locally on first run. Offline mode uses the local intake parser, checked figures, and any cached routes. It can report that a transit duration is missing.

To install or refresh the snapshot independently, use the official download endpoint or supply a local copy of the CSV.

```bash
python -m scripts.ingest_hdb
python -m scripts.ingest_hdb --csv /path/to/official-resale.csv
python -m scripts.snapshot_rules
```

## Run with Gemini and OneMap

Copy `.env.example` to `.env` and set `GOOGLE_API_KEY` with a course project key. The default model is `gemini-3.5-flash-lite`. Set `GEMINI_MODEL` to change it within the team's budget. Run without `--offline` to enable the LLM nodes and the once-per-thread metadata check. The graph caps a request at ten model calls (about 5k tokens on Flash-Lite). [Google Gemini model details](https://ai.google.dev/gemini-api/docs/pricing#gemini-3.5-flash-lite)

```bash
python -m hdb_advisor '4-room under 600k, commute within 45 min to NTU, and grants'
```

For OneMap, set `ONEMAP_EMAIL` and `ONEMAP_PASSWORD` in `.env`. The code requests and renews the 3-day API token itself. The buyer chooses the commute destination (an address, postal code, building or MRT station). The intake reads it from the brief, OneMap resolves it to its top search match, and the plan gate shows the matched address so the buyer can correct it. Resolved destinations and their routes are cached in `data/hdb.db`; the bundle already holds NTU North Spine and Raffles Place MRT, so those work offline. The regular weekday cache uses 08:00 Singapore time with the exact date shown in reports. [OneMap search](https://www.onemap.gov.sg/apidocs/search), [OneMap routing](https://www.onemap.gov.sg/apidocs/routing)

```bash
python -m scripts.onemap_cache --limit 20 --destination 'Raffles Place MRT' \
  --demo-brief '4-room under 600k, weekday commute to Raffles Place MRT'
python -m hdb_advisor --offline --auto '4-room under 600k, weekday commute to NTU'
```

The 20-address sample gate requires at least 18 unique matches and 15 transit itineraries. The demo brief also primes the specific blocks used in that example. Use `--limit 0` to prefill the full eligible sample within the OneMap account quota.

## Follow the code

| Stage | Entry point | What to trace |
| --- | --- | --- |
| Perceive | `guards.py`, `agents.py` | Redaction runs before the brief enters the checkpoint. Intake extracts only stated fields and leaves unknowns empty. |
| Reason | `graph.py` | Routing selects an intent, the planner selects tools, and the graph pauses for clarification or plan confirmation. |
| Action | `market.py`, `onemap.py`, `policy.py` | Market, policy, and affordability branch in parallel. Routes wait for historical block keys, then towns are ranked by commute. The Policy agent issues up to two evidence lookups and assesses each grant; `reconcile_grants` verifies that assessment against the rule engine and keeps the conservative result on disagreement. |
| Report | `report.py`, `graph.py` | Python renders the tool values and links in Markdown. The Report writer (market analyst) receives compact tool summaries and writes a summary, a three-town shortlist with reasons and next steps. The critic first applies hard rules (every figure must match a tool value, picks must be supplied towns, no availability or eligibility claims), then a model judge, and can request one rewrite with specific issues. |
| Learn | `memory.py` | A similarity selector retrieves up to three approved or seeded examples. SQLite stores preferences only when the buyer confirms they should be remembered. |

Market medians summarize transactions matching the supplied filters, including the budget cap. One example block per matching town is routed. When commute data exists, towns are ranked by verified transit time (within the limit first), then straight-line distance, then median price; towns are never removed for commute. The report labels verified routes above the commute limit and keeps straight-line distances in kilometres. The affordability tool reserves declared monthly debt from the payment budget as a conservative modelling assumption and refers lease-proration cases to HDB.

## Session and approval demo

```bash
python -m hdb_advisor --offline --thread demo --save \
  '4-room under 600k, weekday commute to NTU'
python -m hdb_advisor --offline --thread demo
```

The first command asks for plan and export approval. End input at a prompt to pause, then use the second command to resume. To demonstrate MCP, add `--mcp` to both invocations. The agent can use the route tool through FastMCP and LangChain's `MCPAdapter`. The graph makes the save tool available only after the buyer approves the exact report. For a separate MCP client, `python -m hdb_advisor.mcp_tools` serves the cached route tool over stdio. [LangChain MCP adapter](https://docs.langchain.com/oss/python/migrate/langchain-mcp-adapters)

The session directory defaults to `data/`. It holds the checkpoint, preference store, Chroma index and approved local exports. The run log records node names, timings, token counters and guard event labels in `logs/run.jsonl`. Use synthetic samples for traced demos because LangSmith traces can include prompts and state. If the team enables tracing, use `LANGSMITH_TRACING=true` and `LANGSMITH_PROJECT=ca6123-hdb`.

## Observe with Arize Phoenix

Phoenix is open source and runs locally with no account. Install the optional packages, start the server in its own terminal, then add `--trace`.

```bash
python -m pip install -r requirements-observability.txt
PHOENIX_WORKING_DIR=data/phoenix phoenix serve
python -m hdb_advisor --trace '4-room under 600k, commute to NTU within 45 min, and grants'
python -m evals.run_evaluation --gemini --trace
```

Open http://localhost:6006 and choose the `ca6123-hdb` project. Each request appears as one nested trace with the graph nodes, every Gemini call (prompt, structured output, token counts, latency), the Policy agent's lookups and its PII and call-limit middleware. Traces contain model prompts, so record demos with synthetic briefs. The JSON log in `logs/run.jsonl` remains as a lightweight fallback.

## Verify and package

```bash
python -m pytest -q
HDB_ORACLE_CSV=/path/to/official-resale.csv python -m pytest -q evals/test_market.py
python -m evals.run_evaluation
python -m evals.run_evaluation --gate
python -m evals.run_evaluation --gate --gemini
```

The pandas oracle expects the CSV used to build the local database. To run the Gemini comparison, use `python -m evals.run_evaluation --gemini`. Add `--langsmith` and configure `LANGSMITH_API_KEY` only when you want to publish the synthetic cases and scores as LangSmith experiments.

Run the packaging command with the actual group number to produce the assignment filename. The bundle includes the walkthrough, Python source, dependency lock, tests and public SQLite snapshot. Packaging excludes session state, credentials and run logs.

```bash
python -m scripts.package_submission --group GROUP_NUMBER
```

### Recording outline

The assignment calls for a continuous video of up to 12 minutes, with each team member appearing and narrating in their own voice. A possible running order is one minute for the problem and scope, two for Perceive and Reason, three for the Action demo, two for Learn and human oversight, and three for evaluations, limits and contributions. Each member should be able to trace one sample from the brief through their code and explain an evaluated failure mode.

When opening **My memory**, say “This prototype signs you in with just a name. Anyone using this copy of the app can enter the same name, reopen its conversations, and view, export or delete its saved memory.”

Put an accurate AI-use acknowledgment on the final slide. For example, “We used Codex to assist with implementation, tests, and the walkthrough. The team reviewed the code, validated the demo, and prepared and narrated the presentation.” Use that wording only after the team has done those steps.

## Public data attribution

Commute estimates and geocodes come from [OneMap](https://www.onemap.gov.sg) (Singapore Land Authority) and are cached in `data/hdb.db` so the bundle runs without credentials.


Contains information from HDB’s *Resale flat prices based on registration date from Jan-2017 onwards*, accessed on 23 November 2026, from data.gov.sg and made available under the Singapore Open Data Licence version 1.0. [HDB resale dataset](https://data.gov.sg/datasets/d_8b84c4ee58e3cfc0ece0d773c8ca6abc/view), [Singapore Open Data Licence](https://developers.data.gov.sg/open-data-licence)
