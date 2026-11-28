# Reading guide: how to study this codebase

A suggested order for reading the code, from the simplest pieces to the full agent. Each stage builds on the one before, so you never meet a concept you have not seen. The core is about 5,000 lines; plan on roughly 1 to 1.5 hours per stage.

Keep the [architecture document](AGENT_ARCHITECTURE.md) open as your map while you read.

---

## Stage 0. See it work first (30 minutes)

1. Start the app (`docker compose up`, then http://localhost:8501) and choose **Offline (rule-based)** in the sidebar. Offline mode needs no API key and gives the same result every time, which makes it easier to follow.
2. Run the **All-in-one** example. Approve the plan, then click through every tab, including **Agent trace**.
3. Read [AGENT_ARCHITECTURE.md](AGENT_ARCHITECTURE.md), especially the workflow diagram (section 2) and the node table (section 3).

**Check yourself:** can you name the four points where the app stops and waits for the buyer?

---

## Stage 1. The data shapes: [`hdb_advisor/state.py`](../hdb_advisor/state.py)

Start here because every other file passes these objects around.

| Read | Why it matters |
| --- | --- |
| `BuyerProfile` | Everything the agent knows about the buyer. Unknown is `None`, never a guess. |
| `MarketSummary`, `TownSummary` | What the market search returns |
| `RouteResult`, `Destination` | Commute results and the resolved destination |
| `GrantItem`, `Affordability` | Grant screening and the loan estimate, including the loan working |
| `AdvisorState` | The single dictionary that moves through the graph and is checkpointed after every step |

**Check yourself:** why are `destination` and `anchor_id` two separate fields?

---

## Stage 2. The tools: plain Python, no AI

These produce every number in the report. Read each one next to its test.

| File | Read these functions | Compare with |
| --- | --- | --- |
| [`market.py`](../hdb_advisor/market.py) | `ingest_csv` (validated, all-or-nothing loading), `search_market` (parameterised SQL, medians and quartiles) | [`evals/test_market.py`](../evals/test_market.py) |
| [`policy.py`](../hdb_advisor/policy.py) | `evaluate_grants`, `estimate_affordability`, `review_reasons`, `funding_split`, `retrieve_rules` | [`evals/test_policy.py`](../evals/test_policy.py) |
| [`onemap.py`](../hdb_advisor/onemap.py) | `match_address`, `resolve_destination`, `route_to_anchor` | [`evals/test_onemap.py`](../evals/test_onemap.py) |

Also open [`config/rules_2026-11-23.yaml`](../config/rules_2026-11-23.yaml) and follow how `threshold("ltv")` reads from it. Every HDB rule value lives there, next to its source URL.

**Try it:** change `ltv` to `0.8`, predict the new loan ceiling for a S$500,000 flat valued at S$480,000, then check it in the calculator. Change it back afterwards.

---

## Stage 3. The workflow: [`hdb_advisor/graph.py`](../hdb_advisor/graph.py) (the most important file)

1. **Read from the bottom up.** The last ~30 lines (`add_edge` and `add_conditional_edges`) define the whole flow. Match each line with the diagram.
2. **Read `prepare_turn`** to see what every new turn starts with, including follow-ups (`refine`) and form updates (`updates`).
3. **Read the nodes in the order they run:**
   `input_guard` → `intake` (and `with_destination`) → `router` → `route_after_classification` → `clarify` → `planner` → `confirm_plan` → `market` / `routes` / `policy` / `affordability` → `collect` → `writer` → `critic` → `approve_output` → `finish`
4. **Notice `assistant()`.** Every model call goes through it. It enforces the 10-call budget and falls back to the rule-based version when there is no model or a call fails.

**Check yourself:** which three steps run in parallel, and why does `routes` have to wait for `market`?

---

## Stage 4. The agents: [`hdb_advisor/agents.py`](../hdb_advisor/agents.py)

| Read | What you learn |
| --- | --- |
| `ROLES` | The system prompt for each of the six agents (prompt engineering) |
| `RouterOutput`, `PlanOutput`, `PolicyOutput`, `Narrative`, `Critique` | The structured output each agent must return |
| `offline_intake` with `find_destination`, `living_arrangement`, `stated_months_back` | The rule-based parser used offline and to backfill facts the model missed |
| `offline_router`, `default_plan` | Rule-based intent and planning |
| `GeminiAgents.call` | One structured Gemini call |
| `GeminiAgents.policy` | The real tool-using agent: built with `create_agent`, up to 2 rule lookups, wrapped in PII and call-limit middleware (agentic RAG) |

**Try it:** add a new phrasing to `find_destination` (for example "office in X") and write a test for it.

---

## Stage 5. Guardrails and the report

[`guards.py`](../hdb_advisor/guards.py)
- `redact`: removes NRIC, phone, card and email before anything else sees the text.
- `inspect_input`: flags injection attempts, investment advice and "is it for sale" questions.
- `check_narrative`: no invented numbers, overclaiming, links or personal data in model-written text.

[`report.py`](../hdb_advisor/report.py)
- `render_report`: the cited report, built by code rather than the model.
- `ranked_towns`, `commute_groups`: commute ranking and the within/above-limit groups.
- `writer_context`: exactly what the Writer agent is allowed to see.
- `claim_issues`: ties each figure to the town and measure it describes.
- `compose`: puts the narrative and the report together into the final Markdown.

**Check yourself:** why does "Woodlands has a median of S$600,000" fail when S$600,000 is a real number in the data?

---

## Stage 6. Learning and memory: [`hdb_advisor/memory.py`](../hdb_advisor/memory.py)

Read section 8 of the [architecture document](AGENT_ARCHITECTURE.md) first: it maps session, state, short-term and long-term memory to this code.

- **Short-term memory:** in `graph.py`, read `finish` (it appends a turn to `history` and stores `last_profile`/`last_plan`) and `recent_turns` (the last 3 turns given to the Router and Intake).
- **Long-term memory:** `search_summary` (deterministic, redacted), `record_search`, `recall_memories` (keyword plus local embeddings, instruction-like memories dropped), `remember_household` (consent only), `forget` / `forget_everything`.
- **Recall in the graph:** the `recall_memory` node and how `intake` ranks remembered facts below anything stated.
- **Few-shot learning:** `record_example` / `examples_for`; look at [`examples/`](../examples/).
- **In the UI:** `account_panel`, `conversations_panel`, `memory_panel` in `streamlit_app.py`.
- **Tests:** [`evals/test_memory.py`](../evals/test_memory.py) covers recall, isolation between users, consent, history, redaction, injection and deletion.

**Try it:** sign in, run a search, start a new conversation with a related brief, and find "Using what I remember" on the plan step.

---

## Stage 7. Human-in-the-loop and the UI: [`streamlit_app.py`](../streamlit_app.py)

| Read | What it does |
| --- | --- |
| `execute`, `queue` | Runs and resumes the graph with `Command(resume=…)`, bound to the conversation that raised the decision |
| `plan_gate`, `clarify_gate`, `output_gate`, `refresh_gate` | The four decision points as forms |
| `handling_panel`, `timeline`, `agent_roster` | How the app shows which agents ran |
| `calculator`, `follow_up` | Grants and loan calculator, and follow-up refinement |

Compare with `decision_for` in [`hdb_advisor/__main__.py`](../hdb_advisor/__main__.py): the same decisions, handled in the terminal.

---

## Stage 8. Bonus features

- [`mcp_tools.py`](../hdb_advisor/mcp_tools.py): the routes and export tools served over MCP and called through LangChain's `MCPAdapter`.
- [`observability.py`](../hdb_advisor/observability.py): Arize Phoenix tracing.
- [`freshness.py`](../hdb_advisor/freshness.py): checking data.gov.sg for newer data and refreshing only after approval.
- [`exports.py`](../hdb_advisor/exports.py): saving the approved shortlist safely.

---

## Stage 9. Testing and evaluation

- [`evals/scenarios.py`](../evals/scenarios.py): the 44 labelled cases (functional, missing data, honesty, injection, PII, benign).
- [`evals/harness.py`](../evals/harness.py): `check_case` decides pass or fail for each kind.
- [`evals/run_evaluation.py`](../evals/run_evaluation.py): runs every case with and without guardrails and compares.
- [`evals/test_review_fixes.py`](../evals/test_review_fixes.py): one test per bug found in the functional review; a good way to see edge cases.

---

## Habits that help

- **Follow one request end to end.** With tracing on, open Phoenix (http://localhost:6006) and expand a trace while reading `graph.py`.
- **Run one test at a time**, for example `python -m pytest -q -k follow_up`. Then deliberately break the code and watch the test fail.
- **Use offline mode while reading.** Switch to Gemini once you understand a step, and compare what changes.

---

## Splitting the work across the team

Matches the video sections in the CA6123 brief.

| Member | Study | Presents |
| --- | --- | --- |
| 1 | `state.py`, `graph.py`, Router and Planner in `agents.py` | Overview, Reason |
| 2 | `market.py`, `onemap.py`, `mcp_tools.py` | Action, MCP |
| 3 | `policy.py`, the Policy agent, `memory.py`, `examples/` | Perceive (agentic RAG), Learn |
| 4 | `guards.py`, `report.py` (`claim_issues`), `evals/`, `observability.py` | Responsible AI, evaluation, human-in-the-loop |
