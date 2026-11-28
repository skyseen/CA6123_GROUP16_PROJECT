# HDB Advisor Development Summary

Updated 23 November 2026, Singapore time. The technical plan defines the implementation gates. The CA6123 brief defines the assessment criteria for Perceive, Reason, Action, Learn, human oversight, and responsible AI.

| Phase | Modules completed | Verification or remaining gate |
| --- | --- | --- |
| 1. Transactions | Typed state, CSV ingestion, SQLite, provenance, market filters and aggregates | Passed on 245,351 official rows through November 2026. All 11 source columns matched. Three market scenarios agreed with pandas. Corrupt data left the previous snapshot intact. |
| 2. OneMap | Exact address matching, token refresh, anchor configuration, dated transit cache and sample prefill | Three mocked OneMap checks passed. The 20-address live gate needs a OneMap account and confirmed anchor coordinates. |
| 3. Rules | Dated HDB and CPF corpus, Chroma index, Family Grant, EHG, PHG and loan illustration | All 10 top-four retrieval checks and two calculation checks passed. The local corpus uses curated, cited sections because the official HTML endpoints returned HTTP 403 when accessed directly. |
| 4. Workflow | LangGraph orchestration, six model-backed roles, parallel tool branches, deterministic reporting and bounded critic revision | The 10-scenario offline gate passed. A scripted LangChain model verified two policy retrieval calls, rejection of embedded instructions and selection of valid evidence. Live Gemini still needs the team's key. |
| 5. Oversight and evaluation | SQLite checkpoints and preferences, four human decisions, staged refresh, approved examples, local traces, LangSmith evaluation command and MCP | The guarded offline suite passed all 42 labelled cases. Four mocked freshness paths passed. Local MCP route and approved export both ran through the adapter. Hosted LangSmith experiments still need the team's key. |
| 6. Reproduction | README, runnable walkthrough, recording outline, dependency lock and source bundle | All notebook code cells ran on the bundled snapshot. The archive includes public transaction data and is ready for a filename with the team's group number. Recording and submission are team deliverables. |

## Phase 1 learning notes

Start with `hdb_advisor/state.py`. Each Pydantic model defines what can move between modules. For example, an absent income remains `None`, which is distinct from a declared income of zero.

Read `ingest_csv` and `search_market` in `hdb_advisor/market.py`. Ingestion validates and stages the snapshot inside one SQLite transaction. The query binds values through placeholders, then computes quartiles from sorted matching prices. The estimated lease uses the lease commencement year and the current Singapore month. The newest month in the snapshot sets the end of the observation window.

Compare this implementation with the pandas expressions in `evals/test_market.py`. Both answer the same questions using different methods. The official API reported the same exact row count as the CSV. The metadata endpoint supplies the source timestamp as `data.lastUpdatedAt`.

## Phase 2 learning notes

Read `match_address` in `hdb_advisor/onemap.py`. The code expands common road abbreviations, filters by both block and road, then requires one distinct coordinate and postal-code result. An ambiguous match has its own status.

A route cache key includes the actual departure date. A successful route stores the shortest OneMap itinerary in seconds. A straight-line fallback leaves `duration_min` empty. The live gates require at least 18 unique geocodes and 15 transit routes from the 20-address sample. The bundled snapshot has no verified route minutes yet.

## Phase 3 learning notes

The [rules YAML](config/rules_2026-11-23.yaml) puts an official source next to each threshold. Trace a missing household income through `evaluate_grants` and `GrantItem` to see how it reaches the report. The affordability tool computes the lower of the valuation-based limit and the payment-based limit. It signals when HDB needs to assess a pro-rated case. The implemented grant screen covers ordinary first-timer families. Other household schemes require review.

The Chroma corpus stores 12 dated sections across six official policy pages. Local word-feature embeddings allow reproducible retrieval without a paid embedding service. The Gemini policy agent can refine a query and retrieve a second set of passages. The ordinary family grant and housing loan income ceiling follows HDB's August 2026 update. [HDB income ceiling update](https://www.hdb.gov.sg/hdb-pulse/news/2026/increase-in-income-ceilings-and-greater-support-for-families-with-children)

## Phase 4 learning notes

Start at the bottom of `hdb_advisor/graph.py` for the edges. The planner chooses steps, the plan gate lets the user edit filters, and market, policy and affordability run in parallel. The route branch waits for the market query because it needs block keys.

The Report writer acts as the market analyst: it writes a summary, a three-town shortlist and next steps from compact tool summaries, and Python renders the figures and evidence. Every figure in the writer's text must match a tool value. Read `render_report` to see how a town median and a grant citation reach the output. The critic can trigger one rewrite. If the issue persists, the canonical data report reaches the approval gate for review.

The offline gate supports continuing development on Proposal A. The live routing and model checks are still open for the team's go/no-go decision. Proposal B was referenced in the plan, and its specification was outside the supplied files.

## Phase 5 learning notes

Read `confirm_plan` and `approve_output` to see how `interrupt` pauses the graph and `Command(resume=…)` delivers a decision. The checkpoint contains the run state. The preference store holds reusable preferences and approved examples under the local user namespace. The freshness check runs once per thread, and a refresh writes a new market snapshot after approval.

The optional MCP path uses FastMCP and `MCPAdapter`. The local server exposes routes. An approved export uses a short-lived server that binds the exact report before exposing the save tool. The ordinary function remains available as a fallback.

| Offline comparison | Baseline | Guarded |
| --- | ---: | ---: |
| Labelled scenarios passed | 24 / 42 | 42 / 42 |
| Functional scenarios passed | 12 / 12 | 12 / 12 |
| Grounded currency figures | 100% | 100% |
| Grant citation coverage | 100% | 100% |
| Instruction cases blocked | 0 / 10 | 10 / 10 |
| Benign controls blocked | 0 / 6 | 0 / 6 |

The comparison used the deterministic engine and the official transaction snapshot. The shared renderer supplied grounded figures in both modes. The full pytest run passed 70 checks with the official CSV oracle enabled. It also checked mocked refreshes, human approvals, and transport behavior. A separate terminal restart verified persistent resume and approved export. Run `python -m evals.run_evaluation` to measure current latency. Offline model usage is zero.

## Phase 6 learning notes

Use [walkthrough.ipynb](walkthrough.ipynb) to step through a buyer brief, inspect a market query, change the plan, and see a saved preference in a new conversation. Change the budget in the sample and predict the town counts before running it.

The source archive contains the notebook and public transaction snapshot. Run `python -m scripts.package_submission --group GROUP_NUMBER` with the actual group number. The [README](README.md) also covers running the live gates, recording the video and acknowledging AI assistance.

## Multi-agent upgrade (23 November 2026)

| Agent | What it decides | What verifies it |
| --- | --- | --- |
| Intake | Typed buyer profile from the brief | Pydantic contract, per-field validation |
| Router | Intent and confidence | Clarify gate below 0.6 |
| Planner | Steps from a fixed vocabulary | Intent sets the allowed steps |
| Policy | Passage retrieval and query rewriting (two lookups), then a cited status per grant | `reconcile_grants` compares with the rule engine. Disagreement keeps the conservative result and logs `policy_disagreement` |
| Report writer | Summary, three-town shortlist with reasons, next steps | Critic hard rules: grounded figures, supplied towns, no availability or eligibility claims |
| Critic | Whether the narrative overstates the data, with specific issues | One rewrite, then the human approval gate |

The live Gemini run on a grant and commute brief used about 4.7k tokens. The Policy agent agreed with the rule engine on all three grants. The evaluation now reports `narrative_published_rate` and `policy_agent_agreement_rate`.

## Live gates and verification (24 November 2026)

- OneMap: the anchor is set to NTU North Spine. The 20-address sample reached 19 of 20 unique geocodes and 19 of 20 transit routes (targets 18 and 15). The demo briefs cache one block for every matching town, so the 4-room NTU brief ranks Jurong West first at 39.7 minutes.
- Rules: every threshold and amount was re-checked against the live HDB and CPF pages. The EHG table link was updated, and the PHG screen now checks the 20-year lease condition.
- Observability: Arize Phoenix receives one nested trace per request. A live run recorded 115 spans, including 8 Gemini calls and 2 Policy-agent lookups.

## Front end and Docker (24 November 2026)

- `streamlit_app.py` drives the same LangGraph workflow. It streams each node into a live progress panel tagged by agentic stage, renders the four human-in-the-loop gates as forms, and shows the result as Report, Market (price bands and price against commute), Grants and loan, and Agent trace tabs.
- `docker compose up --build` starts the UI on http://localhost:8501 and Arize Phoenix on http://localhost:6006. Both are bound to localhost. `data/` and `logs/` are mounted, and keys are read from `.env` at run time. The test suite passes inside the image, and a live Gemini run exported 139 spans to the Phoenix container.
- The Intake agent sometimes missed explicit facts on live runs. The rule-based parser now backfills any field the model leaves empty (logged as `intake_backfill`). The model's value wins on conflict.

## UI review fixes (24 November 2026)

| Review finding | Fix | Regression test |
| --- | --- | --- |
| Approving the plan ignored an edited budget | One primary **Run with these filters** button submits the on-screen values as an edit | `test_plan_edit_is_the_budget_that_runs`, plus a browser check (S$500,000 applied) |
| A follow-up lost the original criteria | `prepare_turn(refine=True)` carries the last approved profile and plan; the gate states what changed and what was kept | `test_follow_up_keeps_the_search_and_changes_only_the_town`, plus a live Gemini check |
| Clarification re-asked known facts; the loan example missed the purchase price | The parser accepts "purchase price 500000"; clarification lists only missing fields with typed inputs; grants have an **Add household details** form | `test_clarification_asks_only_for_missing_loan_inputs` |
| Over-limit commutes were listed with the match | The report and Answer lead with the routes that met the limit, then group "Above N minutes" and "Travel time unknown", with the sample block on every line | `test_commute_results_lead_with_matches_and_zero_results_offer_no_blocks` |
| Zero results were inconsistent across tabs | One zero-result state with the applied filters, **Raise budget by 20%** and **Widen date range**; no "example blocks" line | same test |

## Destinations, naming and the calculator (25 November 2026)

- **Commute destination is the buyer's choice.** `BuyerProfile.destination` holds what the buyer wrote; `resolve_destination` turns it into coordinates (cache first, then OneMap's top match) and is the only code that sets `anchor_id`. The plan gate shows the matched address and lets the buyer edit it; an unknown destination is reported and no commute is estimated. NTU is only a pre-cached destination, not a default.
- **Look-back window renamed.** "Months of transactions" is now "Look back at sales from the last N months", with help text stating it is a search window and not a loan term. The maximum is 118 months, the length of the dataset.
- **Grants and loan calculator.** The tab runs the same rule engine on the buyer's numbers and shows each grant result, the loan working (repayment period, loan-to-value limit, payment budget, payment-based limit, binding constraint, instalment) and the funding split: price minus grants minus loan.

## Functional review fixes (25 November 2026)

Source: `output/pdf/hdb-functional-review.pdf`. Each finding has a regression test in `evals/test_review_fixes.py` unless noted.

| Finding | Fix |
| --- | --- |
| "We do not live with our parents, 10 km away" became living together and an S$30,000 PHG | `living_arrangement` checks negation and distance first and leaves unclear wording unknown. Grant requests now stop at the plan gate to confirm citizenship, first-timer status and living arrangement |
| Calculator kept a previous loan; financing exceeded the price | Each calculation is keyed to the report that produced it and labelled with its inputs. `funding_split` applies grants, then a loan no larger than the price left, so the parts sum to the price; the borrowing ceiling is shown separately |
| A real number attached to the wrong measure passed validation | `claim_issues` binds each S$ and minute figure to the town and measure it describes (median, range, budget, commute) |
| Approval preview differed from the saved export | The gate previews the composed final Markdown; an edited summary is re-checked and shown again; the approved bytes become the final report, download and export |
| A grants-and-loan request without a budget lost its grants | Finance tasks are planned independently of the main intent (scenario F14) |
| "Last 2 years" became 12 months; 120 months crashed a widget | `stated_months_back` reads months and years; one `MAX_MONTHS_BACK` (120) is used by the profile, parser, plan, refine and widen controls |
| Switching profile during approval raised `KeyError: 'brief'` | Each decision resumes the checkpoint that raised it; changing profile starts a new conversation (browser-verified) |
| Buyer-entered destinations could ship in the ZIP | `scrub_private_destinations` keeps only the listed demo destinations and their routes, then vacuums |
| Cancellation looked like a finished report | Cancel yields no report or metrics; the UI offers **Edit search** and **Start again** (browser-verified) |
| Policy archiver stopped on the first HTTP error | Each source is captured independently with its URL, HTTP status and section count |
| Clearing every preference left the old one stored | `remember` deletes the stored profile when nothing remains |
| Stale generic ZIP in `dist/` | Removed; only `ca6123_group16.zip` is built |

## Second functional review fixes (25 November 2026)

| Finding | Fix | Check |
| --- | --- | --- |
| "We will not live within 4 km of our parents" became Within 4 km and an S$20,000 PHG | Distance negation is handled; distances and negations are read only in clauses about the parent or child (including "…and they live 10 km away"); "our MRT is 2 km away" no longer counts | `test_distance_negation_and_relative_binding`; browser showed "Neither" |
| A failed summary edit restored the draft the Critic rejected | A failed edit changes nothing; the Critic's decision and the last checked preview stand; the edit error is shown separately | `test_failed_summary_edit_keeps_the_critics_rejection` |
| "Compare Woodlands and Tampines" searched all towns; "commute to Woodlands MRT" became a town filter; town capitalisation varied | `BuyerProfile.towns` is a validated list (any capitalisation, old single-town preferences still load); destinations and where relatives live are never town filters; the forms use a multi-select | `test_towns_and_destination_are_separate`; browser searched exactly 2 towns |
| "Remove the commute requirement" kept the commute; resets left the old brief | `cleared_filters` reads removal phrases for commute, budget, towns and lease; **Clear commute** action; new conversation and profile switch empty the brief with a **Reuse previous brief** option | `test_removing_the_commute_clears_destination_and_limit`; browser |
| Approval did not match the task | Grants-only requests show only the household and flat facts that affect screening, under **Check grants** | `test_grants_only_gate_shows_every_screening_fact`; browser |
| The answer was not the strongest element | Metrics follow the result (market, loan with inputs used, or grants); model usage stays in Agent trace; one next action under the answer (**Edit loan details**, **Add missing grant details** or **Compare towns**); new search and other changes are folded away | Browser |
| "First-timers" (plural) was not recognised | Parser accepts plural and "first time buyers" | `test_first_timer_plural_and_second_timer` |

## Memory management (28 November 2026)

- **Session and state (short-term):** each conversation is a LangGraph thread checkpointed after every node. `history` keeps each turn's brief and answer; the last 3 turns reach the Router and Intake. **Your conversations** lists threads and resumes one exactly, including a pending decision.
- **Long-term:** password-free sign-in keys a per-user namespace in the LangGraph store. Completed searches are saved as deterministic, redacted summaries and recalled (keyword plus local embeddings) by the new `recall_memory` node. Household facts are kept only with explicit consent and rank below anything stated. Guests have no long-term memory; **Use my memory** switches it off.
- **Control:** **My memory** shows, deletes and exports everything, with **Forget everything**.
- **Tests:** `evals/test_memory.py` (6 tests) covers cross-conversation recall, isolation between users, consent, history, redaction, injection via memory and deletion. Browser check: a new conversation showed "Using what I remember", and Resume restored the earlier answer.
