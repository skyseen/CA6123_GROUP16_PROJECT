"""Compile the advisor's Perceive, Reason, Action and Learn workflow."""
import asyncio
from dataclasses import dataclass
from functools import wraps
import json
from pathlib import Path
import re
import time

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from .agents import (Critique, Narrative, PlanOutput, RouterOutput, default_plan,
                     cleared_filters, offline_intake, offline_router, stated_months_back)
from .exports import save_research_shortlist
from .freshness import check_freshness, refresh_data
from .guards import INJECTION, inspect_input, redact
from .market import search_market
from .memory import (PREF_FIELDS, examples_for, household_memory, now_sg, preferences, recall_memories,
                     record_example, record_search, remember, remember_household, search_summary)
from .onemap import OneMap, resolve_destination, route_to_anchor
from .policy import (GRANT_PROFILE_FIELDS, estimate_affordability, evaluate_grants, reconcile_grants,
                     retrieve_rules)
from .report import compose, describe_missing, narrative_issues, render_report, writer_context
from .state import AdvisorState, BuyerProfile

# Intake, router and planner use three calls, the Policy agent up to three, and the writer and critic
# two each (one optional revision).
MODEL_CALL_LIMIT = 10
WRITER_RESERVE = 4
# One example block per matched town, so the shortlist can be ranked by commute. There are 26 towns.
ROUTE_LIMIT = 30
RECURSION_LIMIT = 25


@dataclass
class Settings:
    db: Path
    model: object = None
    vectorstore: object = None
    onemap: OneMap | None = None
    online: bool = False
    guards: bool = True
    mcp: bool = False
    log: Path | None = None
    export: Path = Path("data/shortlists")


# Search filters and household facts a follow-up ("What about Woodlands?") keeps from the last approved turn.
CARRIED_FIELDS = tuple(BuyerProfile.model_fields)
# Every household and flat fact the grant screening depends on; the buyer confirms them before it runs.
HOUSEHOLD_FIELDS = {"citizenship", "first_timer", "near_parents", "household_income_monthly", "buyer_ages",
                    "flat_remaining_lease", "continuous_employment_12m", "prerequisites_confirmed", "prior_phg"}
FINANCING_FIELDS = ("purchase_price", "valuation", "household_income_monthly", "buyer_ages",
                    "existing_monthly_debt", "flat_remaining_lease")


def clean_updates(updates: dict | None) -> tuple[dict, list[str]]:
    """Validate structured field edits from the UI (plan form, clarification form, household details)."""
    if not updates:
        return {}, []
    try:
        if not isinstance(updates, dict) or not updates.keys() <= set(CARRIED_FIELDS):
            raise ValueError("Unknown field")
        if any(isinstance(v, str) and (redact(v)[1] or INJECTION.search(v)) for v in updates.values()):
            raise ValueError("Untrusted value")
        values = {key: value for key, value in updates.items() if value is not None}
        BuyerProfile.model_validate(values)
        return updates, []
    except ValueError:
        return {}, ["invalid_update"]


def prepare_turn(brief: str, guards: bool = True, *, save: bool = False, refine: bool = False,
                 updates: dict | None = None, preapproved: bool = False) -> AdvisorState:
    """Start a turn. With refine=True the last approved profile is carried forward under the new brief."""
    clean, events = inspect_input(brief, guards)
    updates, update_events = clean_updates(updates)
    events = events + update_events
    return {
        "refine": refine, "updates": updates, "refinement": {}, "destination": {},
        "approved_report": "", "reapprove": False, "edit_issues": [], "preapproved": preapproved,
        "memories": [], "from_memory": [],
        "brief": clean, "profile": {}, "intent": "out_of_scope", "plan": [], "confidence": 0,
        "rationale": "", "market": {}, "routes": [], "grants": [], "evidence": [], "affordability": None,
        "narrative": {}, "canonical_report": "", "critic_issues": [], "guard_events": events,
        "tool_steps": [], "token_used": 0, "model_calls": 0, "policy_calls": 0, "policy_tokens": 0,
        "policy_events": [], "route_transport": "", "export_transport": "",
        "model_status": "offline", "revision": 0, "clarified": False,
        "cancelled": False, "save_requested": save, "saved_path": None,
    }


def build_graph(settings: Settings, checkpointer, store=None):
    graph = StateGraph(AdvisorState)

    def logged(name, node):
        @wraps(node)
        def run(state: AdvisorState, config):
            start = time.perf_counter()
            result = node(state, config)
            if settings.log:
                settings.log.parent.mkdir(parents=True, exist_ok=True)
                line = {"node": name, "scenario_id": config.get("metadata", {}).get("scenario_id"),
                        "elapsed_ms": round((time.perf_counter() - start) * 1000, 2),
                        "guard_events": result.get("guard_events", result.get("policy_events", [])),
                        "model_tokens_total": result.get("token_used"),
                        "policy_tokens": result.get("policy_tokens"),
                        "transport": result.get("route_transport") or result.get("export_transport")}
                with settings.log.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(line) + "\n")
            return result
        return run

    def assistant(state, config, role, schema, context, fallback, *, example_role=None):
        model = settings.model
        previous_calls, previous_tokens = state["model_calls"], state["token_used"]
        if model is None or previous_calls >= MODEL_CALL_LIMIT:
            return fallback, {}
        user_id = config.get("configurable", {}).get("user_id", "local")
        samples = examples_for(store, user_id, example_role, state["brief"]) if example_role else []
        meta = {"model_calls": previous_calls + 1, "model_status": "gemini"}
        try:
            result, tokens = model.call(role, schema, context, samples)
            return result, {**meta, "token_used": previous_tokens + tokens}
        except Exception:
            return fallback, {**meta, "model_status": "gemini_fallback",
                              "guard_events": list(dict.fromkeys(state["guard_events"] + ["model_fallback"]))}

    def input_guard(state, config):
        clean, events = inspect_input(state["brief"], settings.guards)
        return {"brief": clean, "guard_events": list(dict.fromkeys(state["guard_events"] + events))}

    def freshness(state, config):
        return {} if state.get("freshness") else {"freshness": check_freshness(settings.db, settings.online)}

    def refresh_gate(state, config):
        old = state["freshness"]
        decision = interrupt({"type": "refresh_data", "age_days": old["age_days"],
                              "last_month": old["source"]["last_month"], "options": ["update", "skip"]})
        if decision.get("action") != "update":
            return {"freshness": {**old, "status": "skipped",
                                  "note": "Refresh skipped. Using the local transaction snapshot."}}
        updated = refresh_data(settings.db, old)
        events = state["guard_events"] + (["refresh_failed"] if updated["status"] == "failed" else [])
        return {"freshness": updated, "guard_events": events}

    def memory_on(config) -> bool:
        """Long-term memory is used only for a signed-in user who has not switched it off."""
        return bool(config.get("configurable", {}).get("memory"))

    def recent_turns(state) -> list[dict]:
        """Short-term memory: the last three turns of this conversation, as compact data for the agents."""
        return [{"brief": t["brief"], "answer": t["answer"]} for t in (state.get("history") or [])[-3:]]

    def recall_memory(state, config):
        """Long-term memory: past searches relevant to this brief, recalled before the agents read it."""
        if not memory_on(config) or "direct_instruction" in state["guard_events"]:
            return {"memories": []}
        user_id = config.get("configurable", {}).get("user_id", "local")
        found, dropped = recall_memories(store, user_id, state["brief"])
        events = state["guard_events"] + (["memory_instruction_ignored"] if dropped else []) + (["memory_recalled"] if found else [])
        return {"memories": [{"text": m["text"], "at": m["at"]} for m in found],
                "guard_events": list(dict.fromkeys(events))}

    def with_destination(profile: BuyerProfile) -> tuple[BuyerProfile, dict]:
        """Resolve the buyer's commute destination with OneMap (or the cache). Only this sets anchor_id."""
        if not profile.destination:
            return profile.model_copy(update={"anchor_id": None}), {}
        found = resolve_destination(settings.db, profile.destination, settings.onemap)
        return profile.model_copy(update={"anchor_id": found.anchor_id}), found.model_dump()

    def intake(state, config):
        if "direct_instruction" in state["guard_events"]:
            return {"profile": {}}
        fallback = offline_intake(state["brief"])
        context = {"brief": state["brief"]}
        if recent_turns(state):
            context["earlier_in_this_conversation"] = recent_turns(state)
        if state.get("memories"):
            context["remembered_past_searches"] = [m["text"] for m in state["memories"]]
        result, meta = assistant(state, config, "intake", BuyerProfile, context, fallback, example_role="intake")
        user_id = config.get("configurable", {}).get("user_id", "local")
        fields = result.model_dump(exclude_none=True, exclude_unset=True)
        # The look-back window is read deterministically ("2 years" is 24 months). The schema default is
        # not a stated fact, so it must not override a carried window.
        stated = stated_months_back(state["brief"])
        if stated:
            fields["months_back"] = stated
        else:
            fields.pop("months_back", None)
        if result is not fallback:
            # The deterministic parser only matches explicit phrasings, so it backfills facts the model
            # missed. On a conflict the model's reading wins.
            parsed = fallback.model_dump(exclude_none=True, exclude_unset=True)
            filled = {key: value for key, value in parsed.items() if key not in fields}
            if filled:
                fields = filled | fields
                meta["guard_events"] = list(dict.fromkeys(meta.get("guard_events", state["guard_events"])
                                                          + ["intake_backfill"]))
        known = preferences(store, user_id)
        # Household facts are remembered only with consent, and rank below anything stated or carried.
        remembered = household_memory(store, user_id) if memory_on(config) else {}
        known = remembered | known
        carried = {}
        if state.get("refine") and state.get("last_profile"):
            carried = {k: v for k, v in state["last_profile"].items() if v is not None and v != []}
        updates = state.get("updates") or {}
        # "Remove the commute requirement" clears those filters explicitly, even when they were carried forward.
        cleared = {key: None for key in cleared_filters(state["brief"].split("\nClarification:")[-1])}
        merged = known | carried | fields | cleared | updates
        # An explicit None from a form clears a filter, for example removing the town.
        merged = {k: v for k, v in merged.items() if v is not None and k != "anchor_id"}
        profile, destination = with_destination(BuyerProfile.model_validate(merged))
        refinement = {}
        if carried:
            new = profile.model_dump()
            refinement = {"changed": {k: new[k] for k in new if new[k] != carried.get(k) and (k in carried or new[k] not in (None, []))
                                      and not (k == "months_back" and k not in carried and new[k] == 12)},
                          "kept": [k for k in carried if new.get(k) == carried[k]]}
        dumped = profile.model_dump()
        from_memory = [k for k, v in remembered.items()
                       if dumped.get(k) == v and k not in fields and k not in updates and k not in carried]
        return {"profile": dumped, "refinement": refinement, "destination": destination,
                "from_memory": from_memory, **meta}

    def router(state, config):
        if "direct_instruction" in state["guard_events"]:
            return {"intent": "out_of_scope", "confidence": 1.0}
        if "live_listing_request" in state["guard_events"]:
            return {"intent": "explain", "confidence": 1.0}
        if "investment_advice" in state["guard_events"]:
            return {"intent": "research", "confidence": 1.0}
        fallback = offline_router(state["brief"], BuyerProfile.model_validate(state["profile"]))
        context = {"brief": state["brief"]}
        if recent_turns(state):
            context["earlier_in_this_conversation"] = recent_turns(state)
        if state.get("refinement"):
            # A follow-up such as "What about Woodlands?" is read against the search it refines.
            context["refining_previous_search"] = {k: v for k, v in state["profile"].items() if v not in (None, [])}
        result, meta = assistant(state, config, "router", RouterOutput, context, fallback, example_role="router")
        if state.get("refinement") and result.intent == "out_of_scope":
            result = fallback
        return {**result.model_dump(), **meta}

    def route_after_classification(state):
        profile = state["profile"]
        needs_filter = (state["intent"] == "research" and "investment_advice" not in state["guard_events"]
                        and not any(profile.get(key) for key in ("max_price", "flat_type", "towns", "destination")))
        needs_financing = state["intent"] == "affordability" and any(
            profile.get(field) is None or profile.get(field) == [] for field in FINANCING_FIELDS
        )
        if not state["clarified"] and (state["confidence"] < .6 or needs_filter or needs_financing):
            return "clarify"
        return "planner"

    def clarify(state, config):
        profile = state["profile"]
        known = {k: v for k, v in profile.items() if v not in (None, [])}
        if state["intent"] == "affordability":
            # Ask only for what is still missing, and show what was already understood.
            missing = [f for f in FINANCING_FIELDS if profile.get(f) in (None, [])]
            question = f"To estimate the loan I still need {describe_missing(missing)}."
        else:
            missing = ["flat_type", "max_price", "towns", "destination"]
            question = "What would you like to research? For a market search, give a flat type, budget, town or commute anchor."
        decision = interrupt({"type": "clarify", "question": question, "missing": missing, "known": known,
                              "options": ["answer", "skip"]})
        if decision.get("action") != "answer":
            return {"clarified": True}
        updates, update_events = clean_updates(decision.get("updates"))
        answer, events = inspect_input(str(decision.get("text", "")), settings.guards)
        known_events = {"pii_nric", "pii_phone", "pii_credit_card", "pii_email", "direct_instruction",
                        "investment_advice", "live_listing_request"}
        events = list(dict.fromkeys(events + [x for x in decision.get("input_events", []) if x in known_events]))
        brief = f"{state['brief']}\nClarification: {answer}" if answer else state["brief"]
        return {"brief": brief, "clarified": True, "updates": {**(state.get("updates") or {}), **updates},
                "guard_events": list(dict.fromkeys(state["guard_events"] + events + update_events))}

    def planner(state, config):
        p = BuyerProfile.model_validate(state["profile"])
        fallback = default_plan(state["intent"], p, state["brief"])
        if state["intent"] in ("out_of_scope", "explain"):
            return fallback.model_dump()
        planner_context = {"intent": state["intent"], "profile": p.model_dump(exclude_none=True)}
        if state.get("memories"):
            planner_context["remembered_past_searches"] = [m["text"] for m in state["memories"]]
        result, meta = assistant(state, config, "planner", PlanOutput, planner_context, fallback)
        # The intent sets required tools. The planner may select optional tools within that intent.
        allowed = {"market", "grants", "afford", "report"} if state["intent"] == "research" else set(fallback.plan)
        if state["intent"] == "research" and p.anchor_id:
            allowed.add("route")
        # Advice requests may receive only a historical market summary.
        if "investment_advice" in state["guard_events"]:
            allowed &= {"market", "route", "report"}
        selected = (set(result.plan) | set(fallback.plan)) & allowed
        if state.get("refinement") and "investment_advice" not in state["guard_events"]:
            # A follow-up keeps the steps of the search it refines (for example grants and commute).
            selected |= set(state.get("last_plan") or []) - ({"route"} if not p.anchor_id else set())
        rationale, _ = redact(result.rationale)
        return {"plan": [k for k in ("market", "route", "grants", "afford", "report") if k in selected],
                "rationale": rationale, **meta}

    def confirm_plan(state, config):
        grants = "grants" in state["plan"]
        if "market" not in state["plan"] and not grants:
            return {}
        p = BuyerProfile.model_validate(state["profile"])
        search_fields = {"months_back", "max_price", "flat_type", "min_est_lease_years", "max_commute_min"} | PREF_FIELDS
        # Grant screening depends on household facts the agent interpreted; the buyer confirms them here.
        fields = search_fields | (HOUSEHOLD_FIELDS if grants else set())
        changed = set((state.get("refinement") or {}).get("changed", {}))
        if state.get("preapproved") and state.get("refinement") and not changed & search_fields:
            # The buyer typed these household and loan details and pressed a button to apply them, which is
            # the approval. Any change to the search filters still goes through this gate.
            return {"guard_events": list(dict.fromkeys(state["guard_events"] + ["plan_preapproved_by_buyer"]))}
        decision = interrupt({"type": "confirm_plan", "plan": state["plan"], "rationale": state["rationale"],
                              "filters": p.model_dump(include=search_fields, exclude_none=True),
                              "household": p.model_dump(include=HOUSEHOLD_FIELDS) if grants else {},
                              "options": ["approve", "edit", "cancel"]})
        action = decision.get("action")
        if action not in ("approve", "edit"):
            return {"cancelled": True}
        if action == "edit":
            changes = decision.get("changes", {})
            try:
                if not isinstance(changes, dict) or not changes.keys() <= fields:
                    raise ValueError("Unknown filter")
                if any(isinstance(v, str) and (redact(v)[1] or INJECTION.search(v)) for v in changes.values()):
                    raise ValueError("Untrusted edit")
                p = BuyerProfile.model_validate(p.model_dump() | {k: v for k, v in changes.items() if k != "anchor_id"})
            except ValueError:
                return {"cancelled": True, "guard_events": state["guard_events"] + ["invalid_plan_edit"]}
        user_id = config.get("configurable", {}).get("user_id", "local")
        if decision.get("remember") is True:
            remember(store, user_id, p)
        if decision.get("remember_household") is True and memory_on(config):
            # Explicit consent from the plan gate: keep these household facts for future conversations.
            remember_household(store, user_id, p)
        if action == "edit":
            example = p.model_dump(include=search_fields, exclude_none=True)
            # Record corrected filters with a generic prompt. Household information never enters the example store.
            record_example(store, user_id, "intake", {
                "input": f"Corrected market brief about {p.flat_type or 'resale flats'}",
                "output": json.dumps(example),
            })
        destination = state.get("destination") or {}
        if action == "edit" and "destination" in changes:
            p, destination = with_destination(p)
        plan = [step for step in state["plan"] if step != "route" or p.anchor_id]
        if p.anchor_id and "route" not in plan:
            plan.insert(plan.index("market") + 1, "route")
        return {"profile": p.model_dump(), "plan": plan, "destination": destination}

    def market(state, config):
        if "market" not in state["plan"]:
            return {}
        result = search_market(settings.db, BuyerProfile.model_validate(state["profile"]))
        return {"market": result.model_dump()}

    def routes(state, config):
        if "route" not in state["plan"] or not state["market"].get("towns"):
            return {}
        p = BuyerProfile.model_validate(state["profile"])
        # Sample one block from each town so a routing batch covers multiple areas.
        keys = [item["sample_blocks"][0] for item in state["market"]["towns"][:ROUTE_LIMIT] if item["sample_blocks"]]
        transport = "local"
        results = None
        if settings.mcp:
            try:
                from .mcp_tools import mcp_routes

                results = asyncio.run(mcp_routes(settings.db, settings.onemap, keys, p.anchor_id))
                transport = "mcp"
            except Exception:
                transport = "local_fallback"
        if results is None:
            results = [route_to_anchor(settings.db, key, p.anchor_id, settings.onemap) for key in keys]
        results.sort(key=lambda r: (r.duration_min is None, r.duration_min or 0))
        return {"routes": [row.model_dump() for row in results], "route_transport": transport}

    def policy(state, config):
        if "grants" not in state["plan"] and "afford" not in state["plan"]:
            return {}
        p = BuyerProfile.model_validate(state["profile"])
        query = ("CPF Family Grant EHG PHG first-timer income employment proximity lease" if "grants" in state["plan"]
                 else "HDB loan LTV repayment tenure interest floor CPF lease")
        evidence = retrieve_rules(query, store=settings.vectorstore)
        safe = [chunk for chunk in evidence if not (settings.guards and INJECTION.search(chunk.text))]
        events = ["indirect_instruction"] if len(safe) != len(evidence) else []
        calls = tokens = 0
        assessments = []
        wants_grants = "grants" in state["plan"]
        if settings.model and state["model_calls"] < MODEL_CALL_LIMIT - WRITER_RESERVE:
            budget = MODEL_CALL_LIMIT - state["model_calls"] - WRITER_RESERVE
            facts = p.model_dump(include=set(GRANT_PROFILE_FIELDS)) if wants_grants else None
            try:
                safe, tokens, calls, ignored, assessments = settings.model.policy(
                    query, settings.vectorstore, safe, budget, profile=facts)
                if ignored:
                    events.append("indirect_instruction")
            except Exception:
                calls = min(3, budget)
                events.append("model_fallback")
        grants = evaluate_grants(p) if wants_grants else []
        if grants and assessments:
            # The agent proposes, the rule engine verifies: disagreements keep the conservative rule result.
            grants, disagreements = reconcile_grants(grants, assessments, {c.id: c for c in safe}, p)
            events += disagreements
        return {"grants": [grant.model_dump() for grant in grants],
                "evidence": [chunk.model_dump(exclude={"text"}) for chunk in safe],
                "policy_calls": calls, "policy_tokens": tokens, "policy_events": events}

    def affordability(state, config):
        if "afford" not in state["plan"]:
            return {}
        return {"affordability": estimate_affordability(BuyerProfile.model_validate(state["profile"])).model_dump()}

    def collect(state, config):
        tools = [step for step in state["plan"] if step != "report" and (step != "route" or state["routes"])]
        events = list(dict.fromkeys(state["guard_events"] + state["policy_events"]))
        return {"tool_steps": tools, "guard_events": events, "model_calls": state["model_calls"] + state["policy_calls"],
                "token_used": state["token_used"] + state["policy_tokens"]}

    def writer(state, config):
        reference = render_report(state)
        if state["intent"] == "out_of_scope" or settings.model is None or "direct_instruction" in state["guard_events"]:
            return {"canonical_report": reference, "narrative": {}, "critic_issues": []}
        # Context engineering: the writer receives compact tool summaries, not the rendered report or raw rows.
        result, meta = assistant(state, config, "writer", Narrative,
                                 {"data": writer_context(state), "critique": state["critic_issues"]},
                                 Narrative(), example_role="reports")
        return {"canonical_report": reference, "narrative": result.model_dump() if result.summary.strip() else {},
                **meta}

    def critic(state, config):
        narrative = state["narrative"]
        issues = []
        meta = {}
        if narrative and settings.guards:
            # Hard rules first: grounded figures, supplied towns only, no availability or eligibility claims.
            issues = narrative_issues(narrative, state)
        if narrative and not issues:
            judge, meta = assistant(state, config, "critic", Critique,
                                    {"narrative": narrative, "data": writer_context(state)},
                                    Critique(acceptable=True))
            if settings.guards and not judge.acceptable:
                issues = [redact(item)[0][:300] for item in judge.issues] or [
                    "The narrative overstates the supplied evidence."]
        if any(not item["citations"] for item in state["grants"]):
            issues.append("A grant claim is missing its citation.")
        return {"critic_issues": issues, **meta}

    def revise(state, config):
        return {"revision": state["revision"] + 1}

    def after_critic(state):
        if state["critic_issues"] and state["revision"] == 0:
            return "revise"
        if state["critic_issues"] or (state["save_requested"] and state["intent"] != "out_of_scope"):
            return "approve_output"
        return "finish"

    def approve_output(state, config):
        # The buyer previews the exact Markdown that will be saved, downloaded and shown as the final report.
        narrative = {} if state["critic_issues"] else state["narrative"]
        report = compose(state["canonical_report"], narrative)
        decision = interrupt({"type": "approve_output", "issues": state["critic_issues"],
                              "edit_issues": state.get("edit_issues", []),
                              "report": report, "summary": narrative.get("summary", ""),
                              "options": ["approve", "edit", "reject"]})
        action = decision.get("action")
        if action not in ("approve", "edit"):
            return {"save_requested": False, "narrative": {}, "reapprove": False, "edit_issues": [],
                    "guard_events": state["guard_events"] + ["output_rejected"]}
        if action == "edit":
            # An edit is never saved unseen: it is checked, recomposed and shown again for approval.
            summary = str(decision.get("summary", decision.get("introduction", ""))).strip()
            edited = {**(state["narrative"] or {}), "summary": summary} if summary else {}
            problems = narrative_issues(edited, state) if edited else []
            if problems:
                # A failed edit changes nothing: the last checked preview and the Critic's decision stand.
                return {"edit_issues": problems, "reapprove": True}
            # A human edit that passes the same grounding rules replaces the narrative the Critic rejected.
            return {"narrative": edited, "critic_issues": [], "edit_issues": [], "reapprove": True}
        user_id = config.get("configurable", {}).get("user_id", "local")
        if narrative:
            # Learn from approved experience: the approved summary becomes a few-shot example for this user.
            record_example(store, user_id, "reports", {"input": state["intent"], "output": narrative["summary"]})
        path = None
        events = state["guard_events"]
        transport = ""
        if state["save_requested"]:
            thread = config["configurable"]["thread_id"]
            transport = "local"
            if settings.mcp:
                try:
                    from .mcp_tools import mcp_export

                    path = asyncio.run(mcp_export(settings.db, settings.export, report, thread))
                    transport = "mcp"
                except Exception:
                    transport = "local_fallback"
            if path is None:
                try:
                    path = save_research_shortlist(settings.export, report, thread)
                except OSError:
                    events = events + ["save_failed"]
        return {"narrative": narrative, "saved_path": path, "critic_issues": [], "approved_report": report,
                "reapprove": False, "edit_issues": [], "guard_events": events, "export_transport": transport}

    def finish(state, config):
        if state["cancelled"]:
            # A cancelled plan ran no tools; the UI shows a cancelled state rather than a report.
            return {"canonical_report": "", "narrative": {}, "market": {}, "routes": [], "grants": [],
                    "affordability": None}
        # The approved profile becomes the base for a follow-up refinement in this conversation.
        remembered = ({"last_profile": state["profile"], "last_plan": state["plan"]}
                      if state["intent"] != "out_of_scope" else {})
        if state["intent"] != "out_of_scope" and "direct_instruction" not in state["guard_events"]:
            # Short-term memory: a compact record of this turn, kept in the conversation's state.
            summary = search_summary(state)
            turn = {"at": now_sg().isoformat(timespec="minutes"), "brief": state["brief"][:200],
                    "intent": state["intent"], "plan": state["plan"], "answer": summary.split("Result: ", 1)[-1][:300]}
            remembered["history"] = (state.get("history") or [])[-9:] + [turn]
            if memory_on(config) and state.get("tool_steps"):
                # Long-term memory: a deterministic, redacted summary of the completed search.
                user_id = config.get("configurable", {}).get("user_id", "local")
                record_search(store, user_id, config["configurable"]["thread_id"], state)
        if state.get("approved_report"):
            return {"canonical_report": state["approved_report"], **remembered}
        if state["narrative"] and not state["critic_issues"]:
            return {"canonical_report": compose(state["canonical_report"], state["narrative"]), **remembered}
        return {"narrative": {}, **remembered}

    nodes = {
        "input_guard": input_guard, "recall_memory": recall_memory, "check_freshness": freshness,
        "refresh_gate": refresh_gate,
        "intake": intake, "router": router, "clarify": clarify, "planner": planner, "confirm_plan": confirm_plan,
        "market": market, "routes": routes, "policy": policy, "affordability": affordability, "collect": collect,
        "writer": writer, "critic": critic, "revise": revise, "approve_output": approve_output, "finish": finish,
    }
    for name, node in nodes.items():
        graph.add_node(name, logged(name, node))
    graph.add_edge(START, "input_guard")
    graph.add_edge("input_guard", "recall_memory")
    graph.add_edge("recall_memory", "check_freshness")
    graph.add_conditional_edges("check_freshness",
                                lambda s: "refresh_gate" if s["freshness"]["status"] == "stale" else "intake",
                                ["refresh_gate", "intake"])
    graph.add_edge("refresh_gate", "intake")
    graph.add_edge("intake", "router")
    graph.add_conditional_edges("router", route_after_classification, ["clarify", "planner"])
    graph.add_edge("clarify", "intake")
    graph.add_edge("planner", "confirm_plan")
    graph.add_conditional_edges("confirm_plan",
                                lambda s: ["finish"] if s["cancelled"] else ["market", "policy", "affordability"],
                                ["finish", "market", "policy", "affordability"])
    graph.add_edge("market", "routes")
    graph.add_edge(["routes", "policy", "affordability"], "collect")
    graph.add_edge("collect", "writer")
    graph.add_edge("writer", "critic")
    graph.add_conditional_edges("critic", after_critic, ["revise", "approve_output", "finish"])
    graph.add_edge("revise", "writer")
    graph.add_conditional_edges("approve_output", lambda s: "approve_output" if s.get("reapprove") else "finish",
                                ["approve_output", "finish"])
    graph.add_edge("finish", END)
    return graph.compile(checkpointer=checkpointer, store=store)
