"""Streamlit front end for the HDB Resale Research Advisor.

It drives the same compiled LangGraph workflow as the CLI. Each node is streamed as it completes, and
the four human-in-the-loop interrupts (refresh, clarify, confirm plan, approve output) are rendered
as forms that resume the graph with ``Command(resume=...)``.

Run locally:   streamlit run streamlit_app.py
Run in Docker: docker compose up --build   (then open http://localhost:8501)
"""
import os
import hashlib
import json
import re
from html import escape
import sqlite3
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import altair as alt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore
from langgraph.types import Command

from hdb_advisor.agents import TOWNS, GeminiAgents
from hdb_advisor.graph import RECURSION_LIMIT, Settings, build_graph, prepare_turn
from hdb_advisor.guards import inspect_input, redact
from hdb_advisor.market import provenance
from hdb_advisor.observability import enable_phoenix, phoenix_enabled
from hdb_advisor.onemap import OneMap, place_label
from hdb_advisor.policy import (GRANT_PROFILE_FIELDS, estimate_affordability, evaluate_grants, funding_split,
                                index_rules, rules)
from hdb_advisor.report import commute_headline, describe_missing, ranked_towns
from hdb_advisor.memory import (forget, forget_everything, list_conversations, list_memory, now_sg,
                                record_conversation, remember_household)
from hdb_advisor.state import MAX_MONTHS_BACK, BuyerProfile

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
DATA = Path(os.getenv("HDB_DATA_DIR", ROOT / "data"))
DB = DATA / "hdb.db"
PHOENIX_URL = os.getenv("PHOENIX_PUBLIC_URL", "http://localhost:6006")
FLAT_TYPES = ["1 ROOM", "2 ROOM", "3 ROOM", "4 ROOM", "5 ROOM", "EXECUTIVE", "MULTI-GENERATION"]
# Example prompts only fill the editable brief. The graph decides which agents and tools a brief needs.
EXAMPLES = {
    "All-in-one: price, commute and grants": ("Find 4-room flats under S$600k within 45 minutes of Raffles Place MRT "
                                              "and check what information you need for grants."),
    "Commute shortlist": "4-room under S$600k, last 12 months, weekday commute to NTU within 45 min",
    "Grants for a couple": ("First-timer SC/SC couple, ages 30 and 32, household income 7000, 4-room under 600k, "
                            "commute to Raffles Place MRT within 45 min, and check grants"),
    "Loan estimate": ("Estimate my loan with purchase price 500000, valuation 480000, income 10000, "
                      "ages 32 and 33, existing debt 0, remaining lease 70"),
}
# What each plan step does and which agent or tool performs it.
PLAN_STEPS = {
    "market": ("Market search", "SQLite tool over registered resale transactions"),
    "route": ("Commute estimate", "OneMap transit, weekday 08:00, to your destination"),
    "grants": ("Grant screening", "Policy agent, verified by the rule engine"),
    "afford": ("Loan estimate", "Deterministic HDB loan calculator"),
    "report": ("Cited report", "Report writer agent, checked by the Critic"),
}
# The six model-backed roles behind the single assistant, keyed by their graph node.
AGENTS = [
    ("intake", "Intake", "Turns your words into a typed buyer profile"),
    ("router", "Router", "Classifies what you are asking for"),
    ("planner", "Planner", "Chooses the tools and their order"),
    ("policy", "Policy", "Retrieves rules and assesses each grant"),
    ("writer", "Report writer", "Writes the summary and town shortlist"),
    ("critic", "Critic", "Checks every figure and claim"),
]
STAGES = ["Perceive", "Reason", "Human", "Action", "Report", "Learn"]
# The search window is how far back to read registered sales. It is not a loan term: the loan tool
# sets repayment from HDB's rules (up to 25 years). The dataset starts in January 2017.
LOOKBACK_MAX = MAX_MONTHS_BACK
LOOKBACK_HELP = ("How far back to read registered resale transactions. This is a search window for past sales, "
                 "not a loan repayment period. Older sales say less about today's prices.")
CITIZENSHIP = {"SC_SC": "SC + SC", "SC_SPR": "SC + SPR", "SC_single": "Single SC", "other": "Other"}
PROFILE_LABELS = {
    "flat_type": ("Flat type", str.title),
    "max_price": ("Budget", lambda v: f"up to S${v:,.0f}"),
    "months_back": ("Past sales from", lambda v: f"the last {v} months"),
    "min_est_lease_years": ("Lease left", lambda v: f"at least {v} years"),
    "towns": ("Towns", lambda v: ", ".join(t.title() for t in v)),
    "avoid_floor": ("Avoid floors", str),
    "destination": ("Commute to", str),
    "max_commute_min": ("Commute limit", lambda v: f"{v} min"),
    "citizenship": ("Citizenship", lambda v: CITIZENSHIP.get(v, v)),
    "first_timer": ("First-timer", lambda v: "yes" if v else "no"),
    "household_income_monthly": ("Household income", lambda v: f"S${v:,.0f}/month"),
    "buyer_ages": ("Ages", lambda v: " and ".join(map(str, v))),
    "continuous_employment_12m": ("12-month employment", lambda v: "confirmed" if v else "not met"),
    "near_parents": ("Near parents", lambda v: {"live_with": "living together", "within_4km": "within 4 km",
                                                "no": "no"}.get(v, v)),
    "prerequisites_confirmed": ("Other prerequisites", lambda v: "confirmed" if v else "not met"),
    "prior_phg": ("Previous PHG", lambda v: "yes" if v else "no"),
    "purchase_price": ("Purchase price", lambda v: f"S${v:,.0f}"),
    "valuation": ("Valuation", lambda v: f"S${v:,.0f}"),
    "existing_monthly_debt": ("Monthly debt", lambda v: f"S${v:,.0f}"),
    "flat_remaining_lease": ("Flat's lease left", lambda v: f"{v:g} years"),
}
# Each graph node, the agentic stage it belongs to, and a plain-language label for the live progress feed.
NODES = {
    "input_guard": ("Perceive", "Input guard redacted identifiers and screened for injection"),
    "recall_memory": ("Perceive", "Recalled relevant memories from past conversations"),
    "check_freshness": ("Perceive", "Checked whether the transaction snapshot is current"),
    "refresh_gate": ("Human", "Refresh decision recorded"),
    "intake": ("Perceive", "Intake agent built the buyer profile"),
    "router": ("Reason", "Router agent classified the intent"),
    "clarify": ("Human", "Clarification received"),
    "planner": ("Reason", "Planner agent chose the tools"),
    "confirm_plan": ("Human", "Plan decision recorded"),
    "market": ("Action", "Market tool queried registered transactions"),
    "routes": ("Action", "Commute tool estimated OneMap transit times"),
    "policy": ("Action", "Policy agent retrieved rules and assessed grants"),
    "affordability": ("Action", "Loan tool computed the illustrative ceiling"),
    "collect": ("Action", "Tool results collected"),
    "writer": ("Report", "Report writer drafted the narrative"),
    "critic": ("Report", "Critic checked figures and claims"),
    "revise": ("Report", "Critic requested one rewrite"),
    "approve_output": ("Human", "Output decision recorded"),
    "finish": ("Learn", "Report finalised"),
}
STAGE_COLOURS = {"Perceive": "#2563EB", "Reason": "#7C3AED", "Action": "#0F766E", "Report": "#B45309",
                 "Human": "#BE185D", "Learn": "#15803D"}
STATUS_COLOURS = {"Within limit": "#0F766E", "Above limit": "#D97706", "Straight-line only": "#7C3AED",
                  "No verified time": "#94A3B8"}

st.set_page_config(page_title="HDB Resale Research Advisor", page_icon="🏠", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
.hero {background: linear-gradient(120deg, #0F766E 0%, #115E59 55%, #134E4A 100%); color: #F0FDFA;
       border-radius: 18px; padding: 1.6rem 1.8rem; margin-bottom: 1.2rem;}
.hero h1 {color: #FFFFFF; font-size: 1.9rem; margin: 0 0 .35rem 0;}
.hero p {margin: 0; opacity: .92; font-size: 1rem;}
.chips {margin-top: .8rem;}
.chip {display: inline-block; background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.25);
       border-radius: 999px; padding: .15rem .7rem; margin: 0 .35rem .35rem 0; font-size: .8rem;}
.stage {display: inline-block; border-radius: 6px; padding: .05rem .45rem; font-size: .72rem; font-weight: 600;
        color: white; margin-right: .5rem; min-width: 4.6rem; text-align: center;}
.badge {display: inline-block; border-radius: 999px; padding: .1rem .6rem; font-size: .75rem; font-weight: 600;}
.b-ok {background: #CCFBF1; color: #115E59;} .b-warn {background: #FEF3C7; color: #92400E;}
.b-no {background: #FEE2E2; color: #991B1B;} .b-info {background: #E0E7FF; color: #3730A3;}
.gate {border: 2px solid #BE185D; border-radius: 14px; padding: 1rem 1.2rem; background: #FFF7FB; margin: .6rem 0 1rem;}
.gate h3 {margin-top: 0; color: #9D174D;}
.card {border: 1px solid #E5E3DA; border-radius: 12px; padding: .9rem 1rem; background: #FFFFFF; margin-bottom: .7rem;}
.muted {color: #64748B; font-size: .85rem;}
.panel {border: 1px solid #E5E3DA; border-radius: 14px; background: #FFFFFF; padding: 1rem 1.1rem; height: 100%;}
.panel h4 {margin: 0 0 .6rem 0; font-size: .95rem; color: #134E4A; text-transform: uppercase; letter-spacing: .04em;}
.fact {display: inline-block; background: #F0FDFA; border: 1px solid #99F6E4; color: #134E4A; border-radius: 8px;
       padding: .2rem .55rem; margin: 0 .35rem .4rem 0; font-size: .85rem;}
.fact b {color: #0F766E; font-weight: 600; margin-right: .25rem;}
.unknown {display: inline-block; background: #FFFBEB; border: 1px dashed #F59E0B; color: #92400E; border-radius: 8px;
          padding: .2rem .55rem; margin: 0 .35rem .4rem 0; font-size: .82rem;}
.pipeline {display: flex; flex-wrap: wrap; gap: .5rem; align-items: stretch;}
.step {flex: 1 1 140px; border-radius: 12px; padding: .6rem .7rem; border: 1px solid #E5E3DA; background: #FAFAF7;}
.step .n {font-size: .7rem; font-weight: 700; color: #64748B;}
.step .t {font-weight: 700; font-size: .92rem; margin: .1rem 0;}
.step .d {font-size: .76rem; color: #475569; line-height: 1.25;}
.step .tag {display: inline-block; margin-top: .35rem; font-size: .7rem; font-weight: 700; border-radius: 999px; padding: .05rem .5rem;}
.s-proposed {border-color: #7C3AED; background: #F5F3FF;} .s-proposed .tag {background: #EDE9FE; color: #5B21B6;}
.s-done {border-color: #0F766E; background: #F0FDFA;} .s-done .tag {background: #CCFBF1; color: #115E59;}
.s-skip {opacity: .55;} .s-skip .tag {background: #E2E8F0; color: #475569;}
.s-nodata {border-color: #D97706; background: #FFFBEB;} .s-nodata .tag {background: #FEF3C7; color: #92400E;}
.agents {display: grid; grid-template-columns: repeat(auto-fit, minmax(118px, 1fr)); gap: .5rem;}
.agent {border: 1px solid #E5E3DA; border-radius: 12px; padding: .55rem .7rem; background: #FFFFFF;}
.agent .t {font-weight: 700; font-size: .9rem;} .agent .d {font-size: .75rem; color: #475569;}
.timeline {display: flex; gap: .4rem; flex-wrap: wrap;}
.tl {flex: 1 1 120px; border-radius: 10px; padding: .45rem .6rem; background: #F1F5F9; color: #64748B; font-size: .78rem;}
.tl .h {font-weight: 700; font-size: .8rem; margin-bottom: .15rem;}
.tl.on {color: #FFFFFF;}
.tl.wait {background: #FFF7FB; color: #9D174D; border: 2px dashed #BE185D;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------------------------------
# Shared resources: one checkpointer and preference store per process, one compiled graph per mode.
# ---------------------------------------------------------------------------------------------------
@st.cache_resource
def storage():
    DATA.mkdir(parents=True, exist_ok=True)
    saver = SqliteSaver(sqlite3.connect(DATA / "ckpt.db", check_same_thread=False))
    # The store manages its own transactions, so it needs an autocommit connection (as in from_conn_string).
    store = SqliteStore(sqlite3.connect(DATA / "memory.db", check_same_thread=False, isolation_level=None))
    store.setup()
    return saver, store


@st.cache_resource
def rule_index():
    return index_rules(DATA / "chroma")


@st.cache_resource
def tracing():
    return enable_phoenix() if phoenix_enabled() else None


@st.cache_resource
def advisor(live: bool, mcp: bool):
    model = GeminiAgents.from_env() if live else None
    settings = Settings(DB, model=model, vectorstore=rule_index(), onemap=OneMap.from_env() if live else None,
                        online=live, mcp=mcp, log=ROOT / "logs/run.jsonl", export=DATA / "shortlists")
    saver, store = storage()
    return build_graph(settings, saver, store)


def memory_enabled() -> bool:
    """Long-term memory is on only for a signed-in buyer who has not switched it off."""
    return bool(st.session_state.get("account")) and st.session_state.get("use_memory", True)


def config() -> dict:
    user = st.session_state.user
    # Long-term memory is used only for a signed-in buyer who has it switched on; guests get none.
    memory = memory_enabled()
    return {"configurable": {"thread_id": f"{user}/{st.session_state.thread}", "user_id": user, "memory": memory},
            "recursion_limit": RECURSION_LIMIT}


def money(value) -> str:
    return f"S${value:,.0f}"


# ---------------------------------------------------------------------------------------------------
# Running the graph: stream node updates into a live status panel, then read the checkpoint.
# ---------------------------------------------------------------------------------------------------
def execute(app, payload):
    # A resume goes to the exact checkpoint that raised the decision, whatever the sidebar shows now.
    cfg = st.session_state.get("active_config") or config()
    with st.status("Agents at work…", expanded=True) as status:
        for chunk in app.stream(payload, cfg, stream_mode="updates"):
            for node in chunk:
                if node in NODES:
                    stage, label = NODES[node]
                    st.markdown(f'<span class="stage" style="background:{STAGE_COLOURS[stage]}">{stage}</span>{label}',
                                unsafe_allow_html=True)
                    st.session_state.trail.append(node)
        snapshot = app.get_state(cfg)
        pending = [item for task in snapshot.tasks for item in task.interrupts]
        status.update(label="Waiting for your decision" if pending else "Research brief ready",
                      state="complete", expanded=False)
    st.session_state.result = snapshot.values
    st.session_state.gate = pending[0].value if pending else None


def queue(payload):
    if not isinstance(payload, Command):
        st.session_state.active_config = config()  # A new turn binds the conversation it belongs to.
        # Register the conversation so it can be found and resumed later (title = its first brief).
        record_conversation(storage()[1], st.session_state.user, st.session_state.active_config["configurable"]["thread_id"],
                            payload.get("brief", ""))
    st.session_state.queued = payload


# ---------------------------------------------------------------------------------------------------
# Sidebar: engine, approvals, memory label and data provenance.
# ---------------------------------------------------------------------------------------------------
st.session_state.setdefault("guest_id", f"guest-{uuid.uuid4().hex[:8]}")
defaults = {"thread": uuid.uuid4().hex[:12], "user": st.session_state.guest_id, "account": None, "use_memory": True,
            "result": None, "gate": None, "queued": None,
            "trail": [], "last_brief": EXAMPLES["All-in-one: price, commute and grants"]}
for key, value in defaults.items():
    st.session_state.setdefault(key, value)
# Streamlit drops a widget's state while it is hidden (during a gate), so restore the last brief.
st.session_state.setdefault("brief", st.session_state.last_brief)

def account_panel():
    """Password-free sign-in for this prototype: the name keys the buyer's memory on this machine."""
    store = storage()[1]
    account = st.session_state.get("account")
    if account:
        st.markdown(f"Signed in as **{escape(account)}**")
        st.session_state.use_memory = st.toggle(
            "Use my memory", value=st.session_state.get("use_memory", True),
            help="On: recall and save past searches, preferences and (with consent) household details. "
                 "Off: nothing is read from or saved to long-term memory; this conversation still keeps its own context.")
        if st.button("Sign out", use_container_width=True):
            fresh_start(profile_switched=False, user=st.session_state.guest_id, account=None)
            st.rerun()
        return
    with st.form("sign-in", border=False):
        name = st.text_input("Your name", placeholder="e.g. ken", help="No password in this prototype. "
                             "Your name keys your conversations and memory on this machine.")
        if st.form_submit_button("Sign in", use_container_width=True):
            user = re.sub(r"[^a-z0-9_-]", "-", name.strip().lower())[:32].strip("-")
            if not user or redact(name)[1]:
                st.warning("Use a short name without personal identifiers.")
            else:
                store.put(("users", user), "profile", {"name": name.strip()[:40], "last_seen": now_sg().isoformat(timespec="minutes")})
                # Each account has its own conversations; signing in never resumes a guest's pending decision.
                fresh_start(profile_switched=True, user=user, account=name.strip()[:40])
                st.rerun()
    st.caption("Browsing as a guest: conversations last for this session and nothing is remembered.")


def conversations_panel():
    """Short-term memory per conversation: list this user's conversations and resume one exactly where it was."""
    items = list_conversations(storage()[1], st.session_state.user)
    if not items:
        return
    st.divider()
    st.subheader("Your conversations")
    current = f"{st.session_state.user}/{st.session_state.thread}"
    for item in items[:8]:
        label = f"{item['title'][:42]}{'…' if len(item['title']) > 42 else ''} · {item['updated'][5:10]}"
        if item["thread"] == current:
            st.caption(f"▶ {label}")
        elif st.button(label, key=f"resume-{item['thread']}", use_container_width=True):
            st.session_state.resume_thread = item["thread"]
            st.rerun()


def memory_panel():
    """Everything the assistant remembers about the signed-in buyer, with delete and export."""
    if not st.session_state.get("account"):
        return
    store, user = storage()[1], st.session_state.user
    memory = list_memory(store, user)
    count = len(memory["searches"]) + bool(memory["preferences"]) + bool(memory["household"])
    st.divider()
    with st.expander(f"My memory ({count})", expanded=False):
        if memory["preferences"]:
            prefs = ", ".join(f"{PROFILE_LABELS[k][0]}: {PROFILE_LABELS[k][1](v)}" for k, v in memory["preferences"].items()
                              if k in PROFILE_LABELS)
            st.markdown(f"**Preferences** {escape(prefs)}")
            if st.button("Forget preferences", key="forget-prefs"):
                forget(store, user, "preferences")
                st.rerun()
        if memory["household"]:
            facts = ", ".join(f"{PROFILE_LABELS[k][0]}: {PROFILE_LABELS[k][1](v)}" for k, v in memory["household"].items()
                              if k in PROFILE_LABELS)
            st.markdown(f"**Household (kept with your consent)** {escape(facts)}")
            if st.button("Forget household details", key="forget-household"):
                forget(store, user, "household")
                st.rerun()
        if memory["searches"]:
            st.markdown("**Past searches**")
            for item in memory["searches"][:10]:
                st.caption(item["text"])
                if st.button("Forget this", key=f"forget-{item['key']}"):
                    forget(store, user, "search", item["key"])
                    st.rerun()
        if not count:
            st.caption("Nothing remembered yet. Completed searches are summarised here.")
        st.download_button("Export my memory (JSON)", json.dumps(memory, indent=2, ensure_ascii=False),
                           file_name=f"hdb-advisor-memory-{user}.json", mime="application/json")
        if st.button("Forget everything", type="primary",
                     help="Deletes your saved memory, your conversation list and every saved conversation."):
            forget_everything(store, user, storage()[0])
            # Nothing of the forgotten conversations stays on screen either.
            fresh_start(brief_reuse=False)
            st.session_state.previous_brief = ""
            st.rerun()


def fresh_start(brief_reuse: bool = True, **extra):
    """A new conversation starts empty: no result, no pending decision and no previous brief in the box.

    The previous brief is kept aside so the buyer can choose to reuse it.
    """
    previous = (st.session_state.get("brief") or st.session_state.get("last_brief") or "") if brief_reuse else ""
    st.session_state.update(thread=uuid.uuid4().hex[:12], result=None, gate=None, trail=[], active_config=None,
                            brief="", last_brief="", previous_brief=previous, **extra)


with st.sidebar:
    st.header("Session")
    has_key = bool(os.getenv("GOOGLE_API_KEY"))
    engine = st.radio("Engine", ["Gemini agents (live)", "Offline (rule-based)"], index=0 if has_key else 1,
                      help="Live mode uses Gemini for the six agents, OneMap for routes, and the data.gov.sg freshness check.")
    live = engine.startswith("Gemini")
    if live and not has_key:
        st.error("Set GOOGLE_API_KEY in .env to use the live agents.")
        live = False
    mcp = st.toggle("Route and export through MCP", value=False,
                    help="Calls the route and save tools through a FastMCP server and LangChain's MCPAdapter.")
    save = st.toggle("Save the approved shortlist", value=False,
                     help="Adds an output approval gate. The JSON export is written only after you approve.")
    st.session_state.save_flag = save
    account_panel()
    if st.button("New conversation", use_container_width=True):
        fresh_start()
        st.rerun()
    st.caption(f"Thread `{st.session_state.thread}`")
    if st.session_state.pop("profile_switched", False):
        st.info("Started a new conversation for this account.")
    conversations_panel()
    memory_panel()

    st.divider()
    st.subheader("Data provenance")
    source = provenance(DB) if DB.is_file() else None
    if source:
        retrieved = datetime.fromisoformat(source.retrieved_at).astimezone(ZoneInfo("Asia/Singapore")).date()
        st.markdown(f"**{source.rows:,}** registered resale transactions  \nLatest month **{source.last_month}**  \n"
                    f"Retrieved **{retrieved}** from [data.gov.sg]({source.url})")
    else:
        st.error("No transaction snapshot found. Run `python -m scripts.ingest_hdb`.")
    st.markdown(f"Rules snapshot **{rules()['snapshot_date']}**, verified {rules().get('verified_on', 'n/a')}")
    trace_url = tracing()
    st.divider()
    st.subheader("Observability")
    if trace_url:
        st.markdown(f"Tracing to Arize Phoenix. [Open traces]({PHOENIX_URL}/projects)")
    else:
        st.caption("Phoenix tracing is off. Set PHOENIX_TRACING=true (on by default in Docker).")

app = advisor(live, mcp)

if st.session_state.get("resume_thread"):
    # Resume a past conversation exactly where it was: its state, its result and any pending decision.
    thread_id = st.session_state.pop("resume_thread")
    st.session_state.update(thread=thread_id.split("/", 1)[1], active_config=None, trail=[], brief="")
    snapshot = app.get_state(config())
    pending = [item for task in snapshot.tasks for item in task.interrupts]
    st.session_state.result = snapshot.values or None
    st.session_state.gate = pending[0].value if pending else None
    st.rerun()  # Redraw so the sidebar marks the resumed conversation as current.

st.markdown("""
<div class="hero">
  <h1>HDB Resale Research Advisor</h1>
  <p>One assistant, six specialised agents. Describe your flat search in your own words. The assistant
  analyses registered resale transactions, estimates public-transport commutes with OneMap, screens CPF
  housing grants against dated official rules, and returns a cited shortlist of towns to look up on the
  HDB Flat Portal.</p>
  <div class="chips"><span class="chip">Historical transactions, not listings</span>
  <span class="chip">Every figure comes from a tool</span><span class="chip">You approve each plan</span>
  <span class="chip">HFE letter is the official assessment</span></div>
</div>
""", unsafe_allow_html=True)

# Run any decision queued by a button on the previous script pass, then redraw.
if st.session_state.queued is not None:
    payload, st.session_state.queued = st.session_state.queued, None
    st.session_state.failure = None
    try:
        execute(app, payload)
    except Exception as exc:  # Keep a readable failure on screen; the full traceback goes to the server log.
        traceback.print_exc()
        st.session_state.failure = f"The workflow stopped ({type(exc).__name__}: {str(exc)[:200]})."
    st.rerun()
if st.session_state.get("failure"):
    st.error(st.session_state.failure + " Check the API keys or switch to offline mode.")

# ---------------------------------------------------------------------------------------------------
# Brief input
# ---------------------------------------------------------------------------------------------------
gate = st.session_state.gate
def use_example():
    """Copy the chosen example into the editable brief, then clear the chip so it can be picked again."""
    choice = st.session_state.get("example")
    if choice:
        st.session_state.update(brief=EXAMPLES[choice], last_brief=EXAMPLES[choice], example=None)


def reuse_previous_brief():
    st.session_state.update(brief=st.session_state.get("previous_brief", ""), previous_brief="")


has_result = bool(st.session_state.result and (st.session_state.result.get("canonical_report")
                                               or st.session_state.result.get("cancelled")))
if gate is None:
    # Once there is an answer, a brand-new search steps back so the answer and its next step lead.
    composer = st.expander("Start a new search", expanded=False) if has_result else st.container()
    with composer:
        st.markdown("### What would you like to research?")
        st.caption("Ask in your own words. One request can combine a budget, a commute limit, grant questions and a "
                   "loan estimate. The assistant splits the work across its agents and shows you the plan before any "
                   "tool runs.")
        brief = st.text_area("Research brief", key="brief", height=110, label_visibility="collapsed",
                             placeholder="e.g. 4-room under S$600k within 45 minutes of Raffles Place MRT, and what do you need for grants?")
        run_column, reuse_column, _ = st.columns([1, 1.2, 1.8])
        if run_column.button("Run research", type="primary", use_container_width=True):
            if brief.strip():
                st.session_state.trail = []
                st.session_state.last_brief = brief
                queue(prepare_turn(brief, save=save))
                st.rerun()
            else:
                st.warning("Enter a research brief first.")
        if st.session_state.get("previous_brief") and not brief.strip():
            reuse_column.button("Reuse previous brief", on_click=reuse_previous_brief, use_container_width=True)
        st.pills("Try an example", list(EXAMPLES), key="example", on_change=use_example,
                 help="Examples only fill the box above. You can edit the text before running it.")


# ---------------------------------------------------------------------------------------------------
# How the assistant is handling the request: interpreted profile, plan, agents and workflow timeline.
# ---------------------------------------------------------------------------------------------------
def interpreted_request(values) -> str:
    profile = values.get("profile") or {}
    facts = []
    for key, (label, fmt) in PROFILE_LABELS.items():
        value = profile.get(key)
        if value is None or value == []:
            continue
        facts.append(f"<span class='fact'><b>{escape(label)}</b>{escape(str(fmt(value)))}</span>")
    intent = (values.get("intent") or "unknown").replace("_", " ")
    html = (f"<div class='muted'>Intent <b>{escape(intent)}</b> · router confidence "
            f"{values.get('confidence', 0):.2f}</div><div style='margin-top:.5rem'>"
            + ("".join(facts) or "<span class='muted'>No search filters were found in the brief.</span>") + "</div>")
    html += destination_note(values)
    from_memory = values.get("from_memory") or []
    if from_memory:
        names = ", ".join(PROFILE_LABELS[k][0].lower() for k in from_memory if k in PROFILE_LABELS)
        html += f"<div class='muted' style='margin-top:.4rem'>From your memory: {escape(names)}</div>"
    if values.get("memories"):
        recalled = "".join(f"<li>{escape(m['text'])}</li>" for m in values["memories"])
        html += (f"<div class='muted' style='margin-top:.4rem'>Using what I remember from earlier conversations:"
                 f"<ul style='margin:.2rem 0 0 1rem'>{recalled}</ul></div>")
    if "grants" in (values.get("plan") or []):
        missing = [key for key in GRANT_PROFILE_FIELDS if profile.get(key) in (None, [])]
        if missing:
            html += ("<div class='muted' style='margin-top:.4rem'>Still needed to screen grants</div>"
                     + "".join(f"<span class='unknown'>{escape(PROFILE_LABELS.get(k, (k,))[0])}</span>" for k in missing))
    return html


def destination_note(values) -> str:
    """Show what OneMap matched for the buyer's destination, so a wrong match is visible before anything runs."""
    found = values.get("destination") or {}
    if not found.get("query"):
        return ""
    if found.get("anchor_id"):
        where = f" · {escape(place_label(found['address']))}" if found.get("address") else ""
        return (f"<div class='muted' style='margin-top:.4rem'>Commute destination matched on OneMap: "
                f"<b>{escape(found['label'])}</b>{where}</div>")
    why = "not in the offline cache" if found.get("status") == "offline_unknown" else "not found on OneMap"
    return (f"<div class='unknown' style='margin-top:.4rem'>Destination \"{escape(found['query'])}\" {why}. "
            "Edit it below, for example with a postal code or MRT station.</div>")


def destination_label(values) -> str:
    found = values.get("destination") or {}
    return found.get("label") or (values.get("profile") or {}).get("destination") or "your destination"


def plan_pipeline(values, gate) -> str:
    plan = values.get("plan") or []
    used = set(values.get("tool_steps") or [])
    finished = gate is None and bool(values.get("canonical_report"))
    cards = []
    for index, (key, (title, detail)) in enumerate(PLAN_STEPS.items(), 1):
        if key not in plan:
            state, tag = "skip", "Not needed"
        elif values.get("cancelled"):
            state, tag = "skip", "Cancelled"
        elif not finished:
            state, tag = "proposed", "Awaiting your approval" if gate and gate["type"] == "confirm_plan" else "Planned"
        elif key == "report" or key in used:
            state, tag = "done", "Done"
        else:
            state, tag = "nodata", "No data returned"
        cards.append(f"<div class='step s-{state}'><div class='n'>STEP {index}</div><div class='t'>{title}</div>"
                     f"<div class='d'>{detail}</div><span class='tag'>{tag}</span></div>")
    rationale = values.get("rationale")
    note = f"<div class='muted' style='margin-top:.5rem'>Planner: {escape(rationale)}</div>" if rationale else ""
    return "<div class='pipeline'>" + "".join(cards) + "</div>" + note


def agent_roster(values) -> str:
    trail = set(st.session_state.trail)
    live = values.get("model_status") in ("gemini", "gemini_fallback")
    plan = values.get("plan") or []
    cards = []
    for node, name, job in AGENTS:
        if node == "policy" and not ({"grants", "afford"} & set(plan)):
            badge = "<span class='badge b-info'>Not needed</span>"
        elif node not in trail:
            badge = "<span class='badge b-info'>Waiting</span>"
        else:
            badge = ("<span class='badge b-ok'>Ran on Gemini</span>" if live
                     else "<span class='badge b-warn'>Rule-based (offline)</span>")
        cards.append(f"<div class='agent'><div class='t'>{name}</div><div class='d'>{job}</div>{badge}</div>")
    return "<div class='agents'>" + "".join(cards) + "</div>"


def timeline(gate=None) -> str:
    by_stage = {stage: [] for stage in STAGES}
    for node in dict.fromkeys(st.session_state.trail):
        if node in NODES:
            by_stage[NODES[node][0]].append(node.replace("_", " "))
    blocks = []
    for stage in STAGES:
        nodes = by_stage[stage]
        if stage == "Human" and gate is not None:
            # An open interrupt: the graph is paused on the buyer.
            waiting = gate["type"].replace("_", " ")
            blocks.append(f"<div class='tl wait'><div class='h'>Human</div>waiting for you: {waiting}</div>")
            continue
        style = f" style='background:{STAGE_COLOURS[stage]}'" if nodes else ""
        blocks.append(f"<div class='tl{' on' if nodes else ''}'{style}><div class='h'>{stage}</div>"
                      f"{', '.join(nodes) or 'not reached'}</div>")
    return "<div class='timeline'>" + "".join(blocks) + "</div>"


def handling_panel(values, gate, *, title="How the assistant is handling your request"):
    st.markdown(f"### {title}")
    left, right = st.columns([1, 1.35])
    left.markdown(f"<div class='panel'><h4>Interpreted request</h4>{interpreted_request(values)}</div>",
                  unsafe_allow_html=True)
    right.markdown(f"<div class='panel'><h4>Research plan</h4>{plan_pipeline(values, gate)}</div>",
                   unsafe_allow_html=True)
    st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
    # Technical detail stays one click away; the buyer's decision comes first.
    with st.expander("Agents and workflow progress", expanded=False):
        st.markdown(agent_roster(values), unsafe_allow_html=True)
        st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
        st.markdown(timeline(gate), unsafe_allow_html=True)


def describe(key, value) -> str:
    label, fmt = PROFILE_LABELS.get(key, (key.replace("_", " "), str))
    return f"{label.lower()} {fmt(value)}"


def refinement_sentence(values) -> str | None:
    """Confirm a follow-up in plain language: what changed and what was carried forward."""
    refinement = values.get("refinement") or {}
    if not refinement:
        return None
    shown = set(PROFILE_LABELS)
    changed = [f"{PROFILE_LABELS[k][0].lower()} to {PROFILE_LABELS[k][1](v)}"
               for k, v in refinement.get("changed", {}).items() if k in shown and v not in (None, [])]
    cleared = [PROFILE_LABELS[k][0].lower() for k, v in refinement.get("changed", {}).items()
               if k in PROFILE_LABELS and v in (None, [])]
    kept = [describe(k, values["profile"][k]) for k in refinement.get("kept", []) if k in shown]
    parts = []
    if changed:
        parts.append("Changing " + ", ".join(changed))
    if cleared:
        parts.append("removing " + ", ".join(cleared))
    if kept:
        parts.append("keeping " + ", ".join(kept))
    return ("; ".join(parts) + ".") if parts else "Re-running your previous search."


def applied_filters(values) -> str:
    if not values.get("market"):
        return ""  # Search filters only describe a market search; a loan or grant answer has none.
    profile = values.get("profile") or {}
    keys = ["flat_type", "max_price", "months_back", "min_est_lease_years", "towns", "destination", "max_commute_min"]
    chips = [f"<span class='fact'><b>{escape(PROFILE_LABELS[k][0])}</b>{escape(str(PROFILE_LABELS[k][1](profile[k])))}</span>"
             for k in keys if profile.get(k) not in (None, [])]
    return "".join(chips)


def start_refinement(brief: str, updates: dict | None = None, *, preapproved: bool = False):
    """Continue this conversation, carrying the approved filters and household facts forward."""
    st.session_state.trail = []
    queue(prepare_turn(brief, save=st.session_state.get("save_flag", False), refine=True, updates=updates or {},
                       preapproved=preapproved))
    st.rerun()


# ---------------------------------------------------------------------------------------------------
# Human-in-the-loop gates
# ---------------------------------------------------------------------------------------------------
def refresh_gate(gate):
    st.markdown(f'<div class="gate"><h3>Refresh the transaction data?</h3>The local snapshot is '
                f'{gate["age_days"]} days old (latest month {gate["last_month"]}) and data.gov.sg has newer data. '
                'Updating downloads the official CSV, validates it, and swaps it in only if it passes.</div>',
                unsafe_allow_html=True)
    left, right = st.columns(2)
    if left.button("Update now", type="primary", use_container_width=True):
        queue(Command(resume={"action": "update"}))
        st.rerun()
    if right.button("Skip and use the local snapshot", use_container_width=True):
        queue(Command(resume={"action": "skip"}))
        st.rerun()


FINANCE_INPUTS = {
    "purchase_price": ("Purchase price (S$)", 10_000.0), "valuation": ("Valuation (S$)", 10_000.0),
    "household_income_monthly": ("Gross monthly household income (S$)", 500.0),
    "existing_monthly_debt": ("Existing monthly debt (S$)", 100.0),
    "flat_remaining_lease": ("Flat's remaining lease (years)", 1.0),
}


def clarify_gate(gate):
    known = gate.get("known") or {}
    facts = "".join(f"<span class='fact'><b>{escape(PROFILE_LABELS[k][0])}</b>{escape(str(PROFILE_LABELS[k][1](v)))}</span>"
                    for k, v in known.items() if k in PROFILE_LABELS and k != "months_back")
    understood = f"<div style='margin-top:.5rem'><span class='muted'>Already understood</span><br>{facts}</div>" if facts else ""
    st.markdown(f'<div class="gate"><h3>One more detail</h3>{escape(gate["question"])}{understood}</div>',
                unsafe_allow_html=True)
    missing = [m for m in gate.get("missing", []) if m in FINANCE_INPUTS or m == "buyer_ages"]
    with st.form("clarify"):
        updates = {}
        if missing:
            columns = st.columns(min(3, len(missing)))
            for index, key in enumerate(missing):
                column = columns[index % len(columns)]
                if key == "buyer_ages":
                    ages = column.text_input("Applicant ages (e.g. 32, 33)")
                    updates[key] = [int(a) for a in re.findall(r"\d{2,3}", ages)] or None
                else:
                    label, step = FINANCE_INPUTS[key]
                    updates[key] = column.number_input(label, min_value=0.0, step=step, value=None)
        text = st.text_area("Anything else (optional)" if missing else "Your answer")
        answer, skip = st.columns(2)
        send = answer.form_submit_button("Continue", type="primary", use_container_width=True)
        skipped = skip.form_submit_button("Skip", use_container_width=True)
    if skipped:
        queue(Command(resume={"action": "skip"}))
        st.rerun()
    if send:
        updates = {k: v for k, v in updates.items() if v is not None}
        if not updates and not text.strip():
            st.warning("Fill in at least one detail, or choose Skip.")
            return
        clean, events = inspect_input(text) if text.strip() else ("", [])
        queue(Command(resume={"action": "answer", "text": clean, "input_events": events, "updates": updates}))
        st.rerun()


FACT_SPECS = {
    "citizenship": ("Citizenship", "choice", [None, *CITIZENSHIP], lambda v: "Unknown" if v is None else CITIZENSHIP[v]),
    "first_timer": ("All applicants first-timers?", "yesno"),
    "near_parents": ("Living with or near parents/child", "choice", [None, "live_with", "within_4km", "no"],
                     lambda v: {None: "Unknown", "live_with": "Living together", "within_4km": "Within 4 km",
                                "no": "Neither"}[v]),
    "household_income_monthly": ("Gross monthly household income (S$)", "money"),
    "buyer_ages": ("Applicant ages (e.g. 30, 32)", "ages"),
    "flat_type": ("Flat type", "choice", [None, *FLAT_TYPES], lambda v: "Unknown" if v is None else v.title()),
    "flat_remaining_lease": ("Flat's remaining lease (years)", "years"),
    "continuous_employment_12m": ("Worked continuously for 12 months?", "yesno"),
    "prerequisites_confirmed": ("Other HDB prerequisites checked?", "yesno"),
    "prior_phg": ("Received PHG before?", "yesno"),
    "purchase_price": ("Purchase price (S$)", "money"),
    "valuation": ("Valuation (S$)", "money"),
    "existing_monthly_debt": ("Existing monthly debt (S$)", "money"),
}
YES_NO = [None, True, False]


def fact_inputs(keys, values, prefix) -> dict:
    """Render inputs for household, flat and loan facts, three per row. Unknown stays None."""
    chosen = {}
    columns = st.columns(3)
    for index, key in enumerate(keys):
        label, kind, *extra = FACT_SPECS[key]
        column = columns[index % 3]
        current = values.get(key)
        widget = f"{prefix}-{key}"
        if kind == "choice":
            options, fmt = extra
            chosen[key] = column.selectbox(label, options, index=options.index(current) if current in options else 0,
                                           format_func=fmt, key=widget)
        elif kind == "yesno":
            chosen[key] = column.selectbox(label, YES_NO, index=YES_NO.index(current) if current in YES_NO else 0,
                                           format_func=lambda v: {None: "Unknown", True: "Yes", False: "No"}[v], key=widget)
        elif kind == "ages":
            text = column.text_input(label, value=", ".join(map(str, current or [])), key=widget)
            chosen[key] = [int(a) for a in re.findall(r"\d{2,3}", text)] or None
        else:
            step, top = {"money": (500.0, None), "years": (1.0, 99.0)}[kind]
            chosen[key] = column.number_input(label, min_value=0.0, max_value=top, step=step,
                                              value=float(current) if current is not None else None, key=widget)
    return chosen


def plan_gate(gate):
    steps = " → ".join(PLAN_STEPS[step][0] for step in gate["plan"] if step in PLAN_STEPS)
    searching, screening = "market" in gate["plan"], "grants" in gate["plan"]
    title, action = (("Check your filters and household facts, then run", "Run search and check grants")
                     if searching and screening else
                     ("Check your household facts", "Check grants") if screening else
                     ("Check your filters, then run", "Run with these filters"))
    sentence = refinement_sentence(st.session_state.result or {})
    follow = f"<div style='margin:.3rem 0 .5rem'>{escape(sentence)}</div>" if sentence else ""
    st.markdown(f'<div class="gate"><h3>{title}</h3>{follow}<b>{steps}</b><br>'
                f'<span class="muted">Adjust anything below. The run uses exactly the values shown.</span></div>',
                unsafe_allow_html=True)
    current = gate["filters"]
    household = gate.get("household") or {}
    with st.form("plan"):
        wanted = {}
        if searching:
            wanted = search_filter_inputs(current)
        if screening:
            st.markdown("**Household and flat facts used for the grant screening**")
            facts = ["citizenship", "first_timer", "near_parents", "household_income_monthly", "buyer_ages",
                     "flat_remaining_lease", "continuous_employment_12m", "prerequisites_confirmed", "prior_phg"]
            if not searching:
                facts.insert(5, "flat_type")
            wanted |= fact_inputs(facts, {**household, **current}, "plan")
        remember = keep_household = False
        if memory_enabled():
            remember = (st.checkbox("Remember my towns, floor and commute destination for future conversations")
                        if searching else False)
            keep_household = (st.checkbox("Remember these household facts for future conversations",
                                          help="Stored only for your account. You can view or delete them under My memory.")
                              if screening else False)
        elif st.session_state.get("account"):
            st.caption("Memory is off: nothing from this plan will be remembered.")
        run, cancel, _ = st.columns([1.4, 1, 1.6])
        submitted = run.form_submit_button(action, type="primary", use_container_width=True)
        cancelled = cancel.form_submit_button("Cancel", use_container_width=True)
    if cancelled:
        queue(Command(resume={"action": "cancel"}))
        st.rerun()
    if submitted:
        shown = {**current, **household}
        changes = {key: value for key, value in wanted.items() if shown.get(key) != value
                   and not (value in (None, []) and shown.get(key) in (None, []))}
        # Whatever is on screen is what runs: any difference from the proposal is sent as an edit.
        if changes:
            try:
                BuyerProfile.model_validate({k: v for k, v in changes.items() if v not in (None, [])})
            except ValueError:
                st.error("One of the values is not valid. Check the ages (18 to 120) and the numbers.")
                return
            queue(Command(resume={"action": "edit", "changes": changes, "remember": remember,
                                  "remember_household": keep_household}))
        else:
            queue(Command(resume={"action": "approve", "remember": remember, "remember_household": keep_household}))
        st.rerun()


def search_filter_inputs(current) -> dict:
    """The market search filters, prefilled with the agent's reading of the brief."""
    a, b, c = st.columns(3)
    flat = a.selectbox("Flat type", ["Any", *FLAT_TYPES],
                       index=(FLAT_TYPES.index(current["flat_type"]) + 1) if current.get("flat_type") else 0)
    price = b.number_input("Maximum price (S$, 0 = none)", min_value=0, step=10_000,
                           value=int(current.get("max_price") or 0))
    months = c.slider("Look back at sales from the last … months", 1, LOOKBACK_MAX,
                      min(LOOKBACK_MAX, int(current.get("months_back", 12))),
                      help=LOOKBACK_HELP)
    d, e, f = st.columns(3)
    lease = d.number_input("Minimum estimated lease (years, 0 = none)", 0, 99,
                           int(current.get("min_est_lease_years") or 0))
    commute = e.number_input("Commute limit (minutes, 0 = none)", 0, 300,
                             int(current.get("max_commute_min") or 0))
    towns = f.multiselect("Towns (none = all towns)", TOWNS, default=[t for t in current.get("towns", []) if t in TOWNS],
                          format_func=str.title)
    destination = st.text_input("Commute destination (address, postal code, building or MRT station; blank = none)",
                                value=current.get("destination") or "")
    st.markdown(destination_note(st.session_state.result or {}), unsafe_allow_html=True)
    return {"flat_type": None if flat == "Any" else flat, "max_price": float(price) or None, "months_back": months,
            "min_est_lease_years": lease or None, "max_commute_min": commute or None, "towns": towns,
            "destination": destination.strip() or None}


def output_gate(gate):
    issues = "".join(f"<li>{escape(issue)}</li>" for issue in gate.get("issues", []))
    body = (f"The Critic did not accept the generated summary, so it is left out:<ul>{issues}</ul>"
            "Below is exactly what will be saved and downloaded." if issues
            else "Below is exactly what will be saved and downloaded.")
    if gate.get("edit_issues"):
        edit_problems = "".join(f"<li>{escape(issue)}</li>" for issue in gate["edit_issues"])
        body += f"<div style='margin-top:.5rem'><b>Your last edit was not applied:</b><ul>{edit_problems}</ul></div>"
    st.markdown(f'<div class="gate"><h3>Approve the final report</h3>{body}</div>', unsafe_allow_html=True)
    with st.container(height=420, border=True):
        st.markdown(gate["report"])
    with st.form("approve"):
        summary = st.text_area("Edit the summary (figures must match the report; you will see the result before saving)",
                               value=gate.get("summary", ""))
        a, b, c = st.columns(3)
        if a.form_submit_button("Approve and save", type="primary", use_container_width=True):
            queue(Command(resume={"action": "approve"}))
            st.rerun()
        if b.form_submit_button("Apply edit and preview", use_container_width=True):
            queue(Command(resume={"action": "edit", "summary": redact(summary)[0]}))
            st.rerun()
        if c.form_submit_button("Reject", use_container_width=True):
            queue(Command(resume={"action": "reject"}))
            st.rerun()


values = st.session_state.result
if gate is not None:
    {"refresh_data": refresh_gate, "clarify": clarify_gate, "confirm_plan": plan_gate,
     "approve_output": output_gate}[gate["type"]](gate)
    if values:
        handling_panel(values, gate)


# ---------------------------------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------------------------------
def town_frame(values) -> pd.DataFrame:
    limit = (values.get("profile") or {}).get("max_commute_min")
    rows = []
    for rank, (town, route) in enumerate(ranked_towns(values), 1):
        minutes = route.get("duration_min") if route else None
        km = route.get("distance_km") if route else None
        if minutes is not None:
            status = "Within limit" if limit is None or minutes <= limit else "Above limit"
        elif km is not None:
            status = "Straight-line only"
        else:
            status = "No verified time"
        rows.append({"Rank": rank, "Town": town["town"], "Transactions": town["n_tx"], "Median": town["median"],
                     "P25": town["p25"], "P75": town["p75"], "Commute (min)": minutes,
                     "Example block": town["sample_blocks"][0] if town["sample_blocks"] else "",
                     "Commute status": status})
    frame = pd.DataFrame(rows)
    if not frame.empty:
        # Without routes every value is None, which pandas stores as objects; force a numeric (NaN) column.
        frame["Commute (min)"] = pd.to_numeric(frame["Commute (min)"], errors="coerce")
    return frame


def zero_result_actions(values, key, *, summary=True):
    """A search that matched nothing gets the same message everywhere and two ways forward."""
    profile = values.get("profile") or {}
    if summary:
        st.warning("No registered transactions matched these filters.")
        st.markdown(applied_filters(values), unsafe_allow_html=True)
    a, b, _ = st.columns([1, 1, 1.5])
    if profile.get("max_price") and a.button("Raise budget by 20%", key=f"raise-{key}", use_container_width=True):
        start_refinement("Raise the budget.", {"max_price": round(profile["max_price"] * 1.2, -3)})
    months = profile.get("months_back", 12)
    if months < LOOKBACK_MAX and b.button(f"Widen to {min(LOOKBACK_MAX, months * 2)} months", key=f"widen-{key}",
                                          use_container_width=True):
        start_refinement("Widen the date range.", {"months_back": min(LOOKBACK_MAX, months * 2)})


def market_tab(values):
    market = values.get("market") or {}
    if not market:
        st.info("No market search ran for this request.")
        return
    frame = town_frame(values)
    if frame.empty:
        zero_result_actions(values, "market")
        return
    shown = frame.head(15)
    colour = alt.Color("Commute status:N", scale=alt.Scale(domain=list(STATUS_COLOURS),
                                                           range=list(STATUS_COLOURS.values())),
                       legend=alt.Legend(orient="bottom", title=None))
    base = alt.Chart(shown).encode(y=alt.Y("Town:N", sort=list(shown["Town"]), title=None))
    band = base.mark_rule(strokeWidth=7, opacity=.35).encode(
        x=alt.X("P25:Q", title="Transacted price, middle half and median (S$)", axis=alt.Axis(format="$,.0f"),
                scale=alt.Scale(zero=False)), x2="P75:Q", color=colour)
    median = base.mark_point(filled=True, size=110).encode(
        x="Median:Q", color=colour,
        tooltip=["Town", "Transactions", alt.Tooltip("Median:Q", format="$,.0f"),
                 alt.Tooltip("Commute (min):Q", format=".1f"), "Example block"])
    st.markdown("**Towns in shortlist order** (commute first when OneMap data exists, then median price)")
    st.altair_chart((band + median).properties(height=max(220, 30 * len(shown))), use_container_width=True)
    timed = frame.dropna(subset=["Commute (min)"])
    if not timed.empty:
        st.markdown("**Price against commute**")
        scatter = alt.Chart(timed).mark_circle(opacity=.85).encode(
            x=alt.X("Commute (min):Q", title=f"Weekday 08:00 transit to {destination_label(values)} (min)"),
            y=alt.Y("Median:Q", title="Median price (S$)", axis=alt.Axis(format="$,.0f"), scale=alt.Scale(zero=False)),
            size=alt.Size("Transactions:Q", legend=None, scale=alt.Scale(range=[60, 900])),
            color=colour, tooltip=["Town", "Transactions", alt.Tooltip("Median:Q", format="$,.0f"),
                                   alt.Tooltip("Commute (min):Q", format=".1f")])
        layers = scatter
        limit = (values.get("profile") or {}).get("max_commute_min")
        if limit:
            layers = scatter + alt.Chart(pd.DataFrame({"x": [limit]})).mark_rule(strokeDash=[5, 4], color="#BE185D").encode(x="x:Q")
        st.altair_chart(layers.properties(height=320), use_container_width=True)
    table = frame.assign(**{col: frame[col].map(money) for col in ("Median", "P25", "P75")},
                         **{"Commute (min)": frame["Commute (min)"].round(1)})
    st.dataframe(table, hide_index=True, use_container_width=True)


GRANT_BADGES = {"may_apply": ("b-ok", "May apply"), "need_more_info": ("b-warn", "Need more info"),
                "not_eligible": ("b-no", "Not eligible")}
AGENT_BADGES = {"agreed": ("b-ok", "Policy agent agreed with rule check"),
                "disagreed": ("b-no", "Policy agent disagreed, conservative result shown"),
                "not_assessed": ("b-info", "Rule check only")}


def grant_card(grant) -> str:
    css, text = GRANT_BADGES[grant["status"]]
    acss, atext = AGENT_BADGES[grant.get("agent_check", "not_assessed")]
    amount = f" · indicative <b>{grant['amount_text']}</b>" if grant["amount_text"] else ""
    missing = (f"<div class='muted'>Still needed: {describe_missing(grant['missing_fields'])}</div>"
               if grant["missing_fields"] else "")
    explanation = f"<div>{escape(grant['explanation'])}</div>" if grant.get("explanation") else ""
    sources = " ".join(f"<a href='{url}'>Source {i}</a>" for i, url in enumerate(grant["citations"], 1))
    return (f"<div class='card'><b>{grant['name']}</b> <span class='badge {css}'>{text}</span>{amount}"
            f" <span class='badge {acss}'>{atext}</span><div class='muted'>{escape(grant['note'])}</div>"
            f"{explanation}{missing}<div class='muted'>{sources}</div></div>")


def grants_tab(values):
    grants = values.get("grants") or []
    if grants:
        st.markdown("**Screening in this report**")
        st.markdown("".join(grant_card(g) for g in grants), unsafe_allow_html=True)
    calculator(values)


def amount_of(text: str | None) -> float:
    return float(re.sub(r"[^\d.]", "", text)) if text else 0.0


def calculator(values):
    """Grant and loan calculator: the same deterministic rule engine the agents use, on the buyer's numbers."""
    profile = values.get("profile") or {}
    towns = [t for t, _ in ranked_towns(values)]
    example_price = towns[0]["median"] if towns else None
    st.markdown("### Grant and loan calculator")
    st.caption("Enter your household and the flat you are considering. The calculation uses the same dated HDB "
               "rules as the agents and shows each step. Illustrative only: HDB's HFE letter is the official assessment.")
    # A calculation belongs to the report it was made from; a new report starts from that report's inputs.
    report_id = hashlib.sha1((values.get("canonical_report") or "").encode()).hexdigest()[:10]
    key = f"calc-{st.session_state.thread}-{report_id}"
    with st.form("calculator"):
        st.markdown("**Household**")
        a, b, c = st.columns(3)
        citizenship_options = ["Unknown", *CITIZENSHIP]
        citizenship = a.selectbox("Citizenship", citizenship_options, format_func=lambda v: CITIZENSHIP.get(v, v),
                                  index=citizenship_options.index(profile["citizenship"]) if profile.get("citizenship") else 0)
        yes_no = ["Unknown", "Yes", "No"]

        def pick(column, label, field):
            current = profile.get(field)
            return column.selectbox(label, yes_no, index=0 if current is None else (1 if current else 2), key=f"c-{field}")
        first = pick(b, "All applicants first-timers?", "first_timer")
        income = c.number_input("Gross monthly household income (S$)", min_value=0.0, step=500.0,
                                value=profile.get("household_income_monthly"))
        d, e, f = st.columns(3)
        ages = d.text_input("Applicant ages (e.g. 30, 32)", value=", ".join(map(str, profile.get("buyer_ages") or [])))
        employment = pick(e, "Worked continuously for 12 months?", "continuous_employment_12m")
        near_options = ["Unknown", "live_with", "within_4km", "no"]
        near = f.selectbox("Living with or near parents/child", near_options,
                           index=near_options.index(profile["near_parents"]) if profile.get("near_parents") else 0,
                           format_func=lambda v: {"live_with": "Living together", "within_4km": "Within 4 km",
                                                  "no": "No"}.get(v, v))
        g, h, _ = st.columns(3)
        prerequisites = pick(g, "Other HDB prerequisites checked?", "prerequisites_confirmed")
        prior = pick(h, "Received PHG before?", "prior_phg")
        st.markdown("**The flat and the loan**")
        i, j, k = st.columns(3)
        flat = i.selectbox("Flat type", FLAT_TYPES, index=FLAT_TYPES.index(profile.get("flat_type") or "4 ROOM"))
        price = j.number_input("Purchase price (S$)", min_value=0.0, step=10_000.0,
                               value=profile.get("purchase_price") or example_price,
                               help="Prefilled with the median of the top town in your shortlist when you have not given one.")
        valuation = k.number_input("Valuation (S$, blank = same as price)", min_value=0.0, step=10_000.0,
                                   value=profile.get("valuation"))
        l, m, _ = st.columns(3)
        lease = l.number_input("Flat's remaining lease (years)", min_value=0.0, max_value=99.0, step=1.0,
                               value=profile.get("flat_remaining_lease"))
        debt = m.number_input("Existing monthly debt (S$)", min_value=0.0, step=100.0,
                              value=profile.get("existing_monthly_debt") or 0.0)
        calculate = st.form_submit_button("Calculate", type="primary")
    if calculate:
        as_bool = {"Yes": True, "No": False}
        inputs = {
            "citizenship": None if citizenship == "Unknown" else citizenship, "first_timer": as_bool.get(first),
            "household_income_monthly": income, "buyer_ages": [int(x) for x in re.findall(r"\d{2,3}", ages)],
            "continuous_employment_12m": as_bool.get(employment), "near_parents": None if near == "Unknown" else near,
            "prerequisites_confirmed": as_bool.get(prerequisites), "prior_phg": as_bool.get(prior),
            "flat_type": flat, "purchase_price": price or None, "valuation": valuation or price or None,
            "flat_remaining_lease": lease, "existing_monthly_debt": debt,
        }
        try:
            st.session_state[key] = BuyerProfile.model_validate({k: v for k, v in inputs.items() if v not in (None, [])})
        except ValueError:
            st.error("Check the ages (18 to 120) and the other values.")
    calc = st.session_state.get(key)
    if calc is not None:
        st.caption(f"Calculated for: price {money(calc.purchase_price or 0)}, income "
                   f"{money(calc.household_income_monthly or 0)}/month, ages {', '.join(map(str, calc.buyer_ages)) or 'n/a'}, "
                   f"remaining lease {calc.flat_remaining_lease or 0:g} years. Press Calculate after changing a value.")
    if calc is None and values.get("affordability"):
        # Show the agents' own inputs straight away; the buyer can then change them and recalculate.
        calc = BuyerProfile.model_validate(profile)
    if calc is None:
        return
    grants = evaluate_grants(calc)
    loan = estimate_affordability(calc)
    st.markdown("#### Grants")
    rows = []
    for grant in grants:
        why = grant.note if not grant.missing_fields else "Needs " + describe_missing(grant.missing_fields)
        rows.append({"Grant": grant.name, "Result": GRANT_BADGES[grant.status][1],
                     "Indicative amount": grant.amount_text or "–", "Why": why})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    total_grants = sum(amount_of(g.amount_text) for g in grants if g.status == "may_apply")
    st.markdown(f"Indicative grants that may apply: **{money(total_grants)}**")

    st.markdown("#### HDB loan")
    if loan.status != "illustrative":
        st.warning("Needed for the loan: " + describe_missing(loan.missing_fields) if loan.missing_fields
                   else " ".join(loan.checks))
    else:
        tenure_text = ", ".join(f"{name} = {value:g}" for name, value in loan.tenure_limits.items())
        steps = pd.DataFrame([
            {"Step": "Repayment period", "Working": f"shortest of {tenure_text}", "Result": f"{loan.tenure_years} years"},
            {"Step": "Loan-to-value limit", "Working": f"75% × lower of price {money(calc.purchase_price)} and valuation {money(calc.valuation)}",
             "Result": money(loan.ltv_limit)},
            {"Step": "Monthly payment budget", "Working": f"30% × income {money(calc.household_income_monthly)} − debt {money(calc.existing_monthly_debt)}",
             "Result": f"S${loan.payment_budget:,.2f}"},
            {"Step": "Payment-based limit", "Working": f"budget over {loan.tenure_years} years at {loan.assessment_rate:.1%} assessment rate",
             "Result": money(loan.payment_limit)},
            {"Step": "Loan ceiling", "Working": f"lower of the two limits ({loan.binding.lower()} binds)", "Result": money(loan.loan_ceiling)},
            {"Step": "Monthly instalment", "Working": f"ceiling over {loan.tenure_years} years at {loan.repayment_rate:.2%} HDB rate",
             "Result": f"S${loan.monthly_instalment:,.2f}"},
        ])
        st.dataframe(steps, hide_index=True, use_container_width=True)
        # The parts must add up to the price: grants first, then a loan no larger than what is left.
        split = funding_split(calc.purchase_price, total_grants, loan.loan_ceiling)
        grants_applied, loan_applied, balance = split["grants"], split["loan"], split["balance"]
        a, b, c, d = st.columns(4)
        a.metric("Purchase price", money(calc.purchase_price))
        b.metric("Grants applied", f"− {money(grants_applied)}")
        c.metric("HDB loan applied", f"− {money(loan_applied)}", f"ceiling {money(loan.loan_ceiling)}", delta_color="off")
        d.metric("From CPF savings and/or cash", money(balance))
        st.caption(f"{money(grants_applied)} + {money(loan_applied)} + {money(balance)} = {money(calc.purchase_price)}. "
                   "The loan applied is the smaller of the borrowing ceiling and the price left after grants. Grants are "
                   "credited to CPF and offset the price. Valuation differences, CPF usage limits and fees are not included.")
    keep = (st.checkbox("Remember my household details for future conversations", key=f"keep-{key}",
                        help="Stored only for your account; view or delete under My memory.")
            if memory_enabled() else False)
    if st.button("Use these details in the research", help="Re-runs the agents with these details in this conversation."):
        if keep and memory_enabled():
            remember_household(storage()[1], st.session_state.user, calc)
        updates = {k: v for k, v in calc.model_dump().items() if v not in (None, []) and k not in ("months_back",)}
        st.session_state.notice = ("Updated the research with your household and loan details. "
                                   "The loan and grant results are in the Answer and the Grants and loan tab.")
        # Pressing this button is the approval for a household/loan-only change; search changes still ask.
        start_refinement("Re-check the grants and the loan with my details.", updates, preapproved=True)


def trace_tab(values):
    a, b, c, d = st.columns(4)
    a.metric("Intent", (values.get("intent") or "n/a").replace("_", " "),
             f"confidence {values.get('confidence', 0):.2f}", delta_color="off")
    b.metric("Model calls", values.get("model_calls", 0))
    c.metric("Tokens", f"{values.get('token_used', 0):,}")
    d.metric("Critic rewrites", values.get("revision", 0))
    st.markdown("**Plan** " + " → ".join(step.title() for step in values.get("plan", [])))
    if values.get("rationale"):
        st.caption(f"Planner rationale: {values['rationale']}")
    st.markdown("**Workflow this turn**")
    trail = "".join(f'<span class="stage" style="background:{STAGE_COLOURS[NODES[n][0]]}">{n}</span>'
                    for n in st.session_state.trail if n in NODES)
    st.markdown(trail or "<span class='muted'>No nodes recorded.</span>", unsafe_allow_html=True)
    events = values.get("guard_events") or []
    st.markdown("**Guardrail events** " + (", ".join(f"`{e}`" for e in events) if events else "none"))
    if values.get("critic_issues"):
        st.warning("Critic issues: " + "; ".join(values["critic_issues"]))
    st.caption(f"Engine {values.get('model_status', 'offline')} · route transport "
               f"{values.get('route_transport') or 'n/a'} · freshness "
               f"{(values.get('freshness') or {}).get('status', 'n/a')}")
    st.caption(f"Memory: {len(values.get('memories') or [])} past searches recalled · "
               f"{len(values.get('history') or [])} turns in this conversation · "
               f"{'on' if (st.session_state.get('account') and st.session_state.get('use_memory', True)) else 'off (guest or switched off)'}")
    evidence = values.get("evidence") or []
    if evidence:
        st.markdown("**Policy evidence retrieved**")
        for chunk in evidence:
            st.markdown(f"- [{chunk['title']} · {chunk['section']}]({chunk['source_url']}) "
                        f"<span class='muted'>snapshot {chunk['snapshot_date']}</span>", unsafe_allow_html=True)
    if trace_url:
        st.markdown(f"Full span-level trace: [Arize Phoenix]({PHOENIX_URL}/projects)")


def headline(values) -> str:
    market = values.get("market") or {}
    profile = values.get("profile") or {}
    ranked = ranked_towns(values)
    if market.get("error"):
        return market["error"]
    if market and not market.get("towns"):
        return "No registered transactions matched these filters."
    if values.get("routes") and profile.get("max_commute_min") is not None:
        return commute_headline(ranked, profile["max_commute_min"])
    if market.get("towns"):
        cheapest = min(market["towns"], key=lambda t: t["median"])
        return (f"{len(market['towns'])} towns had matching transactions. The lowest median was "
                f"{cheapest['town'].title()} at {money(cheapest['median'])}.")
    return finance_headline(values) or "Your research brief is ready."


def finance_headline(values) -> str:
    """One line for the loan and grant results, shown in the Answer next to the market result."""
    parts = []
    loan = values.get("affordability") or {}
    if loan.get("status") == "illustrative":
        parts.append(f"Illustrative HDB loan ceiling {money(loan['loan_ceiling'])} over {loan['tenure_years']} years, "
                     f"about S${loan['monthly_instalment']:,.2f} a month.")
    elif loan.get("status") == "need_more_info":
        parts.append("For a loan estimate, add " + describe_missing(loan["missing_fields"]) + " in the Grants and loan tab.")
    elif loan.get("status") == "manual_review":
        parts.append("The loan needs HDB review: " + " ".join(loan.get("checks", [])))
    grants = values.get("grants") or []
    if grants:
        may = [g for g in grants if g["status"] == "may_apply"]
        parts.append(f"Grants: {len(may)} may apply" + (f" ({', '.join(g['name'] + ' ' + g['amount_text'] for g in may)})" if may else "")
                     + f", {sum(g['status'] == 'need_more_info' for g in grants)} need more information.")
    return " ".join(parts)


def follow_up(values):
    """Change this search: carry the approved filters forward and change only what the buyer asks."""
    profile = values.get("profile") or {}
    with st.expander("Change this search", expanded=False):
        with st.form("refine"):
            st.markdown("<span class='muted'>Your filters and household details carry forward. Blank or 0 removes "
                        "a filter.</span>", unsafe_allow_html=True)
            text = st.text_input("Follow-up", placeholder="e.g. What about Woodlands instead?",
                                 label_visibility="collapsed")
            a, b, c = st.columns(3)
            price = a.number_input("Budget (S$, 0 = none)", min_value=0, step=10_000,
                                   value=int(profile.get("max_price") or 0))
            months = b.number_input("Look back at sales (months)", 1, LOOKBACK_MAX,
                                    min(LOOKBACK_MAX, int(profile.get("months_back") or 12)), help=LOOKBACK_HELP)
            towns = c.multiselect("Towns (none = all towns)", TOWNS, default=profile.get("towns") or [],
                                  format_func=str.title)
            d, e = st.columns([2, 1])
            destination = d.text_input("Commute destination (blank = no commute)", value=profile.get("destination") or "")
            commute = e.number_input("Commute limit (min, 0 = none)", 0, 300, int(profile.get("max_commute_min") or 0))
            submitted = st.form_submit_button("Update search", type="primary")
        if profile.get("destination") or profile.get("max_commute_min"):
            if st.button("Clear commute", help="Removes the commute destination and limit, then asks you to confirm."):
                start_refinement("Remove the commute requirement.", {"destination": None, "max_commute_min": None})
    if submitted:
        wanted = {"max_price": float(price) or None, "max_commute_min": commute or None, "months_back": months,
                  "towns": towns, "destination": destination.strip() or None}
        updates = {k: v for k, v in wanted.items() if (profile.get(k) or None) != (v or None)}
        if not text.strip() and not updates:
            st.warning("Type a follow-up or change a filter.")
            return
        start_refinement(text.strip() or "Refine the previous search.", updates)


def next_step(values):
    """One action that fits the answer, placed directly under it."""
    profile = values.get("profile") or {}
    loan = values.get("affordability") or {}
    grants = values.get("grants") or []
    missing = sorted({f for g in grants for f in g["missing_fields"] if f in FACT_SPECS})
    towns = [t["town"] for t, _ in ranked_towns(values)]
    if loan:
        label = "Edit loan details" if loan.get("status") == "illustrative" else "Add loan details"
        keys = ["purchase_price", "valuation", "household_income_monthly", "buyer_ages", "existing_monthly_debt",
                "flat_remaining_lease"]
        brief, preapproved = "Update the loan details.", True
    elif missing:
        label, keys = f"Add missing grant details ({len(missing)})", missing
        brief, preapproved = "Re-check the grants with my details.", True
    elif len(towns) > 1:
        label, keys, brief, preapproved = "Compare towns", None, "Compare these towns.", False
    else:
        return
    with st.expander(label, expanded=False):
        with st.form("next-step"):
            if keys:
                chosen = fact_inputs(keys, profile, "next")
            else:
                chosen = {"towns": st.multiselect("Towns to compare", TOWNS, default=towns[:3], format_func=str.title)}
            go = st.form_submit_button(label, type="primary")
        if go:
            updates = {k: v for k, v in chosen.items() if v not in (None, []) and v != profile.get(k)}
            if keys is None and chosen["towns"]:
                updates = {"towns": chosen["towns"]}
            if not updates:
                st.warning("Change at least one value.")
                return
            try:
                BuyerProfile.model_validate(updates)
            except ValueError:
                st.error("One of the values is not valid. Check the ages (18 to 120) and the numbers.")
                return
            # Household and loan details count as approved here; a town change goes back to the plan gate.
            start_refinement(brief, updates, preapproved=preapproved)


def result_metrics(values):
    """Metrics that belong to this answer: market, loan or grants. Model usage lives in Agent trace."""
    loan = values.get("affordability") or {}
    grants = values.get("grants") or []
    frame = town_frame(values)
    if not frame.empty:
        timed = frame.dropna(subset=["Commute (min)"])
        a, b, c = st.columns(3)
        a.metric("Towns matched", len(frame))
        b.metric("Transactions", f"{int(frame['Transactions'].sum()):,}")
        if values.get("routes"):
            c.metric("Best verified commute", f"{timed['Commute (min)'].min():.1f} min" if not timed.empty else "n/a")
    if loan.get("status") == "illustrative":
        p = values.get("profile") or {}
        a, b, c = st.columns(3)
        a.metric("HDB loan ceiling", money(loan["loan_ceiling"]))
        b.metric("Monthly instalment", f"S${loan['monthly_instalment']:,.2f}")
        c.metric("Repayment period", f"{loan['tenure_years']} years")
        st.caption(f"Inputs used: price {money(p.get('purchase_price') or 0)}, valuation {money(p.get('valuation') or 0)}, "
                   f"income {money(p.get('household_income_monthly') or 0)}/month, ages "
                   f"{', '.join(map(str, p.get('buyer_ages') or []))}, existing debt {money(p.get('existing_monthly_debt') or 0)}"
                   f"/month, remaining lease {p.get('flat_remaining_lease') or 0:g} years.")
    if grants and frame.empty and loan.get("status") != "illustrative":
        a, b, c = st.columns(3)
        a.metric("Grants that may apply", sum(g["status"] == "may_apply" for g in grants))
        b.metric("Need more information", sum(g["status"] == "need_more_info" for g in grants))
        c.metric("Not eligible", sum(g["status"] == "not_eligible" for g in grants))


if gate is None and values and values.get("cancelled"):
    st.divider()
    st.markdown("<div class='panel' style='border-color:#BE185D'><h4>Search cancelled</h4>"
                "No tools ran and no report was produced.</div>", unsafe_allow_html=True)
    a, b, _ = st.columns([1, 1, 2])
    if a.button("Edit search", type="primary", use_container_width=True):
        st.session_state.trail = []
        queue(prepare_turn(st.session_state.last_brief, save=st.session_state.get("save_flag", False)))
        st.rerun()
    if b.button("Start again", use_container_width=True):
        fresh_start()
        st.rerun()
elif gate is None and values and values.get("canonical_report"):
    st.divider()
    if st.session_state.get("notice"):
        st.success(st.session_state.pop("notice"))
    finance = finance_headline(values)
    lead = headline(values)
    extra = f"<div style='margin-top:.4rem'>{escape(finance)}</div>" if finance and finance not in lead else ""
    st.markdown(f"<div class='panel' style='border-color:#0F766E'><h4>Answer</h4>"
                f"<div style='font-size:1.08rem'>{escape(lead)}</div>{extra}"
                f"<div style='margin-top:.6rem'>{applied_filters(values)}</div></div>", unsafe_allow_html=True)
    st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
    result_metrics(values)
    if values.get("market") and not (values["market"].get("towns") or values["market"].get("error")):
        zero_result_actions(values, "answer", summary=False)
    next_step(values)
    follow_up(values)
    if any(event.startswith("pii_") for event in values.get("guard_events", [])):
        st.info("Personal identifiers in the brief were redacted before any model or log saw them.")
    if values.get("saved_path"):
        st.success(f"Approved shortlist saved to `{Path(values['saved_path']).name}`.")
    report, market, grants, trace = st.tabs(["Report", "Market", "Grants and loan", "Agent trace"])
    with report:
        st.markdown(values["canonical_report"])
        st.download_button("Download report (Markdown)", values["canonical_report"],
                           file_name="hdb-research-brief.md", mime="text/markdown")
    with market:
        market_tab(values)
    with grants:
        grants_tab(values)
    with trace:
        trace_tab(values)
    history = values.get("history") or []
    if len(history) > 1:
        with st.expander(f"Earlier in this conversation ({len(history) - 1})", expanded=False):
            for turn in history[:-1][::-1]:
                st.markdown(f"**{escape(turn['brief'])}**  <span class='muted'>{turn['at'][:16].replace('T', ' ')}</span>",
                            unsafe_allow_html=True)
                st.caption(turn["answer"])
    handling_panel(values, None, title="How the assistant handled this request")
