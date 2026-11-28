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
from hdb_advisor.policy import (estimate_affordability, evaluate_grants, funding_split,
                                index_rules, rules)
from hdb_advisor.report import commute_headline, commute_split, describe_missing, ranked_towns
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
    "All-in-one": ("Find 4-room flats under S$600k within 45 minutes of Raffles Place MRT "
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
.block-container {padding-top: 3.2rem; max-width: 1240px;}
h1, h2, h3 {letter-spacing: -.01em;}
/* Narrower sidebar with even spacing. */
section[data-testid="stSidebar"] {width: 315px !important; min-width: 315px !important;}
/* Every bordered container is a card: white, rounded, a faint shadow. */
div[data-testid="stVerticalBlockBorderWrapper"] {background: #FFFFFF; border-radius: 16px !important;
    box-shadow: 0 1px 2px rgba(20, 30, 40, .04), 0 4px 16px rgba(20, 30, 40, .04);}
/* Top bar shared by every screen. */
.brand {display: flex; align-items: center; gap: .6rem; font-weight: 700; font-size: 1.1rem;}
.brand .logo {width: 1.45rem; height: 1.45rem; border-radius: .4rem; background: #0F766E; display: inline-block;}
.topline {border-bottom: 1px solid #E4E2DB; margin: .2rem 0 1.6rem 0;}
.hero {text-align: center; margin: 2.6rem 0 1.6rem 0;}
.hero h1 {font-size: 2.3rem; line-height: 1.15; margin: 0 0 .6rem 0; padding: 0;}
.hero p {font-size: 1.08rem; color: #475569; margin: 0;}
.card-title {font-size: 1.25rem; font-weight: 700; margin: 0 0 .6rem 0;}
.summary {font-size: .98rem; color: #334155;}
.foot-line {border-top: 1px solid #E4E2DB; margin: .4rem 0 .2rem 0;}
/* Larger tab labels with more room between them. */
[data-testid="stTab"] p {font-size: 1.05rem;}
[role="tablist"] {gap: 1.6rem;}
/* Header: title, one-line purpose and a three-step progress bar for the human-in-the-loop flow. */
.app-title {font-size: 2.1rem; font-weight: 700; margin: 0 0 .2rem 0; color: #1C2A33;}
.app-title.small {font-size: 1.3rem; margin: 0;}
.app-sub {color: #475569; font-size: 1rem; margin: 0 0 1.1rem 0;}
.stepper {display: flex; align-items: center; justify-content: flex-end; font-size: .95rem; color: #64748B; margin: 0;
          white-space: nowrap;}
.stepper .s {display: flex; align-items: center; gap: .35rem;}
.stepper .d {width: 1.15rem; height: 1.15rem; border-radius: 50%; border: 1.5px solid #CBD5E1; display: inline-grid;
             place-items: center; font-size: .8rem; font-weight: 700;}
.stepper .done .d {background: #0F766E; border-color: #0F766E; color: #FFFFFF;}
.stepper .now {color: #1C2A33; font-weight: 700;} .stepper .now .d {border-color: #0F766E; color: #0F766E;}
.stepper .bar {width: 2.2rem; height: 1.5px; background: #CBD5E1; margin: 0 .5rem;}
.stepper .bar.on {background: #0F766E;}
.conv-cur {border: 1px solid #D6D3CA; background: #FFFFFF; border-radius: 8px; padding: .4rem .7rem;
            font-size: .95rem; margin-bottom: .4rem;}
.sb-label {font-size: .9rem; color: #1C2A33; margin: .6rem 0 .25rem 0;}
.stage {display: inline-block; border-radius: 6px; padding: .05rem .45rem; font-size: .85rem; font-weight: 600;
        color: white; margin-right: .5rem; min-width: 4.6rem; text-align: center;}
.badge {display: inline-block; border-radius: 999px; padding: .1rem .6rem; font-size: .88rem; font-weight: 600;}
.b-ok {background: #CCFBF1; color: #115E59;} .b-warn {background: #FEF3C7; color: #92400E;}
.b-no {background: #FEE2E2; color: #991B1B;} .b-info {background: #EEF2F1; color: #475569;}
.gate {border: 1px solid #E3E1DA; border-left: 4px solid #0F766E; border-radius: 12px; padding: 1rem 1.2rem;
       background: #FFFFFF; margin: .2rem 0 1rem;}
.gate h3 {margin: 0 0 .3rem 0; font-size: 1.35rem;}
.card {border: 1px solid #E5E3DA; border-radius: 12px; padding: .9rem 1rem; background: #FFFFFF; margin-bottom: .7rem;}
.muted {color: #64748B; font-size: .95rem;}
.panel {border: 1px solid #E5E3DA; border-radius: 12px; background: #FFFFFF; padding: 1rem 1.1rem;}
.panel h4 {margin: 0 0 .4rem 0; font-size: 1rem; color: #1C2A33;}
.sect {font-size: 1rem; font-weight: 700; color: #1C2A33; margin: .9rem 0 .35rem 0;}
.matched {color: #0F766E; font-size: .95rem; margin-top: .2rem;}
.fact.intent {background: #E3F1EE;}
.fact {display: inline-block; background: #EEF2F1; color: #1C2A33; border-radius: 6px;
       padding: .12rem .5rem; margin: 0 .3rem .35rem 0; font-size: .95rem;}
.fact b {color: #475569; font-weight: 600; margin-right: .25rem;}
.unknown {display: inline-block; background: #FFFBEB; border: 1px dashed #F59E0B; color: #92400E; border-radius: 8px;
          padding: .2rem .55rem; margin: 0 .35rem .4rem 0; font-size: .95rem;}
.vsteps {list-style: none; margin: 0; padding: 0;}
.vsteps li {display: flex; gap: .7rem; align-items: flex-start; padding: .35rem 0; font-size: 1rem; margin: 0;}
.vsteps li small {display: block; color: #64748B; font-size: .88rem; line-height: 1.3;}
.vsteps .n {width: 1.45rem; height: 1.45rem; border-radius: 50%; background: #E3F1EE; color: #0F766E; font-weight: 700;
            font-size: .8rem; display: inline-grid; place-items: center; flex: none; margin-top: .1rem;}
.vsteps .done .n {background: #0F766E; color: #FFFFFF;}
.vsteps .off {color: #A0AEC0;} .vsteps .off .n {background: #F1F3F2; color: #A0AEC0;}
.vsteps .nodata .n {background: #FEF3C7; color: #92400E;}
.chain {display: flex; flex-wrap: wrap; align-items: center; gap: .35rem;}
.chain .c {border: 1px solid #0F766E; color: #0F766E; background: #F0FAF8; border-radius: 8px; padding: .18rem .6rem;
           font-weight: 600; font-size: .95rem; cursor: default;}
.chain .c.skip {border-style: dashed; border-color: #CBD5E1; color: #94A3B8; background: transparent; font-weight: 400;}
.chain .c.done {background: #0F766E; color: #FFFFFF;}
.chain .c.nodata {border-color: #D97706; color: #92400E; background: #FFFBEB;}
.chain .ar {color: #94A3B8;}
.agents {display: grid; grid-template-columns: repeat(auto-fit, minmax(118px, 1fr)); gap: .5rem;}
.agent {border: 1px solid #E5E3DA; border-radius: 12px; padding: .55rem .7rem; background: #FFFFFF;}
.agent .t {font-weight: 700; font-size: 1rem;} .agent .d {font-size: .88rem; color: #475569;}
.timeline {display: flex; gap: .4rem; flex-wrap: wrap;}
.tl {flex: 1 1 120px; border-radius: 10px; padding: .45rem .6rem; background: #F1F5F9; color: #64748B; font-size: .9rem;}
.tl .h {font-weight: 700; font-size: .92rem; margin-bottom: .15rem;}
.tl.on {color: #FFFFFF;}
.tl.wait {background: #F0FAF8; color: #0F766E; border: 2px dashed #0F766E;}
/* Answer card: one-line conclusion, the top towns, then the supporting numbers. */
.answer {padding: .2rem .1rem 0;}
.answer .lead {font-size: 1.75rem; font-weight: 700; line-height: 1.25; letter-spacing: -.015em; margin: 0 0 1.1rem 0;}
.answer .lead em {font-style: normal; color: #0F766E;}
.towns {display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: .6rem;}
.town {border: 1px solid #E3E1DA; border-radius: 12px; padding: .8rem 1rem; background: #FCFCFA;}
.town .r {display: inline-grid; place-items: center; width: 1.4rem; height: 1.4rem; border-radius: 50%;
          background: #E3F1EE; color: #0F766E; font-size: .85rem; font-weight: 700; margin-right: .45rem;}
.town .n {font-weight: 700; font-size: 1.08rem;}
.town .k {display: flex; gap: 1.4rem; margin-top: .45rem; font-size: .88rem; color: #475569; flex-wrap: wrap;}
.town .k b {display: block; font-size: 1.3rem; line-height: 1.2; color: #1C2A33;}
.kpis {display: flex; gap: 1.6rem; flex-wrap: wrap; margin-top: .8rem; padding-top: .6rem; border-top: 1px solid #E3E1DA;
       font-size: .95rem; color: #334155;}
.kpis b {color: #1C2A33; font-size: 1.08rem; margin-right: .2rem;}
/* Grant screening as rows rather than cards. */
.grants {border: 1px solid #E3E1DA; border-radius: 12px; background: #FFFFFF; padding: .2rem 1rem; margin-bottom: 1rem;}
.grow {display: grid; grid-template-columns: 1.25fr 1fr 2.6fr .55fr; gap: .8rem; padding: .7rem 0;
       border-bottom: 1px solid #EEECE6; font-size: 1rem; align-items: start;}
.grow:last-child {border-bottom: none;}
.grow.head {font-size: .95rem; font-weight: 700; color: #1C2A33; padding: .6rem 0 .3rem;}
.grow .chk {display: block; font-size: .88rem; color: #0F766E; margin-top: .2rem;}
.grow .chk.no {color: #B45309;} .grow .chk.na {color: #64748B;}
.grow .needs {color: #92400E; margin-top: .2rem; cursor: help; text-decoration: underline dotted;}
.grow .src a {display: block; white-space: nowrap; color: #0F766E; font-size: .92rem;}
@media (max-width: 760px) {.grow {grid-template-columns: 1fr 1fr;} .grow.head {display: none;}}
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
            "trail": [], "last_brief": EXAMPLES["All-in-one"]}
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
        name = st.text_input("Your name", placeholder="e.g. ken", help="No password in this prototype. Your name keys your conversations and memory on "
                             "this machine. As a guest, conversations last for this session only and nothing is "
                             "remembered.")
        if st.form_submit_button("Sign in", use_container_width=True):
            user = re.sub(r"[^a-z0-9_-]", "-", name.strip().lower())[:32].strip("-")
            if not user or redact(name)[1]:
                st.warning("Use a short name without personal identifiers.")
            else:
                store.put(("users", user), "profile", {"name": name.strip()[:40], "last_seen": now_sg().isoformat(timespec="minutes")})
                # Each account has its own conversations; signing in never resumes a guest's pending decision.
                fresh_start(profile_switched=True, user=user, account=name.strip()[:40])
                st.rerun()


def conversations_panel():
    """Short-term memory per conversation: list this user's conversations and resume one exactly where it was."""
    items = list_conversations(storage()[1], st.session_state.user)
    if not items:
        return
    st.markdown("<div class='sb-label'>Your conversations</div>", unsafe_allow_html=True)
    current = f"{st.session_state.user}/{st.session_state.thread}"
    for item in items[:8]:
        label = f"{item['title'][:42]}{'…' if len(item['title']) > 42 else ''} · {item['updated'][5:10]}"
        if item["thread"] == current:
            st.markdown(f"<div class='conv-cur'>{escape(label)}</div>", unsafe_allow_html=True)
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
    st.session_state.update(thread=uuid.uuid4().hex[:12], result=None, gate=None, trail=[], active_config=None, result_panel=None,
                            show_composer=False,
                            brief="", last_brief="", previous_brief=previous, **extra)


with st.sidebar:
    if st.button("New conversation", type="primary", use_container_width=True):
        fresh_start()
        st.rerun()
    if st.session_state.pop("profile_switched", False):
        st.info("Started a new conversation for this account.")
    has_key = bool(os.getenv("GOOGLE_API_KEY"))
    engine = st.segmented_control("Engine", ["Gemini (live)", "Offline"], default="Gemini (live)" if has_key else "Offline",
                                  required=True, width="stretch",
                                  help="Live mode uses Gemini for the six agents, OneMap for routes, and the data.gov.sg "
                                       "freshness check. Offline uses the rule-based agents.")
    live = engine.startswith("Gemini")
    if live and not has_key:
        st.error("Set GOOGLE_API_KEY in .env to use the live agents.")
        live = False
    st.markdown("<div class='sb-label'>Options</div>", unsafe_allow_html=True)
    mcp = st.toggle("Route and export through MCP", value=False,
                    help="Calls the route and save tools through a FastMCP server and LangChain's MCPAdapter.")
    save = st.toggle("Save the approved shortlist", value=False,
                     help="Adds an output approval gate. The JSON export is written only after you approve.")
    st.session_state.save_flag = save
    st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
    account_panel()
    conversations_panel()
    memory_panel()

    st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
    source = provenance(DB) if DB.is_file() else None
    trace_url = tracing()
    if not source:
        st.error("No transaction snapshot found. Run `python -m scripts.ingest_hdb`.")
    # Provenance and tracing stay one click away instead of filling the sidebar.
    with st.expander("Data and tracing"):
        if source:
            retrieved = datetime.fromisoformat(source.retrieved_at).astimezone(ZoneInfo("Asia/Singapore")).date()
            st.markdown(f"**{source.rows:,}** resale transactions, latest month **{source.last_month}**, "
                        f"retrieved {retrieved} from [data.gov.sg]({source.url})")
        st.markdown(f"Grant rules snapshot **{rules()['snapshot_date']}**, verified {rules().get('verified_on', 'n/a')}")
        if trace_url:
            st.markdown(f"[Open traces in Arize Phoenix]({PHOENIX_URL}/projects)")
        else:
            st.markdown("Phoenix tracing is off. Set PHOENIX_TRACING=true (on by default in Docker).")
        st.markdown(f"Thread `{st.session_state.thread}`")

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

def stepper(current: int) -> str:
    """Ask → Confirm plan → Results: where the buyer is in the human-in-the-loop flow."""
    parts = []
    for number, label in enumerate(["Ask", "Confirm plan", "Results"], 1):
        state = "done" if number < current else "now" if number == current else ""
        mark = "✓" if number < current else str(number)
        if number > 1:
            parts.append(f"<span class='bar{' on' if number <= current else ''}'></span>")
        parts.append(f"<div class='s {state}'><span class='d'>{mark}</span>{label}</div>")
    return "<div class='stepper'>" + "".join(parts) + "</div>"


_gate, _result = st.session_state.gate, st.session_state.result or {}
if _gate is not None:
    step = 3 if _gate["type"] == "approve_output" else 2
elif _result.get("canonical_report"):
    step = 4  # Finished: every step is ticked.
elif _result.get("cancelled"):
    step = 2
else:
    step = 1
def toggle_composer():
    st.session_state.show_composer = not st.session_state.get("show_composer", False)


finished = step == 4 or bool(_result.get("cancelled"))
if finished:
    brand_column, steps_column, new_column = st.columns([1.25, 1.65, .55], vertical_alignment="center")
    new_column.button("New search", on_click=toggle_composer, use_container_width=True,
                      type="primary" if st.session_state.get("show_composer") else "secondary")
else:
    brand_column, steps_column = st.columns([1.5, 1.35], vertical_alignment="center")
brand_column.markdown("<div class='brand'><span class='logo'></span>HDB Resale Research Advisor</div>",
                      unsafe_allow_html=True)
steps_column.markdown(stepper(step), unsafe_allow_html=True)
st.markdown("<div class='topline'></div>", unsafe_allow_html=True)

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
def composer_card():
    with st.container(border=True):
        brief = st.text_area("Research brief", key="brief", height=120, label_visibility="collapsed",
                             placeholder="e.g. 4-room under S$600k within 45 minutes of Raffles Place MRT, and what do you need for grants?")
        examples_column, run_column = st.columns([4, 1.15], vertical_alignment="center")
        with examples_column:
            st.pills("Try an example", list(EXAMPLES), key="example", on_change=use_example,
                     label_visibility="collapsed",
                     help="Examples only fill the box above. You can edit the text before running it.")
            if st.session_state.get("previous_brief") and not brief.strip():
                st.button("Reuse previous brief", on_click=reuse_previous_brief)
        if run_column.button("Run research", type="primary", use_container_width=True):
            if brief.strip():
                st.session_state.trail = []
                st.session_state.last_brief = brief
                st.session_state.show_composer = False
                queue(prepare_turn(brief, save=save))
                st.rerun()
            else:
                st.warning("Enter a research brief first.")


if gate is None and not has_result:
    _, middle, _ = st.columns([.45, 4, .45])
    with middle:
        st.markdown("<div class='hero'><h1>Describe the flat you want</h1><p>Past resale prices, commute times and "
                    "grant checks from one request. You approve the plan before anything runs.</p></div>",
                    unsafe_allow_html=True)
        composer_card()
elif gate is None and st.session_state.get("show_composer"):
    composer_card()


# ---------------------------------------------------------------------------------------------------
# How the assistant is handling the request: interpreted profile, plan, agents and workflow timeline.
# ---------------------------------------------------------------------------------------------------
def interpreted_request(values, *, destination=True) -> str:
    profile = values.get("profile") or {}
    facts = []
    for key, (label, fmt) in PROFILE_LABELS.items():
        value = profile.get(key)
        if value is None or value == []:
            continue
        facts.append(f"<span class='fact'><b>{escape(label)}</b>{escape(str(fmt(value)))}</span>")
    intent = (values.get("intent") or "unknown").replace("_", " ")
    intent_chip = (f"<span class='fact intent' title='Router confidence {values.get('confidence', 0):.2f}'>"
                   f"<b>Intent</b>{escape(intent)} · {values.get('confidence', 0):.2f}</span>")
    html = ("<div>" + intent_chip + ("".join(facts) or "<span class='fact'>No search filters found in the brief</span>")
            + "</div>")
    if destination:
        html += destination_note(values)
    from_memory = values.get("from_memory") or []
    if from_memory:
        names = ", ".join(PROFILE_LABELS[k][0].lower() for k in from_memory if k in PROFILE_LABELS)
        html += f"<div class='muted' style='margin-top:.4rem'>From your memory: {escape(names)}</div>"
    if values.get("memories"):
        recalled = "".join(f"<li>{escape(m['text'])}</li>" for m in values["memories"])
        html += (f"<div class='muted' style='margin-top:.4rem'>Using what I remember from earlier conversations:"
                 f"<ul style='margin:.2rem 0 0 1rem'>{recalled}</ul></div>")
    return html


def destination_note(values) -> str:
    """Show what OneMap matched for the buyer's destination, so a wrong match is visible before anything runs."""
    found = values.get("destination") or {}
    if not found.get("query"):
        return ""
    if found.get("anchor_id"):
        where = escape(place_label(found["address"])) if found.get("address") else ""
        return f"<div class='matched' title='Matched on OneMap: {where}'>✓ {escape(found['label'])}</div>"
    why = "not in the offline cache" if found.get("status") == "offline_unknown" else "not found on OneMap"
    return (f"<div class='unknown' style='margin-top:.4rem'>Destination \"{escape(found['query'])}\" {why}. "
            "Edit the destination, for example with a postal code or MRT station.</div>")


def destination_label(values) -> str:
    found = values.get("destination") or {}
    return found.get("label") or (values.get("profile") or {}).get("destination") or "your destination"


# Short "who does it" lines for the plan list; the full descriptions stay in PLAN_STEPS (shown on hover).
PLAN_SHORT = {"market": "SQLite resale transactions", "route": "OneMap transit", "grants": "Policy agent + rule engine",
              "afford": "HDB loan calculator", "report": "Writer, checked by the Critic"}


def plan_pipeline(values, gate) -> str:
    """The research plan as a numbered list; steps that are not needed stay visible but faded."""
    plan = values.get("plan") or []
    used = set(values.get("tool_steps") or [])
    finished = gate is None and bool(values.get("canonical_report"))
    items, number = [], 0
    for key, (title, detail) in PLAN_STEPS.items():
        if key not in plan:
            state, tag = "off", "Not needed"
        elif values.get("cancelled"):
            state, tag = "off", "Cancelled"
        elif not finished:
            state, tag = "", PLAN_SHORT[key]
        elif key == "report" or key in used:
            state, tag = "done", PLAN_SHORT[key]
        else:
            state, tag = "nodata", "No data returned"
        if state != "off":
            number += 1
        mark = "✓" if state == "done" else "–" if state == "off" else str(number)
        items.append(f"<li class='{state}' title='{escape(detail)}'><span class='n'>{mark}</span>"
                     f"<div>{title}<small>{escape(tag)}</small></div></li>")
    return "<ol class='vsteps'>" + "".join(items) + "</ol>"


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


def handling_summary(values, gate, *, destination=True) -> str:
    return (f"<div class='sect'>Understood</div>{interpreted_request(values, destination=destination)}"
            f"<div class='sect'>Plan</div>{plan_pipeline(values, gate)}")


def workflow_details(values, gate):
    # Technical detail stays one click away; the buyer's decision comes first.
    with st.expander("Agents and workflow progress", expanded=False):
        st.markdown(agent_roster(values), unsafe_allow_html=True)
        st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
        st.markdown(timeline(gate), unsafe_allow_html=True)


def handling_panel(values, gate, *, title="How the assistant is handling your request"):
    with st.container(border=True):
        st.markdown(f"<div style='font-weight:700'>{escape(title)}</div>" + handling_summary(values, gate),
                    unsafe_allow_html=True)
    workflow_details(values, gate)


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


def applied_filters(values, *, plain=False) -> str:
    if not values.get("market"):
        return ""  # Search filters only describe a market search; a loan or grant answer has none.
    profile = values.get("profile") or {}
    keys = ["flat_type", "max_price", "months_back", "min_est_lease_years", "towns", "destination", "max_commute_min"]
    if plain:
        return " · ".join(escape(f"{PROFILE_LABELS[k][0]} {PROFILE_LABELS[k][1](profile[k])}")
                          for k in keys if profile.get(k) not in (None, []))
    chips = [f"<span class='fact'><b>{escape(PROFILE_LABELS[k][0])}</b>{escape(str(PROFILE_LABELS[k][1](profile[k])))}</span>"
             for k in keys if profile.get(k) not in (None, [])]
    return "".join(chips)


def start_refinement(brief: str, updates: dict | None = None, *, preapproved: bool = False):
    """Continue this conversation, carrying the approved filters and household facts forward."""
    st.session_state.trail = []
    st.session_state.result_panel = None  # Close the follow-up form under the answer.
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


def fact_inputs(keys, values, prefix, per_row=3) -> dict:
    """Render inputs for household, flat and loan facts, a few per row. Unknown stays None."""
    chosen = {}
    columns = st.columns(per_row)
    for index, key in enumerate(keys):
        label, kind, *extra = FACT_SPECS[key]
        column = columns[index % per_row]
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
    searching, screening = "market" in gate["plan"], "grants" in gate["plan"]
    action = ("Run search and check grants" if searching and screening else
              "Check grants" if screening else "Run with these filters")
    values = st.session_state.result or {}
    sentence = refinement_sentence(values)
    current = gate["filters"]
    household = gate.get("household") or {}
    # Left: what was understood and the plan. Right: the values the run will use, then run.
    left, right = st.columns([1, 1.85], gap="medium")
    with left.container(border=True):
        st.markdown("<div class='card-title'>Your request</div>"
                    + (handling_summary(values, gate, destination=not searching) if values else ""),
                    unsafe_allow_html=True)
    with right.container(border=True):
        follow = f"<div style='margin:-.2rem 0 .6rem'>{escape(sentence)}</div>" if sentence else ""
        st.markdown(f"<div class='card-title'>Confirm the plan</div>{follow}", unsafe_allow_html=True)
        with st.form("plan", border=False):
            wanted = {}
            if searching:
                wanted = search_filter_inputs(current)
            if screening:
                facts = ["citizenship", "first_timer", "near_parents", "household_income_monthly", "buyer_ages",
                         "flat_remaining_lease", "continuous_employment_12m", "prerequisites_confirmed", "prior_phg"]
                if not searching:
                    facts.insert(5, "flat_type")
                shown_facts = {**household, **current}
                given = sum(shown_facts.get(key) not in (None, []) for key in facts)
                # Collapsed by default to keep the plan short; the fields and what they submit are unchanged.
                with st.expander(f"Household details for grant screening · {given} of {len(facts)} given",
                                 expanded=not searching):
                    wanted |= fact_inputs(facts, shown_facts, "plan", per_row=2)
            remember = keep_household = False
            if memory_enabled():
                remember = (st.checkbox("Remember my towns, floor and commute destination for future conversations")
                            if searching else False)
                keep_household = (st.checkbox("Remember these household facts for future conversations",
                                              help="Stored only for your account. You can view or delete them under My memory.")
                                  if screening else False)
            elif st.session_state.get("account"):
                st.caption("Memory is off: nothing from this plan will be remembered.")
            # Confirm-dialog order: Cancel, then the primary action at the far right.
            _, cancel, run = st.columns([1.1, .8, 1.7])
            cancelled = cancel.form_submit_button("Cancel", use_container_width=True)
            submitted = run.form_submit_button(action, type="primary", use_container_width=True)
    if values:
        workflow_details(values, gate)
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
    a, b = st.columns(2)
    flat = a.selectbox("Flat type", ["Any", *FLAT_TYPES],
                       index=(FLAT_TYPES.index(current["flat_type"]) + 1) if current.get("flat_type") else 0)
    price = b.number_input("Max price (S$)", min_value=0, step=10_000, value=int(current.get("max_price") or 0),
                           help="0 = no limit.")
    c, d = st.columns(2)
    destination = c.text_input("Commute to", value=current.get("destination") or "",
                               help="Address, postal code, building or MRT station. Blank = no commute.")
    c.markdown(destination_note(st.session_state.result or {}), unsafe_allow_html=True)
    commute = d.number_input("Commute limit (min)", 0, 300, int(current.get("max_commute_min") or 0),
                             help="0 = no limit.")
    e, f = st.columns(2)
    months = e.slider("Sales from the last … months", 1, LOOKBACK_MAX,
                      min(LOOKBACK_MAX, int(current.get("months_back", 12))),
                      help=LOOKBACK_HELP)
    lease = f.number_input("Min estimated lease (years)", 0, 99, int(current.get("min_est_lease_years") or 0),
                           help="0 = no minimum.")
    towns = st.multiselect("Towns", TOWNS, default=[t for t in current.get("towns", []) if t in TOWNS],
                           format_func=str.title, placeholder="All towns", help="Leave empty to search every town.")
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
    if values and gate["type"] != "confirm_plan":
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


def readable(chart):
    """Axis and legend text at a size people can read on screen and in a recording."""
    return (chart.configure_axis(labelFontSize=14, titleFontSize=15, titleFontWeight="normal")
            .configure_axisY(labelOverlap=False, labelLimit=220)  # Keep every town name, never thin them out.
            .configure_legend(labelFontSize=14, symbolSize=120))


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
    st.markdown("**Price range by town**", help="Middle half and median of transacted prices, in shortlist order: "
                "commute first when OneMap data exists, then median price.")
    st.altair_chart(readable((band + median).properties(height=max(260, 34 * len(shown)))), use_container_width=True)
    timed = frame.dropna(subset=["Commute (min)"])
    if not timed.empty:
        st.markdown("**Price against commute**", help="Bubble size is the number of transactions.")
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
        st.altair_chart(readable(layers.properties(height=360)), use_container_width=True)
    table = frame.assign(**{col: frame[col].map(money) for col in ("Median", "P25", "P75")},
                         **{"Commute (min)": frame["Commute (min)"].round(1)})
    st.markdown("**All matching towns**")
    st.dataframe(table, hide_index=True, use_container_width=True, height=min(38 + 35 * len(table), 390))


GRANT_BADGES = {"may_apply": ("b-ok", "May apply"), "need_more_info": ("b-warn", "Need more info"),
                "not_eligible": ("b-no", "Not eligible")}
AGENT_BADGES = {"agreed": ("b-ok", "Policy agent agreed with rule check"),
                "disagreed": ("b-no", "Policy agent disagreed, conservative result shown"),
                "not_assessed": ("b-info", "Rule check only")}


AGENT_MARKS = {"agreed": ("", "✓ Agent agrees"), "disagreed": ("no", "Agent disagreed"), "not_assessed": ("na", "")}


def grant_row(grant) -> str:
    css, text = GRANT_BADGES[grant["status"]]
    check = grant.get("agent_check", "not_assessed")
    mark_css, mark = AGENT_MARKS[check]
    amount = f"<div class='muted'>indicative <b>{escape(grant['amount_text'])}</b></div>" if grant["amount_text"] else ""
    why = escape(grant.get("explanation") or grant["note"])
    # The rule engine's note is shown too when it adds something beyond the agent's explanation.
    note = (f"<div class='muted'>{escape(grant['note'])}</div>"
            if grant.get("explanation") and grant["note"] and not grant["missing_fields"] else "")
    # A count in the row; the full list of missing details shows on hover.
    count = len(grant["missing_fields"])
    missing = (f"<div class='needs' title='Still needed: {escape(describe_missing(grant['missing_fields']))}'>"
               f"{count} detail{'s' if count != 1 else ''} missing</div>" if count else "")
    sources = "".join(f"<a href='{url}' title='{escape(url)}'>Source {i}</a>" for i, url in enumerate(grant["citations"], 1))
    return (f"<div class='grow'><div><b>{escape(grant['name'])}</b>{amount}</div>"
            f"<div><span class='badge {css}'>{text}</span><span class='chk {mark_css}' "
            f"title='{escape(AGENT_BADGES[check][1])}'>{mark}</span></div>"
            f"<div>{why}{note}{missing}</div><div class='src'>{sources}</div></div>")


def grants_tab(values):
    grants = values.get("grants") or []
    if grants:
        head = "<div class='grow head'><div>Grant</div><div>Result</div><div>Why</div><div>Sources</div></div>"
        st.markdown("<div class='grants'>" + head + "".join(grant_row(g) for g in grants) + "</div>",
                    unsafe_allow_html=True)
    calculator(values)


def amount_of(text: str | None) -> float:
    return float(re.sub(r"[^\d.]", "", text)) if text else 0.0


def calculator(values):
    """Grant and loan calculator: the same deterministic rule engine the agents use, on the buyer's numbers."""
    profile = values.get("profile") or {}
    towns = [t for t, _ in ranked_towns(values)]
    example_price = towns[0]["median"] if towns else None
    # A calculation belongs to the report it was made from; a new report starts from that report's inputs.
    report_id = hashlib.sha1((values.get("canonical_report") or "").encode()).hexdigest()[:10]
    key = f"calc-{st.session_state.thread}-{report_id}"
    calculator_box = st.expander("Calculate with your own numbers", expanded=False)
    with calculator_box.form("calculator", border=False):
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
        d.metric("From CPF savings and/or cash", money(balance),
                 help=f"{money(grants_applied)} + {money(loan_applied)} + {money(balance)} = {money(calc.purchase_price)}. "
                      "The loan applied is the smaller of the borrowing ceiling and the price left after grants. Grants are "
                      "credited to CPF and offset the price. Valuation differences, CPF usage limits and fees are not "
                      "included. Illustrative only: HDB's HFE letter is the official assessment.")
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
    handling_panel(values, None, title="How the assistant handled this request")
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
    with st.container(border=True):
        with st.form("refine", border=False):
            text = st.text_input("Follow-up", placeholder="e.g. What about Woodlands instead?",
                                 label_visibility="collapsed")
            a, b, c = st.columns(3)
            price = a.number_input("Budget (S$)", min_value=0, step=10_000,
                                   value=int(profile.get("max_price") or 0), help="0 = no limit.")
            months = b.number_input("Look back at sales (months)", 1, LOOKBACK_MAX,
                                    min(LOOKBACK_MAX, int(profile.get("months_back") or 12)), help=LOOKBACK_HELP)
            towns = c.multiselect("Towns", TOWNS, default=profile.get("towns") or [],
                                  format_func=str.title, placeholder="All towns")
            d, e = st.columns([2, 1])
            destination = d.text_input("Commute to", value=profile.get("destination") or "",
                                       help="Blank = no commute.")
            commute = e.number_input("Commute limit (min)", 0, 300, int(profile.get("max_commute_min") or 0),
                                     help="0 = no limit.")
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


def next_step_spec(values):
    """The one action that fits the answer: its label, the fields it asks for, its brief and approval."""
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
        return None
    return label, keys, brief, preapproved, towns


def next_step(values, spec):
    """The form for that action, shown under the answer when its button is pressed."""
    profile = values.get("profile") or {}
    label, keys, brief, preapproved, towns = spec
    with st.container(border=True):
        with st.form("next-step", border=False):
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


def answer_card(values) -> tuple[str, str]:
    """Lead with the conclusion and the top towns; the summary line goes in the card's footer. Same figures as before."""
    profile = values.get("profile") or {}
    ranked = ranked_towns(values)
    limit = profile.get("max_commute_min")
    lead, top = escape(headline(values)), ranked[:3]
    if values.get("routes") and limit is not None and ranked:
        fits, above, _ = commute_split(ranked, limit)
        checked = len(fits) + len(above)
        if checked:
            lead = (f"<em>{len(fits)} of {checked}</em> verified {'route' if checked == 1 else 'routes'} reach "
                    f"{escape(destination_label(values))} within {limit} minutes")
            top = fits[:3] or ranked[:3]
    cards = []
    for town, route in top:
        rank = next(i for i, (t, _) in enumerate(ranked, 1) if t is town)
        minutes = route.get("duration_min") if route else None
        commute = f"<span><b>{minutes:.1f} min</b>commute</span>" if minutes is not None else ""
        block = f"Sample block {town['sample_blocks'][0].title()} · " if town["sample_blocks"] else ""
        detail = f"{block}{town['n_tx']:,} sale{'s' if town['n_tx'] != 1 else ''}"
        cards.append(f"<div class='town' title='{escape(detail)}'><div class='n'><span class='r'>{rank}</span>"
                     f"{escape(town['town'].title())}</div>"
                     f"<div class='k'>{commute}<span><b>{money(town['median'])}</b>median</span></div></div>")
    parts = []
    frame = town_frame(values)
    if not frame.empty:
        parts.append(f"<b>{len(frame)}</b> towns · <b>{int(frame['Transactions'].sum()):,}</b> transactions")
    finance = finance_headline(values)
    if finance and finance not in headline(values):
        parts.append(escape(finance))
    body = (f"<div class='answer'><div class='lead'>{lead}</div>"
            + (f"<div class='towns'>{''.join(cards)}</div>" if cards else "") + "</div>")
    return body, "<div class='summary'>" + " · ".join(parts) + "</div>" if parts else ""


def result_metrics(values):
    """Metrics that belong to this answer: market, loan or grants. Model usage lives in Agent trace."""
    loan = values.get("affordability") or {}
    grants = values.get("grants") or []
    frame = town_frame(values)
    if loan.get("status") == "illustrative":
        p = values.get("profile") or {}
        a, b, c = st.columns(3)
        inputs = (f"Inputs used: price {money(p.get('purchase_price') or 0)}, valuation {money(p.get('valuation') or 0)}, "
                  f"income {money(p.get('household_income_monthly') or 0)}/month, ages "
                  f"{', '.join(map(str, p.get('buyer_ages') or []))}, existing debt {money(p.get('existing_monthly_debt') or 0)}"
                  f"/month, remaining lease {p.get('flat_remaining_lease') or 0:g} years.")
        a.metric("HDB loan ceiling", money(loan["loan_ceiling"]), help=inputs)
        b.metric("Monthly instalment", f"S${loan['monthly_instalment']:,.2f}")
        c.metric("Repayment period", f"{loan['tenure_years']} years")
    if grants and frame.empty and loan.get("status") != "illustrative":
        a, b, c = st.columns(3)
        a.metric("Grants that may apply", sum(g["status"] == "may_apply" for g in grants))
        b.metric("Need more information", sum(g["status"] == "need_more_info" for g in grants))
        c.metric("Not eligible", sum(g["status"] == "not_eligible" for g in grants))


if gate is None and values and values.get("cancelled"):
    st.markdown("<div class='panel'><h4>Search cancelled</h4>"
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
    if st.session_state.get("notice"):
        st.success(st.session_state.pop("notice"))
    spec = next_step_spec(values)
    panel = st.session_state.get("result_panel")

    def toggle(name):
        st.session_state.result_panel = None if st.session_state.get("result_panel") == name else name

    body, summary = answer_card(values)
    # One card: the answer, then a footer with the summary on the left and its two actions on the right.
    with st.container(border=True):
        st.markdown(body, unsafe_allow_html=True)
        result_metrics(values)
        if values.get("market") and not (values["market"].get("towns") or values["market"].get("error")):
            zero_result_actions(values, "answer", summary=False)
        st.markdown("<div class='foot-line'></div>", unsafe_allow_html=True)
        summary_column, first, second = st.columns([2.2, 1.45, 1.05], vertical_alignment="center")
        summary_column.markdown(summary, unsafe_allow_html=True)
        if spec:
            first.button(spec[0], key="open-next", on_click=toggle, args=("next",), use_container_width=True,
                         type="primary" if panel == "next" else "secondary")
        current_search = applied_filters(values, plain=True)
        (second if spec else first).button("Change this search", key="open-refine", on_click=toggle, args=("refine",),
                                           use_container_width=True,
                                           type="primary" if panel == "refine" else "secondary",
                                           help=f"Current search: {current_search}" if current_search else None)
    if panel == "next" and spec:
        next_step(values, spec)
    elif panel == "refine":
        follow_up(values)
    if any(event.startswith("pii_") for event in values.get("guard_events", [])):
        st.info("Personal identifiers in the brief were redacted before any model or log saw them.")
    if values.get("saved_path"):
        st.success(f"Approved shortlist saved to `{Path(values['saved_path']).name}`.")
    def report_tab(values):
        st.download_button("Download report (Markdown)", values["canonical_report"],
                           file_name="hdb-research-brief.md", mime="text/markdown")
        with st.container(height=640, border=True):
            st.markdown(values["canonical_report"])

    # Charts first when a market search ran; otherwise the grant and loan results lead.
    order = (["Market", "Grants and loan", "Report", "Agent trace"] if values.get("market")
             else ["Grants and loan", "Report", "Market", "Agent trace"])
    render = {"Market": market_tab, "Grants and loan": grants_tab, "Report": report_tab, "Agent trace": trace_tab}
    for name, tab in zip(order, st.tabs(order)):
        with tab:
            render[name](values)
    history = values.get("history") or []
    if len(history) > 1:
        with st.expander(f"Earlier in this conversation ({len(history) - 1})", expanded=False):
            for turn in history[:-1][::-1]:
                st.markdown(f"**{escape(turn['brief'])}**  <span class='muted'>{turn['at'][:16].replace('T', ' ')}</span>",
                            unsafe_allow_html=True)
                st.caption(turn["answer"])
