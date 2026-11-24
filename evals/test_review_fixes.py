"""Regression tests for the 24 November functional review (one test per reported failure)."""
import sqlite3
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from evals.harness import config_for
from hdb_advisor.agents import Critique, Narrative, Pick, PlanOutput, RouterOutput, offline_intake
from hdb_advisor.graph import Settings, build_graph, prepare_turn
from hdb_advisor.market import ingest_csv
from hdb_advisor.memory import preferences, remember
from hdb_advisor.policy import evaluate_grants, funding_split
from hdb_advisor.report import narrative_issues
from hdb_advisor.state import BuyerProfile

DB = Path("data/hdb.db")
HEADER = ("month,town,flat_type,block,street_name,storey_range,floor_area_sqm,flat_model,"
          "lease_commence_date,remaining_lease,resale_price\n")


def drive(app, payload, config, decide):
    result, gates = app.invoke(payload, config), []
    while result.get("__interrupt__"):
        gate = result["__interrupt__"][0].value
        gates.append(gate)
        result = app.invoke(Command(resume=decide(gate)), config)
    return result, gates


def test_negated_living_arrangement_never_yields_a_proximity_grant():
    brief = ("First-timer SC/SC couple, ages 30 and 32, we do not live with our parents and they live 10 km away, "
             "4-room, income 6000, remaining lease 70, prerequisites confirmed, no prior PHG")
    profile = offline_intake(brief)
    assert profile.near_parents == "no"
    phg = next(g for g in evaluate_grants(profile) if g.name == "Proximity Housing Grant")
    assert phg.status == "not_eligible" and phg.amount_text is None
    assert offline_intake("We don't live with our parents, 4-room").near_parents is None  # asked, not assumed
    assert offline_intake("We will live with our parents").near_parents == "live_with"


def test_grant_requests_confirm_household_facts_before_screening():
    app = build_graph(Settings(DB), InMemorySaver())
    seen = []

    def decide(gate):
        seen.append(gate)
        return {"action": "edit", "changes": {"near_parents": "no"}} if gate["type"] == "confirm_plan" else {"action": "skip"}
    result, _ = drive(app, prepare_turn("SC/SC first-timers, we will live with our parents, check PHG"),
                      config_for("household"), decide)
    assert seen[0]["type"] == "confirm_plan" and seen[0]["household"]["near_parents"] == "live_with"
    assert next(g for g in result["grants"] if g["name"] == "Proximity Housing Grant")["status"] == "not_eligible"


def test_funding_parts_reconcile_to_the_price():
    split = funding_split(300000, 160000, 225000)
    assert split == {"grants": 160000, "loan": 140000, "balance": 0}
    assert sum(split.values()) == 300000
    assert funding_split(500000, 0, 360000) == {"grants": 0, "loan": 360000, "balance": 140000}


def market_state():
    town = lambda n, m: {"town": n, "n_tx": 5, "median": m, "p25": m - 20000, "p75": m + 20000, "latest_month": "2026-11",
                         "sample_blocks": [f"1 {n} RD"], "sample_estimated_leases": {f"1 {n} RD": 60.0}}
    return {"market": {"query": {}, "window": "x", "as_of": "x", "towns": [town("WOODLANDS", 538000), town("BEDOK", 525000)]},
            "routes": [], "profile": {"max_price": 600000}}


def test_a_genuine_number_attached_to_the_wrong_measure_is_rejected():
    state = market_state()
    assert narrative_issues({"summary": "Woodlands has a median of S$600,000."}, state)
    assert narrative_issues({"summary": "Bedok recorded a median of S$538,000."}, state)
    assert not narrative_issues({"summary": "Woodlands has a median of S$538,000, within your S$600,000 budget."}, state)
    assert narrative_issues({"summary": "Two towns matched.",
                             "shortlist": [{"town": "BEDOK", "reason": "Median S$538,000."}]}, state)


def test_approval_previews_the_exact_saved_report_and_edits_are_previewed_again(tmp_path):
    csv = tmp_path / "rows.csv"
    csv.write_text(HEADER + "2026-10,TAMPINES,4 ROOM,10,SAMPLE RD,04 TO 06,92,MODEL A,1990,63 years,500000\n")
    db = tmp_path / "hdb.db"
    ingest_csv(db, csv)

    class Model:
        def call(self, role, schema, context, examples):
            return {"intake": BuyerProfile(flat_type="4 ROOM", max_price=600000),
                    "router": RouterOutput(intent="research", confidence=1),
                    "planner": PlanOutput(plan=["market", "report"], rationale="Search."),
                    "writer": Narrative(summary="Tampines recorded a median of S$500,000.",
                                        shortlist=[Pick(town="TAMPINES", reason="One sale matched.")]),
                    "critic": Critique(acceptable=True)}[role], 1
    app = build_graph(Settings(db, model=Model(), export=tmp_path / "out"), InMemorySaver())
    previews = []

    def decide(gate):
        if gate["type"] == "confirm_plan":
            return {"action": "approve"}
        previews.append(gate["report"])
        if len(previews) == 1:
            return {"action": "edit", "summary": "Tampines had one matching sale at a median of S$500,000."}
        return {"action": "approve"}
    result, _ = drive(app, prepare_turn("4-room under 600k", save=True), config_for("preview"), decide)
    saved = __import__("json").loads(Path(result["saved_path"]).read_text())["report"]
    assert len(previews) == 2 and "## Towns to look up first" in previews[0]
    assert "one matching sale" in previews[1]
    assert saved == previews[1] == result["canonical_report"]


def test_combined_grant_and_loan_request_runs_both_without_a_budget():
    app = build_graph(Settings(DB), InMemorySaver())
    brief = ("Check grants and estimate my loan: purchase price 500000, valuation 480000, income 8000, "
             "ages 30 and 32, existing debt 0, remaining lease 70, first-timer SC/SC")
    result, _ = drive(app, prepare_turn(brief), config_for("combined"),
                      lambda g: {"action": "approve"} if g["type"] == "confirm_plan" else {"action": "skip"})
    assert {"grants", "afford"} <= set(result["tool_steps"])
    assert result["affordability"]["status"] == "illustrative" and result["grants"]


def test_years_are_months_and_the_supported_range_is_shared():
    class Model:
        def call(self, role, schema, context, examples):
            return {"intake": BuyerProfile(flat_type="4 ROOM", max_price=600000, months_back=12),
                    "router": RouterOutput(intent="research", confidence=1),
                    "planner": PlanOutput(plan=["market", "report"], rationale="Search."),
                    "writer": Narrative(), "critic": Critique(acceptable=True)}[role], 1
    app = build_graph(Settings(DB, model=Model()), InMemorySaver())
    result, gates = drive(app, prepare_turn("4-room under 600k in the last 2 years"), config_for("years"),
                          lambda g: {"action": "approve"})
    assert gates[0]["filters"]["months_back"] == 24 and result["market"]["query"]["months_back"] == 24
    assert offline_intake("4-room, last 120 months").months_back == 120
    assert offline_intake("4-room, last 30 years").months_back == 120


def test_cancelling_produces_no_report():
    app = build_graph(Settings(DB), InMemorySaver())
    result, _ = drive(app, prepare_turn("4-room under 600k"), config_for("cancel"), lambda g: {"action": "cancel"})
    assert result["cancelled"] and result["canonical_report"] == "" and not result["market"]


def test_clearing_every_preference_removes_the_stored_profile():
    store = InMemoryStore()
    remember(store, "u", BuyerProfile(town="YISHUN"))
    assert preferences(store, "u") == {"town": "YISHUN"}
    remember(store, "u", BuyerProfile())
    assert preferences(store, "u") == {}


def test_packaged_database_keeps_only_demo_destinations(tmp_path):
    from scripts.package_submission import scrub_private_destinations
    from hdb_advisor.onemap import set_anchor

    csv = tmp_path / "rows.csv"
    csv.write_text(HEADER + "2026-10,TAMPINES,4 ROOM,10,SAMPLE RD,04 TO 06,92,MODEL A,1990,63 years,500000\n")
    db = tmp_path / "hdb.db"
    ingest_csv(db, csv)
    set_anchor(db, 1.3462, 103.6820, anchor_id="NTU_NORTH_SPINE", label="NTU North Spine", aliases=("NTU",))
    set_anchor(db, 1.3000, 103.8000, anchor_id="DEST_999999", label="My Office Tower",
               address="1 PRIVATE ROAD SINGAPORE 999999", aliases=("my office",))
    scrub_private_destinations(db)
    with sqlite3.connect(db) as conn:
        keys = {row[0] for row in conn.execute("SELECT query_key FROM destination_cache")}
        anchors = {row[0] for row in conn.execute("SELECT anchor_id FROM anchors")}
    assert "MY OFFICE" not in keys and "DEST_999999" not in anchors and "NTU" in keys
    assert b"PRIVATE ROAD" not in db.read_bytes() and b"MY OFFICE" not in db.read_bytes()


def test_calculator_details_run_the_loan_without_a_second_plan_approval():
    """The calculator's "Use these details" button is the approval for a household/loan-only change."""
    app = build_graph(Settings(DB), InMemorySaver())
    config = config_for("calculator")
    approve = lambda g: {"action": "approve"} if g["type"] == "confirm_plan" else {"action": "skip"}
    first, _ = drive(app, prepare_turn("4-room under S$600k, commute to NTU, and check grants"), config, approve)
    details = {"citizenship": "SC_SC", "first_timer": True, "household_income_monthly": 7000.0, "buyer_ages": [30, 32],
               "purchase_price": 585000.0, "valuation": 585000.0, "flat_remaining_lease": 70.0,
               "existing_monthly_debt": 0.0}
    second, gates = drive(app, prepare_turn("Re-check the grants and the loan with my details.", refine=True,
                                            updates=details, preapproved=True), config, approve)
    assert gates == []
    assert second["affordability"]["status"] == "illustrative" and second["affordability"]["loan_ceiling"] == 438750
    assert "plan_preapproved_by_buyer" in second["guard_events"]
    # A search change sent the same way still stops for approval.
    _, gates = drive(app, prepare_turn("Refine.", refine=True, updates={"max_price": 500000.0}, preapproved=True),
                     config, approve)
    assert [g["type"] for g in gates] == ["confirm_plan"]
