"""Validated contracts shared by the graph and deterministic tools."""
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

TOWNS = (
    "ANG MO KIO", "BEDOK", "BISHAN", "BUKIT BATOK", "BUKIT MERAH", "BUKIT PANJANG", "BUKIT TIMAH",
    "CENTRAL AREA", "CHOA CHU KANG", "CLEMENTI", "GEYLANG", "HOUGANG", "JURONG EAST", "JURONG WEST",
    "KALLANG/WHAMPOA", "MARINE PARADE", "PASIR RIS", "PUNGGOL", "QUEENSTOWN", "SEMBAWANG",
    "SENGKANG", "SERANGOON", "TAMPINES", "TOA PAYOH", "WOODLANDS", "YISHUN",
)


# How far back a market search may read registered sales. The dataset starts in January 2017.
MAX_MONTHS_BACK = 120
FlatType = Literal["1 ROOM", "2 ROOM", "3 ROOM", "4 ROOM", "5 ROOM", "EXECUTIVE", "MULTI-GENERATION"]
Intent = Literal["research", "grants", "affordability", "explain", "out_of_scope"]
Step = Literal["market", "route", "grants", "afford", "report"]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class BuyerProfile(Contract):
    flat_type: FlatType | None = None
    max_price: float | None = Field(None, gt=0)
    min_est_lease_years: int | None = Field(None, ge=0, le=99)
    months_back: int = Field(12, ge=1, le=MAX_MONTHS_BACK)
    # Towns to search (HDB town names). Empty means all towns. A commute destination is a separate field.
    towns: list[str] = Field(default_factory=list, max_length=len(TOWNS))
    avoid_floor: str | None = Field(None, pattern=r"^\d{2} TO \d{2}$")
    # Where the buyer commutes to, as they wrote it ("NTU", "Raffles Place MRT", a postal code).
    destination: str | None = Field(None, min_length=2, max_length=80)

    @model_validator(mode="before")
    @classmethod
    def _single_town(cls, data):
        """Accept the earlier single `town` field (saved preferences, CLI edits) as a one-town list."""
        if isinstance(data, dict) and "town" in data:
            data = dict(data)
            town = data.pop("town")
            data.setdefault("towns", [town] if town else [])
        return data

    @field_validator("towns", mode="before")
    @classmethod
    def _normalise_towns(cls, value):
        """Accept any capitalisation ("Tampines", "tampines") and keep only real HDB towns, once each."""
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        names = []
        for item in value:
            name = " ".join(str(item).upper().split())
            if name not in TOWNS:
                raise ValueError(f"Unknown HDB town: {item}")
            if name not in names:
                names.append(name)
        return names
    # The OneMap-resolved key for that destination. Set by the resolver, never by a model.
    anchor_id: str | None = Field(None, pattern=r"^[A-Z0-9_]{1,40}$")
    max_commute_min: int | None = Field(None, gt=0, le=300)
    citizenship: Literal["SC_SC", "SC_SPR", "SC_single", "other"] | None = None
    first_timer: bool | None = None
    household_income_monthly: float | None = Field(None, ge=0)
    buyer_ages: list[Annotated[int, Field(ge=18, le=120)]] = Field(default_factory=list, max_length=10)
    continuous_employment_12m: bool | None = None
    near_parents: Literal["live_with", "within_4km", "no"] | None = None
    prerequisites_confirmed: bool | None = None
    prior_phg: bool | None = None
    existing_monthly_debt: float | None = Field(None, ge=0)
    purchase_price: float | None = Field(None, gt=0)
    valuation: float | None = Field(None, gt=0)
    flat_remaining_lease: float | None = Field(None, ge=0, le=99)


class Provenance(Contract):
    source_id: str
    url: str
    retrieved_at: str
    rows: int
    last_month: str


class TownSummary(Contract):
    town: str
    n_tx: int
    median: float
    p25: float
    p75: float
    latest_month: str
    sample_blocks: list[str]
    sample_estimated_leases: dict[str, float] = Field(default_factory=dict)


class MarketSummary(Contract):
    query: dict
    window: str = ""
    as_of: str
    towns: list[TownSummary] = Field(default_factory=list)
    source: Provenance | None = None
    error: str | None = None


class Address(Contract):
    address_key: str
    status: Literal["matched", "ambiguous", "geocode_failed", "pending"]
    lat: float | None = None
    lon: float | None = None
    postal: str | None = None
    matched_address: str | None = None


class Destination(Contract):
    query: str
    status: Literal["matched", "cached", "not_found", "offline_unknown"]
    anchor_id: str | None = None
    label: str | None = None
    address: str | None = None


class RouteResult(Contract):
    address_key: str
    anchor_id: str
    duration_min: float | None = None
    distance_km: float | None = None
    assumption: str
    status: Literal["ok", "cached", "no_route", "geocode_failed", "proxy", "over_limit"]


class RuleChunk(Contract):
    id: str
    title: str
    section: str
    source_url: str
    snapshot_date: str
    text: str


class GrantItem(Contract):
    name: str
    status: Literal["may_apply", "not_eligible", "need_more_info"]
    amount_text: str | None = None
    missing_fields: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    note: str = ""
    # Set when the Policy agent assessed this grant and the rule engine verified that assessment.
    agent_check: Literal["agreed", "disagreed", "not_assessed"] = "not_assessed"
    explanation: str = ""


class Affordability(Contract):
    status: Literal["illustrative", "need_more_info", "manual_review"]
    missing_fields: list[str] = Field(default_factory=list)
    loan_ceiling: float | None = None
    monthly_instalment: float | None = None
    tenure_years: int | None = None
    # The working behind the ceiling, so the buyer can see every step of the calculation.
    tenure_limits: dict[str, float] = Field(default_factory=dict)
    ltv_limit: float | None = None
    payment_budget: float | None = None
    payment_limit: float | None = None
    assessment_rate: float | None = None
    repayment_rate: float | None = None
    binding: str | None = None
    checks: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)


class AdvisorState(TypedDict, total=False):
    brief: str
    profile: dict
    intent: Intent
    plan: list[Step]
    rationale: str
    confidence: float
    market: dict
    routes: list[dict]
    grants: list[dict]
    evidence: list[dict]
    affordability: dict | None
    narrative: dict
    canonical_report: str
    critic_issues: list[str]
    guard_events: list[str]
    tool_steps: list[str]
    token_used: int
    freshness: dict
    revision: int
    clarified: bool
    cancelled: bool
    save_requested: bool
    saved_path: str | None
    model_status: str
    model_calls: int
    policy_calls: int
    policy_tokens: int
    policy_events: list[str]
    route_transport: str
    export_transport: str
    refine: bool
    updates: dict
    refinement: dict
    last_profile: dict
    last_plan: list[str]
    destination: dict
    approved_report: str
    reapprove: bool
    edit_issues: list[str]
    preapproved: bool
    history: list[dict]
    memories: list[dict]
    from_memory: list[str]
