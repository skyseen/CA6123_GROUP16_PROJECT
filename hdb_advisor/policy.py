"""Dated policy evidence and conservative, deterministic screening."""
import hashlib
import json
import math
import re
from functools import lru_cache
from pathlib import Path
from statistics import mean

import yaml
from langchain_core.embeddings import Embeddings

from .state import Affordability, BuyerProfile, GrantItem, RuleChunk

ROOT = Path(__file__).resolve().parent.parent
RULE_FILE = ROOT / "config/rules_2026-11-23.yaml"
HFE = "https://www.hdb.gov.sg/buying-a-flat/flat-grant-and-loan-eligibility/application-for-an-hdb-flat-eligibility-hfe-letter"


@lru_cache
def rules() -> dict:
    with RULE_FILE.open() as stream:
        return yaml.safe_load(stream)


def threshold(key: str) -> float:
    return rules()["thresholds"][key]["value"]


def sources(*keys: str) -> list[str]:
    return [rules()["sources"][key] for key in keys]


def curated_corpus() -> list[RuleChunk]:
    return [RuleChunk(id=key, title=title, section=section, source_url=sources(source)[0],
                      snapshot_date=rules()["snapshot_date"], text=body)
            for key, source, title, section, body in rules()["sections"]]


WORDS = re.compile(r"[a-z0-9]+")
STOPWORDS = set("what who how for can with and the does when which is a to of in on i my".split())


def words(text: str) -> set[str]:
    return {x.removesuffix("s") if len(x) > 4 else x for x in WORDS.findall(text.lower())
            if x not in STOPWORDS}


class LocalEmbeddings(Embeddings):
    """Hash lexical features into vectors so Chroma can run offline."""
    def embed_query(self, text: str) -> list[float]:
        vector = [0.0] * 2048
        for word in words(text):
            slot = int.from_bytes(hashlib.blake2b(word.encode(), digest_size=4).digest()) % len(vector)
            vector[slot] += 1.0
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]


def index_rules(directory: Path, chunks: list[RuleChunk] | None = None, *, rebuild: bool = False):
    from langchain_chroma import Chroma

    chunks = chunks if chunks is not None else curated_corpus()
    directory.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256(json.dumps([c.model_dump() for c in chunks], sort_keys=True).encode()).hexdigest()
    store = Chroma(collection_name=f"hdb-rules-{fingerprint[:20]}", persist_directory=str(directory),
                   embedding_function=LocalEmbeddings(), collection_metadata={"hnsw:space": "cosine"})
    if rebuild or not store.get(limit=1)["ids"]:
        store.add_texts([f"{c.title} {c.section} {c.text}" for c in chunks],
                        metadatas=[{"chunk": c.model_dump_json()} for c in chunks],
                        ids=[c.id for c in chunks])
    return store


def retrieve_rules(query: str, k: int = 4, *, store=None, chunks=None) -> list[RuleChunk]:
    if not query.strip() or k < 1:
        return []
    if store is not None:
        return [RuleChunk.model_validate_json(doc.metadata["chunk"]) for doc in store.similarity_search(query, k=k)]
    tokens = words(query)
    ranking = []
    for chunk in chunks if chunks is not None else curated_corpus():
        headline = words(f"{chunk.title} {chunk.section}")
        relevance = 2 * len(tokens & headline) + len(tokens & words(chunk.text))
        if relevance:
            ranking.append((relevance, chunk))
    ranking.sort(key=lambda item: (-item[0], item[1].id))
    return [chunk for _, chunk in ranking[:k]]


def missing(profile: BuyerProfile, fields: tuple[str, ...]) -> list[str]:
    return [key for key in fields if getattr(profile, key) is None or getattr(profile, key) == []]


def grant_result(name, citations, *, asks=None, ineligible=None, manual=None, amount=None):
    if ineligible:
        return GrantItem(name=name, status="not_eligible", citations=citations, note=ineligible)
    if asks or manual:
        return GrantItem(name=name, status="need_more_info", citations=citations, missing_fields=asks or [],
                         note=manual or "These details are needed to screen the grant.")
    return GrantItem(name=name, status="may_apply", citations=citations, amount_text=amount,
                     note="Indicative screening based on the stated household and proposed flat.")


def money(amount: float) -> str:
    return f"S${amount:,.0f}"


def evaluate_grants(profile: BuyerProfile) -> list[GrantItem]:
    p = profile
    scope = None
    if p.citizenship not in (None, "SC_SC", "SC_SPR") or p.first_timer is False:
        scope = "Check the relevant HDB scheme for singles, mixed timer households, or other household types."
    family_fields = ("citizenship", "first_timer", "household_income_monthly", "buyer_ages",
                     "flat_type", "flat_remaining_lease", "prerequisites_confirmed")
    required = missing(p, family_fields)
    family_fail = None
    if p.flat_type == "1 ROOM" or (p.buyer_ages and min(p.buyer_ages) < threshold("family_min_age")):
        family_fail = "The flat size or applicant age is outside this first-timer family screening."
    if p.household_income_monthly is not None and p.household_income_monthly > threshold("family_income"):
        scope = "Income exceeds the ordinary family ceiling. Extended and multigeneration cases need HDB review."
    family_amount = None
    if not (required or scope or family_fail) and p.prerequisites_confirmed and p.flat_remaining_lease > threshold("minimum_lease"):
        family_amount = money(rules()["amounts"]["family"][p.citizenship][int(p.flat_type not in {"2 ROOM", "3 ROOM", "4 ROOM"})])
    family = grant_result("Family Grant", sources("family"), asks=required, ineligible=family_fail,
                          manual=scope or ("Review remaining lease and other HDB prerequisites." if not required and not family_amount else None),
                          amount=family_amount)

    asks = list(dict.fromkeys(required + missing(p, ("continuous_employment_12m",))))
    ehg_fail = None
    if not scope and p.household_income_monthly is not None and p.household_income_monthly > threshold("ehg_income"):
        ehg_fail = "Declared income exceeds the first-timer family EHG ceiling."
    if not scope and p.continuous_employment_12m is False:
        ehg_fail = "The qualifying employment condition was reported as unmet."
    if p.flat_remaining_lease is not None and p.flat_remaining_lease <= threshold("minimum_lease"):
        ehg_fail = "The stated remaining lease is at or below the minimum for the family EHG."
    ehg_amount = None
    ehg_review = scope
    if not asks and not ehg_fail and not scope:
        if family.status != "may_apply":
            ehg_review = "First establish whether the household qualifies for the Family Grant."
        elif min(p.buyer_ages) + p.flat_remaining_lease < threshold("sufficient_coverage_age"):
            ehg_review = "Ask HDB to assess the pro-rated EHG for this lease."
        else:
            ehg_amount = money(next(amount for cap, amount in rules()["amounts"]["ehg"]["tiers"]
                                    if p.household_income_monthly <= cap))
    ehg = grant_result("Enhanced CPF Housing Grant", sources("ehg", "ehg_table"), asks=asks,
                       ineligible=ehg_fail, manual=ehg_review, amount=ehg_amount)

    phg_fields = ("citizenship", "near_parents", "prior_phg", "buyer_ages", "flat_remaining_lease",
                  "prerequisites_confirmed")
    asks = missing(p, phg_fields)
    phg_manual = None
    if p.citizenship not in (None, "SC_SC", "SC_SPR"):
        phg_manual = "Check the appropriate HDB household scheme."
    if p.prior_phg or p.prerequisites_confirmed is False:
        phg_manual = "Check the prior award and other PHG requirements with HDB."
    phg_fail = "The reported proximity does not meet the family PHG condition." if p.near_parents == "no" else None
    if p.buyer_ages and min(p.buyer_ages) < threshold("family_min_age"):
        phg_fail = "An applicant age is outside this family PHG screening."
    if p.flat_remaining_lease is not None and p.flat_remaining_lease <= threshold("minimum_lease"):
        phg_fail = "The stated remaining lease is at or below the minimum for the PHG."
    phg_amount = money(rules()["amounts"]["phg"][p.near_parents]) if not (asks or phg_manual or phg_fail) else None
    phg = grant_result("Proximity Housing Grant", sources("phg"), asks=asks, ineligible=phg_fail,
                       manual=phg_manual, amount=phg_amount)
    return [family, ehg, phg]


def annuity_factor(annual_rate: float, months: int) -> float:
    monthly_rate = annual_rate / 12
    return (1 - (1 + monthly_rate) ** -months) / monthly_rate if monthly_rate else float(months)


def review_reasons(p: BuyerProfile, limits: dict[str, float]) -> list[str]:
    """Explain, with the buyer's own numbers, why a case is referred to HDB instead of estimated."""
    reasons = []
    youngest = min(p.buyer_ages)
    coverage = threshold("sufficient_coverage_age")
    reach = youngest + p.flat_remaining_lease
    if reach < coverage:
        reasons.append(
            f"The {p.flat_remaining_lease:g}-year remaining lease covers the youngest buyer (age {youngest}) only to age "
            f"{reach:g}, short of {coverage:g}. HDB reduces the {threshold('ltv'):.0%} loan limit for this, and your HFE "
            f"letter gives the exact amount. A flat with at least {coverage - youngest:g} years of lease left would get "
            f"a standard estimate here.")
    shortest = min(limits.values())
    if math.floor(shortest) < 1:
        parts = ", ".join(f"{name} = {value:g}" for name, value in limits.items())
        why = []
        if limits["65 minus average age"] < 1:
            why.append(f"the applicants' average age ({mean(p.buyer_ages):g}) leaves no repayment years before 65")
        if limits["remaining lease minus 20"] < 1:
            why.append(f"HDB lends only on flats with more than {threshold('minimum_lease'):g} years of lease left")
        reasons.append(f"No repayment period is possible: the shortest of {parts} is {shortest:g} years"
                       + (f", because {' and '.join(why)}." if why else "."))
    if p.household_income_monthly > threshold("loan_family_income"):
        reasons.append(f"Household income of S${p.household_income_monthly:,.0f} a month is above the "
                       f"S${threshold('loan_family_income'):,.0f} ceiling for an HDB loan for families. Ask HDB about "
                       "other household schemes or financing from a bank.")
    if p.citizenship == "SC_single":
        reasons.append("Singles buy under a separate scheme with its own income ceiling. Confirm the loan with HDB.")
    return reasons


def estimate_affordability(profile: BuyerProfile) -> Affordability:
    p = profile
    citations = sources("loan", "interest", "cpf", "income_update")
    required = missing(p, ("purchase_price", "valuation", "household_income_monthly", "buyer_ages",
                           "existing_monthly_debt", "flat_remaining_lease"))
    if required:
        return Affordability(status="need_more_info", missing_fields=required, citations=citations)
    checks = []
    limits = {"HDB maximum": threshold("tenure"),
              "65 minus average age": threshold("age_tenure_ceiling") - mean(p.buyer_ages),
              "remaining lease minus 20": p.flat_remaining_lease - threshold("minimum_lease")}
    tenure = math.floor(min(limits.values()))
    checks.extend(review_reasons(p, limits))
    if checks:
        return Affordability(status="manual_review", checks=checks, citations=citations)
    # Reserving existing debt within the same budget is a conservative demo assumption.
    payment_budget = max(0.0, threshold("msr") * p.household_income_monthly - p.existing_monthly_debt)
    assessment_rate = max(threshold("assessment_rate"), threshold("repayment_rate"))
    ltv_ceiling = threshold("ltv") * min(p.purchase_price, p.valuation)
    payment_ceiling = payment_budget * annuity_factor(assessment_rate, tenure * 12)
    principal = math.floor(min(ltv_ceiling, payment_ceiling))
    instalment = round(principal / annuity_factor(threshold("repayment_rate"), tenure * 12), 2)
    binding = "Loan-to-value" if ltv_ceiling <= payment_ceiling else "Monthly payment budget"
    return Affordability(status="illustrative", loan_ceiling=principal, monthly_instalment=instalment,
                         tenure_years=tenure, citations=citations, tenure_limits=limits,
                         ltv_limit=math.floor(ltv_ceiling), payment_budget=round(payment_budget, 2),
                         payment_limit=math.floor(payment_ceiling), assessment_rate=assessment_rate,
                         repayment_rate=threshold("repayment_rate"), binding=binding,
                         checks=[f"{binding} sets the illustrative ceiling.",
                                 "Existing debt is reserved from the payment budget as a modelling assumption."])


GRANT_PROFILE_FIELDS = ("citizenship", "first_timer", "household_income_monthly", "buyer_ages", "flat_type",
                        "flat_remaining_lease", "continuous_employment_12m", "near_parents",
                        "prerequisites_confirmed", "prior_phg")


def reconcile_grants(rule_items: list[GrantItem], assessments, evidence: dict[str, RuleChunk],
                     profile: BuyerProfile) -> tuple[list[GrantItem], list[str]]:
    """Verify the Policy agent's per-grant reading against the deterministic rule engine.

    The agent proposes a status, cites passages and explains the deciding condition. The rule engine
    decides. Agreement adds the explanation and section-level citations; disagreement keeps the
    conservative rule result and logs an event for review and evaluation.
    """
    from .guards import check_narrative, numbers_in

    by_name = {item.grant: item for item in assessments}
    profile_numbers = [float(v) for v in profile.model_dump(include=set(GRANT_PROFILE_FIELDS)).values()
                       if isinstance(v, (int, float)) and not isinstance(v, bool)] + list(map(float, profile.buyer_ages))
    events, output = [], []
    for item in rule_items:
        proposal = by_name.get(item.name)
        if proposal is None:
            output.append(item)
            continue
        if proposal.status != item.status:
            events.append("policy_disagreement")
            output.append(item.model_copy(update={"agent_check": "disagreed"}))
            continue
        cited = [evidence[key] for key in proposal.chunk_ids if key in evidence]
        allowed = profile_numbers + [n for chunk in cited for n in numbers_in(chunk.text)]
        update = {"agent_check": "agreed"}
        if cited and not check_narrative(proposal.explanation, allowed, limit=400):
            update["explanation"] = proposal.explanation.strip()
            update["citations"] = list(dict.fromkeys(item.citations + [chunk.source_url for chunk in cited]))
        output.append(item.model_copy(update=update))
    return output, list(dict.fromkeys(events))


def funding_split(price: float, grants: float, loan_ceiling: float) -> dict[str, float]:
    """How a purchase is funded: grants first, then a loan no larger than the price left. Parts sum to the price."""
    grants_applied = min(max(grants, 0.0), price)
    loan_applied = min(max(loan_ceiling, 0.0), price - grants_applied)
    return {"grants": grants_applied, "loan": loan_applied, "balance": price - grants_applied - loan_applied}
