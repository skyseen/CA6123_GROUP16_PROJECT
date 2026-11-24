"""Render tool results consistently and keep report figures tied to their sources."""
from zoneinfo import ZoneInfo
from datetime import datetime
import json
import re

from .guards import check_narrative, numbers_in
from .onemap import ROUTE_URL, place_label
from .policy import HFE, money, rules
from .state import AdvisorState

DATA_LICENCE = "https://developers.data.gov.sg/open-data-licence"
RFL = "https://www.hdb.gov.sg/buying-a-flat/resale-flats/finding-a-resale-flat"
DISCLAIMER = "Illustrative only. HDB’s HFE letter provides the official assessment."
FIELD_LABELS = {
    "household_income_monthly": "gross monthly household income",
    "buyer_ages": "applicant ages",
    "flat_remaining_lease": "the proposed flat's remaining lease",
    "first_timer": "first-timer status",
    "near_parents": "whether you will live with or near parents or a child",
    "prior_phg": "any previous PHG award",
    "prerequisites_confirmed": "confirmation of other HDB prerequisites",
    "continuous_employment_12m": "confirmation of qualifying employment",
}


def describe_missing(fields: list[str]) -> str:
    return ", ".join(FIELD_LABELS.get(key, key.replace("_", " ")) for key in fields)


def links(urls) -> str:
    return ", ".join(f"[Source {index}]({url})" for index, url in enumerate(urls, 1))


SHOWN_TOWNS = 5
ANALYST_TOWNS = 8


def ranked_towns(state: AdvisorState) -> list[tuple[dict, dict | None]]:
    """Order towns by commute when routes exist, otherwise keep the median-price order.

    Verified transit times within the limit come first, then verified times above it, then
    straight-line distances (never presented as minutes), then towns without any commute data.
    """
    towns = (state.get("market") or {}).get("towns") or []
    routes = {row["address_key"]: row for row in state.get("routes") or []}
    pairs = [(town, routes.get(town["sample_blocks"][0]) if town["sample_blocks"] else None) for town in towns]
    if not routes:
        return pairs
    limit = (state.get("profile") or {}).get("max_commute_min")

    def key(pair):
        town, route = pair
        minutes = route.get("duration_min") if route else None
        km = route.get("distance_km") if route else None
        if minutes is not None:
            return (0 if limit is None or minutes <= limit else 1, minutes, town["median"])
        if km is not None:
            return (2, km, town["median"])
        return (3, 0, town["median"])
    return sorted(pairs, key=key)


def commute_basis(state: AdvisorState) -> str | None:
    routes = state.get("routes") or []
    if not routes:
        return None
    if any(row.get("duration_min") is not None for row in routes):
        return "verified transit time"
    if any(row.get("distance_km") is not None for row in routes):
        return "straight-line distance"
    return None


def writer_context(state: AdvisorState) -> dict:
    """The only data the Report writer sees: compact summaries, never raw transaction rows."""
    profile = state.get("profile") or {}
    market = state.get("market") or {}
    limit = profile.get("max_commute_min")
    towns = []
    for town, route in ranked_towns(state)[:ANALYST_TOWNS]:
        # Figures are pre-formatted so the writer copies them instead of reformatting or computing.
        row = {"town": town["town"], "transactions": town["n_tx"], "median_price": money(town["median"]),
               "middle_half_price_range": f"{money(town['p25'])} to {money(town['p75'])}",
               "example_block": town["sample_blocks"][0] if town["sample_blocks"] else None}
        if route and route.get("duration_min") is not None:
            row["commute_minutes"] = round(route["duration_min"], 1)
            if limit is not None:
                row["within_commute_limit"] = route["duration_min"] <= limit
        elif route and route.get("distance_km") is not None:
            row["straight_line_km_not_travel_time"] = round(route["distance_km"], 1)
        elif route:
            row["commute"] = "no verified transit time"
        towns.append(row)
    context = {
        "intent": state.get("intent"),
        "filters": {key: (money(profile[key]) if key == "max_price" else profile[key]) for key in ("flat_type", "max_price", "months_back", "min_est_lease_years",
                                                   "town", "destination", "max_commute_min")
                    if profile.get(key) is not None},
        "registration_window": market.get("window"),
        "towns_ranked": towns,
        "ranking_basis": commute_basis(state) or "median transacted price",
        "grants": [{"name": g["name"], "status": g["status"], "indicative_amount": g["amount_text"],
                    "missing_fields": g["missing_fields"]} for g in state.get("grants") or []],
    }
    afford = state.get("affordability")
    if afford:
        context["illustrative_loan"] = {
            "status": afford.get("status"), "missing_fields": afford.get("missing_fields"),
            "loan_ceiling": money(afford["loan_ceiling"]) if afford.get("loan_ceiling") is not None else None,
            "monthly_instalment": f"S${afford['monthly_instalment']:,.2f}" if afford.get("monthly_instalment") is not None else None,
            "tenure_years": afford.get("tenure_years")}
    return context


MONEY = re.compile(r"S\$\s?(\d[\d,]*(?:\.\d+)?)\s*(k|m|million)?\b", re.I)
MINUTES = re.compile(r"(\d+(?:\.\d+)?)\s*(?:min(?:ute)?s?)\b", re.I)
METRIC_WORDS = {"median": "median", "middle": "range", "range": "range", "quartile": "range", "between": "range"}
BUDGET_WORDS = re.compile(r"\b(?:budget|under|below|up to|within|cap|maximum|max|limit)\b", re.I)


def _money(match) -> float:
    return float(match[1].replace(",", "")) * {"k": 1e3, "m": 1e6, "million": 1e6}.get((match[2] or "").lower(), 1)


def possessive(town: str) -> str:
    name = town.title()
    return f"{name}'" if name.endswith("s") else f"{name}'s"


def _same(a: float, b: float) -> bool:
    return abs(a - b) <= max(500 if b >= 10_000 else 0.51, b * 0.001)


def claim_issues(text: str, state: AdvisorState, default_town: str | None = None) -> list[str]:
    """Bind every price and commute figure to the town and measure it describes.

    A number that exists somewhere in the tool output is not enough: "Woodlands median S$600,000" fails when
    S$600,000 is the budget and the Woodlands median is S$538,000.
    """
    profile = state.get("profile") or {}
    budget = profile.get("max_price")
    limit = profile.get("max_commute_min")
    towns = {}
    for town, route in ranked_towns(state):
        towns[town["town"]] = {"median": town["median"], "range": (town["p25"], town["p75"]),
                               "minutes": route.get("duration_min") if route else None}
    names = sorted(towns, key=len, reverse=True)
    issues = []
    for sentence in re.split(r"(?<=[.!?;])\s+", text):
        lowered = sentence.lower()
        for kind, pattern in (("money", MONEY), ("minutes", MINUTES)):
            for match in pattern.finditer(sentence):
                value = _money(match) if kind == "money" else float(match[1])
                before = lowered[max(0, match.start() - 40):match.start()]
                mentioned = [n for n in names if n.lower() in lowered[:match.start()]]
                town = (max(mentioned, key=lambda n: lowered.rfind(n.lower(), 0, match.start()))
                        if mentioned else default_town)
                shown = match[0].strip()
                if BUDGET_WORDS.search(before[-20:]):
                    target = budget if kind == "money" else limit
                    if target is not None and _same(value, target):
                        continue
                if kind == "minutes":
                    minutes = towns.get(town, {}).get("minutes") if town else None
                    if limit is not None and _same(value, limit) and not town:
                        continue
                    if town and minutes is not None and not _same(value, round(minutes, 1)):
                        issues.append(f"{shown} is not the sampled commute for {town.title()} ({minutes:.1f} min).")
                    continue
                if not town:
                    if "median" in before and not any(_same(value, t["median"]) for t in towns.values()):
                        issues.append(f"{shown} is not the median of any listed town.")
                    continue
                facts = towns.get(town)
                if facts is None:
                    continue
                metric = next((m for word, m in METRIC_WORDS.items() if word in before), None)
                allowed = {"median": [facts["median"]], "range": list(facts["range"])}.get(
                    metric, [facts["median"], *facts["range"]])
                if not any(_same(value, a) for a in allowed):
                    label = f"{metric} " if metric else ""
                    issues.append(f"{shown} is not {possessive(town)} {label}price. Copy the figure for that town.")
    return list(dict.fromkeys(issues))


def narrative_issues(narrative: dict, state: AdvisorState) -> list[str]:
    """Deterministic critic: every figure must come from the writer's data and every pick must be a supplied town."""
    context = writer_context(state)
    allowed = numbers_in(json.dumps(context))
    names = {row["town"] for row in context["towns_ranked"]}
    issues = check_narrative(narrative.get("summary", ""), allowed) + claim_issues(narrative.get("summary", ""), state)
    for pick in narrative.get("shortlist", []):
        if pick.get("town") not in names:
            issues.append(f"'{pick.get('town')}' is not one of the supplied towns.")
        issues += check_narrative(pick.get("reason", ""), allowed, limit=300)
        issues += claim_issues(pick.get("reason", ""), state, default_town=pick.get("town"))
    for step in narrative.get("next_steps", []):
        issues += check_narrative(step, allowed, limit=250) + claim_issues(step, state)
    if not narrative.get("summary", "").strip():
        issues.append("Write a summary.")
    return list(dict.fromkeys(issues))


def render_narrative(narrative: dict) -> str:
    rows = ["## Summary", "", narrative["summary"].strip()]
    if narrative.get("shortlist"):
        rows.extend(["", "## Towns to look up first", ""])
        rows.extend(f"{index}. **{pick['town']}**: {pick['reason'].strip()}"
                    for index, pick in enumerate(narrative["shortlist"], 1))
    if narrative.get("next_steps"):
        rows.extend(["", "### Next steps", ""])
        rows.extend(f"- {step.strip()}" for step in narrative["next_steps"])
    return "\n".join(rows)


def compose(canonical: str, narrative: dict | None) -> str:
    """Place the verified narrative under the title and dates, above the tool-rendered figures."""
    if not narrative or not narrative.get("summary"):
        return canonical
    lines = canonical.split("\n")
    return "\n".join(lines[:3] + ["", render_narrative(narrative)] + lines[3:])


def commute_text(route: dict | None) -> str:
    if not route:
        return ""
    if route.get("duration_min") is not None:
        return f" · sample block {route['address_key']}, **{route['duration_min']:.1f} min**"
    if route.get("distance_km") is not None:
        return f" · sample block {route['address_key']}, {route['distance_km']:.1f} km straight-line (not a travel time)"
    return f" · sample block {route['address_key']}, travel time unknown"


def town_line(town: dict, route: dict | None = None) -> str:
    leases = town["sample_estimated_leases"]
    blocks = ", ".join(f"{key} ({leases[key]:.1f} years)" for key in town["sample_blocks"])
    count = f"{town['n_tx']:,} transaction{'s' if town['n_tx'] != 1 else ''}"
    return (f"- **{town['town']}**{commute_text(route)} · {count} · median {money(town['median'])} "
            f"· middle half {money(town['p25'])}–{money(town['p75'])} · examples {blocks}.")


def commute_split(ranked, limit):
    fits = [(t, r) for t, r in ranked if r and r.get("duration_min") is not None and r["duration_min"] <= limit]
    above = [(t, r) for t, r in ranked if r and r.get("duration_min") is not None and r["duration_min"] > limit]
    unknown = [(t, r) for t, r in ranked if not r or r.get("duration_min") is None]
    return fits, above, unknown


def commute_headline(ranked, limit) -> str:
    """Lead with the answer: how many sampled journeys met the limit, and which."""
    fits, above, _ = commute_split(ranked, limit)
    checked = len(fits) + len(above)
    if not checked:
        return f"No verified transit times are available yet, so none can be checked against your {limit}-minute limit."
    noun = "route" if checked == 1 else "routes"
    head = f"Of the {checked} sample {noun} with a verified time, {len(fits)} met your {limit}-minute limit"
    if fits:
        head += ": " + "; ".join(f"{r['address_key']} in {t['town'].title()}, {r['duration_min']:.1f} minutes"
                                 for t, r in fits[:3])
    return head + "."


def commute_groups(ranked, limit) -> list[str]:
    fits, above, unknown = commute_split(ranked, limit)
    rows = [commute_headline(ranked, limit), "", f"### Within {limit} minutes", ""]
    rows.extend(town_line(t, r) for t, r in fits[:10]) if fits else rows.append("No sampled block met the limit.")
    if above:
        rows.extend(["", f"### Above {limit} minutes", ""])
        rows.extend(town_line(t, r) for t, r in above[:SHOWN_TOWNS])
        if len(above) > SHOWN_TOWNS:
            rows.append(f"{len(above) - SHOWN_TOWNS} more towns were above the limit.")
    if unknown:
        rows.extend(["", "### Travel time unknown", ""])
        rows.extend(town_line(t, r) for t, r in unknown[:SHOWN_TOWNS])
        if len(unknown) > SHOWN_TOWNS:
            rows.append(f"{len(unknown) - SHOWN_TOWNS} more towns have no verified travel time.")
    return rows


def render_report(state: AdvisorState) -> str:
    market = state.get("market") or {}
    provenance = market.get("source")
    freshness = state.get("freshness", {})
    provenance = provenance or freshness.get("source")
    data_date = "Unavailable"
    if provenance:
        data_date = datetime.fromisoformat(provenance["retrieved_at"]).astimezone(
            ZoneInfo("Asia/Singapore")
        ).date().isoformat()
    rows = ["# HDB resale research brief", "", f"Transaction snapshot as of **{data_date}**. "
            f"Rules snapshot as of **{rules()['snapshot_date']}**."]
    note = freshness.get("note")
    if note:
        rows.extend(["", note])
    events = state.get("guard_events", [])
    if "direct_instruction" in events:
        rows.extend(["", "The request to override app instructions was blocked. You can ask about HDB resale research."])
    elif "investment_advice" in events:
        rows.extend(["", "I can summarize historical prices. Buy or sell recommendations and price forecasts are outside this service."])
    if "live_listing_request" in events or state.get("intent") == "explain":
        rows.extend(["", f"Registered resale transactions describe completed sales. Check current inventory through "
                     f"[HDB’s Resale Flat Listing service]({RFL}) and assessment details through the [HFE letter]({HFE})."])
    if state.get("intent") == "out_of_scope" and "direct_instruction" not in events:
        rows.extend(["", "Ask about registered HDB resale transactions, CPF housing grants, or an illustrative HDB loan."])
    destination = state.get("destination") or {}
    if destination.get("query") and not destination.get("anchor_id"):
        reason = ("is not in the offline destination cache" if destination.get("status") == "offline_unknown"
                  else "was not found on OneMap")
        rows.extend(["", f"Commute destination \"{destination['query']}\" {reason}, so no commute was estimated. "
                         "Try a postal code, a building name or an MRT station."])
    elif destination.get("label"):
        rows.extend(["", f"Commute destination: **{destination['label']}**"
                         + (f" ({place_label(destination['address'])})" if destination.get("address") else "") + "."])
    if market:
        rows.extend(["", "## Historical transactions", ""])
        if market.get("error"):
            rows.append(market["error"])
        else:
            rows.append(f"Past sales registered **{market['window']}** (a search window, not a loan term). The latest month may be partial. Source [HDB resale transactions]({provenance['url']}) under the [Singapore Open Data Licence]({DATA_LICENCE}).")
            lease_filter = market["query"].get("min_est_lease_years")
            if lease_filter is not None:
                rows.append(f"Minimum estimated remaining lease **{lease_filter} years** as of {market['as_of']}.")
            if not market["towns"]:
                rows.append("No matching transactions. A wider observation window or price range may return matches.")
            if market["towns"]:
                rows.append(f"Example block leases are estimates as of {market['as_of']}, based on lease commencement years.")
            ranked = ranked_towns(state)
            limit = (state.get("profile") or {}).get("max_commute_min")
            if state.get("routes") and limit is not None:
                rows.extend(commute_groups(ranked, limit))
            else:
                basis = commute_basis(state)
                if basis:
                    rows.append(f"Towns are ranked by {basis} from one example block per town, then by median price.")
                rows.extend(town_line(town, route) for town, route in ranked[:SHOWN_TOWNS])
            if market["towns"]:
                rows.append(f"Use the example blocks to guide a search on the [HDB Flat Portal]({RFL}).")
    if state.get("routes"):
        assumption = next((r["assumption"] for r in state["routes"] if r.get("status") != "proxy"), state["routes"][0]["assumption"])
        rows.extend(["", f"Commute assumption: {assumption}. One example block is routed per town, so other blocks "
                     f"in the same town may be quicker or slower. [OneMap]({ROUTE_URL})"])
    if state.get("grants"):
        rows.extend(["", "## Grant screening", ""])
        for grant in state["grants"]:
            status = grant["status"].replace("_", " ")
            details = []
            if grant["amount_text"]:
                details.append(f"Indicative {grant['amount_text']}")
            if grant["missing_fields"]:
                details.append("Please provide " + describe_missing(grant["missing_fields"]) + ".")
            details.append(grant["note"])
            if grant.get("explanation"):
                details.append(f"Policy agent: {grant['explanation']}")
            if grant.get("agent_check") == "disagreed":
                details.append("The policy agent's reading differed from the rule check, so the conservative result is shown. Confirm with HDB.")
            details.append(links(grant["citations"]))
            rows.append(f"- **{grant['name']}** · {status}. {' '.join(details)}")
        rows.append(f"Confirm the result through your [HFE letter]({HFE}).")
    if state.get("evidence"):
        rows.extend(["", "### Policy evidence", ""])
        rows.extend(f"- [{chunk['title']} · {chunk['section']}]({chunk['source_url']})"
                    for chunk in state["evidence"])
    if state.get("affordability"):
        result = state["affordability"]
        rows.extend(["", "## Illustrative HDB loan", ""])
        if result["status"] == "illustrative":
            rows.append(f"Ceiling **{money(result['loan_ceiling'])}**. Estimated monthly instalment "
                        f"**S${result['monthly_instalment']:,.2f}** over **{result['tenure_years']} years**.")
        elif result["missing_fields"]:
            rows.append("Please provide " + describe_missing(result["missing_fields"]) + ".")
        else:
            rows.append("HDB review is needed for the reported inputs.")
        rows.extend(result["checks"])
        rows.extend([links(result["citations"]), DISCLAIMER + f" [HFE letter]({HFE})"])
    return "\n".join(rows) + "\n"
