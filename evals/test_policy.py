from pathlib import Path

import pytest

from hdb_advisor.policy import estimate_affordability, evaluate_grants, index_rules, retrieve_rules
from hdb_advisor.state import BuyerProfile


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    return index_rules(tmp_path_factory.mktemp("chroma"))


@pytest.mark.parametrize(("question", "expected"), [
    ("Family Grant flat size SC SPR amount", "family-amount"),
    ("Family Grant income ceiling extended family", "family-eligibility"),
    ("Enhanced CPF grant household income limit", "ehg-income"),
    ("EHG continuous employment 12 months", "ehg-employment"),
    ("Enhanced CPF grant remaining lease youngest 95", "ehg-lease"),
    ("PHG parents live within 4 km", "phg-distance"),
    ("Proximity grant used once previous award", "phg-household"),
    ("HDB loan lower purchase price valuation limit", "loan-ltv"),
    ("HDB mortgage servicing ratio repayment period", "loan-tenure"),
    ("CPF Ordinary Account withdrawal lease", "cpf-usage"),
])
def test_retrieval(question, expected, store):
    assert expected in {item.id for item in retrieve_rules(question, store=store)}


def full_profile(**changes):
    values = dict(citizenship="SC_SC", first_timer=True, flat_type="4 ROOM",
                  household_income_monthly=6000, buyer_ages=[31, 32], flat_remaining_lease=70,
                  continuous_employment_12m=True, prerequisites_confirmed=True,
                  prior_phg=False, near_parents="within_4km", purchase_price=500000,
                  valuation=470000, existing_monthly_debt=200)
    return BuyerProfile(**(values | changes))


def test_grants_match_tables_and_missing_fields():
    assert [x.amount_text for x in evaluate_grants(full_profile())] == ["S$80,000", "S$50,000", "S$20,000"]
    unknown = evaluate_grants(full_profile(household_income_monthly=None))
    assert all("household_income_monthly" in result.missing_fields for result in unknown[:2])
    assert unknown[2].status == "may_apply"


def test_affordability_binds_to_lower_valuation_and_withholds_prorating():
    result = estimate_affordability(full_profile(household_income_monthly=15000))
    assert result.loan_ceiling == 352500 and result.tenure_years == 25
    assert estimate_affordability(full_profile(flat_remaining_lease=56)).status == "manual_review"
    assert estimate_affordability(full_profile(valuation=None)).missing_fields == ["valuation"]


def test_review_reasons_name_the_condition_with_the_buyers_numbers():
    short = estimate_affordability(full_profile(buyer_ages=[30, 32], flat_remaining_lease=60))
    assert short.status == "manual_review"
    assert "covers the youngest buyer (age 30) only to age 90, short of 95" in short.checks[0]
    assert "at least 65 years" in short.checks[0]
    older = estimate_affordability(full_profile(buyer_ages=[66, 68], flat_remaining_lease=80))
    assert "average age (67)" in older.checks[0]
    rich = estimate_affordability(full_profile(household_income_monthly=17000))
    assert "S$17,000" in rich.checks[0] and "S$16,000" in rich.checks[0]
