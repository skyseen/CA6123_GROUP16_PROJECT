from datetime import date
from pathlib import Path
import os
import sqlite3

import pandas as pd
import pytest

from hdb_advisor.market import connect_readonly, ingest_csv, provenance, search_market
from hdb_advisor.state import BuyerProfile


@pytest.mark.parametrize("profile", [
    BuyerProfile(flat_type="4 ROOM", max_price=600_000),
    BuyerProfile(flat_type="3 ROOM", max_price=520_000, min_est_lease_years=60, months_back=6),
    BuyerProfile(town="TAMPINES", max_price=850_000, avoid_floor="01 TO 03", months_back=18),
])
def test_real_csv_matches_pandas(profile):
    csv = os.getenv("HDB_ORACLE_CSV")
    if not csv:
        pytest.skip("Set HDB_ORACLE_CSV to the official CSV used to build data/hdb.db")
    today = date(2026, 11, 23)
    df = pd.read_csv(csv)
    end = pd.Period(df.month.max(), freq="M")
    start = end - (profile.months_back - 1)
    matches = df[df.month.between(str(start), str(end))]
    if profile.flat_type:
        matches = matches[matches.flat_type == profile.flat_type]
    if profile.max_price:
        matches = matches[matches.resale_price <= profile.max_price]
    if profile.min_est_lease_years is not None:
        remaining = 99 - (today.year + today.month / 12 - matches.lease_commence_date)
        matches = matches[remaining >= profile.min_est_lease_years]
    if profile.town:
        matches = matches[matches.town == profile.town]
    if profile.avoid_floor:
        matches = matches[matches.storey_range != profile.avoid_floor]
    result = search_market(Path("data/hdb.db"), profile, today=today)
    assert result.source.rows == len(df)
    assert result.window == f"{start} to {end}"
    expected = matches.groupby("town").resale_price
    assert {t.town for t in result.towns} == set(expected.groups)
    for town in result.towns:
        prices = expected.get_group(town.town)
        assert town.n_tx == prices.size
        assert [town.p25, town.median, town.p75] == pytest.approx(
            list(prices.quantile([.25, .5, .75]))
        )


def test_bad_refresh_rolls_back(tmp_path):
    db = tmp_path / "hdb.db"
    csv = tmp_path / "hdb.csv"
    header = ("month,town,flat_type,block,street_name,storey_range,floor_area_sqm,flat_model,"
              "lease_commence_date,remaining_lease,resale_price\n")
    row = "2026-10,TAMPINES,4 ROOM,10,SAMPLE RD,04 TO 06,92,MODEL A,1990,63 years,"
    csv.write_text(header + row + "500000\n")
    old = ingest_csv(db, csv)
    csv.write_text(header + row + "NaN\n")
    with pytest.raises(ValueError):
        ingest_csv(db, csv)
    assert provenance(db) == old
    with connect_readonly(db) as conn, pytest.raises(sqlite3.OperationalError, match="readonly"):
        conn.execute("CREATE TABLE write_check(id INTEGER)")
