from datetime import date
from types import SimpleNamespace

from hdb_advisor.market import ingest_csv
from hdb_advisor.onemap import OneMap, match_address, resolve_destination, route_to_anchor, set_anchor

ANCHOR_ID = "NTU_NORTH_SPINE"

POINT = {"BLK_NO": "10", "ROAD_NAME": "SAMPLE ROAD", "LATITUDE": "1.34",
         "LONGITUDE": "103.80", "POSTAL": "123456", "ADDRESS": "10 SAMPLE ROAD"}


def test_unique_exact_match():
    assert match_address("10", "SAMPLE RD", [POINT, POINT]).status == "matched"
    assert match_address("11", "SAMPLE RD", [POINT]).status == "ambiguous"
    duplicate = {**POINT, "POSTAL": "123457"}
    assert match_address("10", "SAMPLE RD", [POINT, duplicate]).status == "ambiguous"
    tenant = {**POINT, "LATITUDE": "1.34002", "ADDRESS": "PRESCHOOL"}
    assert match_address("10", "SAMPLE RD", [tenant, POINT]).status == "matched"
    far = {**POINT, "LATITUDE": "1.36"}
    assert match_address("10", "SAMPLE RD", [POINT, far]).status == "ambiguous"
    north = {**POINT, "ROAD_NAME": "TOA PAYOH NORTH"}
    assert match_address("10", "TOA PAYOH NTH", [north]).status == "matched"


def test_shortest_itinerary(monkeypatch):
    client = OneMap("example", "example")
    monkeypatch.setattr(client, "get", lambda path, params: {
        "plan": {"itineraries": [{"duration": 3100}, {"duration": 2500}, {"duration": None}]}
    })
    assert client.transit((1.3, 103.7), (1.34, 103.8), date(2026, 11, 24)) == 2500


def test_cache_and_proxy_remain_distinct(tmp_path):
    csv = tmp_path / "rows.csv"
    csv.write_text("month,town,flat_type,block,street_name,storey_range,floor_area_sqm,flat_model,"
                   "lease_commence_date,remaining_lease,resale_price\n"
                   "2026-10,TAMPINES,4 ROOM,10,SAMPLE RD,04 TO 06,92,MODEL A,1990,63 years,500000\n")
    db = tmp_path / "hdb.db"
    ingest_csv(db, csv)
    set_anchor(db, 1.35, 103.70, anchor_id=ANCHOR_ID, label="NTU North Spine", aliases=("NTU",))
    client = SimpleNamespace(search=lambda term: [POINT], transit=lambda *args: 1800)
    day = date(2026, 11, 24)
    assert route_to_anchor(db, "10 SAMPLE RD", ANCHOR_ID, client, day).status == "ok"
    cached = route_to_anchor(db, "10 SAMPLE RD", ANCHOR_ID, None, day)
    assert cached.status == "cached" and cached.duration_min == 30
    proxy = route_to_anchor(db, "10 SAMPLE RD", ANCHOR_ID, None, date(2026, 11, 25))
    assert proxy.status == "proxy" and proxy.duration_min is None and proxy.distance_km > 0


def test_destination_resolves_from_cache_then_onemap(tmp_path):
    """Any destination the buyer names is resolved: the cache first, then OneMap's top match, then cached."""
    csv = tmp_path / "rows.csv"
    csv.write_text("month,town,flat_type,block,street_name,storey_range,floor_area_sqm,flat_model,"
                   "lease_commence_date,remaining_lease,resale_price\n"
                   "2026-10,TAMPINES,4 ROOM,10,SAMPLE RD,04 TO 06,92,MODEL A,1990,63 years,500000\n")
    db = tmp_path / "hdb.db"
    ingest_csv(db, csv)
    assert resolve_destination(db, "Raffles Place MRT", None).status == "offline_unknown"
    place = {"SEARCHVAL": "RAFFLES PLACE MRT STATION (NS26 / EW14)", "LATITUDE": "1.2840", "LONGITUDE": "103.8514",
             "POSTAL": "048618", "ADDRESS": "5 RAFFLES PLACE SINGAPORE 048618", "BLK_NO": "5", "ROAD_NAME": "RAFFLES PLACE"}
    calls = []
    client = SimpleNamespace(search=lambda term, max_pages=None: (calls.append(term), iter([place]))[1])
    found = resolve_destination(db, "Raffles Place MRT", client)
    assert (found.status, found.anchor_id) == ("matched", "DEST_048618") and "Raffles" in found.label
    again = resolve_destination(db, "raffles place mrt", None)
    assert (again.status, again.anchor_id) == ("cached", "DEST_048618") and calls == ["Raffles Place MRT"]
    empty = SimpleNamespace(search=lambda term, max_pages=None: iter([]))
    assert resolve_destination(db, "Nowhere Land", empty).status == "not_found"
