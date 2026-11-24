"""OneMap search, dated transit schedules and small, durable caches."""
import hashlib
import math
import os
import re
import sqlite3
import time
from contextlib import closing
from datetime import date, timedelta
from pathlib import Path

import requests

from .market import address_key, connect_readonly, singapore_today
from .market import CACHE_SCHEMA
from .state import Address, Destination, RouteResult

BASE = "https://www.onemap.gov.sg"
SEARCH_URL = f"{BASE}/apidocs/search"
ROUTE_URL = f"{BASE}/apidocs/routing"
DEPARTURE = "08:00:00"
MODE = "transit"
ROAD_ABBREVIATIONS = {
    "AVE": "AVENUE", "BT": "BUKIT", "CTR": "CENTRE", "CL": "CLOSE",
    "CRES": "CRESCENT", "DR": "DRIVE", "GDNS": "GARDENS", "HTS": "HEIGHTS",
    "JLN": "JALAN", "LOR": "LORONG", "PL": "PLACE", "RD": "ROAD",
    "ST": "STREET", "TER": "TERRACE", "UPP": "UPPER", "NTH": "NORTH", "STH": "SOUTH",
    "C'WEALTH": "COMMONWEALTH", "CTRL": "CENTRAL", "KG": "KAMPONG", "MKT": "MARKET",
    "TG": "TANJONG", "PK": "PARK", "SQ": "SQUARE",
}
# Tenants inside one block (preschools, shelters) share its postal code with slightly different points.
SAME_BUILDING_KM = 0.15


def road_name(value: str) -> str:
    return " ".join(ROAD_ABBREVIATIONS.get(word, word) for word in value.upper().split())


def next_weekday(today: date | None = None) -> date:
    day = (today or singapore_today()) + timedelta(days=1)
    while day.weekday() >= 5:
        day += timedelta(days=1)
    return day


class OneMapError(Exception):
    pass


class OneMap:
    def __init__(self, email: str, password: str):
        self.email = email
        self.password = password
        self.token = ""
        self.expires = 0

    @classmethod
    def from_env(cls):
        email, password = os.getenv("ONEMAP_EMAIL"), os.getenv("ONEMAP_PASSWORD")
        return cls(email, password) if email and password else None

    def authorize(self):
        response = requests.post(f"{BASE}/api/auth/post/getToken",
                                 json={"email": self.email, "password": self.password}, timeout=8)
        response.raise_for_status()
        payload = response.json()
        self.token = payload["access_token"]
        self.expires = float(payload["expiry_timestamp"])

    def get(self, path: str, params: dict) -> dict:
        try:
            for attempt in range(2):
                if time.time() >= self.expires - 60:
                    self.authorize()
                response = requests.get(f"{BASE}{path}", params=params,
                                        headers={"Authorization": self.token}, timeout=8)
                if response.status_code == 401 and attempt == 0:
                    self.expires = 0
                    continue
                response.raise_for_status()
                data = response.json()
                if "error" in data and "token" in str(data["error"]).lower():
                    raise OneMapError("Authentication failed")
                return data
        except (requests.RequestException, KeyError, TypeError, ValueError) as exc:
            raise OneMapError("OneMap request failed") from exc
        raise OneMapError("OneMap authentication failed")

    def search(self, query: str, max_pages: int | None = None):
        params = {"searchVal": query, "returnGeom": "Y", "getAddrDetails": "Y", "pageNum": 1}
        result = self.get("/api/common/elastic/search", params)
        if "error" in result or "results" not in result:
            raise OneMapError("Invalid search response")
        yield from result["results"]
        last = int(result.get("totalNumPages", 1))
        for page in range(2, (min(last, max_pages) if max_pages else last) + 1):
            result = self.get("/api/common/elastic/search", {**params, "pageNum": page})
            if "results" not in result:
                raise OneMapError("Missing search page")
            yield from result["results"]

    def transit(self, start: tuple, end: tuple, day: date) -> int | None:
        data = self.get("/api/public/routingsvc/route", {
            "start": f"{start[0]},{start[1]}", "end": f"{end[0]},{end[1]}",
            "routeType": "pt", "date": day.strftime("%m-%d-%Y"), "time": DEPARTURE,
            "mode": MODE, "numItineraries": 3,
        })
        itineraries = (data.get("plan") or {}).get("itineraries", [])
        durations = [round(item["duration"]) for item in itineraries
                     if isinstance(item.get("duration"), (int, float))
                     and math.isfinite(item["duration"]) and item["duration"] > 0]
        return min(durations, default=None)


def point(result: dict) -> tuple[float, float]:
    lat, lon = float(result["LATITUDE"]), float(result["LONGITUDE"])
    if not (1.1 < lat < 1.6 and 103.5 < lon < 104.2):
        raise ValueError("Coordinates outside Singapore")
    return lat, lon


def km_between(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    half_chord = math.sin((lat2 - lat1) / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2)**2
    return 2 * 6371 * math.asin(math.sqrt(min(1, half_chord)))


def match_address(block: str, street: str, results) -> Address:
    """Accept only one postal code for the exact block and road, with all its points inside one building."""
    key = address_key(block, street)
    by_postal = {}
    for result in results:
        if (result.get("BLK_NO", "").strip().upper() != block.strip().upper()
                or road_name(result.get("ROAD_NAME", "")) != road_name(street)):
            continue
        by_postal.setdefault(result.get("POSTAL"), []).append((point(result), result))
    if len(by_postal) != 1:
        return Address(address_key=key, status="ambiguous")
    (postal, entries), = by_postal.items()
    points = [location for location, _ in entries]
    if any(km_between(points[0], other) > SAME_BUILDING_KM for other in points[1:]):
        return Address(address_key=key, status="ambiguous")
    # Prefer the plain block address over a tenant's entry at the same postal code.
    plain = next((item for _, item in entries if item.get("SEARCHVAL", "").startswith(f"{block} ")), entries[0][1])
    lat, lon = point(plain)
    return Address(address_key=key, status="matched", lat=lat, lon=lon, postal=postal,
                   matched_address=plain.get("ADDRESS"))


def geocode_hdb_address(db: Path, origin: str, client: OneMap | None) -> Address:
    try:
        with closing(connect_readonly(db)) as conn:
            row = conn.execute("SELECT * FROM address_cache WHERE address_key = ?", (origin,)).fetchone()
        if row and row["geocode_status"] in ("matched", "ambiguous"):
            return Address(address_key=origin, status=row["geocode_status"], lat=row["lat"], lon=row["lon"],
                           postal=row["postal"], matched_address=row["matched_address"])
        if row is None or client is None:
            return Address(address_key=origin, status="geocode_failed")
        block, street = origin.split(" ", 1)
        result = match_address(block, street, client.search(f"{origin} SINGAPORE"))
        with closing(sqlite3.connect(db)) as conn, conn:
            conn.execute("INSERT OR REPLACE INTO address_cache VALUES (?, ?, ?, ?, ?, ?)",
                         (origin, result.lat, result.lon, result.postal, result.matched_address, result.status))
        return result
    except (OneMapError, sqlite3.Error, ValueError, TypeError, KeyError):
        return Address(address_key=origin, status="geocode_failed")


ACRONYMS = {"MRT", "LRT", "NTU", "NUS", "SMU", "SUTD", "SIT", "SUSS", "HDB", "CBD", "SGH", "NUH", "TTSH", "KKH",
            "CPF", "IRAS", "MOM", "ITE", "JTC", "PSA", "DBS", "OCBC", "UOB"}


def place_label(name: str) -> str:
    """Title-case a OneMap name but keep acronyms and station codes such as MRT or NS26."""
    words = []
    for word in name.split():
        core = word.strip("()/,")
        keep = core in ACRONYMS or any(ch.isdigit() for ch in core)
        words.append(word if keep else word.title())
    return " ".join(words)[:60]


def destination_key(text: str) -> str:
    return " ".join(re.sub(r"[^A-Z0-9 ]", " ", text.upper()).split())


def set_anchor(db: Path, lat: float, lon: float, *, anchor_id: str, label: str, address: str | None = None,
               aliases: tuple[str, ...] = ()) -> None:
    """Store a commute destination and the phrasings that refer to it."""
    point({"LATITUDE": lat, "LONGITUDE": lon})
    with closing(sqlite3.connect(db)) as conn, conn:
        for statement in CACHE_SCHEMA:
            conn.execute(statement)
        previous = conn.execute("SELECT lat, lon FROM anchors WHERE anchor_id=?", (anchor_id,)).fetchone()
        if previous and previous != (lat, lon):
            conn.execute("DELETE FROM route_cache WHERE anchor_id=?", (anchor_id,))
        conn.execute("INSERT OR REPLACE INTO anchors VALUES (?, ?, ?, ?)", (anchor_id, label, lat, lon))
        for alias in {label, *aliases}:
            conn.execute("INSERT OR REPLACE INTO destination_cache VALUES (?, ?, ?)",
                         (destination_key(alias), anchor_id, address))


def resolve_destination(db: Path, text: str, client: OneMap | None) -> Destination:
    """Turn the buyer's destination into coordinates: cache first, then OneMap's top search match.

    The match is shown to the buyer at the plan gate, so a wrong guess is visible and editable.
    """
    key = destination_key(text)
    result = Destination(query=text, status="offline_unknown" if client is None else "not_found")
    if not key:
        return result
    try:
        with closing(connect_readonly(db)) as conn:
            row = conn.execute("""
                SELECT d.anchor_id, d.address, a.label FROM destination_cache d
                JOIN anchors a USING (anchor_id) WHERE d.query_key = ?""", (key,)).fetchone()
        if row:
            return Destination(query=text, status="cached", anchor_id=row["anchor_id"], label=row["label"],
                               address=row["address"])
        if client is None:
            return result
        top = next(iter(client.search(text, max_pages=1)), None)
        if top is None:
            return result
        lat, lon = point(top)
        postal = str(top.get("POSTAL") or "")
        suffix = postal if postal.isdigit() else hashlib.sha1(top.get("SEARCHVAL", key).encode()).hexdigest()[:10].upper()
        anchor_id = f"DEST_{suffix}"
        label = place_label(top.get("SEARCHVAL", text))
        set_anchor(db, lat, lon, anchor_id=anchor_id, label=label, address=top.get("ADDRESS"), aliases=(text,))
        return Destination(query=text, status="matched", anchor_id=anchor_id, label=label, address=top.get("ADDRESS"))
    except (OneMapError, sqlite3.Error, ValueError, TypeError, KeyError):
        return result


def anchor_label(db: Path, anchor: str) -> str:
    try:
        with closing(connect_readonly(db)) as conn:
            row = conn.execute("SELECT label FROM anchors WHERE anchor_id=?", (anchor,)).fetchone()
        return row["label"] if row else anchor
    except sqlite3.Error:
        return anchor


def route_to_anchor(db: Path, origin: str, anchor: str, client: OneMap | None,
                    day: date | None = None, *, allow_proxy: bool = True) -> RouteResult:
    explicit_date = day is not None
    day = day or next_weekday()
    label = anchor_label(db, anchor)
    assumption = f"{label}, weekday 08:00 Singapore time, {day.isoformat()}, public transport via OneMap"
    result = RouteResult(address_key=origin, anchor_id=anchor, assumption=assumption, status="no_route")
    if day.weekday() >= 5:
        return result
    try:
        with closing(connect_readonly(db)) as conn:
            cached = conn.execute("""
                SELECT duration_sec, dep_date FROM route_cache
                WHERE address_key=? AND anchor_id=? AND dep_time=? AND mode=?
                  AND source='OneMap' AND (dep_date=? OR ?)
                ORDER BY dep_date=? DESC, dep_date DESC LIMIT 1
            """, (origin, anchor, DEPARTURE, MODE, day.isoformat(),
                  not explicit_date and client is None, day.isoformat())).fetchone()
            target = conn.execute("SELECT lat, lon FROM anchors WHERE anchor_id=?", (anchor,)).fetchone()
        if cached:
            return result.model_copy(update={"duration_min": cached["duration_sec"] / 60,
                                             "status": "cached",
                                             "assumption": assumption.replace(day.isoformat(), cached["dep_date"])})
        if not target:
            return result
        location = geocode_hdb_address(db, origin, client)
        if location.status != "matched":
            return result.model_copy(update={"status": "geocode_failed"})
        start, end = (location.lat, location.lon), tuple(target)
        try:
            seconds = client.transit(start, end, day) if client else None
        except OneMapError:
            seconds = None
        if seconds:
            with closing(sqlite3.connect(db)) as conn, conn:
                conn.execute("INSERT OR REPLACE INTO route_cache VALUES (?, ?, ?, ?, ?, ?, ?)",
                             (origin, anchor, day.isoformat(), DEPARTURE, MODE, seconds, "OneMap"))
            return result.model_copy(update={"duration_min": seconds / 60, "status": "ok"})
        if allow_proxy:
            km = km_between(start, end)
            return result.model_copy(update={"status": "proxy", "distance_km": km,
                                             "assumption": f"Straight-line distance to {label}. This is not a travel time."})
    except (sqlite3.Error, TypeError, ValueError):
        pass
    return result
