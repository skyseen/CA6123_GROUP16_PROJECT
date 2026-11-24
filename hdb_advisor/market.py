"""The dataset has one writer and short-lived, read-only query connections."""
import csv
import math
import re
import sqlite3
import time
from collections import defaultdict
from contextlib import closing
from datetime import date, datetime, timezone
from pathlib import Path
from typing import TextIO
from zoneinfo import ZoneInfo

import requests

from .state import BuyerProfile, MarketSummary, Provenance, TownSummary

DATASET_ID = "d_8b84c4ee58e3cfc0ece0d773c8ca6abc"
DATASET_URL = f"https://data.gov.sg/datasets/{DATASET_ID}/view"
DOWNLOAD_API = f"https://api-open.data.gov.sg/v1/public/api/datasets/{DATASET_ID}"
METADATA_API = f"https://api-production.data.gov.sg/v2/public/api/datasets/{DATASET_ID}/metadata"
CSV_COLUMNS = (
    "month", "town", "flat_type", "block", "street_name", "storey_range",
    "floor_area_sqm", "flat_model", "lease_commence_date", "remaining_lease", "resale_price",
)
TX_COLUMNS = (
    "month", "town", "flat_type", "block", "street_name", "storey_range",
    "floor_area_sqm", "flat_model", "lease_commence_year", "remaining_lease_raw",
    "resale_price", "address_key",
)
TX_SCHEMA = """
    month TEXT NOT NULL, town TEXT NOT NULL, flat_type TEXT NOT NULL,
    block TEXT NOT NULL, street_name TEXT NOT NULL, storey_range TEXT NOT NULL,
    floor_area_sqm REAL NOT NULL, flat_model TEXT NOT NULL,
    lease_commence_year INTEGER NOT NULL, remaining_lease_raw TEXT,
    resale_price REAL NOT NULL, address_key TEXT NOT NULL
"""
CACHE_SCHEMA = (
    """CREATE TABLE IF NOT EXISTS address_cache (
        address_key TEXT PRIMARY KEY, lat REAL, lon REAL, postal TEXT,
        matched_address TEXT, geocode_status TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS route_cache (
        address_key TEXT, anchor_id TEXT, dep_date TEXT, dep_time TEXT,
        mode TEXT, duration_sec INTEGER NOT NULL, source TEXT NOT NULL,
        PRIMARY KEY (address_key, anchor_id, dep_date, dep_time, mode))""",
    """CREATE TABLE IF NOT EXISTS anchors (
        anchor_id TEXT PRIMARY KEY, label TEXT NOT NULL, lat REAL NOT NULL, lon REAL NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS destination_cache (
        query_key TEXT PRIMARY KEY, anchor_id TEXT NOT NULL, address TEXT)""",
    """CREATE TABLE IF NOT EXISTS provenance (
        source_id TEXT PRIMARY KEY, url TEXT NOT NULL, retrieved_at TEXT NOT NULL,
        rows INTEGER NOT NULL, last_month TEXT NOT NULL)""",
)


def singapore_today() -> date:
    return datetime.now(ZoneInfo("Asia/Singapore")).date()


def connect_readonly(db: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def address_key(block: str, road: str) -> str:
    return " ".join(f"{block} {road}".upper().split())


def normalized_rows(source: TextIO):
    reader = csv.DictReader(source)
    if len(reader.fieldnames or []) != len(CSV_COLUMNS) or set(reader.fieldnames or []) != set(CSV_COLUMNS):
        raise ValueError("Expected all 11 official HDB CSV columns")
    for row in reader:
        try:
            if None in row or any(row[name] is None for name in CSV_COLUMNS):
                raise ValueError("Incomplete row")
            month = row["month"].strip()
            if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", month):
                raise ValueError("Invalid registration month")
            year = int(row["lease_commence_date"])
            price = float(row["resale_price"])
            area = float(row["floor_area_sqm"])
            if not (1900 <= year <= int(month[:4]) and math.isfinite(price) and price > 0
                    and math.isfinite(area) and area > 0):
                raise ValueError("Invalid transaction values")
            town, flat, block, road, storey, model = (
                " ".join(row[key].upper().split())
                for key in ("town", "flat_type", "block", "street_name", "storey_range", "flat_model")
            )
            if not all((town, flat, block, road, storey, model)):
                raise ValueError("Missing transaction identifiers")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid HDB CSV row {reader.line_num}") from exc
        yield (month, town, flat, block, road, storey, area, model, year,
               row["remaining_lease"].strip(), price, address_key(block, road))


def provenance(db: Path) -> Provenance | None:
    if not db.is_file():
        return None
    with closing(connect_readonly(db)) as conn:
        exists = conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'provenance'").fetchone()
        if not exists:
            return None
        row = conn.execute("SELECT * FROM provenance WHERE source_id = ?", (DATASET_ID,)).fetchone()
    return Provenance(**dict(row)) if row else None


def ingest_csv(db: Path, csv_path: Path, *, retrieved_at: str | None = None,
               expected_rows: int | None = None) -> Provenance:
    db.parent.mkdir(parents=True, exist_ok=True)
    captured = retrieved_at or datetime.now(timezone.utc).isoformat()
    with closing(sqlite3.connect(db, timeout=30)) as conn, conn, csv_path.open(encoding="utf-8-sig", newline="") as source:
        # Taking the writer lock before reading provenance prevents overlapping refreshes from
        # comparing against an out-of-date baseline.
        conn.execute("BEGIN IMMEDIATE")
        for statement in CACHE_SCHEMA:
            conn.execute(statement)
        previous = conn.execute(
            "SELECT rows, last_month, retrieved_at FROM provenance WHERE source_id = ?", (DATASET_ID,)
        ).fetchone()
        conn.execute(f"CREATE TABLE resale_staging (id INTEGER PRIMARY KEY, {TX_SCHEMA})")
        placeholders = ", ".join("?" for _ in TX_COLUMNS)
        conn.executemany(
            f"INSERT INTO resale_staging ({', '.join(TX_COLUMNS)}) VALUES ({placeholders})",
            normalized_rows(source),
        )
        rows, last_month = conn.execute("SELECT COUNT(*), MAX(month) FROM resale_staging").fetchone()
        if not rows or (expected_rows is not None and rows != expected_rows):
            raise ValueError("HDB CSV row count failed validation")
        if previous and (rows < previous[0] or last_month < previous[1]
                         or datetime.fromisoformat(captured) < datetime.fromisoformat(previous[2])):
            raise ValueError("HDB CSV is older or smaller than the existing snapshot")
        conn.execute("DROP TABLE IF EXISTS resale_tx")
        conn.execute("ALTER TABLE resale_staging RENAME TO resale_tx")
        conn.execute("CREATE INDEX ix_tx ON resale_tx(flat_type, month, town)")
        conn.execute("CREATE INDEX ix_tx_month ON resale_tx(month, town)")
        conn.execute("""
            INSERT OR IGNORE INTO address_cache(address_key, geocode_status)
            SELECT DISTINCT address_key, 'pending' FROM resale_tx
        """)
        conn.execute("INSERT OR REPLACE INTO provenance VALUES (?, ?, ?, ?, ?)",
                     (DATASET_ID, DATASET_URL, captured, rows, last_month))
    return Provenance(source_id=DATASET_ID, url=DATASET_URL, retrieved_at=captured,
                      rows=rows, last_month=last_month)


def download_csv(destination: Path) -> None:
    """Use the fixed official dataset ID and stream the full CSV once."""
    response = requests.get(f"{DOWNLOAD_API}/initiate-download", timeout=15)
    response.raise_for_status()
    url = ""
    for attempt in range(4):
        response = requests.get(f"{DOWNLOAD_API}/poll-download", timeout=15)
        response.raise_for_status()
        payload = response.json()
        url = payload.get("data", {}).get("url", "")
        if payload.get("code") == 0 and url.startswith("https://"):
            break
        if attempt < 3:
            time.sleep(1)
    else:
        raise ValueError("data.gov.sg did not return a download link")
    with requests.get(url, timeout=45, stream=True) as response:
        response.raise_for_status()
        with destination.open("wb") as stream:
            for chunk in response.iter_content(chunk_size=256 * 1024):
                stream.write(chunk)


def month_start(last_month: str, months_back: int) -> str:
    year, month = map(int, last_month.split("-"))
    year, index = divmod(year * 12 + month - months_back, 12)
    return f"{year:04d}-{index + 1:02d}"


def lease_reference(today: date) -> float:
    return today.year + today.month / 12


def quantile(values: list[float], fraction: float) -> float:
    """Linear interpolation, matching pandas' default quantile."""
    index = (len(values) - 1) * fraction
    lower = math.floor(index)
    return values[lower] + (values[math.ceil(index)] - values[lower]) * (index - lower)


def search_market(db: Path, profile: BuyerProfile, *, today: date | None = None) -> MarketSummary:
    today = today or singapore_today()
    fields = {"flat_type", "months_back", "max_price", "min_est_lease_years", "town", "avoid_floor"}
    summary = MarketSummary(query=profile.model_dump(include=fields, exclude_none=True), as_of=today.isoformat())
    try:
        with closing(connect_readonly(db)) as conn, conn:
            # Both SELECTs must see the same snapshot if a refresh runs concurrently.
            conn.execute("BEGIN")
            row = conn.execute("SELECT * FROM provenance WHERE source_id = ?", (DATASET_ID,)).fetchone()
            if not row:
                raise ValueError("Missing snapshot")
            summary.source = Provenance(**dict(row))
            end = summary.source.last_month
            start = month_start(end, profile.months_back)
            summary.window = f"{start} to {end}"
            matches = conn.execute("""
                SELECT month, town, address_key, resale_price, lease_commence_year FROM resale_tx
                WHERE month BETWEEN ? AND ?
                  AND (? IS NULL OR flat_type = ?)
                  AND (? IS NULL OR resale_price <= ?)
                  AND (? IS NULL OR 99 - (? - lease_commence_year) >= ?)
                  AND (? IS NULL OR town = ?)
                  AND (? IS NULL OR storey_range != ?)
                ORDER BY town, month DESC, resale_price, address_key
            """, (start, end, profile.flat_type, profile.flat_type, profile.max_price, profile.max_price,
                  profile.min_est_lease_years, lease_reference(today), profile.min_est_lease_years,
                  profile.town, (profile.town or "").strip().upper(), profile.avoid_floor, profile.avoid_floor))
            by_town = defaultdict(list)
            for transaction in matches:
                by_town[transaction["town"]].append(transaction)
    except (sqlite3.Error, ValueError):
        summary.error = "Historical transaction data could not be opened. Run the ingestion command first."
        return summary
    for town, transactions in by_town.items():
        prices = sorted(row["resale_price"] for row in transactions)
        leases = {}
        for row in transactions:
            if len(leases) == 3:
                break
            leases.setdefault(row["address_key"], round(99 - (lease_reference(today) - row["lease_commence_year"]), 1))
        summary.towns.append(TownSummary(town=town, n_tx=len(prices), median=quantile(prices, .5),
                                         p25=quantile(prices, .25), p75=quantile(prices, .75),
                                         latest_month=transactions[0]["month"], sample_blocks=list(leases),
                                         sample_estimated_leases=leases))
    summary.towns.sort(key=lambda town: (town.median, -town.n_tx, town.town))
    return summary
