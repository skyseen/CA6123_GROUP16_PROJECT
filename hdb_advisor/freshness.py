"""Check the official source once and publish refreshes only after approval."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import sqlite3

import requests
import yaml

from .market import METADATA_API, download_csv, ingest_csv, provenance
from .policy import ROOT


def check_freshness(db: Path, online: bool, *, now=None) -> dict:
    try:
        source = provenance(db)
    except (sqlite3.Error, ValueError):
        source = None
    result = {"source": source.model_dump() if source else None, "status": "unknown", "note": ""}
    if source is None:
        return {**result, "note": "The local transaction snapshot is missing."}
    if not online:
        return {**result, "status": "offline", "note": "Freshness check skipped in offline mode."}
    config = yaml.safe_load((ROOT / "config/freshness.yaml").read_text())
    try:
        response = requests.get(METADATA_API, timeout=config["metadata_timeout_seconds"])
        response.raise_for_status()
        payload = response.json()
        if payload["code"] != 0:
            raise ValueError("Invalid metadata response")
        remote = datetime.fromisoformat(payload["data"]["lastUpdatedAt"])
        captured = datetime.fromisoformat(source.retrieved_at)
        age = (now or datetime.now(timezone.utc)) - captured
        stale = remote > captured and age > timedelta(days=config["max_age_days"])
        return {**result, "status": "stale" if stale else "current", "age_days": age.days}
    except (requests.RequestException, KeyError, TypeError, ValueError):
        return {**result, "note": "Source freshness could not be checked. Using the local snapshot."}


def refresh_data(db: Path, previous: dict) -> dict:
    try:
        captured = datetime.now(timezone.utc).isoformat()
        with TemporaryDirectory() as temp:
            csv = Path(temp) / "resale.csv"
            download_csv(csv)
            new = ingest_csv(db, csv, retrieved_at=captured)
        rows_added = new.rows - previous["source"]["rows"]
        return {"status": "refreshed", "source": new.model_dump(),
                "note": f"Transaction snapshot refreshed. {rows_added:,} rows added."}
    except (requests.RequestException, OSError, ValueError, sqlite3.Error):
        return {**previous, "status": "failed",
                "note": "The refresh failed validation or could not download. The previous snapshot was retained."}
