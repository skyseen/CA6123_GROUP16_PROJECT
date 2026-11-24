"""Prime OneMap caches and report the independent sample gates."""
import argparse
from contextlib import closing
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

from hdb_advisor.agents import offline_intake
from hdb_advisor.market import connect_readonly, month_start, provenance, search_market
from hdb_advisor.onemap import OneMap, geocode_hdb_address, next_weekday, resolve_destination, route_to_anchor


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/hdb.db"))
    parser.add_argument("--limit", default=20, type=int, help="Use 0 to process the full address set")
    parser.add_argument("--route-date", type=date.fromisoformat)
    parser.add_argument("--destination", help="Commute destination for the sample gate, e.g. 'Raffles Place MRT'")
    parser.add_argument("--demo-brief", action="append", default=[], help="Also cache the blocks a demo brief would route")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be non-negative")
    if args.route_date and args.route_date.weekday() >= 5:
        parser.error("--route-date must be a weekday")
    client = OneMap.from_env()
    if client is None:
        parser.exit(2, "OneMap needs ONEMAP_EMAIL and ONEMAP_PASSWORD in .env.\n")
    source = provenance(args.db)
    if source is None:
        parser.error("Run ingestion first")
    with closing(connect_readonly(args.db)) as conn:
        keys = [row[0] for row in conn.execute("""
            SELECT DISTINCT address_key FROM resale_tx
            WHERE month BETWEEN ? AND ? AND resale_price < ?
            ORDER BY address_key
        """, (month_start(source.last_month, 12), source.last_month, 1_200_000))]
    if args.limit:
        # Spread the sample across the sorted addresses and keep it reproducible.
        n = min(args.limit, len(keys))
        keys = [keys[index * len(keys) // n] for index in range(n)]
    sample = set(keys)
    anchor = None
    if args.destination:
        found = resolve_destination(args.db, args.destination, client)
        print(f"Destination {args.destination!r}: {found.status} {found.label or ''} {found.address or ''}")
        anchor = found.anchor_id
    # Each demo brief is routed to its own destination; ranking needs one block for every matching town.
    jobs = [(key, anchor) for key in keys] if anchor else [(key, None) for key in keys]
    for brief in args.demo_brief:
        profile = offline_intake(brief)
        found = resolve_destination(args.db, profile.destination, client) if profile.destination else None
        print(f"Demo brief {brief!r}: destination {found.label if found else 'none'}")
        market = search_market(args.db, profile)
        jobs += [(town.sample_blocks[0], found.anchor_id if found else None) for town in market.towns if town.sample_blocks]
    matched = 0
    routed = 0
    day = args.route_date or next_weekday()
    for key, target in dict.fromkeys(jobs):
        address = geocode_hdb_address(args.db, key, client)
        matched += key in sample and target == anchor and address.status == "matched"
        route = (route_to_anchor(args.db, key, target, client, day, allow_proxy=False)
                 if address.status == "matched" and target else None)
        routed += key in sample and target == anchor and route is not None and route.duration_min is not None
        print(key, address.status, route.status if route else "")
    print(f"Sample unique geocodes {matched}/{len(sample)}. Valid {day} sample transit routes {routed}/{len(sample)}.")


if __name__ == "__main__":
    main()
