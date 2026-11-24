"""Load an official local CSV or download the fixed data.gov.sg dataset."""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from hdb_advisor.market import download_csv, ingest_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/hdb.db"))
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--expected-rows", type=int)
    args = parser.parse_args()
    with TemporaryDirectory() as temp:
        csv_path = args.csv or Path(temp) / "resale.csv"
        if args.csv is None:
            download_csv(csv_path)
        result = ingest_csv(args.db, csv_path, expected_rows=args.expected_rows)
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
