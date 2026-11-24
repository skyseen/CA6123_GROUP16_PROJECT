"""Build a reproducible source archive with the public transaction snapshot."""
import argparse
from contextlib import closing
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

from hdb_advisor.market import connect_readonly

ROOT = Path(__file__).resolve().parent.parent
FILES = (".gitignore", ".env.example", ".dockerignore", "README.md", "PHASE_SUMMARY.md", "requirements.lock",
         "requirements-app.txt", "requirements-observability.txt", "walkthrough.ipynb", "streamlit_app.py",
         "Dockerfile", "docker-compose.yml", ".streamlit/config.toml",
         "docs/AGENT_ARCHITECTURE.md")
SOURCE_DIRS = ("hdb_advisor", "scripts", "evals", "config", "examples")
SOURCE_EXTENSIONS = {".py", ".jsonl", ".yaml"}


# Only these commute destinations, and only these phrasings of them, ship in the submission database.
# Destinations typed by buyers in live sessions (and their routes) stay on the machine that ran them.
DEMO_DESTINATIONS = {
    "NTU_NORTH_SPINE": ("NTU", "NTU NORTH SPINE", "NANYANG TECHNOLOGICAL UNIVERSITY", "NORTH SPINE"),
    "DEST_048618": ("RAFFLES PLACE MRT", "RAFFLES PLACE MRT STATION"),
}


def scrub_private_destinations(db: Path) -> None:
    """Keep public transactions, public HDB block geocodes and the listed demo destinations only."""
    anchors = list(DEMO_DESTINATIONS)
    aliases = [alias for names in DEMO_DESTINATIONS.values() for alias in names]
    marks = lambda items: ", ".join("?" for _ in items)
    with closing(sqlite3.connect(db)) as conn:
        with conn:
            conn.execute(f"DELETE FROM destination_cache WHERE query_key NOT IN ({marks(aliases)})", aliases)
            conn.execute(f"DELETE FROM anchors WHERE anchor_id NOT IN ({marks(anchors)})", anchors)
            conn.execute(f"DELETE FROM route_cache WHERE anchor_id NOT IN ({marks(anchors)})", anchors)
        conn.execute("VACUUM")  # Drop the deleted rows from free pages as well.


def package(output: Path) -> None:
    snapshot = ROOT / "data/hdb.db"
    if not snapshot.is_file():
        raise FileNotFoundError("Ingest data/hdb.db before packaging")
    files = [ROOT / name for name in FILES]
    for folder in SOURCE_DIRS:
        files.extend(path for path in (ROOT / folder).rglob("*") if path.suffix in SOURCE_EXTENSIONS)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temp:
        db_copy = Path(temp) / "hdb.db"
        with closing(connect_readonly(snapshot)) as source, closing(sqlite3.connect(db_copy)) as copy:
            source.backup(copy)
        scrub_private_destinations(db_copy)
        draft = Path(temp) / output.name
        with ZipFile(draft, "w", compression=ZIP_DEFLATED) as archive:
            for file in sorted(files):
                archive.write(file, file.relative_to(ROOT))
            archive.write(db_copy, "data/hdb.db")
        draft.replace(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", type=int)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dist")
    args = parser.parse_args()
    if args.group is not None and args.group < 1:
        parser.error("--group must be positive")
    output_name = f"ca6123_group{args.group}.zip" if args.group else "ca6123_hdb_source.zip"
    output = args.output_dir / output_name
    package(output)
    print(output)


if __name__ == "__main__":
    main()
