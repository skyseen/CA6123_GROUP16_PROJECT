"""Build the offline policy index and optionally capture official HTML sections."""
import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup

from hdb_advisor.policy import curated_corpus, index_rules
from hdb_advisor.state import RuleChunk


def archive_sections(output: Path) -> list[dict]:
    """Capture each official page independently and report the status of every source."""
    pages = {chunk.source_url: chunk.title for chunk in curated_corpus()}
    chunks, statuses = [], []
    snapshot_date = datetime.now(ZoneInfo("Asia/Singapore")).date().isoformat()
    for page_index, (url, title) in enumerate(pages.items()):
        status = {"url": url, "title": title, "sections": 0}
        try:
            response = requests.get(url, timeout=12)
            status["http_status"] = response.status_code
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            content = soup.find("main") or soup.find("article")
            if content is None or not content.find(["h2", "h3"]):
                raise ValueError("No policy headings found")
            for index, heading in enumerate(content.find_all(["h2", "h3", "h4"])):
                paragraphs = []
                for sibling in heading.next_siblings:
                    if getattr(sibling, "name", None) in ("h2", "h3", "h4"):
                        break
                    if getattr(sibling, "get_text", None):
                        paragraphs.append(sibling.get_text(" ", strip=True))
                text = " ".join(paragraphs).strip()
                if text:
                    chunks.append(RuleChunk(id=f"{page_index}-{index}", title=title,
                                            section=heading.get_text(" ", strip=True), source_url=url,
                                            snapshot_date=snapshot_date, text=text))
                    status["sections"] += 1
            status["result"] = "captured"
        except (requests.RequestException, ValueError) as exc:
            status["result"] = f"failed: {type(exc).__name__}"
        statuses.append(status)
        print(f"{status['result']:<28} HTTP {status.get('http_status', '-')}  {status['sections']:>2} sections  {url}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"snapshot_date": snapshot_date, "sources": statuses,
                                  "chunks": [c.model_dump() for c in chunks]}, indent=2, ensure_ascii=False) + "\n")
    if not chunks:
        print("No sections captured. The curated snapshot in config/ remains in use.")
    return statuses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path("data/chroma"))
    parser.add_argument("--archive-html", type=Path)
    args = parser.parse_args()
    store = index_rules(args.index)
    print(f"Indexed {len(store.get()['ids'])} policy sections.")
    if args.archive_html:
        archive_sections(args.archive_html)


if __name__ == "__main__":
    main()
