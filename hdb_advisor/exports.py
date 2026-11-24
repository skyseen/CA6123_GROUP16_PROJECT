"""Save an approved shortlist atomically and idempotently."""
import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile


def save_research_shortlist(directory: Path, report: str, thread: str) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(f"{thread}\n{report}".encode()).hexdigest()[:20]
    output = directory / f"shortlist-{key}.json"
    if output.is_file():
        return str(output)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory, delete=False) as stream:
        staged = Path(stream.name)
        try:
            json.dump({"report": report}, stream, indent=2, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
            staged.replace(output)
        finally:
            staged.unlink(missing_ok=True)
    return str(output)
