"""Read approved examples and remember explicitly confirmed preferences."""
import hashlib
import json

from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.vectorstores import InMemoryVectorStore

from .policy import LocalEmbeddings, ROOT

PREF_FIELDS = {"destination", "avoid_floor", "towns"}


def preferences(store, user_id):
    if store is None:
        return {}
    item = store.get(("prefs", user_id), "profile")
    return item.value if item else {}


def remember(store, user_id, profile):
    if store is None:
        return
    values = {k: v for k, v in profile.model_dump(include=PREF_FIELDS).items() if v not in (None, [])}
    if values:
        store.put(("prefs", user_id), "profile", values)
    else:
        # Remembering an empty set means the buyer cleared every preference.
        store.delete(("prefs", user_id), "profile")


def record_example(store, user_id, role, example):
    if store is None:
        return
    encoded = json.dumps(example, sort_keys=True, ensure_ascii=False)
    key = hashlib.sha256(encoded.encode()).hexdigest()
    store.put(("examples", user_id, role), key, example)


def examples_for(store, user_id, role, query):
    file = ROOT / "examples" / f"{role}.jsonl"
    examples = [json.loads(line) for line in file.read_text().splitlines() if line.strip()] if file.is_file() else []
    if store is not None:
        examples.extend(item.value for item in store.search(("examples", user_id, role), limit=20))
    if not examples:
        return []
    selector = SemanticSimilarityExampleSelector.from_examples(
        examples, LocalEmbeddings(), InMemoryVectorStore, k=3, input_keys=["input"]
    )
    return selector.select_examples({"input": query})
