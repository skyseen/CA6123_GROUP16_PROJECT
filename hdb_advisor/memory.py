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


# ---------------------------------------------------------------------------------------------------------
# Long-term memory across conversations (per signed-in user)
#
# Short-term memory is the LangGraph checkpointer: one thread per conversation, holding the full state.
# Long-term memory is the LangGraph store, one namespace per user:
#   ("prefs", user)          "profile"    towns, destination, floor  (when the buyer ticks "remember")
#                            "household"  household facts            (only with explicit consent)
#   ("memories", user)       <id>         deterministic summary of each completed search
#   ("conversations", user)  <thread>     title and timestamps, so a conversation can be resumed
#   ("examples", user, role) <hash>       approved corrections used as few-shot examples
# Recall is keyword plus local-embedding similarity over the user's own memories (no external service).
# ---------------------------------------------------------------------------------------------------------
import math
import re
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

HOUSEHOLD_MEMORY_FIELDS = ("citizenship", "first_timer", "near_parents", "household_income_monthly", "buyer_ages",
                           "continuous_employment_12m", "prior_phg")
MAX_MEMORIES = 50


def now_sg() -> datetime:
    return datetime.now(ZoneInfo("Asia/Singapore"))


def remember_household(store, user_id, profile):
    """Store household facts the buyer explicitly agreed to keep. Unknown facts are not stored."""
    if store is None:
        return
    values = {k: v for k, v in profile.model_dump(include=set(HOUSEHOLD_MEMORY_FIELDS)).items() if v not in (None, [])}
    if values:
        store.put(("prefs", user_id), "household", {**values, "saved_at": now_sg().isoformat(timespec="minutes")})


def household_memory(store, user_id) -> dict:
    if store is None:
        return {}
    item = store.get(("prefs", user_id), "household")
    return {k: v for k, v in (item.value if item else {}).items() if k in HOUSEHOLD_MEMORY_FIELDS}


def search_summary(state) -> str:
    """A deterministic, redacted one-paragraph memory of a completed search (no model call)."""
    from .guards import redact
    from .report import commute_headline, ranked_towns

    def money(v):
        return f"S${v:,.0f}"
    p = state.get("profile") or {}
    asked = []
    if p.get("flat_type"):
        asked.append(p["flat_type"].title())
    if p.get("max_price"):
        asked.append(f"budget up to {money(p['max_price'])}")
    if p.get("towns"):
        asked.append("towns " + " and ".join(t.title() for t in p["towns"]))
    if p.get("destination"):
        asked.append(f"commute to {p['destination']}" + (f" within {p['max_commute_min']} min" if p.get("max_commute_min") else ""))
    if state.get("market"):
        asked.append(f"sales from the last {p.get('months_back', 12)} months")
    found = []
    ranked = ranked_towns(state)
    if state.get("market"):
        if ranked:
            top = ", ".join(f"{t['town'].title()} (median {money(t['median'])})" for t, _ in ranked[:3])
            found.append(f"{len(ranked)} towns matched; first in the shortlist: {top}")
        else:
            found.append("no transactions matched")
    if state.get("routes") and p.get("max_commute_min"):
        found.append(commute_headline(ranked, p["max_commute_min"]).rstrip("."))
    loan = state.get("affordability") or {}
    if loan.get("status") == "illustrative":
        found.append(f"illustrative HDB loan ceiling {money(loan['loan_ceiling'])} over {loan['tenure_years']} years")
    elif loan.get("status"):
        found.append("loan estimate needed more details or HDB review")
    grants = state.get("grants") or []
    if grants:
        may = [f"{g['name']} {g['amount_text']}" for g in grants if g["status"] == "may_apply"]
        found.append(f"grants: {len(may)} may apply" + (f" ({', '.join(may)})" if may else "")
                     + f", {sum(g['status'] == 'need_more_info' for g in grants)} need more information")
    when = now_sg().strftime("%d %b %Y")
    text = f"On {when} the buyer researched {', '.join(asked) or 'HDB resale options'}. Result: {'; '.join(found) or 'no tool results'}."
    return redact(text)[0]


def record_search(store, user_id, thread_id, state) -> str | None:
    if store is None:
        return None
    key = uuid.uuid4().hex[:16]
    store.put(("memories", user_id), key, {"kind": "search", "text": search_summary(state), "thread": thread_id,
                                           "at": now_sg().isoformat(timespec="minutes")})
    # Keep the newest memories only.
    items = sorted(store.search(("memories", user_id), limit=500), key=lambda i: i.value.get("at", ""), reverse=True)
    for old in items[MAX_MEMORIES:]:
        store.delete(("memories", user_id), old.key)
    return key


def _cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def recall_memories(store, user_id, query: str, k: int = 3) -> tuple[list[dict], bool]:
    """Return up to k of this user's past-search memories relevant to the query, and whether any were dropped.

    Relevance mixes keyword overlap with local-embedding similarity. Memories that look like instructions are
    dropped (they are data, never commands).
    """
    from .guards import INJECTION
    from .policy import words

    if store is None or not query.strip():
        return [], False
    embed = LocalEmbeddings()
    query_vector, query_words = embed.embed_query(query), words(query)
    scored, dropped = [], False
    for item in store.search(("memories", user_id), limit=500):
        text = item.value.get("text", "")
        if INJECTION.search(text):
            dropped = True
            continue
        overlap = len(query_words & words(text)) / max(1, len(query_words))
        score = 0.6 * overlap + 0.4 * _cosine(query_vector, embed.embed_query(text))
        if score > 0.15:
            scored.append((score, item.value.get("at", ""), {"key": item.key, **item.value}))
    scored.sort(key=lambda row: (row[0], row[1]), reverse=True)
    return [row[2] for row in scored[:k]], dropped


def record_conversation(store, user_id, thread_id, brief: str):
    """Register or touch a conversation so the buyer can find and resume it later."""
    from .guards import redact

    if store is None:
        return
    item = store.get(("conversations", user_id), thread_id)
    stamp = now_sg().isoformat(timespec="minutes")
    title = (item.value["title"] if item else re.sub(r"\s+", " ", redact(brief)[0]).strip()[:70]) or "Untitled search"
    store.put(("conversations", user_id), thread_id,
              {"title": title, "created": item.value["created"] if item else stamp, "updated": stamp})


def list_conversations(store, user_id) -> list[dict]:
    if store is None:
        return []
    items = store.search(("conversations", user_id), limit=200)
    return sorted(({"thread": i.key, **i.value} for i in items), key=lambda c: c["updated"], reverse=True)


def list_memory(store, user_id) -> dict:
    """Everything the assistant remembers about this user, for the "My memory" panel and export."""
    if store is None:
        return {"preferences": {}, "household": {}, "searches": []}
    searches = sorted(({"key": i.key, **i.value} for i in store.search(("memories", user_id), limit=500)),
                      key=lambda m: m.get("at", ""), reverse=True)
    return {"preferences": preferences(store, user_id), "household": household_memory(store, user_id),
            "searches": searches}


def forget(store, user_id, kind: str, key: str | None = None):
    if store is None:
        return
    if kind == "preferences":
        store.delete(("prefs", user_id), "profile")
    elif kind == "household":
        store.delete(("prefs", user_id), "household")
    elif kind == "search" and key:
        store.delete(("memories", user_id), key)


def forget_everything(store, user_id):
    """Delete every long-term memory for this user: preferences, household facts, searches, examples, history."""
    if store is None:
        return
    for namespace in (("prefs", user_id), ("memories", user_id), ("conversations", user_id),
                      ("examples", user_id, "intake"), ("examples", user_id, "reports"), ("examples", user_id, "router")):
        for item in store.search(namespace, limit=1000):
            store.delete(namespace, item.key)
