"""Keep user identifiers and untrusted instructions outside model prompts."""
import re

NRIC = r"\b[STFGM]\d{7}[A-Z]\b"
PHONE = r"(?<![\w$])(?:\+65[ -]?)?[689]\d{3}[ -]?\d{4}(?!\w)"
CARD = r"(?<![\d$])(?:\d[ -]?){12,18}\d(?!\d)"
EMAIL = r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"
PATTERNS = {"nric": NRIC, "phone": PHONE, "credit_card": CARD, "email": EMAIL}
INJECTION = re.compile(
    r"(?:ignore|override|bypass|disregard|forget)\b.{0,55}\b(?:system|instructions?|rules|guardrails?)"
    r"|\b(?:reveal|print|show)\b.{0,26}\b(?:system prompt|api key|secret token)\b"
    r"|\b(?:drop|truncate|alter)\s+table\b"
    r"|\b(?:mark|declare|label)\b.{0,35}\b(?:me|everyone|user)\b.{0,22}\beligible\b",
    re.I | re.S,
)
ADVICE = re.compile(
    r"\b(?:should (?:i|we) (?:buy|sell|invest|wait)|predict\b|will (?:hdb |resale |flat )?prices? "
    r"(?:rise|fall|go)|investment advice|buy now or wait|market timing)\b", re.I,
)
AVAILABILITY = re.compile(r"\b(?:for sale|still available|current listings?|asking prices?|available now)\b", re.I)


def redact(text: str) -> tuple[str, list[str]]:
    events = []
    for name, pattern in PATTERNS.items():
        text, count = re.subn(pattern, f"[REDACTED_{name.upper()}]", text, flags=re.I)
        if count:
            events.append(f"pii_{name}")
    return text, events


def inspect_input(text: str, enabled: bool = True) -> tuple[str, list[str]]:
    text = text.strip()[:4000]
    if not enabled:
        return text, []
    text, events = redact(text)
    if INJECTION.search(text):
        events.append("direct_instruction")
    if ADVICE.search(text):
        events.append("investment_advice")
    if AVAILABILITY.search(text):
        events.append("live_listing_request")
    return text, events


OVERCLAIM = re.compile(
    r"\b(?:for sale|on sale|available (?:now|for purchase)|still available|currently (?:available|listed)"
    r"|asking prices?|guaranteed?|definitely|should (?:buy|sell|wait)|will (?:rise|fall|increase|drop|appreciate)"
    r"|you (?:are|will be) eligible|you qualify|will receive|approved)\b", re.I,
)
NUMBER = re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)(?:\s*(k|m|million|thousand)\b)?", re.I)
SCALE = {"": 1, "k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6}
# Small counts ("three towns", "two steps") and list positions need no tool source.
FREE_NUMBERS = frozenset(range(13))


def numbers_in(text: str) -> list[float]:
    """Extract written figures, expanding S$518k-style suffixes."""
    values = []
    for match in NUMBER.finditer(text):
        try:
            value = float(match[1].replace(",", ""))
        except ValueError:
            continue
        values.append(value * SCALE[(match[2] or "").lower()])
    return values


def ungrounded(text: str, allowed) -> list[float]:
    """Return figures that do not match a tool value (rounding to the nearest unit or S$1,000 is allowed)."""
    allowed = list(allowed)

    def known(value):
        return value in FREE_NUMBERS or any(
            abs(value - a) <= max(0.51, 500 if abs(a) >= 10_000 else 0, abs(a) * 0.001) for a in allowed)
    return [value for value in numbers_in(text) if not known(value)]


def check_narrative(text: str, allowed, limit: int = 700) -> list[str]:
    """Deterministic critic rules for model-written prose. An empty list means the text passed."""
    issues = []
    if len(text) > limit:
        issues.append(f"Keep the text under {limit} characters.")
    if re.search(r"[<>{}\[\]]|https?://|www\.", text):
        issues.append("Remove links and markup. The renderer adds the sources.")
    claim = OVERCLAIM.search(text)
    if claim:
        issues.append(f"Remove the overclaiming phrase '{claim[0]}'. Transactions are completed sales, not listings or decisions.")
    if redact(text)[1]:
        issues.append("Remove personal identifiers.")
    if INJECTION.search(text):
        issues.append("Remove instruction-like text.")
    invented = ungrounded(text, allowed)
    if invented:
        shown = ", ".join(f"{value:,.10g}" for value in invented[:4])
        issues.append(f"These figures are not in the tool data: {shown}. Copy figures exactly and do not compute new ones.")
    return issues


def safe_narrative(text: str) -> bool:
    return (len(text) <= 400 and not re.search(r"\d|[$€£%<>{}\[\]\r\n]", text)
            and not re.search(
                r"\b(?:available|for sale|listing|eligible|guaranteed|approved?|will rise|will fall|should buy)\b",
                text, re.I,
            ) and not redact(text)[1] and not INJECTION.search(text))
