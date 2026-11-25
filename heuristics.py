"""Simple, explainable red-flag checks that sit next to the ML prediction."""
import re

SENSATIONAL = [
    "shocking", "you won't believe", "exposed", "secret", "miracle", "hoax",
    "they don't want you to know", "banned", "cover-up", "cover up", "bombshell",
    "wake up", "share before", "doctors hate", "100% proof", "leaked",
]


def red_flags(text: str) -> list:
    flags = []
    words = re.findall(r"[A-Za-z']+", text)
    caps = [w for w in words if len(w) > 3 and w.isupper()]
    if words and len(caps) / len(words) > 0.10:
        flags.append("Lots of ALL-CAPS words (emotional / shouting tone)")
    if text.count("!") >= 2:
        flags.append("Multiple exclamation marks")
    low = text.lower()
    hits = [s for s in SENSATIONAL if s in low]
    if hits:
        flags.append("Sensational phrases: " + ", ".join(hits[:4]))
    if not re.search(r"according to|said|told|reported|study|report|ministry|official|police|court", low):
        flags.append("No source or attribution mentioned (who said this?)")
    if len(words) < 8:
        flags.append("Very short text - the prediction may be unreliable")
    return flags
