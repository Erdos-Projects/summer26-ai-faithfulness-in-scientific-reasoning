import re
_PATTERNS = [r"is:?\s*\*{0,2}\s*([0-5])\b", r"level\s+is\s*([0-5])\b", r"score:?\s*([0-5])\b"]

def parse_score(text):
    if not text:
        return None
    for pat in _PATTERNS:
        m = re.findall(pat, text, flags=re.IGNORECASE)
        if m:
            return int(m[-1])
    return None
