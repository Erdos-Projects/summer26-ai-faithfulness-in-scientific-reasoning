"""Extract the yes/no verdict from a subagent's final text."""
import re

_ANS = re.compile(r"answer\s*:?\s*\*{0,2}\s*(yes|no)\b", re.IGNORECASE)


def parse_answer(text):
    """Last explicit 'Answer: yes|no' -> 'yes'/'no'; None if no such line."""
    if not text:
        return None
    m = _ANS.findall(text)
    return m[-1].lower() if m else None
