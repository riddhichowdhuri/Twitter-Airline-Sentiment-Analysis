"""
clean_text.py
Small helper module: turns raw tweet text into cleaned text ready for
vectorising. Kept separate so both the EDA script and the model script
use exactly the same cleaning logic.
"""

import re

URL_RE = re.compile(r"http\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_SYMBOL_RE = re.compile(r"#")
NON_ALPHA_RE = re.compile(r"[^a-zA-Z\s]")
MULTI_SPACE_RE = re.compile(r"\s+")


def clean_tweet(text: str) -> str:
    """Lowercase a tweet and strip URLs, @mentions, punctuation/numbers,
    and extra whitespace. Hashtag symbols are removed but the word after
    a hashtag is kept, since it usually carries sentiment (e.g. #delayed).
    """
    text = str(text).lower()
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HASHTAG_SYMBOL_RE.sub("", text)
    text = NON_ALPHA_RE.sub(" ", text)
    text = MULTI_SPACE_RE.sub(" ", text).strip()
    return text
