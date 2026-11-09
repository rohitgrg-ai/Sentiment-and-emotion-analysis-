"""Emoji feature extraction for the experimental models (Models B and C).

Uses only the Python standard library (unicodedata) to turn emojis into words,
so no extra dependency is needed and the output is identical on every machine:
    "really happy today 😭"  ->  "really happy today loudly crying face"
(The third-party `emoji` package gives similar CLDR names; it is not used here
because it is not installed everywhere and preprocessing.py silently skips it
when missing. This module never skips: every emoji gets a name, or "emoji" if
the running Python's Unicode table is too old to know it.)
"""
import re
import unicodedata
from collections import Counter

import numpy as np
from scipy.sparse import csr_matrix

VS16, ZWJ, KEYCAP = "️", "‍", "⃣"
SKIN_TONES = {chr(c) for c in range(0x1F3FB, 0x1F400)}
_EMOJI_RANGES = [(0x1F000, 0x1FAFF), (0x2600, 0x27BF), (0x2B00, 0x2BFF), (0x2300, 0x23FF), (0x2190, 0x21FF)]


def is_emoji_char(ch):
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _EMOJI_RANGES) and ch not in SKIN_TONES


def extract_emojis(text):
    """Return the emojis in text as clusters (keeps ZWJ sequences and skin tones together)."""
    text, clusters, i = str(text), [], 0
    while i < len(text):
        if is_emoji_char(text[i]):
            j = i + 1
            while j < len(text):
                if text[j] in (VS16, KEYCAP) or text[j] in SKIN_TONES:
                    j += 1
                elif text[j] == ZWJ and j + 1 < len(text) and is_emoji_char(text[j + 1]):
                    j += 2
                else:
                    break
            clusters.append(text[i:j])
            i = j
        else:
            i += 1
    return clusters


# Modern CLDR short names (the names the `emoji` package uses) for emojis whose
# official Unicode character name is outdated or misleading. These are names only,
# not emotion labels.
CLDR_NAMES = {
    "\u2764": "red heart", "\U0001F621": "enraged face", "\U0001F52A": "kitchen knife",
    "\U0001F624": "face with steam from nose", "\u2600": "sun", "\U0001F44D": "thumbs up",
    "\U0001F44E": "thumbs down", "\U0001F44F": "clapping hands", "\U0001F64F": "folded hands",
    "\U0001F970": "smiling face with hearts", "\U0001F973": "partying face",
    "\U0001F92F": "exploding head", "\U0001F630": "anxious face with sweat",
    "\U0001F60D": "smiling face with heart eyes", "\U0001F618": "face blowing a kiss",
    "\U0001F4A9": "pile of poo", "\u26C8": "cloud with lightning and rain",
    "\U0001F625": "sad but relieved face", "\U0001F613": "downcast face with sweat",
    "\U0001F605": "grinning face with sweat", "\U0001F606": "grinning squinting face",
    "\U0001F601": "beaming face with smiling eyes", "\U0001F603": "grinning face with big eyes",
    "\U0001F604": "grinning face with smiling eyes", "\U0001F92C": "face with symbols on mouth",
    "\U0001F633": "flushed face", "\U0001F62E": "face with open mouth", "\U0001F62F": "hushed face",
    "\U0001F616": "confounded face", "\U0001F623": "persevering face", "\U0001F494": "broken heart",
    "\U0001F495": "two hearts", "\u2639": "frowning face", "\u263A": "smiling face",
}


def emoji_name(cluster):
    """'😭' -> 'loudly crying face'; ZWJ sequences give the names of their parts."""
    names = []
    for ch in cluster:
        if ch in (VS16, ZWJ, KEYCAP) or ch in SKIN_TONES:
            continue
        if ch in CLDR_NAMES:
            names.append(CLDR_NAMES[ch])
            continue
        try:
            names.append(unicodedata.name(ch).lower().replace("-", " "))
        except ValueError:
            names.append("emoji")
    return " ".join(names) or "emoji"


def strip_emojis(text):
    out = str(text)
    for cluster in sorted(set(extract_emojis(text)), key=len, reverse=True):
        out = out.replace(cluster, " ")
    return re.sub(r"\s+", " ", out).strip()


def collapse_repeats(text):
    """'😂😂😂' -> '😂' (repetition is kept as a separate numeric feature in Model C)."""
    clusters = extract_emojis(text)
    out = str(text)
    for c in sorted(set(clusters), key=len, reverse=True):
        out = re.sub(f"(?:{re.escape(c)}\\s*){{2,}}", c + " ", out)
    return out


def demojize(text):
    """Model B input: emojis replaced inline by their names (repeats collapsed first)."""
    out = collapse_repeats(text)
    for c in sorted(set(extract_emojis(out)), key=len, reverse=True):
        out = out.replace(c, f" {emoji_name(c)} ")
    return re.sub(r"\s+", " ", out).strip()


def emoji_description(text):
    """Model C emoji channel: only the emoji names, one per emoji, e.g. 'loudly crying face skull'."""
    return " ".join(emoji_name(c) for c in extract_emojis(collapse_repeats(text)))


NUMERIC_FEATURE_NAMES = ["log_emoji_count", "emoji_per_token", "has_repeat", "max_repeat_log", "distinct_emojis_log"]


def numeric_emoji_features(texts):
    """Emoji count, frequency, repetition (Model C). Returns a sparse matrix."""
    rows = []
    for t in texts:
        clusters = extract_emojis(t)
        n = len(clusters)
        tokens = max(1, len(strip_emojis(t).split()) + n)
        counts = Counter(clusters)
        max_rep = max(counts.values()) if counts else 0
        rows.append([np.log1p(n), n / tokens, float(max_rep > 1), np.log1p(max_rep), np.log1p(len(counts))])
    return csr_matrix(np.array(rows, dtype=float))
