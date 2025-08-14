import re
import pandas as pd

# PDTB-style discourse connectives
PDTB_CONNECTIVES = {
    "contrast": [
        "but", "however", "yet", "nevertheless", "nonetheless", "on the other hand",
        "in contrast", "by contrast", "still", "instead", "conversely", "whereas", "while"
    ],
    "concession": [
        "although", "though", "even though", "while", "albeit", "granted that",
        "nonetheless", "nevertheless", "despite", "in spite of", "of course"
    ]
}

# Flatten to one dict: marker -> category
CONNECTIVE_CATEGORY = {}
for category, markers in PDTB_CONNECTIVES.items():
    for m in markers:
        CONNECTIVE_CATEGORY[m] = category

# Regex patterns for connectives
CONNECTIVE_PATTERNS = {m: re.compile(r'\b' + re.escape(m) + r'\b', re.IGNORECASE)
                       for m in CONNECTIVE_CATEGORY}

def message_balance_score(text):
    words = text.split()
    total_words = len(words)
    if total_words == 0:
        return 0.0, []

    matched_spans = []

    for marker, pattern in CONNECTIVE_PATTERNS.items():
        for match in pattern.finditer(text):
            marker_start_index = len(text[:match.start()].split())
            after_words = words[marker_start_index:]
            span = []
            for w in after_words:
                span.append(w)
                if re.search(r'[.,;!?]$', w):  # stop at comma OR end punctuation
                    break
            matched_spans.append((marker, CONNECTIVE_CATEGORY[marker], len(span)))

    total_span_words = sum(length for _, _, length in matched_spans)
    score = total_span_words / total_words

    return score, matched_spans

# --- DataFrame version ---
def classify_message_balance(df, text_col='caption'):
    scores = []
    spans_list = []

    for text in df[text_col]:
        score, spans = message_balance_score(str(text))
        scores.append(score)
        spans_list.append(spans)

    return pd.Series(scores)

# --- Example usage ---
data = {
    'caption': [
        "I agree with you, but we must also consider the risks.",
        "While it is true that the economy is growing, some sectors struggle.",
        "Although the plan is solid, some people have doubts.",
        "No discourse markers here, just a simple sentence."
    ]
}

df = pd.DataFrame(data)
scores = classify_message_balance(df)
print(scores)
