import pandas as pd
import itertools
import numpy as np
import re
from Levenshtein import distance as levenshtein_distance

def classify_levdist(captions: pd.Series) -> pd.Series:
    results = []
    for caption in captions:
        if not isinstance(caption, str):
            results.append(0)
            continue

        # Tokenize, keeping numbers and symbols
        words = re.findall(r"[^\s]+", caption.lower())

        if len(words) < 2:
            results.append(0)
            continue

        distances = [
            levenshtein_distance(w1, w2)
            for w1, w2 in itertools.combinations(words, 2)
        ]
        results.append(round(np.mean(distances), 2) if distances else 0)

    return pd.Series(results, index=captions.index)


if __name__ == "__main__":
    df = pd.DataFrame({
        "caption": [
            "Cat bat rat",
            "Love dove move",
            "Quick brown fox jumps over the lazy dog",
            "Word",
            "This has numbers 123 and symbols #hashtag",
            None
        ]
    })

    df["avg_letter_edit_distance"] = classify_levdist(df["caption"])
    print(df)
