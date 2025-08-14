import pandas as pd
import itertools
import numpy as np
import re
from Levenshtein import distance as levenshtein_distance

def classify_levdist(caption):
    # Extract only alphabetic words, lowercase
    words = re.findall(r"\b[a-z]+\b", caption.lower())
    
    if len(words) < 2:
        return 0  # Skip captions with fewer than 2 words
    
    distances = []
    # Compare every pair of words in the caption
    for w1, w2 in itertools.combinations(words, 2):
        distances.append(levenshtein_distance(w1, w2))
    
    return round(np.mean(distances),2) if distances else 0

# Example dataframe
df = pd.DataFrame({
    "caption": [
        "Cat bat rat",
        "Love dove move",
        "Quick brown fox jumps over the lazy dog",
        "Word"
    ]
})

# Apply to each caption
df["avg_letter_edit_distance"] = df["caption"].apply(classify_levdist)

print(df)
