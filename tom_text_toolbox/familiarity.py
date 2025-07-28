import pandas as pd
from typing import Optional

def load_familiarity_dict(dict_path: str) -> dict:
    df = pd.read_csv(dict_path)
    return dict(zip(df["Word"].str.lower(), df["Familiarity"]))

def score_caption(text: str, fam_dict: dict) -> Optional[float]:
    tokens = str(text).lower().split()
    scores = [fam_dict[token] for token in tokens if token in fam_dict]
    if scores:
        return round(sum(scores) / len(scores), 2)
    else:
        return None

def familiarity_scores(captions: pd.Series, fam_dict: dict) -> pd.Series:
    return captions.apply(lambda x: score_caption(x, fam_dict))


# Example usage:
# fam_dict = load_familiarity_dict("../dictionaries/peatzold_dict.csv")
# df["Familiarity Score"] = familiarity_score(df["cleaned_caption"], fam_dict)
