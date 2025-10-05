import pandas as pd
from typing import Optional, Union
from tqdm import tqdm

import pandas as pd
from typing import Optional, Union, List
from tqdm import tqdm

# -----------------------------
# Load familiarity dictionary
# -----------------------------
def load_familiarity_dict(dict_path: str = "tom_text_toolbox/linguistic_dictionaries/fam_peatzold_dict.csv") -> dict:
    df = pd.read_csv(dict_path)
    df = df[df["Word"].apply(lambda x: isinstance(x, str))]
    return dict(zip(df["Word"].str.lower(), df["Familiarity"]))

# -----------------------------
# Score a single caption
# -----------------------------
def score_caption(tokens_or_text: Union[str, List[str]], fam_dict: dict) -> Optional[float]:
    # Handle NaNs / empty values
    if tokens_or_text is None or (isinstance(tokens_or_text, float) and pd.isna(tokens_or_text)):
        return None
    
    # If it's already tokenized
    if isinstance(tokens_or_text, list):
        tokens = [str(t).lower() for t in tokens_or_text]
    else:
        # Assume it's a string
        tokens = str(tokens_or_text).lower().split()
    
    if not tokens:
        return None
    
    # Compute familiarity score
    scores = [fam_dict[token] for token in tokens if token in fam_dict]
    return round(sum(scores) / len(scores), 2) if scores else None

# -----------------------------
# Apply scoring to a pandas Series
# -----------------------------
def classify_familiarity(captions: pd.Series, show_progress: bool = False) -> pd.Series:
    fam_dict = load_familiarity_dict()
    if show_progress:
        tqdm.pandas(desc="Calculating familiarity scores")
        return captions.progress_apply(lambda x: score_caption(x, fam_dict))
    else:
        return captions.apply(lambda x: score_caption(x, fam_dict))


# -----------------------------
# Main execution
# -----------------------------
if __name__ == "__main__":
    df = pd.read_csv("tom_text_toolbox/text_data_TEST.csv")
    
    # This will now work whether "caption" is tokenized (list) or a raw string
    df["familiarity_score"] = classify_familiarity(df["caption"])
    print(df[["caption", "familiarity_score"]].head(10))
