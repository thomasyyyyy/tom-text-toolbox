import pandas as pd
import readability
from tqdm import tqdm

# Constants for readability method and subkey
READABILITY_METHOD = "Flesch-Kincaid"
KINCAID_KEY = "Kincaid"

def parse_readability_measures(measures: dict, method: str = READABILITY_METHOD) -> dict:
    """
    Extracts the specified readability score from the 'readability grades' section.
    """
    grades = measures.get("readability grades", {})
    if method == READABILITY_METHOD and KINCAID_KEY in grades:
        return {f"readability_{KINCAID_KEY.lower()}": grades[KINCAID_KEY]}
    return {}

def get_readability_safe(text: str) -> dict | None:
    """
    Safely computes readability measures for a single text.
    Returns None if computation fails.
    """
    try:
        measures = readability.getmeasures(text)
        return parse_readability_measures(measures)
    except ValueError:
        return None

def readability_scores(captions: pd.Series) -> pd.DataFrame:
    """
    Computes readability scores for a pandas Series of captions.
    Returns a DataFrame aligned with the original Series index.
    """
    tqdm.pandas(desc="Computing readability")
    results = captions.progress_apply(get_readability_safe)

    # Keep only successful results
    valid_results = results[results.notnull()]

    # Convert to DataFrame and align with original index
    df = pd.DataFrame.from_records(valid_results)
    df = df.reindex(captions.index)

    skipped = len(captions) - len(valid_results)
    success_rate = (len(valid_results) / len(captions)) * 100
    print(f"Skipped {skipped} rows, success rate: {success_rate:.2f}%")

    return df

if __name__ == "__main__":
    captions = pd.Series([
        "This is a test.",
        "Another sentence!",
        "Third one."
    ])
    df = readability_scores(captions)
    print(df)
