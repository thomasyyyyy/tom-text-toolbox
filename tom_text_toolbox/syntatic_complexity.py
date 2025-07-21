import pandas as pd
from tokenizer import tokenize_caption
import readability
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map


def parse_readability_measures(measures, method: str = "Flesch-Kincaid"):
    parsed_dict = {}
    for key, value in measures.items():
        if isinstance(value, dict) and key == "readability grades":
            for subkey, subvalue in value.items():
                if method == "Flesch-Kincaid" and subkey == "Kincaid":
                    parsed_dict[(f"readability_{subkey}").lower()] = subvalue
                else:
                    parsed_dict[(f"readability_{subkey}").lower()] = subvalue
    return parsed_dict


def get_readability_safe(text):
    try:
        measures = readability.getmeasures(text)
        return parse_readability_measures(measures)
    except ValueError as e:
        return None  # Or return empty dict


def readability_scores(captions: pd.Series) -> pd.DataFrame:
    tqdm.pandas(desc="Computing readability")

    results = captions.progress_apply(get_readability_safe)

    # Filter out None (failed results)
    valid_results = results[results.notnull()]

    # Expand into DataFrame
    df = pd.json_normalize(valid_results)

    # Reindex to original (to align with input if you need to merge)
    df = df.reindex(captions.index)

    success_rate = (len(valid_results) / len(captions)) * 100
    print(f"The function skipped {len(captions) - len(valid_results)} rows, leaving a success rate of {success_rate:.2f}%")

    return df

if __name__ == "__main__":
    captions = pd.Series(["This is a test.", "Another sentence!", "Third one."])
    df = readability_scores(captions)
    print(df.head())
