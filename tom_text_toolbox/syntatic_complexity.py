from tokenizer import process_caption
import readability
import pandas as pd
from tqdm import tqdm

def parse_readability_measures(measures):
    parsed_dict = {}
    for key, value in measures.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                parsed_dict[f"{key}_{subkey}"] = subvalue
        else:
            parsed_dict[key] = value
    return parsed_dict

def readability_scores(captions: list | pd.Series) -> dict:
    new_columns = {}
    skips = 0

    # Ensure tqdm works with index if it's a Series
    if isinstance(captions, pd.Series):
        iterable = captions.items()
    else:
        iterable = enumerate(captions)

    for index, caption in tqdm(iterable, desc="Processing Readability Scores..."):
        try:
            measures = readability.getmeasures(caption)
            parsed_measures = parse_readability_measures(measures)
        except ValueError as e:
            print(f"Error processing row {index}: {e}")
            skips += 1
            continue

        # Initialize keys only once, when we get the first valid parsed_measures
        if not new_columns:
            new_columns = {key: [] for key in parsed_measures.keys()}

        for key in new_columns:
            value = parsed_measures.get(key, float('nan'))
            new_columns[key].append(value)

    # Pad all lists to match the length of input
    for key, values in new_columns.items():
        if len(values) < len(captions):
            values.extend([float('nan')] * (len(captions) - len(values)))

    success_rate = ((len(captions) - skips) / len(captions)) * 100
    print(f"The function skipped {skips} rows, leaving a success rate of {success_rate:.2f}%")

    return new_columns

if __name__ == "__main__":
    captions = ["This is a test.", "Another sentence!", "Third one."]
    results = readability_scores(captions)

    # Convert to DataFrame for funsies
    df = pd.DataFrame(results)
    print(df.head())
