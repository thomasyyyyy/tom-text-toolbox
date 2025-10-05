import pandas as pd
from tqdm import tqdm

def classify_whissell_scores(captions: pd.Series, dictionary: str|pd.DataFrame = "tom_text_toolbox/linguistic_dictionaries/whissell_dict.csv") -> pd.DataFrame:
    """
    Calculate Whissell scores for a series of captions and return as a dictionary of Series.

    Parameters:
        captions (pd.Series): Series of caption strings.
        dictionary (pd.DataFrame): Whissell dictionary with 'pleas', 'activ', 'image' columns, indexed by 'word'.

    Returns:
        dict[str, pd.Series]: Dictionary with keys:
            - 'mean_pleasant'
            - 'mean_active'
            - 'mean_image'
    """
    if isinstance(dictionary, str):
        dictionary = pd.read_csv(dictionary)

    if dictionary.index.name != 'word':
        dictionary = dictionary.set_index('word')

    pleasant = []
    active = []
    image = []

    for caption in tqdm(captions, desc="Scoring captions"):
        matched_words = [w for w in caption if w in dictionary.index]

        if matched_words:
            scores = dictionary.loc[matched_words]
            means = scores[["pleas", "activ", "image"]].mean()
            pleasant.append(means["pleas"])
            active.append(means["activ"])
            image.append(means["image"])
        else:
            pleasant.append(0.0)
            active.append(0.0)
            image.append(0.0)

    return pd.DataFrame({
        "whissell_pleasant": pd.Series(pleasant, index=captions.index),
        "whissell_active": pd.Series(active, index=captions.index),
        "whissell_image": pd.Series(image, index=captions.index),
    })

if __name__ == "__main__":
    df = pd.DataFrame({
        "caption": ["This is a test.", "Another sentence!", "Third one."]
    })

    results = {
        "caption": df["caption"]
    }

    # Add Whissell scores
    results.update(classify_whissell_scores(df["caption"]))

    # Create final DataFrame
    final_df = pd.DataFrame(results)
    print(final_df.head())
