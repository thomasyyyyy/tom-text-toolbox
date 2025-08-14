import pandas as pd
import logging
from nltk.tokenize import word_tokenize

### Single Score Features (returns a Series)
from features.abstract_concrete_score import classify_abstract_concrete # Abstract/Concrete Scores
from features.familiarity_score import classify_familiarity # Familiarity Score
from features.mind_miner_score import classify_mind_miner # Mind Miner Score
from features.mistakes_score import count_spelling_mistakes # Spelling Mistake Count
from features.passive_voice_score import count_passive # Passive Voice Count
from tom_text_toolbox.features.levdist_scores import classify_levdist

### Multiple Score Features (returns a DataFrame)
from features.dictionary_scores import TermCounter # All custom dictionary scores (including Harvard, excluding nrc)
from features.spacy_measure_scores import SpacyAnalyzer # Spacy-Based Scores
from features.nrc_scores import classify_nrc_dict # Score Joy and Anger
from features.whissell_scores import classify_whissell_scores # Score Whissel Dictionary Scores
from features.figurative_speech_scores import classify_figures_of_speech # Score Figure of Speech
from features.liwc_scores import classify_liwc # Classify all liwc scores

### Read in the target file

def read_file(file: str):
    try:
        if file.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        return df
    except ValueError:
        print("Please enter either a csv file or an excel file.")
        return None

### Process the Captions
def process_captions(df: pd.DataFrame, column: str):
    df[column] = df[column].fillna("")
    df["token_captions"] = df[column].apply(word_tokenize)
    return df

### Main function to run the analysis

# Configure logging once at the top
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def analyse_features(file: str, column: str = "caption", method: str = "complete", liwc: bool = False):
    logging.info("Running main function")

    # Read the input file
    df = read_file(file)

    # Process the DataFrame if it's valid
    if df is not None and column in df.columns:
        df = process_captions(df, column)
    else:
        logging.error(f"Column '{column}' not found in the DataFrame.")
        return None
    
    # "Complete" analysis method
    if method == "complete":
        logging.info("Running complete analysis...")

        tc = TermCounter.from_json("tom_text_toolbox/dictionaries/term_dict.json")
        term_counts_df = tc.count_all(df["caption"])
        df = pd.concat([df, term_counts_df], axis=1)

        sc = SpacyAnalyzer()
        sc_df = sc.score_spacy_measures(df["caption"])
        df = pd.concat([df, sc_df], axis = 1)

        nrc_scores_df = classify_nrc_dict(df["caption"])
        df = pd.concat([df, nrc_scores_df], axis=1)

        figurative_scores_df = classify_figures_of_speech(df["cpation"])
        df = pd.concat([df, figurative_scores_df], axis = 1)

        df["passive_count"] = count_passive(df)

        df["abstract_concrete_score"] = classify_abstract_concrete(df["token_captions"])
        df["familiarity_score"] = classify_familiarity(df["token_captions"])
        df["mistakes_count"] = count_spelling_mistakes(df["caption"])
        df["percept_dist"] = classify_levdist(df["caption"])
        df[["whissell_pleasant", "whissell_active", "whissell_image"]] = classify_whissell_scores(df["token_captions"])

        # Save with the new column(s)
        df.to_csv("processed_captions.csv", index=False)

        logging.info("Complete analysis done")

        logging.info("Main function done running")

        if liwc:
            logging.info("Starting to analyse liwc features...")
            classify_liwc(file = "processed_captions.csv", column = "caption", dependent = True, merge_back= True, concise = True)

        return df


if __name__ == "__main__":
    file = "tom_text_toolbox/text_data_TEST.csv"
    result_df = analyse_features(file)
    
    if result_df is not None:
        print(result_df.head())  # Display the first few rows of the processed DataFrame

