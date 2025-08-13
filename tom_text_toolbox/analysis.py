import pandas as pd
import logging
from nltk.tokenize import word_tokenize

from scripts.abstract_concrete_score import classify_abstract_concrete # Abstract/Concrete Scores
from scripts.familiarity_score import classify_familiarity # Familiarity Scores
from scripts.mind_miner_score import classify_mind_miner # Mind Miner Scores
from scripts.mistakes_score import count_spelling_mistakes # Spelling Mistake Counts
from scripts.passive_voice_score import count_passive # Passive Voice Count

from scripts.dictionary_scores import TermCounter # All custom dictionary scores (including Harvard, excluding nrc)
from scripts.spacy_measure_scores import SpacyAnalyzer # Spacy-Based Scores
from scripts.nrc_scores import classify_nrc_dict # Score Joy and Anger
from scripts.whissell import classify_whissell_scores

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

def main(file: str, column: str = "caption", method: str = "complete", liwc: bool = False):
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

        df["abstract_concrete_score"] = classify_abstract_concrete(df["token_captions"])
        df["familiarity_score"] = classify_familiarity(df["token_captions"])
        df["mistakes_count"] = count_spelling_mistakes(df["caption"])
        df["passive_count"] = count_passive(df)
        df[["whissell_pleasant", "whissell_active", "whissell_image"]] = classify_whissell_scores(df["token_captions"])

        # Save with the new column(s)
        #df.to_csv("processed_captions.csv", index=False)

        logging.info("Complete analysis done")

        logging.info("Main function done running")
        return df


if __name__ == "__main__":
    file = "tom_text_toolbox/text_data_TEST.csv"
    result_df = main(file)
    
    if result_df is not None:
        print(result_df.head())  # Display the first few rows of the processed DataFrame

