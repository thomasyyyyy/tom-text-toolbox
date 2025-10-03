import pandas as pd
import logging
from nltk.tokenize import word_tokenize

### Single Score Features (returns a Series)
from .linguistic_features.abstract_concrete_score import classify_abstract_concrete # Abstract/Concrete Scores
from .linguistic_features.familiarity_score import classify_familiarity # Familiarity Score
from .linguistic_features.mind_miner_score import classify_mind_miner # Mind Miner Score
from .linguistic_features.mistakes_score import count_spelling_mistakes # Spelling Mistake Count
from .linguistic_features.passive_voice_score import count_passive # Passive Voice Count
from .linguistic_features.levdist_scores import classify_levdist

### Multiple Score Features (returns a DataFrame)
from .linguistic_features.dictionary_scores import TermCounter # All custom dictionary scores (including Harvard, excluding nrc)
from .linguistic_features.spacy_measure_scores import SpacyAnalyzer # Spacy-Based Scores
from .linguistic_features.nrc_scores import classify_nrc_dict # Score Joy and Anger
from .linguistic_features.whissell_scores import classify_whissell_scores # Score Whissel Dictionary Scores
from .linguistic_features.figurative_speech_scores import classify_figures_of_speech # Score Figure of Speech
from .linguistic_features.liwc_scores import classify_liwc # Classify all liwc scores

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

def analyse_features(file: str, column: str = "caption", method: str = "complete", liwc: bool = False,
                     custom_dictionary: str = None):
    logging.info("Running Main Function")

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
        logging.info("Running Complete Analysis...")

        logging.info("Running TermCounter...")
        tc = TermCounter.from_json("linguistic_dictionaries/term_dict.json")
        term_counts_df = tc.count_all(df["caption"])
        df = pd.concat([df, term_counts_df], axis=1)

        logging.info("Running SpacyScores...")
        sc = SpacyAnalyzer()
        sc_df = sc.score_spacy_measures(df["caption"])
        df = pd.concat([df, sc_df], axis = 1)

        logging.info("Running nrcdict...")
        nrc_scores_df = classify_nrc_dict(df["caption"])
        df = pd.concat([df, nrc_scores_df], axis=1)

        #logging.info("Classifying Figurative Language...")
        #figurative_scores_df = classify_figures_of_speech(df["caption"])
        #df = pd.concat([df, figurative_scores_df], axis = 1)

        #logging.info("Counting passives... this might take a while...")
        #df["passive_count"] = count_passive(df)

        logging.info("Scoring Abstract Concrete...")
        df["abstract_concrete_score"] = classify_abstract_concrete(df["token_captions"])
        logging.info("Scoring Familiarity...")
        df["familiarity_score"] = classify_familiarity(df["token_captions"])
        logging.info("Scoring Mistakes...")
        df["mistakes_count"] = count_spelling_mistakes(df["caption"])
        logging.info("Scoring Mind Miner...")
        df["mind_miner_score"] = classify_mind_miner(df["caption"])
        logging.info("Scoring Perceptual Distance...")
        df["percept_dist"] = classify_levdist(df["caption"])
        logging.info("Scoring Whissell...")
        df[["whissell_pleasant", "whissell_active", "whissell_image"]] = classify_whissell_scores(df["token_captions"])

        # Save with the new column(s)
        output_file = "processed_captions.csv"
        df.to_csv(output_file, index=False)

        logging.info(f"Complete analysis done. File saved as {output_file}. If liwc is true, further analysis will be done...")

        if liwc:
            logging.info("Starting to analyse liwc features...")
            classify_liwc(file = "processed_captions.csv", column = "caption", dependent = True, merge_back= True, concise = True,
                          custom_dictionary= custom_dictionary)
            logging.info("All Done!")

        return df


if __name__ == "__main__":
    file = "text_data_TEST.csv"
    result_df = analyse_features(file, liwc = True)

