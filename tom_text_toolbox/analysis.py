import pandas as pd
import nltk
from nltk.tokenize import word_tokenize

# Import analysis libraries
from tokenizer import tokenize_caption
from whissell import whissell_scores
from syntatic_complexity import readability_scores
from mind_miner import mind_miner


from passive_voice import passive_scores




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
    df["token_captions"] = df[column].apply(tokenize_caption)
    return df

### Main function to run the analysis

def main(file: str, column: str = "caption", method:str = "complete"):

    # Read the input file
    df = read_file(file)

    # Process the DataFrame if it's valid
    if df is not None and column in df.columns:
        df = process_captions(df, column)
        return df
    else:
        print(f"Column '{column}' not found in the DataFrame.")
        return None
    
    results = {}
    
    # "Complete" analysis method
    if method == "complete":
        # All Analysis Here
        print("Running complete analysis...")

        results["whissell"] = whissell_scores(df["token_captions"])
        results["readability"] = readability_scores(df["token_captions"])
        results["mind_miner"] = mind_miner(df["token_captions"])

        print("Skipping sentistrength... See Documentation for details.")
        print("Skipping hedonometer... See Documentation for details.")

        # Passive Voice Requires a DataFrame, so is included at the end
        df = passive_scores(df)

        return df
    elif method == "summary":
        # Summary Analysis Here
        print("Running summary analysis...")
        # Add your summary analysis logic here
        # For now, just return the processed DataFrame
        return df

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process and analyze captions from a file.")
    parser.add_argument("file", type=str, help="Path to the input file (CSV or Excel).")
    parser.add_argument("column", type=str, help="Column name containing captions.")
    parser.add_argument("--method", type=str, default="complete", choices=["complete", "summary"], help="Analysis method to use.")

    args = parser.parse_args()
    
    result_df = main(args.file, args.column, args.method)
    
    if result_df is not None:
        print(result_df.head())  # Display the first few rows of the processed DataFrame

