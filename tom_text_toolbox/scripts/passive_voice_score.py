from PassivePySrc import PassivePy
import pandas as pd
from tqdm import tqdm

def count_passive(df: pd.DataFrame, captions: str = "caption", n_process: int = 10, batch_size: int = 1000):
    """
    Analyze passive voice in captions using PassivePy.

    Parameters:
        df (pd.DataFrame): DataFrame containing captions.
        captions (str): Column name containing the captions.
        n_process (int): Number of processes to use for parallel processing.
        batch_size (int): Size of each batch for processing.

    Returns:
        pd.DataFrame: DataFrame with passive voice analysis results.
    """
    # Ensure captions are strings
    df[captions] = df[captions].astype(str)

    passivepy = PassivePy.PassivePyAnalyzer(spacy_model="en_core_web_lg")
    
    df1 = passivepy.match_corpus_level(
        df, captions, n_process, batch_size, add_other_columns=True
    )

    return df1

if __name__ == "__main__":
    df = pd.read_csv("tom_text_toolbox/text_data_TEST.csv")

    df = count_passive(df)
    print(df)
