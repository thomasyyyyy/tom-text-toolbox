import pandas as pd
from tokenizer import process_caption
from tqdm import tqdm
import os

def whissell_score(caption: str | list[str], dictionary: pd.DataFrame):
    ### Function to score words in a caption
    pleasant = []
    active = []
    image = []
    for word in caption:
        if word in dictionary.index:
            pleasant.append(dictionary.loc[word, "pleas"])
            active.append(dictionary.loc[word, "activ"])
            image.append(dictionary.loc[word, "image"])
    
    # Calculate mean scores
    mean_pleasant = sum(pleasant) / len(pleasant) if pleasant else 0
    mean_active = sum(active) / len(active) if active else 0
    mean_image = sum(image) / len(image) if image else 0

    return mean_pleasant, mean_active, mean_image

def whissell_scores(captions: list | pd.Series, dictionary: pd.DataFrame):
    ### Scores all captions

    # Intiialise lists to store the scores
    mean_pleasant_scores = []
    mean_active_scores = []
    mean_image_scores = []

    # Process each caption to extract words and compute scores
    for caption in tqdm(captions, desc = f"Processing Whissell Scores..."):
        words = process_caption(caption)
        mean_pleasant, mean_active, mean_image = whissell_score(words, dictionary)
        mean_pleasant_scores.append(mean_pleasant)
        mean_active_scores.append(mean_active)
        mean_image_scores.append(mean_image)
    
    return mean_pleasant_scores, mean_active_scores, mean_image_scores

if __name__ == "__main__":
    captions = ["This is a test.", "Another sentence!", "Third one."]
    basepath = os.path.dirname(os.path.realpath(__file__))

    # Path to a sample image for debugging   
    data_folder = basepath + "/dictionaries/"
    
    # Path to a sample image
    dictionary_file = data_folder + "whissell_dict.csv"

    # Assigning Dictionary
    dictionary = pd.read_csv(dictionary_file)

    results = whissell_scores(captions, dictionary = dictionary)

    # Convert to DataFrame for funsies
    df = pd.DataFrame(results)
    print(df.head())

