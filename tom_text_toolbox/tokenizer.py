import nltk
from nltk.tokenize import word_tokenize

nltk.download("punkt")
nltk.download("punkt_tab")

def tokenize_caption(caption):
    ### Filters out non-words and links. Sets them to lowercase to match the dictionary.
    tokens = word_tokenize(caption)
    filtered_tokens = [word.lower() for word in tokens if word.isalpha() and "https" not in word]
    return filtered_tokens
