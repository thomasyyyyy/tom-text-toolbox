"""Main module."""

from main import analyse_features

file = "text_data_TEST.csv"

if __name__ == "__main__":
    analyse_features(file = file, liwc = True)