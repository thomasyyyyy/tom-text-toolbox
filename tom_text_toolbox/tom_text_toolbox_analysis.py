"""Main module."""

from tom_text_toolbox import analyse_features

file = "text_data_TEST.csv"

def main():
    analyse_features(file = file, liwc = True)