"""Main module."""

from tom_text_toolbox.main import analyse_features

file = "Input file name here"

def main():
    analyse_features(file = file, liwc = True)