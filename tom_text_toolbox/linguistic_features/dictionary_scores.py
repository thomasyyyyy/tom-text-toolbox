import pandas as pd
import numpy as np
import re
import json
import os

from emosent import get_emoji_sentiment_rank_multiple
from collections import Counter
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor

class TermCounter:
    def __init__(self, term_dict: Dict[str, List[str]]):
        """
        Initialize TermCounter with a dictionary of term categories.
        term_dict: {category_name: [list_of_terms]}
        """
        if not isinstance(term_dict, dict):
            raise ValueError("term_dict must be a dictionary.")
        if not all(isinstance(v, list) for v in term_dict.values()):
            raise ValueError("Each value in term_dict must be a list of terms.")

        self.term_dict = term_dict
        self.patterns = {name: self.build_pattern(terms) for name, terms in term_dict.items()}

    @classmethod
    def from_json(cls, json_path: str = "term_dict.json"):
        """
        Initialize TermCounter directly from a JSON file.
        json_path: Path to the JSON file containing {category_name: [list_of_terms]}.
                   If a relative path is given, it will be resolved relative to the
                   'linguistic_dictionaries' folder inside the package.
        """
        # If only a filename or relative path is passed, resolve relative to package
        if not os.path.isabs(json_path):
            base_dir = os.path.dirname(__file__)  # current file: dictionary_scores.py
            package_root = os.path.abspath(os.path.join(base_dir, ".."))  # tom_text_toolbox/
            json_path = os.path.join(package_root, "linguistic_dictionaries", json_path)

        with open(json_path, "r", encoding="utf-8") as f:
            term_dict = json.load(f)

        if not isinstance(term_dict, dict):
            raise ValueError("JSON must contain a dictionary at the top level.")
        if not all(isinstance(v, list) for v in term_dict.values()):
            raise ValueError("Each value in the JSON must be a list of terms.")

        return cls(term_dict)

    def build_pattern(self, terms: List[str]) -> re.Pattern:
        """Compile a regex pattern for a list of terms (supports '*' wildcard)."""
        pattern_parts = [
            rf"\b{re.escape(term[:-1])}\w*" if term.endswith("*") else rf"\b{re.escape(term)}\b"
            for term in terms
        ]
        return re.compile(rf"(?:{'|'.join(pattern_parts)})", re.IGNORECASE)

    def count_terms(self, captions: pd.Series, category: str) -> pd.Series:
        """Count term matches for a specific category."""
        if category not in self.patterns:
            raise ValueError(f"Category '{category}' not found in term_dict.")
        return captions.str.count(self.patterns[category].pattern)

    def extract_emoji_dict(self, captions: pd.Series, parallel: bool = True, verbose: bool = False) -> Counter:
        """Extract emoji sentiment counts from text."""
        emoji_counter = Counter()
        captions = captions.astype(str)

        def process_caption(caption: str) -> List[str]:
            try:
                results = get_emoji_sentiment_rank_multiple(caption)
                return [item.get('emoji_sentiment_rank', {}).get('unicode_name', 'unknown') for item in results]
            except Exception as e:
                if verbose:
                    print(f"Error processing text: {e}")
                return []

        if parallel:
            with ThreadPoolExecutor() as executor:
                all_results = list(executor.map(process_caption, captions))
            for result in all_results:
                emoji_counter.update(result)
        else:
            for caption in captions:
                emoji_counter.update(process_caption(caption))

        return emoji_counter

    @staticmethod
    def exclamation_count(captions: pd.Series) -> pd.Series:
        return captions.str.count(r'!')

    @staticmethod
    def question_count(captions: pd.Series) -> pd.Series:
        return captions.str.count(r'\?')

    @staticmethod
    def hashtag_count(captions: pd.Series) -> pd.Series:
        return captions.str.count(r'#\S+')

    @staticmethod
    def mention_count(captions: pd.Series) -> pd.Series:
        return captions.str.count(r'@\w+')

    @staticmethod
    def caption_length(captions: pd.Series) -> pd.Series:
        return captions.str.len()

    def type_token_ratio(self, captions: pd.Series, segment_size: int = 5) -> pd.Series:
        """Calculate segmental type-token ratio (TTR) for each caption."""
        def calculate_segmental_ttr(text: str) -> Optional[float]:
            words = str(text).lower().split()
            if not words:
                return None
            segments = [words[i:i + segment_size] for i in range(0, len(words), segment_size)]
            ttrs = [len(set(seg)) / len(seg) for seg in segments]
            return round(float(np.mean(ttrs)), 3)
        return captions.apply(calculate_segmental_ttr)
    
    def count_all(self, captions: pd.Series) -> pd.DataFrame:
        """Count matches for all categories AND include additional text features."""
        # Count term matches
        df_counts = pd.DataFrame({cat: captions.str.count(pat.pattern) for cat, pat in self.patterns.items()})

        # Add extra features
        df_counts['exclamation_count'] = self.exclamation_count(captions)
        df_counts['question_count'] = self.question_count(captions)
        df_counts['hashtag_count'] = self.hashtag_count(captions)
        df_counts['mention_count'] = self.mention_count(captions)
        df_counts['caption_length'] = self.caption_length(captions)
        df_counts['type_token_ratio'] = self.type_token_ratio(captions)

        return df_counts

if __name__ == "__main__":
    tc = TermCounter.from_json("tom_text_toolbox/dictionaries/term_dict.json")
    df = pd.read_csv("tom_text_toolbox/text_data_TEST.csv")

    # Count all term categories
    term_counts_df = tc.count_all(df["caption"])
    
    # Add counts as new columns to the original DataFrame
    df = pd.concat([df, term_counts_df], axis=1)

    print(df.head())