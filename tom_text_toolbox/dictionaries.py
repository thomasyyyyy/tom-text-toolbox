import re
import pandas as pd
from typing import List, Dict, Optional

from collections import Counter
from emosent import get_emoji_sentiment_rank_multiple

import Levenshtein as Lev

class TermCounter:
    def __init__(self, term_dict: Dict[str, List[str]], spacy_model: Optional[str] = None):
        self.term_dict = term_dict
        self.patterns = {
            name: self.build_pattern(terms)
            for name, terms in term_dict.items()
        }

    def build_pattern(self, terms: List[str]) -> re.Pattern:
        pattern_parts = [
            rf"\b{re.escape(term[:-1])}\w*" if term.endswith("*") else rf"\b{re.escape(term)}\b"
            for term in terms
        ]
        return re.compile(rf"(?:{'|'.join(pattern_parts)})", re.IGNORECASE)

    def count_terms(self, captions: pd.Series, category: str) -> pd.Series:
        """Count term matches in a specific category."""
        if category not in self.patterns:
            raise ValueError(f"Category '{category}' not found in term_dict.")
        pattern = self.patterns[category]
        return captions.str.count(pattern)

    def count_all(self, captions: pd.Series) -> pd.DataFrame:
        """Count all categories in one go."""
        return pd.DataFrame({
            category: captions.str.count(pattern)
            for category, pattern in self.patterns.items()
        })
    
    def extract_emoji_dict(self, captions: pd.Series, verbose:bool = False):
        # Need to test this function
        """
        Extract emoji sentiment counts from text using get_emoji_sentiment_rank_multiple.
        Returns a dictionary with emoji unicode names as keys and counts as values.
        """

        emoji_counter = Counter()
        for caption in captions:
            try:
                results = get_emoji_sentiment_rank_multiple(str(caption))
                emoji_counter.update(
                    item.get('emoji_sentiment_rank', {}).get('unicode_name', 'unknown')
                    for item in results
                )
            except Exception as e:
                if verbose:
                    print(f"Error processing text: {e}")
                continue
        return emoji_counter
    
    def exclamation_count(self, captions: pd.Series) -> pd.Series:
        """Count the number of exclamation marks in each caption."""
        return captions.str.count(r'!')
    
    def question_count(self, captions: pd.Series) -> pd.Series:
        """Count the number of question marks in each caption."""
        return captions.str.count(r'\?')
    
    def hashtag_count(self, captions: pd.Series) -> pd.Series:
        """Count the number of hashtags in each caption."""
        return captions.str.count(r'#\S+')
    
    def mention_count(self, captions: pd.Series) -> pd.Series:
        """Count the number of mentions (e.g., @username) in each caption."""
        return captions.str.count(r'@\w+')
    
    def caption_length(self, captions: pd.Series) -> pd.Series:
        """Calculate the length of each caption."""
        return captions.str.len()