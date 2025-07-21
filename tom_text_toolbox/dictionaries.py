import re
import pandas as pd
import spacy
from typing import List, Dict, Optional

from collections import Counter
from emosent import get_emoji_sentiment_rank_multiple

import Levenshtein as Lev

import pyenchant as pe

class TermCounter:
    def __init__(self, term_dict: Dict[str, List[str]], spacy_model: Optional[str] = None):
        self.term_dict = term_dict
        self.patterns = {
            name: self._build_pattern(terms)
            for name, terms in term_dict.items()
        }

    def _build_pattern(self, terms: List[str]) -> re.Pattern:
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

class SpacyTermCounter(TermCounter):
    def __init__(self, term_dict: Dict[str, List[str]], spacy_model: str = "en_core_web_sm"):
        super().__init__(term_dict)
        try:
            self.nlp = spacy.load(spacy_model, disable=["ner", "parser"])
            print(f"spaCy model '{spacy_model}' loaded.")
        except OSError:
            raise ValueError(f"spaCy model '{spacy_model}' not found. Install with: python -m spacy download {spacy_model}")

    def count_terms_with_spacy(self, captions: pd.Series, category: str) -> pd.Series:
        """Count terms using spaCy for more complex patterns."""
        if not self.nlp:
            raise RuntimeError("spaCy model not loaded.")
        if category not in self.patterns:
            raise ValueError(f"Category '{category}' not found in term_dict.")
        
        counts = []
        pattern = self.patterns[category]
        for doc in self.nlp.pipe(captions, batch_size=1000, n_process=4):
            count = sum(1 for token in doc if pattern.search(token.text))
            counts.append(count)
        return pd.Series(counts, index=captions.index)

    def informativeness_ratio(self, captions: pd.Series) -> pd.Series:
        """
        Calculate the ratio of content words (NOUN, VERB, ADJ, ADV)
        to total alphabetic tokens in each caption.
        Returns a Series of ratios.
        """
        ratios = []
        docs = self.nlp.pipe(captions.astype(str), batch_size=100, n_process=1)  # or n_process=4 if safe

        for doc in docs:
            content_count = sum(1 for token in doc if token.pos_ in {"NOUN", "VERB", "ADJ", "ADV"} and token.is_alpha)
            total_count = sum(1 for token in doc if token.is_alpha)
            ratio = round(content_count / total_count, 3) if total_count else None
            ratios.append(ratio)

        return pd.Series(ratios, index=captions.index)
    
    def narrativity_score(self, text: str) -> float:
        #Double check functionality
        """
        Compute a narrativity score based on presence of named characters (PERSON, ORG, NORP)
        and ratio of state/event verbs to all verbs in the text.
        """
        doc = self.nlp(str(text))

        # Check for at least one narrative character/entity
        has_character = any(ent.label_ in {"PERSON", "ORG", "NORP"} for ent in doc.ents)
        if not has_character:
            return 0.0

        # Get all verb lemmas
        verb_lemmas = [token.lemma_ for token in doc if token.pos_ == "VERB"]
        if not verb_lemmas:
            return 0.0

        # Verb sets
        state_verbs = {"feel", "become", "change", "transform", "realize", "understand", "decide"}
        event_verbs = {"happen", "occur", "cause", "trigger", "lead", "result", "start", "end"}

        # Count matches
        state_count = sum(1 for v in verb_lemmas if v in state_verbs)
        event_count = sum(1 for v in verb_lemmas if v in event_verbs)

        if (state_count + event_count) == 0:
            return 0.0

        score = (state_count + event_count) / len(verb_lemmas)
        return round(score, 3)
    
    def narrativity_scores(self, captions: pd.Series) -> pd.Series:
        return captions.apply(self.narrativity_score)

    def syntax_complexity(self, text: str) -> float | None:
        doc = self.nlp(str(text))
        if not doc:
            return None

        # Features to consider
        num_clauses = sum(1 for token in doc if token.dep_ in ("ccomp", "advcl", "acl", "relcl"))
        max_depth = max((len(list(token.ancestors)) for token in doc), default=0)
        num_subtrees = sum(1 for token in doc if len(list(token.children)) > 1)

        # Weighted score (tweak weights as needed)
        score = num_clauses * 1.5 + max_depth * 1.2 + num_subtrees * 1.0
        return round(score, 2)
    
    def syntax_complexity_scores(self, captions: pd.Series) -> pd.Series:
        return captions.apply(self.syntax_complexity)

    def count_verb_tenses(self, text: str) -> pd.Series:
        doc = self.nlp(text)
        past = 0
        present = 0

        for token in doc:
            if token.pos_ == "VERB" and "VerbForm=Fin" in token.morph:
                if "Tense=Past" in token.morph:
                    past += 1
                elif "Tense=Pres" in token.morph:
                    present += 1
        return pd.Series([past, present])
    
    def verb_tense_counts(self, captions: pd.Series) -> pd.DataFrame:
        """
        Count past and present tense verbs in each caption.
        Returns a DataFrame with 'Past' and 'Present' columns.
        """
        counts = captions.apply(self.count_verb_tenses)
        return pd.DataFrame(counts.tolist(), index=captions.index, columns=["Past", "Present"])

class SpellingMistakes:
    def __init__(self, language: str = "en_US"):
        """Initialize the spell checker with a specific language."""
        self.spell_checker = pe.Dict(language)

    def sum_spelling_mistakes(self, captions: pd.Series) -> pd.Series:
        """Count spelling mistakes in each caption."""
        def count_mistakes(text: str) -> int:
            words = re.findall(r"\b[a-zA-Z]+\b", text)
            return sum(not self.spell_checker.check(word) for word in words)

        return captions.apply(count_mistakes)
    
class FamiliarityScorer:
    def __init__(self, dict_path: str = "../dictionaries/peatzold_dict.csv"):
        """Load the familiarity dictionary from a CSV file."""
        df = pd.read_csv(dict_path)
        self.fam_dict = dict(zip(df["Word"].str.lower(), df["Familiarity"]))

    def _score_caption(self, text: str) -> float | None:
        """Compute average familiarity score for a single caption."""
        tokens = str(text).lower().split()
        scores = [self.fam_dict[token] for token in tokens if token in self.fam_dict]
        return round(sum(scores) / len(scores), 2) if scores else None

    def familiarity_score(self, captions: pd.Series) -> pd.Series:
        """Apply scoring to a pandas Series of captions."""
        return captions.apply(self._score_caption)

    # scorer = FamiliarityScorer("../dictionaries/peatzold_dict.csv")
    # df["Familiarity Score"] = scorer.familiarity_score(df["cleaned_caption"])
