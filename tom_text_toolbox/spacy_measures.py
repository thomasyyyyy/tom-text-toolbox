from dictionaries import TermCounter
from collections import Counter
import re
import pandas as pd
from typing import List, Dict
import spacy

class SpacyMeasures(TermCounter):
    def __init__(self, term_dict: Dict[str, List[str]], spacy_model: str = "en_core_web_sm"):
        super().__init__(term_dict)
        try:
            self.nlp = spacy.load(spacy_model, disable=["ner", "parser"])
            print(f"spaCy model '{spacy_model}' loaded.")
        except OSError:
            raise ValueError(f"spaCy model '{spacy_model}' not found. Install with: python -m spacy download {spacy_model}")

    def count_terms_with_spacy(self, captions: pd.Series, category: str) -> pd.Series:
        if not self.nlp:
            raise RuntimeError("spaCy model not loaded.")
        if category not in self.patterns:
            raise ValueError(f"Category '{category}' not found in term_dict.")
        
        pattern = self.patterns[category]
        return pd.Series(
            [sum(1 for token in doc if pattern.search(token.text)) 
             for doc in self.nlp.pipe(captions, batch_size=1000, n_process=4)],
            index=captions.index
        )

    def informativeness_ratio(self, captions: pd.Series) -> pd.Series:
        return pd.Series([
            round(sum(1 for t in doc if t.pos_ in {"NOUN", "VERB", "ADJ", "ADV"} and t.is_alpha) / 
                  sum(1 for t in doc if t.is_alpha), 3)
            if sum(1 for t in doc if t.is_alpha) > 0 else None
            for doc in self.nlp.pipe(captions.astype(str), batch_size=100, n_process=1)
        ], index=captions.index)

    def narrativity_scores(self, captions: pd.Series) -> pd.Series:
        state_verbs = {"feel", "become", "change", "transform", "realize", "understand", "decide"}
        event_verbs = {"happen", "occur", "cause", "trigger", "lead", "result", "start", "end"}
        
        def compute(text: str) -> float:
            doc = self.nlp(str(text))
            if not any(ent.label_ in {"PERSON", "ORG", "NORP"} for ent in doc.ents):
                return 0.0
            verb_lemmas = [t.lemma_ for t in doc if t.pos_ == "VERB"]
            if not verb_lemmas:
                return 0.0
            state_count = sum(1 for v in verb_lemmas if v in state_verbs)
            event_count = sum(1 for v in verb_lemmas if v in event_verbs)
            if (state_count + event_count) == 0:
                return 0.0
            return round((state_count + event_count) / len(verb_lemmas), 3)

        return captions.apply(compute)

    def syntax_complexity_scores(self, captions: pd.Series) -> pd.Series:
        def compute(text: str) -> float:
            doc = self.nlp(str(text))
            num_clauses = sum(1 for t in doc if t.dep_ in ("ccomp", "advcl", "acl", "relcl"))
            max_depth = max((len(list(t.ancestors)) for t in doc), default=0)
            num_subtrees = sum(1 for t in doc if len(list(t.children)) > 1)
            score = num_clauses * 1.5 + max_depth * 1.2 + num_subtrees * 1.0
            return round(score, 2)
        
        return captions.apply(compute)

    def count_verb_tenses(self, captions: pd.Series) -> pd.DataFrame:
        results = []
        for doc in self.nlp.pipe(captions.astype(str), batch_size=100, n_process=1):
            past = present = 0
            for token in doc:
                if token.pos_ == "VERB" and "VerbForm=Fin" in token.morph:
                    if "Tense=Past" in token.morph:
                        past += 1
                    elif "Tense=Pres" in token.morph:
                        present += 1
            results.append([past, present])
        return pd.DataFrame(results, columns=["Past", "Present"], index=captions.index)
    

    def score_all(self, captions: pd.Series) -> Dict[str, pd.Series | pd.DataFrame]:
        return {
            "informativeness": self.informativeness_ratio(captions),
            "narrativity": self.narrativity_scores(captions),
            "syntax_complexity": self.syntax_complexity_scores(captions),
            "verb_tenses": self.count_verb_tenses(captions)
        }
