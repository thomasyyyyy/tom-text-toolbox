import logging
import spacy
import pandas as pd
import re
import json
from typing import Dict

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

class SpacyMeasures:
    def __init__(self, cb_ratio_file: str = "tom_text_toolbox/dictionaries/cb_ratio.json", spacy_model: str = "en_core_web_lg"):
        # Load spaCy model
        try:
            self.nlp = spacy.load(spacy_model, disable=["ner", "parser"])
            logger.info(f"spaCy model '{spacy_model}' loaded.")
        except OSError:
            raise ValueError(f"spaCy model '{spacy_model}' not found. Install with: python -m spacy download {spacy_model}")

        # Load CB ratio dictionary and merge product terms into brand
        with open(cb_ratio_file, "r", encoding="utf-8") as f:
            cb_data = json.load(f)
        self.cb_dict = {
            "brand": set(cb_data.get("brand", []) + cb_data.get("product", [])),
            "user": set(cb_data.get("user", []))
        }
        self.cb_patterns = {k: re.compile(r"\b(" + "|".join(map(re.escape, v)) + r")\b", re.IGNORECASE)
                            for k, v in self.cb_dict.items() if v}

    # -------------------------
    # Core scoring functions
    # -------------------------
    def informativeness_ratio(self, captions: pd.Series) -> pd.Series:
        docs = self.nlp.pipe(captions.astype(str), batch_size=1000, n_process=4)
        ratios = []
        for doc in docs:
            alpha_tokens = [t for t in doc if t.is_alpha]
            if not alpha_tokens:
                ratios.append(0.0)
            else:
                content_tokens = [t for t in alpha_tokens if t.pos_ in {"NOUN", "VERB", "ADJ", "ADV"}]
                ratios.append(round(len(content_tokens)/len(alpha_tokens), 3))
        return pd.Series(ratios, index=captions.index)

    def narrativity_scores(self, captions: pd.Series) -> pd.Series:
        state_verbs = {"feel", "become", "change", "transform", "realize", "understand", "decide"}
        event_verbs = {"happen", "occur", "cause", "trigger", "lead", "result", "start", "end"}

        def score(doc):
            verb_lemmas = [t.lemma_ for t in doc if t.pos_ == "VERB"]
            if not verb_lemmas:
                return 0.0
            state_count = sum(1 for v in verb_lemmas if v in state_verbs)
            event_count = sum(1 for v in verb_lemmas if v in event_verbs)
            return round((state_count + event_count)/len(verb_lemmas), 3)

        return pd.Series([score(doc) for doc in self.nlp.pipe(captions.astype(str), batch_size=1000, n_process=4)],
                         index=captions.index)

    def syntax_complexity_scores(self, captions: pd.Series) -> pd.Series:
        def score(doc):
            num_clauses = sum(1 for t in doc if t.dep_ in ("ccomp", "advcl", "acl", "relcl"))
            max_depth = max((len(list(t.ancestors)) for t in doc), default=0)
            num_subtrees = sum(1 for t in doc if len(list(t.children)) > 1)
            return round(num_clauses*1.5 + max_depth*1.2 + num_subtrees*1.0, 2)

        return pd.Series([score(doc) for doc in self.nlp.pipe(captions.astype(str), batch_size=1000, n_process=4)],
                         index=captions.index)

    def count_verb_tenses(self, captions: pd.Series) -> pd.DataFrame:
        def tense_counts(doc):
            counts = {"Past": 0, "Present": 0}
            for t in doc:
                if t.pos_ == "VERB" and "VerbForm=Fin" in t.morph:
                    if "Tense=Past" in t.morph:
                        counts["Past"] += 1
                    elif "Tense=Pres" in t.morph:
                        counts["Present"] += 1
            return counts

        docs = self.nlp.pipe(captions.astype(str), batch_size=1000, n_process=4)
        df = pd.DataFrame([tense_counts(doc) for doc in docs], index=captions.index)
        return df

    def compute_cb_ratio(self, captions: pd.Series) -> pd.DataFrame:
        brand_counts = []
        user_counts = []

        for text in captions.astype(str):
            brand_counts.append(len(self.cb_patterns.get("brand", re.compile("$")).findall(text)))
            user_counts.append(len(self.cb_patterns.get("user", re.compile("$")).findall(text)))

        df = pd.DataFrame({
            "brand": brand_counts,
            "user": user_counts
        }, index=captions.index)
        df["ratio"] = (df["brand"] / df["user"].replace(0, 1)).replace([float("inf")], 0)
        return df

    # -------------------------
    # All-in-one scoring
    # -------------------------
    def score_all(self, captions: pd.Series) -> Dict[str, pd.Series | pd.DataFrame]:
        return {
            "informativeness": self.informativeness_ratio(captions),
            "narrativity": self.narrativity_scores(captions),
            "syntax_complexity": self.syntax_complexity_scores(captions),
            "verb_tenses": self.count_verb_tenses(captions),
            "cb_ratio": self.compute_cb_ratio(captions)
        }

if __name__ == "__main__":
    # Example captions
    captions = pd.Series([
        "I love using this brand! It always makes me happy.",
        "Yesterday I tried their new product, and it really worked.",
        "Users often complain about slow delivery."
    ])

    # Initialize the SpacyMeasures class
    spacy_measures = SpacyMeasures()

    # Compute all scores
    results = spacy_measures.score_all(captions)

    # Display results using logging
    logger.info(f"Informativeness:\n{results['informativeness']}")
    logger.info(f"\nNarrativity:\n{results['narrativity']}")
    logger.info(f"\nSyntax Complexity:\n{results['syntax_complexity']}")
    logger.info(f"\nVerb Tenses:\n{results['verb_tenses']}")
    logger.info(f"\nCB Ratio:\n{results['cb_ratio']}")
