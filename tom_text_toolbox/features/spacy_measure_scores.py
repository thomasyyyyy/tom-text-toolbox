import spacy
from spacy.symbols import NOUN, VERB, ADJ, ADV
import pandas as pd
import json

class SpacyAnalyzer:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_lg")

        # Load cb_ratio.json from dictionaries folder
        cb_path = "tom_text_toolbox/dictionaries/cb_ratio.json"
        with open(cb_path, "r", encoding="utf-8") as f:
            self.term_dict = json.load(f)

        # Convert lists to sets for fast lookup
        self.term_dict = {k: set(v) for k, v in self.term_dict.items()}

    def score_spacy_measures(self, captions: pd.Series) -> pd.DataFrame:
        docs = list(self.nlp.pipe(captions.astype(str), batch_size=2000, n_process=4))

        # Predefine sets
        state_verbs = {"feel", "become", "change", "transform", "realize", "understand", "decide"}
        event_verbs = {"happen", "occur", "cause", "trigger", "lead", "result", "start", "end"}

        # Containers
        informativeness = []
        narrativity = []
        boastful = []
        syntax_complexity = []
        tense_data = []

        consumer_counts = []
        brand_counts = []

        for doc in docs:
            alpha_tokens = [t for t in doc if t.is_alpha]
            n_tokens = len(alpha_tokens)

            # ---- Informativeness ----
            content_count = doc.count_by(spacy.attrs.POS)
            content_tokens = content_count.get(NOUN, 0) + content_count.get(VERB, 0) + \
                             content_count.get(ADJ, 0) + content_count.get(ADV, 0)
            informativeness.append(round(content_tokens / n_tokens, 3) if n_tokens else 0.0)

            # ---- Narrativity ----
            verbs = [t.lemma_ for t in alpha_tokens if t.pos_ == "VERB"]
            n_verbs = len(verbs)
            if n_verbs:
                state_count = sum(1 for v in verbs if v in state_verbs)
                event_count = sum(1 for v in verbs if v in event_verbs)
                narrativity.append(round((state_count + event_count) / n_verbs, 3))
            else:
                narrativity.append(0.0)

            # ---- Syntax complexity ----
            dep_counts = doc.count_by(spacy.attrs.DEP)
            num_clauses = sum(dep_counts.get(doc.vocab.strings[dep], 0) for dep in ["ccomp", "advcl", "acl", "relcl"])
            max_depth = max((len(list(t.ancestors)) for t in alpha_tokens), default=0)
            num_subtrees = sum(1 for t in alpha_tokens if len(list(t.children)) > 1)
            syntax_complexity.append(round(num_clauses * 1.5 + max_depth * 1.2 + num_subtrees * 1.0, 2))

            # ---- Verb tenses ----
            counts = {"Past": 0, "Present": 0}
            for t in alpha_tokens:
                if t.pos_ == "VERB" and "VerbForm=Fin" in t.morph:
                    if "Tense=Past" in t.morph:
                        counts["Past"] += 1
                    elif "Tense=Pres" in t.morph:
                        counts["Present"] += 1
            tense_data.append(counts)

            # ---- Consumer/Brand ratio ----
            subjects = [t for t in doc if t.dep_ in ("nsubj", "nsubjpass")]
            c_count = sum(1 for tok in subjects if tok.lemma_.lower() in self.term_dict["user"] or tok.text.lower() in self.term_dict["user"])
            b_count = sum(1 for tok in subjects if tok.lemma_.lower() in self.term_dict["brand"] or tok.text.lower() in self.term_dict["brand"] 
                          or tok.lemma_.lower() in self.term_dict["product"] or tok.text.lower() in self.term_dict["product"])
            consumer_counts.append(c_count)
            brand_counts.append(b_count)

            # ---- Boastful Language ----
            boast_count = sum(1 for token in doc if token.tag_ in ["JJS", "RBS"])
            boastful.append(boast_count)

        # Build final DataFrame
        df = pd.DataFrame({
            "informativeness": informativeness,
            "narrativity": narrativity,
            "syntax_complexity": syntax_complexity,
            "cb_ratio": [
                float('inf') if b == 0 and c > 0 else 0.0 if b == 0 else c / b
                for c, b in zip(consumer_counts, brand_counts)
            ],
            "tense_past": [t["Past"] for t in tense_data],
            "tense_present": [t["Present"] for t in tense_data]
        }, index=captions.index)

        return df

if __name__ == "__main__":
    captions = pd.Series([
        "The customer loves the new Apple iPhone!",
        "Nike releases its latest sports shoes for athletes.",
        "Clients are happy with Coca-Cola's new flavors.",
        "Users feel excited when they see a new product."
    ])

    analyzer = SpacyAnalyzer()
    results_df = analyzer.score_spacy_measures(captions)
    print(results_df)
