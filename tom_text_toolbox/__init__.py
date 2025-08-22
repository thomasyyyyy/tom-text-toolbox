"""Top-level package for tom-text-toolbox."""

__author__ = """Thomas Young"""
__email__ = "thomasyoung0416@gmail.com"
__version__ = "0.0.1"

# Core pipeline functions
from .main import read_file, process_captions, analyse_features

# Expose linguistic feature scorers directly
from .linguistic_features.abstract_concrete_score import classify_abstract_concrete
from .linguistic_features.familiarity_score import classify_familiarity
from .linguistic_features.mind_miner_score import classify_mind_miner
from .linguistic_features.mistakes_score import count_spelling_mistakes
from .linguistic_features.passive_voice_score import count_passive
from .linguistic_features.levdist_scores import classify_levdist
from .linguistic_features.dictionary_scores import TermCounter
from .linguistic_features.spacy_measure_scores import SpacyAnalyzer
from .linguistic_features.nrc_scores import classify_nrc_dict
from .linguistic_features.whissell_scores import classify_whissell_scores
from .linguistic_features.figurative_speech_scores import classify_figures_of_speech
from .linguistic_features.liwc_scores import classify_liwc

__all__ = [
    "read_file",
    "process_captions",
    "analyse_features",
    "classify_abstract_concrete",
    "classify_familiarity",
    "classify_mind_miner",
    "count_spelling_mistakes",
    "count_passive",
    "classify_levdist",
    "TermCounter",
    "SpacyAnalyzer",
    "classify_nrc_dict",
    "classify_whissell_scores",
    "classify_figures_of_speech",
    "classify_liwc",
]
