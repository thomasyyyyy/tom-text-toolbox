from transformers.pipelines import pipeline
import torch
import time
import pandas as pd

def load_model(model_name:str = "valhalla/distilbart-mnli-12-1"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model {model_name} on {device}...")

    classifier = pipeline("zero-shot-classification", model = model_name, device = device)

    return classifier

def speech_act_labels():
    label_definitions = {
        "Commissive": "Promising or committing to a future action.",
        "Directive": "Attempting to get the listener to do something.",
        "Assertive": "Stating facts, beliefs, or opinions."
    }

    return list(label_definitions.keys())

def main(captions: pd.Series|list):
    print("Loading model and labels...")
    classifier = load_model()
    labels = speech_act_labels()
    if isinstance(captions, pd.Series):
        captions = captions.tolist()
    print("Running inference... This may take a while for large datasets.")
    results = classifier(captions, candidate_labels=labels)

    predicted_labels = [r['labels'][0] for r in results]
    predicted_scores = [r['scores'][0] for r in results]
    return pd.DataFrame({'label': predicted_labels, 'score': predicted_scores})


####################################

if __name__ == "__main__":
    df = pd.read_csv('text_data_TEST.csv')
    df = df[df["caption"].notna()]

    results = main(df["caption"])

    print(results.head())

