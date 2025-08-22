from transformers.pipelines import pipeline
import torch
import time
import pandas as pd

df = pd.read_csv("tom_text_toolbox/text_data_TEST.csv")
df = df[df["caption"].notna()]

print(df.shape)

# Use GPU if available
device = 0 if torch.cuda.is_available() else -1

# Load model directly (no save/load)
classifier = pipeline("zero-shot-classification", model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", device=device)

start = time.time()

# Input sentences
label_sentences = [
    "We’ll launch a loyalty program for our customers next month.",           # Commissive
    "Sign up today to receive your free sample.",                             # Directive
    "We’re thrilled to have been nominated for this award—thank you!",        # Expressive
    "Our company has served over 100,000 satisfied customers since 2010.",    # Assertive
    "We hereby certify that your account has been permanently deactivated.",  # Declarative
    "The store opens at 9:00 AM on weekdays and closes at 6:00 PM."           # Literal
]


# Label definitions (can be used in output)
label_definitions = {
    "Commissive": "Promising or committing to a future action.",
    "Directive": "Attempting to get the listener to do something.",
    "Assertive": "Stating facts, beliefs, or opinions."
}

# Labels
labels = list(label_definitions.keys())

# Run batch inference
results = classifier(df["caption"].tolist(), candidate_labels=labels, multi_label=False)

df["results"] = results

end = time.time()
print(f"Total time taken: {end - start:.2f} seconds")

results_subset = results[:3]

# Display results with definitions
for i, res in enumerate(results_subset):
    print(f"Sentence: {res['sequence']}")
    for label, score in zip(res['labels'], res['scores']):
        definition = label_definitions[label]
        print(f"  {label}: {score:.4f} — {definition}")
    print("-" * 40)

#valhalla/distilbart-mnli-12-1: 42.89
#facebook/bart-large-mnli: 86.87
#MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli: 80.33