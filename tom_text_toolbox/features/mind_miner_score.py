#Import libraries
import pandas as pd
from transformers.pipelines import pipeline
import torch

#Convert the tokenised column to a list
def classify_mind_miner(captions: pd.Series | list):
    """Analyze captions using MindMiner.
    Parameters:
        captions (pd.Series | list): Series or list of caption strings.
    Returns:          
        list: List of scores for each caption.
    """

    model_name = "j-hartmann/MindMiner"
    
    mindminer = pipeline(model = model_name, function_to_apply = "none", device = 0 if torch.cuda.is_available() else -1)

    # Check if the GPU is available
    if mindminer.device == 0:
        print("Using GPU for inference.")
    else:  
        print("Using CPU for inference. This may be slower.")

    p: list = [mindminer(caption) for caption in captions]
    scores = [entry[0]['score'] for entry in p]

    return pd.Series(scores)

if __name__ == "__main__":
    # Example input DataFrame
    df = pd.DataFrame({
        "caption": [
            "This product is amazing!",
            "I'm not sure how I feel about this.",
            "Worst experience ever."
        ]
    }) 

    # Run MindMiner
    results = classify_mind_miner(df["caption"])

    # Print results
    print(results)


