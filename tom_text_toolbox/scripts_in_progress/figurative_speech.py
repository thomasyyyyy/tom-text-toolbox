import torch
from transformers import T5Tokenizer, MT5ForConditionalGeneration
import pandas as pd
from tqdm import tqdm

def classify_figures_of_speech(captions, tasks=None, batch_size=4):
    if tasks is None:
        tasks = ['Idiom', 'Hyperbole', 'Metaphor']
    
    # Load tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained('laihuiyuan/MMFLD')
    model = MT5ForConditionalGeneration.from_pretrained('laihuiyuan/MMFLD')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    prompt_template = 'Which figure of speech does this text contain? (A) Literal. (B) {}. | Text: {}'

    if isinstance(captions, pd.Series):
        captions = captions.tolist()
    else:
        captions = captions

    results = []

    for i in tqdm(range(0, len(captions), batch_size), desc="Processing Captions"):
        batch = captions[i:i+batch_size]
        batch_result = {"Caption": batch}

        for task in tasks:
            prompts = [prompt_template.format(task, cap) for cap in batch]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
            outputs = model.generate(**inputs)
            preds = [tokenizer.decode(o, skip_special_tokens=True).strip() for o in outputs]

            # Convert to binary: 1 if task detected, else 0
            binary_preds = []
            for pred in preds:
                pred_lower = pred.lower()
                if task.lower() in pred_lower:
                    binary_preds.append(1)
                elif "literal" in pred_lower:
                    binary_preds.append(0)
                else:
                    binary_preds.append(0)  # default to 0 if unclear

            batch_result[task] = binary_preds

        batch_df = pd.DataFrame(batch_result)
        results.append(batch_df)

    final_df = pd.concat(results, ignore_index=True)
    return final_df

if __name__ == "__main__":
    df = pd.DataFrame({
        'captions': [
            "This is a perfect way to break the ice and start the conversation.",
            "He’s a ticking time bomb ready to explode.",
            "The data speaks for itself.",
            "She’s walking on sunshine after the news.",
            "They are burning the midnight oil again.",
        ]
    })

    binary_df = classify_figures_of_speech(df['captions'])
    print(binary_df)
