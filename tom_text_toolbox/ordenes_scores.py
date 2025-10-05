import requests
import json
import time
import csv
import pandas as pd
from typing import List, Dict, Any, Optional
import concurrent.futures

# --- Gemini API Configuration ---
API_KEY = "AIzaSyA6yltpx1YP7nzI6EZo9nHukJSO_FKV3tU"
MODEL_NAME = "gemini-2.5-flash-preview-05-20"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={API_KEY}"

# --- System Instruction & Scoring Configuration ---
SYSTEM_INSTRUCTION = (
    "You are a professional Computational Linguistic Analyst. "
    "You have three tasks: "
    "1. Classify a given social media caption as Assertive, Commissive, and/or Directive. A caption can be classified as more than one. "
    "2. Rate the Specificity of the caption on a strict 1-10 scale based on concrete details, numbers, or measurable terms. "
    "3. Classify the use of Figurative Language in the caption (percentage 0-100%). "
    "Adhere strictly to the provided JSON schema for the output."
)

JSON_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "Assertive_Language": {"type": "INTEGER"},
        "Commissive_Language": {"type": "INTEGER"},
        "Directive_Language": {"type": "INTEGER"},
        "Specificity": {"type": "INTEGER"},
        "Figurative_Language": {"type": "INTEGER"}
    },
    "required": ["Assertive_Language", "Commissive_Language", "Directive_Language", "Specificity", "Figurative_Language"],
    "propertyOrdering": ["Assertive_Language", "Commissive_Language", "Directive_Language", "Specificity", "Figurative_Language"]
}

def score_caption(caption: str, max_retries: int = 5) -> Optional[Dict[str, Any]]:
    """Score a single social media caption using Gemini API."""
    user_query = f"Score the following social media caption based on the provided dimensions: '{caption}'"
    
    payload = {
        "contents": [{"parts": [{"text": user_query}]}],
        "systemInstruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]},
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": JSON_SCHEMA
        }
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, headers={'Content-Type': 'application/json'}, data=json.dumps(payload), timeout=30)
            response.raise_for_status()
            result = response.json()
            
            json_text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')
            if json_text:
                scores = json.loads(json_text)
                scores['original_caption'] = caption
                return scores
        except (requests.exceptions.RequestException, json.JSONDecodeError):
            pass

        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)

    return None

def analyze_captions(input_file: str, output_file: str, caption_column_name: str, max_workers: int = 20):
    """Read captions from CSV or Excel, analyze them, and save results to CSV."""
    
    # Load the data
    if input_file.endswith('.csv'):
        df = pd.read_csv(input_file)
    elif input_file.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(input_file)
    else:
        raise ValueError("Input file must be CSV or Excel (.xls/.xlsx)")
    
    if caption_column_name not in df.columns:
        raise ValueError(f"Column '{caption_column_name}' not found in the input file.")
    
    captions = df[caption_column_name].astype(str).tolist()
    
    print(f"Starting analysis of {len(captions)} captions using {max_workers} threads...")

    results: List[Dict[str, Any]] = []
    
    # Concurrent API calls
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(score_caption, cap): cap for cap in captions}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            caption = futures[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
                    print(f"[{i+1}/{len(captions)}] Success: '{caption[:50]}...'")
                else:
                    raise Exception("API call returned None")
            except Exception:
                failed_result = {
                    'original_caption': caption, 
                    'Assertive_Language': 0, 
                    'Commissive_Language': 0, 
                    'Directive_Language': 0, 
                    'Specificity': 0, 
                    'Figurative_Language': 0
                }
                results.append(failed_result)
                print(f"[{i+1}/{len(captions)}] Failed: '{caption[:50]}...' (Placeholder inserted)")

    # Merge results back into DataFrame
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False, quoting=csv.QUOTE_ALL)
    print(f"\nAnalysis complete! Results saved to '{output_file}'")

# ------------------ Usage ------------------
if __name__ == "__main__":
    # Set your input/output files here
    INPUT_FILE = "social_media_captions.xlsx"  # or .csv
    OUTPUT_FILE = "gemini_results.csv"
    CAPTION_COLUMN = "caption"  # Column name in your file containing captions

    analyze_captions(INPUT_FILE, OUTPUT_FILE, CAPTION_COLUMN)
