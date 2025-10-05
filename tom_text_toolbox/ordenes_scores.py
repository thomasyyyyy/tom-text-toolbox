import requests
import json
import time
import csv
from io import StringIO
import concurrent.futures
from typing import List, Dict, Any, Optional

# --- Gemini API Configuration ---
API_KEY = "AIzaSyA6yltpx1YP7nzI6EZo9nHukJSO_FKV3tU"
MODEL_NAME = "gemini-2.5-flash-preview-05-20"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={API_KEY}"

# --- System Instruction & Scoring Configuration ---

SYSTEM_INSTRUCTION = (
    "You are a professional Computational Linguistic Analyst."
    "You have three tasks:"
    "The first is to critically classify a given social media caption as either an Assertive, Commissive, and/or Directive statement. A caption can be classified as more than one of these categories if applicable."
    "The second is to rate the Specificity of the caption based on the level of detail in its use of concrete details, numbers, or measurable terms using a strict 1-10 scale (1=Low Presence/Weak, 10=High Presence/Strong).."
    "The third is to classify a caption's use of Figurative Language, such as metaphors or similes. Get the percentage of figurative language used in the caption (i.e. 0% = No Figurative Language, 100% = Entirely Figurative Language)."
    "You MUST adhere strictly to the provided JSON schema for the output. "
    "The scores should reflect the strength and presence of each linguistic category."
)

# JSON Schema defining the required output structure for data analysis
JSON_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "Assertive_Language": {
            "type": "INTEGER",
            "description": "Score 1-10 on the strength and presence of statements of fact or belief (e.g., 'We are the best...')."
        },
        "Commissive_Language": {
            "type": "INTEGER",
            "description": "Score 1-10 on the presence of commitments, promises, or pledges made by the brand (e.g., 'We promise to deliver...')."
        },
        "Directive_Language": {
            "type": "INTEGER",
            "description": "Score 1-10 on the clarity and forcefulness of calls to action (CTAs), commands, or requests (e.g., 'Click the link...', 'Tell us your thoughts...')."
        },
        "Specificity": {
            "type": "INTEGER",
            "description": "Score 1-10 on the use of concrete details, numbers, or measurable terms (e.g., 'Save 35% in 48 hours')."
        },
        "Figurative_Language": {
            "type": "INTEGER",
            "description": "Classify a caption as either using figurative language or not. Figurative language includes words or phrases that are meaningful, but not literally true, including metaphor, simile, sarcasm, and hyperbole (e.g., 'This product is a magic wand for your skin')."
        },
    },
    "required": ["Assertive_Language", "Commissive_Language", "Directive_Language", "Specificity", "Figurative_Language"],
    "propertyOrdering": [
        "Assertive_Language", "Commissive_Language", "Directive_Language",
        "Specificity", "Figurative_Language"
    ]
}


def score_caption(caption: str, max_retries: int = 5) -> Optional[Dict[str, Any]]:
    """
    Calls the Gemini API to score a social media caption based on linguistic dimensions.

    Args:
        caption (str): The social media caption text to score.
        max_retries (int): The maximum number of retries for the API call.

    Returns:
        dict | None: The structured JSON response with scores and critique, or None on failure.
    """
    user_query = f"Score the following social media caption based on the provided dimensions: '{caption}'"

    payload = {
        "contents": [
            {"parts": [{"text": user_query}]}
        ],
        "systemInstruction": {
            "parts": [{"text": SYSTEM_INSTRUCTION}]
        },
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": JSON_SCHEMA
        }
    }

    # Implement exponential backoff for robust API calling
    for attempt in range(max_retries):
        try:
            response = requests.post(
                API_URL,
                headers={'Content-Type': 'application/json'},
                data=json.dumps(payload),
                timeout=30 # Set a timeout for the request
            )
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

            result = response.json()
            
            # Extract the JSON string from the response
            json_text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')
            
            if json_text:
                # The response text is a JSON string, so we need to parse it
                scores = json.loads(json_text)
                # Add the original caption to the result for context
                scores['original_caption'] = caption
                return scores
            else:
                # Fail fast on empty content if we can't retry
                pass

        except requests.exceptions.RequestException:
            # Silence logging of retries 
            pass 
        except json.JSONDecodeError:
            pass # Silence logging of retries

        # If it's not the last attempt, wait before retrying (exponential backoff)
        if attempt < max_retries - 1:
            wait_time = 2 ** attempt
            time.sleep(wait_time)

    # Note: Failure logging is now handled in the main execution block for clarity
    return None

# --- Example Usage (Simulating a Dataset Column) ---

if __name__ == "__main__":
    
    # This list simulates a column in your pandas DataFrame or CSV file
    caption_column: List[str] = [
        "Unleash your inner explorer! Our new titanium watch is built like a tank and will track your altitude with 99.9% accuracy. Buy one today!",
        "We promise to always prioritize sustainable materials. It is our vow to the planet, and we believe our commitment speaks for itself.",
        "This product is truly life-changing. It’s a magic wand for your skin. Tell us what you think in the comments below!",
        "Sale! Everything is discounted right now. Don't miss out.",
        "Just received my order! It arrived in record time and the quality is simply unmatched.",
        "Join our community, sign up for our newsletter, and get 15% off your first purchase right now."
    ]

    # ----------------------------------------------------------------------
    # CONCURRENCY IMPLEMENTATION: ThreadPoolExecutor for I/O-bound API calls
    # ----------------------------------------------------------------------
    MAX_WORKERS = 20  # Adjust this number based on your API rate limits and network capacity
    
    print(f"--- Starting analysis of {len(caption_column)} captions using {MAX_WORKERS} concurrent threads ---")
    
    all_results: List[Dict[str, Any]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Map the score_caption function to every item in caption_column.
        # This executes the calls concurrently and returns results in the same order as the inputs.
        results_iterator = executor.map(score_caption, caption_column)
        
        for i, result in enumerate(results_iterator):
            caption = caption_column[i]
            
            if result:
                all_results.append(result)
                print(f"[Finished {i+1}/{len(caption_column)}] Success: '{caption[:50]}...'")
            else:
                # Create a placeholder result for failed attempts to keep the CSV format consistent
                failed_result = {
                    'original_caption': caption, 
                    'Assertive_Language': 0, 
                    'Commissive_Language': 0, 
                    'Directive_Language': 0, 
                    'Specificity': 0, 
                    'Figurative_Language': 0, 
                    'Critique': 'API_CALL_FAILED'
                }
                all_results.append(failed_result)
                print(f"[Finished {i+1}/{len(caption_column)}] Failed: '{caption[:50]}...' (Placeholder inserted)")


    print("\n\n=============== FINAL ANALYTICAL OUTPUT (CSV FORMAT) ===============")
    
    # 1. Define the exact order of the columns for the CSV header
    CSV_COLUMNS = [
        "original_caption", 
        "Assertive_Language", 
        "Commissive_Language", 
        "Directive_Language", 
        "Specificity", 
        "Figurative_Language"
    ]

    # 2. Use the csv module to format the output for robustness
    # Using StringIO allows us to capture the CSV output as a string.
    output = StringIO()
    # csv.QUOTE_ALL ensures that any string fields containing commas, quotes, or newlines 
    # are handled correctly, which is vital for the 'original_caption' and 'Critique' fields.
    writer = csv.DictWriter(output, fieldnames=CSV_COLUMNS, quoting=csv.QUOTE_ALL)
    
    # Write Header
    writer.writeheader()
    
    # Write Rows
    writer.writerows(all_results)
    
    # Print the full CSV content
    print(output.getvalue().strip())

    print("\n--- End of CSV Data ---")
