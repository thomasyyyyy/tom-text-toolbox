import re
import json
from pathlib import Path

# Hardcoded category-to-file mapping
category_files = {
    "overstated": Path(r"tom_text_toolbox\dictionaries\md_files\ovrst.md"),
    "understated": Path(r"tom_text_toolbox\dictionaries\md_files\undrst.md"),
    "power": Path(r"tom_text_toolbox\dictionaries\md_files\power.md"),
    "wellbeing":Path(r"tom_text_toolbox\dictionaries\md_files\wlbtot.md")
}

def extract_terms(md_file):
    terms = []
    with md_file.open("r", encoding="utf-8") as f:
        for line in f:
            # Only process table rows (skip headers and separators)
            if line.startswith("|") and not set(line.strip()) <= {"|", "-", " "}:
                cols = [c.strip() for c in line.strip().split("|")[1:-1]]
                if len(cols) >= 2:
                    term = re.sub(r"#\d+", "", cols[1]).lower()  # remove #1, #2, etc. and lowercase
                    terms.append(term)
    return terms

# Build dictionary
terms_by_category = {cat: extract_terms(file) for cat, file in category_files.items()}

# Save JSON
output_file = Path(r"md.json")
with output_file.open("w", encoding="utf-8") as f:
    json.dump(terms_by_category, f, indent=2, ensure_ascii=False)

print(f"Saved JSON to {output_file}")
