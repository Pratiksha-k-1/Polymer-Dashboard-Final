import glob
import re
import pandas as pd
from transformers import pipeline
from pathlib import Path

# -----------------------------------------------------------
# PATHS (relative, clean)
# -----------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
RAW_TEXT_DIR = BASE_DIR / "raw"
SOURCE_FILE = BASE_DIR / "sources" / "polymer_sentiment_sources_new.xlsx"
OUTPUT_FILE = BASE_DIR / "weekly_market_sentiment.csv"

# -----------------------------------------------------------
# Load pretrained financial sentiment model
# -----------------------------------------------------------
classifier = pipeline(
    "text-classification",
    model="ProsusAI/finbert",
    return_all_scores=False
)

# -----------------------------------------------------------
# Helper: extract date from filename
# Example: Week3_2024-05-17.txt → 2024-05-17
# -----------------------------------------------------------
def extract_date_from_filename(filename: str) -> pd.Timestamp:
    match = re.search(r"\d{4}-\d{2}-\d{2}", filename)
    if not match:
        raise ValueError(f"No date found in filename: {filename}")
    return pd.to_datetime(match.group())

# -----------------------------------------------------------
# Optional: load source metadata (for traceability)
# -----------------------------------------------------------
if SOURCE_FILE.exists():
    source_df = pd.read_excel(SOURCE_FILE)
else:
    source_df = None

# -----------------------------------------------------------
# Sentiment label → numeric mapping
# -----------------------------------------------------------
label_map = {
    "positive": 1,
    "neutral": 0,
    "negative": -1
}

rows = []

# -----------------------------------------------------------
# Process weekly sentiment text files
# -----------------------------------------------------------
for txt_file in sorted(RAW_TEXT_DIR.glob("Week*_*.txt")):
    with open(txt_file, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if len(text) < 50:
        # Skip empty or meaningless files
        continue

    date = extract_date_from_filename(txt_file.name)

    result = classifier(text)[0]
    label = result["label"].lower()
    confidence = result["score"]

    sentiment_index = round(label_map[label] * confidence, 3)

    rows.append({
        "date": date,
        "sentiment_index": sentiment_index,
        "sentiment_label": label.capitalize(),
        "model_confidence": round(confidence, 3)
    })

# -----------------------------------------------------------
# Create weekly sentiment dataframe
# -----------------------------------------------------------
sentiment_df = pd.DataFrame(rows).sort_values("date")

# -----------------------------------------------------------
# Save final output
# -----------------------------------------------------------
sentiment_df.to_csv(OUTPUT_FILE, index=False)

print("Sentiment preprocessing completed successfully.")
print(sentiment_df)
