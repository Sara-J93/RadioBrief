# 📘 RadioBrief Project – Daily Summarizer & Translator (Colab + OpenAI + Drive)

# ✅ SECTION 1: Install & Setup
!pip install openai==0.28 --quiet

# ✅ SECTION 2: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# ✅ SECTION 3: Import Libraries & Load API Key
import openai
import pandas as pd
import os
from datetime import datetime

# Load the OpenAI API key from Google Drive (safe way)
with open("/content/drive/MyDrive/RadioBrief/openai_key.txt") as f:
    openai.api_key = f.read().strip()

# ✅ SECTION 4: Load Existing CSV (or create if not found)
drive_path = "/content/drive/MyDrive/RadioBrief"
csv_path = f"{drive_path}/radio_summaries.csv"

try:
    df = pd.read_csv(csv_path)
    print("✅ Loaded existing summary file.")
except FileNotFoundError:
    df = pd.DataFrame(columns=["Date", "Article", "Summary_FR", "Translation_AR", "Topic"])
    print("📁 No file found – starting a new table.")

# ✅ SECTION 5: Define GPT Functions (Summarize + Translate)

def summarize_article(text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{
            "role": "user",
            "content": f"Résume cet article en 3 lignes dans un style journalistique :\n\n{text}"
        }]
    )
    return response['choices'][0]['message']['content']

def translate_to_arabic(text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{
            "role": "user",
            "content": f"Traduis ce texte en arabe littéraire, en gardant le style journalistique :\n\n{text}"
        }]
    )
    return response['choices'][0]['message']['content']

# ✅ SECTION 6: Paste Your Article Text Here
article_text = """Collez votre article ici..."""

# ✅ SECTION 7: Run Summary + Translation
summary = summarize_article(article_text)
arabic_translation = translate_to_arabic(summary)

# ✅ SECTION 8: Add Entry to the Table
new_row = {
    "Date": datetime.now().strftime("%Y-%m-%d"),
    "Article": article_text,
    "Summary_FR": summary,
    "Translation_AR": arabic_translation,
    "Topic": "À compléter",  # e.g. Gaza, Diplomatie...
}
df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

# ✅ SECTION 9: Save to Google Drive
os.makedirs(drive_path, exist_ok=True)
df.to_csv(csv_path, index=False)
print(f"✅ Data saved to: {csv_path}")