# RadioBrief – AI-Powered Multilingual Press Review Assistant

RadioBrief is a professional tool designed for radio newsrooms to automate their daily press review process.  
It generates multilingual summaries from French news articles, translating them into Arabic and preparing outputs for radio broadcast, social media, and internal use.

## Features

- Summarizes French press articles using GPT
- Translates to Arabic in a journalistic style
- Stores results in structured CSV format
- Integrated with Google Drive for persistent storage
- Ready for fine-tuning to match radio newsroom needs

## How It Works

1. Open the `RadioBrief_Starter_Notebook.ipynb` in Google Colab
2. Paste or upload your article(s)
3. Automatically generate:
   - French summary
   - Arabic translation
4. Save everything to your Google Drive
5. Repeat daily with new articles

## Tech Stack

- Google Colab
- OpenAI GPT-3.5 / GPT-4
- Python (Pandas, datetime)
- Google Drive API (via Colab)
- Optionally: LangChain for prompt pipelines

## Installation

```bash
pip install -r requirements.txt

Files
RadioBrief_Starter_Notebook.ipynb – main working notebook

RadioBrief_Daily_Workflow.pdf – daily usage guide

requirements.txt – libraries needed

Author
Sara Jabali – Final Project at Developers Institute
AI Bootcamp – GenAI 2025

Status
Minimum Viable Product (MVP) complete
Streamlit app version in progress
