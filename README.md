# RadioBrief – AI-Powered Multilingual Press Review Assistant

RadioBrief is a professional tool designed for radio newsrooms to automate their daily press review process. It processes French news articles to generate summaries, Arabic translations, and topic classifications, saving the results in a structured CSV format.

## Key Components

This project is divided into several components, each addressing a specific aspect of the workflow:

1.  **Main Article Processing Pipeline**

    * Notebook: `Perfect_results_WithReport.ipynb`
    * Description: This notebook handles the core article processing steps, including:
        * PDF text extraction
        * Article segmentation
        * Summarization (using OpenAI)
        * Translation to Arabic (using OpenAI)
        * Data organization and saving to CSV
    * Key Dependencies:
        * OpenAI API key (user-provided, see setup instructions)
        * Python libraries: `pdfplumber`, `pandas`, `asyncio`, etc. (see `requirements.txt`)
    * Usage: Run this notebook in Google Colab to process articles and generate output.

2.  **Fine-Tuning for Topic Classification**

    * Notebook: `FineTune_Topic_Classifier.ipynb`
    * Description: This notebook demonstrates the fine-tuning of a pre-trained CamemBERT model to improve the accuracy of topic classification for French news articles. It focuses on adapting the model to a specific set of topic categories.
    * Key Functionality:
        * Dataset loading and preprocessing
        * Model fine-tuning
        * Evaluation (accuracy, confusion matrix)
    * Requirements:
        * Google Colab environment
        * Hugging Face datasets library
        * Sufficient training data (MasakhaNEWS dataset used as a demonstration)
    * Relevance to Project: This notebook illustrates the methodology for creating a more accurate topic classifier, which could be integrated into the main pipeline.

3.  **Streamlit Web Application**

    * Script: `app.py`
    * Description: This script provides a user-friendly interface for interacting with the RadioBrief system. It allows users to upload PDF files or paste article text and view the processed results.
    * Key Functionality:
        * File upload/text input
        * Triggering the article processing pipeline
        * Displaying summaries, translations, and topic classifications
    * Requirements:
        * Python 3.7+
        * Streamlit library
        * Dependencies listed in `requirements.txt`
        * OpenAI API key (user-provided)
    * Running the app:
        1.  Install dependencies: `pip install -r requirements.txt`
        2.  Run the app: `streamlit run app.py`

4.  **Streamlit App Logic Test Notebook**

    * Notebook: `Streamlit_App_Logic_Test_Final.ipynb`
    * Description: This notebook contains code for testing the core logic of the Streamlit app within a Colab environment.
    * Key Functionality:
        * Tests the article processing, summarization, and translation functions.
    * Requirements:
        * Google Colab environment
        * Libraries: pandas, openai, transformers, etc. (see `requirements.txt`)
    * Usage: Run all cells to test the app logic.

## Installation
```bash
pip install -r requirements.txt

Files
Perfect_results_WithReport.ipynb – Main article processing notebook
FineTune_Topic_Classifier.ipynb – Notebook for fine-tuning the topic classifier
app.py – Streamlit web application script
Streamlit_App_Logic_Test_Final.ipynb - Notebook for testing the Streamlit app logic
requirements.txt – Python library dependencies

Author
Sara Jabali – Final Project at Developers Institute
AI Bootcamp – GenAI 2025

Status
Minimum Viable Product (MVP) complete
Streamlit app version in progress
