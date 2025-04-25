# app.py
# --- Core Imports ---
import streamlit as st
import pandas as pd
import os
import pdfplumber # For PDF extraction
import re # For keyword regex
import asyncio # For async functions
import nest_asyncio # For running async
from openai import AsyncOpenAI # OpenAI client
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification # For classification
import torch
from dotenv import load_dotenv # To load API key from .env file
import datetime # For timestamps (if needed)
import traceback # For detailed error printing
import logging # For logging errors from functions

# Configure basic logging (optional, helps debug if run from terminal)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Apply nest_asyncio patch (Important!) ---
# Allows running asyncio event loops within Streamlit/Jupyter environments
try:
    nest_asyncio.apply()
    logging.info("Applied nest_asyncio patch.")
except RuntimeError:
    logging.warning("nest_asyncio patch might already be applied or is not needed.")
except Exception as e:
    logging.error(f"Could not apply nest_asyncio: {e}")


# --- Page Configuration (Set Title and Layout) ---
st.set_page_config(page_title="RadioBrief Processor", layout="wide", initial_sidebar_state="collapsed")

# --- Title and Description ---
st.title("📰 RadioBrief: Political News Processor")
st.write("""
Upload a PDF or paste text containing French news articles.
The app will attempt to:
1. Extract text (if PDF).
2. Split the text into potential articles (using the `ftp\\n` method).
3. Identify the *first* article (skipping the very first chunk) deemed political using keywords OR a **fine-tuned classification model**.
4. Summarize the identified article using OpenAI.
5. Translate the summary into Arabic using OpenAI.
6. Display the results.
""")
st.info("ℹ️ Classification uses a fine-tuned CamemBERT model (trained on MasakhaNEWS). Summary/Translation use OpenAI GPT-3.5.")

# --- Load API Key from .env file ---
load_dotenv() # Looks for .env in the current directory
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Stop the app if the API key is missing
if not OPENAI_API_KEY:
    st.error("❌ OpenAI API key not found. Please create a `.env` file in the same folder as `app.py` with the line: OPENAI_API_KEY='your_key_here'")
    st.stop()

# --- Initialize OpenAI Client (Cached) ---
# Use st.cache_resource to load the client only once per session
@st.cache_resource
def get_openai_client():
    logging.info("Initializing OpenAI Async Client...")
    try:
        client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        logging.info("✅ OpenAI Client Initialized.")
        return client
    except Exception as e:
        st.error(f"❌ Failed to initialize OpenAI client: {e}")
        logging.error(f"OpenAI Client Init Error: {e}", exc_info=True)
        return None

async_client = get_openai_client()
if not async_client:
    st.stop() # Stop if client failed to initialize

# --- Load Fine-Tuned Classification Model (Cached) ---
# ** IMPORTANT: Assumes 'final_model' folder is in the same directory as app.py **
fine_tuned_model_path = "./final_model" # Local path relative to app.py

# Define the expected labels and mappings from your fine-tuning notebook
id2label_map = {0: 'business', 1: 'entertainment', 2: 'health', 3: 'politics', 4: 'religion', 5: 'sports', 6: 'technology'}
label2id_map = {v: k for k, v in id2label_map.items()}
num_model_labels = len(id2label_map)

@st.cache_resource # Cache the loaded model/pipeline
def load_finetuned_classifier(model_path):
    logging.info(f"Loading fine-tuned classification model from: {model_path}")
    # Check if the path exists before trying to load
    if not os.path.isdir(model_path): # Use isdir for checking folder
         st.error(f"❌ Fine-tuned model directory not found at: '{model_path}'. Please ensure the 'final_model' folder exists here.")
         return None
    try:
        # Determine device (use GPU if available)
        device_id = 0 if torch.cuda.is_available() else -1
        device_name = 'GPU 0' if device_id == 0 else 'CPU'
        logging.info(f"Attempting to load model on device: {device_name}")

        # Load model and tokenizer from the local path
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=num_model_labels,
            id2label=id2label_map,
            label2id=label2id_map
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Create the pipeline
        classifier_pipeline = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=device_id
        )
        logging.info(f"✅ Fine-tuned classifier loaded successfully on {device_name}.")
        return classifier_pipeline
    except Exception as e:
        st.error(f"❌ Failed to load fine-tuned classification model: {e}")
        logging.error(f"Model Loading Error: {e}", exc_info=True)
        return None

# Load the classifier when the app starts
classifier_finetuned = load_finetuned_classifier(fine_tuned_model_path)
# App continues even if classifier fails, but classification step will be skipped later


# --- Define Helper Functions (Copied/Adapted from Colab Notebooks) ---

# --- Configuration Variables ---
# (Copied from Cell 4 of Streamlit_App_Logic_Test_Final.ipynb)
political_keywords = [
    "Politique", "Géopolitique", "Économie politique", "Relations internationales", "Souveraineté", "Frontières", "Sécurité", "Défense", "Libéralisme", "Conflit", "Guerre", "Crise", "Instabilité", "Tensions géopolitiques", "Cessez-le-feu", "Trêve", "Élections", "Scrutin", "Parité", "Diplomatie", "Négociations", "Sommet", "Traité", "Accord", "Sanctions", "Embargo", "Coercition économique", "Aide humanitaire", "Crise humanitaire", "Droits de l'homme", "Terrorisme", "Extrémisme", "Ingérence étrangère", "Assemblée Nationale", "Sénat", "Matignon", "Bercy", "Élysée", "Quai d'Orsay", "Ministère des Affaires étrangères", "Ministre des Affaires étrangères", "France", "ONU", "Nations Unies", "Conseil de sécurité", "Résolution", "Casques bleus", "OTAN", "NATO", "Union Européenne", "UE", "Bruxelles", "Commission européenne", "Parlement européen", "G7", "G20", "OMC", "FMI", "Banque mondiale", "Union Africaine", "UA", "Cour pénale internationale", "CPI", "Cour internationale de Justice", "CIJ", "Moyen-Orient", "Proche-Orient", "Gaza", "Palestinien", "Palestine", "Cisjordanie", "Jérusalem", "Autorité palestinienne", "Israël", "Israélien", "Tel Aviv", "Liban", "Libanais", "Beyrouth", "Syrie", "Syrien", "Damas", "Jordanie", "Jordanien", "Amman", "Égypte", "Égyptien", "Le Caire", "Irak", "Irakien", "Bagdad", "Iran", "Iranien", "Téhéran", "Arabie Saoudite", "Saoudien", "Riyad", "Yémen", "Yéménite", "Sanaa", "Qatar", "Qatari", "Doha", "Émirats arabes unis", "EAU", "Émirati", "Abou Dhabi", "Dubaï", "Turquie", "Turc", "Ankara", "Afrique du Nord", "Maghreb", "Algérie", "Algérien", "Alger", "Tunisie", "Tunisien", "Tunis", "Maroc", "Marocain", "Rabat", "Libye", "Libyen", "Tripoli", "États-Unis", "USA", "Américain", "Washington", "Maison Blanche", "Pentagone", "Département d'État", "Chine", "Chinois", "Pékin", "Taïwan", "Russie", "Russe", "Moscou", "Kremlin", "Ukraine", "Ukrainien", "Kiev", "Migration", "Migrants", "Réfugiés", "Asile", "Immigration", "Immigrés", "Flux migratoires", "Frontière", "Guerre commerciale", "Commerce international", "Droit de douane", "Accords commerciaux", "Protectionnisme", "Multilateralisme", "Nucléaire", "Prolifération", "AIEA", "Énergie", "Pétrole", "Gaz", "OPEP", "Hamas", "Jihad islamique", "Hezbollah", "Houthis", "Ansar Allah", "Talibans", "Afghanistan", "État islamique", "Daech", "EI", "ISIS", "Al-Qaïda", "Extrême droite", "Populisme", "Autoritarisme", "Souverainisme", "Nationalisme", "Islamisme"
]
min_keyword_hits = 3

# --- FUNCTION DEFINITIONS ---

# 1. PDF Extraction Function (Adapted for Streamlit UploadedFile)
def extract_text_from_pdf(uploaded_file_object):
    """Extracts text from an uploaded PDF file object using pdfplumber."""
    full_text = ""
    if uploaded_file_object is None:
        return None
    try:
        # pdfplumber can often open file-like objects directly
        with pdfplumber.open(uploaded_file_object) as pdf:
            all_pages_text = []
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    all_pages_text.append(page_text)
                # else:
                #     logging.warning(f"No text extracted from page {i+1} in {uploaded_file_object.name}")
            full_text = "\\n".join(all_pages_text) # Join pages with newline
        if not full_text:
             logging.warning(f"No text could be extracted from the PDF: {uploaded_file_object.name}")
             st.warning(f"⚠️ Could not extract any text from the PDF: {uploaded_file_object.name}")
             return None
        logging.info(f"✅ Extracted text from PDF: {uploaded_file_object.name}")
        return full_text
    except Exception as e:
        st.error(f"❌ Error extracting text from PDF '{uploaded_file_object.name}': {e}")
        logging.error(f"PDF Extraction Error: {e}", exc_info=True)
        return None

# 2. Article Splitting Function (Using 'ftp\\n' separator)
# (Copied from Cell 4 of Streamlit_App_Logic_Test_Final.ipynb)
def smart_split_articles(full_text):
    """Splits text into potential articles based on the 'ftp\\n' separator
       and applies basic cleaning and filtering."""
    if not full_text: return []
    logging.info("--- Running smart_split_articles (Splitting by 'ftp\\n') ---")
    # Basic cleaning (remove form feed)
    cleaned_text = full_text.replace('\\x0c', '\\n')
    # Split the text wherever 'ftp\n' occurs (case-insensitive)
    split_pattern = r'ftp\\n'
    possible_articles = re.split(split_pattern, cleaned_text, flags=re.IGNORECASE)
    logging.info(f"Number of chunks after splitting by '{split_pattern}': {len(possible_articles)}")
    # Filter out empty strings and apply length/content checks
    final_articles = []
    min_article_length = 150 # Minimum characters
    logging.info(f"Filtering chunks shorter than {min_article_length} characters...")
    for i, article_chunk in enumerate(possible_articles):
        if article_chunk:
            trimmed_chunk = article_chunk.strip()
            # Remove potential leftover page/section headers
            lines = trimmed_chunk.split('\\n')
            if lines and re.match(r'^[A-Z\sIVXLCDM]+\s*\d*$', lines[0].strip()):
                 # logging.info(f"  -> Removing potential header from chunk {i}: '{lines[0].strip()}'")
                 trimmed_chunk = '\\n'.join(lines[1:]).strip()
            # Check length and content
            if len(trimmed_chunk) > min_article_length and any(c.isalpha() for c in trimmed_chunk):
                 final_articles.append(trimmed_chunk)
    logging.info(f"  (Splitting resulted in {len(final_articles)} potential articles after filtering)")
    logging.info("--- Finished smart_split_articles ---")
    return final_articles

# 3. Summarization Function (with Input Truncation)
# (Copied from Cell 4 of Streamlit_App_Logic_Test_Final.ipynb)
async def async_summarize_article(
    aclient: AsyncOpenAI,
    text: str,
    max_lines: int = 4,
    style: str = "journalistique radio",
    tone: str = "neutre et informatif",
    focus: str = "faits politiques",
    max_input_chars: int = 12000 # Safety limit
) -> str:
    """Asynchronously summarizes an article using the provided AsyncOpenAI client,
       truncating input text if it's too long."""
    if not text or not isinstance(text, str) or len(text.strip()) < 50:
        logging.warning(f"Skipping summary for short/invalid text: {text[:50]}...")
        return "Résumé non disponible (Texte d'entrée invalide)."
    text_to_summarize = text
    truncated = False
    if len(text) > max_input_chars:
        logging.warning(f"Input text length ({len(text)}) exceeds limit ({max_input_chars}). Truncating.")
        st.warning(f"⚠️ Input article text too long ({len(text)} chars), truncating to {max_input_chars} chars for summary.")
        text_to_summarize = text[:max_input_chars]
        truncated = True
    prompt = f"""
Résume cet article en {max_lines} lignes maximum, dans un style {style},
en mettant l'accent sur les {focus}. Utilise un ton {tone}.

Texte de l'article :
{text_to_summarize}
"""
    logging.info(f"--> Preparing async summary request for article snippet: {text_to_summarize[:50]}...")
    if not aclient: return "Résumé non disponible (Erreur Client OpenAI)."
    try:
        response = await aclient.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200
        )
        summary = response.choices[0].message.content.strip()
        logging.info(f"--> Received async summary for article snippet: {text_to_summarize[:50]}...")
        if truncated:
             summary += "\\n\\n*(Note: Résumé basé sur le début de l'article car le texte original était trop long.)*"
        return summary
    except Exception as e:
        logging.error(f"Error summarizing article asynchronously: {e}", exc_info=True)
        error_code = getattr(e, 'code', None) or getattr(getattr(e, 'body', {}), 'code', None)
        if error_code == 'context_length_exceeded':
            st.error(f"❌ Error during summary: Input text still too long for the model even after truncation attempt.")
            return "Résumé non disponible (Erreur: Texte trop long)."
        else:
            st.error(f"❌ Error during async summary request: {e}")
            return "Résumé non disponible (Erreur API)."

# 4. Translation Function
# (Copied from Cell 4 of Streamlit_App_Logic_Test_Final.ipynb)
async def async_translate_to_arabic(
    aclient: AsyncOpenAI,
    text: str,
    style: str = "journalistique",
    formality: str = "formel",
    target_audience: str = "public général",
    context: str = "actualités"
) -> str:
    """Asynchronously translates text into Modern Standard Arabic."""
    prompt = f"""
Traduis ce texte en arabe standard moderne (MSA),
avec une précision élevée et un style {style},
adapté à un public {target_audience} dans un contexte de {context}.
Utilise un registre {formality}.
Conserve la structure et le sens du texte original.

Texte à traduire :
{text}
"""
    logging.info(f"--> Preparing async translation request for text snippet: {text[:50]}...")
    if not aclient: return "Traduction non disponible (Erreur Client OpenAI)."
    # Robust check for summary errors before translating
    if not text or any(err_msg in text for err_msg in ["Erreur API", "non disponible", "Texte d'entrée invalide", "Texte trop long"]):
         logging.warning(f"Skipping translation because input text indicates summary error: '{text[:50]}...'")
         return "Traduction non disponible car le résumé initial n'était pas disponible ou contenait une erreur."
    try:
        response = await aclient.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300 # Allow more tokens for translation
        )
        translation = response.choices[0].message.content.strip()
        logging.info(f"--> Received async translation for text snippet: {text[:50]}...")
        return translation
    except Exception as e:
        logging.error(f"Error translating text asynchronously: {e}", exc_info=True)
        st.error(f"❌ Error during async translation request: {e}")
        return "Traduction non disponible (Erreur API)."

# 5. Classification Function (using the fine-tuned model)
# (Copied from Cell 4 of Streamlit_App_Logic_Test_Final.ipynb)
def classify_topic_finetuned(pipeline_obj, text_to_classify):
    """Classifies text using the fine-tuned model pipeline."""
    if not pipeline_obj:
        logging.warning("Fine-tuned classifier not loaded/available. Skipping classification.")
        return {"label": "Classifier Unavailable", "score": 0.0}
    if not text_to_classify or not isinstance(text_to_classify, str) or len(text_to_classify.strip()) < 20:
        logging.warning(f"Skipping fine-tuned classification for invalid/short input text: '{str(text_to_classify)[:50]}...'")
        return {"label": "Input Too Short/Invalid", "score": 0.0}
    try:
        logging.info(f"  (Classifying with fine-tuned model: {text_to_classify[:50]}...)")
        max_chars_for_classifier = 2000 # Limit input characters
        result = pipeline_obj(text_to_classify[:max_chars_for_classifier])
        if result and isinstance(result, list):
            top_prediction = result[0]
            label = top_prediction.get('label', 'Error')
            score = top_prediction.get('score', 0.0)
            logging.info(f"  (Fine-tuned classification: Label='{label}', Score={score:.4f})")
            return {"label": label, "score": score}
        else:
             logging.warning(f"Unexpected result format from fine-tuned pipeline: {result}")
             return {"label": "Classification Error (Format)", "score": 0.0}
    except Exception as e:
        logging.error(f"Error during fine-tuned classification: {e}", exc_info=True)
        st.error(f"❌ Error during fine-tuned classification: {e}")
        return {"label": "Classification Error (Runtime)", "score": 0.0}

# --- End Helper Functions ---


# --- Streamlit UI Elements ---
st.divider()
st.subheader("1. Input Text or PDF")

col1, col2 = st.columns(2)

with col1:
    # Option 1: File Uploader
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf", label_visibility="collapsed")

with col2:
    # Option 2: Text Area
    pasted_text_input = st.text_area("Or paste text here:", height=150, placeholder="Paste article text...", label_visibility="collapsed")

# Process Button
st.divider()
if st.button("✨ Process Input", type="primary", use_container_width=True):
    full_text = None
    source_name = ""
    processing_error = False

    # --- Get Input ---
    if uploaded_file is not None:
        # Give priority to uploaded file if both are provided
        with st.spinner(f"⏳ Extracting text from {uploaded_file.name}..."):
            full_text = extract_text_from_pdf(uploaded_file) # Use function adapted for UploadedFile
            source_name = uploaded_file.name
            if full_text is None:
                processing_error = True # Error handled in function
    elif pasted_text_input.strip():
        full_text = pasted_text_input
        source_name = "Pasted Text"
        logging.info("Processing pasted text.")
    else:
        st.warning("⚠️ Please upload a PDF or paste text first.")
        processing_error = True

    # --- Process if Input is Valid ---
    if full_text and not processing_error:
        st.info(f"ℹ️ Processing content from: {source_name}")
        with st.spinner("⏳ Splitting articles and finding first political one..."):
            # 1. Split Articles
            articles = smart_split_articles(full_text) # Use the 'ftp\\n' version
            if not articles:
                st.error("❌ Could not split the text into articles using the 'ftp\\n' separator.")
                processing_error = True
            else:
                st.write(f"Found {len(articles)} potential articles after splitting. Identifying the first political one...")

                # 2. Find First Political Article
                first_political_article_text = None
                first_political_topic = "N/A"
                first_political_score = 0.0
                found_article = False

                # Compile keyword regex once
                keyword_pattern = r'(?i)\\b(?:' + '|'.join(re.escape(kw) for kw in political_keywords) + r')\\b'
                keyword_regex = re.compile(keyword_pattern)
                relevant_masakha_labels = ['politics'] # Only 'politics' now

                # Loop through articles, SKIPPING THE FIRST CHUNK (index 0)
                for i, current_article_text in enumerate(articles):
                    if i == 0: # Skip the first chunk
                        logging.info("Skipping chunk 0 (likely preamble).")
                        continue

                    logging.info(f"--- Checking Article Chunk {i+1}/{len(articles)} ---")
                    is_political = False
                    assigned_topic_source = "N/A"

                    # Classification
                    if classifier_finetuned:
                        classification_result = classify_topic_finetuned(classifier_finetuned, current_article_text)
                        predicted_topic_label = classification_result.get('label', 'Error')
                        predicted_topic_score = classification_result.get('score', 0.0)
                        if predicted_topic_label in relevant_masakha_labels:
                            is_political = True
                            assigned_topic_source = f"Classifier ({predicted_topic_label})"
                            logging.info(f"-> Chunk {i+1} identified as '{predicted_topic_label}' by classifier (Score: {predicted_topic_score:.2f}).")
                    else:
                        predicted_topic_label = "Classifier Unavailable"
                        logging.info("-> Skipping classification (model not loaded). Checking keywords...")

                    # Keyword Check
                    keyword_matches = keyword_regex.findall(current_article_text)
                    number_of_hits = len(keyword_matches)
                    if number_of_hits >= min_keyword_hits:
                        logging.info(f"-> Chunk {i+1} met keyword threshold ({number_of_hits} hits).")
                        if not is_political:
                            is_political = True
                            assigned_topic_source = f"Keywords ({number_of_hits} hits)"

                    # Store and break if political
                    if is_political:
                        st.success(f"✅ Political article found (Chunk Index {i+1}) identified via {assigned_topic_source}.")
                        first_political_article_text = current_article_text
                        first_political_topic = predicted_topic_label # Store the classifier's prediction
                        first_political_score = predicted_topic_score
                        found_article = True
                        break # Stop after finding the first one

        # --- Process the first political article if found ---
        if found_article and not processing_error:
            st.divider()
            st.subheader("2. Processing First Political Article Found")
            try:
                # Run async tasks (Summarize & Translate)
                # Define the async function to run
                async def process_article_async():
                    summary_task = asyncio.create_task(async_summarize_article(async_client, first_political_article_text))
                    summary = await summary_task
                    translation = "Translation skipped (summary error)."
                    if summary and not any(err_msg in summary for err_msg in ["Erreur API", "non disponible", "Texte d'entrée invalide", "Texte trop long"]):
                        translation_task = asyncio.create_task(async_translate_to_arabic(async_client, summary))
                        translation = await translation_task
                    else:
                        translation = f"Traduction non effectuée ({summary})."
                    return summary, translation

                # Run the async processing function in Streamlit
                with st.spinner("⏳ Running Summary and Translation via OpenAI..."):
                    # Get or create an event loop for the current thread
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)

                    # Run the coroutine using the loop
                    summary_result, translation_result = loop.run_until_complete(process_article_async())


                # --- Display Results ---
                st.subheader("3. Results")

                st.markdown(f"**📌 Detected Topic (Fine-Tuned Model):**")
                topic_display = f"{first_political_topic} (Score: {first_political_score:.2f})"
                if first_political_topic == "Classifier Unavailable":
                    topic_display = "Classifier Unavailable (Identified by Keywords)"
                elif assigned_topic_source.startswith("Keywords"):
                     topic_display += f" - Identified via Keywords ({number_of_hits} hits)" # Show keyword trigger too

                st.info(topic_display)

                st.markdown(f"**📝 Summary (French):**")
                if "non disponible" in summary_result or "Erreur" in summary_result:
                    st.error(summary_result)
                else:
                    st.success(summary_result)

                st.markdown(f"**🌍 Translation (Arabic):**")
                if "non disponible" in translation_result or "Erreur" in translation_result or "non effectuée" in translation_result:
                     st.warning(translation_result)
                else:
                    # Display Arabic text with right-to-left alignment using markdown
                    st.markdown(f'<div style="direction: rtl; text-align: right; font-size: 1.1em;">{translation_result}</div>', unsafe_allow_html=True)


                with st.expander("📰 Show Original Article Snippet"):
                    st.text_area("Article Snippet", value=first_political_article_text[:2000]+"...", height=200, disabled=True, label_visibility="collapsed")

            except Exception as e:
                st.error(f"❌ An error occurred during API processing: {e}")
                logging.error(f"API Processing Error: {e}", exc_info=True)

        elif not processing_error: # Only show if no prior errors occurred
            st.warning("⚠️ No political articles meeting the criteria were found in the input after checking all chunks.")

# --- Footer or additional info ---
st.divider()
st.caption("RadioBrief App v1.1 - Using Fine-Tuned Classifier & 'ftp\\n' Split")
