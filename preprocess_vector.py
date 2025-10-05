import os
import pandas as pd
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError
import pickle
from sentence_transformers import SentenceTransformer

# --- Sumy Summarizer Imports ---
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

# --- Configuration ---
CSV_FILE = 'SB_publication_PMC.csv'
PDF_FOLDER = 'data'
OUTPUT_FILE = 'processed_data_vectors.pkl'
MODEL_NAME = 'all-MiniLM-L6-v2'
LANGUAGE = "english"
SENTENCES_COUNT = 7 # The length of the pre-computed summaries

def extract_text_from_pdf(pdf_path):
    """
    Robustly extracts text from a single PDF file, handling errors.
    """
    text = ""
    try:
        with open(pdf_path, 'rb') as f:
            reader = PdfReader(f)
            if reader.is_encrypted:
                print(f"  ⚠️  Skipping encrypted PDF: {os.path.basename(pdf_path)}")
                return None
            
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            
            if not text.strip():
                print(f"  ⚠️  No text extracted from {os.path.basename(pdf_path)}. It might be image-based.")
            return text
    except PdfReadError:
        print(f"  ❌ Error: Could not read {os.path.basename(pdf_path)}. File may be corrupted. Skipping.")
        return None
    except Exception as e:
        print(f"  ❌ An unexpected error occurred with {os.path.basename(pdf_path)}: {e}. Skipping.")
        return None

def summarize_text(text: str) -> str:
    """
    Generates a concise summary for a given string of text.
    """
    parser = PlaintextParser.from_string(text, Tokenizer(LANGUAGE))
    stemmer = Stemmer(LANGUAGE)
    summarizer = Summarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)
    summary_sentences = summarizer(parser.document, SENTENCES_COUNT)
    return " ".join([str(sentence) for sentence in summary_sentences])

def run_full_preprocessing():
    """
    Reads PDFs, extracts text, generates vector embeddings, creates summaries,
    and saves everything to a single data file for the API.
    """
    print(f"🚀 Starting full preprocessing with vectors and summaries...")
    print(f"   Using model: '{MODEL_NAME}'")
    print("   This process will take a significant amount of time. Please be patient.")

    # Load the AI model for creating embeddings
    model = SentenceTransformer(MODEL_NAME)
    
    # Load the source CSV
    df = pd.read_csv(CSV_FILE, dtype={'Title': 'string', 'Link': 'string'}).dropna(subset=['Title', 'Link'])
    
    all_documents = []

    # Loop through each document 
    for index, row in df.iterrows():
        title = row['Title']
        link = row['Link']
        
        try:
            pmc_id = [part for part in link.strip().split('/') if part.startswith('PMC')][0]
            pdf_filename = f"{pmc_id}.pdf"
            pdf_path = os.path.join(PDF_FOLDER, pdf_filename)
        except (IndexError, AttributeError):
            continue

        if os.path.exists(pdf_path):
            print(f"Processing '{pdf_filename}'...")
            
            full_text = extract_text_from_pdf(pdf_path)
            
            
            if full_text and len(full_text) > 200:
                print("  - Generating vector embedding...")
                embedding = model.encode(full_text, convert_to_tensor=False)
                
                print("  - Generating summary...")
                summary = summarize_text(full_text)
                
               
                document_data = {
                    'title': title,
                    'link': link,
                    'pdf_filename': pdf_filename,
                    'text': full_text,       
                    'embedding': embedding,
                    'summary': summary
                }
                all_documents.append(document_data)

    
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(all_documents, f)

    print(f"\n✅ Full preprocessing complete!")
    print(f"   {len(all_documents)} documents were successfully processed, embedded, and summarized.")
    print(f"   Data saved to '{OUTPUT_FILE}'")

if __name__ == "__main__":
    run_full_preprocessing()