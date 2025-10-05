import os
import pandas as pd
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError
import pickle
from sentence_transformers import SentenceTransformer

# --- Configuration ---
CSV_FILE = 'SB_publication_PMC.csv'
PDF_FOLDER = 'data'
OUTPUT_FILE = 'processed_data_vectors.pkl' # New output file
MODEL_NAME = 'all-MiniLM-L6-v2' # A fast and effective model

def extract_text_from_pdf(pdf_path):
    # (This function remains the same as before)
    text = ""
    try:
        with open(pdf_path, 'rb') as f:
            reader = PdfReader(f)
            if reader.is_encrypted: return None
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text: text += page_text + "\n"
            return text
    except Exception:
        return None

def run_preprocessing_with_vectors():
    """
    Extracts text and also creates vector embeddings for semantic search.
    """
    print(f" Starting preprocessing with vector generation using '{MODEL_NAME}'...")
    print("This may take a few minutes...")

    # 1. Load the AI model
    # The model will be downloaded from the internet on the first run
    model = SentenceTransformer(MODEL_NAME)
    
    df = pd.read_csv(CSV_FILE, dtype={'Title': 'string', 'Link': 'string'}).dropna(subset=['Title', 'Link'])
    all_documents = []

    # 2. Loop through documents
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
            
            if full_text:
                # 3. Generate a vector embedding for the full text
                # This vector represents the "meaning" of the entire document
                embedding = model.encode(full_text, convert_to_tensor=False)
                
                document_data = {
                    'title': title,
                    'link': link,
                    'pdf_filename': pdf_filename,
                    'text': full_text,
                    'embedding': embedding # 4. Store the vector
                }
                all_documents.append(document_data)

    # 5. Save the new data structure to a new file
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(all_documents, f)

    print(f"\n✅ Preprocessing complete! {len(all_documents)} documents processed and embedded.")
    print(f"Data saved to '{OUTPUT_FILE}'")

if __name__ == "__main__":
    run_preprocessing_with_vectors()