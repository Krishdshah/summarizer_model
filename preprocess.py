import os
import pandas as pd
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError # Import the specific error
import pickle

# --- Configuration ---
CSV_FILE = 'SB_publication_PMC.csv'
PDF_FOLDER = 'data'
OUTPUT_FILE = 'processed_data.pkl'

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a single PDF file with enhanced error handling.
    """
    text = ""
    try:
        with open(pdf_path, 'rb') as f:
            reader = PdfReader(f)
            # Check for encryption
            if reader.is_encrypted:
                print(f"  ⚠️  Skipping encrypted PDF: {os.path.basename(pdf_path)}")
                return None
            
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            # If no text was extracted at all, it's a sign of a problem
            if not text.strip():
                print(f"  ⚠️  Warning: No text extracted from {os.path.basename(pdf_path)}. It might be an image-based PDF.")
            return text
    # Catch specific PDF reading errors
    except PdfReadError:
        print(f"  ❌ Error: Could not read {os.path.basename(pdf_path)}. The file may be corrupted. Skipping.")
        return None
    # Catch any other unexpected errors
    except Exception as e:
        print(f"  ❌ An unexpected error occurred with {os.path.basename(pdf_path)}: {e}. Skipping.")
        return None

def run_preprocessing():
    """
    Reads the CSV, finds corresponding PDFs, extracts text,
    and saves everything into a single file for the app to use.
    """
    print("🚀 Starting preprocessing...")
    
    try:
        # Read CSV, ensuring the 'Title' and 'Link' columns are treated as strings
        df = pd.read_csv(CSV_FILE, dtype={'Title': 'string', 'Link': 'string'}).dropna(subset=['Title', 'Link'])
    except FileNotFoundError:
        print(f"Error: The file '{CSV_FILE}' was not found.")
        return

    all_documents = []

    for index, row in df.iterrows():
        title = row['Title']
        link = row['Link']
        
        try:
            pmc_id = [part for part in link.strip().split('/') if part.startswith('PMC')][0]
            pdf_filename = f"{pmc_id}.pdf"
            pdf_path = os.path.join(PDF_FOLDER, pdf_filename)
        except (IndexError, AttributeError):
            print(f"  ⚠️  Could not parse link to get PMC ID: {link}. Skipping row.")
            continue

        print(f"Processing '{pdf_filename}'...")

        if os.path.exists(pdf_path):
            full_text = extract_text_from_pdf(pdf_path)
            
            if full_text:
                document_data = {
                    'title': title,
                    'link': link,
                    'pdf_filename': pdf_filename,
                    'text': full_text
                }
                all_documents.append(document_data)
        else:
            # This is just a warning, not an error
            pass # We can suppress the "not found" warning to make the output cleaner

    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(all_documents, f)

    print(f"\n✅ Preprocessing complete! {len(all_documents)} documents were successfully processed.")
    print(f"Data saved to '{OUTPUT_FILE}'")

if __name__ == "__main__":
    run_preprocessing()