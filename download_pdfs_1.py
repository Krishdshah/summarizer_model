import os
import pandas as pd
import requests
import time

# --- Configuration ---
CSV_FILE = 'SB_publication_PMC.csv'
OUTPUT_DIR = 'data'

def download_all_pdfs():
    """
    Reads a CSV file with links, constructs the direct PDF download URL for each,
    and downloads them into the specified output directory.
    """
    # 1. Create the output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"📁 Created directory: '{OUTPUT_DIR}'")

    # 2. Read the CSV file
    try:
        df = pd.read_csv(CSV_FILE)
        if 'Link' not in df.columns:
            print("❌ Error: CSV file must have a column named 'Link'.")
            return
        page_urls = df['Link'].dropna().tolist() # Drop any empty rows
    except FileNotFoundError:
        print(f"❌ Error: The file '{CSV_FILE}' was not found in this directory.")
        return

    print(f"Found {len(page_urls)} links in the CSV file. Starting downloads...\n")

    # 3. Loop through each URL to construct the PDF link and download
    for i, page_url in enumerate(page_urls):
        print(f"--- Processing link {i+1}/{len(page_urls)} ---")
        try:
            # Ensure the URL has a trailing slash for consistent joining
            if not page_url.endswith('/'):
                page_url += '/'

            # Extract PMC ID for a clean filename (e.g., PMC4136787)
            # This is more robust than splitting by '/'
            try:
                pmc_id = [part for part in page_url.split('/') if part.startswith('PMC')][0]
                pdf_filename = f"{pmc_id}.pdf"
            except IndexError:
                print(f"⚠️  Could not extract PMC ID from URL: {page_url}. Skipping.")
                continue

            pdf_filepath = os.path.join(OUTPUT_DIR, pdf_filename)

            # Skip if the file has already been downloaded
            if os.path.exists(pdf_filepath):
                print(f"👍 '{pdf_filename}' already exists. Skipping.")
                continue

            # Construct the direct PDF download URL
            # The pattern is simply to add "pdf/" to the end of the article URL
            pdf_download_url = page_url + 'pdf/'

            print(f"⬇️  Attempting to download from: {pdf_download_url}")

            # Set headers to mimic a real browser visit
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }

            # Download the PDF file with a timeout and stream=True for large files
            pdf_response = requests.get(pdf_download_url, headers=headers, timeout=30, stream=True)
            pdf_response.raise_for_status()  # Raise an error for bad status codes (like 404)

            # Save the PDF to the data folder
            with open(pdf_filepath, 'wb') as f:
                for chunk in pdf_response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"✅ Successfully saved '{pdf_filename}'")

            # Add a small delay to be respectful to the server
            time.sleep(1)

        except requests.exceptions.HTTPError as e:
            print(f"❌ HTTP Error for {pdf_download_url}: {e}")
            print("   This might mean the PDF does not exist at this URL.")
        except requests.exceptions.RequestException as e:
            print(f"❌ Network Error for {pdf_download_url}: {e}")
        except Exception as e:
            print(f"❌ An unexpected error occurred: {e}")
        finally:
            print("-" * 35)


if __name__ == "__main__":
    download_all_pdfs()
    print("\n🎉 Script finished.")