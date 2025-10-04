import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# --- Configuration ---
CSV_FILE = 'SB_publication_PMC.csv'
OUTPUT_DIR = 'data'
BASE_URL = 'https://www.ncbi.nlm.nih.gov' # Base URL for constructing full links

def download_all_pdfs():
    """
    Reads a CSV file with links, finds PDF download URLs on each page,
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
        page_urls = df['Link'].tolist()
    except FileNotFoundError:
        print(f"❌ Error: The file '{CSV_FILE}' was not found in this directory.")
        return

    print(f"Found {len(page_urls)} links in the CSV file. Starting downloads...\n")

    # 3. Loop through each URL to find and download the PDF
    for i, page_url in enumerate(page_urls):
        print(f"--- Processing link {i+1}/{len(page_urls)} ---")
        try:
            # Generate a clean filename from the URL (e.g., PMC4136787.pdf)
            pmc_id = page_url.strip('/').split('/')[-1]
            pdf_filename = f"{pmc_id}.pdf"
            pdf_filepath = os.path.join(OUTPUT_DIR, pdf_filename)

            # Skip if the file has already been downloaded
            if os.path.exists(pdf_filepath):
                print(f"👍 '{pdf_filename}' already exists. Skipping.")
                continue

            print(f"🔗 Visiting: {page_url}")

            # Fetch the HTML content of the page
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(page_url, headers=headers, timeout=15)
            response.raise_for_status()  # Raise an error for bad status codes (4xx or 5xx)

            # Parse the HTML to find the PDF link
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # The PDF link is in an 'a' tag with a specific data-ga-action attribute
            pdf_link_tag = soup.find('a', {'data-ga-action': 'pdf'})

            if pdf_link_tag and pdf_link_tag.has_attr('href'):
                relative_pdf_url = pdf_link_tag['href']
                
                # Construct the absolute URL (e.g., https://.../file.pdf)
                full_pdf_url = urljoin(BASE_URL, relative_pdf_url)
                
                print(f"⬇️  Downloading PDF from: {full_pdf_url}")

                # Download the PDF file
                pdf_response = requests.get(full_pdf_url, stream=True, headers=headers, timeout=30)
                pdf_response.raise_for_status()

                # Save the PDF to the data folder
                with open(pdf_filepath, 'wb') as f:
                    for chunk in pdf_response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                print(f"✅ Successfully saved '{pdf_filename}'")
            else:
                print(f"⚠️  Could not find a PDF download link on the page.")

        except requests.exceptions.RequestException as e:
            print(f"❌ Network Error for {page_url}: {e}")
        except Exception as e:
            print(f"❌ An unexpected error occurred: {e}")
        finally:
            print("-" * 35)


if __name__ == "__main__":
    download_all_pdfs()
    print("\n🎉 Script finished.")