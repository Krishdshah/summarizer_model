import os
import pandas as pd
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.common.exceptions import SessionNotCreatedException

# --- Configuration ---
CSV_FILE = 'SB_publication_PMC.csv'
OUTPUT_DIR = os.path.join(os.getcwd(), 'data')

# --- PASTE YOUR PATHS HERE ---
# 1. Path to the driver you downloaded (for version 141)
CHROME_DRIVER_PATH = os.path.join(os.getcwd(), 'chromedriver.exe')

# 2. Path to your actual chrome.exe file (from your shortcut's properties)
#    IMPORTANT: Keep the 'r' before the string and use double backslashes if needed.
CHROME_BINARY_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe" # <-- PASTE YOUR PATH HERE


def setup_driver():
    """Sets up Selenium with manual paths for BOTH the driver and the browser."""
    if not os.path.exists(CHROME_DRIVER_PATH):
        raise FileNotFoundError(f"ChromeDriver not found at: {CHROME_DRIVER_PATH}")
    if not os.path.exists(CHROME_BINARY_PATH):
        raise FileNotFoundError(f"Chrome Browser not found at: {CHROME_BINARY_PATH}")

    service = ChromeService(executable_path=CHROME_DRIVER_PATH)
    
    chrome_options = webdriver.ChromeOptions()
    # This is the crucial line that tells Selenium which browser to use
    chrome_options.binary_location = CHROME_BINARY_PATH
    
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--disable-gpu")
    prefs = {
        "download.default_directory": OUTPUT_DIR,
        "download.prompt_for_download": False,
        "plugins.always_open_pdf_externally": True
    }
    chrome_options.add_experimental_option("prefs", prefs)
    
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

# The rest of the script remains the same...
def download_all_pdfs():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"📁 Created directory: '{OUTPUT_DIR}'")

    try:
        df = pd.read_csv(CSV_FILE)
        page_urls = df['Link'].dropna().tolist()
    except FileNotFoundError:
        print(f"❌ Error: The file '{CSV_FILE}' was not found.")
        return

    print(f"Found {len(page_urls)} links. Setting up browser for download...")
    
    try:
        driver = setup_driver()
    except Exception as e:
        print(f"❌ Critical Error: Could not start the browser session.")
        print(f"   Please double-check your CHROME_DRIVER_PATH and CHROME_BINARY_PATH variables in the script.")
        print(f"   Details: {e}")
        return
        
    print("✅ Browser ready. Starting downloads...\n")

    for i, page_url in enumerate(page_urls):
        print(f"--- Processing link {i+1}/{len(page_urls)} ---")
        try:
            pdf_download_url = (page_url.strip() if page_url.strip().endswith('/') else page_url.strip() + '/') + 'pdf/'
            pmc_id = [part for part in page_url.strip().split('/') if part.startswith('PMC')][0]
            target_filename = f"{pmc_id}.pdf"
            target_filepath = os.path.join(OUTPUT_DIR, target_filename)

            if os.path.exists(target_filepath):
                print(f"👍 '{target_filename}' already exists. Skipping.")
                continue

            print(f"🔗 Navigating to: {pdf_download_url}")
            driver.get(pdf_download_url)
            
            seconds_waited = 0
            download_wait_time = 60
            
            while seconds_waited < download_wait_time:
                if any(fname.endswith('.crdownload') for fname in os.listdir(OUTPUT_DIR)):
                    time.sleep(1)
                    seconds_waited += 1
                else:
                    time.sleep(2)
                    break
            
            if seconds_waited >= download_wait_time:
                 print(f"⚠️  Download timed out for {pdf_download_url}")
                 continue

            files = [os.path.join(OUTPUT_DIR, f) for f in os.listdir(OUTPUT_DIR)]
            if not files:
                print(f"⚠️ No file was downloaded from {pdf_download_url}")
                continue
            latest_file = max(files, key=os.path.getctime)

            if latest_file.endswith('.pdf'):
                if os.path.basename(latest_file) != target_filename:
                   os.rename(latest_file, target_filepath)
                   print(f"✅ Downloaded and renamed to '{target_filename}'")
                else:
                   print(f"✅ Downloaded successfully as '{target_filename}'")
            else:
                 print(f"⚠️ Downloaded file was not a PDF. File: {os.path.basename(latest_file)}")

        except Exception as e:
            print(f"❌ An unexpected error occurred for URL {page_url}: {e}")
        finally:
            print("-" * 35)
    
    driver.quit()

if __name__ == "__main__":
    download_all_pdfs()
    print("\n🎉 Script finished.")