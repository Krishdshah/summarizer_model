import streamlit as st
import pickle

# --- Sumy Summarizer Setup ---
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

# --- Configuration for Summarizer ---
LANGUAGE = "english"
SENTENCES_COUNT = 7

@st.cache_data
def load_data(file_path):
    """Loads the preprocessed data from the pickle file."""
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

def summarize_text(text):
    """Generates a summary for the given text using Sumy."""
    parser = PlaintextParser.from_string(text, Tokenizer(LANGUAGE))
    stemmer = Stemmer(LANGUAGE)
    summarizer = Summarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)
    summary_sentences = summarizer(parser.document, SENTENCES_COUNT)
    summary = " ".join([str(sentence) for sentence in summary_sentences])
    return summary

# --- Main Application Logic ---
st.set_page_config(layout="wide")

st.title("🔬 Semantic PDF Search & Summarizer")
st.markdown("Search through your downloaded research papers on space and microgravity.")

documents = load_data('processed_data.pkl')

if documents is None:
    st.error("Error: `processed_data.pkl` not found. Please run `preprocess.py` first!")
else:
    search_query = st.text_input("Enter your search query (e.g., microgravity, osteoclasts, spaceflight)", "")

    if search_query:
        results = [
            doc for doc in documents 
            if search_query.lower() in doc['text'].lower()
        ]

        st.subheader(f"Found {len(results)} matching documents:")

        if not results:
            st.warning("No documents found matching your query. Try another term.")
        else:
            # Use enumerate to get a unique index 'i' for each result
            for i, doc in enumerate(results):
                with st.expander(f"**{doc['title']}**"):
                    st.markdown("---")
                    
                    # --- THE FIX IS HERE ---
                    # We add the unique index 'i' to the key to prevent duplicates.
                    if st.button("Generate Summary", key=f"{doc['pdf_filename']}_{i}"):
                        st.subheader("📄 Summary")
                        with st.spinner("Generating summary..."):
                            summary = summarize_text(doc['text'])
                            st.write(summary)
                    
                    st.markdown("---")
                    st.markdown(f"**[➡️ Read Original Article Online]({doc['link']})**")