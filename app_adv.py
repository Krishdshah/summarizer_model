import streamlit as st
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- Sumy Summarizer (remains the same) ---
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

# --- Configuration ---
LANGUAGE = "english"
SENTENCES_COUNT = 7
DATA_FILE = 'processed_data_vectors.pkl'

# --- Caching Models and Data ---
@st.cache_resource
def load_model(model_name):
    """Loads the sentence transformer model and caches it."""
    return SentenceTransformer(model_name)

@st.cache_data
def load_data(file_path):
    """Loads the preprocessed data."""
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

# Summarizer function (remains the same)
def summarize_text(text):
    parser = PlaintextParser.from_string(text, Tokenizer(LANGUAGE))
    stemmer = Stemmer(LANGUAGE)
    summarizer = Summarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)
    summary_sentences = summarizer(parser.document, SENTENCES_COUNT)
    return " ".join([str(sentence) for sentence in summary_sentences])

# --- Main App ---
st.set_page_config(layout="wide")
st.title("🔬 Semantic PDF Search & Summarizer")
st.markdown("Ask a question or enter keywords to find relevant research papers.")

# Load data and the model
documents = load_data(DATA_FILE)
model = load_model('all-MiniLM-L6-v2')

if documents is None:
    st.error(f"Error: `{DATA_FILE}` not found. Please run `preprocess.py` first!")
else:
    # Separate embeddings from the document info
    doc_embeddings = np.array([doc['embedding'] for doc in documents])
    
    search_query = st.text_input("Enter your search query (e.g., what are the effects of microgravity on bone?)", "")

    if search_query:
        # --- NEW Semantic Search Logic ---
        # 1. Encode the user's query into a vector
        query_embedding = model.encode(search_query, convert_to_tensor=False).reshape(1, -1)

        # 2. Calculate similarity between the query and all document embeddings
        similarities = cosine_similarity(query_embedding, doc_embeddings)[0]

        # 3. Get the top 10 most similar documents
        # We use argsort to get the indices of the highest scores
        top_indices = np.argsort(similarities)[-10:][::-1]
        
        results = [(documents[i], similarities[i]) for i in top_indices]

        st.subheader(f"Top {len(results)} relevant documents:")

        # --- Display Results ---
        for i, (doc, score) in enumerate(results):
            with st.expander(f"**{doc['title']}** (Relevance: {score:.2f})"):
                st.markdown("---")
                
                if st.button("Generate Summary", key=f"{doc['pdf_filename']}_{i}"):
                    st.subheader("📄 Summary")
                    with st.spinner("Generating summary..."):
                        summary = summarize_text(doc['text'])
                        st.write(summary)
                
                st.markdown("---")
                st.markdown(f"**[➡️ Read Original Article Online]({doc['link']})**")