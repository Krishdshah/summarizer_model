import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk # --- ADD THIS IMPORT ---

# --- Sumy Summarizer Imports (same as before) ---
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

# --- Configuration ---
LANGUAGE = "english"
SENTENCES_COUNT = 7
DATA_FILE = 'processed_data_vectors.pkl'
MODEL_NAME = 'all-MiniLM-L6-v2'

# --- Initialize the FastAPI app ---
app = FastAPI(
    title="Semantic PDF Search API",
    description="An API to search and summarize research papers.",
    version="1.0.0",
)

# --- ADD THIS ENTIRE MIDDLEWARE SECTION ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- App State for Models and Data ---
app.state.documents = None
app.state.doc_embeddings = None
app.state.model = None

# --- Startup Event ---
@app.on_event("startup")
def load_models_and_data():
    """Load all necessary data and models when the API starts."""
    # --- NEW: Programmatically download NLTK data ---
    try:
        nltk.data.find("tokenizers/punkt")
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        print("Downloading NLTK data packages...")
        nltk.download("punkt")
        nltk.download("punkt_tab")
        print("NLTK data downloaded successfully.")
    # -----------------------------------------------

    print("Loading data and models...")
    app.state.model = SentenceTransformer(MODEL_NAME)
    
    try:
        with open(DATA_FILE, 'rb') as f:
            app.state.documents = pickle.load(f)
            app.state.doc_embeddings = np.array([doc['embedding'] for doc in app.state.documents])
        print("Data and models loaded successfully.")
    except FileNotFoundError:
        print(f"FATAL ERROR: Data file '{DATA_FILE}' not found. Please run preprocess.py.")
        app.state.documents = []


def summarize_text(text: str) -> str:
    """Generates a summary for the given text using Sumy."""
    parser = PlaintextParser.from_string(text, Tokenizer(LANGUAGE))
    stemmer = Stemmer(LANGUAGE)
    summarizer = Summarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)
    summary_sentences = summarizer(parser.document, SENTENCES_COUNT)
    return " ".join([str(sentence) for sentence in summary_sentences])


# --- API Endpoints ---

@app.get("/search/", summary="Search for relevant documents")
def search_documents(q: str):
    """
    Performs semantic search based on the query `q`.

    Returns a list of all relevant documents, sorted by relevance score.
    """
    if not app.state.documents:
        raise HTTPException(status_code=503, detail="Server is not ready, data not loaded.")

    query_embedding = app.state.model.encode(q, convert_to_tensor=False).reshape(1, -1)
    similarities = cosine_similarity(query_embedding, app.state.doc_embeddings)[0]

    # --- CHANGED: Get ALL results, not just the top 10 ---
    # We create a list of tuples (index, score)
    all_results_with_scores = list(enumerate(similarities))
    
    # Filter out results with low similarity (optional, but recommended)
    relevant_results = [res for res in all_results_with_scores if res[1] > 0.3]
    
    # Sort the relevant results by score in descending order
    sorted_results = sorted(relevant_results, key=lambda item: item[1], reverse=True)
    
    # Format the final response
    results = []
    for index, score in sorted_results:
        doc = app.state.documents[index]
        results.append({
            "title": doc['title'],
            "link": doc['link'],
            "pdf_filename": doc['pdf_filename'],
            "relevance_score": float(score)
        })
        
    return {"query": q, "results": results}


@app.get("/summarize/", summary="Generate a summary for a specific document")
def get_summary(filename: str):
    """
    Finds a document by its `pdf_filename` and generates a summary.
    """
    if not app.state.documents:
        raise HTTPException(status_code=503, detail="Server is not ready, data not loaded.")

    doc_found = next((doc for doc in app.state.documents if doc['pdf_filename'] == filename), None)
    
    if doc_found:
        summary = summarize_text(doc_found['text'])
        return {
            "pdf_filename": filename,
            "title": doc_found['title'],
            "summary": summary
        }
    else:
        raise HTTPException(status_code=404, detail=f"Document with filename '{filename}' not found.")