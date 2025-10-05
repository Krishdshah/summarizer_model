import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
# ADD THIS IMPORT
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

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
# This section enables CORS, allowing your frontend to call the API.
# The wildcard ["*"] allows requests from any origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- App State for Models and Data ---
# We will load models into the app's state on startup to avoid reloading them on every request.
app.state.documents = None
app.state.doc_embeddings = None
app.state.model = None

# --- Startup Event ---
@app.on_event("startup")
def load_models_and_data():
    """Load all necessary data and models when the API starts."""
    print("Loading data and models...")
    # Load the sentence transformer model
    app.state.model = SentenceTransformer(MODEL_NAME)
    
    # Load the preprocessed data
    try:
        with open(DATA_FILE, 'rb') as f:
            app.state.documents = pickle.load(f)
            # Pre-calculate embeddings into a NumPy array for performance
            app.state.doc_embeddings = np.array([doc['embedding'] for doc in app.state.documents])
        print("Data and models loaded successfully.")
    except FileNotFoundError:
        print(f"FATAL ERROR: Data file '{DATA_FILE}' not found. Please run preprocess.py.")
        app.state.documents = [] # Ensure app can start but will return empty results


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

    Returns a list of the top 10 most relevant documents, including their title,
    link, and a relevance score.
    """
    if not app.state.documents:
        raise HTTPException(status_code=503, detail="Server is not ready, data not loaded.")

    # 1. Encode the user's query
    query_embedding = app.state.model.encode(q, convert_to_tensor=False).reshape(1, -1)

    # 2. Calculate similarities
    similarities = cosine_similarity(query_embedding, app.state.doc_embeddings)[0]

    # 3. Get top 10 results
    top_indices = np.argsort(similarities)[-10:][::-1]

    # Format results
    results = []
    for i in top_indices:
        doc = app.state.documents[i]
        results.append({
            "title": doc['title'],
            "link": doc['link'],
            "pdf_filename": doc['pdf_filename'],
            "relevance_score": float(similarities[i])
        })
        
    return {"query": q, "results": results}


@app.get("/summarize/", summary="Generate a summary for a specific document")
def get_summary(filename: str):
    """
    Finds a document by its `pdf_filename` and generates a summary.
    """
    if not app.state.documents:
        raise HTTPException(status_code=503, detail="Server is not ready, data not loaded.")

    # Find the document by filename
    doc_found = next((doc for doc in app.state.documents if doc['pdf_filename'] == filename), None)
    
    if doc_found:
        summary = summarize_text(doc_found['text'])
        return {
            "pdf_filename": filename,
            "title": doc_found['title'],
            "summary": summary
        }
    else:
        # If no document is found, raise a 404 error
        raise HTTPException(status_code=404, detail=f"Document with filename '{filename}' not found.")