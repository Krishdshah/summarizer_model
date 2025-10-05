import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

# --- Configuration ---
DATA_FILE = 'processed_data_vectors.pkl'
MODEL_NAME = 'all-MiniLM-L6-v2'

# --- Initialize the FastAPI app ---
app = FastAPI(
    title="Semantic PDF Search API",
    description="An API to search and summarize research papers.",
    version="1.0.0",
)

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
    # (The NLTK downloader can be removed, but it's harmless to keep)
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

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

# --- API Endpoints ---
@app.get("/search/", summary="Search for relevant documents")
def search_documents(q: str):
    if not app.state.documents:
        raise HTTPException(status_code=503, detail="Server is not ready, data not loaded.")

    query_embedding = app.state.model.encode(q, convert_to_tensor=False).reshape(1, -1)
    similarities = cosine_similarity(query_embedding, app.state.doc_embeddings)[0]
    
    all_results_with_scores = list(enumerate(similarities))
    relevant_results = [res for res in all_results_with_scores if res[1] > 0.3]
    sorted_results = sorted(relevant_results, key=lambda item: item[1], reverse=True)
    
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
    Finds a document by its `pdf_filename` and returns its pre-computed summary.
    """
    if not app.state.documents:
        raise HTTPException(status_code=503, detail="Server is not ready, data not loaded.")

    doc_found = next((doc for doc in app.state.documents if doc['pdf_filename'] == filename), None)
    
    if doc_found:
        # --- THE FIX IS HERE ---
        # We now simply return the summary that's already in the data.
        return {
            "pdf_filename": filename,
            "title": doc_found['title'],
            "summary": doc_found.get('summary', 'Summary not available.') # .get for safety
        }
    else:
        raise HTTPException(status_code=404, detail=f"Document with filename '{filename}' not found.")