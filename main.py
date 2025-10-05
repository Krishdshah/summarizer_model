import os
import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import uvicorn

# --- Configuration ---
DATA_FILE = 'processed_data_vectors.pkl'
MODEL_NAME = 'all-MiniLM-L6-v2'

# --- Initialize the FastAPI app ---
app = FastAPI(
    title="Semantic PDF Search API",
    description="An API to search and summarize research papers.",
    version="1.0.0",
)

# --- Allow all origins (for frontend integration) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- App State for Models and Data ---
app.state.model = None
app.state.documents = None
app.state.doc_embeddings = None


# --- Lazy Loader Function ---
def ensure_loaded():
    """Ensure model and data are loaded (lazy load)."""
    if app.state.model is None:
        print("Loading SentenceTransformer model...")
        app.state.model = SentenceTransformer(MODEL_NAME)
        print("Model loaded successfully.")

    if app.state.documents is None:
        print("Loading document data...")
        try:
            with open(DATA_FILE, 'rb') as f:
                app.state.documents = pickle.load(f)
                app.state.doc_embeddings = np.array(
                    [doc['embedding'] for doc in app.state.documents]
                )
            print(f"Loaded {len(app.state.documents)} documents successfully.")
        except FileNotFoundError:
            print(f"FATAL ERROR: Data file '{DATA_FILE}' not found.")
            app.state.documents = []
            app.state.doc_embeddings = np.array([])


# --- Routes ---

@app.get("/")
def root():
    return {"message": "Semantic PDF Search API is running."}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/search/", summary="Search for relevant documents")
def search_documents(q: str):
    ensure_loaded()  # Lazy load model and data

    if not app.state.documents:
        raise HTTPException(status_code=503, detail="Server not ready — data not loaded.")

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
    ensure_loaded()  # Lazy load model and data

    if not app.state.documents:
        raise HTTPException(status_code=503, detail="Server not ready — data not loaded.")

    doc_found = next((doc for doc in app.state.documents if doc['pdf_filename'] == filename), None)
    if doc_found:
        return {
            "pdf_filename": filename,
            "title": doc_found['title'],
            "summary": doc_found.get('summary', 'Summary not available.')
        }
    else:
        raise HTTPException(status_code=404, detail=f"Document with filename '{filename}' not found.")


# --- Run App (Render-compatible) ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
