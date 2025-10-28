import os
import json
import faiss
import numpy as np
from dotenv import load_dotenv
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from google.cloud import storage
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="GPT-5 Mini Railway RAG API", description="Retrieval-Augmented Generation Server", version="1.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# MongoDB setup
MONGO_URI = os.getenv("MONGODB_URI")
MONGO_DBNAME = os.getenv("MONGODB_DBNAME", "gpt_integration")
MONGO_COLLECTION = os.getenv("MONGODB_COLLECTION", "documents")

mongo_client = MongoClient(MONGO_URI)
db = mongo_client[MONGO_DBNAME]
collection = db[MONGO_COLLECTION]

# Embedding model
embed_model = SentenceTransformer(os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"))

# FAISS index setup (optional fallback)
FAISS_PATH = os.getenv("FAISS_PATH", "./knowledge_pack/index_hnsw.faiss")
index = None
if os.path.exists(FAISS_PATH):
    index = faiss.read_index(FAISS_PATH)

# GCS setup
storage_client = None
if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    storage_client = storage.Client.from_service_account_json(os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
    bucket_name = os.getenv("GCS_BUCKET_NAME")
else:
    print("⚠️ Google Cloud credentials not found. Skipping GCS setup.")


# ---------------------- Helper Functions ----------------------

def get_top_documents(query, top_k=3):
    """Retrieve top relevant documents from MongoDB using vector similarity."""
    query_embedding = embed_model.encode([query])[0].tolist()

    # Retrieve all stored embeddings from MongoDB
    docs = list(collection.find({}, {"text": 1, "embedding": 1, "source": 1}))
    if not docs:
        return []

    embeddings = np.array([d["embedding"] for d in docs])
    query_vec = np.array(query_embedding)

    # Compute cosine similarity
    similarity = np.dot(embeddings, query_vec) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_vec)
    )

    top_indices = similarity.argsort()[-top_k:][::-1]
    top_docs = [docs[i] for i in top_indices]
    return top_docs


def construct_prompt(contexts, question):
    """Constructs the final prompt for OpenAI."""
    combined_context = "\n\n".join(
        [f"Source: {doc.get('source', 'Unknown')}\nContent: {doc['text']}" for doc in contexts]
    )
    prompt = f"""You are a helpful assistant for answering queries about railway information.

Use the following retrieved context to answer the question concisely and accurately.

Context:
{combined_context}

Question: {question}

Answer:"""
    return prompt


def call_openai(prompt):
    """Call OpenAI model to generate an answer."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a knowledgeable assistant for Indian Railways data."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=500,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return "Sorry, I encountered an issue while generating a response."


# ---------------------- FastAPI Routes ----------------------

@app.get("/ping")
def ping():
    return {"status": "ok", "message": "Server is running successfully."}


@app.get("/retrieve_and_answer")
def retrieve_and_answer(question: str = Query(..., description="User's query text")):
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    top_docs = get_top_documents(question, top_k=3)
    if not top_docs:
        raise HTTPException(status_code=404, detail="No relevant documents found in the database.")

    prompt = construct_prompt(top_docs, question)
    answer = call_openai(prompt)

    return {
        "question": question,
        "answer": answer,
        "sources": [d.get("source", "Unknown") for d in top_docs],
    }


@app.get("/list_gcs_pdfs")
def list_gcs_pdfs():
    """List available PDFs in the GCS bucket."""
    if not storage_client:
        raise HTTPException(status_code=500, detail="GCS not configured.")

    try:
        bucket = storage_client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix="pdfs/")
        pdf_files = [blob.name for blob in blobs if blob.name.lower().endswith(".pdf")]
        return {"pdfs": pdf_files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list PDFs: {e}")


# ---------------------- Main ----------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "retrieve_and_answer:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )