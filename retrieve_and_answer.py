import os
import base64
import logging
import pickle
from typing import List, Dict, Any, Tuple
import numpy as np
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import openai
import faiss

from utils.logging_config import setup_logging

# ------------------------------------------------------
# Setup Logging
# ------------------------------------------------------
setup_logging()
log = logging.getLogger("retrieve")

# ------------------------------------------------------
# Load Environment
# ------------------------------------------------------
load_dotenv()

MONGO_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("MONGODB_DBNAME", "gpt_integration")
COLLECTION_NAME = os.getenv("MONGODB_COLLECTION", "documents")

FAISS_PATH = os.getenv("FAISS_PATH", "./knowledge_pack/index_hnsw.faiss")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

if not MONGO_URI or not OPENAI_API_KEY:
    raise RuntimeError("MONGODB_URI or OPENAI_API_KEY missing in .env")

openai.api_key = OPENAI_API_KEY

# ------------------------------------------------------
# Initialize MongoDB and Model
# ------------------------------------------------------
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
col = db[COLLECTION_NAME]

log.info("Connected to MongoDB successfully.")

embed_model = SentenceTransformer(EMBED_MODEL_NAME)
log.info(f"Loaded embedding model: {EMBED_MODEL_NAME}")

# ------------------------------------------------------
# Load FAISS Index
# ------------------------------------------------------
faiss_index = None
faiss_id_map = {}

try:
    if os.path.exists(FAISS_PATH):
        faiss_index = faiss.read_index(FAISS_PATH)
        with open(FAISS_PATH + ".pkl", "rb") as f:
            saved = pickle.load(f)
            # If pickle contains tuple (index, map)
            if isinstance(saved, tuple) and len(saved) == 2:
                _, faiss_id_map = saved
            else:
                faiss_id_map = saved
        log.info("FAISS index and ID map loaded successfully.")
    else:
        log.warning("FAISS index not found locally.")
except Exception as e:
    log.error(f"Failed to load FAISS index: {e}")

# ------------------------------------------------------
# Helper Functions
# ------------------------------------------------------
def _encode(queries: List[str]) -> np.ndarray:
    return embed_model.encode(queries, convert_to_numpy=True, normalize_embeddings=True).astype("float32")

def _read_image_as_dataurl(path: str) -> str:
    """Convert local image file to base64 data URL."""
    try:
        with open(path, "rb") as f:
            return "data:image/png;base64," + base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return ""

# ------------------------------------------------------
# Core Retrieval
# ------------------------------------------------------
TOP_K = 4

def find_relevant(query: str) -> Tuple[List[str], List[str]]:
    q_vec = _encode([query])[0]
    contexts, images = [], []

    if faiss_index is not None and faiss_index.ntotal > 0:
        try:
            D, I = faiss_index.search(np.expand_dims(q_vec, 0), TOP_K * 2)
            matched_ids = [faiss_id_map.get(idx) for idx in I[0] if idx in faiss_id_map]
            hits = list(col.find({"doc_id": {"$in": matched_ids}}))
        except Exception as e:
            log.warning(f"FAISS search failed: {e}")
            hits = []
    else:
        log.warning("FAISS index not loaded or empty.")
        hits = []

    # Fallback if FAISS unavailable — search MongoDB directly
    if not hits:
        log.info("Falling back to MongoDB full-text search.")
        hits = list(col.find({"$text": {"$search": query}})) if col.index_information() else []

    for h in hits:
        text_chunk = h.get("text_for_search") or h.get("text") or h.get("ocr_text")
        if text_chunk and len(contexts) < TOP_K:
            contexts.append(text_chunk[:1200])
        if h.get("type") == "image" and h.get("image_path"):
            img_path = h["image_path"]
            if os.path.exists(img_path):
                img_data = _read_image_as_dataurl(img_path)
                if img_data:
                    images.append(img_data)

    # Deduplicate
    seen = set()
    unique_contexts = [c for c in contexts if not (c in seen or seen.add(c))]

    return unique_contexts[:TOP_K], images[:TOP_K]

# ------------------------------------------------------
# Prompt and LLM Call
# ------------------------------------------------------
def build_prompt(contexts: List[str], query: str) -> str:
    ctx = "\n\n".join(contexts)
    return (
        "You are a concise and helpful assistant.\n"
        "Answer ONLY from the context provided.\n"
        "If context is insufficient, say 'Information not found in knowledge base.'\n\n"
        f"Context:\n{ctx}\n\nQuestion: {query}\nAnswer:"
    )

def generate_answer(prompt: str) -> str:
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        log.error(f"OpenAI call failed: {e}")
        return "⚠️ Error generating answer. Please try again."

# ------------------------------------------------------
# Main Handler
# ------------------------------------------------------
def answer(question: str) -> Dict[str, Any]:
    if not question.strip():
        return {"answer": "Please ask a valid question.", "contexts": [], "images": []}
    contexts, images = find_relevant(question)
    prompt = build_prompt(contexts, question)
    text = generate_answer(prompt)
    return {"answer": text, "contexts": contexts, "images": images}

# ------------------------------------------------------
# CLI Mode
# ------------------------------------------------------
if __name__ == "__main__":
    q = input("Enter question: ")
    result = answer(q)
    print("\nAnswer:", result["answer"])