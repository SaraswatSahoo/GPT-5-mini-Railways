import os
import base64
import logging
from typing import List, Dict, Any, Tuple
import numpy as np
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import openai
import faiss
import pickle # Added

from utils.logging_config import setup_logging

setup_logging()
log = logging.getLogger("retrieve")

load_dotenv()
MONGO_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("MONGODB_DBNAME", "gpt_integration")
COLLECTION_NAME = os.getenv("MONGODB_COLLECTION", "documents")

if not MONGO_URI:
    raise RuntimeError("MONGODB_URI not set")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
col = db[COLLECTION_NAME]

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")
openai.api_key = OPENAI_API_KEY

EMBED_MODEL_NAME = "all-MiniLM-L6-v2" # Standardized model
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

FAISS_PATH = os.getenv("FAISS_PATH", "./knowledge_pack/index_hnsw.faiss")
faiss_index = None
faiss_id_map: Dict[int, str] = {} # Added
FAISS_AVAILABLE = False

try:
    if os.path.exists(FAISS_PATH):
        faiss_index = faiss.read_index(FAISS_PATH)
        with open(FAISS_PATH + ".pkl", "rb") as f:
            faiss_id_map = pickle.load(f)
        log.info("FAISS index and map loaded successfully.")
        FAISS_AVAILABLE = True
except Exception as e:
    log.warning(f"Could not read FAISS index or map: {e}")
    FAISS_AVAILABLE = False

TOP_K = 4
IMAGE_KEYWORDS = {"image","diagram","photo","picture","figure","chart","graph","देख","तस्वीर","चित्र","कैसा"}

def fallback_mongodb_search(query: str, top_k: int = TOP_K) -> List[Dict]:
    try:
        results = list(col.find(
            {"$text": {"$search": query}},
            {"score": {"$meta": "textScore"}}
        ).sort([("score", {"$meta": "textScore"})]).limit(top_k))
        if results:
            return results
        else:
            return list(col.find().limit(top_k))
    except Exception as e:
        log.error(f"MongoDB fallback search failed: {e}")
        return []

def _encode(q: List[str]) -> np.ndarray:
    return embed_model.encode(q, convert_to_numpy=True, normalize_embeddings=True).astype("float32")

def _get_docs_by_ids(ids: List[int]) -> List[Dict]:
    # FAISS indices don't map directly to Mongo IDs unless we store mapping; fallback to text-match search
    # We'll fetch best matches from Atlas as fallback
    return []

def _read_image_as_dataurl(path: str) -> str:
    try:
        with open(path, "rb") as f:
            b = f.read()
            return "data:image/jpeg;base64," + base64.b64encode(b).decode("utf-8")
    except Exception:
        return ""

def find_relevant(query: str) -> Tuple[List[str], List[str]]:
    q_vec = _encode([query])[0]
    contexts = []
    images = []

    if FAISS_AVAILABLE and faiss_index is not None and faiss_index.ntotal > 0:
        try:
            D, I = faiss_index.search(np.expand_dims(q_vec, 0), TOP_K * 2)
            mongo_doc_ids = [faiss_id_map[idx] for idx in I[0] if idx in faiss_id_map]
            hits = list(col.find({"doc_id": {"$in": mongo_doc_ids}}))
            log.info(f"Retrieved {len(hits)} documents from FAISS search.")
        except Exception as e:
            log.exception(f"FAISS search failed: {e}. Falling back to MongoDB.")
            hits = fallback_mongodb_search(query)
    else:
        log.info("Local FAISS index not available or empty. Using fallback search.")
        hits = fallback_mongodb_search(query)

    # --- 2. Process Hits ---
    # The Atlas vector search is removed here to simplify and rely on the local FAISS index as intended.
    # If Atlas search is desired, it should be re-introduced with proper configuration.
    
    # Process the hits from FAISS/Mongo retrieval
    for h in hits:
        txt = h.get("text_for_search") or h.get("text") or h.get("ocr_text") or ""
        if txt and len(contexts) < TOP_K:
            contexts.append(txt[:1200]) # Truncate context
        
        # Images are also handled here if they were retrieved
        if h.get("type") == "image" and h.get("image_path") and len(images) < TOP_K:
            # Image path is relative to the `ingest_to_mongodb.py` execution environment
            full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", h["image_path"].lstrip("./"))
            data_url = _read_image_as_dataurl(full_path)
            if data_url:
                images.append(data_url)
    
    # Dedupe contexts (optional, but good practice)
    out_contexts = []
    seen_contexts = set()
    for c in contexts:
        if c not in seen_contexts:
            seen_contexts.add(c)
            out_contexts.append(c)
    
    return out_contexts[:TOP_K], images[:TOP_K]

def build_prompt(contexts: List[str], query: str) -> str:
    context_text = "\n\n".join(contexts) if contexts else ""
    return (
        "You are a helpful assistant. If the user uses Hindi, answer in Hindi; if they use English, answer in English.\n"
        "Use only the provided context to answer. If you do not have enough information, say so concisely.\n\n"
        f"Context:\n{context_text}\n\nQuestion: {query}\nAnswer:"
    )

def generate_answer(prompt: str) -> str:
    try:
        # Changed to a real, available model
        resp = openai.chat.completions.create(
            model="gpt-3.5-turbo", # Changed from gpt-5-mini
            messages=[{"role":"user","content":prompt}],
            temperature=0.2
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        log.exception("GPT call failed")
        return "I encountered an error while generating the answer."

def answer(question: str) -> Dict[str,Any]:
    if not question or not question.strip():
        return {"answer":"Please ask a non-empty question.", "images":[], "contexts":[]}
    contexts, images = find_relevant(question)
    prompt = build_prompt(contexts, question)
    text = generate_answer(prompt)
    return {"answer": text, "images": images, "contexts": contexts}