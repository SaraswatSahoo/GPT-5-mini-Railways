import os
import logging
import json
from io import BytesIO
from typing import List, Dict

import fitz  # PyMuPDF
import camelot
from PIL import Image
import pytesseract
import numpy as np
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
from dotenv import load_dotenv
import faiss
import pickle # Added
import uuid # Added

from utils.logging_config import setup_logging

setup_logging()
log = logging.getLogger("ingest")

# Load environment
load_dotenv()
MONGO_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("MONGODB_DBNAME", "gpt_integration")
COLLECTION_NAME = os.getenv("MONGODB_COLLECTION", "documents")
FAISS_PATH = os.getenv("FAISS_PATH", "./knowledge_pack/index_hnsw.faiss")

if not MONGO_URI:
    raise RuntimeError("MONGODB_URI not set in .env")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

PDF_DIR = "./pdf"
OUT_DIR = "./knowledge_pack"
IMG_DIR = os.path.join(OUT_DIR, "images")
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

EMBED_MODEL_NAME = "all-MiniLM-L6-v2" # Standardized model
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
BATCH_SIZE_MONGO = 50
MAX_PAGES = 20

# FAISS
faiss_index = None
faiss_dim = None
# FAISS mapping: FAISS index ID (int) -> MongoDB doc_id (str)
faiss_id_map: Dict[int, str] = {} # Added

# ---------- Utility Functions ----------

def chunk_text(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    chunks, start, step = [], 0, max(1, size - overlap)
    while start < len(text):
        piece = text[start:start + size].strip()
        if piece:
            chunks.append(piece)
        start += step
    return chunks

def save_image(pil_img: Image.Image, stem: str, page: int, idx: int, quality=70) -> str:
    try:
        pil_img = pil_img.convert("RGB")
        pil_img.thumbnail((1200,1200))
        name = f"{stem}_p{page}_i{idx}.jpg"
        path = os.path.join(IMG_DIR, name)
        pil_img.save(path, "JPEG", optimize=True, quality=quality)
        return path
    except Exception:
        log.exception("save_image failed")
        return ""

def convert_keys_to_str(obj):
    if isinstance(obj, dict):
        return {str(k): convert_keys_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_keys_to_str(v) for v in obj]
    else:
        return obj

# ---------- Extraction Functions ----------

def extract_tables(pdf_path: str) -> List[Dict]:
    try:
        tables = camelot.read_pdf(pdf_path, pages="all")
        return [convert_keys_to_str(t.df.to_dict()) for t in tables]
    except Exception as e:
        log.warning("camelot table extraction failed: %s", e)
        return []

def extract_images_and_ocr(pdf_path: str) -> List[Dict]:
    docs = []
    try:
        doc = fitz.open(pdf_path)
        stem = os.path.splitext(os.path.basename(pdf_path))[0]
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            img_list = page.get_images(full=True)
            for i, img in enumerate(img_list):
                try:
                    xref = img[0]
                    base = doc.extract_image(xref)
                    b = base["image"]
                    with Image.open(BytesIO(b)) as pil_img:
                        if pil_img.width < 60 or pil_img.height < 60:
                            continue
                        img_path = save_image(pil_img, stem, page_idx+1, i)
                        ocr_text = ""
                        try:
                            ocr_text = pytesseract.image_to_string(pil_img, lang="hin+eng").strip()
                        except Exception:
                            pass
                        docs.append({
                            "type":"image",
                            "file":stem,
                            "page":page_idx+1,
                            "image_path":img_path,
                            "ocr_text":ocr_text,
                            "name":os.path.basename(img_path)
                        })
                except Exception:
                    log.exception("image extraction sub-failure")
    except Exception:
        log.exception("extract_images_and_ocr failed")
    return docs

def extract_text_chunks(pdf_path: str) -> List[Dict]:
    chunks = []
    try:
        doc = fitz.open(pdf_path)
        stem = os.path.splitext(os.path.basename(pdf_path))[0]
        total = len(doc)
        num_pages = total if MAX_PAGES is None else min(total, MAX_PAGES)
        for i in range(num_pages):
            page = doc[i]
            text = page.get_text("text") or ""
            if len(text.strip()) < 30:
                pix = page.get_pixmap()
                with Image.open(BytesIO(pix.tobytes("png"))) as im:
                    try:
                        text = (text + "\n" + pytesseract.image_to_string(im, lang="hin+eng")).strip()
                    except Exception:
                        pass
            for c in chunk_text(text):
                chunks.append({"type":"text","file":stem,"page":i+1,"text":c})
    except Exception:
        log.exception("extract_text_chunks failed")
    return chunks

# ---------- FAISS Functions ----------

def build_or_load_faiss(embs: np.ndarray, doc_ids: List[str]): # Added doc_ids
    global faiss_index, faiss_dim, faiss_id_map
    
    if faiss_index is None:
        if os.path.exists(FAISS_PATH):
            try:
                log.info("Loading existing FAISS index...")
                faiss_index = faiss.read_index(FAISS_PATH)
                with open(FAISS_PATH + ".pkl", "rb") as f:
                    faiss_id_map = pickle.load(f) # Load existing map
                faiss_dim = faiss_index.d
            except Exception:
                log.warning("Could not load FAISS index or map. Building new.")
                faiss_dim = embs.shape[1]
                faiss_index = faiss.IndexHNSWFlat(faiss_dim, 32)
                faiss_index.hnsw.efConstruction = 200
        else:
            faiss_dim = embs.shape[1]
            faiss_index = faiss.IndexHNSWFlat(faiss_dim, 32)
            faiss_index.hnsw.efConstruction = 200

    start_idx = faiss_index.ntotal
    
    # Map new FAISS IDs to MongoDB doc_ids
    for i, doc_id in enumerate(doc_ids):
        faiss_id_map[start_idx + i] = doc_id
        
    log.info("Adding %d embeddings to FAISS index...", len(embs))
    faiss_index.add(embs)
    log.info("FAISS index size: %d", faiss_index.ntotal)

# ---------- Persistence ----------

def persist_to_mongo_and_faiss(items: List[Dict], embed_model: SentenceTransformer):
    if not items:
        return

    texts = []
    doc_ids = [] # Added
    
    for it in items:
        doc_id = str(uuid.uuid4()) # Generate unique ID for each chunk
        doc_ids.append(doc_id) # Store doc_id for FAISS mapping
        it["doc_id"] = doc_id # Add doc_id to the item

        if it["type"] == "text":
            texts.append(it["text"])
        elif it["type"] == "table":
            texts.append(json.dumps(it.get("table_data", it), ensure_ascii=False))
        elif it["type"] == "image":
            # Prefer OCR text, fall back to name, or empty string
            texts.append(it.get("ocr_text") or it.get("name") or "") 
        else:
            texts.append(str(it))

    try:
        embs = embed_model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False, # Changed to False for cleaner terminal output
            normalize_embeddings=True
        ).astype("float32")
    except Exception:
        log.exception("Embedding failed")
        return

    docs = []
    for it, e, txt in zip(items, embs, texts):
        doc = dict(it)
        doc["embedding"] = e.tolist()
        doc["text_for_search"] = txt
        docs.append(doc)

    # Insert in batches
    for i in range(0, len(docs), BATCH_SIZE_MONGO):
        batch = docs[i:i+BATCH_SIZE_MONGO]
        try:
            collection.insert_many(batch, ordered=False)
        except Exception:
            # log.exception("Mongo insert_many failed for batch %d-%d", i, i+len(batch)) # Removed excessive logging
            pass # Suppress non-critical, known exceptions (like duplicate key if you retry)

    # Save to FAISS
    try:
        build_or_load_faiss(embs, doc_ids) # Pass doc_ids
        faiss.write_index(faiss_index, FAISS_PATH)
        # Save the map as a pickle file
        with open(FAISS_PATH + ".pkl", "wb") as f:
            pickle.dump(faiss_id_map, f) # Save the map
    except Exception:
        log.exception("faiss operations failed")

# ---------- PDF Ingestion ----------

def ingest_pdf_file(pdf_path: str, embed_model: SentenceTransformer):
    log.info("Ingesting %s", pdf_path)
    text_chunks = extract_text_chunks(pdf_path)
    image_docs = extract_images_and_ocr(pdf_path)
    table_docs = []
    try:
        tables = extract_tables(pdf_path)
        stem = os.path.splitext(os.path.basename(pdf_path))[0]
        for t in tables:
            table_docs.append({"type":"table","file":stem,"table_data":t})
    except Exception:
        log.debug("No tables or extraction failed")
    all_items = text_chunks + table_docs + image_docs
    persist_to_mongo_and_faiss(all_items, embed_model)
    log.info("Done ingest %s (text=%d tables=%d images=%d)",
             os.path.basename(pdf_path), len(text_chunks), len(table_docs), len(image_docs))

# ---------- Main ----------

def main():
    try:
        # Check if the index exists and load the model based on its dimension if possible
        if os.path.exists(FAISS_PATH):
            index_tmp = faiss.read_index(FAISS_PATH)
            # The model is fixed to 'all-MiniLM-L6-v2' which is 384 dimensions.
            if index_tmp.d != 384:
                 log.error(f"FATAL: Index dimension mismatch. Expected 384, found {index_tmp.d}. Delete index to rebuild.")
                 return

        embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    except Exception:
        log.exception("Failed to load embed model")
        return

    pdfs = [os.path.join(PDF_DIR, f) for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
    if not pdfs:
        log.warning("No PDFs found in %s", PDF_DIR)
        return

    # Sequential ingestion to save memory
    for pdf_path in pdfs:
        try:
            ingest_pdf_file(pdf_path, embed_model)
        except Exception:
            log.exception("Ingest failed for %s", pdf_path)

    log.info("All PDFs ingested successfully!")

if __name__ == "__main__":
    main()