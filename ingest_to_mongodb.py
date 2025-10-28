import os
import logging
import json
from io import BytesIO
from typing import List, Dict
import tempfile

import fitz  # PyMuPDF
import camelot
from PIL import Image
import pytesseract
import numpy as np
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
from dotenv import load_dotenv
import faiss
import pickle
import uuid

from google.cloud import storage
from utils.logging_config import setup_logging

# ----------------------------------
# Setup and Environment
# ----------------------------------
setup_logging()
log = logging.getLogger("ingest")
load_dotenv()

# Mongo
MONGO_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("MONGODB_DBNAME", "gpt_integration")
COLLECTION_NAME = os.getenv("MONGODB_COLLECTION", "documents")

# GCP
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
GCS_KEY_PATH = os.getenv("GCS_KEY_PATH")
GCS_INPUT_PREFIX = os.getenv("GCS_INPUT_PREFIX", "pdfs/")
GCS_OUTPUT_PREFIX = os.getenv("GCS_OUTPUT_PREFIX", "knowledge_pack/")
LOCAL_TMP_DIR = os.getenv("LOCAL_TMP_DIR", "/tmp")

# FAISS paths (temp local)
FAISS_PATH = os.path.join(LOCAL_TMP_DIR, "index_hnsw.faiss")

# Initialize GCS client
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GCS_KEY_PATH
storage_client = storage.Client(project=GCP_PROJECT_ID)
bucket = storage_client.bucket(GCS_BUCKET_NAME)

# Mongo client
if not MONGO_URI:
    raise RuntimeError("MONGODB_URI not set in .env")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# Embedding & OCR Config
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
BATCH_SIZE_MONGO = 50
MAX_PAGES = 20

# Globals for FAISS
faiss_index = None
faiss_dim = None
faiss_id_map: Dict[int, str] = {}

# ----------------------------------
# Utility Functions
# ----------------------------------
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

def download_blob_to_temp(gcs_path: str) -> str:
    """Download a GCS object to local /tmp path"""
    blob = bucket.blob(gcs_path)
    local_path = os.path.join(LOCAL_TMP_DIR, os.path.basename(gcs_path))
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    blob.download_to_filename(local_path)
    return local_path

def upload_file_to_gcs(local_path: str, gcs_path: str):
    """Upload local file to GCS under output prefix"""
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)
    log.info(f"Uploaded {local_path} → gs://{GCS_BUCKET_NAME}/{gcs_path}")

def save_image(pil_img: Image.Image, stem: str, page: int, idx: int, quality=70) -> str:
    try:
        pil_img = pil_img.convert("RGB")
        pil_img.thumbnail((1200, 1200))
        name = f"{stem}_p{page}_i{idx}.jpg"
        local_path = os.path.join(LOCAL_TMP_DIR, name)
        pil_img.save(local_path, "JPEG", optimize=True, quality=quality)

        # Upload image to GCS
        gcs_path = f"{GCS_OUTPUT_PREFIX}images/{name}"
        upload_file_to_gcs(local_path, gcs_path)
        return gcs_path
    except Exception:
        log.exception("save_image failed")
        return ""

# ----------------------------------
# Extraction Functions
# ----------------------------------
def extract_tables(pdf_path: str) -> List[Dict]:
    try:
        tables = camelot.read_pdf(pdf_path, pages="all")
        return [t.df.to_dict() for t in tables]
    except Exception as e:
        log.warning(f"camelot failed for {pdf_path}: {e}")
        return []

def extract_images_and_ocr(pdf_path: str) -> List[Dict]:
    docs = []
    try:
        doc = fitz.open(pdf_path)
        stem = os.path.splitext(os.path.basename(pdf_path))[0]
        for page_idx, page in enumerate(doc, start=1):
            for i, img in enumerate(page.get_images(full=True)):
                try:
                    base = doc.extract_image(img[0])
                    with Image.open(BytesIO(base["image"])) as pil_img:
                        if pil_img.width < 60 or pil_img.height < 60:
                            continue
                        gcs_image_path = save_image(pil_img, stem, page_idx, i)
                        ocr_text = ""
                        try:
                            ocr_text = pytesseract.image_to_string(pil_img, lang="hin+eng").strip()
                        except Exception:
                            pass
                        docs.append({
                            "type": "image",
                            "file": stem,
                            "page": page_idx,
                            "image_path": gcs_image_path,
                            "ocr_text": ocr_text,
                            "name": os.path.basename(gcs_image_path)
                        })
                except Exception:
                    log.exception("image extraction failed")
    except Exception:
        log.exception("extract_images_and_ocr failed")
    return docs

def extract_text_chunks(pdf_path: str) -> List[Dict]:
    chunks = []
    try:
        doc = fitz.open(pdf_path)
        stem = os.path.splitext(os.path.basename(pdf_path))[0]
        for i, page in enumerate(doc[:MAX_PAGES]):
            text = page.get_text("text") or ""
            if len(text.strip()) < 30:
                pix = page.get_pixmap()
                with Image.open(BytesIO(pix.tobytes("png"))) as im:
                    text += "\n" + pytesseract.image_to_string(im, lang="hin+eng")
            for c in chunk_text(text):
                chunks.append({"type": "text", "file": stem, "page": i + 1, "text": c})
    except Exception:
        log.exception("extract_text_chunks failed")
    return chunks

# ----------------------------------
# FAISS Functions
# ----------------------------------
def build_or_load_faiss(embs: np.ndarray, doc_ids: List[str]):
    global faiss_index, faiss_dim, faiss_id_map

    if faiss_index is None:
        if os.path.exists(FAISS_PATH):
            try:
                faiss_index = faiss.read_index(FAISS_PATH)
                with open(FAISS_PATH + ".pkl", "rb") as f:
                    faiss_id_map = pickle.load(f)
                faiss_dim = faiss_index.d
                log.info("Loaded existing FAISS index")
            except Exception:
                log.warning("Could not load FAISS index. Creating new.")
                faiss_dim = embs.shape[1]
                faiss_index = faiss.IndexHNSWFlat(faiss_dim, 32)
        else:
            faiss_dim = embs.shape[1]
            faiss_index = faiss.IndexHNSWFlat(faiss_dim, 32)

    start_idx = faiss_index.ntotal
    for i, doc_id in enumerate(doc_ids):
        faiss_id_map[start_idx + i] = doc_id

    faiss_index.add(embs)

# ----------------------------------
# Main Persistence Logic
# ----------------------------------
def persist_to_mongo_and_faiss(items: List[Dict], embed_model: SentenceTransformer):
    if not items:
        return

    texts, doc_ids = [], []
    for it in items:
        doc_id = str(uuid.uuid4())
        it["doc_id"] = doc_id
        doc_ids.append(doc_id)
        if it["type"] == "text":
            texts.append(it["text"])
        elif it["type"] == "table":
            texts.append(json.dumps(it.get("table_data", {})))
        elif it["type"] == "image":
            texts.append(it.get("ocr_text") or it.get("name") or "")
        else:
            texts.append(str(it))

    embs = embed_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype("float32")

    docs = []
    for it, e, txt in zip(items, embs, texts):
        doc = dict(it)
        doc["embedding"] = e.tolist()
        doc["text_for_search"] = txt
        docs.append(doc)

    # Insert to Mongo
    for i in range(0, len(docs), BATCH_SIZE_MONGO):
        try:
            collection.insert_many(docs[i:i + BATCH_SIZE_MONGO], ordered=False)
        except Exception:
            pass

    build_or_load_faiss(embs, doc_ids)
    faiss.write_index(faiss_index, FAISS_PATH)
    with open(FAISS_PATH + ".pkl", "wb") as f:
        pickle.dump(faiss_id_map, f)

    # Upload FAISS & map to GCS
    upload_file_to_gcs(FAISS_PATH, f"{GCS_OUTPUT_PREFIX}index_hnsw.faiss")
    upload_file_to_gcs(FAISS_PATH + ".pkl", f"{GCS_OUTPUT_PREFIX}index_hnsw.faiss.pkl")

# ----------------------------------
# Ingest Pipeline
# ----------------------------------
def ingest_pdf_file(pdf_blob_path: str, embed_model: SentenceTransformer):
    log.info(f"Processing PDF: {pdf_blob_path}")
    local_pdf = download_blob_to_temp(pdf_blob_path)

    text_chunks = extract_text_chunks(local_pdf)
    image_docs = extract_images_and_ocr(local_pdf)
    tables = extract_tables(local_pdf)
    table_docs = [{"type": "table", "file": os.path.basename(local_pdf), "table_data": t} for t in tables]

    persist_to_mongo_and_faiss(text_chunks + image_docs + table_docs, embed_model)
    log.info(f"Completed {pdf_blob_path}")

# ----------------------------------
# Main Entry
# ----------------------------------
def main():
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    # List PDFs from GCS
    blobs = storage_client.list_blobs(GCS_BUCKET_NAME, prefix=GCS_INPUT_PREFIX)
    pdfs = [b.name for b in blobs if b.name.lower().endswith(".pdf")]

    if not pdfs:
        log.warning("No PDFs found in GCS input prefix.")
        return

    for pdf_blob_path in pdfs:
        ingest_pdf_file(pdf_blob_path, embed_model)

    log.info("All PDFs processed successfully!")

if __name__ == "__main__":
    main()