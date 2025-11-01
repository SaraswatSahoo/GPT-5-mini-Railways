# app.py
import os
import faiss
import pickle
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from dotenv import load_dotenv
from typing import List
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# MongoDB config
MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DBNAME = os.getenv("MONGODB_DBNAME")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION")

if not all([MONGODB_URI, MONGODB_DBNAME, MONGODB_COLLECTION]):
    raise RuntimeError("MongoDB config not set in .env")

# Connect MongoDB
client = MongoClient(MONGODB_URI)
db = client[MONGODB_DBNAME]
collection = db[MONGODB_COLLECTION]

# FastAPI app
app = FastAPI(title="Industry-Ready RAG API", version="2.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------- Models -------------------
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


# ------------------- Routes -------------------
@app.get("/")
async def root():
    """Health check"""
    # Changed message to reflect its purpose as an old/non-main server file
    return {"message": "✅ Secondary server running (Upload only)"}


@app.get("/documents")
async def list_documents():
    """List up to 50 ingested documents"""
    docs = collection.find({}, {"_id": 0}).limit(50)
    return {"documents": list(docs)}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a file (PDF/docx/txt).
    - Saves file to ./uploads
    - Real parsing & ingestion handled by `ingest_to_mongodb.py`
    """
    os.makedirs("./uploads", exist_ok=True)
    file_path = os.path.join("./uploads", file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    doc_meta = {
        "filename": file.filename,
        "path": file_path,
        "status": "uploaded",
    }
    collection.insert_one(doc_meta)
    # NOTE: This endpoint only uploads and *registers* the file. 
    # Real ingestion needs to be triggered separately, e.g., via the /ingest/pdf endpoint in server.py, or a background worker.
    return {"message": f"📄 File {file.filename} uploaded successfully (requires ingest job)", "metadata": doc_meta}


@app.post("/query") # Kept only one /query route for client compatibility
async def query_documents(request: QueryRequest):
    """
    Placeholder/removed RAG logic. Use server.py for /ask.
    """
    return {"query": request.query, "answer": "Use the /ask endpoint on the main server."}
