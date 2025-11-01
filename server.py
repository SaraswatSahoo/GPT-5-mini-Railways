from fastapi import Header, HTTPException, Depends
import os
import base64
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from gtts import gTTS
from datetime import datetime

from utils.logging_config import setup_logging
from retrieve_and_answer import answer as rag_answer
from ingest_to_mongodb import ingest_pdf_file, SentenceTransformer, EMBED_MODEL_NAME  # ingest function

from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from retrieve_and_answer import client, FAISS_AVAILABLE, faiss_index, OPENAI_API_KEY

# Note: ingest_to_mongodb defines ingest_pdf_file and model name

setup_logging()
log = logging.getLogger("server")
load_dotenv()

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")
ALLOW_ORIGINS = [origin.strip() for origin in CORS_ORIGINS.split(",") if origin.strip()]

app = FastAPI(title="GPT Integration Backend")

# Create Limiter instance, key function is the client's IP address
limiter = Limiter(key_func=get_remote_address)
RATE_LIMIT_QUESTIONS = os.getenv("RATE_LIMIT_QUESTIONS", "10/minute")
RATE_LIMIT_INGEST = os.getenv("RATE_LIMIT_INGEST", "5/hour")

# Attach limiter instance to your FastAPI app
app.state.limiter = limiter

# app.add_middleware(CORSMiddleware, allow_origins=ALLOW_ORIGINS, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,  # ONLY allow these listed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=600,
)

#authentication

log = logging.getLogger(__name__)
API_KEY = os.getenv("API_KEY")

def verify_api_key(authorization: str = Header(None)):
    if not API_KEY:
        # If no API key is set, skip authentication (e.g., dev mode)
        return True
    if not authorization:
        log.warning("Missing Authorization header")
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise ValueError("Invalid scheme")
        if token != API_KEY:
            raise ValueError("Invalid token")
    except Exception as e:
        log.warning(f"Authorization failure: {e}")
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True


class Query(BaseModel):
    question: str

@app.exception_handler(Exception)
async def global_handler(request: Request, exc: Exception):
    log.exception("Unhandled error: %s", exc)
    return JSONResponse(status_code=500, content={"error":"Internal server error"})

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Too many requests, please try again later."}
    )


@app.get("/health")
async def health_check():
    log = logging.getLogger("server")
    checks = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "mongodb": False,
            "faiss": False,
            "openai": False,
        }
    }
    
    # Check MongoDB connectivity
    try:
        client.admin.command("ping")
        checks["components"]["mongodb"] = True
        log.info("MongoDB connectivity: OK")
    except Exception as e:
        log.error(f"MongoDB connectivity failed: {e}")
        checks["status"] = "degraded"

    # Check FAISS availability
    try:
        if FAISS_AVAILABLE and faiss_index is not None:
            # Test FAISS index search to ensure functionality
            test_vec = np.zeros((1, faiss_index.d), dtype="float32")
            faiss_index.search(test_vec, 1)
            checks["components"]["faiss"] = True
            log.info("FAISS index: OK")
        else:
            checks["status"] = "degraded"
            log.warning("FAISS index not loaded or unavailable")
    except Exception as e:
        log.error(f"FAISS search check failed: {e}")
        checks["status"] = "degraded"

    # Check OpenAI API key presence (basic check)
    if OPENAI_API_KEY:
        checks["components"]["openai"] = True
        log.info("OpenAI API key present")
    else:
        checks["status"] = "unhealthy"
        log.error("OpenAI API key missing")

    # Determine appropriate HTTP status code
    if checks["status"] == "healthy":
        status_code = 200
    elif checks["status"] == "degraded":
        status_code = 503  # Service Unavailable
    else:
        status_code = 500

    return JSONResponse(content=checks, status_code=status_code)


@app.get("/")
def root():
    return {"message": "RAG API running. Use /health for status."}

@app.post("/ask")
@limiter.limit(RATE_LIMIT_QUESTIONS)
def ask(request: Request, q: Query, _=Depends(verify_api_key)):
    res = rag_answer(q.question)
    return res

@app.post("/ingest/pdf")
@limiter.limit(RATE_LIMIT_INGEST)
async def ingest_pdf_upload(request: Request, file: UploadFile = File(...), _=Depends(verify_api_key)):
    try:
        # save file to disk
        os.makedirs("pdf", exist_ok=True)
        path = os.path.join("pdf", file.filename)
        with open(path, "wb") as f:
            f.write(await file.read())
        # embed model
        embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        ingest_pdf_file(path, embed_model)
        return {"status":"ok","file":file.filename}
    except Exception:
        log.exception("ingest endpoint failed")
        return JSONResponse(status_code=500, content={"error":"ingest failed"})

@app.post("/tts")
async def tts(text: str = Form(...), lang: str = Form("en")):
    try:
        if lang not in {"en","hi"}:
            lang = "en"
        tts = gTTS(text=text, lang=lang)
        out = f"tts_{lang}.mp3"
        tts.save(out)
        with open(out, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return {"audio_base64": f"data:audio/mpeg;base64,{b64}"}
    except Exception:
        log.exception("TTS failed")
        return JSONResponse(status_code=500, content={"error":"tts failed"})

# Optional STT endpoint using whisper if enabled in .env
ENABLE_STT = os.getenv("ENABLE_STT","false").lower()=="true"
if ENABLE_STT:
    try:
        import whisper
        WHISPER_MODEL = os.getenv("WHISPER_MODEL","small")
        _whisper = whisper.load_model(WHISPER_MODEL)
    except Exception:
        log.exception("Whisper initialization failed; disabling STT")
        ENABLE_STT = False

@app.post("/stt")
async def stt(audio: UploadFile = File(...), lang_hint: str = Form("auto")):
    if not ENABLE_STT:
        return JSONResponse(status_code=400, content={"error":"STT disabled on server"})
    try:
        tmp = "_tmp_audio_input"
        with open(tmp, "wb") as f:
            f.write(await audio.read())
        result = _whisper.transcribe(tmp, language=None if lang_hint=="auto" else lang_hint)
        os.remove(tmp)
        return {"text": result.get("text","").strip()}
    except Exception:
        log.exception("STT failed")
        return JSONResponse(status_code=500, content={"error":"stt failed"})