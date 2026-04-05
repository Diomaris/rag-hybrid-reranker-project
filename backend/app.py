# C:\RAG_Project\backend\app.py

import os
import re
import io
import gc
import uuid
import string
import logging
import time
from typing import List, Optional
from functools import lru_cache

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import threading
from watcher import start_watcher

# ---- Chroma ----
import chromadb
from chromadb.config import Settings
# (El embedder se cargará en lazy-load, no aquí.)

# ---- PDF + OCR ----
import fitz  # PyMuPDF
from PIL import Image
try:
    import pytesseract
    TESSERACT_OK = True
    # Ajusta si tu Tesseract está en otra ruta:
    # pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
except Exception:
    TESSERACT_OK = False

# ---- BM25 ----
from rank_bm25 import BM25Okapi
LEX_CORPUS: List[str] = []
LEX_META: List[dict] = []
BM25 = None

# ================== Config ==================
load_dotenv()
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
COLLECTION  = os.getenv("COLLECTION", "kb_docs")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

GEM_BASE = os.getenv("GEM_BASE", "https://generativelanguage.googleapis.com/v1beta/openai/")
GEM_KEY  = os.getenv("GEMINI_API_KEY", "AIzaSyDH7ib0ejDwxBTjiRVJ-udzQWdIJTljoIk")
GEM_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GEM_KEY")
if not GEM_KEY:
    raise RuntimeError("Falta GEMINI_API_KEY/GEM_KEY en el entorno")
DEFAULT_TOP_K        = int(os.getenv("DEFAULT_TOP_K", "6"))
DEFAULT_GEMINI_MODEL = os.getenv("DEFAULT_GEMINI_MODEL", "gemini-2.5-flash")

if not GEM_BASE.endswith("/"):
    raise RuntimeError("GEM_BASE debe terminar con '/'. Ej: https://.../v1beta/openai/")

# --- Reranker config ---
ENABLE_RERANKER = os.getenv("ENABLE_RERANKER", "true").lower() == "true"
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
RERANKER_TOP_K = int(os.getenv("RERANKER_TOP_K", "6"))
RERANKER_MAXLEN = int(os.getenv("RERANKER_MAXLEN", "512"))


# ================== App & Middleware ==================
app = FastAPI(title="RAG Backend – Denso + Léxico + Gemini")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("rag-pro")

# ================== Chroma (rápido) ==================
client = chromadb.HttpClient(
    host=CHROMA_HOST,
    port=CHROMA_PORT,
    settings=Settings(allow_reset=True)
)
collection = client.get_or_create_collection(name=COLLECTION)

# ================== Hybrid Retrieval (RRF) ==================
def rrf_fusion(dense_blocks: list[str], dense_scores: list[float],
               lex_blocks: list[str], lex_scores: list[float],
               k: int = 60, top_k: int = 6):
    """
    Fusión RRF clásica:
    score = 1 / (k + rank)
    Donde rank es la posición (0,1,2...) en cada lista.
    """

    fusion = {}

    # Normalizamos entradas a la misma longitud mínima
    d_len = len(dense_blocks)
    l_len = len(lex_blocks)

    # Asignar scores densos por ranking
    for rank, blk in enumerate(dense_blocks):
        fusion.setdefault(blk, 0.0)
        fusion[blk] += 1.0 / (k + rank)

    # Asignar scores de BM25 por ranking
    for rank, blk in enumerate(lex_blocks):
        fusion.setdefault(blk, 0.0)
        fusion[blk] += 1.0 / (k + rank)

    # Ordenar por score RRF total
    fused_sorted = sorted(fusion.items(), key=lambda x: x[1], reverse=True)

    # Tomar los top_k
    final = [blk for blk, sc in fused_sorted[:top_k]]
    return final

# ================== Lazy‑load del embedder ==================
@lru_cache(maxsize=1)
def get_embedder():
    """
    Carga diferida y cacheada de SentenceTransformer.
    Evita bloquear la apertura del puerto durante el arranque del backend.
    """
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL)
    
# ================== Re-Ranker (Cross‑Encoder) ==================
from functools import lru_cache
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def _pick_device() -> torch.device:
    # Usa GPU si está disponible; si no, CPU.
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

@lru_cache(maxsize=1)
def get_reranker():
    """
    Carga diferida del cross‑encoder (tokenizer + model) y fija eval().
    """
    tok = AutoTokenizer.from_pretrained(RERANKER_MODEL, use_fast=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL)
    mdl.eval()
    mdl.to(_pick_device())
    return tok, mdl

def rerank_blocks(query: str, blocks: list[str], keep_top_k: int = None, max_length: int = None) -> list[str]:
    """
    Ordena los blocks por relevancia con un cross‑encoder (score mayor = más relevante).
    """
    if not ENABLE_RERANKER or not blocks:
        return blocks

    keep = keep_top_k if keep_top_k is not None else RERANKER_TOP_K
    max_len = max_length if max_length is not None else RERANKER_MAXLEN

    tok, mdl = get_reranker()
    dev = _pick_device()

    scores = []
    with torch.no_grad():
        for b in blocks:
            # Empaqueta (query, block) en un solo input
            inputs = tok(
                query, b,
                truncation=True,
                max_length=max_len,
                padding="max_length",
                return_tensors="pt"
            ).to(dev)
            out = mdl(**inputs).logits.squeeze()
            # Para la mayoría de re‑rankers, mayor logit = más relevante.
            score = float(out.item()) if out.dim() == 0 else float(out[0].item())
            scores.append((score, b))

    scores.sort(key=lambda x: x[0], reverse=True)
    return [b for (_, b) in scores[:max(1, min(keep, len(scores)))]]   

# ================== Modelos de datos ==================
class RetrieveRequest(BaseModel):
    query: str
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=50)

class AnswerRequest(BaseModel):
    query: str
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=50)
    model: str = Field(default=DEFAULT_GEMINI_MODEL)
    temperature: float = Field(default=0.6, ge=0.0, le=1.0)

class LexicalRequest(BaseModel):
    query: str
    top_k: int = Field(default=10, ge=1, le=50)

# ================== Utilidades ==================
def _clean_text(t: str) -> str:
    t = t.replace("\x00", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{2,}", "\n", t)
    return t.strip()

def _chunk(text: str, max_chars=1200, overlap=150, min_len=30) -> List[str]:
    text = text.strip()
    n = len(text)
    if n == 0:
        return []
    out, start = [], 0
    while start < n:
        end = min(start + max_chars, n)
        part = text[start:end].strip()
        if len(part) > min_len:
            out.append(part)
        if end == n:
            break
        next_start = end - overlap
        if next_start <= start:
            next_start = end
        start = next_start
    return out

def _ocr_page(page, dpi=120, lang="spa+eng") -> str:
    if not TESSERACT_OK:
        return ""
    mat = fitz.Matrix(dpi/72, dpi/72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img_bytes = pix.tobytes("png")
    img = Image.open(io.BytesIO(img_bytes)).convert("L")
    text = pytesseract.image_to_string(img, lang=lang, config="--oem 1 --psm 6") or ""
    return _clean_text(text)

def build_context_blocks(docs: List[str], metas: List[dict]) -> List[str]:
    blocks = []
    for d, m in zip(docs, metas):
        src = m.get("source", "desconocido")
        pg  = m.get("page", "?")
        blocks.append(f"[{src} p.{pg}] {d}")
    return blocks

def build_prompt(query: str, blocks: List[str]) -> str:
    ctx = "\n\n".join(blocks)
    return (
        "Eres un asistente experto en RAG. Responde exclusivamente con la información "
        "del contexto. Si algo no está en el contexto, dilo explícitamente.\n\n"
        f"Contexto:\n{ctx}\n\n"
        f"Pregunta: {query}\n\n"
        "Responde en español y cita las fuentes entre corchetes [archivo p.página]."
    )

# ================== LLM (Gemini vía OpenAI‑compatible) ==================
import httpx, asyncio, random

async def call_gemini_chat_completions(model: str, prompt: str, temperature: float = 0.6) -> str:
    if not GEM_KEY:
        raise RuntimeError("Falta GEMINI_API_KEY en .env")

    headers = {"Authorization": f"Bearer {GEM_KEY}"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Eres un asistente experto en RAG."},
            {"role": "user",    "content": prompt}
        ],
        "temperature": temperature
    }

    # Política de reintentos: 429 y 5xx (3 intentos, backoff exponencial con jitter)
    max_attempts = 3
    base = 1.0  # segundos

    async with httpx.AsyncClient(timeout=120) as s:
        for attempt in range(1, max_attempts + 1):
            try:
                r = await s.post(f"{GEM_BASE}chat/completions", headers=headers, json=payload)
                r.raise_for_status()  # lanza en 4xx/5xx
                data = r.json()
                return data["choices"][0]["message"]["content"]

            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                body   = e.response.text

                # Reintentar en 429 o 5xx
                if status == 429 or 500 <= status < 600:
                    if attempt < max_attempts:
                        # Exponential backoff con jitter [base*2^(n-1) ± 20%]
                        wait = base * (2 ** (attempt - 1))
                        wait = wait * random.uniform(0.8, 1.2)
                        await asyncio.sleep(wait)
                        continue
                # Otros códigos: propagar con detalle
                raise RuntimeError(f"{status} {body}") from e

            except httpx.RequestError as e:
                # Errores de red/transporte: reintenta si quedan intentos
                if attempt < max_attempts:
                    wait = base * (2 ** (attempt - 1)) * random.uniform(0.8, 1.2)
                    await asyncio.sleep(wait)
                    continue
                raise RuntimeError(f"RequestError: {e}") from e

    # Si llegó aquí, agotó intentos
    raise RuntimeError("LLM no respondió tras reintentos")

# ================== ENDPOINTS ==================
@app.get("/health")
def health():
    return {"ok": True}

# ---------- Ingesta ----------
@app.post("/ingest")
async def ingest_pdf(file: UploadFile = File(...), ocr: bool = Form(False)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Solo se aceptan PDFs")
    pdf_bytes = await file.read()
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF inválido: {e}")

    fname = file.filename
    pages_total = len(doc)
    pages_with_text = 0
    ocr_pages = 0
    chunks_total = 0

    batch_docs, batch_metas, batch_ids = [], [], []

    try:
        for i in range(pages_total):
            page = doc[i]
            txt = _clean_text(page.get_text("text") or "")
            used_ocr = False
            if not txt and ocr:
                txt = _ocr_page(page)
                used_ocr = True
            if txt:
                pages_with_text += 1
                if used_ocr:
                    ocr_pages += 1
                for c in _chunk(txt, max_chars=1200, overlap=150, min_len=30):
                    batch_docs.append(c)
                    batch_metas.append({"source": fname, "page": i + 1})
                    batch_ids.append(str(uuid.uuid4()))
                    chunks_total += 1

            if len(batch_docs) >= 256:
                emb  = get_embedder()
                vecs = emb.encode(batch_docs, normalize_embeddings=True).tolist()
                collection.add(ids=batch_ids, documents=batch_docs, embeddings=vecs, metadatas=batch_metas)
                batch_docs.clear(); batch_metas.clear(); batch_ids.clear()
                gc.collect()

        if batch_docs:
            emb  = get_embedder()
            vecs = emb.encode(batch_docs, normalize_embeddings=True).tolist()
            collection.add(ids=batch_ids, documents=batch_docs, embeddings=vecs, metadatas=batch_metas)

    except Exception as e:
        logger.exception("Error en ingest")
        raise HTTPException(status_code=500, detail=f"Error de ingesta: {e}")
    finally:
        doc.close()

    return {
        "file": fname,
        "pages_total": pages_total,
        "pages_with_text": pages_with_text,
        "ocr_pages": ocr_pages,
        "chunks_added": chunks_total
    }

# ---------- Retrieve (denso) ----------
@app.post("/retrieve", operation_id="rag_retrieve", summary="Dense retrieval desde Chroma (devuelve contextos con [source p.page])")
def retrieve(req: RetrieveRequest):
    emb  = get_embedder()
    qemb = emb.encode([req.query], normalize_embeddings=True).tolist()
    res = collection.query(
        query_embeddings=qemb,
        n_results=req.top_k,
        include=["documents", "metadatas", "distances"]
    )
    docs  = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    blocks = build_context_blocks(docs, metas)
    # ---- opcional: re‑rank en /retrieve también ----
    try:
        blocks = rerank_blocks(req.query, blocks, keep_top_k=req.top_k)
    except Exception as e:
        logger.warning(f"[reranker] en /retrieve deshabilitado por error: {e}")
    # -----------------------------------------------
    return {
        "query": req.query,
        "top_k": req.top_k,
        "count": len(blocks),
        "contexts": blocks,
        "distances": res.get("distances", [[]])[0]
    }

# ---------- Answer (RAG + Gemini) ----------

@app.post(
    "/answer",
    operation_id="rag_answer",
    summary="RAG + Gemini (usa contextos y devuelve respuesta redactada; cita si el prompt lo pide)"
)
async def answer(req: AnswerRequest):
    emb = get_embedder()
    qemb = emb.encode([req.query], normalize_embeddings=True).tolist()
    dense = collection.query(
        query_embeddings=qemb,
        n_results=max(req.top_k, 12),
        include=["documents", "metadatas", "distances"]
    )
    docs  = dense.get("documents", [[]])[0]
    metas = dense.get("metadatas", [[]])[0]
    if not docs:
        return {
            "query": req.query,
            "answer": "No encuentro evidencia en el contexto.",
            "contexts": [],
            "model": req.model
        }

    # Armado de bloques de contexto (con [archivo p.X] al inicio de cada línea)
    blocks = build_context_blocks(docs, metas)[:req.top_k]

    # ---- NUEVO: reordenar por cross‑encoder (local) ----
    try:
        blocks = rerank_blocks(req.query, blocks, keep_top_k=req.top_k)
    except Exception as e:
        logger.warning(f"[reranker] deshabilitado por error: {e}")
    # -----------------------------------------------------

    # Prompt para el modelo
    prompt = build_prompt(req.query, blocks)
    
    # Respuesta del LLM
    #text = await call_gemini_chat_completions(
    #    model=req.model,
    #    prompt=prompt,
    #    temperature=req.temperature
    #)
    # --- Diagnóstico: captura el error del LLM y lo devuelve en claro ---
    try:
        text = await call_gemini_chat_completions(
            model=req.model,
            prompt=prompt,
            temperature=req.temperature
        )
    except Exception as e:
        from fastapi.responses import JSONResponse
        import logging
        logging.getLogger("uvicorn.error").exception("Error en call_gemini_chat_completions")
        return JSONResponse(
            status_code=502,
            content={
                "query": req.query,
                "answer": f"Error al consultar el LLM: {type(e).__name__}: {e}",
                "contexts": blocks,
                "model": req.model
            }
        )
    # ------- Bloque "Fuentes" seguro (añade citas reales al final) -------
    sources = []
    seen = set()
    for b in blocks[:req.top_k]:
        b = (b or "").strip()
        if b.startswith('[') and (']' in b):
            tag = b.split(']')[0] + ']'
            if tag not in seen:
                seen.add(tag)
                sources.append(tag)
    if sources:
        text = text + "\n\n---\n**Fuentes**\n" + "\n".join("- " + s for s in sources)
    # ---------------------------------------------------------------------

    return {"query": req.query, "answer": text, "contexts": blocks, "model": req.model}

# ---------- BM25 (léxico) ----------
def _tok(s: str):
    s = s.lower().translate(str.maketrans("", "", string.punctuation))
    return re.findall(r"\w+", s, flags=re.UNICODE)

def build_bm25_from_chroma(batch=1000, cap: Optional[int] = 5000):
    global LEX_CORPUS, LEX_META, BM25
    LEX_CORPUS, LEX_META = [], []

    total = collection.count()
    logger.info(f"[BM25] collection.count() = {total}")
    if total == 0:
        BM25 = None
        return 0

    offset = 0
    leidos = 0
    while True:
        if cap is not None and leidos >= cap:
            break
        this_limit = batch if cap is None else min(batch, cap - leidos)
        r = collection.get(include=["documents", "metadatas"], limit=this_limit, offset=offset)
        docs  = r.get("documents", [])
        metas = r.get("metadatas", [])
        n = len(docs)
        logger.info(f"[BM25] Lote offset={offset}, n={n}, leídos={leidos}")
        if n == 0:
            break
        LEX_CORPUS.extend(docs)
        LEX_META.extend(metas)
        leidos += n
        offset += this_limit
        if offset >= total:
            break

    if not LEX_CORPUS:
        BM25 = None
        logger.warning("[BM25] No hay documentos para indexar (colección vacía o cap=0).")
        return 0

    logger.info(f"[BM25] Tokenizando {len(LEX_CORPUS)} documentos...")
    tokenized = [_tok(d) for d in LEX_CORPUS]
    t0 = time.time()
    BM25 = BM25Okapi(tokenized)
    dt = round(time.time() - t0, 2)
    logger.info(f"[BM25] Índice construido con {len(LEX_CORPUS)} chunks en {dt}s.")
    return len(LEX_CORPUS)

@app.post("/retrieve_lexical")
def retrieve_lexical(req: LexicalRequest):
    if not BM25:
        raise HTTPException(status_code=500, detail="BM25 no disponible (colección vacía o fallo de carga).")
    qtok = _tok(req.query)
    scores = BM25.get_scores(qtok)
    idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:req.top_k]
    blocks = [f"[{LEX_META[i].get('source','?')} p.{LEX_META[i].get('page','?')}] {LEX_CORPUS[i]}" for i in idx]
    return {"query": req.query, "top_k": req.top_k, "contexts": blocks}

# ---------- Hybrid Retrieve (RRF) ----------
class HybridRetrieveRequest(BaseModel):
    query: str
    top_k: int = Field(default=6, ge=1, le=50)

@app.post("/retrieve_hybrid")
def retrieve_hybrid(req: HybridRetrieveRequest):
    # 1) DENSO
    emb = get_embedder()
    qemb = emb.encode([req.query], normalize_embeddings=True).tolist()
    dense = collection.query(
        query_embeddings=qemb,
        n_results=max(req.top_k, 12),
        include=["documents", "metadatas", "distances"]
    )
    docs_dense  = dense.get("documents", [[]])[0]
    metas_dense = dense.get("metadatas", [[]])[0]
    blocks_dense = build_context_blocks(docs_dense, metas_dense)

    # 2) LÉXICO
    if not BM25:
        raise HTTPException(status_code=500, detail="BM25 no está disponible. Reconstruye con /rebuild_bm25")

    qtok = _tok(req.query)
    scores = BM25.get_scores(qtok)
    idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:max(req.top_k, 12)]
    blocks_lex = [
        f"[{LEX_META[i].get('source','?')} p.{LEX_META[i].get('page','?')}] {LEX_CORPUS[i]}"
        for i in idx
    ]

    # 3) FUSIÓN RRF (denso + léxico)
    fused_blocks = rrf_fusion(
        dense_blocks=blocks_dense,
        dense_scores=[],           # (scores no necesarios, RRF usa rank)
        lex_blocks=blocks_lex,
        lex_scores=[],
        k=60,
        top_k=req.top_k,
    )

    # 4) Re‑Ranker Cross‑Encoder (opcional, recomendado)
    try:
        fused_blocks = rerank_blocks(req.query, fused_blocks, keep_top_k=req.top_k)
    except Exception as e:
        logger.warning(f"[reranker] error en hybrid: {e}")

    return {
        "query": req.query,
        "top_k": req.top_k,
        "contexts": fused_blocks
    }


@app.post("/rebuild_bm25")
def rebuild_bm25(batch: int = 1000, cap: Optional[int] = 5000):
    t0 = time.time()
    try:
        docs = build_bm25_from_chroma(batch=batch, cap=cap)
        dt = round(time.time() - t0, 2)
        return {"ok": BM25 is not None, "docs_indexed": docs, "seconds": dt, "batch": batch, "cap": cap}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fallo reconstruyendo BM25: {e}")

@app.get("/bm25_status")
def bm25_status():
    return {"ready": BM25 is not None, "docs_indexed": len(LEX_CORPUS), "meta_indexed": len(LEX_META)}
    
class RerankTestRequest(BaseModel):
    query: str
    blocks: list[str]
    top_k: int = Field(default=6, ge=1, le=50)

@app.post("/rerank_test")
def rerank_test(req: RerankTestRequest):
    try:
        out = rerank_blocks(req.query, req.blocks, keep_top_k=req.top_k)
        return {"query": req.query, "top_k": req.top_k, "contexts": out}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Re-ranker error: {e}")


#Arrancar el watcher automáticamente en docker
threading.Thread(target=start_watcher, daemon=True).start()