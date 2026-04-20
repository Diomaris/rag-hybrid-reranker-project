import os, sys, uuid, re
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from config import settings


client = chromadb.HttpClient(
    host=settings.CHROMA_HOST,
    port=settings.CHROMA_PORT,
    settings=Settings(allow_reset=True)
)


def info(msg: str):
    print(f"[INFO] {msg}")

def warn(msg: str):
    print(f"[WARN] {msg}")

def err(msg: str):
    print(f"[ERROR] {msg}")

def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()

def load_pdf(path: str) -> Iterable[tuple[int, str]]:
    """Devuelve (num_pagina, texto_limpio) por cada página."""
    reader = PdfReader(path)
    if getattr(reader, "is_encrypted", False):
        warn("El PDF está cifrado. Si requiere contraseña no se podrá extraer texto.")
        try:
            # intento de decrypt sin contraseña (algunos PDFs permiten)
            reader.decrypt("")
        except Exception as e:
            err(f"No se pudo descifrar el PDF: {e}")
            return
    n = len(reader.pages)
    info(f"PDF con {n} páginas.")
    for i, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception as e:
            warn(f"Fallo extrayendo texto en la página {i}: {e}")
            raw = ""
        yield i, clean_text(raw)

def chunk(text: str, max_chars=1200, overlap=150):
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        part = text[start:end]
        if len(part.strip()) > 50:
            yield part.strip()
        start = end - overlap

def main(pdf_path: str):
    # 0) Validaciones de ruta
    p = Path(pdf_path)
    info(f"Ruta recibida: {p}")
    if not p.exists():
        err("El archivo no existe. Revisa la ruta.")
        return
    if not p.is_file():
        err("La ruta no es un archivo. Revisa.")
        return

    info("Conectando a Chroma...")

    # heartbeat rápido (algunas builds exponen v2; si falla, no es crítico)
    try:
        hb = client.heartbeat()
        info(f"Chroma heartbeat: {hb}")
    except Exception as e:
        warn(f"No se pudo obtener heartbeat vía cliente: {e}")

    collection = client.get_or_create_collection(
        name=COLLECTION, metadata={"hnsw:space": "cosine"}
    )
    info(f"Colección OK: {COLLECTION}")

    info("Cargando modelo de embeddings...")
    embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    # 1) Ingesta
    fname = p.name
    ids, docs, metas = [], [], []
    total_pages = 0
    total_with_text = 0

    try:
        for page_num, page_text in load_pdf(str(p)):
            total_pages += 1
            if page_text:
                total_with_text += 1
                for c in chunk(page_text):
                    docs.append(c)
                    metas.append({"source": fname, "page": page_num})
                    ids.append(str(uuid.uuid4()))
    except Exception as e:
        err(f"Fallo durante la extracción/troceo: {e}")
        return

    info(f"Páginas totales: {total_pages}, páginas con texto: {total_with_text}")
    if not docs:
        warn("No se encontraron chunks (¿PDF escaneado sin texto?).")
        warn("Si es escaneado, necesitaremos OCR (puedo darte receta con Tesseract/Docling).")
        return

    try:
        info(f"Generando embeddings de {len(docs)} chunks...")
        vecs = embedder.encode(docs, normalize_embeddings=True).tolist()
        info("Subiendo a Chroma...")
        collection.add(ids=ids, documents=docs, embeddings=vecs, metadatas=metas)
        info(f"Ingestados {len(docs)} chunks desde {fname}")
    except Exception as e:
        err(f"Fallo al generar embeddings o subir a Chroma: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        err("Uso: python ingest_verbose.py <ruta_al_pdf>")
        sys.exit(1)
    main(sys.argv[1])