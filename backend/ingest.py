import os, uuid, re
from pypdf import PdfReader
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from config import settings


CHROMA_HOST = settings.CHROMA_HOST
CHROMA_PORT = settings.CHROMA_PORT
COLLECTION  = settings.COLLECTION

client = chromadb.HttpClient(
    host=CHROMA_HOST,
    port=CHROMA_PORT,
    settings=Settings(allow_reset=True)
)

# Crear/obtener colección
collection = client.get_or_create_collection(
    name=COLLECTION,
    metadata={"hnsw:space": "cosine"}
)

# Modelo embeddings local (rápido, multilingüe)
embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# --- LIMPIEZA ---
def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()

# --- CARGA PDF ---
def load_pdf(path: str):
    reader = PdfReader(path)
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        yield page_num, clean_text(text)

# --- CHUNKING ---
def chunk(text: str, max_chars=1200, overlap=150):
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        part = text[start:end]
        if len(part.strip()) > 50:
            yield part.strip()
        start = end - overlap

# --- INGESTA ---
def ingest(pdf_path: str):
    fname = os.path.basename(pdf_path)

    docs = []
    metas = []
    ids   = []

    for page_num, page_text in load_pdf(pdf_path):
        for c in chunk(page_text):
            docs.append(c)
            metas.append({"source": fname, "page": page_num})
            ids.append(str(uuid.uuid4()))

    if not docs:
        print("No se encontraron chunks.")
        return

    embeddings = embedder.encode(docs, normalize_embeddings=True).tolist()

    collection.add(
        documents=docs,
        embeddings=embeddings,
        metadatas=metas,
        ids=ids
    )

    print(f"Ingestados {len(docs)} chunks desde {fname}")

if __name__ == "__main__":
    import sys
    ingest(sys.argv[1])