# chroma_delete_source.py
import sys
import chromadb
from chromadb.config import Settings
from config import settings

if len(sys.argv) < 2:
    print("Uso: python chroma_delete_source.py <nombre_archivo_pdf>")
    sys.exit(1)

FNAME = sys.argv[1]
#client = chromadb.HttpClient(host="localhost", port=8000, settings=Settings(allow_reset=True))

client = chromadb.HttpClient(
    host=settings.CHROMA_HOST,
    port=settings.CHROMA_PORT,
    settings=Settings(allow_reset=True)
)

col = client.get_or_create_collection("kb_docs")

def count_by_source(fname):
    total = 0
    batch, off = 5000, 0
    while True:
        r = col.get(where={"source": fname}, include=["metadatas"], limit=batch, offset=off)
        n = len(r.get("ids", []))
        total += n
        if n < batch: break
        off += batch
    return total

before = count_by_source(FNAME)
print(f"[ANTES] {FNAME}: {before} chunks")
col.delete(where={"source": FNAME})
after = count_by_source(FNAME)
print(f"[DESPUÉS] {FNAME}: {after} chunks (borrados {before - after})")