# stats_by_source.py
import sys
import chromadb
from chromadb.config import Settings
from config import settings

if len(sys.argv) < 2:
    print("Uso: python stats_by_source.py <nombre_archivo_pdf>")
    sys.exit(1)

FNAME = sys.argv[1]

client = chromadb.HttpClient(
    host=settings.CHROMA_HOST,
    port=settings.CHROMA_PORT,
    settings=Settings(allow_reset=True)
)

col = client.get_or_create_collection(settings.COLLECTION)

# Chroma no tiene "count(where=...)" en todas las builds,
# así que usamos get() filtrado y contamos ids.
batch_size = 5000
offset = 0
total = 0

while True:
    res = col.get(
        where={"source": FNAME},
        include=["metadatas"],
        limit=batch_size,
        offset=offset
    )
    ids = res.get("ids", [])
    n = len(ids)
    total += n
    if n < batch_size:
        break
    offset += batch_size

print(f"Fuente: {FNAME}")
print("Chunks registrados con ese 'source':", total)