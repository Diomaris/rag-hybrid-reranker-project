# stats_overall.py
import chromadb
from chromadb.config import Settings

CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
COLLECTION  = "kb_docs"

client = chromadb.HttpClient(
    host=CHROMA_HOST, port=CHROMA_PORT,
    settings=Settings(allow_reset=True)
)
col = client.get_or_create_collection(COLLECTION)

print("Colección:", COLLECTION)
print("Total de items (chunks):", col.count())