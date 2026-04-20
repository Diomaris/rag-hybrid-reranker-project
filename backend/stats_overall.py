# stats_overall.py
import chromadb
from chromadb.config import Settings
from config import settings

client = chromadb.Client(
    host=settings.CHROMA_HOST,
    port=settings.CHROMA_PORT
)

col = client.get_or_create_collection(COLLECTION)

print("Colección:", COLLECTION)
print("Total de items (chunks):", col.count())