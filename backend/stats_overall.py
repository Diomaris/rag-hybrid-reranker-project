# stats_overall.py
import chromadb
from chromadb.config import Settings
from config import settings

client = chromadb.HttpClient(
    host=settings.CHROMA_HOST,
    port=settings.CHROMA_PORT,
    settings=Settings(allow_reset=True)
)

col = client.get_or_create_collection(settings.COLLECTION)

print("Colección:", settings.COLLECTION)
print("Total de items (chunks):", col.count())