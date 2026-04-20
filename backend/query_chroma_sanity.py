# query_chroma_sanity.py
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from config import settings


client = chromadb.HttpClient(
    host=settings.CHROMA_HOST,
    port=settings.CHROMA_PORT,
    settings=Settings(allow_reset=True)
)

col = client.get_or_create_collection(name=settings.COLLECTION, metadata={"hnsw:space": "cosine"})

embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")  # mismo modelo que en ingesta

query = input("Escribe tu consulta: ").strip()
qv = embedder.encode([query], normalize_embeddings=True).tolist()

res = col.query(query_embeddings=qv, n_results=5, include=["documents", "metadatas", "distances"])
for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
    print(f"[{meta['source']} p.{meta['page']}] dist={dist:.4f}\n{doc[:200]}...\n")