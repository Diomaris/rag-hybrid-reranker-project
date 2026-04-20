import chromadb
from chromadb.config import Settings
from config import settings

client = chromadb.HttpClient(
    host=settings.CHROMA_HOST,
    port=settings.CHROMA_PORT,
    settings=Settings(allow_reset=True)
)

def get_collection():
    return client.get_or_create_collection(name="kb_docs", metadata={"hnsw:space": "cosine"})

def get_docs_by_source(col, source_name: str):
    # Si hay muchos, pagina por 'where' extendido (no implementado aquí)
    return col.get(where={"source": source_name}, include=["documents", "metadatas", "ids"])

def check_sizes_and_overlap(docs, max_chars=1200, min_chars=50, expected_overlap=150, tolerance=25):
    problems = {"too_short": [], "too_long": [], "overlap_mismatch": []}
    # Agrupar por página
    by_page = {}
    for d, m in zip(docs["documents"], docs["metadatas"]):
        by_page.setdefault(m["page"], []).append(d)

    for page, chunks in by_page.items():
        for i, c in enumerate(chunks):
            if len(c) < min_chars: problems["too_short"].append((page, i, len(c)))
            if len(c) > max_chars: problems["too_long"].append((page, i, len(c)))
            if i < len(chunks) - 1:
                a, b = chunks[i], chunks[i + 1]
                overlap_a = a[-expected_overlap:]
                # Comprobación heurística del solape
                if overlap_a not in b[:expected_overlap + tolerance]:
                    problems["overlap_mismatch"].append((page, i))
    return problems

def ids_unique(docs):
    ids = docs["ids"]
    return len(ids) == len(set(ids))

if __name__ == "__main__":
    col = get_collection()
    source = input("Nombre de archivo (tal como metadato 'source'): ").strip()
    data = get_docs_by_source(col, source)
    probs = check_sizes_and_overlap(data)
    print("Problemas:", probs)
    print("IDs únicos:", ids_unique(data))
    print("Total docs:", len(data.get("documents", [])))