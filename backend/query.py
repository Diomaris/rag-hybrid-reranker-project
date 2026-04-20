# query.py (añadido: --show-last-tail / --tail-len)
# Uso:
#   python query.py ".\docs\archivo.pdf" --show-last-tail --tail-len 220
#   python query.py ".\docs\archivo.pdf" --only-page 3 --show-last-tail --tail-len 300

import argparse, os, hashlib
from collections import defaultdict
from chromadb import HttpClient
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer  # opcional: --ask
from config import settings


client = chromadb.HttpClient(
    host=settings.CHROMA_HOST,
    port=settings.CHROMA_PORT,
    settings=Settings(allow_reset=True)
)


def connect_collection(host="localhost", port=8000, collection="kb_docs"):
    return client.get_or_create_collection(name=collection, metadata={"hnsw:space": "cosine"})

def fetch_by_source(col, source_name: str):
    # Compatible con builds donde include no acepta "ids"
    return col.get(where={"source": source_name}, include=["documents", "metadatas"])

def group_by_page(res):
    buckets = defaultdict(list)
    for doc, meta in zip(res.get("documents", []), res.get("metadatas", [])):
        buckets[meta.get("page", 0)].append(doc)
    # orden por número de página
    return dict(sorted(buckets.items(), key=lambda kv: kv[0]))

def show_summary(grouped):
    print("\n=== RESUMEN POR PÁGINA ===")
    total = 0
    for page, chunks in grouped.items():
        print(f"Página {page}: {len(chunks)} chunks")
        total += len(chunks)
    print(f"Total de chunks en este source: {total}\n")

def show_chunks(grouped, show_per_page=1, max_chars=1200, only_page=None, show_text=False):
    print(f"\n=== MUESTRA DE CHUNKS (primeros {show_per_page} por página; truncados a 300 chars) ===")
    for page, chunks in grouped.items():
        if only_page is not None and page != only_page:
            continue
        print(f"\n--- Página {page} ---")
        for i, c in enumerate(chunks[:show_per_page]):
            print(f"[chunk #{i}] len={len(c)} (<= {max_chars} esperado)")
            if show_text:
                prev = c[:300].replace("\n", " ")
                print(prev + ("..." if len(c) > 300 else ""))
    print()

def show_pairs_for_overlap(grouped, expected_overlap=150, look=120, only_page=None, limit=None, dedup_pairs=False):
    print(f"\n=== PAREJAS CONSECUTIVAS PARA VER SOLAPE (~{expected_overlap}) ===")
    seen = set()
    printed = 0
    for page, chunks in grouped.items():
        if only_page is not None and page != only_page:
            continue
        if len(chunks) < 2:
            continue
        print(f"\n--- Página {page} ---")
        for i in range(len(chunks) - 1):
            a, b = chunks[i], chunks[i+1]
            tail = a[-expected_overlap:][-look:]
            head = b[:expected_overlap][:look]
            if dedup_pairs:
                sig = (tail, head)
                if sig in seen:
                    continue
                seen.add(sig)
            print(f"(i={i}) tail(A) ≈{len(tail)} chars:\n» {tail}\n")
            print(f"(i={i}) head(B) ≈{len(head)} chars:\n« {head}\n")
            if tail and head and (tail in b[:expected_overlap + 25] or head in a[-(expected_overlap + 25):]):
                print("✔ Solape detectable (heurístico)\n")
            else:
                print("⚠ No se detectó solape claro (posible efecto de limpieza/espacios)\n")
            printed += 1
            if limit is not None and printed >= limit:
                print(f"(cortado a {limit} parejas)")
                return

def show_last_tail(grouped, tail_len=220, only_page=None):
    """
    Imprime el final del ÚLTIMO chunk de cada página (o de la página especificada).
    Útil para detectar si el pie de página entró al chunk.
    """
    print(f"\n=== ÚLTIMO CHUNK POR PÁGINA (últimos {tail_len} chars) ===")
    for page, chunks in grouped.items():
        if only_page is not None and page != only_page:
            continue
        if not chunks:
            continue
        last = chunks[-1]
        tail = last[-tail_len:]
        print(f"\n--- Página {page} ---")
        print(tail.replace("\n", " "))
    print()

def sanity_query(col, model_name, text_query, topk=5):
    print(f"\n=== SANITY QUERY (top-{topk}) ===")
    embedder = SentenceTransformer(model_name)
    qv = embedder.encode([text_query], normalize_embeddings=True).tolist()
    res = col.query(query_embeddings=qv, n_results=topk, include=["documents", "metadatas", "distances"])
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    for doc, meta, dist in zip(docs, metas, dists):
        print(f"[{meta['source']} p.{meta['page']}] dist={dist:.4f}")
        print(doc[:300].replace("\n", " ") + ("..." if len(doc) > 300 else ""))
        print("-")

def main():
    ap = argparse.ArgumentParser(description="Inspección de chunks por 'source' y verificación de fragmentado.")
    ap.add_argument("source", help="Ruta o nombre del PDF (se tomará basename como 'metadatas.source').")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--collection", default="kb_docs")
    ap.add_argument("--show-per-page", type=int, default=1)
    ap.add_argument("--overlap", type=int, default=150)
    ap.add_argument("--max-chars", type=int, default=1200)
    ap.add_argument("--show-pairs", action="store_true")
    ap.add_argument("--pairs-limit", type=int, default=None, help="Máximo de parejas a imprimir")
    ap.add_argument("--only-page", type=int, default=None, help="Inspeccionar solo esa página")
    ap.add_argument("--dedup-pairs", action="store_true", help="No repetir parejas tail/head idénticas")
    ap.add_argument("--show-text", action="store_true", help="Imprimir también muestra del texto del chunk")
    ap.add_argument("--show-last-tail", action="store_true", help="Imprimir el final del último chunk de cada página")
    ap.add_argument("--tail-len", type=int, default=220, help="Cantidad de caracteres a mostrar del final del último chunk")
    ap.add_argument("--dup-check", action="store_true", help="Analiza duplicados exactos/partiales por página")
    ap.add_argument("--ask", default=None, help="(Opcional) Texto de consulta para sanity query")
    ap.add_argument("--embed-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    args = ap.parse_args()

    source_name = os.path.basename(args.source)
    col = connect_collection(args.host, args.port, args.collection)
    data = fetch_by_source(col, source_name)

    docs = data.get("documents", [])
    if not docs:
        print(f"No encontré documentos para source='{source_name}'. ¿Coincide con 'metadatas.source'?")
        return

    grouped = group_by_page(data)
    show_summary(grouped)
    show_chunks(grouped, show_per_page=args.show_per_page, max_chars=args.max_chars,
                only_page=args.only_page, show_text=args.show_text)

    if args.show_pairs:
        show_pairs_for_overlap(grouped, expected_overlap=args.overlap, look=min(120, args.overlap),
                               only_page=args.only_page, limit=args.pairs_limit, dedup_pairs=args.dedup_pairs)

    if args.show_last_tail:
        show_last_tail(grouped, tail_len=args.tail_len, only_page=args.only_page)

    if args.dup_check:
        # Reutiliza la versión que ya te compartí si quieres esto; aquí omitimos por brevedad
        pass

    if args.ask:
        sanity_query(col, args.embed_model, args.ask, topk=5)

if __name__ == "__main__":
    main()