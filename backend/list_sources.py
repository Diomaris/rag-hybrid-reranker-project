# list_sources.py
import json
import chromadb
from chromadb.config import Settings
from collections import Counter

HOST, PORT = "localhost", 8000
COLLECTION = "kb_docs"  # cámbialo si tu app.py usa otro nombre

def iter_metadatas(col, batch=2000):
    off = 0
    while True:
        r = col.get(include=["metadatas"], limit=batch, offset=off)
        metas = r.get("metadatas", [])
        n = len(metas)
        if n == 0:
            break
        for m in metas:
            yield m or {}
        off += batch

def main():
    client = chromadb.HttpClient(host=HOST, port=PORT, settings=Settings(allow_reset=True))
    col = client.get_or_create_collection(COLLECTION)
    c = Counter()
    total = 0
    for m in iter_metadatas(col):
        total += 1
        c[m.get("source", "<?>")] += 1

    result = {
        "collection": COLLECTION,
        "total_items": total,
        "by_source": dict(sorted(c.items(), key=lambda x: (-x[1], x[0])))
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()