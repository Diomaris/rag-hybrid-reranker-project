# read_ocr_progress.py
import json, pathlib

pdf_name = input("Nombre del PDF (incluida extensión): ").strip()
progress_path = pathlib.Path(".ingest_progress") / f"{pdf_name}.json"
with open(progress_path, "r", encoding="utf-8") as f:
    p = json.load(f)

print("\n--- Progreso OCR ---")
for k, v in p.items():
    print(f"{k}: {v}")