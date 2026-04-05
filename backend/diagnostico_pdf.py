# -*- coding: utf-8 -*-
"""
diagnostico_pdf.py
------------------
Herramienta de diagnóstico para entender qué extrae PyMuPDF de un PDF.
Úsala para depurar problemas de drop caps, tablas, páginas vacías, etc.

Uso:
  python diagnostico_pdf.py "docs\MiPDF.pdf" --page 1
  python diagnostico_pdf.py "docs\MiPDF.pdf" --page 21 --mode ocr-check
  python diagnostico_pdf.py "docs\MiPDF.pdf" --page 1 --mode blocks

Modos:
  blocks     → muestra todos los bloques de texto con bbox, tamaño de fuente y texto
  raw        → texto crudo extraído por PyMuPDF (sin ningún procesado)
  ocr-check  → comprueba si la página tiene texto nativo o es imagen pura
  drop-cap   → muestra bloques ordenados por tamaño de fuente (para ver drop caps)
  full       → todo lo anterior
"""

import sys, os, re, argparse
from pathlib import Path

try:
    import fitz
except ImportError:
    print("ERROR: PyMuPDF no está instalado. Ejecuta: pip install pymupdf")
    sys.exit(1)


def hr(char="─", width=70):
    print(char * width)


def show_raw(page):
    hr()
    print("MODO: TEXTO CRUDO (page.get_text('text'))")
    hr()
    txt = page.get_text("text")
    if not txt.strip():
        print("  [VACÍO — la página no tiene texto nativo]")
    else:
        print(txt[:3000])
        if len(txt) > 3000:
            print(f"  ... ({len(txt)} chars total, mostrando primeros 3000)")


def show_ocr_check(page, page_num):
    hr()
    print(f"MODO: OCR CHECK — página {page_num}")
    hr()

    txt = page.get_text("text").strip()
    blocks = page.get_text("dict").get("blocks", [])
    text_blocks = [b for b in blocks if b.get("type") == 0]
    img_blocks  = [b for b in blocks if b.get("type") == 1]

    print(f"  Texto nativo:   {len(txt)} chars")
    print(f"  Bloques texto:  {len(text_blocks)}")
    print(f"  Bloques imagen: {len(img_blocks)}")
    print()

    if not txt:
        print("  ⚠ PÁGINA SIN TEXTO NATIVO → necesita OCR")
        print("  → Usa --enable-ocr al ingestar")
    elif len(txt) < 50:
        print(f"  ⚠ TEXTO MUY ESCASO ({len(txt)} chars) → posiblemente necesita OCR")
    else:
        print(f"  ✓ Página con texto nativo suficiente")

    if img_blocks:
        print(f"\n  Imágenes encontradas:")
        for i, b in enumerate(img_blocks[:5]):
            x0,y0,x1,y1 = b["bbox"]
            w, h = x1-x0, y1-y0
            print(f"    [{i}] bbox=({x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f})  tamaño={w:.0f}×{h:.0f}pt")
        if len(img_blocks) > 5:
            print(f"    ... ({len(img_blocks)} imágenes total)")


def show_blocks(page, max_blocks=30):
    hr()
    print("MODO: BLOQUES DE TEXTO (con bbox y tamaño de fuente)")
    hr()

    blocks = page.get_text("dict").get("blocks", [])
    text_blocks = [b for b in blocks if b.get("type") == 0]
    W, H = page.rect.width, page.rect.height

    print(f"  Página: {W:.0f} × {H:.0f} pt  ({len(text_blocks)} bloques de texto)\n")

    for bi, b in enumerate(text_blocks[:max_blocks]):
        x0, y0, x1, y1 = b["bbox"]
        # Extraer todos los spans del bloque
        spans = [sp for ln in b.get("lines", []) for sp in ln.get("spans", [])]
        sizes = sorted(set(round(sp.get("size", 0), 1) for sp in spans if sp.get("size", 0) > 0))
        txt_raw = "".join(sp.get("text", "") for sp in spans).strip()
        txt_display = txt_raw[:80] + ("…" if len(txt_raw) > 80 else "")

        y_pct = y0 / H * 100
        flags = []
        if y_pct < 12:  flags.append("HEADER-ZONE")
        if y_pct > 90:  flags.append("FOOTER-ZONE")
        if len(txt_raw) <= 3 and sizes and max(sizes) > 20:
            flags.append("⚠ DROP-CAP?")

        print(f"  [{bi:02d}] y={y0:.0f}-{y1:.0f} ({y_pct:.0f}%)  x={x0:.0f}-{x1:.0f}"
              f"  sizes={sizes}  {' '.join(flags)}")
        print(f"       '{txt_display}'")
        print()

    if len(text_blocks) > max_blocks:
        print(f"  ... ({len(text_blocks) - max_blocks} bloques más, usa --max-blocks N para ver más)")


def show_drop_cap(page):
    hr()
    print("MODO: DROP CAP — bloques ordenados por tamaño de fuente (desc)")
    hr()

    blocks = page.get_text("dict").get("blocks", [])
    text_blocks = [b for b in blocks if b.get("type") == 0]
    W, H = page.rect.width, page.rect.height

    # Calcular body size (moda de fuentes)
    all_sizes = []
    for b in text_blocks:
        for ln in b.get("lines", []):
            for sp in ln.get("spans", []):
                s = sp.get("size", 0)
                if s > 0:
                    all_sizes.append(round(s, 1))

    if all_sizes:
        from collections import Counter
        body_size = Counter(all_sizes).most_common(1)[0][0]
        print(f"  Tamaño de cuerpo (moda): {body_size}pt")
        print(f"  Threshold drop cap (×2): {body_size*2:.1f}pt\n")
    else:
        body_size = 12
        print("  No se pudo calcular body_size, usando 12pt\n")

    # Ordenar por tamaño máximo descendente
    block_info = []
    for b in text_blocks:
        spans = [sp for ln in b.get("lines", []) for sp in ln.get("spans", [])]
        sizes = [sp.get("size", 0) for sp in spans if sp.get("size", 0) > 0]
        txt = "".join(sp.get("text", "") for sp in spans).strip()
        max_size = max(sizes) if sizes else 0
        x0,y0,x1,y1 = b["bbox"]
        block_info.append((max_size, y0, x0, txt, b["bbox"]))

    block_info.sort(reverse=True)

    for max_size, y0, x0, txt, bbox in block_info[:20]:
        is_dc = max_size >= body_size * 2 and len(txt) <= 3
        marker = "  ⚠ DROP CAP" if is_dc else ""
        txt_d = txt[:60] + ("…" if len(txt) > 60 else "")
        print(f"  size={max_size:.1f}pt  y={y0:.0f}  x={x0:.0f}  '{txt_d}'{marker}")


def main():
    ap = argparse.ArgumentParser(description="Diagnóstico de extracción PDF con PyMuPDF")
    ap.add_argument("pdf_path", help="Ruta al PDF")
    ap.add_argument("--page", type=int, default=1, help="Página a inspeccionar (1-based, default=1)")
    ap.add_argument("--mode", default="full",
                    choices=["blocks", "raw", "ocr-check", "drop-cap", "full"],
                    help="Modo de inspección (default=full)")
    ap.add_argument("--max-blocks", type=int, default=30, help="Máx bloques a mostrar en modo blocks")
    ap.add_argument("--scan-all", action="store_true",
                    help="Escanea todas las páginas y muestra cuáles están vacías")
    args = ap.parse_args()

    pdf_path = args.pdf_path
    if not os.path.exists(pdf_path):
        print(f"ERROR: No existe el archivo '{pdf_path}'")
        sys.exit(1)

    with fitz.open(pdf_path) as doc:
        total = len(doc)
        print(f"\n{'='*70}")
        print(f"PDF: {Path(pdf_path).name}  ({total} páginas)")
        print(f"{'='*70}\n")

        if args.scan_all:
            print("ESCANEO COMPLETO — texto nativo por página:")
            hr("-")
            for i in range(total):
                txt = doc[i].get_text("text").strip()
                blocks = doc[i].get_text("dict").get("blocks", [])
                imgs = len([b for b in blocks if b.get("type") == 1])
                status = "✓" if len(txt) >= 50 else ("⚠ ESCASA" if txt else "✗ VACÍA")
                print(f"  Pág {i+1:3d}: {status:10s}  {len(txt):5d} chars  {imgs} imgs")
            return

        page_idx = args.page - 1
        if not (0 <= page_idx < total):
            print(f"ERROR: Página {args.page} fuera de rango (1–{total})")
            sys.exit(1)

        page = doc[page_idx]
        print(f"Inspeccionando página {args.page} de {total}\n")

        if args.mode in ("raw", "full"):
            show_raw(page)
            print()

        if args.mode in ("ocr-check", "full"):
            show_ocr_check(page, args.page)
            print()

        if args.mode in ("blocks", "full"):
            show_blocks(page, args.max_blocks)
            print()

        if args.mode in ("drop-cap", "full"):
            show_drop_cap(page)
            print()


if __name__ == "__main__":
    main()
