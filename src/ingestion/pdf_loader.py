# src/ingestion/pdf_loader.py
from pathlib import Path

# PyMuPDF renamed its public module; keep the fallback for older installations.
try:
    import pymupdf as fitz
except ImportError:
    import fitz  # PyMuPDF < 1.24
from PIL import Image


def pdf_to_images(pdf_path: str, dpi: int = 200) -> list[dict]:
    """
    Convert every page of a PDF into a PIL image.

    Returns a list of dicts:
        {"page_number": int, "image": PIL.Image, "source_file": str}

    Page numbers start at 1, matching what a human would call "page 1."
    """
    doc = fitz.open(pdf_path)
    zoom = dpi / 72  # PDF points are 72 per inch, this scales to the requested dpi
    matrix = fitz.Matrix(zoom, zoom)

    pages = []
    for page_index in range(len(doc)):
        page = doc.load_page(page_index)
        pix = page.get_pixmap(matrix=matrix)
        image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

        pages.append({
            "page_number": page_index + 1,
            "image": image,
            "source_file": Path(pdf_path).name,
        })

    doc.close()
    return pages


def save_page_images(pages: list[dict], output_dir: str) -> list[dict]:
    """
    Write each page image to disk and add its saved path to the dict.
    Needed because ChromaDB will store a path, not the raw image bytes.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for page in pages:
        stem = Path(page["source_file"]).stem
        filename = f"{stem}_page{page['page_number']:03d}.png"
        file_path = output_path / filename
        page["image"].save(file_path)
        page["image_path"] = str(file_path)

    return pages
