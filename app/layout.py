from typing import List, Dict, Any
import pdfplumber
from pdf2image import convert_from_path
from PIL import Image
import io

def extract_pdf_blocks(path: str) -> List[Dict[str, Any]]:
    """
    Minimal layout detector:
    - uses pdfplumber to get text boxes (with bbox)
    - rasterize page for image processing if needed
    Returns list of blocks {type: text|table|image, page: int, bbox:..., content: ...}
    """
    blocks = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            # text extraction (verbatim)
            text = page.extract_text(x_tolerance=1)
            if text and text.strip():
                blocks.append({"type":"text","page":i,"content":text})
            # table detection fallback - Cursor should replace with camelot detection
            # We include images via rasterization for vision processing
            pil_page = page.to_image(resolution=150).original
            buf = io.BytesIO()
            pil_page.save(buf, format="PNG")
            img_bytes = buf.getvalue()
            blocks.append({"type":"image","page":i,"content_bytes":img_bytes})
    return blocks
