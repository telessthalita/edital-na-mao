import io
import re

from pypdf import PdfReader


class PdfError(ValueError):
    pass


def clean_page_text(text: str) -> str:
    text = text.replace(" ", " ").replace("\r", "")
    # Junta palavra quebrada com hífen no fim da linha: "inscri-\nção" -> "inscrição"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pages(data: bytes, max_pages: int | None = None) -> list[str]:
    """Texto de cada página, na ordem. Índice 0 = página 1."""
    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as exc:  # pypdf solta vários tipos
        raise PdfError("Não consegui abrir esse PDF.") from exc
    if reader.is_encrypted:
        raise PdfError("PDF protegido por senha.")
    total = len(reader.pages)
    if max_pages and total > max_pages:
        raise PdfError(f"PDF tem {total} páginas; o limite é {max_pages}.")
    pages = [clean_page_text(p.extract_text() or "") for p in reader.pages]
    if sum(len(p) for p in pages) < 200:
        raise PdfError("Quase não achei texto. Parece PDF escaneado (imagem), e OCR ainda não é suportado.")
    return pages
