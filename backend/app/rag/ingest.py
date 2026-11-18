from __future__ import annotations

import hashlib

from ..llm.base import Embedder
from .chunking import chunk_pages
from .models import Document
from .pdf import extract_pages
from .store import Store
from .text import slugify


def ingest_pdf(
    data: bytes,
    titulo: str,
    store: Store,
    embedder: Embedder,
    fonte_url: str | None = None,
    max_pages: int | None = None,
    doc_id: str | None = None,
) -> Document:
    pages = extract_pages(data, max_pages=max_pages)
    doc_id = doc_id or f"{slugify(titulo, 32)}-{hashlib.sha1(data).hexdigest()[:6]}"
    chunks = chunk_pages(doc_id, pages)
    vectors = embedder.embed([c.texto_para_embedding() for c in chunks], "documento")
    doc = Document(id=doc_id, titulo=titulo.strip(), paginas=len(pages), fonte_url=fonte_url or None)
    store.add_document(doc, chunks, vectors)
    return doc
