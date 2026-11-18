from __future__ import annotations

import json
import logging

from ..config import ROOT_DIR
from ..llm.base import Embedder
from .ingest import ingest_pdf
from .models import Document
from .store import Store

EDITAIS = ROOT_DIR / "data" / "editais"
log = logging.getLogger("edital")


def seed_from_manifest(store: Store, embedder: Embedder, force: bool = False) -> list[Document]:
    """Carrega os editais listados em data/editais/editais.json."""
    manifesto = EDITAIS / "editais.json"
    if not manifesto.exists():
        return []
    carregados = []
    for item in json.loads(manifesto.read_text(encoding="utf-8")):
        if not force and store.get_document(item["id"]):
            continue
        data = (EDITAIS / item["arquivo"]).read_bytes()
        doc = ingest_pdf(data, item["titulo"], store, embedder, fonte_url=item.get("fonte_url"), doc_id=item["id"])
        log.warning("edital carregado: %s (%s páginas, %s trechos)", doc.id, doc.paginas, doc.total_trechos)
        carregados.append(doc)
    return carregados
