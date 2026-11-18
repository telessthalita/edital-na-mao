"""Carrega os editais de data/editais/editais.json no banco configurado.

Uso:
    python -m scripts.seed           # pula os que já existem
    python -m scripts.seed --force   # recarrega todos
"""

from __future__ import annotations

import argparse
import json

from app.config import ROOT_DIR, get_settings
from app.main import build_services
from app.rag.ingest import ingest_pdf

EDITAIS = ROOT_DIR / "data" / "editais"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    services = build_services(get_settings())
    if services.embedder is None:
        raise SystemExit("Sem GEMINI_API_KEY no backend/.env")
    manifesto = json.loads((EDITAIS / "editais.json").read_text(encoding="utf-8"))
    for item in manifesto:
        if not args.force and services.store.get_document(item["id"]):
            print("já existe:", item["id"])
            continue
        data = (EDITAIS / item["arquivo"]).read_bytes()
        doc = ingest_pdf(
            data,
            item["titulo"],
            services.store,
            services.embedder,
            fonte_url=item.get("fonte_url"),
            doc_id=item["id"],
        )
        print(f"carregado: {doc.id} ({doc.paginas} páginas, {doc.total_trechos} trechos) em {services.store.kind}")


if __name__ == "__main__":
    main()
