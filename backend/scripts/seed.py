"""Carrega os editais de data/editais/editais.json no banco configurado.

Uso:
    python -m scripts.seed           # pula os que já existem
    python -m scripts.seed --force   # recarrega todos
"""

from __future__ import annotations

import argparse

from app.config import get_settings
from app.main import build_services
from app.rag.seed import seed_from_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    services = build_services(get_settings())
    if services.embedder is None:
        raise SystemExit("Sem GEMINI_API_KEY no backend/.env")
    docs = seed_from_manifest(services.store, services.embedder, force=args.force)
    for d in docs:
        print(f"carregado: {d.id} ({d.paginas} páginas, {d.total_trechos} trechos) em {services.store.kind}")
    if not docs:
        print("nada novo pra carregar")


if __name__ == "__main__":
    main()
