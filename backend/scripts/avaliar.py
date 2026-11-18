"""Avalia a busca em três modos: só vetor, só texto (BM25) e híbrida (RRF).

Para cada pergunta-gabarito, a página certa é a que contém o trecho esperado.
Métricas:
- hit@1 / hit@5: a página certa apareceu no 1º / entre os 5 primeiros trechos?
- MRR: média de 1/posição do primeiro acerto (1.0 = sempre em primeiro).

Uso:
    python -m scripts.avaliar              # embeddings do Gemini (precisa da chave)
    python -m scripts.avaliar --offline    # embeddings por hashing, sem rede
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from app.config import ROOT_DIR, get_settings
from app.llm.fake import HashEmbedder
from app.rag.ingest import ingest_pdf
from app.rag.pdf import extract_pages
from app.rag.retrieval import search
from app.rag.store import MemoryStore
from app.rag.text import strip_accents

EDITAIS = ROOT_DIR / "data" / "editais"
PERGUNTAS = ROOT_DIR / "eval" / "perguntas.json"
MODOS = ["vetor", "texto", "hibrida"]


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", strip_accents(s.lower())).strip()


class CachedEmbedder:
    """Não paga duas vezes o embedding da mesma pergunta (modos vetor e híbrida)."""

    def __init__(self, inner):
        self.inner, self.name, self.dim, self.cache = inner, inner.name, inner.dim, {}

    def embed(self, texts, task):
        import numpy as np

        if task != "consulta" or len(texts) != 1:
            return self.inner.embed(texts, task)
        if texts[0] not in self.cache:
            self.cache[texts[0]] = self.inner.embed(texts, task)
        return np.array(self.cache[texts[0]])


def avaliar(embedder, k: int = 5) -> tuple[dict, list[dict]]:
    manifesto = json.loads((EDITAIS / "editais.json").read_text(encoding="utf-8"))
    store = MemoryStore()
    paginas: dict[str, list[str]] = {}
    for item in manifesto:
        data = (EDITAIS / item["arquivo"]).read_bytes()
        ingest_pdf(data, item["titulo"], store, embedder, doc_id=item["id"])
        paginas[item["id"]] = [norm(p) for p in extract_pages(data)]

    perguntas = json.loads(PERGUNTAS.read_text(encoding="utf-8"))
    metricas = {m: {"hit1": 0, "hit5": 0, "mrr": 0.0} for m in MODOS}
    detalhes = []
    for q in perguntas:
        alvo = {i + 1 for i, p in enumerate(paginas[q["edital"]]) if norm(q["esperado"]) in p}
        if not alvo:
            raise SystemExit(f"Trecho esperado não existe no edital: {q}")
        linha = {"pergunta": q["pergunta"], "paginas_certas": sorted(alvo)}
        for modo in MODOS:
            hits = search(store, embedder, q["pergunta"], k=k, doc_id=q["edital"], mode=modo)
            pos = next((i + 1 for i, h in enumerate(hits) if h.chunk.pagina in alvo and
                        norm(q["esperado"]) in norm(h.chunk.texto)), None)
            if pos is None:  # aceita a página certa mesmo se o trecho esperado foi cortado
                pos = next((i + 1 for i, h in enumerate(hits) if h.chunk.pagina in alvo), None)
            linha[modo] = pos
            if pos:
                metricas[modo]["hit1"] += pos == 1
                metricas[modo]["hit5"] += 1
                metricas[modo]["mrr"] += 1 / pos
        detalhes.append(linha)
    n = len(perguntas)
    for m in MODOS:
        metricas[m] = {"hit@1": metricas[m]["hit1"] / n, "hit@5": metricas[m]["hit5"] / n, "MRR": metricas[m]["mrr"] / n}
    return metricas, detalhes


def tabela(metricas: dict, n: int, embedder_nome: str) -> str:
    nomes = {"vetor": "Só vetor", "texto": "Só texto (BM25)", "hibrida": "Híbrida (RRF)"}
    linhas = [
        f"Perguntas: {n} | embeddings: {embedder_nome}",
        "",
        "| Modo | hit@1 | hit@5 | MRR |",
        "| --- | --- | --- | --- |",
    ]
    for m in MODOS:
        r = metricas[m]
        linhas.append(f"| {nomes[m]} | {r['hit@1']:.0%} | {r['hit@5']:.0%} | {r['MRR']:.2f} |")
    return "\n".join(linhas)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="usa HashEmbedder em vez do Gemini")
    parser.add_argument("--saida", type=Path, help="salva a tabela em markdown")
    args = parser.parse_args()

    if args.offline:
        base = HashEmbedder()
    else:
        s = get_settings()
        if not s.gemini_api_key:
            raise SystemExit("Sem GEMINI_API_KEY. Use --offline ou configure backend/.env")
        from app.llm.gemini import GeminiEmbedder

        base = GeminiEmbedder(s.gemini_api_key, s.gemini_embed_model, s.embed_dim)
    embedder = CachedEmbedder(base)
    metricas, detalhes = avaliar(embedder)
    out = tabela(metricas, len(detalhes), base.name)
    print(out)
    print("\nPosição do primeiro acerto (- = não achou no top 5):")
    for d in detalhes:
        pos = "  ".join(f"{m}={d[m] or '-'}" for m in MODOS)
        print(f"  {pos}  | {d['pergunta']}")
    if args.saida:
        args.saida.write_text(out + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
