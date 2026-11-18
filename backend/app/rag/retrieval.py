"""Busca híbrida: vetor + texto, unidas por Reciprocal Rank Fusion.

RRF olha só a posição de cada trecho em cada lista (1/(60 + posição)), então
não precisa calibrar score de cosseno contra score de BM25, que vivem em
escalas diferentes.
"""

from __future__ import annotations

from typing import Literal

from ..llm.base import Embedder
from .models import Hit
from .store import Store

Mode = Literal["hibrida", "vetor", "texto"]
RRF_K = 60


def rrf(lists: list[list[Hit]], k: int, rrf_k: int = RRF_K) -> list[Hit]:
    scores: dict[str, float] = {}
    best: dict[str, Hit] = {}
    for hits in lists:
        for rank, hit in enumerate(hits):
            cid = hit.chunk.id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)
            if cid not in best:
                best[cid] = Hit(hit.chunk, 0.0, list(hit.fontes))
            else:
                for f in hit.fontes:
                    if f not in best[cid].fontes:
                        best[cid].fontes.append(f)
    ranked = sorted(scores, key=lambda cid: scores[cid], reverse=True)[:k]
    out = []
    for cid in ranked:
        hit = best[cid]
        hit.score = scores[cid]
        out.append(hit)
    return out


def search(
    store: Store,
    embedder: Embedder,
    query: str,
    k: int = 6,
    doc_id: str | None = None,
    mode: Mode = "hibrida",
    candidates: int = 20,
) -> list[Hit]:
    if mode == "texto":
        return store.text_search(query, k, doc_id)
    qvec = embedder.embed([query], "consulta")[0]
    if mode == "vetor":
        return store.vector_search(qvec, k, doc_id)
    by_vector = store.vector_search(qvec, candidates, doc_id)
    by_text = store.text_search(query, candidates, doc_id)
    return rrf([by_vector, by_text], k)
