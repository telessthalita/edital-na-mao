"""Onde os trechos moram.

Dois stores com a mesma interface:
- MemoryStore: numpy + BM25, salvo em disco. Roda sem banco nenhum.
- PgStore (pg_store.py): Postgres + pgvector + full text search. Usado no deploy.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Protocol

import numpy as np

from .models import Chunk, Document, Hit
from .text import tokenize


class Store(Protocol):
    kind: str

    def add_document(self, doc: Document, chunks: list[Chunk], vectors: np.ndarray) -> None: ...
    def list_documents(self) -> list[Document]: ...
    def get_document(self, doc_id: str) -> Document | None: ...
    def delete_document(self, doc_id: str) -> None: ...
    def vector_search(self, query_vec: np.ndarray, k: int, doc_id: str | None = None) -> list[Hit]: ...
    def text_search(self, query: str, k: int, doc_id: str | None = None) -> list[Hit]: ...


class MemoryStore:
    kind = "memoria"

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.docs: dict[str, Document] = {}
        self.chunks: list[Chunk] = []
        self.vectors: np.ndarray | None = None
        self._tokens: list[Counter] = []
        self._df: Counter = Counter()
        if self.path and (self.path / "meta.json").exists():
            self._load()

    # --- escrita -----------------------------------------------------------
    def add_document(self, doc: Document, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if doc.id in self.docs:
            self.delete_document(doc.id)
        doc.total_trechos = len(chunks)
        self.docs[doc.id] = doc
        self.chunks.extend(chunks)
        vecs = np.asarray(vectors, dtype=np.float32)
        self.vectors = vecs if self.vectors is None else np.vstack([self.vectors, vecs])
        self._reindex()
        self._save()

    def delete_document(self, doc_id: str) -> None:
        keep = [i for i, c in enumerate(self.chunks) if c.doc_id != doc_id]
        self.chunks = [self.chunks[i] for i in keep]
        if self.vectors is not None:
            self.vectors = self.vectors[keep] if keep else None
        self.docs.pop(doc_id, None)
        self._reindex()
        self._save()

    # --- leitura -----------------------------------------------------------
    def list_documents(self) -> list[Document]:
        return sorted(self.docs.values(), key=lambda d: d.criado_em)

    def get_document(self, doc_id: str) -> Document | None:
        return self.docs.get(doc_id)

    def _mask(self, doc_id: str | None) -> np.ndarray:
        return np.array([doc_id is None or c.doc_id == doc_id for c in self.chunks], dtype=bool)

    def vector_search(self, query_vec: np.ndarray, k: int, doc_id: str | None = None) -> list[Hit]:
        if self.vectors is None or not self.chunks:
            return []
        scores = self.vectors @ np.asarray(query_vec, dtype=np.float32).ravel()
        scores = np.where(self._mask(doc_id), scores, -np.inf)
        order = np.argsort(-scores)[:k]
        return [Hit(self.chunks[i], float(scores[i]), ["vetor"]) for i in order if np.isfinite(scores[i])]

    def text_search(self, query: str, k: int, doc_id: str | None = None) -> list[Hit]:
        """BM25 clássico (k1=1.5, b=0.75)."""
        q = set(tokenize(query))
        if not q or not self.chunks:
            return []
        n = len(self.chunks)
        avg = sum(sum(t.values()) for t in self._tokens) / n
        scored = []
        for i, (chunk, tf) in enumerate(zip(self.chunks, self._tokens)):
            if doc_id and chunk.doc_id != doc_id:
                continue
            length = sum(tf.values())
            score = 0.0
            for term in q:
                if term not in tf:
                    continue
                idf = math.log(1 + (n - self._df[term] + 0.5) / (self._df[term] + 0.5))
                f = tf[term]
                score += idf * f * 2.5 / (f + 1.5 * (0.25 + 0.75 * length / avg))
            if score > 0:
                scored.append((score, i))
        scored.sort(reverse=True)
        return [Hit(self.chunks[i], s, ["texto"]) for s, i in scored[:k]]

    # --- internos ----------------------------------------------------------
    def _reindex(self) -> None:
        self._tokens = [Counter(tokenize(c.texto_para_embedding())) for c in self.chunks]
        self._df = Counter()
        for tf in self._tokens:
            self._df.update(tf.keys())

    def _save(self) -> None:
        if not self.path:
            return
        self.path.mkdir(parents=True, exist_ok=True)
        meta = {
            "docs": [d.to_dict() for d in self.docs.values()],
            "chunks": [c.__dict__ for c in self.chunks],
        }
        (self.path / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        if self.vectors is not None:
            np.save(self.path / "vectors.npy", self.vectors)
        elif (self.path / "vectors.npy").exists():
            (self.path / "vectors.npy").unlink()

    def _load(self) -> None:
        meta = json.loads((self.path / "meta.json").read_text(encoding="utf-8"))
        self.docs = {d["id"]: Document(**d) for d in meta["docs"]}
        self.chunks = [Chunk(**c) for c in meta["chunks"]]
        vec_file = self.path / "vectors.npy"
        self.vectors = np.load(vec_file) if vec_file.exists() else None
        self._reindex()
