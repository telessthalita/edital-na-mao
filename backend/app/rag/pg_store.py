"""Store em Postgres + pgvector."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from .models import Chunk, Document, Hit

SCHEMA = Path(__file__).resolve().parents[2] / "sql" / "schema.sql"


def _configure(conn):
    register_vector(conn)


class PgStore:
    kind = "postgres"

    def __init__(self, dsn: str, ensure_schema: bool = True):
        # prepare_threshold=None: o pooler do Supabase (modo transaction)
        # não suporta prepared statements.
        conn_kwargs = {"prepare_threshold": None, "autocommit": True}
        if ensure_schema:
            # Roda antes do pool porque o tipo vector só existe depois da extensão.
            with psycopg.connect(dsn, **conn_kwargs) as conn:
                conn.execute(SCHEMA.read_text(encoding="utf-8"))
        self.pool = ConnectionPool(dsn, min_size=1, max_size=5, kwargs=conn_kwargs, configure=_configure, open=True)

    def add_document(self, doc: Document, chunks: list[Chunk], vectors: np.ndarray) -> None:
        doc.total_trechos = len(chunks)
        with self.pool.connection() as conn, conn.transaction():
            conn.execute("delete from editais where id = %s", (doc.id,))
            conn.execute(
                "insert into editais (id, titulo, paginas, total_trechos, fonte_url) values (%s, %s, %s, %s, %s)",
                (doc.id, doc.titulo, doc.paginas, doc.total_trechos, doc.fonte_url),
            )
            with conn.cursor() as cur:
                cur.executemany(
                    "insert into trechos (id, edital_id, ordem, pagina, secao, texto, embedding)"
                    " values (%s, %s, %s, %s, %s, %s, %s)",
                    [
                        (c.id, c.doc_id, c.ordem, c.pagina, c.secao, c.texto, np.asarray(v, dtype=np.float32))
                        for c, v in zip(chunks, vectors)
                    ],
                )

    def list_documents(self) -> list[Document]:
        with self.pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            rows = cur.execute("select * from editais order by criado_em").fetchall()
        return [self._doc(r) for r in rows]

    def get_document(self, doc_id: str) -> Document | None:
        with self.pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            row = cur.execute("select * from editais where id = %s", (doc_id,)).fetchone()
        return self._doc(row) if row else None

    def delete_document(self, doc_id: str) -> None:
        with self.pool.connection() as conn:
            conn.execute("delete from editais where id = %s", (doc_id,))

    def vector_search(self, query_vec: np.ndarray, k: int, doc_id: str | None = None) -> list[Hit]:
        sql = (
            "select id, edital_id, ordem, pagina, secao, texto, 1 - (embedding <=> %(q)s) as score"
            " from trechos where (%(doc)s::text is null or edital_id = %(doc)s)"
            " order by embedding <=> %(q)s limit %(k)s"
        )
        params = {"q": np.asarray(query_vec, dtype=np.float32), "doc": doc_id, "k": k}
        return self._hits(sql, params, "vetor")

    def text_search(self, query: str, k: int, doc_id: str | None = None) -> list[Hit]:
        # plainto_tsquery junta os termos com AND, e pergunta em linguagem
        # natural quase nunca bate todos. Troca por OR e deixa o ranking
        # (ts_rank_cd) premiar quem bate mais termos.
        sql = (
            "with q as (select nullif(replace(plainto_tsquery('portuguese', %(q)s)::text, '&', '|'), '')"
            " as txt)"
            " select t.id, t.edital_id, t.ordem, t.pagina, t.secao, t.texto,"
            " ts_rank_cd(t.tsv, to_tsquery('portuguese', q.txt)) as score"
            " from trechos t, q"
            " where q.txt is not null and t.tsv @@ to_tsquery('portuguese', q.txt)"
            " and (%(doc)s::text is null or t.edital_id = %(doc)s)"
            " order by score desc limit %(k)s"
        )
        return self._hits(sql, {"q": query, "doc": doc_id, "k": k}, "texto")

    def _hits(self, sql: str, params: dict, fonte: str) -> list[Hit]:
        with self.pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            rows = cur.execute(sql, params).fetchall()
        return [
            Hit(
                Chunk(
                    id=r["id"],
                    doc_id=r["edital_id"],
                    ordem=r["ordem"],
                    pagina=r["pagina"],
                    secao=r["secao"],
                    texto=r["texto"],
                ),
                float(r["score"]),
                [fonte],
            )
            for r in rows
        ]

    @staticmethod
    def _doc(r: dict) -> Document:
        return Document(
            id=r["id"],
            titulo=r["titulo"],
            paginas=r["paginas"],
            total_trechos=r["total_trechos"],
            fonte_url=r["fonte_url"],
            criado_em=r["criado_em"].isoformat(),
        )
