from datetime import date
from pathlib import Path

import pytest

from app.agent.tools import AgentContext
from app.llm.fake import HashEmbedder
from app.rag.ingest import ingest_pdf
from app.rag.store import MemoryStore
from scripts.gerar_editais_exemplo import main as gerar_editais


@pytest.fixture(scope="session")
def pdfs(tmp_path_factory) -> list[Path]:
    return gerar_editais(tmp_path_factory.mktemp("editais"))


@pytest.fixture
def embedder():
    return HashEmbedder()


@pytest.fixture
def store(pdfs, embedder) -> MemoryStore:
    s = MemoryStore()
    ingest_pdf(pdfs[0].read_bytes(), "Prefeitura de Vila Serena", s, embedder, doc_id="vila-serena")
    ingest_pdf(pdfs[1].read_bytes(), "Saneamento Vale Azul", s, embedder, doc_id="vale-azul")
    return s


@pytest.fixture
def ctx(store, embedder) -> AgentContext:
    return AgentContext(store=store, embedder=embedder, hoje=date(2026, 11, 18))
