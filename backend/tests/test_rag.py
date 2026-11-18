import numpy as np

from app.rag.chunking import chunk_pages, is_heading
from app.rag.models import Chunk, Hit
from app.rag.pdf import clean_page_text
from app.rag.retrieval import rrf, search
from app.rag.store import MemoryStore
from app.rag.text import tokenize


def test_clean_page_text_junta_hifenizacao():
    assert clean_page_text("as inscri-\nções   abertas") == "as inscrições abertas"


def test_is_heading():
    assert is_heading("3. DAS INSCRIÇÕES")
    assert is_heading("ANEXO II - ATRIBUIÇÕES DOS CARGOS")
    assert is_heading("DAS PROVAS")
    assert not is_heading("3.1 As inscrições serão realizadas pela internet.")
    assert not is_heading("R$ 85,00")


def test_chunk_respeita_pagina_secao_e_tamanho():
    paginas = [
        "1. DAS INSCRIÇÕES\n" + ("texto da inscrição " * 120),
        "continua na página dois\n2. DAS PROVAS\nA prova será em 14 de março.",
    ]
    chunks = chunk_pages("d", paginas, max_chars=400, overlap=80)
    assert all(len(c.texto) <= 400 + 80 for c in chunks)
    assert {c.pagina for c in chunks} == {1, 2}
    # trecho da página 2 antes do título herda a seção anterior
    p2 = [c for c in chunks if c.pagina == 2]
    assert p2[0].secao == "1. DAS INSCRIÇÕES"
    assert p2[-1].secao == "2. DAS PROVAS"
    assert "14 de março" in p2[-1].texto
    # sobreposição: o começo do 2º trecho repete o fim do 1º
    assert chunks[1].texto.split()[0] in chunks[0].texto


def test_tokenize_tira_acento_e_stopword():
    toks = tokenize("Quais são as inscrições de R$ 85,00?")
    assert "inscricoes" in toks and "85,00" in toks and "as" not in toks


def _hit(cid: str) -> Hit:
    return Hit(Chunk(doc_id="d", ordem=0, pagina=1, secao=None, texto=cid, id=cid), 1.0, ["x"])


def test_rrf_premia_quem_aparece_nas_duas_listas():
    vetor = [_hit("a"), _hit("b"), _hit("c")]
    texto = [_hit("c"), _hit("d")]
    out = rrf([vetor, texto], k=4)
    assert out[0].chunk.id == "c"
    assert out[1].chunk.id == "a"
    assert {h.chunk.id for h in out[2:]} == {"b", "d"}
    assert out[0].fontes == ["x"]


def test_memory_store_persiste(tmp_path, pdfs, embedder):
    from app.rag.ingest import ingest_pdf

    s = MemoryStore(tmp_path)
    ingest_pdf(pdfs[1].read_bytes(), "Vale Azul", s, embedder, doc_id="va")
    s2 = MemoryStore(tmp_path)
    assert [d.id for d in s2.list_documents()] == ["va"]
    assert len(s2.chunks) == len(s.chunks)
    assert np.allclose(s2.vectors, s.vectors)
    s2.delete_document("va")
    assert MemoryStore(tmp_path).list_documents() == []


def test_busca_hibrida_acha_a_pagina_certa(store, embedder):
    hits = search(store, embedder, "valor da inscrição nível médio R$ 85,00", k=3, doc_id="vila-serena")
    assert any("R$ 85,00" in h.chunk.texto for h in hits)
    assert all(h.chunk.doc_id == "vila-serena" for h in hits)


def test_busca_texto_filtra_por_edital(store):
    hits = store.text_search("salário analista", 5, doc_id="vale-azul")
    assert hits and all(h.chunk.doc_id == "vale-azul" for h in hits)
