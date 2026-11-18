import pytest
from fastapi.testclient import TestClient

from app.agent.tools import build_tools
from app.config import Settings
from app.llm.base import ModelTurn, ToolCall
from app.llm.fake import ScriptedChatModel
from app.main import Services, create_app
from app.rag.store import MemoryStore


@pytest.fixture
def services(embedder):
    settings = Settings(_env_file=None, frontend_dist="nao-existe", rate_limit_chat_per_hour=3)
    return Services(settings, MemoryStore(), embedder, ScriptedChatModel([]), build_tools())


def test_upload_e_chat(services, pdfs):
    client = TestClient(create_app(services))
    assert client.get("/api/health").json()["ia"] is True

    r = client.post(
        "/api/editais",
        files={"arquivo": ("e.pdf", pdfs[0].read_bytes(), "application/pdf")},
        data={"titulo": "Vila Serena"},
    )
    assert r.status_code == 201, r.text
    doc = r.json()
    assert doc["paginas"] == 5 and doc["total_trechos"] > 5
    assert client.get("/api/editais").json()[0]["id"] == doc["id"]

    services.model = ScriptedChatModel(
        [
            ModelTurn(None, [ToolCall("buscar_trecho", {"pergunta": "salário analista"})]),
            ModelTurn("O vencimento é R$ 6.850,00 [1].", []),
        ]
    )
    r = client.post("/api/chat", json={"mensagem": "salário?", "edital_id": doc["id"]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["citacoes"][0]["pagina"] >= 1
    assert body["trace"][0]["ferramenta"] == "buscar_trecho"


def test_rejeita_nao_pdf_e_upload_desligado(services):
    client = TestClient(create_app(services))
    r = client.post("/api/editais", files={"arquivo": ("a.pdf", b"oi", "application/pdf")}, data={"titulo": "abc"})
    assert r.status_code == 415
    services.settings.allow_upload = False
    r = client.post("/api/editais", files={"arquivo": ("a.pdf", b"%PDF", "application/pdf")}, data={"titulo": "abc"})
    assert r.status_code == 403


def test_rate_limit_do_chat(services):
    client = TestClient(create_app(services))
    codes = [client.post("/api/chat", json={"mensagem": "oi"}).status_code for _ in range(4)]
    assert codes == [200, 200, 200, 429]


def test_sem_chave_responde_503(services):
    services.model = None
    client = TestClient(create_app(services))
    assert client.post("/api/chat", json={"mensagem": "oi"}).status_code == 503
