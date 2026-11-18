from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .agent.loop import run_agent
from .agent.tools import AgentContext, Tool, build_tools
from .config import Settings, get_settings
from .llm.base import ChatModel, Embedder, LLMError
from .rag.ingest import ingest_pdf
from .rag.pdf import PdfError
from .rag.seed import seed_from_manifest
from .rag.store import MemoryStore, Store

log = logging.getLogger("edital")
BRASILIA = timezone(timedelta(hours=-3))


@dataclass
class Services:
    settings: Settings
    store: Store
    embedder: Embedder | None
    model: ChatModel | None
    tools: dict[str, Tool]


def build_services(settings: Settings) -> Services:
    if settings.database_url:
        from .rag.pg_store import PgStore

        store: Store = PgStore(settings.database_url)
    else:
        store = MemoryStore(settings.memory_store_dir)
    embedder = model = None
    if settings.gemini_api_key:
        from .llm.fallback import FallbackChatModel
        from .llm.gemini import GeminiChatModel, GeminiEmbedder

        embedder = GeminiEmbedder(settings.gemini_api_key, settings.gemini_embed_model, settings.embed_dim)
        nomes = [settings.gemini_model] + [n.strip() for n in settings.gemini_fallback_models.split(",") if n.strip()]
        nomes = list(dict.fromkeys(nomes))
        if len(nomes) == 1:
            model = GeminiChatModel(settings.gemini_api_key, nomes[0])
        else:
            # com fallback, 1 retentativa por modelo basta: melhor trocar de modelo do que esperar
            model = FallbackChatModel([GeminiChatModel(settings.gemini_api_key, n, retries=1) for n in nomes])
    return Services(settings, store, embedder, model, build_tools())


class RateLimiter:
    """Janela deslizante por IP, em memória. Suficiente pra uma instância só."""

    def __init__(self, limit: int, window_s: int = 3600):
        self.limit, self.window = limit, window_s
        self.hits: dict[str, deque] = defaultdict(deque)

    def check(self, key: str) -> None:
        now = time.monotonic()
        q = self.hits[key]
        while q and now - q[0] > self.window:
            q.popleft()
        if len(q) >= self.limit:
            raise HTTPException(429, "Muitas requisições. Espera um pouco e tenta de novo.")
        q.append(now)


def client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    return fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "?")


class ChatIn(BaseModel):
    mensagem: str = Field(min_length=1, max_length=2000)
    historico: list[dict] = Field(default_factory=list, max_length=20)
    edital_id: str | None = None


def create_app(services: Services | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not hasattr(app.state, "services"):
            app.state.services = build_services(get_settings())
        s = app.state.services
        if s.settings.seed_on_startup and s.embedder and not s.store.list_documents():
            try:
                seed_from_manifest(s.store, s.embedder)
            except Exception:  # sem edital a API ainda sobe; o erro fica no log
                log.exception("Falha ao carregar os editais de exemplo")
        log.warning(
            "Edital na mão: banco=%s ia=%s upload=%s",
            s.store.kind,
            s.model.name if s.model else "desligada (sem GEMINI_API_KEY)",
            s.settings.allow_upload,
        )
        yield

    app = FastAPI(title="Edital na mão", lifespan=lifespan)
    if services is not None:
        app.state.services = services
    settings = services.settings if services else get_settings()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in settings.cors_origins.split(",")],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    chat_limit = RateLimiter(settings.rate_limit_chat_per_hour)
    upload_limit = RateLimiter(settings.rate_limit_upload_per_hour)

    def svc(request: Request) -> Services:
        return request.app.state.services

    @app.get("/api/health")
    def health(request: Request):
        s = svc(request)
        return {
            "ok": True,
            "banco": s.store.kind,
            "ia": s.model is not None,
            "modelo": s.model.name if s.model else None,
            "upload": s.settings.allow_upload,
        }

    @app.get("/api/editais")
    def listar(request: Request):
        return [d.to_dict() for d in svc(request).store.list_documents()]

    @app.post("/api/editais", status_code=201)
    async def enviar(
        request: Request,
        arquivo: UploadFile = File(...),
        titulo: str = Form(..., min_length=3, max_length=120),
        fonte_url: str | None = Form(None),
    ):
        s = svc(request)
        if not s.settings.allow_upload:
            raise HTTPException(403, "Upload desligado nessa demo.")
        if s.embedder is None:
            raise HTTPException(503, "IA desligada: configure GEMINI_API_KEY.")
        upload_limit.check(client_ip(request))
        if fonte_url and not fonte_url.startswith(("http://", "https://")):
            raise HTTPException(422, "O link da fonte precisa começar com http.")
        data = await arquivo.read(s.settings.max_upload_mb * 1024 * 1024 + 1)
        if len(data) > s.settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(413, f"Arquivo maior que {s.settings.max_upload_mb} MB.")
        if not data.startswith(b"%PDF"):
            raise HTTPException(415, "Mande um PDF.")
        try:
            doc = ingest_pdf(data, titulo, s.store, s.embedder, fonte_url, s.settings.max_pages)
        except PdfError as exc:
            raise HTTPException(422, str(exc)) from exc
        except LLMError as exc:
            raise HTTPException(exc.status, str(exc)) from exc
        return doc.to_dict()

    @app.post("/api/chat")
    def chat(body: ChatIn, request: Request):
        s = svc(request)
        if s.model is None or s.embedder is None:
            raise HTTPException(503, "IA desligada: configure GEMINI_API_KEY.")
        chat_limit.check(client_ip(request))
        if body.edital_id and s.store.get_document(body.edital_id) is None:
            raise HTTPException(404, "Edital não encontrado.")
        ctx = AgentContext(
            store=s.store,
            embedder=s.embedder,
            hoje=datetime.now(BRASILIA).date(),
            top_k=s.settings.search_top_k,
            edital_padrao=body.edital_id,
        )
        try:
            result = run_agent(s.model, s.tools, ctx, body.mensagem, body.historico, s.settings.agent_max_steps)
        except LLMError as exc:
            raise HTTPException(exc.status, str(exc)) from exc
        return result.to_dict()

    # Frontend buildado (frontend/dist) servido pela própria API: um serviço só no deploy.
    dist = Path(settings.frontend_dist)
    if (dist / "index.html").exists():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):
            if path.startswith("api/"):
                raise HTTPException(404, "Rota não encontrada.")
            arquivo = (dist / path).resolve()
            if path and arquivo.is_file() and dist.resolve() in arquivo.parents:
                return FileResponse(arquivo)
            # index.html sem cache: depois de um deploy o navegador pega o bundle novo
            return FileResponse(dist / "index.html", headers={"Cache-Control": "no-cache"})

    return app


app = create_app()
