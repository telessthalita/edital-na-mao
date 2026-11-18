from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = BACKEND_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BACKEND_DIR / ".env", extra="ignore")

    # IA
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    gemini_embed_model: str = "gemini-embedding-001"
    embed_dim: int = 768

    # Banco: vazio = store em memória (salvo em disco em memory_store_dir)
    database_url: str = ""
    memory_store_dir: str = str(ROOT_DIR / "data" / "store")
    # Banco vazio ao subir? Carrega os editais de data/editais/editais.json.
    seed_on_startup: bool = True

    # Agente
    agent_max_steps: int = 6
    search_top_k: int = 6

    # Proteção da demo pública
    allow_upload: bool = True
    max_upload_mb: int = 15
    max_pages: int = 200
    rate_limit_chat_per_hour: int = 40
    rate_limit_upload_per_hour: int = 5

    cors_origins: str = "*"
    frontend_dist: str = str(ROOT_DIR / "frontend" / "dist")


@lru_cache
def get_settings() -> Settings:
    return Settings()
