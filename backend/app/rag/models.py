from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone


@dataclass
class Document:
    id: str
    titulo: str
    paginas: int
    total_trechos: int = 0
    fonte_url: str | None = None
    criado_em: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Chunk:
    doc_id: str
    ordem: int
    pagina: int
    secao: str | None
    texto: str
    id: str = ""

    def __post_init__(self):
        if not self.id:
            self.id = f"{self.doc_id}#{self.ordem}"

    def texto_para_embedding(self) -> str:
        # Prefixar a seção dá contexto pro vetor: um trecho solto "R$ 95,00"
        # vira "3. DAS INSCRIÇÕES ... R$ 95,00".
        return f"{self.secao}\n{self.texto}" if self.secao else self.texto


@dataclass
class Hit:
    chunk: Chunk
    score: float
    fontes: list[str] = field(default_factory=list)  # quais buscas acharam: vetor, texto
