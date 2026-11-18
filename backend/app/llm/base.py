"""Contratos neutros de provedor.

O agente só conversa com essas classes. Trocar Gemini por Groq, Ollama ou
Claude é escrever outro ChatModel/Embedder, sem tocar no loop do agente.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import numpy as np


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict  # JSON Schema (type object)


@dataclass
class ToolCall:
    name: str
    args: dict
    id: str | None = None


@dataclass
class ToolResult:
    name: str
    content: dict
    id: str | None = None


@dataclass
class Message:
    role: Literal["user", "assistant", "tool"]
    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    # Conteúdo original do provedor (ex: Gemini manda "thought signatures"
    # que precisam voltar intactas no próximo turno).
    raw: Any = None


@dataclass
class ModelTurn:
    text: str | None
    tool_calls: list[ToolCall]
    raw: Any = None


class LLMError(RuntimeError):
    def __init__(self, message: str, status: int = 502):
        super().__init__(message)
        self.status = status


class ChatModel(Protocol):
    name: str

    def generate(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> ModelTurn: ...


EmbedTask = Literal["documento", "consulta"]


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str], task: EmbedTask) -> np.ndarray:
        """Devolve matriz (len(texts), dim) com vetores normalizados."""
        ...
