"""Implementações sem rede, usadas nos testes e no modo offline da avaliação."""

from __future__ import annotations

import hashlib

import numpy as np

from ..rag.text import tokenize
from .base import Message, ModelTurn, ToolSpec


class HashEmbedder:
    """Bag of words com hashing. Não entende sinônimo, mas é determinístico."""

    name = "hash"

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, texts: list[str], task: str) -> np.ndarray:
        arr = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in tokenize(text):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                arr[i, h % self.dim] += 1.0
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms


class ScriptedChatModel:
    """Devolve turnos pré-definidos, em ordem. Guarda o que recebeu."""

    name = "roteiro"

    def __init__(self, turns: list[ModelTurn]):
        self._turns = list(turns)
        self.calls: list[dict] = []

    def generate(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> ModelTurn:
        self.calls.append({"system": system, "messages": list(messages), "tools": [t.name for t in tools]})
        if not self._turns:
            return ModelTurn(text="(fim do roteiro)", tool_calls=[])
        return self._turns.pop(0)
