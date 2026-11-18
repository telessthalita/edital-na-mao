"""Fallback entre modelos, com pausa pro que falhou.

No tier gratuito o modelo principal às vezes responde 503 ("high demand").
Em vez de devolver erro, tenta o próximo da lista. O que falhou fica de castigo
por alguns minutos, pra próxima pergunta não perder tempo batendo nele de novo.
Serve pra qualquer provedor que implemente ChatModel.
"""

from __future__ import annotations

import logging
import time
from typing import Callable

from .base import ChatModel, LLMError, Message, ModelTurn, ToolSpec

log = logging.getLogger("edital")


class FallbackChatModel:
    def __init__(self, models: list[ChatModel], cooldown_s: float = 300, clock: Callable[[], float] = time.monotonic):
        if not models:
            raise ValueError("Precisa de pelo menos um modelo")
        self.models = models
        self.cooldown_s = cooldown_s
        self._clock = clock
        self._pausado_ate: dict[str, float] = {}
        self.name = models[0].name  # último modelo que respondeu

    def _ordem(self) -> list[ChatModel]:
        agora = self._clock()
        livres = [m for m in self.models if self._pausado_ate.get(m.name, 0) <= agora]
        pausados = [m for m in self.models if m not in livres]
        return livres + pausados  # se todos estão pausados, tenta assim mesmo

    def generate(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> ModelTurn:
        ultimo_erro: LLMError | None = None
        for model in self._ordem():
            try:
                turn = model.generate(system, messages, tools)
            except LLMError as exc:
                if not exc.transient:
                    raise
                log.warning("modelo %s indisponível (%s), tentando o próximo", model.name, exc)
                self._pausado_ate[model.name] = self._clock() + self.cooldown_s
                ultimo_erro = exc
                continue
            self.name = model.name
            return turn
        assert ultimo_erro is not None
        raise ultimo_erro
