import pytest

from app.llm.base import LLMError, ModelTurn
from app.llm.fallback import FallbackChatModel


class Modelo:
    def __init__(self, name, falhas=0, transient=True):
        self.name, self.falhas, self.transient, self.chamadas = name, falhas, transient, 0

    def generate(self, system, messages, tools):
        self.chamadas += 1
        if self.falhas:
            self.falhas -= 1
            raise LLMError("ocupado", 503, transient=self.transient)
        return ModelTurn(f"oi de {self.name}", [])


def test_cai_pro_proximo_e_pausa_o_que_falhou():
    agora = [0.0]
    a, b = Modelo("a", falhas=1), Modelo("b")
    m = FallbackChatModel([a, b], cooldown_s=300, clock=lambda: agora[0])
    assert m.generate("", [], []).text == "oi de b" and m.name == "b"
    # na pausa, nem tenta o "a"
    m.generate("", [], [])
    assert a.chamadas == 1
    # passou a pausa, volta pro principal
    agora[0] = 301
    assert m.generate("", [], []).text == "oi de a" and m.name == "a"


def test_erro_nao_transitorio_sobe_direto():
    a, b = Modelo("a", falhas=1, transient=False), Modelo("b")
    with pytest.raises(LLMError):
        FallbackChatModel([a, b]).generate("", [], [])
    assert b.chamadas == 0


def test_todos_falhando_devolve_o_ultimo_erro():
    with pytest.raises(LLMError):
        FallbackChatModel([Modelo("a", 5), Modelo("b", 5)]).generate("", [], [])
