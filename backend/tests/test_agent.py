import pytest

from app.agent.loop import run_agent
from app.agent.tools import ToolError, build_tools, calcular_dias_ate_prova, montar_cronograma, resolver_edital
from app.llm.base import ModelTurn, ToolCall
from app.llm.fake import ScriptedChatModel


def test_calcular_dias(ctx):
    out = calcular_dias_ate_prova({"data_prova": "2027-03-14"}, ctx)
    assert out["dias"] == 116
    assert out["semanas"] == 16.6
    assert out["ja_passou"] is False
    assert 0 < out["dias_uteis"] < out["dias"]


def test_calcular_dias_data_invalida(ctx):
    with pytest.raises(ToolError):
        calcular_dias_ate_prova({"data_prova": "14/03/2027"}, ctx)


def test_cronograma_proporcional_ao_peso(ctx):
    out = montar_cronograma(
        {
            "materias": [{"nome": "Português", "peso": 10}, {"nome": "Específicos", "peso": 40}],
            "horas_por_semana": 10,
            "data_prova": "2027-03-14",
        },
        ctx,
    )
    dist = {d["materia"]: d["horas_semana"] for d in out["distribuicao"]}
    assert dist == {"Específicos": 8.0, "Português": 2.0}
    assert out["semanas_total"] == 16 and out["ultima_semana_revisao"] is True
    assert out["distribuicao"][0]["horas_total"] == 8.0 * 15


def test_resolver_edital_por_nome(ctx):
    assert resolver_edital(ctx, "vale azul").id == "vale-azul"
    with pytest.raises(ToolError):
        resolver_edital(ctx, "edital que não existe")


def test_agente_busca_calcula_e_cita(ctx):
    modelo = ScriptedChatModel(
        [
            ModelTurn(None, [ToolCall("buscar_trecho", {"pergunta": "data da prova objetiva", "edital_id": "vila-serena"})]),
            ModelTurn(None, [ToolCall("calcular_dias_ate_prova", {"data_prova": "2027-03-14"})]),
            ModelTurn("A prova é em 14/3/2027 [1] e faltam 116 dias. [99]", []),
        ]
    )
    res = run_agent(modelo, build_tools(), ctx, "quando é a prova e quantos dias faltam?")

    assert [t.ferramenta for t in res.trace] == ["buscar_trecho", "calcular_dias_ate_prova"]
    assert res.trace[1].resumo.startswith("116 dias")
    # [99] não veio de ferramenta nenhuma: some da resposta
    assert "[99]" not in res.resposta and "[1]" in res.resposta
    assert len(res.citacoes) == 1 and res.citacoes[0].edital_id == "vila-serena"
    # o resultado da ferramenta voltou pro modelo
    ultima = modelo.calls[1]["messages"][-1]
    assert ultima.role == "tool" and ultima.tool_results[0].name == "buscar_trecho"
    assert "Vila Serena" in modelo.calls[0]["system"]


def test_agente_erro_de_ferramenta_volta_pro_modelo(ctx):
    modelo = ScriptedChatModel(
        [
            ModelTurn(None, [ToolCall("ferramenta_inventada", {})]),
            ModelTurn("Não achei isso no edital.", []),
        ]
    )
    res = run_agent(modelo, build_tools(), ctx, "?")
    assert res.trace[0].erro is True
    assert "não existe" in modelo.calls[1]["messages"][-1].tool_results[0].content["erro"]
    assert res.citacoes == []


def test_agente_respeita_limite_de_passos(ctx):
    busca = ModelTurn(None, [ToolCall("buscar_trecho", {"pergunta": "salário"})])
    modelo = ScriptedChatModel([busca, busca, busca, ModelTurn("Resposta final [1].", [])])
    res = run_agent(modelo, build_tools(), ctx, "salário?", max_steps=3)
    assert res.limite_atingido and res.passos == 3
    assert modelo.calls[-1]["tools"] == []
    assert res.resposta == "Resposta final [1]."


def test_historico_tira_refs_antigas(ctx):
    modelo = ScriptedChatModel([ModelTurn("ok", [])])
    run_agent(
        modelo,
        build_tools(),
        ctx,
        "e a taxa?",
        historico=[{"role": "user", "content": "salário?"}, {"role": "assistant", "content": "R$ 6.850 [3]"}],
    )
    msgs = modelo.calls[0]["messages"]
    assert [m.role for m in msgs] == ["user", "assistant", "user"]
    assert "[3]" not in msgs[1].text
