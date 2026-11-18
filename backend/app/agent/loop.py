"""O loop do agente.

    pergunta -> modelo -> pediu ferramenta? -> executa -> devolve resultado -> modelo -> ... -> resposta

Sem framework: dá pra ler de cima a baixo e explicar cada passo.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field

from ..llm.base import ChatModel, Message, ToolResult
from .prompts import system_prompt
from .tools import AgentContext, Tool, ToolError

_REF = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


@dataclass
class TraceStep:
    passo: int
    ferramenta: str
    argumentos: dict
    resumo: str
    ms: int
    erro: bool = False


@dataclass
class Citation:
    n: int
    edital_id: str
    edital: str
    pagina: int
    secao: str | None
    trecho: str
    fonte_url: str | None


@dataclass
class AgentResult:
    resposta: str
    citacoes: list[Citation]
    trace: list[TraceStep]
    passos: int
    modelo: str
    ms: int
    limite_atingido: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def _resumo(nome: str, out: dict) -> str:
    if "erro" in out:
        return f"erro: {out['erro']}"
    if nome == "buscar_trecho":
        trechos = out.get("trechos", [])
        if not trechos:
            return "nenhum trecho encontrado"
        paginas = sorted({t["pagina"] for t in trechos})
        return f"{len(trechos)} trechos (pág. {', '.join(map(str, paginas))})"
    if nome == "calcular_dias_ate_prova":
        return f"{out['dias']} dias ({out['semanas']} semanas, {out['dias_uteis']} dias úteis)"
    if nome == "montar_cronograma":
        return f"{out['semanas_total']} semanas, {len(out['distribuicao'])} matérias"
    if nome == "comparar_editais":
        return f"{len(out['edital_a']['trechos'])} + {len(out['edital_b']['trechos'])} trechos"
    if nome == "listar_editais":
        return f"{len(out['editais'])} editais"
    return json.dumps(out, ensure_ascii=False)[:160]


def _limpar_refs(texto: str, validas: set[int]) -> tuple[str, list[int]]:
    """Tira refs inventadas ([9] que nenhuma ferramenta devolveu) e lista as usadas."""
    usadas: list[int] = []

    def troca(m: re.Match) -> str:
        nums = [int(x) for x in re.split(r"\s*,\s*", m.group(1))]
        boas = [n for n in nums if n in validas]
        for n in boas:
            if n not in usadas:
                usadas.append(n)
        return "".join(f"[{n}]" for n in boas)

    return _REF.sub(troca, texto), usadas


def _historico_limpo(historico: list[dict]) -> list[Message]:
    msgs = []
    for h in historico[-8:]:
        role = "assistant" if h.get("role") == "assistant" else "user"
        texto = str(h.get("content", ""))[:4000]
        if role == "assistant":
            texto = _REF.sub("", texto)  # refs antigas não valem nessa rodada
        msgs.append(Message(role=role, text=texto))
    return msgs


def run_agent(
    model: ChatModel,
    tools: dict[str, Tool],
    ctx: AgentContext,
    pergunta: str,
    historico: list[dict] | None = None,
    max_steps: int = 6,
) -> AgentResult:
    t_inicio = time.perf_counter()
    system = system_prompt(ctx.hoje, ctx.store.list_documents(), ctx.edital_padrao)
    specs = [t.spec for t in tools.values()]
    messages = _historico_limpo(historico or []) + [Message(role="user", text=pergunta)]
    trace: list[TraceStep] = []
    resposta: str | None = None
    passos = 0
    limite = False

    for passo in range(1, max_steps + 1):
        passos = passo
        turn = model.generate(system, messages, specs)
        messages.append(Message(role="assistant", text=turn.text, tool_calls=turn.tool_calls, raw=turn.raw))
        if not turn.tool_calls:
            resposta = turn.text
            break

        resultados = []
        for call in turn.tool_calls:
            t0 = time.perf_counter()
            tool = tools.get(call.name)
            erro = False
            try:
                if tool is None:
                    raise ToolError(f"Ferramenta '{call.name}' não existe.")
                out = tool.run(call.args or {}, ctx)
            except ToolError as exc:
                out, erro = {"erro": str(exc)}, True
            trace.append(
                TraceStep(
                    passo=passo,
                    ferramenta=call.name,
                    argumentos=call.args or {},
                    resumo=_resumo(call.name, out),
                    ms=int((time.perf_counter() - t0) * 1000),
                    erro=erro,
                )
            )
            resultados.append(ToolResult(name=call.name, content=out, id=call.id))
        messages.append(Message(role="tool", tool_results=resultados))
    else:
        # Bateu o limite de passos: pede uma resposta final sem ferramentas.
        limite = True
        turn = model.generate(
            system + "\nVocê atingiu o limite de buscas. Responda agora só com o que já encontrou.", messages, []
        )
        resposta = turn.text

    texto, usadas = _limpar_refs(resposta or "Não consegui montar uma resposta. Tenta reformular?", set(ctx.citacoes))
    docs = {d.id: d for d in ctx.store.list_documents()}
    citacoes = []
    for n in usadas:
        hit = ctx.citacoes[n]
        doc = docs.get(hit.chunk.doc_id)
        citacoes.append(
            Citation(
                n=n,
                edital_id=hit.chunk.doc_id,
                edital=doc.titulo if doc else hit.chunk.doc_id,
                pagina=hit.chunk.pagina,
                secao=hit.chunk.secao,
                trecho=hit.chunk.texto,
                fonte_url=doc.fonte_url if doc else None,
            )
        )
    return AgentResult(
        resposta=texto.strip(),
        citacoes=citacoes,
        trace=trace,
        passos=passos,
        modelo=model.name,
        ms=int((time.perf_counter() - t_inicio) * 1000),
        limite_atingido=limite,
    )
