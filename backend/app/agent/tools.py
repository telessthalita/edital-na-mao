"""Ferramentas que o agente pode chamar.

Cada ferramenta é uma função Python comum com um JSON Schema do lado. O modelo
escolhe qual chamar e com quais argumentos; quem executa é o nosso código.
Conta de data e divisão de horas ficam aqui, determinísticas, porque LLM erra
aritmética.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Callable

from ..llm.base import Embedder, ToolSpec
from ..rag.models import Document, Hit
from ..rag.retrieval import search
from ..rag.store import Store
from ..rag.text import strip_accents


class ToolError(ValueError):
    """Erro que volta pro modelo como resultado, pra ele poder se corrigir."""


@dataclass
class AgentContext:
    store: Store
    embedder: Embedder
    hoje: date
    top_k: int = 6
    edital_padrao: str | None = None
    citacoes: dict[int, Hit] = field(default_factory=dict)
    _por_trecho: dict[str, int] = field(default_factory=dict)

    def citar(self, hit: Hit) -> int:
        if hit.chunk.id not in self._por_trecho:
            n = len(self.citacoes) + 1
            self.citacoes[n] = hit
            self._por_trecho[hit.chunk.id] = n
        return self._por_trecho[hit.chunk.id]


@dataclass
class Tool:
    spec: ToolSpec
    run: Callable[[dict, AgentContext], dict]


# --- helpers ------------------------------------------------------------------


def _norm(s: str) -> str:
    return strip_accents(s.lower()).strip()


def resolver_edital(ctx: AgentContext, ref: str | None) -> Document | None:
    ref = (ref or "").strip() or ctx.edital_padrao
    if not ref:
        return None
    docs = ctx.store.list_documents()
    for d in docs:
        if d.id == ref:
            return d
    alvo = _norm(ref)
    achados = [d for d in docs if alvo in _norm(d.titulo) or alvo in d.id or _norm(d.titulo) in alvo]
    if len(achados) == 1:
        return achados[0]
    opcoes = ", ".join(f"{d.id} ({d.titulo})" for d in docs) or "nenhum"
    raise ToolError(f"Não sei qual edital é '{ref}'. Disponíveis: {opcoes}")


def _parse_data(valor: str) -> date:
    try:
        return date.fromisoformat(str(valor).strip()[:10])
    except ValueError as exc:
        raise ToolError(f"Data inválida '{valor}'. Use o formato AAAA-MM-DD.") from exc


def _trechos(ctx: AgentContext, hits: list[Hit]) -> list[dict]:
    titulos = {d.id: d.titulo for d in ctx.store.list_documents()}
    return [
        {
            "ref": ctx.citar(h),
            "edital": titulos.get(h.chunk.doc_id, h.chunk.doc_id),
            "pagina": h.chunk.pagina,
            "secao": h.chunk.secao,
            "texto": h.chunk.texto,
        }
        for h in hits
    ]


# --- ferramentas --------------------------------------------------------------


def listar_editais(args: dict, ctx: AgentContext) -> dict:
    return {
        "editais": [
            {"id": d.id, "titulo": d.titulo, "paginas": d.paginas} for d in ctx.store.list_documents()
        ]
    }


def buscar_trecho(args: dict, ctx: AgentContext) -> dict:
    pergunta = str(args.get("pergunta", "")).strip()
    if not pergunta:
        raise ToolError("Faltou a 'pergunta'.")
    doc = resolver_edital(ctx, args.get("edital_id"))
    hits = search(ctx.store, ctx.embedder, pergunta, k=ctx.top_k, doc_id=doc.id if doc else None)
    if not hits:
        return {"trechos": [], "aviso": "Nada encontrado. Tente outras palavras."}
    return {"trechos": _trechos(ctx, hits)}


def calcular_dias_ate_prova(args: dict, ctx: AgentContext) -> dict:
    prova = _parse_data(args.get("data_prova", ""))
    dias = (prova - ctx.hoje).days
    uteis = sum(1 for i in range(1, max(dias, 0) + 1) if (ctx.hoje + timedelta(days=i)).weekday() < 5)
    return {
        "hoje": ctx.hoje.isoformat(),
        "data_prova": prova.isoformat(),
        "dias": dias,
        "semanas": round(dias / 7, 1),
        "dias_uteis": uteis,
        "ja_passou": dias < 0,
    }


def _meia_hora(x: float) -> float:
    return max(0.5, round(x * 2) / 2)


def montar_cronograma(args: dict, ctx: AgentContext) -> dict:
    materias = args.get("materias") or []
    if not materias:
        raise ToolError("Passe pelo menos uma matéria em 'materias'.")
    try:
        horas = float(args.get("horas_por_semana", 0))
    except (TypeError, ValueError) as exc:
        raise ToolError("'horas_por_semana' precisa ser número.") from exc
    if not 1 <= horas <= 80:
        raise ToolError("'horas_por_semana' precisa estar entre 1 e 80.")

    itens = []
    for m in materias:
        nome = str(m.get("nome", "")).strip()
        try:
            peso = float(m.get("peso") or 1)
        except (TypeError, ValueError):
            peso = 1.0
        if nome:
            itens.append((nome, max(peso, 0.1)))
    if not itens:
        raise ToolError("Nenhuma matéria com nome.")

    if args.get("data_prova"):
        dias = (_parse_data(args["data_prova"]) - ctx.hoje).days
        if dias <= 0:
            raise ToolError("A data da prova já passou.")
        semanas = max(1, min(52, dias // 7))
        base = f"{dias} dias até a prova"
    else:
        semanas, base = 8, "sem data de prova, considerei 8 semanas"

    revisao = semanas >= 3
    semanas_estudo = semanas - 1 if revisao else semanas
    total_peso = sum(p for _, p in itens)
    por_semana = sorted(
        ({"materia": n, "peso": p, "horas_semana": _meia_hora(horas * p / total_peso)} for n, p in itens),
        key=lambda x: -x["peso"],
    )
    for linha in por_semana:
        linha["horas_total"] = round(linha["horas_semana"] * semanas_estudo, 1)
    return {
        "base": base,
        "semanas_total": semanas,
        "semanas_de_estudo": semanas_estudo,
        "ultima_semana_revisao": revisao,
        "inicio": ctx.hoje.isoformat(),
        "horas_por_semana": horas,
        "distribuicao": por_semana,
        "observacao": "Horas proporcionais ao peso (questões) de cada matéria, arredondadas em meia hora.",
    }


def comparar_editais(args: dict, ctx: AgentContext) -> dict:
    aspecto = str(args.get("aspecto", "")).strip()
    if not aspecto:
        raise ToolError("Faltou o 'aspecto' a comparar (ex: salário, requisitos, data da prova).")
    a = resolver_edital(ctx, args.get("edital_a"))
    b = resolver_edital(ctx, args.get("edital_b"))
    if not a or not b or a.id == b.id:
        raise ToolError("Preciso de dois editais diferentes em 'edital_a' e 'edital_b'.")
    k = max(2, ctx.top_k // 2)
    return {
        "aspecto": aspecto,
        "edital_a": {"titulo": a.titulo, "trechos": _trechos(ctx, search(ctx.store, ctx.embedder, aspecto, k, a.id))},
        "edital_b": {"titulo": b.titulo, "trechos": _trechos(ctx, search(ctx.store, ctx.embedder, aspecto, k, b.id))},
    }


def build_tools() -> dict[str, Tool]:
    obj = lambda props, req=(): {"type": "object", "properties": props, "required": list(req)}  # noqa: E731
    texto = lambda d: {"type": "string", "description": d}  # noqa: E731
    tools = [
        Tool(
            ToolSpec(
                "listar_editais",
                "Lista os editais carregados (id, título e número de páginas).",
                obj({}),
            ),
            listar_editais,
        ),
        Tool(
            ToolSpec(
                "buscar_trecho",
                "Busca trechos do edital que respondem a uma pergunta (busca híbrida: significado + palavras exatas). "
                "Devolve texto, página e um número 'ref' pra citar como [ref]. Use pra qualquer fato do edital.",
                obj(
                    {
                        "pergunta": texto("O que procurar, em linguagem natural ou com termos do edital."),
                        "edital_id": texto("Id do edital. Omita pra buscar em todos."),
                    },
                    ["pergunta"],
                ),
            ),
            buscar_trecho,
        ),
        Tool(
            ToolSpec(
                "calcular_dias_ate_prova",
                "Calcula quantos dias, semanas e dias úteis faltam de hoje até a data da prova.",
                obj({"data_prova": texto("Data da prova no formato AAAA-MM-DD.")}, ["data_prova"]),
            ),
            calcular_dias_ate_prova,
        ),
        Tool(
            ToolSpec(
                "montar_cronograma",
                "Monta um cronograma semanal de estudo dividindo as horas entre as matérias, proporcional ao peso "
                "(número de questões) de cada uma. A última semana vira revisão.",
                obj(
                    {
                        "materias": {
                            "type": "array",
                            "description": "Matérias da prova e peso (número de questões ou peso no edital).",
                            "items": obj(
                                {"nome": texto("Nome da matéria."), "peso": {"type": "number", "description": "Peso."}},
                                ["nome", "peso"],
                            ),
                        },
                        "horas_por_semana": {"type": "number", "description": "Horas de estudo disponíveis por semana."},
                        "data_prova": texto("Data da prova AAAA-MM-DD, se souber."),
                    },
                    ["materias", "horas_por_semana"],
                ),
            ),
            montar_cronograma,
        ),
        Tool(
            ToolSpec(
                "comparar_editais",
                "Busca o mesmo aspecto (ex: salário, escolaridade, data da prova) em dois editais e devolve os trechos "
                "lado a lado.",
                obj(
                    {
                        "edital_a": texto("Id ou nome do primeiro edital."),
                        "edital_b": texto("Id ou nome do segundo edital."),
                        "aspecto": texto("O que comparar."),
                    },
                    ["edital_a", "edital_b", "aspecto"],
                ),
            ),
            comparar_editais,
        ),
    ]
    return {t.spec.name: t for t in tools}
