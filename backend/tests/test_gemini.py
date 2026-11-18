import json

import httpx
import numpy as np

from app.llm.base import Message, ToolCall, ToolResult, ToolSpec
from app.llm.gemini import GeminiChatModel, GeminiEmbedder, _to_gemini_schema, to_gemini_contents


def test_schema_vira_tipos_maiusculos():
    schema = {
        "type": "object",
        "properties": {"m": {"type": "array", "items": {"type": "object", "properties": {"p": {"type": "number"}}}}},
        "required": ["m"],
        "additionalProperties": False,
    }
    out = _to_gemini_schema(schema)
    assert out["type"] == "OBJECT"
    assert out["properties"]["m"]["items"]["properties"]["p"]["type"] == "NUMBER"
    assert "additionalProperties" not in out


def test_conteudo_preserva_raw_e_function_response():
    raw = {"role": "model", "parts": [{"functionCall": {"name": "x", "args": {}}, "thoughtSignature": "abc"}]}
    contents = to_gemini_contents(
        [
            Message("user", "oi"),
            Message("assistant", tool_calls=[ToolCall("x", {})], raw=raw),
            Message("tool", tool_results=[ToolResult("x", {"ok": 1})]),
        ]
    )
    assert contents[1] is raw  # assinatura de pensamento volta intacta
    assert contents[2] == {"role": "user", "parts": [{"functionResponse": {"name": "x", "response": {"ok": 1}}}]}


def test_generate_parseia_function_call():
    pedidos = []

    def handler(req: httpx.Request) -> httpx.Response:
        pedidos.append(json.loads(req.content))
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [
                                {"text": "pensando", "thought": True},
                                {"functionCall": {"name": "buscar_trecho", "args": {"pergunta": "taxa"}}},
                            ],
                        }
                    }
                ]
            },
        )

    model = GeminiChatModel("k", client=httpx.Client(transport=httpx.MockTransport(handler)))
    turn = model.generate("sys", [Message("user", "taxa?")], [ToolSpec("buscar_trecho", "d", {"type": "object", "properties": {}})])
    assert turn.text is None
    assert turn.tool_calls[0].name == "buscar_trecho" and turn.tool_calls[0].args == {"pergunta": "taxa"}
    body = pedidos[0]
    assert body["systemInstruction"]["parts"][0]["text"] == "sys"
    assert body["tools"][0]["functionDeclarations"][0]["parameters"]["type"] == "OBJECT"


def test_embedder_em_lotes_e_normaliza():
    tamanhos = []

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        tamanhos.append(len(body["requests"]))
        assert body["requests"][0]["taskType"] == "RETRIEVAL_QUERY"
        return httpx.Response(200, json={"embeddings": [{"values": [3.0, 4.0]} for _ in body["requests"]]})

    emb = GeminiEmbedder("k", dim=2, client=httpx.Client(transport=httpx.MockTransport(handler)))
    out = emb.embed(["t"] * 150, "consulta")
    assert tamanhos == [100, 50]
    assert out.shape == (150, 2) and np.allclose(out[0], [0.6, 0.8])
