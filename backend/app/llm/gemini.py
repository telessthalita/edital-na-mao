"""Gemini via REST (generativelanguage.googleapis.com).

REST direto em vez do SDK: o protocolo de tool calling fica visível aqui e o
projeto não quebra quando o SDK muda de versão.
"""

from __future__ import annotations

import time

import httpx
import numpy as np

from .base import LLMError, Message, ModelTurn, ToolCall, ToolSpec

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


def _to_gemini_schema(schema: dict) -> dict:
    """JSON Schema -> subconjunto OpenAPI que o Gemini aceita em `parameters`."""
    out: dict = {}
    for key, value in schema.items():
        if key == "type":
            out["type"] = str(value).upper()
        elif key == "properties":
            out["properties"] = {k: _to_gemini_schema(v) for k, v in value.items()}
        elif key == "items":
            out["items"] = _to_gemini_schema(value)
        elif key in ("description", "required", "enum", "format", "nullable"):
            out[key] = value
    return out


# Assinatura "coringa" documentada pelo Google pra quando o histórico veio de
# outro modelo (ex: depois de um fallback) e não tem thought signature válida.
SKIP_SIGNATURE = "skip_thought_signature_validator"


def to_gemini_contents(messages: list[Message], model: str | None = None) -> list[dict]:
    contents: list[dict] = []
    for msg in messages:
        if msg.role == "user":
            contents.append({"role": "user", "parts": [{"text": msg.text or ""}]})
        elif msg.role == "assistant":
            raw = msg.raw if isinstance(msg.raw, dict) else None
            if raw and raw.get("model") == model:
                contents.append(raw["content"])  # thought signatures voltam intactas
                continue
            parts: list[dict] = []
            if msg.text:
                parts.append({"text": msg.text})
            for call in msg.tool_calls:
                parts.append({"functionCall": {"name": call.name, "args": call.args}, "thoughtSignature": SKIP_SIGNATURE})
            contents.append({"role": "model", "parts": parts or [{"text": ""}]})
        elif msg.role == "tool":
            parts = []
            for res in msg.tool_results:
                fr: dict = {"name": res.name, "response": res.content}
                if res.id:
                    fr["id"] = res.id
                parts.append({"functionResponse": fr})
            contents.append({"role": "user", "parts": parts})
    return contents


def parse_turn(data: dict, model: str | None = None) -> ModelTurn:
    candidates = data.get("candidates") or []
    if not candidates:
        motivo = (data.get("promptFeedback") or {}).get("blockReason", "sem candidatos")
        raise LLMError(f"Gemini não devolveu resposta ({motivo})")
    content = candidates[0].get("content") or {"role": "model", "parts": []}
    texts: list[str] = []
    calls: list[ToolCall] = []
    for part in content.get("parts", []):
        if part.get("thought"):
            continue
        if "functionCall" in part:
            fc = part["functionCall"]
            calls.append(ToolCall(name=fc["name"], args=fc.get("args") or {}, id=fc.get("id")))
        elif "text" in part:
            texts.append(part["text"])
    text = "".join(texts).strip() or None
    return ModelTurn(text=text, tool_calls=calls, raw={"model": model, "content": content})


class _GeminiHTTP:
    def __init__(self, api_key: str, client: httpx.Client | None = None, retries: int = 2):
        if not api_key:
            raise ValueError("GEMINI_API_KEY vazia")
        self._client = client or httpx.Client(timeout=60)
        self._headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
        self._retries = retries

    def post(self, path: str, body: dict) -> dict:
        for attempt in range(self._retries + 1):
            try:
                res = self._client.post(f"{BASE_URL}/{path}", json=body, headers=self._headers)
            except httpx.HTTPError as exc:
                if attempt < self._retries:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise LLMError(f"Falha de rede falando com o Gemini: {exc}", transient=True) from exc
            if res.status_code in (429, 500, 503) and attempt < self._retries:
                time.sleep(2.0 * (attempt + 1))
                continue
            if res.status_code == 429:
                raise LLMError("Cota do Gemini estourou. Tenta de novo daqui a pouco.", 429, transient=True)
            if res.status_code in (500, 503):
                raise LLMError(f"Gemini sobrecarregado ({res.status_code}). Tenta de novo daqui a pouco.", 503, transient=True)
            if res.status_code >= 400:
                raise LLMError(f"Gemini respondeu {res.status_code}: {res.text[:300]}")
            return res.json()
        raise LLMError("Gemini indisponível")  # pragma: no cover


class GeminiChatModel:
    def __init__(
        self, api_key: str, model: str = "gemini-3.8-flash", client: httpx.Client | None = None, retries: int = 2
    ):
        self.name = model
        self._http = _GeminiHTTP(api_key, client, retries)

    def generate(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> ModelTurn:
        body: dict = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": to_gemini_contents(messages, self.name),
        }
        if tools:
            body["tools"] = [
                {
                    "functionDeclarations": [
                        {
                            "name": t.name,
                            "description": t.description,
                            "parameters": _to_gemini_schema(t.parameters),
                        }
                        for t in tools
                    ]
                }
            ]
        data = self._http.post(f"models/{self.name}:generateContent", body)
        return parse_turn(data, self.name)


class GeminiEmbedder:
    BATCH = 100

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-embedding-001",
        dim: int = 768,
        client: httpx.Client | None = None,
    ):
        self.name = model
        self.dim = dim
        self._http = _GeminiHTTP(api_key, client)

    def embed(self, texts: list[str], task: str) -> np.ndarray:
        task_type = "RETRIEVAL_DOCUMENT" if task == "documento" else "RETRIEVAL_QUERY"
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.BATCH):
            batch = texts[i : i + self.BATCH]
            body = {
                "requests": [
                    {
                        "model": f"models/{self.name}",
                        "content": {"parts": [{"text": t}]},
                        "taskType": task_type,
                        "outputDimensionality": self.dim,
                    }
                    for t in batch
                ]
            }
            data = self._http.post(f"models/{self.name}:batchEmbedContents", body)
            vectors.extend(e["values"] for e in data["embeddings"])
        arr = np.asarray(vectors, dtype=np.float32).reshape(len(texts), self.dim)
        # Com dimensão reduzida (<3072) o Google recomenda normalizar.
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms
