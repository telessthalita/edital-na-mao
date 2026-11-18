"""Quebra o edital em trechos que respeitam página e seção.

Regras:
- um trecho nunca atravessa página (a citação "pág. X" tem que ser exata);
- um título de seção sempre abre trecho novo;
- trechos grandes ganham sobreposição com o anterior pra não cortar uma
  informação no meio.
"""

from __future__ import annotations

import re

from .models import Chunk

_PREFIXOS = re.compile(r"^(ANEXO|CAP[IÍ]TULO|T[IÍ]TULO|SE[CÇ][AÃ]O)\b", re.IGNORECASE)
_NUMERADO = re.compile(r"^\d{1,2}(\.\d{1,2})*\.?\s*[-–]?\s*\S")
_DAS_DOS = re.compile(r"^D[AO]S?\s+\S")


def is_heading(line: str) -> bool:
    line = line.strip()
    if not (4 <= len(line) <= 90):
        return False
    letters = [c for c in line if c.isalpha()]
    if len(letters) < 4:
        return False
    all_caps = line == line.upper()
    if _PREFIXOS.match(line) and all_caps:
        return True
    if _NUMERADO.match(line) and all_caps and len(line.split()) <= 10:
        return True
    return bool(_DAS_DOS.match(line) and all_caps and len(line.split()) <= 8)


def _tail(text: str, size: int) -> str:
    if len(text) <= size:
        return text
    cut = text[-size:]
    space = cut.find(" ")
    return cut[space + 1 :] if space != -1 else cut


def _split_long(line: str, max_chars: int) -> list[str]:
    if len(line) <= max_chars:
        return [line]
    parts, cur = [], ""
    for sentence in re.split(r"(?<=[.;:])\s+", line):
        while len(sentence) > max_chars:  # frase gigante: corta por palavra
            idx = sentence.rfind(" ", 0, max_chars)
            if idx <= 0:
                idx = max_chars
            parts.append((cur + " " + sentence[:idx]).strip())
            cur, sentence = "", sentence[idx:].strip()
        if len(cur) + len(sentence) + 1 > max_chars and cur:
            parts.append(cur.strip())
            cur = ""
        cur = f"{cur} {sentence}".strip()
    if cur:
        parts.append(cur)
    return parts


def chunk_pages(doc_id: str, pages: list[str], max_chars: int = 900, overlap: int = 150) -> list[Chunk]:
    chunks: list[Chunk] = []
    section: str | None = None

    def flush(buf: list[str], page_no: int) -> str:
        text = "\n".join(buf).strip()
        if text:
            chunks.append(Chunk(doc_id=doc_id, ordem=len(chunks), pagina=page_no, secao=section, texto=text))
        return text

    for page_idx, page in enumerate(pages):
        page_no = page_idx + 1
        buf: list[str] = []
        size = 0
        for raw_line in page.split("\n"):
            line = raw_line.strip()
            if not line:
                continue
            if is_heading(line):
                flush(buf, page_no)
                section = line[:90]
                buf, size = [line], len(line)
                continue
            for piece in _split_long(line, max_chars):
                if size + len(piece) + 1 > max_chars and buf:
                    previous = flush(buf, page_no)
                    tail = _tail(previous, overlap)
                    buf, size = ([tail], len(tail)) if tail else ([], 0)
                buf.append(piece)
                size += len(piece) + 1
        flush(buf, page_no)
    return chunks
