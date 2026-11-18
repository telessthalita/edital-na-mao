import io
import re

from pypdf import PdfReader


class PdfError(ValueError):
    pass


def clean_page_text(text: str) -> str:
    text = text.replace(" ", " ").replace("\r", "")
    # Junta palavra quebrada com hífen no fim da linha: "inscri-\nção" -> "inscrição"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _assinatura(linha: str) -> str:
    return re.sub(r"\d+", "#", linha.strip().lower())


def remove_repeated_lines(pages: list[str], min_share: float = 0.6) -> list[str]:
    """Tira cabeçalho e rodapé que se repetem em quase toda página.

    Edital costuma ter "Página 3 de 80", brasão e nome do órgão em toda folha.
    Isso polui os trechos e confunde a busca. Números viram # pra "Página 2" e
    "Página 3" contarem como a mesma linha. Limite conhecido: uma linha de
    conteúdo na borda que só muda o número de página pra página também sai.
    """
    if len(pages) < 2:
        return pages

    def bordas(linhas: list[str]) -> set[int]:
        # Só as 2 primeiras e 2 últimas linhas: é onde mora cabeçalho e rodapé.
        n = len(linhas)
        return {i for i in (0, 1, n - 2, n - 1) if 0 <= i < n}

    contagem: dict[str, int] = {}
    for page in pages:
        linhas = [l for l in page.split("\n") if l.strip()]
        for assinatura in {_assinatura(linhas[i]) for i in bordas(linhas)}:
            contagem[assinatura] = contagem.get(assinatura, 0) + 1
    limite = max(2, int(len(pages) * min_share + 0.999))
    repetidas = {a for a, n in contagem.items() if n >= limite}
    out = []
    for page in pages:
        linhas = [l for l in page.split("\n") if l.strip()]
        tirar = {i for i in bordas(linhas) if _assinatura(linhas[i]) in repetidas}
        out.append("\n".join(l for i, l in enumerate(linhas) if i not in tirar))
    return out


def extract_pages(data: bytes, max_pages: int | None = None) -> list[str]:
    """Texto de cada página, na ordem. Índice 0 = página 1."""
    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as exc:  # pypdf solta vários tipos
        raise PdfError("Não consegui abrir esse PDF.") from exc
    if reader.is_encrypted:
        raise PdfError("PDF protegido por senha.")
    total = len(reader.pages)
    if max_pages and total > max_pages:
        raise PdfError(f"PDF tem {total} páginas; o limite é {max_pages}.")
    pages = remove_repeated_lines([clean_page_text(p.extract_text() or "") for p in reader.pages])
    if sum(len(p) for p in pages) < 200:
        raise PdfError("Quase não achei texto. Parece PDF escaneado (imagem), e OCR ainda não é suportado.")
    return pages
