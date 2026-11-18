import re
import unicodedata

STOPWORDS = set(
    """
a ao aos as até com como da das de dela dele do dos e é ela ele em entre essa esse esta este eu
foi há isso já mais mas me meu minha na nas não nem no nos nós o os ou para pela pelas pelo pelos
por qual quais quando que quem se sem ser seu seus sua suas são também te tem um uma umas uns vai
você vocês vou the
""".split()
)

_TOKEN = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def tokenize(text: str) -> list[str]:
    """Minúsculas, sem acento, sem stopword. Mantém números como 85,00."""
    norm = strip_accents(text.lower())
    stop = {strip_accents(s) for s in STOPWORDS}
    return [t for t in _TOKEN.findall(norm) if t not in stop and len(t) > 1]


def slugify(text: str, max_len: int = 40) -> str:
    norm = strip_accents(text.lower())
    slug = re.sub(r"[^a-z0-9]+", "-", norm).strip("-")
    return slug[:max_len].strip("-") or "edital"
