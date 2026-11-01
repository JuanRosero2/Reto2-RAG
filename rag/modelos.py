"""Conexión con Ollama: embeddings y chat."""
import sys
from collections.abc import Iterator

import ollama

from rag import config

_cliente = ollama.Client(host=config.OLLAMA_HOST)

# nomic-embed-text rinde mejor si se marca qué es documento y qué es pregunta
_PREFIJOS = {
    "documento": "search_document: ",
    "consulta": "search_query: ",
}


def verificar_ollama() -> None:
    """Comprueba que Ollama esté encendido y que los dos modelos estén descargados."""
    try:
        instalados = {m.model for m in _cliente.list().models}
    except Exception:
        sys.exit(
            f"No pude conectarme a Ollama en {config.OLLAMA_HOST}.\n"
            "Ábralo (o ejecute `ollama serve`) y vuelva a intentarlo."
        )
    faltan = [
        m for m in (config.MODELO_CHAT, config.MODELO_EMBEDDINGS)
        if m not in instalados and f"{m}:latest" not in instalados
    ]
    if faltan:
        sys.exit("Faltan modelos en Ollama. Ejecute:\n"
                 + "\n".join(f"  ollama pull {m}" for m in faltan))


def _prefijar(textos: list[str], tipo: str) -> list[str]:
    if "nomic" not in config.MODELO_EMBEDDINGS:
        return textos
    return [_PREFIJOS[tipo] + t for t in textos]


def embeber(textos: list[str], tipo: str = "documento", lote: int = 32) -> list[list[float]]:
    """Convierte una lista de textos en vectores con el modelo de embeddings."""
    vectores: list[list[float]] = []
    textos = _prefijar(textos, tipo)
    for i in range(0, len(textos), lote):
        respuesta = _cliente.embed(model=config.MODELO_EMBEDDINGS, input=textos[i:i + lote])
        vectores.extend(respuesta.embeddings)
    return vectores


def conversar(mensajes: list[dict]) -> Iterator[str]:
    """Envía los mensajes al modelo de chat y devuelve la respuesta por partes (streaming)."""
    for parte in _cliente.chat(model=config.MODELO_CHAT, messages=mensajes, stream=True):
        yield parte.message.content or ""
