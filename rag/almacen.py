"""Acceso a la base vectorial ChromaDB."""
import chromadb

from rag import config


def obtener_coleccion():
    """Abre (o crea) la colección persistente que guarda los fragmentos del PDF.

    No se le pasa función de embeddings a Chroma: los vectores se calculan
    con Ollama (rag.modelos) y se entregan ya hechos al guardar y al consultar.
    """
    cliente = chromadb.PersistentClient(path=config.RUTA_CHROMA)
    return cliente.get_or_create_collection(
        name=config.COLECCION,
        embedding_function=None,
        metadata={"hnsw:space": "cosine"},
    )
