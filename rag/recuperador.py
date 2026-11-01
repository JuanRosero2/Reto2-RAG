"""Paso 2 del RAG: buscar los fragmentos más parecidos a la pregunta."""
from dataclasses import dataclass

from rag import config
from rag.almacen import obtener_coleccion
from rag.modelos import embeber


@dataclass
class Fragmento:
    texto: str
    archivo: str
    pagina: int
    similitud: float  # 1 = idéntico, 0 = nada que ver


def buscar(pregunta: str, k: int = config.K_RESULTADOS,
           pagina: int | None = None) -> list[Fragmento]:
    """Devuelve los k fragmentos más similares. Si se da `pagina`, busca solo en esa página."""
    coleccion = obtener_coleccion()
    if coleccion.count() == 0:
        return []

    vector = embeber([pregunta], tipo="consulta")[0]
    filtro = {"pagina": pagina} if pagina is not None else None
    resultado = coleccion.query(
        query_embeddings=[vector],
        n_results=k,
        where=filtro,
        include=["documents", "metadatas", "distances"],
    )

    fragmentos = []
    for texto, meta, distancia in zip(resultado["documents"][0],
                                      resultado["metadatas"][0],
                                      resultado["distances"][0]):
        fragmentos.append(Fragmento(texto=texto, archivo=meta["archivo"],
                                    pagina=int(meta["pagina"]),
                                    similitud=1 - distancia))
    return fragmentos
