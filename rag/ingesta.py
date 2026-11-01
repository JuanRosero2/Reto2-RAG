"""Paso 1 del RAG: leer un PDF, trocearlo y guardarlo en ChromaDB.

Uso:
    uv run python -m rag.ingesta datos/documento.pdf
"""
import argparse
import hashlib
import re
from pathlib import Path

from pypdf import PdfReader

from rag import config
from rag.almacen import obtener_coleccion
from rag.modelos import embeber, verificar_ollama


def leer_paginas(ruta: Path) -> list[tuple[int, str]]:
    """Devuelve [(numero_de_pagina, texto), ...] con números que empiezan en 1."""
    lector = PdfReader(ruta)
    paginas = []
    for numero, pagina in enumerate(lector.pages, start=1):
        texto = pagina.extract_text() or ""
        texto = re.sub(r"[ \t]+", " ", texto)          # espacios repetidos
        texto = re.sub(r"\n{3,}", "\n\n", texto).strip()  # saltos de línea de sobra
        if texto:
            paginas.append((numero, texto))
    return paginas


def trocear(texto: str, tamano: int = config.TAMANO_CHUNK,
            solapamiento: int = config.SOLAPAMIENTO) -> list[str]:
    """Divide el texto en fragmentos de ~`tamano` caracteres que se solapan.

    Intenta cortar en un final de párrafo, de frase o en un espacio
    para no partir palabras por la mitad.
    """
    fragmentos = []
    inicio = 0
    while inicio < len(texto):
        fin = min(inicio + tamano, len(texto))
        if fin < len(texto):
            ventana = texto[inicio:fin]
            for separador in ("\n\n", ". ", "\n", " "):
                corte = ventana.rfind(separador)
                if corte > tamano // 2:
                    fin = inicio + corte + len(separador)
                    break
        fragmento = texto[inicio:fin].strip()
        if fragmento:
            fragmentos.append(fragmento)
        if fin >= len(texto):
            break
        inicio = max(fin - solapamiento, inicio + 1)
    return fragmentos


def huella(ruta: Path) -> str:
    """Identificador corto basado en el contenido del archivo."""
    return hashlib.sha1(ruta.read_bytes()).hexdigest()[:10]


def ingerir(ruta: Path) -> int:
    """Carga un PDF en ChromaDB. Si ya estaba cargado, lo reemplaza. Devuelve cuántos fragmentos guardó."""
    coleccion = obtener_coleccion()
    doc_id = huella(ruta)

    ids, textos, metadatos = [], [], []
    for pagina, texto in leer_paginas(ruta):
        for i, fragmento in enumerate(trocear(texto)):
            ids.append(f"{doc_id}-p{pagina}-c{i}")
            textos.append(fragmento)
            metadatos.append({"archivo": ruta.name, "doc_id": doc_id,
                              "pagina": pagina, "chunk": i})

    if not textos:
        print(f"⚠ {ruta.name}: no se encontró texto (¿es un PDF escaneado?).")
        return 0

    # Evita duplicados: borra lo que hubiera de este mismo PDF antes de volver a guardarlo
    coleccion.delete(where={"doc_id": doc_id})

    print(f"Calculando embeddings de {len(textos)} fragmentos con {config.MODELO_EMBEDDINGS}...")
    vectores = embeber(textos, tipo="documento")
    coleccion.add(ids=ids, documents=textos, embeddings=vectores, metadatas=metadatos)
    return len(textos)


def main():
    parser = argparse.ArgumentParser(description="Carga uno o varios PDF en ChromaDB.")
    parser.add_argument("pdfs", nargs="+", type=Path, help="Rutas de los PDF")
    args = parser.parse_args()

    verificar_ollama()
    for ruta in args.pdfs:
        if not ruta.is_file():
            print(f"✗ No existe: {ruta}")
            continue
        n = ingerir(ruta)
        if n:
            print(f"✓ {ruta.name}: {n} fragmentos guardados en {config.RUTA_CHROMA}")
    print(f"Total en la colección '{config.COLECCION}': {obtener_coleccion().count()} fragmentos")


if __name__ == "__main__":
    main()
