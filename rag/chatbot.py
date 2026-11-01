"""Paso 3 del RAG: chatbot de consola que responde con base en el PDF.

Uso:
    uv run python -m rag.chatbot

Comandos dentro del chat:
    /pagina 5   → solo busca en la página 5
    /todo       → vuelve a buscar en todo el documento
    salir       → termina
"""
from rag import config
from rag.almacen import obtener_coleccion
from rag.modelos import conversar, verificar_ollama
from rag.recuperador import Fragmento, buscar

SISTEMA = """Usted es un asistente que responde preguntas sobre un documento PDF.
Reglas:
- Responda en español, de forma clara y concisa.
- Use ÚNICAMENTE la información de los fragmentos del documento que se le entregan.
- Si la respuesta no está en esos fragmentos, diga exactamente:
  "No encontré esa información en el documento."
- No invente datos. Cuando sea útil, mencione la página, por ejemplo: (p. 4)."""

TURNOS_HISTORIAL = 3  # pares pregunta/respuesta que se recuerdan para preguntas de seguimiento


def armar_contexto(fragmentos: list[Fragmento]) -> str:
    bloques = [f"[{f.archivo}, página {f.pagina}]\n{f.texto}" for f in fragmentos]
    return "\n\n---\n\n".join(bloques)


def armar_mensajes(pregunta: str, fragmentos: list[Fragmento],
                   historial: list[dict]) -> list[dict]:
    contenido = (
        "Fragmentos del documento:\n\n"
        f"{armar_contexto(fragmentos)}\n\n"
        f"Pregunta: {pregunta}"
    )
    return [{"role": "system", "content": SISTEMA},
            *historial,
            {"role": "user", "content": contenido}]


def fuentes(fragmentos: list[Fragmento]) -> str:
    por_archivo: dict[str, set[int]] = {}
    for f in fragmentos:
        por_archivo.setdefault(f.archivo, set()).add(f.pagina)
    return "; ".join(f"{archivo} p. {', '.join(map(str, sorted(pags)))}"
                     for archivo, pags in por_archivo.items())


def main():
    verificar_ollama()
    total = obtener_coleccion().count()
    if total == 0:
        print("La base está vacía. Primero cargue un PDF:\n"
              "  uv run python -m rag.ingesta datos/su_documento.pdf")
        return

    print(f"Chatbot RAG · modelo: {config.MODELO_CHAT} · {total} fragmentos cargados")
    print("Comandos: /pagina N, /todo, salir\n")

    historial: list[dict] = []
    pagina: int | None = None

    while True:
        try:
            pregunta = input("Usted: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not pregunta:
            continue
        if pregunta.lower() in {"salir", "exit"}:
            break
        if pregunta.lower().startswith("/pagina"):
            partes = pregunta.split()
            if len(partes) == 2 and partes[1].isdigit():
                pagina = int(partes[1])
                print(f"→ Ahora solo busco en la página {pagina}.\n")
            else:
                print("→ Uso: /pagina 5\n")
            continue
        if pregunta.lower() == "/todo":
            pagina = None
            print("→ Busco en todo el documento.\n")
            continue

        fragmentos = buscar(pregunta, pagina=pagina)
        if not fragmentos:
            print("Asistente: No encontré esa información en el documento.\n")
            continue

        print("Asistente: ", end="", flush=True)
        respuesta = ""
        for parte in conversar(armar_mensajes(pregunta, fragmentos, historial)):
            print(parte, end="", flush=True)
            respuesta += parte
        print(f"\n   📄 Fuentes: {fuentes(fragmentos)}\n")

        # En el historial se guarda la pregunta sin los fragmentos para no inflar el prompt
        historial += [{"role": "user", "content": pregunta},
                      {"role": "assistant", "content": respuesta}]
        historial = historial[-2 * TURNOS_HISTORIAL:]

    print("Hasta pronto.")


if __name__ == "__main__":
    main()
