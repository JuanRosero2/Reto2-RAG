# Chatbot RAG sobre PDF (Ollama + ChromaDB)

Chatbot de consola que responde preguntas sobre un PDF usando RAG, todo en local.

## Cómo funciona

1. **Ingesta** (`rag/ingesta.py`): lee el PDF página por página, lo divide en fragmentos
   de ~800 caracteres (con 150 de solapamiento), calcula sus embeddings con
   `nomic-embed-text` y los guarda en ChromaDB (`datos/chroma`) con el número de página.
2. **Recuperación** (`rag/recuperador.py`): convierte la pregunta en vector y busca los
   4 fragmentos más parecidos (distancia coseno).
3. **Generación** (`rag/chatbot.py`): le pasa esos fragmentos a `llama3.2` con la
   instrucción de responder solo con esa información, y muestra las páginas usadas.

```
rag/
├── config.py       # modelos, rutas, tamaño de fragmento, k
├── modelos.py      # conexión con Ollama (embeddings y chat)
├── almacen.py      # colección de ChromaDB
├── ingesta.py      # PDF → fragmentos → ChromaDB
├── recuperador.py  # búsqueda semántica
└── chatbot.py      # bucle de chat con RAG
```

## Requisitos

- Ollama instalado y abierto, con los modelos:
  ```bash
  ollama pull llama3.2
  ollama pull nomic-embed-text
  ```

## Uso

```bash
uv sync                                                  # instala dependencias (solo la primera vez)
uv run python -m rag.ingesta datos/tutorial_memoria_rag.pdf
uv run python -m rag.chatbot
```

Dentro del chat:

| Comando     | Qué hace                               |
|-------------|----------------------------------------|
| `/pagina 5` | Solo busca en la página 5              |
| `/todo`     | Vuelve a buscar en todo el documento   |
| `salir`     | Termina                                |

Para usar otro PDF, cópielo a `datos/` y ejecute la ingesta con su ruta. Si vuelve a cargar
el mismo PDF, se reemplaza (no se duplica). Para empezar de cero, borre la carpeta `datos/chroma`.

## Cambiar de modelo

Sin tocar el código, con variables de entorno:

```bash
RAG_MODELO_CHAT=qwen2.5:7b uv run python -m rag.chatbot
```

Si cambia `RAG_MODELO_EMB`, borre `datos/chroma` y vuelva a hacer la ingesta: los vectores
de modelos distintos no son compatibles.
