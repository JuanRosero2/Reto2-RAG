"""Configuración del chatbot RAG. Todo se puede cambiar con variables de entorno."""
import os

# Servidor de Ollama (por defecto, el local)
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# Modelos: uno para conversar y otro, distinto, para los embeddings
MODELO_CHAT = os.environ.get("RAG_MODELO_CHAT", "llama3.2")
MODELO_EMBEDDINGS = os.environ.get("RAG_MODELO_EMB", "nomic-embed-text")

# Dónde se guarda la base vectorial y con qué nombre
RUTA_CHROMA = os.environ.get("RAG_RUTA_CHROMA", "datos/chroma")
COLECCION = os.environ.get("RAG_COLECCION", "pdf")

# Troceado del texto
TAMANO_CHUNK = 800      # caracteres por fragmento
SOLAPAMIENTO = 150      # caracteres que comparten dos fragmentos seguidos

# Recuperación
K_RESULTADOS = 4        # fragmentos que se le pasan al modelo
