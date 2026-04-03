import os
from dotenv import load_dotenv

load_dotenv()

# Qdrant
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6335))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "genai_docs")

# LLM - Only need API key and model name
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.1-8b-instant"  # Just the model name

# Redis
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))

# Embedding
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Cache
SEMANTIC_SIM_THRESHOLD = 0.85
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
TOP_K = 5