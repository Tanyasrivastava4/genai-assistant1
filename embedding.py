from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL_NAME

model = SentenceTransformer(EMBEDDING_MODEL_NAME)

def generate_embedding(text):
    """Single embedding (for queries)"""
    return model.encode(text).tolist()

def generate_embeddings_batch(texts):
    """Batch embedding for documents (FASTER)"""
    return model.encode(texts).tolist()