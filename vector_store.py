from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from config import QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION, EMBEDDING_DIM, TOP_K

print(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# Ensure collection exists
try:
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]
    
    if QDRANT_COLLECTION not in collection_names:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE
            )
        )
        print(f"Created collection: {QDRANT_COLLECTION}")
    else:
        print(f"Collection already exists: {QDRANT_COLLECTION}")
except Exception as e:
    print(f"Qdrant error: {e}")

def store_chunks(chunks, embeddings):
    points = [
        PointStruct(id=i, vector=embeddings[i], payload={"text": chunks[i]})
        for i in range(len(chunks))
    ]
    client.upsert(collection_name=QDRANT_COLLECTION, points=points)
    print(f"Stored {len(chunks)} chunks")

def search_chunks(embedding, top_k=TOP_K):
    # Try different methods based on qdrant client version
    try:
        # Method 1: using search (older versions)
        results = client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=embedding,
            limit=top_k
        )
        return [hit.payload["text"] for hit in results]
    except AttributeError:
        # Method 2: using query_points (newer versions)
        results = client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=embedding,
            limit=top_k
        )
        return [hit.payload["text"] for hit in results.points]