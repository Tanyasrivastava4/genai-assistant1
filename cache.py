import redis
import json
import uuid
import numpy as np
import hashlib
from sklearn.metrics.pairwise import cosine_similarity
from config import REDIS_HOST, REDIS_PORT, REDIS_DB, SEMANTIC_SIM_THRESHOLD

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)


# LEVEL 1: Exact Query Cache (L1) 

def get_exact(query):
    """Check if exact query exists in cache"""
    key = f"exact:{query.lower().strip()}"
    result = r.get(key)
    if result:
        data = json.loads(result)
        return data.get("answer"), data.get("chunks")
    return None, None

def set_exact(query, answer, chunks):
    """Store exact query response - NO EXPIRATION"""
    key = f"exact:{query.lower().strip()}"
    r.set(key, json.dumps({"answer": answer, "chunks": chunks}))  # Removed setex


# LEVEL 2: Semantic Cache (L2) 

def get_semantic(query_embedding):
    """Find semantically similar query in cache using SCAN"""
    query_vec = np.array(query_embedding).reshape(1, -1)
    
    for key in r.scan_iter("semantic:*"):
        data = json.loads(r.get(key))
        stored_embedding = np.array(data["embedding"]).reshape(1, -1)
        similarity = cosine_similarity(query_vec, stored_embedding)[0][0]
        
        if similarity >= SEMANTIC_SIM_THRESHOLD:
            return data["answer"], data["chunks"]
    
    return None, None

def get_semantic_optimized(query_embedding, max_checks=100):
    """Optimized semantic cache with limit on number of checks"""
    query_vec = np.array(query_embedding).reshape(1, -1)
    checked = 0
    
    for key in r.scan_iter("semantic:*"):
        if checked >= max_checks:
            break
        
        data = json.loads(r.get(key))
        stored_embedding = np.array(data["embedding"]).reshape(1, -1)
        similarity = cosine_similarity(query_vec, stored_embedding)[0][0]
        
        if similarity >= SEMANTIC_SIM_THRESHOLD:
            return data["answer"], data["chunks"]
        
        checked += 1
    
    return None, None

def set_semantic(query_embedding, answer, chunks):
    """Store query embedding and response - NO EXPIRATION"""
    key = f"semantic:{uuid.uuid4()}"
    r.set(key, json.dumps({  # Removed setex
        "embedding": query_embedding, 
        "answer": answer, 
        "chunks": chunks
    }))


# LEVEL 3: Retrieval Cache with Answer (L3) 

def get_retrieval(query_embedding):
    """Check if similar query has already retrieved same chunks AND generated answer"""
    query_vec = np.array(query_embedding).reshape(1, -1)
    
    for key in r.scan_iter("retrieval:*"):
        data = json.loads(r.get(key))
        stored_embeddings = data.get("embeddings", [])
        for stored_emb in stored_embeddings:
            stored_embedding = np.array(stored_emb).reshape(1, -1)
            similarity = cosine_similarity(query_vec, stored_embedding)[0][0]
            if similarity >= SEMANTIC_SIM_THRESHOLD:
                return data["chunks"], data["answer"]
    
    return None, None

def set_retrieval(query_embedding, chunks, answer):
    """Store retrieved chunks AND generated answer with chunk hashing - NO EXPIRATION"""
    chunks_str = "".join(chunks)
    chunk_hash = hashlib.md5(chunks_str.encode()).hexdigest()
    key = f"retrieval:{chunk_hash}"
    
    existing = r.get(key)
    if existing:
        existing_data = json.loads(existing)
        existing_embeddings = existing_data.get("embeddings", [])
        existing_embeddings.append(query_embedding)
        r.set(key, json.dumps({  # Removed setex
            "embeddings": existing_embeddings,
            "chunks": chunks,
            "answer": answer
        }))
    else:
        r.set(key, json.dumps({  # Removed setex
            "embeddings": [query_embedding],
            "chunks": chunks,
            "answer": answer
        }))


# Cache Management

def clear_all_cache():
    """Clear all cache levels"""
    count = 0
    
    for key in r.scan_iter("exact:*"):
        r.delete(key)
        count += 1
    
    for key in r.scan_iter("semantic:*"):
        r.delete(key)
        count += 1
    
    for key in r.scan_iter("retrieval:*"):
        r.delete(key)
        count += 1
    
    return count

def get_cache_stats():
    """Get cache statistics"""
    stats = {"exact_count": 0, "semantic_count": 0, "retrieval_count": 0}
    
    for _ in r.scan_iter("exact:*"):
        stats["exact_count"] += 1
    
    for _ in r.scan_iter("semantic:*"):
        stats["semantic_count"] += 1
    
    for _ in r.scan_iter("retrieval:*"):
        stats["retrieval_count"] += 1
    
    return stats