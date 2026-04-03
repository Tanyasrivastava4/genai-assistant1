from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
import tempfile
import time

from ingestion import process_document
from embedding import generate_embedding
from vector_store import search_chunks
from cache import (
    get_exact, set_exact,
    get_semantic_optimized, set_semantic,
    get_retrieval, set_retrieval,
    clear_all_cache, get_cache_stats
)
from llm_client import generate_answer

app = FastAPI(title="GenAI Knowledge Assistant")

class QueryRequest(BaseModel):
    query: str
    use_llm: bool = True

class QueryResponse(BaseModel):
    answer: str
    retrieved_chunks: List[str]
    cache_hit: Optional[str] = None
    latency_ms: float
    source: str



@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    allowed = [".pdf", ".txt", ".md"]
    
    print(f"Received file: {file.filename}")
    print(f"Content-Type: {file.content_type}")
    
    # Try to get extension from filename
    ext = os.path.splitext(file.filename)[1].lower()
    
    # If no extension, detect from content_type or file header
    if not ext:
        content_type = file.content_type
        if content_type == "application/pdf":
            ext = ".pdf"
        elif content_type == "text/plain":
            ext = ".txt"
        elif content_type == "text/markdown":
            ext = ".md"
        else:
            # Read first 4 bytes to check for PDF signature
            header = await file.read(4)
            await file.seek(0)
            if header == b"%PDF":
                ext = ".pdf"
    
    print(f"Extension detected: '{ext}'")
    
    if ext not in allowed:
        raise HTTPException(
            400, 
            f"Unsupported file type: '{ext}'. Allowed: {allowed}. Filename: '{file.filename}'"
        )
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        result = process_document(tmp_path)
        return {"message": "Document ingested", "chunks_stored": result.get("chunks_stored", 0)}
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(500, str(e))
    finally:
        os.unlink(tmp_path)


# Query Endpoint with 3-Level Cache

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    start_time = time.time()
    query_text = request.query.strip()
    
    if not query_text:
        raise HTTPException(400, "Query cannot be empty")
    
    query_embedding = generate_embedding(query_text)
    
    
    # LEVEL 1: Exact Query Cache
    
    answer, chunks = get_exact(query_text)
    if answer:
        latency_ms = (time.time() - start_time) * 1000
        return QueryResponse(
            answer=answer,
            retrieved_chunks=chunks,
            cache_hit="exact",
            latency_ms=round(latency_ms, 2),
            source="cache"
        )
    
    
    # LEVEL 2: Semantic Cache (Optimized with max_checks=100)
    
    answer, chunks = get_semantic_optimized(query_embedding, max_checks=100)
    if answer:
        latency_ms = (time.time() - start_time) * 1000
        return QueryResponse(
            answer=answer,
            retrieved_chunks=chunks,
            cache_hit="semantic",
            latency_ms=round(latency_ms, 2),
            source="cache"
        )
    
    
    # LEVEL 3: Retrieval Cache (Fixed edge case)
    
    chunks, cached_answer = get_retrieval(query_embedding)
    
    if chunks:
        if cached_answer:
            # Perfect L3 hit: chunks AND answer exist
            latency_ms = (time.time() - start_time) * 1000
            return QueryResponse(
                answer=cached_answer,
                retrieved_chunks=chunks,
                cache_hit="retrieval",
                latency_ms=round(latency_ms, 2),
                source="cache"
            )
        else:
            # Chunks exist but no answer - still need LLM
            retrieved_chunks = chunks
    else:
        # No chunks at all - must search Qdrant
        retrieved_chunks = search_chunks(query_embedding)
    
    # Handle case when no chunks found
    if not retrieved_chunks:
        latency_ms = (time.time() - start_time) * 1000
        return QueryResponse(
            answer="No relevant information found in the knowledge base.",
            retrieved_chunks=[],
            cache_hit=None,
            latency_ms=round(latency_ms, 2),
            source="retrieval_only"
        )
    
    # Generate answer using LLM
    if request.use_llm:
        answer = generate_answer(query_text, retrieved_chunks)
        source = "llm"
    else:
        answer = "\n\n---\n\n".join(retrieved_chunks)
        source = "retrieval_only"
    
    # Store in all cache levels
    set_exact(query_text, answer, retrieved_chunks)
    set_semantic(query_embedding, answer, retrieved_chunks)
    set_retrieval(query_embedding, retrieved_chunks, answer)
    
    latency_ms = (time.time() - start_time) * 1000
    
    return QueryResponse(
        answer=answer,
        retrieved_chunks=retrieved_chunks,
        cache_hit=None,
        latency_ms=round(latency_ms, 2),
        source=source
    )



# Utility Endpoints

@app.delete("/cache")
async def clear_cache():
    count = clear_all_cache()
    return {"message": f"Cleared {count} cache entries"}

@app.get("/cache/stats")
async def cache_stats():
    return get_cache_stats()

@app.get("/health")
async def health():
    return {"status": "healthy", "qdrant": "connected"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)