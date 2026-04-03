🚀 GenAI Knowledge Assistant (Low-Latency, Cache-Optimized)
📌 Overview

This project implements a GenAI-powered knowledge assistant that answers user queries using a given set of documents. The system is designed with a strong focus on:

⚡ Low latency
 Multi-level caching
 Scalability (multi-user readiness)
 Clean modular architecture

The assistant uses a Retrieval-Augmented Generation (RAG) pipeline with Qdrant (vector DB), Redis (cache), and Groq LLaMA 3.1-8B for response generation.




- Architecture Overview

Tech Stack
API Layer	               FastAPI (uvicorn) — /ingest and /query endpoints
Vector Database	Qdrant     stores chunk embeddings for semantic retrieval
Cache Layer	Redis          3-level caching (exact, semantic, retrieval)
Embedding Model	           all-MiniLM-L6-v2 via SentenceTransformers (384-dim)
LLM	Groq API               llama-3.1-8b-instant (fast, low-latency inference)
UI (Bonus)	Streamlit      document upload, query input, results display
Document Support	       .pdf (PyPDF2), .txt, .md


🔹 High-Level Flow
User Query
   ↓
Embedding Generation
   ↓
L1 Cache (Exact Match)
   ↓
L2 Cache (Semantic Match)
   ↓
L3 Cache (Retrieval + Answer)
   ↓
Qdrant Vector Search (if cache miss)
   ↓
LLM (Groq - LLaMA 3.1-8B)
   ↓
Store in Cache
   ↓
Return Response
Store result in all 3 cache levels for future queries

Components -

1. Document Ingestion
Accepts .pdf, .txt, .md
Performs:
Chunking (with overlap)
Embedding generation (batch)
Storage in Qdrant

2. Query Pipeline
Input: User query
Steps:
Generate embedding
Check multi-level cache
Retrieve relevant chunks (if needed)
Generate response using LLM (optional)

3. Technologies Used
Component    	Tool Used
Backend API	    FastAPI
Vector DB	    Qdrant
Cache	        Redis
Embeddings	    Sentence Transformers
LLM	            Groq(LLaMA 3.1-8B Instant)
UI     	        Streamlit




- Caching Strategy (Multi-Level)

The system implements a 3-level caching strategy to minimize redundant computation and reduce latency.


🔹 Level 1: Exact Query Cache (L1)
The most aggressive optimization. If the exact same query (lowercased and stripped) has been asked before, the cached answer and chunks are returned instantly. No embedding, no Qdrant, no LLM.

 What it does:

Stores:
query → answer + chunks
Matches:
Exact query string (case-insensitive)
⚡ Benefit:
Fastest response (~1–5 ms)
No embedding, no DB, no LLM


🔹 Level 2: Semantic Cache (L2)
Catches queries that are semantically equivalent but worded differently. For example, "What is AI?" and "Can you explain Artificial Intelligence?" would be considered similar enough to reuse the same cached answer.
If similarity >= 0.85 , the cached answer is returned.

 What it does:

Stores:
query_embedding → answer + chunks
Matches:
Similar queries using cosine similarity
 Example:
"What is ML?"
"Explain machine learning"
⚡ Benefit:
Avoids Qdrant search
Avoids LLM call
Handles paraphrased queries


🔹 Level 3: Retrieval Cache (L3)
Level 3 — Retrieval Cache (with Answer)
Handles the case where different queries retrieve the same set of document chunks. Instead of re-running Qdrant search and LLM generation, the cached answer for that specific chunk set is reused.
    • Chunks are hashed with MD5 to create a stable, deduplicated cache key
    • The cache stores both the retrieved chunks and the generated answer
    • Each new query that maps to the same chunk set appends its embedding to the entry for better future matching.
    
 What it does:
Stores:
query_embedding → chunks + answer
Uses:
Chunk hashing to avoid duplication
Multiple embeddings per chunk set
 Key Idea:

Different queries → same retrieved chunks → same answer

⚡ Benefit:
Skips BOTH:
Qdrant search ❌
LLM generation ❌


- Why 3 Levels?
Level   	Saves
L1	       Everything
L2	       LLM + DB
L3	       DB + LLM (for similar context)

## Query Flow with Edge Case Handling

Query: "How do computers learn from data without programming?"
│
▼
┌───────────────┐
│ L1: Exact Cache│ ── Miss (different string)
└───────────────┘
│
▼
┌───────────────┐
│ L2: Semantic │ ── Miss (similarity 0.78 < 0.85)
│ (max_checks) │
└───────────────┘
│
▼
┌───────────────┐
│ L3: Retrieval │
│ Cache with │
│ Answer │
└───────────────┘
│
▼
┌───────────────┐
│ If chunks exist:│
│ - Has answer? │ → Return cached answer ✅
│ - No answer? │ → Still need





⚡ Latency Optimization Techniques

The system is optimized to reduce response time using multiple strategies aligned with the assignment's low-latency requirement.

1.  Multi-Level Caching Strategy

Cache Level	                 Hit Condition	            Latency	          What is Saved
L1 (Exact)	              Exact query string match   	 ~2-5ms	          Qdrant search + LLM call
L2 (Semantic)	          Cosine similarity ≥ 0.85	     ~10-30ms	      Qdrant search + LLM call
L3 (Retrieval)	          Same chunk set retrieved	     ~10-30ms	      Qdrant search + LLM call
Full Pipeline (no hit)	  All caches miss	             500ms-2000ms	  Nothing (Qdrant + Groq LLM)

2. Batch Embedding During Ingestion

When a document is ingested, all chunks are embedded in a single batch call to SentenceTransformer instead of one call per chunk.

Code implementation (embedding.py):
python

def generate_embeddings_batch(texts):
    """Batch embedding for documents (FASTER)"""
    return model.encode(texts).tolist()

3. Non-Blocking Redis SCAN

Redis KEYS command blocks the server for its entire duration. This system uses scan_iter() instead, which is non-blocking and production-safe.

The Problem: Redis KEYS Command (Blocking)
What Happens Internally:

Your Code → Redis → "Give me ALL keys matching semantic:*"
                          │
                          ▼
                    Redis LOCKS itself
                    (Cannot process other requests)
                          │
                          ▼
                    Scans through 100,000 keys
                    (Takes 2-5 seconds)
                          │
                          ▼
                    Returns ALL keys at once
                          │
                          ▼
                    Redis UNLOCKS

Redis freezes for 2-5 seconds
All other users waiting for their queries are blocked
Your system appears unresponsive.

The Solution: Redis SCAN (Non-Blocking)
What Happens Internally:

Your Code → Redis → "Give me BATCH 1 of keys matching semantic:*"
                          │
                          ▼
                    Returns 100 keys
                    (Takes 10 milliseconds)
                          │
                          ▼
                    Your code processes those 100 keys
                          │
                          ▼
                    Meanwhile, Redis handles OTHER requests
                    (Not blocked!)
                          │
                          ▼
                    Your code asks: "Give me BATCH 2"
                          │
                          ▼
                    Returns next 100 keys
                    ..continues until done

4. L2 Semantic Cache with max_checks

The semantic cache limits comparisons to a maximum of entries per query to prevent unbounded scanning as cache grows.

5. Embedding Model Choice

all-MiniLM-L6-v2 is a lightweight 384-dimension model that runs efficiently on CPU, balancing embedding quality and inference speed.

6. Groq for LLM Inference

Groq's LPU hardware provides fast inference. llama-3.1-8b-instant is chosen specifically for speed, making LLM calls the smallest bottleneck when cache is missed.

7. Avoid Redundant Embedding Computation

Query embedding is generated only once per request and reused across all cache levels (L2 similarity check, L3 retrieval check, and Qdrant search).

8. Redis In-Memory Cache

Redis provides fast read/write operations suitable for low-latency systems. All cache levels use Redis as the storage backend.
Code implementation (config.py):
python
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))



## Trade-offs & Limitations

1. Linear Scan in Semantic & Retrieval Cache (L2 & L3)
Issue: Both L2 and L3 caches use scan_iter() which iterates over keys sequentially. As the cache grows, this becomes slower.

2. Embedding Growth in Retrieval Cache (L3)
Issue: Each new query that maps to the same chunk set appends its embedding to the entry.
Limitation: Embeddings list grows unboundedly as more queries map to the same chunk set, increasing memory usage over time.
Future improvement: Implement max embeddings per entry (e.g., keep most recent 100)

3. No Cache Eviction Policy (LRU / Size-based)
Issue: Currently no TTL (after your change) and no LRU/size-based eviction.
Your Code: cache.py → No TTL, no eviction policy
Limitation: Cache grows indefinitely until memory is exhausted.

4. Qdrant Sequential IDs (No Deduplication)
Issue: Chunks are stored with sequential integer IDs (0, 1, 2...).
Your Code: vector_store.py → PointStruct(id=i, ...)
Limitation: Re-ingesting the same document creates duplicate vectors instead of updating existing ones.
Impact: Qdrant collection grows with duplicate chunks over time.
Future improvement: Use document hash as ID for idempotent ingestion.

5. No Persistent Storage for Redis
Issue: Redis cache is in-memory only with no persistence configured.
Your Code: No RDB/AOF configuration
Limitation: A Redis restart clears all cached data.
Future improvement: Enable Redis RDB snapshots or AOF persistence.

6. No Authentication on API Endpoints
Limitation: Your /ingest and /query endpoints have no authentication.





## How The System Is Working

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              USER (Browser)                                          │
│                                   │                                                  │
│                          Click "Browse files"                                        │
│                          Select document.pdf                                         │
│                          Click "Upload" button                                       │
│                                   │                                                  │
│                                   ▼                                                  │
└──────────────────────────────────│──────────────────────────────────────────────────┘
                                    │
                                    │ HTTP POST (multipart/form-data)
                                    │ file = document.pdf
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           STREAMLIT UI (ui.py)                                       │
│                                                                                      │
│   uploaded_file = st.file_uploader(...)                                              │
│                                                                                      │
│   files = {"file": uploaded_file.getvalue()}                                         │
│   response = requests.post("http://localhost:8000/ingest", files=files)              │
│                                   │                                                  │
│                                   │ HTTP POST to FastAPI                            │
│                                   ▼                                                  │
└──────────────────────────────────│──────────────────────────────────────────────────┘
                                    │
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         FASTAPI BACKEND (app.py)                                     │
│                                                                                      │
│   @app.post("/ingest")                                                               │
│   async def ingest(file: UploadFile = File(...)):                                    │
│       │                                                                              │
│       │ 1. Save file temporarily                                                     │
│       │ 2. Call process_document(file_path)                                          │
│       │                                                                              │
│       ▼                                                                              │
└───────────────────────│─────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                        INGESTION (ingestion.py)                                      │
│                                                                                      │
│   def process_document(file_path):                                                   │
│       │                                                                              │
│       │ ┌─────────────────────────────────────────────────────────────────────┐     │
│       │ │ STEP 1: Read file based on extension                                 │     │
│       │ │                                                                       │     │
│       │ │ if file_path.endswith(".pdf"):                                       │     │
│       │ │     text = read_pdf(file_path)    → PyPDF2 extracts text             │     │
│       │ │ elif file_path.endswith(".txt") or file_path.endswith(".md"):        │     │
│       │ │     text = read_txt(file_path)    → Direct read                      │     │
│       │ └─────────────────────────────────────────────────────────────────────┘     │
│       │                                                                              │
│       │ ▼                                                                              │
│       │ ┌─────────────────────────────────────────────────────────────────────┐     │
│       │ │ STEP 2: Chunking with overlap                                        │     │
│       │ │                                                                       │     │
│       │ │ chunks = chunk_text_with_overlap(text)                               │     │
│       │ │                                                                       │     │
│       │ │ Example: "Machine learning is a branch of artificial intelligence"   │     │
│       │ │          ↓                                                            │     │
│       │ │ Chunk 1: "Machine learning is a branch of artificial"                │     │
│       │ │ Chunk 2: "artificial intelligence that allows computers to learn"    │     │
│       │ │          (50 word overlap)                                            │     │
│       │ └─────────────────────────────────────────────────────────────────────┘     │
│       │                                                                              │
│       │ ▼                                                                              │
│       │ ┌─────────────────────────────────────────────────────────────────────┐     │
│       │ │ STEP 3: Generate Embeddings (Batch)                                  │     │
│       │ │                                                                       │     │
│       │ │ embeddings = generate_embeddings_batch(chunks)                       │     │
│       │ │                                                                       │     │
│       │ │ SentenceTransformer("all-MiniLM-L6-v2")                              │     │
│       │ │ Each chunk → 384-dimension vector                                    │     │
│       │ └─────────────────────────────────────────────────────────────────────┘     │
│       │                                                                              │
│       │ ▼                                                                              │
│       │ ┌─────────────────────────────────────────────────────────────────────┐     │
│       │ │ STEP 4: Store in Qdrant Vector Database                              │     │
│       │ │                                                                       │     │
│       │ │ store_chunks(chunks, embeddings)                                     │     │
│       │ │                                                                       │     │
│       │ │ Qdrant Collection: "genai_docs"                                      │     │
│       │ │ Each point: {id, vector, payload: {text, document_name}}            │     │
│       │ └─────────────────────────────────────────────────────────────────────┘     │
│       │                                                                              │
│       │ ▼                                                                              │
│       │ return {"status": "success", "chunks_stored": len(chunks)}                    │
│       │                                                                              │
│       ▼                                                                              │
└───────────────────────│─────────────────────────────────────────────────────────────┘
                        │
                        │ Response: {"chunks_stored": 42}
                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           STREAMLIT UI (ui.py)                                       │
│                                                                                      │
│   st.success(f"✅ Ingested: {response.json()['chunks_stored']} chunks")             │
│                                   │                                                  │
│                                   ▼                                                  │
└──────────────────────────────────│──────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              USER (Browser)                                          │
│                                                                                      │
│   Sees: "✅ Ingested: 42 chunks"                                                     │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘




## Instructions to Run Locally

🔹 1. Clone Repository
git clone <your-repo-link>
cd genai-assistant

🔹 2. Setup Environment

Create .env file:

GROQ_API_KEY=your_api_key
QDRANT_HOST=localhost
QDRANT_PORT=6333
REDIS_HOST=localhost
REDIS_PORT=6379

🔹 3. Install Dependencies
pip install -r requirements.txt

4. Start Services

Make sure:

✅ Redis is running
✅ Qdrant is running

🔹 5. Run FastAPI Server
python app.py
or
uvicorn app:app --reload

🔹 6. Run UI 
streamlit run ui.py







