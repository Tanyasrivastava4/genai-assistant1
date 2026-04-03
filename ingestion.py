import os
import PyPDF2
from config import CHUNK_SIZE, CHUNK_OVERLAP
from embedding import generate_embeddings_batch
from vector_store import store_chunks

def read_pdf(file_path):
    text = ""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def read_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def chunk_text_with_overlap(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Chunk text with overlap for better context preservation"""
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks

def process_document(file_path):
    # Read file
    if file_path.endswith(".pdf"):
        text = read_pdf(file_path)
    elif file_path.endswith(".txt") or file_path.endswith(".md"):
        text = read_txt(file_path)
    else:
        raise ValueError("Unsupported file type")

    print(f"Extracted {len(text)} characters")
    
    # Chunking with overlap
    chunks = chunk_text_with_overlap(text)
    print(f"Created {len(chunks)} chunks")

    # Batch embedding
    embeddings = generate_embeddings_batch(chunks)
    print(f"Generated {len(embeddings)} embeddings")

    # Store in Qdrant
    store_chunks(chunks, embeddings)
    print(f"Stored in Qdrant")

    return {"status": "success", "chunks_stored": len(chunks)}