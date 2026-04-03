from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL

# Initialize client once
client = Groq(api_key=GROQ_API_KEY)

def generate_answer(query, context_chunks):
    """Generate answer using Groq LLM - SDK handles endpoint internally"""
    
    # FALLBACK: If no API key, return retrieved chunks
    if not GROQ_API_KEY:
        return "\n\n---\n\n".join(context_chunks)
    
    context = "\n\n".join(context_chunks)
    
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "Answer based only on the context provided. Be concise."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ],
            temperature=0.3,
            max_tokens=500
        )
        return response.choices[0].message.content
    
    except Exception as e:
        # Fallback to chunks if LLM fails
        return f"LLM Error: {str(e)}\n\nRetrieved context:\n{context}"