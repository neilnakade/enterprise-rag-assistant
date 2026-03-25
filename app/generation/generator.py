import os
from groq import Groq
from app.retrieval.retriever import retrieve_documents

api_key = os.getenv("GROQ_API_KEY")

def generate_answer(query):
    if not api_key:
        return "❌ API key not found. Check Streamlit secrets."

    client = Groq(api_key=api_key)

    results = retrieve_documents(query, k=3)
    context = "\n".join([doc.page_content for doc, score in results])

    prompt = f"""
Answer ONLY from the context below.

Context:
{context}

Question:
{query}
"""

    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}]
        )

        answer = response.choices[0].message.content

    except Exception as e:
        return f"❌ ERROR: {str(e)}"

    sources = [results[0][0].metadata.get("source", "")] if results else []

    return f"{answer}\n\nSources:\n" + "\n".join(sources)