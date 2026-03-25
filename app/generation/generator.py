import os
import requests
from app.retrieval.retriever import retrieve_documents

API_URL = "https://router.huggingface.co/hf-inference/models/google/flan-t5-large"

HEADERS = {
    "Authorization": f"Bearer {os.getenv('HUGGINGFACEHUB_API_TOKEN')}"
}

def query_hf(payload):
    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload)

        # Handle non-200 responses
        if response.status_code != 200:
            return {"error": f"HTTP {response.status_code}: {response.text}"}

        # Try parsing JSON
        try:
            return response.json()
        except:
            return {"error": "Invalid JSON response from API"}

    except Exception as e:
        return {"error": str(e)}


def generate_answer(query):
    results = retrieve_documents(query, k=3)

    # Build context
    context = "\n".join([doc.page_content for doc, score in results])

    prompt = f"""
Answer ONLY from the context below.

Context:
{context}

Question:
{query}
"""

    response = query_hf({"inputs": prompt})

    # Handle different API responses
    if isinstance(response, list) and "generated_text" in response[0]:
        answer = response[0]["generated_text"]

    elif isinstance(response, dict) and "error" in response:
        answer = "⚠️ Model is busy or loading. Please try again in a few seconds."

    else:
        answer = "⚠️ Unexpected response. Try again."

    # Show only top source (clean UI)
    sources = [results[0][0].metadata.get("source", "")] if results else []

    return f"{answer}\n\nSources:\n" + "\n".join(sources)
