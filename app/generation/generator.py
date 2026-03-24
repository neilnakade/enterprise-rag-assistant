import os
import requests
from app.retrieval.retriever import retrieve_documents

API_URL = API_URL = "https://router.huggingface.co/hf-inference/models/google/flan-t5-large"

HEADERS = {
    "Authorization": f"Bearer {os.getenv('HUGGINGFACEHUB_API_TOKEN')}"
}

def query_hf(payload):
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    return response.json()

def generate_answer(query):
    results = retrieve_documents(query, k=3)

    context = "\n".join([doc.page_content for doc, score in results])

    prompt = f"""
Answer ONLY from the context below.

Context:
{context}

Question:
{query}
"""

    response = query_hf({"inputs": prompt})

    if isinstance(response, list) and "generated_text" in response[0]:
        answer = response[0]["generated_text"]
    elif isinstance(response, dict) and "error" in response:
        answer = f"API Error: {response['error']}"
    else:
        answer = "Model is loading or rate limited. Try again."

    sources = list(set([doc.metadata.get("source", "") for doc, _ in results]))

    return f"{answer}\n\nSources:\n" + "\n".join(sources)
