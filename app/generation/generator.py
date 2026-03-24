import os
import requests
from app.retrieval.retriever import retrieve_documents

API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"

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

    try:
        answer = response[0]["generated_text"]
    except:
        answer = "Error generating response. Try again."

    sources = list(set([doc.metadata.get("source", "") for doc, _ in results]))

    return f"{answer}\n\nSources:\n" + "\n".join(sources)
