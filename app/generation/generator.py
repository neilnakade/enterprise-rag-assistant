import os
import requests
import time
from app.retrieval.retriever import retrieve_documents

API_URL = API_URL = "https://router.huggingface.co/hf-inference/models/distilgpt2"

HEADERS = {
    "Authorization": f"Bearer {os.getenv('HUGGINGFACEHUB_API_TOKEN')}"
}

def query_hf(payload):
    for _ in range(3):  # retry 3 times
        try:
            response = requests.post(API_URL, headers=HEADERS, json=payload)

            # If API fails, retry
            if response.status_code != 200:
                time.sleep(2)
                continue

            try:
                return response.json()
            except:
                time.sleep(2)
                continue

        except:
            time.sleep(2)

    return {"error": "Model unavailable after retries"}


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

    # Handle response
    if isinstance(response, list) and "generated_text" in response[0]:
        answer = response[0]["generated_text"]

    elif isinstance(response, dict) and "error" in response:
        answer = "⏳ Loading model... please retry once."

    else:
        answer = "⚠️ Unexpected response. Try again."

    # Show only top source
    sources = [results[0][0].metadata.get("source", "")] if results else []

    return f"{answer}\n\nSources:\n" + "\n".join(sources)
