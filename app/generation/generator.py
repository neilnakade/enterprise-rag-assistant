from langchain_community.llms import HuggingFaceHub
from app.retrieval.retriever import retrieve_documents
import os

def generate_answer(query):
    results = retrieve_documents(query, k=3)

    context = "\n".join([doc.page_content for doc, score in results])

    llm = HuggingFaceHub(
    repo_id="google/flan-t5-large",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    model_kwargs={"temperature": 0.2, "max_length": 512}
    )

    prompt = f"""
    Answer ONLY from the context below.

    Context:
    {context}

    Question:
    {query}
    """

    answer = llm(prompt)

    sources = list(set([doc.metadata.get("source", "") for doc, _ in results]))

    return f"{answer}\n\nSources:\n" + "\n".join(sources)
