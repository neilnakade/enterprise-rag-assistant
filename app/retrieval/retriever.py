from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS


def retrieve_documents(query, k=5):

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    vector_store = FAISS.load_local(
        "vector_db",
        embeddings,
        allow_dangerous_deserialization=True
    )

    results = vector_store.similarity_search_with_score(query, k=k)

    return results
    