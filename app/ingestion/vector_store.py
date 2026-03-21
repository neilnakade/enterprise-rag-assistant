from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

vector_store = None


def create_vector_store(chunks):
    global vector_store

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    vector_store = FAISS.from_documents(chunks, embeddings)

    # Save locally (optional but good)
    vector_store.save_local("vector_db")

    return vector_store


def get_vector_store():
    return vector_store