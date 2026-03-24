from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from app.ingestion.loader import load_documents
from app.ingestion.splitter import split_documents

def retrieve_documents(query, k=3):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 🔥 Create vector DB dynamically
    documents = load_documents("data/raw")
    chunks = split_documents(documents)

    vector_store = FAISS.from_documents(chunks, embeddings)

    results = vector_store.similarity_search_with_score(query, k=k)
    return results
    