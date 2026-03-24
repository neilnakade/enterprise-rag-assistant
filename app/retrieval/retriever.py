from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

def retrieve_documents(query, k=3):
    docs = []

    # Load all files
    for file in os.listdir("data/raw"):
        loader = TextLoader(os.path.join("data/raw", file))
        docs.extend(loader.load())

    # Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Vector store (dynamic)
    vector_store = FAISS.from_documents(chunks, embeddings)

    return vector_store.similarity_search_with_score(query, k=k)
    