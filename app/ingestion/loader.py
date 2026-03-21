import os
from langchain_community.document_loaders import TextLoader


def load_documents(folder_path):
    """
    Loads all .txt files recursively from a folder.
    """

    documents = []

    for root, dirs, files in os.walk(folder_path):   # 👈 key change
        for filename in files:
            if filename.endswith(".txt"):
                file_path = os.path.join(root, filename)
                loader = TextLoader(file_path)
                documents.extend(loader.load())

    return documents