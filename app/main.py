from app.ingestion.loader import load_documents
from app.ingestion.chunking import split_documents
from app.ingestion.vector_store import create_vector_store
from app.generation.generator import generate_answer


def main():

    print("Loading documents...")
    documents = load_documents("data")
    print(f"Loaded {len(documents)} documents")

    print("Splitting documents...")
    chunks = split_documents(documents)
    print(f"Created {len(chunks)} chunks") 

    print("Creating vector store...")
    create_vector_store(chunks)

    print("\nRAG Assistant Ready!")
    print("Type 'exit' to quit.\n")

    while True:

        query = input("You: ")

        if query.lower() == "exit":
            break

        answer = generate_answer(query)

        print("\nAI:", answer)
        print()


if __name__ == "__main__":
    main()
