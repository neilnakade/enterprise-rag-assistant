import os
from groq import Groq
from app.retrieval.retriever import retrieve_documents

api_key = os.getenv("GROQ_API_KEY")

def generate_answer(query):
    if not api_key:
        return "❌ API key not found. Check Streamlit secrets."

    client = Groq(api_key=api_key)

    # Step 1: Retrieve documents
    results = retrieve_documents(query, k=3)
    context = "\n".join([doc.page_content for doc, score in results])

    # Step 2: Strict RAG prompt
    prompt = f"""
Answer using ONLY the context below.

If the answer is NOT in the context, reply EXACTLY:
NOT_FOUND

Context:
{context}

Question:
{query}
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}]
        )

        answer = response.choices[0].message.content.strip()

    except Exception as e:
        return f"❌ ERROR: {str(e)}"

    # Step 3: Fallback if not found
    if answer == "NOT_FOUND":
        fallback_prompt = f"Answer this question clearly:\n{query}"

        try:
            fallback_response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": fallback_prompt}]
            )

            fallback_answer = fallback_response.choices[0].message.content

            return f"Answering from general knowledge:\n\n{fallback_answer}\n\n(No document sources used)"

        except Exception as e:
            return f"❌ ERROR: {str(e)}"

    # Step 4: Correct source attribution
    sources = list(set([doc.metadata.get("source", "") for doc, _ in results]))

    return f"{answer}\n\nSources:\n" + "\n".join(sources)