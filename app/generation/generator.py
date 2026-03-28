import os
from groq import Groq
from app.retrieval.retriever import retrieve_documents

api_key = os.getenv("GROQ_API_KEY")

def generate_answer(query):
    if not api_key:
        return "❌ API key not found. Check Streamlit secrets."

    client = Groq(api_key=api_key)

    # Step 1: Retrieve documents (with scores)
    results = retrieve_documents(query, k=3)

    if not results:
        return "No relevant documents found."

    # Sort by similarity score (best first)
    results = sorted(results, key=lambda x: x[1], reverse=True)

    # Take ONLY best document
    best_doc, best_score = results[0]
    context = best_doc.page_content

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

    # Step 3: Fallback
    if answer == "NOT_FOUND":
        fallback_prompt = f"""
Answer the question directly and confidently.

Do NOT ask for clarification.

Question:
{query}
"""

        try:
            fallback_response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": fallback_prompt}]
            )

            fallback_answer = fallback_response.choices[0].message.content

            return f"Answering from general knowledge:\n\n{fallback_answer}\n\n(No document sources used)"

        except Exception as e:
            return f"❌ ERROR: {str(e)}"

    # Step 4: Show ONLY best source
    source = best_doc.metadata.get("source", "Unknown")

    return f"{answer}\n\nSource:\n{source}"