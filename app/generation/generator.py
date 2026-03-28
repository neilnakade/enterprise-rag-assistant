import os
from groq import Groq
from app.retrieval.retriever import retrieve_documents

api_key = os.getenv("GROQ_API_KEY")

# simple in-memory chat history
chat_history = []

def generate_answer(query):
    global chat_history

    if not api_key:
        return "❌ API key not found."

    client = Groq(api_key=api_key)

    # 🔹 Step 1: Retrieve documents
    results = retrieve_documents(query, k=3)

    if not results:
        return "❌ No relevant information found in documents."

    # 🔹 Step 2: Pick BEST doc only
    results = sorted(results, key=lambda x: x[1], reverse=True)
    best_doc, best_score = results[0]

    # 🔹 Strict relevance filter (important)
    if best_score < 0.5:
        return "❌ Answer not found in provided documents."

    context = best_doc.page_content

    # 🔹 Step 3: Add memory (last 3 messages)
    history_text = ""
    for q, a in chat_history[-3:]:
        history_text += f"User: {q}\nAssistant: {a}\n"

    # 🔹 Step 4: Strict RAG prompt
    prompt = f"""
You are an enterprise assistant.

Rules:
- Answer ONLY from the provided context
- Do NOT use outside knowledge
- If answer not in context → reply EXACTLY: NOT_FOUND
- Be clear and concise

Conversation History:
{history_text}

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

    # 🔹 Step 5: Handle NOT_FOUND (NO fallback now)
    if answer == "NOT_FOUND":
        return "❌ Answer not found in provided documents."

    # 🔹 Step 6: Save memory
    chat_history.append((query, answer))

    # 🔹 Step 7: Show ONLY correct source
    source = best_doc.metadata.get("source", "Unknown")

    return f"{answer}\n\nSource:\n{source}"