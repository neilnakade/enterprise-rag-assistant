import os
from groq import Groq
from app.retrieval.retriever import retrieve_documents

api_key = os.getenv("GROQ_API_KEY")

chat_history = []

def generate_answer(query):
    global chat_history

    if not api_key:
        return "❌ API key not found."

    client = Groq(api_key=api_key)

    # 🔹 Step 1: Retrieve top 3
    results = retrieve_documents(query, k=3)

    if not results:
        return "❌ Answer not found in provided documents."

    # 🔹 Step 2: Sort by score
    results = sorted(results, key=lambda x: x[1], reverse=True)

    # 🔹 Step 3: Filter relevant docs
    relevant_docs = [(doc, score) for doc, score in results if score > 0.5]

    if not relevant_docs:
        return "❌ Answer not found in provided documents."

    # 🔹 Step 4: Take TOP 2 docs (better context)
    top_docs = relevant_docs[:2]

    context = "\n\n".join([doc.page_content for doc, _ in top_docs])

    # 🔹 Memory (last 3)
    history_text = ""
    for q, a in chat_history[-3:]:
        history_text += f"User: {q}\nAssistant: {a}\n"

    # 🔹 Better strict prompt
    prompt = f"""
You are an enterprise knowledge assistant.

Rules:
- Answer ONLY using the given context
- Provide COMPLETE and SPECIFIC answers
- Do NOT be vague
- If answer not present → reply EXACTLY: NOT_FOUND

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

    if answer == "NOT_FOUND":
        return "❌ Answer not found in provided documents."

    # 🔹 Save memory
    chat_history.append((query, answer))

    # 🔹 Show UNIQUE sources only
    sources = list(set([doc.metadata.get("source", "Unknown") for doc, _ in top_docs]))

    return f"{answer}\n\nSource:\n" + "\n".join(sources)