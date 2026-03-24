from langchain_community.llms import HuggingFaceHub
from app.retrieval.retriever import retrieve_documents

chat_history = []


def generate_answer(query):
    global chat_history

    results = retrieve_documents(query, k=3)

    # 🚫 No results case
    if not results:
        return "I don't know. This information is not available in the provided documents.\n\nSources:\n- None"

    # 🎯 Best score check
    best_score = results[0][1]

    # 🚫 Reject irrelevant queries
    if best_score > 1.2:
        return "I don't know. This information is not available in the provided documents.\n\nSources:\n- None"

    # ✅ Filter relevant results
    relevant_results = [r for r in results if r[1] < 1.2]

    # 📚 Build context
    context = "\n".join([
        doc.page_content for doc, score in relevant_results
    ])

    # 🧠 Build chat history
    history_text = ""
    for q, a in chat_history:
        history_text += f"User: {q}\nAI: {a}\n"

    # 🧾 Strict prompt
    prompt = f"""
You are an Enterprise Knowledge Assistant.

STRICT RULES:
- Answer ONLY from the given context
- DO NOT use outside knowledge
- If answer is not in context, say exactly: I don't know

Conversation History:
{history_text}

Context:
{context}

Question:
{query}

Answer:
"""

    llm = HuggingFaceHub(
    repo_id="google/flan-t5-large",
    model_kwargs={"temperature": 0.2, "max_length": 512}
    )

    response = llm.invoke(prompt)

    # 💾 Save memory
    chat_history.append((query, response))

    # 🚫 If model says "I don't know"
    if "i don't know" in response.lower():
        return "I don't know. This information is not available in the provided documents.\n\nSources:\n- None"

    # 📌 Clean sources (top 2)
    sorted_results = sorted(relevant_results, key=lambda x: x[1])[:1]

    sources = list(set([
        doc.metadata.get("source", "Unknown").split("\\")[-1]
        for doc, score in sorted_results
    ]))

    source_text = "\nSources:\n" + "\n".join([
        f"- {s}" for s in sources
    ])

    return response + source_text
