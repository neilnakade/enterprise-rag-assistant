import streamlit as st
from app.generation.generator import generate_answer

st.set_page_config(page_title="Enterprise AI Assistant", layout="wide")

st.title("💼 Enterprise Knowledge Assistant")

# Chat history
if "chat" not in st.session_state:
    st.session_state.chat = []

# Input box
query = st.chat_input("Ask your question...")

# Generate response
if query:
    with st.spinner("Thinking..."):
        answer = generate_answer(query)

    st.session_state.chat.append(("user", query))
    st.session_state.chat.append(("ai", answer))

# Display chat (ChatGPT style)
for role, msg in st.session_state.chat:
    if role == "user":
        with st.chat_message("user"):
            st.markdown(msg)
    else:
        with st.chat_message("assistant"):
            st.markdown(msg)