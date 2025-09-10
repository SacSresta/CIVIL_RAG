
import streamlit as st
import os
import tempfile
from civic_rag.backend import database
from civic_rag.backend import rag

st.set_page_config(page_title="Civic RAG for Nepal", layout="wide")
database.init_db()
st.title("🇳🇵 Civic Protest Guidance & Insights")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat messages from history (latest at bottom)
st.subheader("Chat History")
chat_container = st.container()
with chat_container:
    # Display messages in chronological order (oldest first, latest at bottom)
    for sender, message in st.session_state.chat_history:
        if sender == "user":
            st.markdown(f"**🧑 You:** {message}")
        else:
            st.markdown(f"**🤖 Bot:** {message}")

# Input section at the bottom
st.markdown("---")
st.subheader("Ask a Question")

# Create columns for better layout
col1, col2 = st.columns([4, 1])

with col1:
    user_input = st.text_input("Type your question about protests:", key="user_input", placeholder="e.g., How can protests be conducted peacefully?")

with col2:
    send_button = st.button("Send", type="primary")

# Handle message submission
if send_button and user_input:
    # Add user message to history
    st.session_state.chat_history.append(("user", user_input))
    
    # Get bot response
    with st.spinner("Thinking..."):
        answer = rag.answer_query(user_input)
    
    # Add bot response to history
    st.session_state.chat_history.append(("bot", answer))
    
    # Save to database
    database.save_query(user_input, answer)
    
    # Rerun to show new messages
    st.rerun()

# Clear chat button
if st.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.rerun()
