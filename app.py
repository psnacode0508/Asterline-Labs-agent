"""Streamlit interface for the Asterline Labs research assistant."""

import os
import re
from typing import Dict, List

import streamlit as st

from graph import app as research_graph
from memory import UserMemoryStore


USER_OPTIONS = ["user_alice", "user_bob", "user_carol", "Custom user ID"]
CITATION_PATTERN = re.compile(r"\[([^\]\n]+)\]")


def get_memory(user_id: str) -> Dict[str, str]:
    """Read only the selected user's personal context."""
    store = UserMemoryStore()
    try:
        return store.get_user_memory(user_id)
    finally:
        store.close()


def clear_memory(user_id: str) -> int:
    """Clear only the selected user's personal context."""
    store = UserMemoryStore()
    try:
        return store.clear_user_memory(user_id)
    finally:
        store.close()


def get_history(user_id: str) -> List[Dict[str, object]]:
    """Return session chat history scoped to the active user."""
    histories = st.session_state.setdefault("chat_histories", {})
    return histories.setdefault(user_id, [])


def render_citations(response: str) -> None:
    """Render verified citations as visible grounding badges."""
    citations = sorted(set(CITATION_PATTERN.findall(response)))
    if citations:
        badges = " ".join(
            f"<span style='background:#e8f1ff; border-radius:12px; color:#174ea6; "
            f"font-size:0.8rem; margin-right:4px; padding:3px 8px;'>[{citation}]</span>"
            for citation in citations
        )
        st.markdown(badges, unsafe_allow_html=True)


def render_diagnostics(result: Dict[str, object]) -> None:
    """Display safe execution metadata without displaying credentials."""
    st.caption(f"Intent: `{result.get('intent', 'unknown')}`")
    documents = result.get("retrieved_docs", [])
    if documents:
        st.caption("Retrieved chunks")
        st.write(
            [
                {
                    "chunk_id": document.metadata.get("chunk_id"),
                    "title": document.metadata.get("title"),
                    "published": document.metadata.get("published"),
                    "version": document.metadata.get("version"),
                }
                for document in documents
            ]
        )


st.set_page_config(page_title="Asterline Research Assistant", page_icon="🔎", layout="wide")
st.title("Asterline Research Assistant")
st.caption("Grounded answers from Asterline documentation, with private user preferences kept separate.")

with st.sidebar:
    st.header("Workspace")
    selected_user = st.selectbox("User", USER_OPTIONS)
    if selected_user == "Custom user ID":
        user_id = st.text_input("Custom user ID", placeholder="e.g., user_dana").strip()
        if not user_id:
            st.info("Enter a custom user ID to begin.")
            st.stop()
    else:
        user_id = selected_user

    st.subheader("Active Memory")
    active_memory = get_memory(user_id)
    if active_memory:
        st.json(active_memory)
    else:
        st.caption("No personal preferences saved for this user.")

    if st.button("Clear My Memory", type="secondary"):
        deleted_count = clear_memory(user_id)
        st.success(f"Cleared {deleted_count} memory entr{'y' if deleted_count == 1 else 'ies'}.")
        st.rerun()

    st.subheader("Tracing & Diagnostics")
    show_diagnostics = st.toggle("Show agent diagnostics", value=False)
    tracing_enabled = os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"
    project_name = os.getenv("LANGCHAIN_PROJECT", "not configured")
    st.caption(
        f"LangSmith tracing: {'enabled' if tracing_enabled else 'disabled'} "
        f"({project_name})"
    )

history = get_history(user_id)
for message in history:
    with st.chat_message(str(message["role"])):
        st.markdown(str(message["content"]))
        if message["role"] == "assistant":
            render_citations(str(message["content"]))
            if show_diagnostics and message.get("diagnostics"):
                with st.expander("Execution details"):
                    render_diagnostics(message["diagnostics"])

user_input = st.chat_input("Ask about Asterline documentation or save a personal preference")
if user_input:
    history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Researching documentation..."):
            try:
                result = research_graph.invoke(
                    {
                        "user_id": user_id,
                        "user_input": user_input,
                        "user_memory": {},
                        "intent": "",
                        "memory_payload": {},
                        "retrieved_docs": [],
                        "response": "",
                    }
                )
                response = result["response"]
            except Exception:
                result = {}
                response = (
                    "I could not complete that request. Confirm that the local index, "
                    "Gemini configuration, and network connection are available."
                )

        st.markdown(response)
        render_citations(response)
        if show_diagnostics and result:
            with st.expander("Execution details", expanded=True):
                render_diagnostics(result)

    history.append({"role": "assistant", "content": response, "diagnostics": result})
