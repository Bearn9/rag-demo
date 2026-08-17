"""Streamlit chat UI for the Bearn Assistant (Nova).

Run with:  streamlit run app.py
"""

import streamlit as st

from ragbot.graph import get_graph

st.set_page_config(page_title="Bearn Assistant", page_icon="\N{COMPASS}")


@st.cache_resource(show_spinner=False)
def _graph():
    return get_graph()


with st.sidebar:
    st.markdown("### Bearn Assistant")
    st.caption("Powered by Claude + LangGraph + FAISS")
    st.markdown(
        "Nova is an agentic RAG demo. It decides per-query whether to search "
        "Bearn Analytics' internal knowledge base (HR policies, IT setup, "
        "security policy, remote-work handbook, product FAQs, company financials) "
        "or answer directly."
    )
    show_routing = st.toggle("Show routing details", value=True)

st.title("\N{SPEECH BALLOON} Nova — Bearn Assistant")

def _render_meta(meta: dict) -> None:
    if meta["route"] == "retrieve":
        st.caption(f"\N{LEFT-POINTING MAGNIFYING GLASS} Retrieved from knowledge base — {meta['route_reasoning']}")
        if meta.get("chunks"):
            with st.expander("Sources used"):
                for chunk in meta["chunks"]:
                    st.markdown(
                        f"**{chunk['source']}** — {chunk['section']} "
                        f"(score: {chunk['score']:.2f})"
                    )
    else:
        st.caption(f"\N{SPEECH BALLOON} Answered directly — {meta['route_reasoning']}")


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and show_routing and message.get("meta"):
            _render_meta(message["meta"])


query = st.chat_input(
    "Ask about PTO, benefits, IT setup, security policy, pricing, or company financials..."
)

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        status_box = st.status("Deciding whether to search the knowledge base...", expanded=False)
        meta: dict = {}
        final_state: dict = {}

        for event in _graph().stream({"query": query}, stream_mode="updates"):
            for node_name, node_output in event.items():
                final_state.update(node_output)
                if node_name == "route_query":
                    if node_output["route"] == "retrieve":
                        status_box.update(label="Searching knowledge base...")
                    else:
                        status_box.update(label="Answering directly...")
                elif node_name == "retrieve":
                    status_box.update(label="Generating answer from retrieved sources...")
                elif node_name in ("generate_with_context", "generate_direct"):
                    status_box.update(label="Done", state="complete")

        answer = final_state.get("answer", "Sorry, something went wrong generating a response.")
        meta = {
            "route": final_state.get("route"),
            "route_reasoning": final_state.get("route_reasoning", ""),
            "chunks": final_state.get("retrieved_chunks") or [],
        }

        st.markdown(answer)
        if show_routing:
            _render_meta(meta)

    st.session_state.messages.append({"role": "assistant", "content": answer, "meta": meta})
