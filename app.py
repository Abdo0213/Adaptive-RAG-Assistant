import streamlit as st
import os
from src.config import require_api_key, OLLAMA_MODEL, MAX_RETRIES, WEB_IF_ANY_IRRELEVANT
from src.graph import stream_graph, get_mermaid, get_mermaid_png

st.set_page_config(page_title="Adaptive RAG · LangGraph", page_icon="🧠", layout="wide")

try:
    require_api_key()
except RuntimeError as e:
    st.error(str(e))
    st.stop()

st.title("🧠 Adaptive RAG Assistant")
st.markdown("A stateful, conditional and cyclic research assistant built with LangGraph.")

with st.sidebar:
    st.markdown(f"**Model:** `{OLLAMA_MODEL}`")
    st.markdown("**API Key:** ✅ Set")
    max_retries = st.slider("Max retries", 0, 4, MAX_RETRIES)
    strict_grading = st.toggle("Strict grading", WEB_IF_ANY_IRRELEVANT)
    
    st.subheader("Example questions")
    example_qs = [
        "What is a conditional edge in LangGraph?",
        "How does RAG help reduce hallucinations?",
        "What is the capital of Australia?"
    ]
    for q in example_qs:
        if st.button(q):
            st.session_state.current_query = q
            
    st.subheader("Graph architecture")
    mermaid_png = get_mermaid_png()
    if mermaid_png:
        st.image(mermaid_png)
    else:
        st.code(get_mermaid(), language="text")
        
    if st.button("Clear history"):
        st.session_state.history = []
        
if "history" not in st.session_state:
    st.session_state.history = []

query = st.chat_input("Ask a question...") or st.session_state.get("current_query")
if query:
    if "current_query" in st.session_state:
        del st.session_state["current_query"]
        
    st.chat_message("user").write(query)
    
    with st.status("Running graph…", expanded=True) as status:
        state_updates = []
        try:
            for node, update in stream_graph(query, max_retries):
                if "path" in update and update["path"]:
                    path_str = update["path"][-1]
                    emoji = {"route_question": "🧭", "retrieve_documents": "📥", "grade_documents": "⚖️", 
                             "web_search": "🌐", "generate_answer": "✍️", "check_answer": "✅"}.get(node, "▪️")
                    st.write(f"{emoji} {path_str}")
                state_updates.append(update)
            status.update(label="Graph complete!", state="complete")
        except Exception as e:
            status.update(label="Graph failed!", state="error")
            st.error(str(e))
            st.stop()
            
    if state_updates:
        final_state = state_updates[-1]
        answer = final_state.get("generation", "No answer generated.")
        grounded = final_state.get("grounded", False)
        useful = final_state.get("useful", False)
        retries = final_state.get("retries", 0)
        docs = final_state.get("documents", [])
        
        st.subheader("Answer")
        if grounded and useful:
            st.success("✅ Verified (grounded & answers)")
        else:
            st.warning("⚠️ Low confidence (retry limit reached / not verified)")
        st.write(answer)
        
        st.subheader("Sources")
        for d in docs:
            st.markdown(f"- **{d['origin'].upper()}**: {d['source']}")
            
        st.session_state.history.append({"question": query, "answer": answer, "path": final_state.get("path", [])})
        
if st.session_state.history:
    with st.expander("History"):
        for h in st.session_state.history:
            st.markdown(f"**Q:** {h['question']}")
            st.markdown(f"**A:** {h['answer']}")
            st.divider()
