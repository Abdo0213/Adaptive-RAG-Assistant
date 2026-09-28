from langgraph.graph import StateGraph, START, END
from .state import GraphState
from .nodes import route_question, retrieve_documents, grade_documents, web_search, generate_answer, check_answer
from .edges import decide_after_routing, decide_after_grading, decide_after_check
from .config import MAX_RETRIES

def build_graph():
    builder = StateGraph(GraphState)
    builder.add_node("route_question", route_question)
    builder.add_node("retrieve_documents", retrieve_documents)
    builder.add_node("grade_documents", grade_documents)
    builder.add_node("web_search", web_search)
    builder.add_node("generate_answer", generate_answer)
    builder.add_node("check_answer", check_answer)

    builder.add_edge(START, "route_question")
    builder.add_edge("retrieve_documents", "grade_documents")
    builder.add_edge("web_search", "generate_answer")
    builder.add_edge("generate_answer", "check_answer")

    builder.add_conditional_edges("route_question", decide_after_routing,
        {"retrieve_documents": "retrieve_documents", "web_search": "web_search"})
    builder.add_conditional_edges("grade_documents", decide_after_grading,
        {"generate_answer": "generate_answer", "web_search": "web_search"})
    builder.add_conditional_edges("check_answer", decide_after_check,
        {"web_search": "web_search", "end": END})

    return builder.compile()

graph = build_graph()

def initial_state(question: str, max_retries: int = MAX_RETRIES) -> GraphState:
    return {
        "question": question,
        "route": "",
        "documents": [],
        "generation": "",
        "web_search_needed": False,
        "grounded": False,
        "useful": False,
        "retries": 0,
        "max_retries": max_retries,
        "path": []
    }

def run_graph(question: str, max_retries: int = MAX_RETRIES):
    state = initial_state(question, max_retries)
    return graph.invoke(state, config={"recursion_limit": 25})

def stream_graph(question: str, max_retries: int = MAX_RETRIES):
    state = initial_state(question, max_retries)
    return graph.stream(state, stream_mode="updates", config={"recursion_limit": 25})

def get_mermaid():
    return graph.get_graph().draw_mermaid()

def get_mermaid_png():
    try:
        return graph.get_graph().draw_mermaid_png()
    except Exception:
        return None
