def decide_after_routing(state):
    return "retrieve_documents" if state["route"] == "vectorstore" else "web_search"

def decide_after_grading(state):
    return "web_search" if state["web_search_needed"] else "generate_answer"

def decide_after_check(state):
    if state.get("grounded") and state.get("useful"):
        return "end"
    if state.get("retries", 0) <= state.get("max_retries", 2):
        return "web_search"
    return "end"
