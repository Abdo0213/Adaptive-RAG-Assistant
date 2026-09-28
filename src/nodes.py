import re
from .state import GraphState
from .prompts import ROUTER_PROMPT, GRADER_PROMPT, GENERATE_PROMPT, CHECK_PROMPT, REWRITE_PROMPT
from .llm import ask
from .knowledge_base import get_retriever, KB_TOPICS
from .web_tools import web_search_tool
from .config import WEB_IF_ANY_IRRELEVANT

def route_question(state: GraphState):
    question = state["question"]
    prompt = ROUTER_PROMPT.format(question=question)
    out = ask(prompt)
    route = "web_search" if "web_search" in out.lower() else "vectorstore"
    return {"route": route, "path": [f"route_question → {route}"]}

def retrieve_documents(state: GraphState):
    question = state["question"]
    retriever = get_retriever()
    docs = retriever.invoke(question)
    doc_dicts = [{"content": d.page_content, "source": d.metadata.get("source", ""), "origin": d.metadata.get("origin", "kb")} for d in docs]
    return {"documents": doc_dicts, "path": [f"retrieve_documents ({len(doc_dicts)} docs)"]}

def grade_documents(state: GraphState):
    question = state["question"]
    docs = state.get("documents", [])
    if not docs:
        return {"documents": [], "web_search_needed": True, "path": ["grade_documents (0 docs → web_search)"]}
        
    docs_text = "\n".join([f"[{i+1}] {d['content']}" for i, d in enumerate(docs)])
    prompt = GRADER_PROMPT.format(question=question, docs=docs_text)
    out = ask(prompt)
    
    if "none" in out.lower():
        relevant_indices = []
    else:
        relevant_indices = [int(i) - 1 for i in re.findall(r"\d+", out)]
        
    kept_docs = [docs[i] for i in relevant_indices if 0 <= i < len(docs)]
    total = len(docs)
    kept = len(kept_docs)
    
    web_search_needed = (kept == 0) or (WEB_IF_ANY_IRRELEVANT and kept < total)
    
    return {
        "documents": kept_docs,
        "web_search_needed": web_search_needed,
        "path": [f"grade_documents ({kept}/{total} relevant → {'web_search' if web_search_needed else 'generate'})"]
    }

def web_search(state: GraphState):
    question = state["question"]
    retries = state.get("retries", 0)
    
    if retries > 0:
        query = ask(REWRITE_PROMPT.format(question=question))
    else:
        query = question
        
    new_results = web_search_tool(query)
    
    docs = state.get("documents", [])
    seen = {d["content"][:80] for d in docs}
    
    added = 0
    for r in new_results:
        if r["content"][:80] not in seen:
            docs.append(r)
            seen.add(r["content"][:80])
            added += 1
            
    return {"documents": docs, "path": [f"web_search (+{added} results, query='{query}')"]}

def generate_answer(state: GraphState):
    question = state["question"]
    docs = state.get("documents", [])
    
    if not docs:
        return {"generation": "I don't have enough information.", "path": ["generate_answer"]}
        
    context = "\n".join([f"[{i+1}] ({d['source']}) {d['content']}" for i, d in enumerate(docs)])
    prompt = GENERATE_PROMPT.format(context=context, question=question)
    answer = ask(prompt)
    
    return {"generation": answer, "path": ["generate_answer"]}

def check_answer(state: GraphState):
    question = state["question"]
    answer = state.get("generation", "")
    docs = state.get("documents", [])
    
    if answer == "I don't have enough information.":
        return {"grounded": False, "useful": False, "retries": state.get("retries", 0) + 1, "path": ["check_answer (not enough info)"]}
        
    context = "\n".join([f"[{i+1}] {d['content']}" for i, d in enumerate(docs)])
    prompt = CHECK_PROMPT.format(context=context, question=question, answer=answer)
    out = ask(prompt)
    
    grounded = "GROUNDED: yes" in out.lower()
    useful = "ANSWERS: yes" in out.lower()
    
    retries = state.get("retries", 0)
    if not (grounded and useful):
        retries += 1
        
    return {
        "grounded": grounded, 
        "useful": useful, 
        "retries": retries, 
        "path": [f"check_answer (grounded={grounded}, answers={useful})"]
    }
