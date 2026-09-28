import pytest
from src.graph import run_graph

def mock_ask(prompt, system=None):
    if "question router" in prompt.lower():
        return "web_search"
    if "grading retrieved documents" in prompt.lower():
        return "NONE"
    if "strict answer checker" in prompt.lower():
        return "GROUNDED: no\nANSWERS: no"
    return "Mock generation"

def test_graph_smoke(monkeypatch):
    import src.nodes
    import src.llm
    monkeypatch.setattr(src.nodes, "ask", mock_ask)
    monkeypatch.setattr(src.llm, "ask", mock_ask)
    
    state = run_graph("Smoke test question", max_retries=1)
    
    assert state["route"] == "web_search"
    assert "generation" in state
    assert state["retries"] == 2  # max_retries + 1 before stopping
