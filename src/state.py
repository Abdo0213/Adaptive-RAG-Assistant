import operator
from typing import TypedDict, Annotated, List

class GraphState(TypedDict):
    question: str
    route: str                       # "vectorstore" | "web_search"
    documents: List[dict]            # {"content", "source", "origin": "kb"|"web"}
    generation: str
    web_search_needed: bool
    grounded: bool
    useful: bool
    retries: int
    max_retries: int
    path: Annotated[List[str], operator.add]   # execution trace (append-only reducer)
