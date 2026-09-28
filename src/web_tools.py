import logging
import requests
from duckduckgo_search import DDGS
from .config import WEB_MAX_RESULTS, OLLAMA_HOST, OLLAMA_API_KEY
from .llm import get_client

def web_search_tool(query: str, n: int = WEB_MAX_RESULTS) -> list[dict]:
    n = min(n, 10)
    try:
        url = f"{OLLAMA_HOST.rstrip('/')}/api/web_search"
        headers = {"Authorization": f"Bearer {OLLAMA_API_KEY}"}
        resp = requests.post(url, headers=headers, json={"query": query, "max_results": n}, timeout=10)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        out = [{"content": r["content"][:1500], "source": r["url"], "origin": "web"} for r in results]
        return out
    except Exception as e:
        logging.warning(f"Ollama web search failed: {e}. Falling back to DDGS.")
        try:
            ddgs_results = DDGS().text(query, max_results=n)
            out = []
            for r in ddgs_results:
                out.append({"content": r.get("body", "")[:1500], "source": r.get("href", ""), "origin": "web"})
            return out
        except Exception as e2:
            logging.warning(f"DDGS web search also failed: {e2}")
            return []
