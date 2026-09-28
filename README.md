# Adaptive RAG Assistant (LangGraph + Ollama Cloud + Streamlit)

A stateful, conditional and **cyclic** research assistant built with **LangGraph**.
The app runs **locally** (Python venv + Streamlit UI). The LLM and web search are served by **Ollama Cloud** (hosted API at `https://ollama.com`), authenticated with an API key stored in a `.env` file.

> **Note for the code generator:** this README is the full specification. Implement every file listed in [Project structure](#project-structure), following the contracts, prompts and graph wiring below exactly. Do not simplify the graph into a linear pipeline.

---

## 1. What the system does

```
Question → Router ─┬─ related to KB  → Retrieve → Grade ─┬─ relevant ────→ Generate → Check ─┬─ OK ─────────────→ END
                   │                                     └─ irrelevant ──→ Web Search        ├─ fail + retries → Web Search (LOOP)
                   └─ unrelated ─────────────────────────────────────────→ Web Search        └─ fail + limit ──→ END (low confidence)
```

1. A **Question Router** decides whether the question is about the local knowledge base (KB) or needs web search.
2. KB questions: **retrieve** documents from a local vector store, then **grade** them. Irrelevant results trigger web search.
3. **Generate** an answer from the context.
4. An **Answer Checker** verifies the answer is *grounded* in the context and *actually answers* the question. If not, the graph loops back to web search (with a rewritten query) and regenerates, up to `MAX_RETRIES`.

### LangGraph concepts that must be demonstrated
| Concept | Where |
|---|---|
| **State** | One shared `GraphState` TypedDict; `path` uses the `operator.add` reducer |
| **Nodes** | `route_question`, `retrieve_documents`, `grade_documents`, `web_search`, `generate_answer`, `check_answer` |
| **Edges** | `START→route_question`, `retrieve_documents→grade_documents`, `web_search→generate_answer`, `generate_answer→check_answer` |
| **Conditional edges** | after `route_question`, after `grade_documents`, after `check_answer` |
| **Routing** | Question router (KB vs web) |
| **Loops** | `check_answer → web_search → generate_answer → check_answer`, bounded by `max_retries` and `recursion_limit` |
| **START / END** | Entry point and terminal node |

---

## 2. Tech stack

| Purpose | Library |
|---|---|
| Workflow | `langgraph`, `langchain-core` |
| LLM (Ollama Cloud) | `ollama` (official Python client) **or** `langchain-ollama` |
| Web search | Ollama Cloud web search API (same API key), optional DuckDuckGo fallback (`ddgs`) |
| Vector store | `langchain-chroma` (in-memory or persisted in `./chroma_db`) |
| Embeddings | `langchain-huggingface` + `sentence-transformers/all-MiniLM-L6-v2` (runs locally on CPU) |
| UI | `streamlit` |
| Config | `python-dotenv` |
| Tests | `pytest`, `pandas` |

`requirements.txt`:
```
langgraph
langchain-core
langchain-ollama
langchain-chroma
langchain-huggingface
sentence-transformers
ollama
ddgs
streamlit
python-dotenv
pandas
pytest
requests
```

---

## 3. Project structure

```
adaptive-rag/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore                # must include: .env, .venv/, chroma_db/, __pycache__/
├── app.py                    # Streamlit UI
├── run_tests.py              # runs the 10 test questions, prints table, saves results.csv
├── src/
│   ├── __init__.py
│   ├── config.py             # loads .env, constants
│   ├── llm.py                # Ollama Cloud client + ask()
│   ├── knowledge_base.py     # KB documents + Chroma vector store + retriever
│   ├── web_tools.py          # Ollama web search (+ DDGS fallback)
│   ├── state.py              # GraphState
│   ├── prompts.py            # all prompt templates
│   ├── nodes.py              # the 6 nodes
│   ├── edges.py              # 3 conditional-edge functions
│   └── graph.py              # builds & compiles the graph, run_graph(), stream_graph()
└── tests/
    └── test_graph.py         # smoke tests (routing logic, retry cap) with mocked LLM
```

---

## 4. Installation (virtual environment)

### 4.1 Prerequisites
- Python 3.10+
- A free Ollama account and an **API key**: https://ollama.com/settings/keys

### 4.2 Setup

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
```

### 4.3 Configure `.env`
```env
# Required: your Ollama Cloud API key
OLLAMA_API_KEY=your_api_key_here

# Ollama Cloud endpoint (hosted API)
OLLAMA_HOST=https://ollama.com

# Model name for the HOSTED API (no ":cloud" / "-cloud" suffix)
OLLAMA_MODEL=gpt-oss:120b

# Graph settings
MAX_RETRIES=2
TOP_K=3
WEB_IF_ANY_IRRELEVANT=false
WEB_MAX_RESULTS=4
```
`.env.example` must contain the same keys with placeholder values. **Never commit `.env`.**

### 4.4 Run

Make sure your virtual environment is activated and you have set your `OLLAMA_API_KEY` in `.env`.

**To run the interactive Streamlit UI:**
```bash
streamlit run app.py
```
This will open the web app at `http://localhost:8501`.

**To run the test suite:**
```bash
python run_tests.py
```
This runs 10 evaluation questions and saves the results to `results.csv`.

**To run offline smoke tests:**
```bash
pytest -q tests/
```


---

## 5. Module specifications

### 5.1 `src/config.py`
- `load_dotenv()` at import.
- Expose: `OLLAMA_API_KEY`, `OLLAMA_HOST` (default `https://ollama.com`), `OLLAMA_MODEL` (default `gpt-oss:120b`), `MAX_RETRIES` (int, 2), `TOP_K` (int, 3), `WEB_IF_ANY_IRRELEVANT` (bool, False), `WEB_MAX_RESULTS` (int, 4).
- `def require_api_key()` raises a clear `RuntimeError("OLLAMA_API_KEY is missing. Add it to your .env file.")` when unset. The Streamlit app must catch this and show `st.error(...)` + `st.stop()`.

### 5.2 `src/llm.py`
Use the official client pointed at the hosted API:
```python
import os
from ollama import Client

client = Client(
    host=OLLAMA_HOST,                                   # https://ollama.com
    headers={"Authorization": "Bearer " + OLLAMA_API_KEY},
)

def ask(prompt: str, system: str = "You are a precise assistant. Follow the requested output format exactly.") -> str:
    resp = client.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": prompt}],
        options={"temperature": 0},
    )
    text = (resp.message.content or "").strip()
    if not text:
        raise ValueError("empty response from LLM")
    return text
```
Requirements:
- Create the client lazily (first call) so importing the module does not fail without a key.
- Retry up to 2 times with exponential backoff (1s, 2s) on transient errors (HTTP 429/5xx, timeouts) before raising.
- Track the last model used in a module-level variable for display in the UI.
- Reasoning models (e.g. `gpt-oss`) may return a separate `thinking` field; only use `message.content`.

### 5.3 `src/knowledge_base.py`
- Define `KB` as a list of `(source, text)` tuples covering **LLM / RAG / LangGraph engineering** (14 documents, 2–3 sentences each): `langgraph_state`, `langgraph_nodes`, `langgraph_edges`, `langgraph_conditional_edges`, `langgraph_cycles`, `langgraph_checkpointer`, `rag_basics`, `rag_hallucination`, `chunking`, `embeddings`, `vector_databases`, `agents_react`, `adaptive_rag`, `prompt_engineering`.
- The KB must **not** cover: Reflexion, fine-tuning details, legal/medical topics, news, sports, geography (so tests can trigger web fallback).
- Build a Chroma collection with `HuggingFaceEmbeddings("sentence-transformers/all-MiniLM-L6-v2")`; persist to `./chroma_db`; rebuild only if the collection is empty.
- Expose `get_retriever()` (cached, `k=TOP_K`) and `KB_TOPICS` (string used by the router prompt).
- (Nice to have) allow adding documents from uploaded `.txt`/`.md` files through the Streamlit sidebar.

### 5.4 `src/web_tools.py`
Primary: **Ollama Cloud web search** (same API key).
```python
def web_search_tool(query: str, n: int = WEB_MAX_RESULTS) -> list[dict]:
    # Option A (python client):  client.web_search(query=query, max_results=n)
    # Option B (REST):           POST https://ollama.com/api/web_search
    #                            headers: Authorization: Bearer <OLLAMA_API_KEY>
    #                            json: {"query": query, "max_results": n}
    # Response: {"results": [{"title", "url", "content"}, ...]}
    ...
    return [{"content": r["content"], "source": r["url"], "origin": "web"} for r in results]
```
- `max_results` max is 10.
- On any exception: log a warning and fall back to `ddgs` (`DDGS().text(query, max_results=n)` → map `body`→`content`, `href`→`source`). If that also fails, return `[]`.
- Truncate each result `content` to ~1500 characters.

### 5.5 `src/state.py`
```python
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
```

### 5.6 `src/prompts.py`
Use these exact templates (`str.format`).

```python
ROUTER_PROMPT = """You are a question router. The local knowledge base covers ONLY: LangGraph (state, nodes, edges, checkpointers), RAG, embeddings, vector databases, chunking, LLM agents, and prompt engineering.

Reply with exactly one word:
- vectorstore  -> if the question is about those topics
- web_search   -> for anything else (news, sports, geography, people, current events, general knowledge)

Question: {question}
Answer:"""

GRADER_PROMPT = """You are grading retrieved documents for relevance to a question.
A document is relevant if it contains information that helps answer the question.

Question: {question}

Documents:
{docs}

Reply ONLY with the numbers of the relevant documents separated by commas (e.g. 1,3). If none are relevant reply NONE."""

GENERATE_PROMPT = """Answer the question using ONLY the context below. Be concise (max 5 sentences).
If the context is insufficient, say exactly: "I don't have enough information."

Context:
{context}

Question: {question}
Answer:"""

CHECK_PROMPT = """You are a strict answer checker.

Documents:
{context}

Question: {question}
Answer: {answer}

Reply in exactly this format:
GROUNDED: yes/no    (is every claim in the answer supported by the documents?)
ANSWERS: yes/no     (does the answer actually resolve the question?)"""

REWRITE_PROMPT = """Rewrite the question as a short, effective web search query (max 12 words). Output only the query.
Question: {question}"""
```
> Design choice: grading is done in **one** LLM call for all documents, and answer checking (grounded + useful) in **one** call, to minimise API usage (~4 LLM calls per question).

### 5.7 `src/nodes.py`
Each node takes `state` and returns a **partial** update including a `path` entry (list with one string).

| Node | Behaviour | `path` entry example |
|---|---|---|
| `route_question` | `ask(ROUTER_PROMPT)`; `route = "web_search" if "web_search" in out.lower() else "vectorstore"` | `route_question → web_search` |
| `retrieve_documents` | `retriever.invoke(question)` → list of dicts with `origin="kb"` | `retrieve_documents (3 docs)` |
| `grade_documents` | One LLM call with numbered docs; parse numbers with `re.findall(r"\d+")`; `NONE` → empty set; keep only relevant docs; `web_search_needed = (kept == 0) or (WEB_IF_ANY_IRRELEVANT and kept < total)` | `grade_documents (2/3 relevant → generate)` |
| `web_search` | If `retries > 0`, rewrite the query with `REWRITE_PROMPT`; call `web_search_tool`; **append** new results to existing documents, de-duplicating by first 80 chars of content | `web_search (+4 results, query='...')` |
| `generate_answer` | Build numbered context `[i] (source) text`; if no context → return `"I don't have enough information."`; else `ask(GENERATE_PROMPT)` | `generate_answer` |
| `check_answer` | Parse `GROUNDED: yes/no` and `ANSWERS: yes/no` with regex (missing → `False`); `retries += 1` only when not OK | `check_answer (grounded=True, answers=True)` |

### 5.8 `src/edges.py`
```python
def decide_after_routing(state):   # -> "retrieve_documents" | "web_search"
    return "retrieve_documents" if state["route"] == "vectorstore" else "web_search"

def decide_after_grading(state):   # -> "generate_answer" | "web_search"
    return "web_search" if state["web_search_needed"] else "generate_answer"

def decide_after_check(state):     # -> "web_search" | "end"
    if state["grounded"] and state["useful"]:
        return "end"
    if state["retries"] <= state["max_retries"]:
        return "web_search"        # LOOP
    return "end"                   # retry budget exhausted
```
With `max_retries=2`: 1st failure → `retries=1` (retry), 2nd → `retries=2` (retry), 3rd → `retries=3` (stop). So max **2 retry loops**.

### 5.9 `src/graph.py`
```python
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

graph = builder.compile()
```
Also provide:
- `initial_state(question, max_retries)` → full initial state dict (`retries=0`, `path=[]`, etc.).
- `run_graph(question, max_retries=MAX_RETRIES)` → `graph.invoke(state, config={"recursion_limit": 25})`.
- `stream_graph(question, max_retries)` → generator over `graph.stream(..., stream_mode="updates")` yielding `(node_name, update)`.
- `get_mermaid()` → `graph.get_graph().draw_mermaid()`; `get_mermaid_png()` → `draw_mermaid_png()` wrapped in try/except (returns `None` on failure).

---

## 6. Streamlit UI (`app.py`)

**Layout**
- `st.set_page_config(page_title="Adaptive RAG · LangGraph", page_icon="🧠", layout="wide")`.
- Title and one-line description.
- **Sidebar**
  - Model name (read-only, from `.env`) and API key status (✅/❌, never show the key).
  - Slider **Max retries** (0–4, default `MAX_RETRIES`).
  - Toggle **Strict grading** (`WEB_IF_ANY_IRRELEVANT`).
  - Section **Example questions**: the 10 test questions as buttons that fill the input.
  - Expander **Graph architecture**: the Mermaid PNG if available, otherwise a `st.code(mermaid, language="text")`.
  - Button **Clear history**.
- **Main area**
  - `st.text_input`/`st.chat_input` for the question and an **Ask** button.
  - While running, use `st.status("Running graph…", expanded=True)` and write each node's `path` entry live as `graph.stream` yields updates (with emoji per node: 🧭 route, 📥 retrieve, ⚖️ grade, 🌐 web, ✍️ generate, ✅ check).
  - After completion show:
    1. **Answer** in a card, with a badge: ✅ *Verified* (grounded & answers) or ⚠️ *Low confidence* (retry limit reached / not verified).
    2. **Execution path**: numbered list of steps + a compact `A → B → C` summary, and a metric row: route, retries used, number of docs, grounded, answers.
    3. **Sources**: table/list with origin badge (`KB` / `WEB`) and source name/URL (clickable for web).
  - Keep a **History** (in `st.session_state`) of previous questions with their answer and path, shown in an expander list.
- Handle errors gracefully (`st.error`), including a missing API key, HTTP 401 (invalid key), 429 (rate limited) and network errors. Cache the vector store with `@st.cache_resource`.

---

## 7. Test suite (`run_tests.py`)

Run these **10 questions** through `run_graph` and print a table (question, expected, route, path, retries, grounded, useful, answer preview); save to `results.csv`. Also print the detailed numbered trace of at least 5 of them.

| # | Question | Expected path |
|---|---|---|
| 1 | What is a conditional edge in LangGraph? | KB → relevant → answer |
| 2 | How does RAG help reduce hallucinations? | KB → relevant → answer |
| 3 | What is a checkpointer in LangGraph and why is it useful? | KB → relevant → answer |
| 4 | What is the difference between a vector database and SQL? | KB → relevant → answer |
| 5 | What is the capital of Australia? | Router → web directly |
| 6 | Who is the current CEO of Anthropic? | Router → web directly |
| 7 | What is the latest stable version of Python? | Router → web directly |
| 8 | Explain the Reflexion agent architecture and how it differs from ReAct. | KB (related) → irrelevant docs → web fallback |
| 9 | What is the best chunk size recommended for legal documents in RAG? | KB → partial → web / retry loop |
| 10 | What is Self-RAG and which reflection tokens does it use? | KB partly → answer check → possible retry loop |

Actual paths depend on the LLM; the table reports what really happened.

`tests/test_graph.py` (offline, mock `ask`, `retriever` and `web_search_tool` with `monkeypatch`):
- Router returns `web_search` → path skips retrieval and grading.
- Grader returns `NONE` → graph goes to web search.
- Checker always fails → loop stops after `max_retries` (assert `retries == max_retries + 1` and the run terminates).
- Checker passes on first try → no retry.

---

## 8. Acceptance criteria

- [ ] `python -m venv` + `pip install -r requirements.txt` works from scratch.
- [ ] The app starts with `streamlit run app.py` and shows a clear error if `OLLAMA_API_KEY` is missing.
- [ ] All LLM calls go to `https://ollama.com` with `Authorization: Bearer $OLLAMA_API_KEY`; the key is only read from the environment/`.env`.
- [ ] Graph has the 6 required nodes, normal edges, 3 conditional edges, `START`, `END`, shared `GraphState`, and a bounded retry loop.
- [ ] Each node appends a human-readable entry to `state["path"]`; the UI shows it live.
- [ ] Different queries visibly take different paths (KB, direct web, web fallback after grading, retry loop).
- [ ] The retry loop always terminates (`max_retries` + `recursion_limit`).
- [ ] `run_tests.py` executes 10 questions and writes `results.csv`.
- [ ] `.env` is git-ignored; no secrets in code or logs.

---

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| `401 Unauthorized` | Check `OLLAMA_API_KEY` in `.env`; generate a new key at https://ollama.com/settings/keys |
| `404 model not found` | On the hosted API use the name **without** the cloud suffix (`gpt-oss:120b`, not `gpt-oss:120b-cloud`). List models: `curl -H "Authorization: Bearer $OLLAMA_API_KEY" https://ollama.com/api/tags` |
| `429 Too Many Requests` | Usage limit reached; wait, or switch `OLLAMA_MODEL` to a smaller model such as `gpt-oss:20b` |
| First run is slow | The embedding model (`all-MiniLM-L6-v2`) is downloaded once |
| Web search returns nothing | The DDGS fallback kicks in automatically; check your network |
| Mermaid PNG missing | Rendering needs internet access (mermaid.ink); the Mermaid text is shown instead |

---

## 10. Optional extensions
- Add `MemorySaver` checkpointer + `thread_id` for chat memory across turns.
- Add a local Ollama fallback (`OLLAMA_HOST=http://localhost:11434`, no auth header) when the cloud limit is reached.
- Add a document-upload page to extend the KB from PDFs.
