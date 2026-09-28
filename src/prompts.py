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
