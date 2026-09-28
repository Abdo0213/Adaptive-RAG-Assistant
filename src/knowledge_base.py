import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from .config import TOP_K

KB = [
    ("langgraph_state", "LangGraph uses a shared State to pass data between nodes. The State is defined as a TypedDict and often uses reducers like operator.add to append updates."),
    ("langgraph_nodes", "Nodes in LangGraph are python functions that take the state as input and return a partial state update. Each node performs a specific task in the workflow."),
    ("langgraph_edges", "Edges connect nodes in a LangGraph, defining the flow of execution. A standard edge always transitions from one specific node to another."),
    ("langgraph_conditional_edges", "Conditional edges use a routing function to decide the next node dynamically based on the current state. They enable complex branching logic."),
    ("langgraph_cycles", "LangGraph supports cycles (loops) natively, unlike LangChain Expression Language (LCEL). Loops are essential for building iterative agentic behaviors."),
    ("langgraph_checkpointer", "Checkpointers save the state at each step, enabling human-in-the-loop interactions and time travel. The MemorySaver is a simple in-memory checkpointer."),
    ("rag_basics", "Retrieval-Augmented Generation (RAG) improves LLM responses by fetching relevant facts from an external knowledge base before generating an answer."),
    ("rag_hallucination", "RAG reduces hallucinations by grounding the LLM's answers in retrieved documents rather than relying solely on its internal training data."),
    ("chunking", "Chunking splits large documents into smaller, manageable pieces before embedding. Overlapping chunks help maintain context across boundaries."),
    ("embeddings", "Embeddings are vector representations of text that capture semantic meaning. Similar texts have embeddings that are close together in the vector space."),
    ("vector_databases", "Vector databases store and index embeddings for fast similarity search. They are fundamentally different from SQL databases, which are relational."),
    ("agents_react", "ReAct (Reasoning and Acting) is an agent paradigm where the LLM interleaves thought processes with tool use to solve complex tasks."),
    ("adaptive_rag", "Adaptive RAG dynamically routes queries to different retrieval strategies based on complexity, optionally falling back to web search if local data is insufficient."),
    ("prompt_engineering", "Prompt engineering is the practice of designing inputs to guide an LLM's output. Effective prompts are clear, specific, and often provide examples.")
]

KB_TOPICS = "LangGraph (state, nodes, edges, checkpointers), RAG, embeddings, vector databases, chunking, LLM agents, and prompt engineering"

_retriever = None

def get_retriever():
    global _retriever
    if _retriever is None:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        persist_dir = "./chroma_db"
        
        if not os.path.exists(persist_dir) or not os.listdir(persist_dir):
            texts = [text for _, text in KB]
            metadatas = [{"source": source, "origin": "kb"} for source, _ in KB]
            vectorstore = Chroma.from_texts(
                texts,
                embeddings,
                metadatas=metadatas,
                persist_directory=persist_dir
            )
        else:
            vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
            
        _retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    return _retriever
