import pandas as pd
from src.graph import run_graph

QUESTIONS = [
    "What is a conditional edge in LangGraph?",
    "How does RAG help reduce hallucinations?",
    "What is a checkpointer in LangGraph and why is it useful?",
    "What is the difference between a vector database and SQL?",
    "What is the capital of Australia?",
    "Who is the current CEO of Anthropic?",
    "What is the latest stable version of Python?",
    "Explain the Reflexion agent architecture and how it differs from ReAct.",
    "What is the best chunk size recommended for legal documents in RAG?",
    "What is Self-RAG and which reflection tokens does it use?"
]

def main():
    results = []
    print("Running 10 questions...")
    for i, q in enumerate(QUESTIONS):
        print(f"[{i+1}/10] {q}")
        try:
            state = run_graph(q)
            results.append({
                "Question": q,
                "Route": state.get("route", ""),
                "Path": " -> ".join(state.get("path", [])),
                "Retries": state.get("retries", 0),
                "Grounded": state.get("grounded", False),
                "Useful": state.get("useful", False),
                "Answer": state.get("generation", "")[:100] + "..."
            })
            if i < 5:
                print(f"  Path: {state.get('path', [])}")
        except Exception as e:
            print(f"Error on {q}: {e}")
            
    df = pd.DataFrame(results)
    print(df.to_string())
    df.to_csv("results.csv", index=False)
    print("Saved results.csv")

if __name__ == "__main__":
    main()
