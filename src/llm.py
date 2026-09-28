import os
import time
from ollama import Client
from .config import OLLAMA_HOST, OLLAMA_API_KEY, OLLAMA_MODEL

_client = None
last_model_used = None

def get_client():
    global _client
    if _client is None:
        _client = Client(
            host=OLLAMA_HOST,
            headers={"Authorization": "Bearer " + OLLAMA_API_KEY} if OLLAMA_API_KEY else {},
        )
    return _client

def ask(prompt: str, system: str = "You are a precise assistant. Follow the requested output format exactly.") -> str:
    global last_model_used
    client = get_client()
    last_model_used = OLLAMA_MODEL
    
    retries = 2
    backoff = 1
    
    for attempt in range(retries + 1):
        try:
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
        except Exception as e:
            if attempt == retries:
                raise e
            time.sleep(backoff)
            backoff *= 2
