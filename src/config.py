import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "https://ollama.com")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gpt-oss:120b")
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "2"))
TOP_K = int(os.environ.get("TOP_K", "3"))
WEB_IF_ANY_IRRELEVANT = os.environ.get("WEB_IF_ANY_IRRELEVANT", "false").lower() == "true"
WEB_MAX_RESULTS = int(os.environ.get("WEB_MAX_RESULTS", "4"))

def require_api_key():
    if not OLLAMA_API_KEY or OLLAMA_API_KEY == "your_api_key_here":
        raise RuntimeError("OLLAMA_API_KEY is missing. Add it to your .env file.")
