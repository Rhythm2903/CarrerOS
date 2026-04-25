import numpy as np
import faiss
import requests
import time

from backend.config import get_embedding_provider, get_gemini_api_key, get_openai_api_key

OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 1536
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
GEMINI_RETRY_DELAYS = [1.0, 2.5, 5.0]

_store: dict = {}


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def _embed_openai(texts: list[str]) -> np.ndarray:
    from openai import OpenAI

    client = OpenAI(api_key=get_openai_api_key())
    response = client.embeddings.create(model=OPENAI_EMBEDDING_MODEL, input=texts)
    vectors = [item.embedding for item in response.data]
    return np.array(vectors, dtype="float32")


def _embed_gemini(texts: list[str], task_type: str) -> np.ndarray:
    vectors = []
    for text in texts:
        payload = None
        last_error = None
        for delay in [0.0] + GEMINI_RETRY_DELAYS:
            if delay:
                time.sleep(delay)
            response = requests.post(
                f"{GEMINI_API_BASE}/{GEMINI_EMBEDDING_MODEL}:embedContent",
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": get_gemini_api_key(),
                },
                json={
                    "taskType": task_type,
                    "outputDimensionality": EMBEDDING_DIM,
                    "content": {"parts": [{"text": text}]},
                },
                timeout=120,
            )
            if response.status_code == 429:
                last_error = requests.HTTPError(
                    f"429 Client Error: Too Many Requests for url: {response.url}",
                    response=response,
                )
                continue
            response.raise_for_status()
            payload = response.json()
            break
        if payload is None:
            raise last_error or RuntimeError("Gemini embedding request failed")
        vectors.append(payload["embedding"]["values"])
    return _normalize_rows(np.array(vectors, dtype="float32"))


def _embed(texts: list[str], task_type: str) -> np.ndarray:
    if get_embedding_provider() == "openai":
        return _embed_openai(texts)
    return _embed_gemini(texts, task_type)


def chunk_text(text: str, chunk_size: int = 350, overlap: int = 70) -> list:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(' '.join(words[start:end]))
        start += chunk_size - overlap
    return chunks


def index_resume(session_id: str, resume_text: str) -> int:
    chunks = chunk_text(resume_text)
    embeddings = _embed(chunks, "RETRIEVAL_DOCUMENT")
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings)
    _store[session_id] = {"index": index, "chunks": chunks}
    return len(chunks)


def retrieve_context(
    session_id: str, query: str, top_k: int = 5, max_chars: int = 1200
) -> str:
    if session_id not in _store:
        return ""
    store = _store[session_id]
    query_vec = _embed([query], "RETRIEVAL_QUERY")
    k = min(top_k, len(store["chunks"]))
    _, indices = store["index"].search(query_vec, k)
    results = [store["chunks"][i] for i in indices[0] if i < len(store["chunks"])]
    return '\n\n'.join(results)[:max_chars]


def delete_session(session_id: str):
    _store.pop(session_id, None)
