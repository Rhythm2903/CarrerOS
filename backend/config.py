from pathlib import Path

from dotenv import dotenv_values

_root = Path(__file__).resolve().parents[1]
_env_path = _root / ".env"


def get_env_values() -> dict:
    values = dotenv_values(_env_path)
    return {str(k): str(v) for k, v in values.items() if k is not None and v is not None}


def get_openai_api_key() -> str:
    return get_env_values().get("OPENAI_API_KEY", "")


def get_gemini_api_key() -> str:
    return get_env_values().get("GEMINI_API_KEY", "")


def get_groq_api_key() -> str:
    return get_env_values().get("GROQ_API_KEY", "")


def get_chroma_persist_dir() -> str:
    return get_env_values().get("CHROMA_PERSIST_DIR", "./chroma_db")


def get_llm_provider() -> str:
    provider = get_env_values().get("LLM_PROVIDER", "gemini").strip().lower()
    return provider if provider in {"gemini", "groq", "openai"} else "gemini"


def get_embedding_provider() -> str:
    provider = get_env_values().get("EMBEDDING_PROVIDER", "gemini").strip().lower()
    return provider if provider in {"gemini", "openai"} else "gemini"


def get_fallback_provider() -> str:
    provider = get_env_values().get("FALLBACK_PROVIDER", "groq").strip().lower()
    return provider if provider in {"groq", "openai", "none"} else "groq"


def get_model_tier() -> str:
    tier = get_env_values().get("MODEL_TIER", "budget").strip().lower()
    return tier if tier in {"budget", "balanced", "premium"} else "budget"


def get_env_path() -> Path:
    return _env_path


def preview_key(key: str) -> str:
    return key[:8] + "..." + key[-4:] if len(key) > 12 else key
