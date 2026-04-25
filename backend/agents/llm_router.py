import json
import time
from typing import Optional

import requests
from openai import OpenAI

from backend.config import (
    get_fallback_provider,
    get_gemini_api_key,
    get_groq_api_key,
    get_llm_provider,
    get_model_tier,
    get_openai_api_key,
)

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
GROQ_API_BASE = "https://api.groq.com/openai/v1"
OPENAI_API_BASE = "https://api.openai.com/v1"
OPENAI_MINI = "gpt-4o-mini"
OPENAI_FULL = "gpt-4o"
GROQ_MINI = "llama-3.1-8b-instant"
GROQ_FULL = "llama-3.3-70b-versatile"
GEMINI_RETRY_DELAYS = [1.0, 2.5, 5.0]


def _openai_client() -> OpenAI:
    return OpenAI(api_key=get_openai_api_key())


def _groq_client() -> OpenAI:
    return OpenAI(api_key=get_groq_api_key(), base_url=GROQ_API_BASE)


def get_synthesis_model() -> str:
    provider = get_llm_provider()
    tier = get_model_tier()
    if provider == "gemini":
        if tier == "premium":
            return "gemini-2.5-flash"
        return "gemini-2.5-flash-lite"
    if provider == "groq":
        return GROQ_FULL if tier in {"balanced", "premium"} else GROQ_MINI
    if tier == "premium":
        return OPENAI_FULL
    return OPENAI_MINI


def get_routing_model() -> str:
    provider = get_llm_provider()
    if provider == "gemini":
        return "gemini-2.5-flash-lite"
    if provider == "groq":
        return GROQ_MINI
    return OPENAI_MINI


def _fallback_chat_completion(
    messages: list[dict], temperature: float, max_tokens: int, json_mode: bool
) -> tuple[str, int]:
    fallback = get_fallback_provider()
    if fallback == "groq":
        return _call_openai_compatible(
            _groq_client(),
            GROQ_MINI,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
        )
    if fallback == "openai":
        return _call_openai_compatible(
            _openai_client(),
            OPENAI_MINI,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
        )
    raise


def _messages_to_prompt(messages: list[dict]) -> str:
    role_map = {"system": "SYSTEM", "user": "USER", "assistant": "ASSISTANT"}
    lines = []
    for message in messages:
        role = role_map.get(message.get("role", "user"), "USER")
        lines.append(f"{role}:\n{message.get('content', '')}")
    lines.append("ASSISTANT:")
    return "\n\n".join(lines)


def _extract_gemini_text(payload: dict) -> str:
    candidates = payload.get("candidates", [])
    if not candidates:
        raise ValueError(f"Gemini returned no candidates: {payload}")
    parts = candidates[0].get("content", {}).get("parts", [])
    texts = [part.get("text", "") for part in parts if part.get("text")]
    if not texts:
        raise ValueError(f"Gemini returned no text parts: {payload}")
    return "".join(texts)


def _extract_gemini_tokens(payload: dict) -> int:
    usage = payload.get("usageMetadata", {})
    return int(usage.get("totalTokenCount", 0))


def _call_gemini(
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    json_mode: bool,
    system: Optional[str] = None,
) -> tuple[str, int]:
    body: dict = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        },
    }
    if system:
        body["system_instruction"] = {"parts": [{"text": system}]}
    if json_mode:
        body["generationConfig"]["responseMimeType"] = "application/json"
        body["generationConfig"]["responseSchema"] = {"type": "object"}

    last_error = None
    for delay in [0.0] + GEMINI_RETRY_DELAYS:
        if delay:
            time.sleep(delay)
        response = requests.post(
            f"{GEMINI_API_BASE}/{model}:generateContent",
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": get_gemini_api_key(),
            },
            json=body,
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
        return _extract_gemini_text(payload), _extract_gemini_tokens(payload)
    raise last_error or RuntimeError("Gemini request failed without a response")


def _call_openai_compatible(
    client: OpenAI,
    model: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    json_mode: bool,
) -> tuple[str, int]:
    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )
    return response.choices[0].message.content, response.usage.total_tokens


def call_mini(
    system: str, user: str, json_mode: bool = False, max_tokens: int = 450
) -> tuple[str, int]:
    provider = get_llm_provider()
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    if provider == "gemini":
        try:
            return _call_gemini(
                prompt=user,
                model=get_routing_model(),
                temperature=0.1,
                max_tokens=max_tokens,
                json_mode=json_mode,
                system=system,
            )
        except requests.HTTPError as exc:
            if getattr(exc.response, "status_code", None) == 429:
                return _fallback_chat_completion(
                    messages, temperature=0.1, max_tokens=max_tokens, json_mode=json_mode
                )
            raise
    if provider == "groq":
        return _call_openai_compatible(
            _groq_client(),
            get_routing_model(),
            messages,
            temperature=0.1,
            max_tokens=max_tokens,
            json_mode=json_mode,
        )
    return _call_openai_compatible(
        _openai_client(),
        get_routing_model(),
        messages,
        temperature=0.1,
        max_tokens=max_tokens,
        json_mode=json_mode,
    )


def call_full(
    system: str, user: str, json_mode: bool = False, max_tokens: int = 700
) -> tuple[str, int]:
    provider = get_llm_provider()
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    if provider == "gemini":
        try:
            return _call_gemini(
                prompt=user,
                model=get_synthesis_model(),
                temperature=0.3,
                max_tokens=max_tokens,
                json_mode=json_mode,
                system=system,
            )
        except requests.HTTPError as exc:
            if getattr(exc.response, "status_code", None) == 429:
                return _fallback_chat_completion(
                    messages, temperature=0.3, max_tokens=max_tokens, json_mode=json_mode
                )
            raise
    if provider == "groq":
        return _call_openai_compatible(
            _groq_client(),
            get_synthesis_model(),
            messages,
            temperature=0.3,
            max_tokens=max_tokens,
            json_mode=json_mode,
        )
    return _call_openai_compatible(
        _openai_client(),
        get_synthesis_model(),
        messages,
        temperature=0.3,
        max_tokens=max_tokens,
        json_mode=json_mode,
    )


def call_chat(messages: list, max_tokens: int = 600) -> tuple[str, int]:
    provider = get_llm_provider()
    if provider == "gemini":
        try:
            return _call_gemini(
                prompt=_messages_to_prompt(messages),
                model=get_routing_model(),
                temperature=0.4,
                max_tokens=max_tokens,
                json_mode=False,
            )
        except requests.HTTPError as exc:
            if getattr(exc.response, "status_code", None) == 429:
                return _fallback_chat_completion(
                    messages, temperature=0.4, max_tokens=max_tokens, json_mode=False
                )
            raise
    if provider == "groq":
        return _call_openai_compatible(
            _groq_client(),
            get_routing_model(),
            messages,
            temperature=0.4,
            max_tokens=max_tokens,
            json_mode=False,
        )
    return _call_openai_compatible(
        _openai_client(),
        get_routing_model(),
        messages,
        temperature=0.4,
        max_tokens=max_tokens,
        json_mode=False,
    )


def _extract_json_object(content: str) -> dict:
    candidates = []
    stripped = content.strip()
    candidates.append(stripped)

    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            candidates.append("\n".join(lines[1:-1]).strip())

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(stripped[start : end + 1].strip())

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return json.loads(candidates[-1])


def _repair_json(content: str, max_tokens: int = 500) -> tuple[dict, int]:
    system = """You repair malformed JSON.
Return ONLY one valid JSON object.
Do not add commentary, markdown, or code fences."""
    user = f"""Repair this into one valid JSON object.
Preserve the original meaning and keys as much as possible.

MALFORMED JSON:
{content}"""
    repaired, tokens = call_mini(system, user, json_mode=False, max_tokens=max_tokens)
    return _extract_json_object(repaired), tokens


def _call_json_with_retries(
    caller, system: str, user: str, max_tokens: int, retry_max_tokens: int
) -> tuple[dict, int]:
    total_tokens = 0

    content, tokens = caller(system, user, json_mode=True, max_tokens=max_tokens)
    total_tokens += tokens
    try:
        return _extract_json_object(content), total_tokens
    except json.JSONDecodeError:
        shorter_user = (
            user
            + "\n\nImportant: return one compact valid JSON object only. Keep values short enough to fit in one response."
        )
        content, tokens = caller(
            system, shorter_user, json_mode=True, max_tokens=retry_max_tokens
        )
        total_tokens += tokens
        try:
            return _extract_json_object(content), total_tokens
        except json.JSONDecodeError:
            try:
                repaired, repair_tokens = _repair_json(content, max_tokens=retry_max_tokens)
                total_tokens += repair_tokens
                return repaired, total_tokens
            except json.JSONDecodeError:
                final_user = (
                    user
                    + "\n\nReturn a minified valid JSON object only. No markdown. No explanations. Keep arrays short."
                )
                content, tokens = caller(
                    system, final_user, json_mode=True, max_tokens=retry_max_tokens + 120
                )
                total_tokens += tokens
                return _extract_json_object(content), total_tokens


def call_mini_json(
    system: str, user: str, max_tokens: int = 450, retry_max_tokens: int = 650
) -> tuple[dict, int]:
    return _call_json_with_retries(call_mini, system, user, max_tokens, retry_max_tokens)


def call_full_json(
    system: str, user: str, max_tokens: int = 700, retry_max_tokens: int = 950
) -> tuple[dict, int]:
    return _call_json_with_retries(call_full, system, user, max_tokens, retry_max_tokens)
