"""
llm_service.py - now just a thin wrapper for the chat endpoint.
Full analysis is handled by the orchestrator.
"""

from backend.agents.llm_router import call_chat
from backend.rag_pipeline.rag_engine import retrieve_context


def chat_with_resume(
    session_id: str, user_message: str, chat_history: list
) -> tuple[str, int]:
    """
    RAG-powered chat using gpt-4o-mini (cheap for conversational turns).
    Returns (response_text, tokens_used).
    """

    context = retrieve_context(session_id, user_message, top_k=2, max_chars=700)

    system = f"""You are an AI career advisor. Answer using the candidate's resume context.
Be specific, concise, and actionable. Max 3 paragraphs.

    RESUME CONTEXT:
{context}"""

    messages = [{"role": "system", "content": system}]
    for msg in chat_history[-2:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})

    return call_chat(messages, max_tokens=220)
