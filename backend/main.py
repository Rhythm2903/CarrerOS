import os
import sys
import uuid
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root))

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.agents.orchestrator import run_full_analysis
from backend.config import (
    get_embedding_provider,
    get_env_path,
    get_gemini_api_key,
    get_groq_api_key,
    get_llm_provider,
    get_model_tier,
    get_openai_api_key,
    preview_key,
)
from backend.models.schemas import (
    AnalyzeJobRequest,
    ChatRequest,
    ChatResponse,
    UploadResponse,
)
from backend.rag_pipeline.rag_engine import delete_session, index_resume
from backend.services.llm_service import chat_with_resume
from backend.services.resume_parser import extract_text_from_pdf, parse_resume

app = FastAPI(
    title="AI Career Copilot API - Agentic v2",
    description="Multi-agent orchestrated career intelligence system",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

session_store: dict = {}


@app.get("/")
def root():
    return {
        "message": "AI Career Copilot v2 - Agentic Orchestrator",
        "version": "2.0.0",
    }


@app.get("/debug-env")
def debug_env():
    provider = get_llm_provider()
    key_map = {
        "gemini": get_gemini_api_key(),
        "groq": get_groq_api_key(),
        "openai": get_openai_api_key(),
    }
    key = key_map.get(provider, "")
    return {
        "key_found": bool(key),
        "key_preview": preview_key(key) if key else "NOT FOUND",
        "llm_provider": provider,
        "embedding_provider": get_embedding_provider(),
        "model_tier": get_model_tier(),
        "env_file": str(get_env_path()),
        "env_exists": get_env_path().exists(),
    }


@app.post("/upload_resume", response_model=UploadResponse)
async def upload_resume(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files supported.")
    file_bytes = await file.read()
    if len(file_bytes) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Max file size 5MB.")
    try:
        raw_text = extract_text_from_pdf(file_bytes)
        resume_data = parse_resume(raw_text)
        session_id = str(uuid.uuid4())[:8]
        chunks_indexed = index_resume(session_id, raw_text)
        session_store[session_id] = {
            "resume_data": resume_data,
            "resume_raw": raw_text,
        }
        return UploadResponse(
            session_id=session_id,
            message="Resume uploaded and indexed successfully.",
            chunks_indexed=chunks_indexed,
            resume_data=resume_data,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Resume processing failed: {str(exc)}"
        ) from exc


@app.post("/analyze_job")
async def analyze_job(request: AnalyzeJobRequest):
    if request.session_id not in session_store:
        raise HTTPException(status_code=404, detail="Session not found. Upload resume first.")
    sess = session_store[request.session_id]
    try:
        result = run_full_analysis(
            session_id=request.session_id,
            resume_raw=sess["resume_raw"],
            resume_sections=sess["resume_data"],
            job_description=request.job_description,
        )
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(exc)}") from exc


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if request.session_id not in session_store:
        raise HTTPException(status_code=404, detail="Session not found. Upload resume first.")
    try:
        response, tokens = chat_with_resume(
            request.session_id, request.message, request.chat_history
        )
        return ChatResponse(response=response, session_id=request.session_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(exc)}") from exc


@app.delete("/session/{session_id}")
def delete_user_session(session_id: str):
    delete_session(session_id)
    session_store.pop(session_id, None)
    return {"message": "Session deleted."}
