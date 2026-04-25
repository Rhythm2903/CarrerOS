from pydantic import BaseModel
from typing import Optional


class AnalyzeJobRequest(BaseModel):
    session_id: str
    job_description: str


class ChatRequest(BaseModel):
    session_id: str
    message: str
    chat_history: list[dict] = []


class UploadResponse(BaseModel):
    session_id: str
    message: str
    chunks_indexed: int
    resume_data: dict


class MatchResponse(BaseModel):
    match_score: int
    matched_skills: list[str]
    missing_skills: list[str]
    strengths: list[str]
    weaknesses: list[str]
    overall_verdict: str


class SkillGapResponse(BaseModel):
    technical_gaps: list[dict]
    soft_skill_gaps: list[dict]
    experience_gaps: list[str]
    candidate_level: str
    role_level_required: str
    gap_summary: str


class RoadmapResponse(BaseModel):
    roadmap_title: str
    total_duration: str
    phases: list[dict]
    quick_wins: list[str]
    milestone_projects: list[str]


class ChatResponse(BaseModel):
    response: str
    session_id: str
