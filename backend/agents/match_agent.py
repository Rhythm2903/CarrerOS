"""
Match Agent (uses gpt-4o - synthesis quality matters here)
Responsibility: Score candidate vs job using career_profile + jd_decomposition.
Does NOT re-read the raw resume - uses parsed ctx data only.
Output saved to ctx.match_result
"""

import json
import re

from .base_agent import BaseAgent, ReasoningContext
from .llm_router import call_full_json


AI_STACK_SKILLS = {
    "llm",
    "llms",
    "large language models",
    "generative ai",
    "rag",
    "ai agents",
    "agentic ai",
    "prompt engineering",
    "langchain",
    "langgraph",
    "llamaindex",
    "hugging face",
    "huggingface",
    "openai api",
    "openai",
    "gemini api",
    "faiss",
    "chromadb",
    "pinecone",
    "weaviate",
    "qdrant",
    "embeddings",
    "semantic search",
    "transformers",
    "fastapi",
}


def _normalize(skill: str) -> str:
    return " ".join(skill.lower().strip().replace("/", " ").replace("-", " ").split())


SKILL_VARIANTS = {
    "openai api": ["openai api", "openai", "gpt api", "chatgpt api"],
    "gemini api": ["gemini api", "gemini"],
    "hugging face": ["hugging face", "huggingface"],
    "llm": ["llm", "llms", "large language model", "large language models"],
    "generative ai": ["generative ai", "genai"],
    "ai agents": ["ai agents", "agentic ai", "agents"],
    "rag": ["rag", "retrieval augmented generation"],
    "prompt engineering": ["prompt engineering", "prompt design"],
}

SOFT_SKILL_MARKERS = {
    "leadership": ["leadership", "led", "lead", "managed", "owned"],
    "team coordination": ["team coordination", "coordinated", "collaborated", "cross-functional", "team"],
    "communication": ["communication", "presented", "stakeholder", "explained"],
}

GENERIC_SKILL_VARIANTS = {
    "machine learning": ["machine learning", "ml"],
    "artificial intelligence": ["artificial intelligence", "ai"],
    "deep learning": ["deep learning", "dl"],
    "javascript": ["javascript", "js"],
    "typescript": ["typescript", "ts"],
    "react": ["react", "reactjs", "react.js"],
    "node": ["node", "nodejs", "node.js"],
    "postgresql": ["postgresql", "postgres", "psql"],
    "mysql": ["mysql", "my sql"],
    "mongodb": ["mongodb", "mongo db", "mongo"],
    "rest api": ["rest api", "restful api", "rest apis"],
    "graphql": ["graphql", "graph ql"],
    "ci cd": ["ci cd", "ci/cd", "continuous integration", "continuous deployment"],
    "c++": ["c++", "cpp", "c plus plus"],
    "c#": ["c#", "c sharp"],
    ".net": [".net", "dotnet", "asp.net", "asp net"],
    "aws": ["aws", "amazon web services"],
    "gcp": ["gcp", "google cloud", "google cloud platform"],
}


def _variants(skill: str) -> list[str]:
    normalized = _normalize(skill)
    variants = set(SKILL_VARIANTS.get(normalized, []))
    variants.update(GENERIC_SKILL_VARIANTS.get(normalized, []))
    variants.add(normalized)
    if normalized.endswith(" api"):
        variants.add(normalized.replace(" api", ""))
    if normalized.endswith("s"):
        variants.add(normalized[:-1])
    return [variant for variant in variants if variant]


def _token_match(variant: str, resume_text: str) -> bool:
    tokens = [token for token in variant.split() if token]
    if len(tokens) < 2:
        return False
    return all(re.search(r"\b" + re.escape(token) + r"\b", resume_text) for token in tokens)


def _resume_contains_skill(skill: str, resume_text: str, resume_skills: list[str]) -> bool:
    normalized_resume_skills = {_normalize(item) for item in resume_skills}
    for variant in _variants(skill):
        if variant in normalized_resume_skills:
            return True
        pattern = r"\b" + re.escape(variant) + r"\b"
        if re.search(pattern, resume_text):
            return True
        if _token_match(variant, resume_text):
            return True
    return False


def _find_soft_strengths(resume_text: str) -> list[str]:
    found = []
    for label, variants in SOFT_SKILL_MARKERS.items():
        if any(re.search(r"\b" + re.escape(variant) + r"\b", resume_text) for variant in variants):
            found.append(label)
    return found


def _is_ai_role(jd: dict) -> bool:
    role_text = " ".join(
        [
            jd.get("role_title", ""),
            jd.get("role_family", ""),
            " ".join(jd.get("domain_focus", [])),
            " ".join(jd.get("must_have_skills", [])),
        ]
    ).lower()
    markers = ["ai", "ml", "machine learning", "llm", "genai", "generative", "rag"]
    return any(marker in role_text for marker in markers)


class MatchAgent(BaseAgent):
    name = "matcher"

    def run(self, ctx: ReasoningContext, **kwargs) -> ReasoningContext:
        ctx.log(self.name, "start", "Computing match score")

        system = """You are a senior technical recruiter.
Only count skills as matched if they are genuinely relevant to this role family and supported by resume evidence.
Do not over-credit generic tools or adjacent technologies unless they clearly transfer.
Return ONLY valid JSON:
{
  "match_score": integer 0-100,
  "matched_skills": ["skills from candidate matching JD must-haves"],
  "missing_critical": ["must-have skills the candidate lacks"],
  "missing_preferred": ["nice-to-have skills the candidate lacks"],
  "strengths_for_role": ["2-3 specific reasons candidate is strong for this role"],
  "concerns": ["2-3 honest gaps or risks"],
  "hiring_signal": "strong_yes|yes|maybe|no",
  "one_line_verdict": "string"
}"""

        user = f"""CANDIDATE PROFILE:
{json.dumps(ctx.career_profile, separators=(",", ":"))}

EXTRACTED RESUME SKILLS:
{json.dumps(ctx.resume_sections.get("skills", []), separators=(",", ":"))}

PROJECTS SECTION:
{ctx.resume_sections.get("projects", "")[:800]}

JD REQUIREMENTS:
{json.dumps(ctx.jd_decomposition, separators=(",", ":"))}

RELEVANT RESUME CONTEXT:
{ctx.retrieved_chunks[:700]}

Score the match precisely. Be conservative and honest about gaps."""

        parsed, tokens = call_full_json(system, user, max_tokens=420, retry_max_tokens=620)

        resume_skills = ctx.resume_sections.get("skills", [])
        resume_text = " ".join(
            [
                ctx.resume_raw.lower(),
                ctx.resume_sections.get("projects", "").lower(),
                ctx.resume_sections.get("experience", "").lower(),
            ]
        )
        normalized_resume = {_normalize(skill): skill for skill in resume_skills}
        normalized_matched = {_normalize(skill) for skill in parsed.get("matched_skills", [])}
        must_have = ctx.jd_decomposition.get("must_have_skills", [])
        nice_to_have = ctx.jd_decomposition.get("nice_to_have_skills", [])

        deterministic_matched = []
        deterministic_missing_critical = []
        deterministic_missing_preferred = []

        for skill in must_have:
            if _resume_contains_skill(skill, resume_text, resume_skills):
                deterministic_matched.append(skill)
            else:
                deterministic_missing_critical.append(skill)

        for skill in nice_to_have:
            if _resume_contains_skill(skill, resume_text, resume_skills):
                if skill not in deterministic_matched:
                    deterministic_matched.append(skill)
            else:
                deterministic_missing_preferred.append(skill)

        if _is_ai_role(ctx.jd_decomposition):
            for normalized, original in normalized_resume.items():
                if normalized in AI_STACK_SKILLS and normalized not in normalized_matched:
                    deterministic_matched.append(original)
                    normalized_matched.add(normalized)

        merged_matched = []
        for skill in deterministic_matched + parsed.get("matched_skills", []):
            if _normalize(skill) not in {_normalize(item) for item in merged_matched}:
                merged_matched.append(skill)

        parsed["matched_skills"] = merged_matched[:15]
        parsed["missing_critical"] = deterministic_missing_critical[:12]
        parsed["missing_preferred"] = deterministic_missing_preferred[:12]

        soft_strengths = _find_soft_strengths(resume_text)
        if soft_strengths:
            for item in soft_strengths:
                strength_line = f"Resume shows evidence of {item}"
                if strength_line not in parsed.get("strengths_for_role", []):
                    parsed.setdefault("strengths_for_role", []).append(strength_line)
            filtered_concerns = []
            for concern in parsed.get("concerns", []):
                lowered = concern.lower()
                if any(marker in lowered for marker in soft_strengths):
                    continue
                filtered_concerns.append(concern)
            parsed["concerns"] = filtered_concerns[:3]

        parsed["strengths_for_role"] = parsed.get("strengths_for_role", [])[:4]
        ctx.match_result = parsed
        ctx.token_usage["full"] += tokens
        ctx.log(
            self.name,
            "done",
            f"Score: {ctx.match_result.get('match_score')} | "
            f"Signal: {ctx.match_result.get('hiring_signal')}",
        )
        return ctx
