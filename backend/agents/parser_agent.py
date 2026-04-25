"""
Parser Agent (uses gpt-4o-mini)
Responsibility: Extract a structured career profile from raw resume text.
Output saved to ctx.career_profile
"""

import json

from .base_agent import BaseAgent, ReasoningContext
from .llm_router import call_mini_json


class ParserAgent(BaseAgent):
    name = "parser"

    def run(self, ctx: ReasoningContext, **kwargs) -> ReasoningContext:
        ctx.log(self.name, "start", "Extracting structured career profile")

        resume_snippet = ctx.resume_raw[:2600]
        extracted_skills = ctx.resume_sections.get("skills", [])
        projects = ctx.resume_sections.get("projects", "")[:1200]
        experience = ctx.resume_sections.get("experience", "")[:1200]
        summary = ctx.resume_sections.get("summary", "")[:500]

        system = """You are a resume parsing engine.
Use the explicit resume sections and extracted skills as primary evidence.
If tools like LangChain, OpenAI, RAG, FAISS, Streamlit, LlamaIndex, Hugging Face,
or similar AI-stack technologies appear in the resume, include them when relevant.
Return ONLY valid JSON with these exact keys:
{
  "name": "string",
  "current_role": "string or null",
  "years_experience": number,
  "education_level": "high_school|bachelor|master|phd",
  "education_field": "string",
  "top_skills": ["max 12 most important skills"],
  "domains": ["domains of expertise e.g. backend, ml, data, devops"],
  "notable_projects": ["max 3 project names with 1 line description"],
  "career_level": "fresher|junior|mid|senior",
  "strengths": ["max 3 strongest areas based on resume evidence"],
  "red_flags": ["any gaps, short tenures, missing info"]
}"""

        user = f"""Extract career profile from this resume.

EXTRACTED SKILLS:
{json.dumps(extracted_skills)}

PROJECTS SECTION:
{projects}

EXPERIENCE SECTION:
{experience}

SUMMARY SECTION:
{summary}

RAW RESUME SNIPPET:
{resume_snippet}"""
        parsed, tokens = call_mini_json(system, user, max_tokens=350, retry_max_tokens=520)

        merged_skills = []
        for skill in parsed.get("top_skills", []) + extracted_skills:
            normalized = skill.strip()
            if normalized and normalized.lower() not in [s.lower() for s in merged_skills]:
                merged_skills.append(normalized)
        parsed["top_skills"] = merged_skills[:12]

        ctx.career_profile = parsed
        ctx.token_usage["mini"] += tokens
        ctx.log(
            self.name,
            "done",
            f"Profile extracted. Level: {ctx.career_profile.get('career_level')}",
        )
        return ctx
