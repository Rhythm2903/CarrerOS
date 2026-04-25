"""
JD Decomposition Agent
Responsibility: Break job description into weighted requirement buckets.
Output saved to ctx.jd_decomposition
"""

from .base_agent import BaseAgent, ReasoningContext
from .llm_router import call_full_json


class JDAgent(BaseAgent):
    name = "jd_decomposer"

    def run(
        self, ctx: ReasoningContext, job_description: str = "", **kwargs
    ) -> ReasoningContext:
        ctx.log(self.name, "start", "Decomposing job description")

        jd_snippet = job_description[:2600]

        system = """You are a technical recruiter specializing in role calibration.
Extract only skills and requirements that are truly relevant to succeeding in this specific job.
Prefer concrete role skills over generic workplace terms.
For AI/ML/LLM internships and early-career roles, explicitly prioritize stack skills like:
LLMs, Generative AI, RAG, AI Agents, Prompt Engineering, LangChain, Hugging Face,
OpenAI API, Gemini API, vector databases, and model-serving APIs when the JD suggests that work.
Return ONLY valid JSON:
{
  "role_title": "string",
  "role_family": "frontend|backend|fullstack|data|ml|ai_ml|devops|qa|security|product|other",
  "seniority": "junior|mid|senior|lead",
  "required_years_experience": "string",
  "must_have_skills": ["skills explicitly required"],
  "nice_to_have_skills": ["skills mentioned as preferred"],
  "domain_focus": ["primary technical domains"],
  "soft_skills_needed": ["communication, leadership, etc"],
  "key_responsibilities": ["max 4 core responsibilities"],
  "geography": "india|us|europe|remote_global|unknown",
  "company_type": "startup|mid-size|enterprise|unknown",
  "industry": "fintech|edtech|healthtech|ecommerce|saas|other"
}"""

        user = f"Decompose this job description:\n\n{jd_snippet}"
        parsed, tokens = call_full_json(system, user, max_tokens=700, retry_max_tokens=950)
        ctx.jd_decomposition = parsed
        ctx.token_usage["full"] += tokens
        ctx.log(
            self.name,
            "done",
            f"JD decomposed. Role: {ctx.jd_decomposition.get('role_title')}",
        )
        return ctx
