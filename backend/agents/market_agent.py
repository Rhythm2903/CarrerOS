"""
Market Intelligence Agent (uses gpt-4o-mini)
Responsibility: Enrich analysis with real-world market context.
This is the NOVEL component - no simple ChatGPT call does this.
Output saved to ctx.market_intel
"""

from .base_agent import BaseAgent, ReasoningContext
from .llm_router import call_full_json


class MarketAgent(BaseAgent):
    name = "market_intel"

    def run(self, ctx: ReasoningContext, **kwargs) -> ReasoningContext:
        ctx.log(self.name, "start", "Generating market intelligence")

        role = ctx.jd_decomposition.get("role_title", "software engineer")
        geography = ctx.jd_decomposition.get("geography", "india")
        domain = ctx.jd_decomposition.get("domain_focus", [])
        industry = ctx.jd_decomposition.get("industry", "tech")
        level = ctx.career_profile.get("career_level", "mid")
        tier1 = [g["skill"] for g in ctx.skill_gap_result.get("tier1_blockers", [])]

        system = """You are a tech industry market analyst and compensation researcher.
Estimate compensation based on this candidate's likely entry point for the role,
their evidence level in the resume, and how their gaps affect expected pay.
Be realistic, conservative, and avoid inflated salary bands that would create false expectations.
Default to India compensation unless the job clearly points to another geography.
Return ONLY valid JSON:
{
  "market_demand": "high|medium|low - for this role right now",
  "salary_range_inr_lpa": {"min": number, "max": number, "currency": "INR LPA"},
  "candidate_salary_estimate_inr_lpa": {"min": number, "max": number, "currency": "INR LPA"},
  "candidate_level_for_role": "entry|junior|mid|senior",
  "salary_reasoning": "string - why this candidate likely fits this band",
  "salary_caution": "string - what might keep the candidate at the lower end for now",
  "trending_skills_in_domain": ["top 5 skills trending in this domain in 2025"],
  "fastest_to_learn_blocker": "which tier1 gap can be fixed fastest and how",
  "competitive_candidates_have": ["3 things typical hired candidates have"],
  "interview_focus_areas": ["top 3 topics this company likely tests"],
  "red_ocean_vs_blue_ocean": "string - is this role saturated or in-demand niche?",
  "market_tip": "1 specific, actionable tip for this candidate to stand out"
}"""

        user = f"""Role: {role}
Job geography: {geography}
Domain focus: {domain}
Industry: {industry}
Candidate level: {level}
Tier-1 skill gaps to address: {tier1}
Biggest strength to highlight: {ctx.skill_gap_result.get("biggest_strength_to_highlight", "")}
Experience gap: {ctx.skill_gap_result.get("experience_gap", "")}
Resume strengths: {ctx.skill_gap_result.get("resume_strength_signals", [])}

Provide 2025 market intelligence for this specific candidate and role.
If the candidate is entry-level or clearly underqualified for the target role, keep the salary estimate conservative and practical."""

        parsed, tokens = call_full_json(system, user, max_tokens=750, retry_max_tokens=1000)
        ctx.market_intel = parsed
        ctx.token_usage["full"] += tokens
        ctx.log(
            self.name,
            "done",
            f"Market demand: {ctx.market_intel.get('market_demand')}",
        )
        return ctx
