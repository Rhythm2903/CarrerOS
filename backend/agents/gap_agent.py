"""
Skill Gap Agent (uses gpt-4o-mini - structured extraction)
Responsibility: Deep gap analysis with priority tiers.
Output saved to ctx.skill_gap_result
"""

import json

from .base_agent import BaseAgent, ReasoningContext
from .llm_router import call_full_json


class GapAgent(BaseAgent):
    name = "gap_analyzer"

    def run(self, ctx: ReasoningContext, **kwargs) -> ReasoningContext:
        ctx.log(self.name, "start", "Analyzing skill gaps")

        system = """You are a senior career gap analyst for technical hiring.
Judge severity based on missing must-haves, transferability of current strengths,
experience mismatch, and how fast the candidate can realistically close the gaps.
Return ONLY valid JSON:
{
  "tier1_blockers": [{"skill": "string", "why_critical": "string", "estimated_learn_time": "string"}],
  "tier2_differentiators": [{"skill": "string", "impact": "string", "estimated_learn_time": "string"}],
  "tier3_nice_to_have": ["skill names only, max 5"],
  "experience_gap": "string - 1 sentence on experience level mismatch if any",
  "biggest_strength_to_highlight": "string",
  "gap_severity_reason": "string - why this severity is justified",
  "resume_strength_signals": ["2-3 strongest positive signals from the resume for this role"],
  "gap_severity": "low|medium|high|critical"
}"""

        user = f"""CANDIDATE:
{json.dumps(ctx.career_profile, separators=(",", ":"))}

JD REQUIREMENTS:
{json.dumps(ctx.jd_decomposition, separators=(",", ":"))}

MATCH ANALYSIS (for context):
Missing critical: {ctx.match_result.get('missing_critical', [])}
Missing preferred: {ctx.match_result.get('missing_preferred', [])}
Strengths for role: {ctx.match_result.get('strengths_for_role', [])}
Concerns: {ctx.match_result.get('concerns', [])}

Identify gaps in 3 priority tiers. Be specific to this candidate and target role."""

        parsed, tokens = call_full_json(system, user, max_tokens=650, retry_max_tokens=900)
        ctx.skill_gap_result = parsed
        ctx.token_usage["full"] += tokens
        ctx.log(
            self.name,
            "done",
            f"Gap severity: {ctx.skill_gap_result.get('gap_severity')}",
        )
        return ctx
