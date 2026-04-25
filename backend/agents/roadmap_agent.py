"""
Roadmap Agent (uses gpt-4o - quality matters for the final plan)
Responsibility: Build a personalized, market-aware 90-day roadmap.
Uses ALL prior agent outputs - this is the synthesis step.
Output saved to ctx.roadmap_result
"""

import json

from .base_agent import BaseAgent, ReasoningContext
from .llm_router import call_full_json


class RoadmapAgent(BaseAgent):
    name = "roadmap_builder"

    @staticmethod
    def _ensure_three_phases(phases: list[dict]) -> list[dict]:
        defaults = [
            {"phase": 1, "title": "Foundation", "days": "Days 1-30"},
            {"phase": 2, "title": "Proof of Skill", "days": "Days 31-60"},
            {"phase": 3, "title": "Interview Readiness", "days": "Days 61-90"},
        ]
        completed = []
        for index, default in enumerate(defaults):
            source = phases[index] if index < len(phases) else {}
            completed.append(
                {
                    "phase": default["phase"],
                    "title": source.get("title", default["title"]),
                    "days": source.get("days", default["days"]),
                    "primary_goal": source.get("primary_goal", "Complete the most important outcome for this phase"),
                    "tasks": source.get("tasks", []),
                    "milestone": source.get("milestone", "Phase milestone completed"),
                }
            )
        return completed

    def run(self, ctx: ReasoningContext, **kwargs) -> ReasoningContext:
        ctx.log(self.name, "start", "Building personalized roadmap")

        system = """You are an elite career coach. Build a hyper-personalized 90-day roadmap.
Use ALL provided context - don't give generic advice. Reference the candidate's
actual gaps, the market trends, and the specific role requirements.
Return exactly 3 phases covering Days 1-30, Days 31-60, and Days 61-90.
Return ONLY valid JSON:
{
  "title": "string - personalized to candidate name and role",
  "executive_summary": "2 sentence summary of the plan and why it will work",
  "phases": [
    {
      "phase": 1,
      "title": "string",
      "days": "Days 1-30",
      "primary_goal": "string - one specific outcome",
      "tasks": [
        {"task": "string", "resource": "string", "hours_per_week": number}
      ],
      "milestone": "string - how to know phase 1 is complete"
    }
  ],
  "quick_wins_week1": ["3 things to do THIS WEEK"],
  "portfolio_projects": [
    {"name": "string", "description": "string", "skills_demonstrated": ["list"]}
  ],
  "interview_prep_focus": ["3 specific areas to prepare for THIS company type"]
}"""

        user = f"""CANDIDATE: {json.dumps(ctx.career_profile, separators=(",", ":"))}
TARGET ROLE: {json.dumps(ctx.jd_decomposition, separators=(",", ":"))}
TIER-1 BLOCKERS: {json.dumps(ctx.skill_gap_result.get('tier1_blockers', []), separators=(",", ":"))}
TIER-2 DIFFERENTIATORS: {json.dumps(ctx.skill_gap_result.get('tier2_differentiators', []), separators=(",", ":"))}
MARKET INTEL: {json.dumps(ctx.market_intel, separators=(",", ":"))}
MATCH SCORE: {ctx.match_result.get('match_score')}
HIRING SIGNAL: {ctx.match_result.get('hiring_signal')}

Build a roadmap that directly addresses tier-1 blockers first,
incorporates the fastest-to-learn tip from market intel,
        and ends with portfolio projects that prove competence."""

        parsed, tokens = call_full_json(system, user, max_tokens=650, retry_max_tokens=900)
        parsed["phases"] = self._ensure_three_phases(parsed.get("phases", []))
        ctx.roadmap_result = parsed
        ctx.token_usage["full"] += tokens
        ctx.log(self.name, "done", "Roadmap built")
        return ctx
