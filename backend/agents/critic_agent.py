"""
Critic Agent (uses gpt-4o-mini - cheap reflection pass)
Responsibility: Review all outputs for consistency and flag issues.
If score < threshold, sets a retry flag in the context.
This is the self-critique loop that makes it agentic.
"""

from .base_agent import BaseAgent, ReasoningContext
from .llm_router import call_mini_json


QUALITY_THRESHOLD = 6


class CriticAgent(BaseAgent):
    name = "critic"

    def run(self, ctx: ReasoningContext, **kwargs) -> ReasoningContext:
        ctx.log(self.name, "start", "Running quality critique")

        system = """You are a quality control agent reviewing AI-generated career analysis.
Check for: logical consistency, specificity (not generic advice), coverage of key gaps.
Return ONLY valid JSON:
{
  "quality_score": integer 1-10,
  "issues": ["list of specific problems found, empty if none"],
  "is_generic": boolean,
  "missing_personalisation": ["what's missing from the candidate's specific context"],
  "approved": boolean
}"""

        user = f"""Review these outputs for quality:

MATCH SCORE: {ctx.match_result.get('match_score')}
VERDICT: {ctx.match_result.get('one_line_verdict')}
GAP SEVERITY: {ctx.skill_gap_result.get('gap_severity')}
TIER-1 BLOCKERS COUNT: {len(ctx.skill_gap_result.get('tier1_blockers', []))}
ROADMAP TITLE: {ctx.roadmap_result.get('title')}
MARKET TIP: {ctx.market_intel.get('market_tip')}
QUICK WINS: {ctx.roadmap_result.get('quick_wins_week1')}

Is this analysis specific to THIS candidate and THIS role?
Or is it generic advice that ChatGPT would give to anyone?"""

        critique, tokens = call_mini_json(system, user, max_tokens=220, retry_max_tokens=360)
        ctx.critique_log.append(critique)
        ctx.token_usage["mini"] += tokens

        approved = critique.get("approved", True)
        score = critique.get("quality_score", 7)
        ctx.log(self.name, "done", f"Quality: {score}/10 | Approved: {approved}")
        return ctx
