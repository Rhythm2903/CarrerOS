"""
Master Orchestrator
This is the brain of the system. It:
  1. Runs agents in dependency order
  2. Passes ReasoningContext between agents
  3. Runs the Critic agent and handles the self-critique loop
  4. Returns the final enriched context

Agent execution plan:
  ParserAgent -> JDAgent -> (retrieve context) -> MatchAgent
  -> GapAgent -> MarketAgent -> RoadmapAgent -> CriticAgent
"""

from backend.rag_pipeline.rag_engine import retrieve_context

from .base_agent import ReasoningContext
from .critic_agent import CriticAgent
from .gap_agent import GapAgent
from .jd_agent import JDAgent
from .market_agent import MarketAgent
from .match_agent import MatchAgent
from .parser_agent import ParserAgent
from .roadmap_agent import RoadmapAgent

MAX_RETRIES = 0


def run_full_analysis(
    session_id: str, resume_raw: str, resume_sections: dict, job_description: str
) -> dict:
    """
    Main entry point. Runs the full agentic pipeline.
    Returns a dict with all results + agent trace + token usage.
    """

    ctx = ReasoningContext(
        session_id=session_id, resume_raw=resume_raw, resume_sections=resume_sections
    )

    agents = [
        ParserAgent(),
        JDAgent(),
        MatchAgent(),
        GapAgent(),
        MarketAgent(),
        RoadmapAgent(),
        CriticAgent(),
    ]

    for agent in agents:
        if agent.name == "jd_decomposer":
            ctx = agent.run(ctx, job_description=job_description)
        elif agent.name == "matcher":
            ctx.retrieved_chunks = retrieve_context(
                session_id, job_description, top_k=2, max_chars=800
            )
            ctx = agent.run(ctx)
        else:
            ctx = agent.run(ctx)

    critique = ctx.critique_log[-1] if ctx.critique_log else {}
    if not critique.get("approved", True) and MAX_RETRIES > 0:
        ctx.log("orchestrator", "retry", "Quality below threshold - re-running roadmap")
        ctx = RoadmapAgent().run(ctx)
        ctx = CriticAgent().run(ctx)

    return _build_response(ctx)


def _build_response(ctx: ReasoningContext) -> dict:
    """Package the final response from context."""

    total_tokens = ctx.token_usage["mini"] + ctx.token_usage["full"]
    mini_pct = round(ctx.token_usage["mini"] / max(total_tokens, 1) * 100)

    return {
        "match_analysis": ctx.match_result,
        "skill_gap": ctx.skill_gap_result,
        "market_intel": ctx.market_intel,
        "learning_roadmap": ctx.roadmap_result,
        "career_profile": ctx.career_profile,
        "meta": {
            "agent_trace": ctx.agent_trace,
            "critique": ctx.critique_log[-1] if ctx.critique_log else {},
            "token_usage": ctx.token_usage,
            "total_tokens": total_tokens,
            "efficiency": f"{mini_pct}% tokens used mini model",
        },
    }
