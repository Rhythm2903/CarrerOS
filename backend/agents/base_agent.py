"""
Base agent interface. Every agent in the orchestrator inherits from this.
Agents are stateless - all state lives in ReasoningContext.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ReasoningContext:
    """
    Shared scratchpad passed between all agents.
    Holds all intermediate results so no agent re-reads the resume.
    """

    session_id: str
    resume_raw: str = ""
    resume_sections: dict = field(default_factory=dict)
    career_profile: dict = field(default_factory=dict)
    jd_decomposition: dict = field(default_factory=dict)
    retrieved_chunks: str = ""
    match_result: dict = field(default_factory=dict)
    skill_gap_result: dict = field(default_factory=dict)
    market_intel: dict = field(default_factory=dict)
    roadmap_result: dict = field(default_factory=dict)
    critique_log: list = field(default_factory=list)
    agent_trace: list = field(default_factory=list)
    token_usage: dict = field(default_factory=lambda: {"mini": 0, "full": 0})

    def log(self, agent_name: str, action: str, detail: str = ""):
        self.agent_trace.append(
            {"agent": agent_name, "action": action, "detail": detail}
        )


class BaseAgent(ABC):
    name: str = "base"

    @abstractmethod
    def run(self, ctx: ReasoningContext, **kwargs) -> ReasoningContext:
        """Execute agent logic, mutate ctx, return ctx."""
        pass

