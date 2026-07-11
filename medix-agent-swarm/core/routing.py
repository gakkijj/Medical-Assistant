"""Deterministic, explainable routing for the MediX agent system.

The router intentionally does not call an LLM. Simple requests can therefore
skip the LeadAgent planning call, while complex or high-risk requests still use
the multi-agent workflow.
"""
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Optional, Tuple


@dataclass(frozen=True)
class RouteDecision:
    """A serializable routing decision with evidence for later evaluation."""

    mode: str
    primary_agent: str
    recommended_agents: Tuple[str, ...]
    complexity_score: int
    risk_level: str
    reason_codes: Tuple[str, ...]
    lead_planning_required: bool

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["recommended_agents"] = list(self.recommended_agents)
        data["reason_codes"] = list(self.reason_codes)
        return data


class AdaptiveRouter:
    """Budget-aware rule router used before any LLM planning call.

    The rules are deliberately small and auditable. They are not a diagnostic
    model: they only decide how much compute and which specialist workflow a
    request needs.
    """

    EMERGENCY_TERMS = (
        "胸痛",
        "呼吸困难",
        "喘不上气",
        "意识不清",
        "昏迷",
        "晕厥",
        "抽搐",
        "大出血",
        "冒冷汗",
        "嘴唇发紫",
        "服毒",
        "自杀",
        "过敏性休克",
    )
    EVIDENCE_TERMS = (
        "指南",
        "循证",
        "文献",
        "研究",
        "证据",
        "共识",
        "最新进展",
        "标准治疗",
        "治疗方案",
    )
    COMPLEXITY_TERMS = (
        "越来越严重",
        "持续",
        "反复",
        "既往史",
        "用药史",
        "同时",
        "鉴别",
        "严重吗",
        "需要就医",
        "孕妇",
        "婴儿",
        "老年人",
    )
    SYMPTOM_TERMS = (
        "发热",
        "发烧",
        "头痛",
        "恶心",
        "呕吐",
        "咳嗽",
        "胸闷",
        "胸痛",
        "气短",
        "腹痛",
        "腹泻",
        "皮疹",
        "乏力",
        "眩晕",
        "视力模糊",
        "心悸",
        "麻木",
    )

    def __init__(self, swarm_threshold: int = 2):
        self.swarm_threshold = swarm_threshold

    @staticmethod
    def _matched_terms(text: str, terms: Iterable[str]) -> Tuple[str, ...]:
        return tuple(term for term in terms if term in text)

    def decide(
        self,
        question: str,
        *,
        has_history: bool = False,
        enable_swarm: bool = True,
        force_mode: Optional[str] = None,
    ) -> RouteDecision:
        """Return a routing decision without network or model calls."""
        text = "".join(question.lower().split())
        emergencies = self._matched_terms(text, self.EMERGENCY_TERMS)
        evidence = self._matched_terms(text, self.EVIDENCE_TERMS)
        complexity = self._matched_terms(text, self.COMPLEXITY_TERMS)
        symptoms = self._matched_terms(text, self.SYMPTOM_TERMS)

        score = 0
        reasons = []
        if emergencies:
            score += 5
            reasons.append("emergency_signal")
        if evidence:
            score += 3
            reasons.append("evidence_required")
        if len(symptoms) >= 3:
            score += 2
            reasons.append("multiple_symptoms")
        if complexity:
            score += 2
            reasons.append("complex_context")
        if has_history:
            score += 1
            reasons.append("conversation_context")
        if len(text) >= 80:
            score += 1
            reasons.append("long_question")

        risk_level = "emergency" if emergencies else "normal"
        primary_agent = self._select_primary_agent(
            has_emergency=bool(emergencies),
            has_evidence=bool(evidence),
            symptom_count=len(symptoms),
            has_complexity=bool(complexity),
        )
        recommended_agents = self._select_agents(
            primary_agent=primary_agent,
            has_emergency=bool(emergencies),
            has_evidence=bool(evidence),
        )

        normalized_force_mode = (force_mode or "auto").strip().lower()
        if normalized_force_mode not in {"auto", "single", "swarm"}:
            raise ValueError("force_mode must be one of: auto, single, swarm")

        if normalized_force_mode != "auto":
            mode = normalized_force_mode
            reasons.append("forced_route")
        else:
            mode = "swarm" if score >= self.swarm_threshold else "single"

        if not enable_swarm and mode == "swarm":
            mode = "single"
            reasons.append("swarm_disabled")

        unique_agents = tuple(dict.fromkeys(recommended_agents))
        if mode == "single":
            unique_agents = (primary_agent,)
        elif len(unique_agents) < 2:
            secondary = (
                "diagnostic_agent"
                if primary_agent == "consultation_agent"
                else "consultation_agent"
            )
            unique_agents = (primary_agent, secondary)

        if not reasons:
            reasons.append("simple_request")

        return RouteDecision(
            mode=mode,
            primary_agent=primary_agent,
            recommended_agents=unique_agents,
            complexity_score=score,
            risk_level=risk_level,
            reason_codes=tuple(reasons),
            lead_planning_required=mode == "swarm",
        )

    @staticmethod
    def _select_primary_agent(
        *,
        has_emergency: bool,
        has_evidence: bool,
        symptom_count: int,
        has_complexity: bool,
    ) -> str:
        if has_emergency or symptom_count >= 3 or has_complexity:
            return "diagnostic_agent"
        if has_evidence:
            return "research_agent"
        return "consultation_agent"

    @staticmethod
    def _select_agents(
        *,
        primary_agent: str,
        has_emergency: bool,
        has_evidence: bool,
    ) -> Tuple[str, ...]:
        if has_emergency and has_evidence:
            return ("diagnostic_agent", "research_agent", "consultation_agent")
        if has_emergency:
            return ("diagnostic_agent", "consultation_agent")
        if has_evidence:
            return ("research_agent", "consultation_agent")
        return (primary_agent, "consultation_agent")
