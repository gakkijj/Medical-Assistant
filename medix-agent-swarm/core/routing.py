"""Deterministic and explainable routing for the MediX agent system.

The router separates three concerns that are often mixed together:

1. high-recall safety rules detect urgent expressions;
2. an auditable semantic lexicon selects the most suitable worker;
3. low-confidence ties are marked for an optional LLM planning fallback.

It is a compute/workflow router, not a diagnostic model.
"""
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class RouteDecision:
    """A serializable routing decision with evidence for evaluation."""

    mode: str
    intent: str
    primary_agent: str
    recommended_agents: Tuple[str, ...]
    complexity_score: int
    confidence: float
    risk_level: str
    reason_codes: Tuple[str, ...]
    matched_signals: Tuple[str, ...]
    router_stage: str
    fallback_required: bool
    lead_planning_required: bool

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["recommended_agents"] = list(self.recommended_agents)
        data["reason_codes"] = list(self.reason_codes)
        data["matched_signals"] = list(self.matched_signals)
        return data


class AdaptiveRouter:
    """Budget-aware router that makes no network or model calls."""

    NEGATIONS = ("没有", "并无", "无", "未", "不", "否认", "排除")
    EMERGENCY_SIGNALS: Mapping[str, Tuple[str, ...]] = {
        "chest_pain": ("胸痛", "心口疼", "胸口疼", "心口像压着石头", "胸口压榨感"),
        "breathing_difficulty": ("呼吸困难", "喘不上气", "无法呼吸", "嘴唇发紫"),
        "altered_consciousness": ("意识不清", "昏迷", "叫不醒", "晕厥"),
        "seizure": ("抽搐", "癫痫持续发作"),
        "major_bleeding": ("大出血", "血流不止"),
        "poisoning_or_self_harm": ("服毒", "自杀", "吞了农药"),
        "anaphylaxis": ("过敏性休克", "喉咙肿无法呼吸"),
        "autonomic_warning": ("冒冷汗",),
    }
    EVIDENCE_TERMS = (
        "指南", "循证", "文献", "研究", "证据", "共识", "最新进展",
        "标准治疗", "治疗方案",
    )
    COMPLEXITY_TERMS = (
        "越来越严重", "持续", "反复", "既往史", "用药史", "同时", "鉴别",
        "严重吗", "需要就医", "孕妇", "婴儿", "老年人", "合并", "检查结果",
    )
    SYMPTOM_TERMS = (
        "发热", "发烧", "头痛", "恶心", "呕吐", "咳嗽", "胸闷", "胸痛",
        "气短", "腹痛", "腹泻", "皮疹", "乏力", "眩晕", "视力模糊",
        "心悸", "麻木", "水肿", "失眠",
    )
    INTENT_TERMS: Mapping[str, Tuple[str, ...]] = {
        "diagnosis": (
            "可能是什么", "是什么病", "什么原因", "怎么判断", "如何判断",
            "需要做哪些检查", "检查什么", "鉴别", "严重吗", "风险",
        ),
        "research": EVIDENCE_TERMS + ("诊断标准", "参考资料", "来源", "医学原理"),
        "consultation": (
            "怎么办", "怎么缓解", "注意什么", "能不能", "可以吃", "如何预防",
            "怎么预防", "饮食", "运动", "生活方式", "日常", "护理",
        ),
    }
    INTENT_AGENT = {
        "diagnosis": "diagnostic_agent",
        "research": "research_agent",
        "consultation": "consultation_agent",
    }

    def __init__(self, swarm_threshold: int = 2, fallback_confidence: float = 0.55):
        self.swarm_threshold = swarm_threshold
        self.fallback_confidence = fallback_confidence

    @staticmethod
    def _matched_terms(text: str, terms: Iterable[str]) -> Tuple[str, ...]:
        return tuple(term for term in terms if term in text)

    def _is_negated(self, text: str, start: int) -> bool:
        prefix = text[max(0, start - 6):start]
        return any(prefix.endswith(negation) for negation in self.NEGATIONS)

    def _active_emergency_signals(self, text: str) -> Tuple[str, ...]:
        active = []
        for signal, phrases in self.EMERGENCY_SIGNALS.items():
            for phrase in phrases:
                start = text.find(phrase)
                if start >= 0 and not self._is_negated(text, start):
                    active.append(signal)
                    break
        return tuple(active)

    def _select_intent(
        self,
        text: str,
        *,
        emergencies: Sequence[str],
        symptom_count: int,
        has_complexity: bool,
    ) -> Tuple[str, float, str, bool, Tuple[str, ...]]:
        """Select worker intent independently from the single/swarm decision."""
        if emergencies:
            return "diagnosis", 1.0, "safety_rule", False, tuple(emergencies)

        scores: Dict[str, int] = {}
        matched: Dict[str, Tuple[str, ...]] = {}
        for intent, terms in self.INTENT_TERMS.items():
            matched[intent] = self._matched_terms(text, terms)
            # Explicit user intent outweighs the weak signal of mentioning one symptom.
            scores[intent] = len(matched[intent]) * 2

        if symptom_count >= 3 or has_complexity:
            scores["diagnosis"] += 2
        elif symptom_count:
            scores["diagnosis"] += 1

        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_intent, best_score = ordered[0]
        second_score = ordered[1][1]

        if best_score == 0:
            return "consultation", 0.65, "default_rule", False, ()

        margin = best_score - second_score
        confidence = min(0.99, 0.55 + 0.12 * best_score + 0.08 * margin)
        fallback_required = margin == 0 or confidence < self.fallback_confidence
        stage = "llm_fallback" if fallback_required else "semantic_rule"
        return best_intent, round(confidence, 2), stage, fallback_required, matched[best_intent]

    def decide(
        self,
        question: str,
        *,
        has_history: bool = False,
        enable_swarm: bool = True,
        force_mode: Optional[str] = None,
    ) -> RouteDecision:
        """Return an auditable routing decision without network/model calls."""
        text = "".join(question.lower().split())
        emergencies = self._active_emergency_signals(text)
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

        intent, confidence, router_stage, fallback_required, intent_matches = self._select_intent(
            text,
            emergencies=emergencies,
            symptom_count=len(symptoms),
            has_complexity=bool(complexity),
        )
        primary_agent = self.INTENT_AGENT[intent]
        risk_level = "emergency" if emergencies else "normal"

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

        recommended_agents = self._select_agents(
            primary_agent=primary_agent,
            has_emergency=bool(emergencies),
            has_evidence=bool(evidence),
        )
        unique_agents = tuple(dict.fromkeys(recommended_agents))
        if mode == "single":
            unique_agents = (primary_agent,)
        elif len(unique_agents) < 2:
            secondary = "consultation_agent" if primary_agent != "consultation_agent" else "diagnostic_agent"
            unique_agents = (primary_agent, secondary)

        if fallback_required:
            reasons.append("low_confidence_route")
        if not reasons:
            reasons.append("simple_request")

        matched_signals = tuple(dict.fromkeys(
            tuple(emergencies) + tuple(evidence) + tuple(complexity) + tuple(intent_matches)
        ))
        return RouteDecision(
            mode=mode,
            intent=intent,
            primary_agent=primary_agent,
            recommended_agents=unique_agents,
            complexity_score=score,
            confidence=confidence,
            risk_level=risk_level,
            reason_codes=tuple(reasons),
            matched_signals=matched_signals,
            router_stage=router_stage,
            fallback_required=fallback_required,
            lead_planning_required=mode == "swarm" or fallback_required,
        )

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
