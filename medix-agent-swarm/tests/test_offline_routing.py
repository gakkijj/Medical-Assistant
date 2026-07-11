"""Dependency-free tests for the adaptive router."""
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.routing import AdaptiveRouter


class AdaptiveRouterTest(unittest.TestCase):
    def setUp(self):
        self.router = AdaptiveRouter()

    def test_simple_question_skips_lead_planning(self):
        decision = self.router.decide("感冒了应该怎么办？")
        self.assertEqual(decision.mode, "single")
        self.assertEqual(decision.primary_agent, "consultation_agent")
        self.assertFalse(decision.lead_planning_required)

    def test_evidence_question_uses_research_swarm(self):
        decision = self.router.decide("高血压最新诊疗指南和循证证据是什么？")
        self.assertEqual(decision.mode, "swarm")
        self.assertEqual(decision.primary_agent, "research_agent")
        self.assertIn("research_agent", decision.recommended_agents)
        self.assertIn("consultation_agent", decision.recommended_agents)

    def test_multiple_symptoms_use_diagnostic_swarm(self):
        decision = self.router.decide("头痛、恶心并且视力模糊，可能是什么原因？")
        self.assertEqual(decision.mode, "swarm")
        self.assertEqual(decision.primary_agent, "diagnostic_agent")

    def test_emergency_signal_has_full_recall_path(self):
        decision = self.router.decide("突然胸闷、气短、冒冷汗怎么办？")
        self.assertEqual(decision.risk_level, "emergency")
        self.assertEqual(decision.mode, "swarm")
        self.assertIn("emergency_signal", decision.reason_codes)

    def test_swarm_can_be_disabled_for_cost_control(self):
        decision = self.router.decide("请给出糖尿病最新诊疗指南", enable_swarm=False)
        self.assertEqual(decision.mode, "single")
        self.assertIn("swarm_disabled", decision.reason_codes)

    def test_force_mode_is_available_for_ablation(self):
        decision = self.router.decide("什么是高血压？", force_mode="swarm")
        self.assertEqual(decision.mode, "swarm")
        self.assertGreaterEqual(len(decision.recommended_agents), 2)

    def test_simple_diagnostic_intent_can_stay_single_agent(self):
        decision = self.router.decide("轻微头痛可能是什么原因？")
        self.assertEqual(decision.mode, "single")
        self.assertEqual(decision.intent, "diagnosis")
        self.assertEqual(decision.primary_agent, "diagnostic_agent")

    def test_simple_research_intent_can_stay_single_agent(self):
        decision = self.router.decide("高血压的诊断标准是什么？")
        self.assertEqual(decision.mode, "single")
        self.assertEqual(decision.primary_agent, "research_agent")
        self.assertEqual(decision.intent, "research")

    def test_negated_emergency_phrase_does_not_raise_emergency(self):
        decision = self.router.decide("我没有胸痛，只是想了解日常运动建议")
        self.assertEqual(decision.risk_level, "normal")
        self.assertNotIn("emergency_signal", decision.reason_codes)

    def test_colloquial_emergency_expression_is_detected(self):
        decision = self.router.decide("心口像压着石头，还喘不上气")
        self.assertEqual(decision.risk_level, "emergency")
        self.assertEqual(decision.router_stage, "safety_rule")
        self.assertEqual(decision.confidence, 1.0)

    def test_route_exposes_auditable_metadata(self):
        decision = self.router.decide("应该如何预防高血压？")
        payload = decision.to_dict()
        self.assertIn("confidence", payload)
        self.assertIn("router_stage", payload)
        self.assertIn("matched_signals", payload)


if __name__ == "__main__":
    unittest.main()
