"""Dependency-free tests for reproducible evaluation utilities."""
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.benchmark import evaluate_predictions, evaluate_router, load_cases


class BenchmarkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = load_cases(PROJECT_ROOT / "evaluation" / "cases.jsonl")

    def test_router_meets_regression_gates(self):
        metrics = evaluate_router(self.cases)
        self.assertGreaterEqual(metrics["route_accuracy"], 0.85)
        self.assertEqual(metrics["emergency_recall"], 1.0)

    def test_saved_prediction_metrics(self):
        case = {
            "id": "urgent",
            "question": "胸痛怎么办",
            "expected_mode": "swarm",
            "expected_agent": "diagnostic_agent",
            "emergency": True,
            "citation_required": True,
            "relevant_doc_ids": ["emergency_doc"],
            "required_terms": ["急诊"],
        }
        prediction = {
            "id": "urgent",
            "answer": "请立即前往急诊。",
            "citations": [{"source": "test", "doc_id": "emergency_doc"}],
            "latency_seconds": 1.2,
            "total_tokens": 120,
            "llm_call_count": 2,
        }
        metrics = evaluate_predictions([case], [prediction])
        self.assertEqual(metrics["emergency_advice_recall"], 1.0)
        self.assertEqual(metrics["citation_coverage"], 1.0)
        self.assertEqual(metrics["keyword_coverage"], 1.0)
        self.assertEqual(metrics["retrieval_recall_at_k"], 1.0)
        self.assertEqual(metrics["retrieval_mrr"], 1.0)


if __name__ == "__main__":
    unittest.main()
