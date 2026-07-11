"""Dependency-free tests for request metrics and sanitized trace events."""
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.request_metrics import (
    get_request_metrics,
    record_llm_call,
    record_route_decision,
    record_tool_call,
    reset_request_metrics,
    start_request_metrics,
)


class RequestMetricsTest(unittest.TestCase):
    def test_counts_tokens_tools_and_route(self):
        token = start_request_metrics()
        try:
            record_route_decision({
                "mode": "single",
                "primary_agent": "consultation_agent",
                "complexity_score": 0,
            }, 0.001)
            record_llm_call(
                0.2,
                prompt_tokens=100,
                completion_tokens=40,
                total_tokens=140,
            )
            record_tool_call("search_knowledge", 0.05)
            metrics = get_request_metrics()
        finally:
            reset_request_metrics(token)

        self.assertEqual(metrics["llm_call_count"], 1)
        self.assertEqual(metrics["tool_call_count"], 1)
        self.assertEqual(metrics["total_tokens"], 140)
        self.assertEqual(metrics["route"]["mode"], "single")
        self.assertEqual([event["name"] for event in metrics["events"]], [
            "routing",
            "llm_call",
            "tool_call",
        ])

    def test_metrics_are_empty_outside_request_context(self):
        self.assertEqual(get_request_metrics()["llm_call_count"], 0)


if __name__ == "__main__":
    unittest.main()
