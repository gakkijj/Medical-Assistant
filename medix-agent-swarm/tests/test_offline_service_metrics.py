"""Dependency-free aggregate service metrics tests."""
import unittest

from core.service_metrics import ServiceMetrics


class ServiceMetricsTest(unittest.TestCase):
    def test_metrics_track_outcome_route_and_duration(self):
        metrics = ServiceMetrics()
        metrics.start_request()
        self.assertEqual(metrics.snapshot()["requests_in_flight"], 1)
        metrics.finish_request("success", 0.25, "single")
        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["requests_total"], 1)
        self.assertEqual(snapshot["requests_in_flight"], 0)
        self.assertEqual(snapshot["outcomes"]["success"], 1)
        self.assertEqual(snapshot["routes"]["single"], 1)
        self.assertEqual(snapshot["duration_seconds_sum"], 0.25)

    def test_prometheus_export_contains_no_request_payload(self):
        metrics = ServiceMetrics()
        metrics.start_request()
        metrics.finish_request("timeout", 1.0)
        output = metrics.prometheus()
        self.assertIn("medix_requests_total 1", output)
        self.assertIn('outcome="timeout"', output)


if __name__ == "__main__":
    unittest.main()
