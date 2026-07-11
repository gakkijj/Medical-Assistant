"""Process-local service counters with a dependency-free Prometheus export."""
from collections import Counter
from copy import deepcopy
from threading import Lock
from typing import Any, Dict, Optional


class ServiceMetrics:
    """Track aggregate operational metrics without storing request payloads."""

    def __init__(self):
        self._lock = Lock()
        self._requests_total = 0
        self._requests_in_flight = 0
        self._outcomes: Counter = Counter()
        self._routes: Counter = Counter()
        self._duration_sum = 0.0

    def start_request(self) -> None:
        with self._lock:
            self._requests_total += 1
            self._requests_in_flight += 1

    def finish_request(
        self,
        outcome: str,
        duration: float,
        route_mode: Optional[str] = None,
    ) -> None:
        with self._lock:
            self._requests_in_flight = max(0, self._requests_in_flight - 1)
            self._outcomes[outcome] += 1
            if route_mode:
                self._routes[route_mode] += 1
            self._duration_sum += max(0.0, duration)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "requests_total": self._requests_total,
                "requests_in_flight": self._requests_in_flight,
                "outcomes": deepcopy(dict(self._outcomes)),
                "routes": deepcopy(dict(self._routes)),
                "duration_seconds_sum": self._duration_sum,
            }

    def prometheus(self) -> str:
        snapshot = self.snapshot()
        lines = [
            "# HELP medix_requests_total Total accepted chat requests.",
            "# TYPE medix_requests_total counter",
            f"medix_requests_total {snapshot['requests_total']}",
            "# HELP medix_requests_in_flight Chat requests currently executing.",
            "# TYPE medix_requests_in_flight gauge",
            f"medix_requests_in_flight {snapshot['requests_in_flight']}",
            "# HELP medix_request_duration_seconds_sum Accumulated request duration.",
            "# TYPE medix_request_duration_seconds_sum counter",
            f"medix_request_duration_seconds_sum {snapshot['duration_seconds_sum']:.6f}",
        ]
        for outcome, count in sorted(snapshot["outcomes"].items()):
            lines.append(f'medix_request_outcomes_total{{outcome="{outcome}"}} {count}')
        for route, count in sorted(snapshot["routes"].items()):
            lines.append(f'medix_routes_total{{mode="{route}"}} {count}')
        return "\n".join(lines) + "\n"


service_metrics = ServiceMetrics()

