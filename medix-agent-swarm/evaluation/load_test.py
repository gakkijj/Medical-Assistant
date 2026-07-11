"""Small asynchronous load-test client for a configured MediX deployment."""
import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4

import httpx


def percentile(values: List[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(round((len(ordered) - 1) * ratio), len(ordered) - 1)
    return ordered[index]


async def run_load_test(args: argparse.Namespace) -> Dict[str, Any]:
    semaphore = asyncio.Semaphore(args.concurrency)
    latencies: List[float] = []
    statuses: List[int] = []
    tokens: List[int] = []
    llm_calls: List[int] = []

    async with httpx.AsyncClient(timeout=args.timeout) as client:
        async def one_request(index: int) -> None:
            async with semaphore:
                started = time.perf_counter()
                try:
                    response = await client.post(
                        args.url,
                        json={
                            "message": args.message,
                            "session_id": f"load-{index}-{uuid4().hex[:8]}",
                            "routing_mode": args.routing_mode,
                        },
                    )
                    statuses.append(response.status_code)
                    if response.status_code == 200:
                        payload = response.json()
                        tokens.append(int(payload.get("total_tokens") or 0))
                        llm_calls.append(int(payload.get("llm_call_count") or 0))
                except httpx.HTTPError:
                    statuses.append(0)
                finally:
                    latencies.append(time.perf_counter() - started)

        wall_start = time.perf_counter()
        await asyncio.gather(*(one_request(index) for index in range(args.requests)))
        wall_time = time.perf_counter() - wall_start

    successes = sum(status == 200 for status in statuses)
    return {
        "requests": args.requests,
        "concurrency": args.concurrency,
        "routing_mode": args.routing_mode,
        "success_rate": round(successes / max(args.requests, 1), 4),
        "throughput_requests_per_second": round(args.requests / max(wall_time, 1e-9), 4),
        "latency_p50_seconds": round(percentile(latencies, 0.50), 4),
        "latency_p95_seconds": round(percentile(latencies, 0.95), 4),
        "latency_p99_seconds": round(percentile(latencies, 0.99), 4),
        "mean_total_tokens": round(statistics.fmean(tokens), 2) if tokens else 0.0,
        "mean_llm_calls": round(statistics.fmean(llm_calls), 2) if llm_calls else 0.0,
        "status_counts": {str(status): statuses.count(status) for status in sorted(set(statuses))},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/api/chat")
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--routing-mode", choices=("auto", "single", "swarm"), default="auto")
    parser.add_argument("--message", default="高血压患者日常饮食和运动应该注意什么？")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.requests < 1 or args.concurrency < 1:
        parser.error("requests and concurrency must be positive")
    report = asyncio.run(run_load_test(args))
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0 if report["success_rate"] == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
