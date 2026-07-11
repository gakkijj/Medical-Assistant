"""CLI for reproducible MediX routing and saved-output evaluation."""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.benchmark import (  # noqa: E402
    evaluate_predictions,
    evaluate_router,
    evaluate_static_baseline,
    load_cases,
    load_predictions,
)


def _markdown(report: dict) -> str:
    strategies = report["routing"]
    lines = [
        "# MediX Offline Benchmark",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "> This is a routing regression benchmark, not a clinical-accuracy claim.",
        "",
        "| Strategy | Route accuracy | Swarm rate | Lead planning rate |",
        "|---|---:|---:|---:|",
    ]
    for name in ("always_single", "always_swarm", "adaptive"):
        metrics = strategies[name]
        lines.append(
            f"| {name} | {metrics['route_accuracy']:.1%} | "
            f"{metrics['swarm_rate']:.1%} | {metrics['lead_planning_rate']:.1%} |"
        )
    adaptive = strategies["adaptive"]
    lines.extend([
        "",
        f"- Cases: {adaptive['case_count']}",
        f"- Primary-agent accuracy: {adaptive['primary_agent_accuracy']:.1%}",
        f"- Emergency-signal recall: {adaptive['emergency_recall']:.1%}",
        f"- Routing failures: {len(adaptive['failures'])}",
    ])
    if "end_to_end" in report:
        lines.extend(["", "## Saved-output evaluation", "", "```json"])
        lines.append(json.dumps(report["end_to_end"], ensure_ascii=False, indent=2))
        lines.append("```")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    default_cases = Path(__file__).with_name("cases.jsonl")
    parser.add_argument("--cases", type=Path, default=default_cases)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("reports"))
    parser.add_argument("--check", action="store_true", help="fail when routing gates regress")
    args = parser.parse_args()

    cases = load_cases(args.cases)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "routing": {
            "always_single": evaluate_static_baseline(cases, "single"),
            "always_swarm": evaluate_static_baseline(cases, "swarm"),
            "adaptive": evaluate_router(cases),
        },
    }
    if args.predictions:
        report["end_to_end"] = evaluate_predictions(cases, load_predictions(args.predictions))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "latest.json"
    markdown_path = args.output_dir / "latest.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    print(_markdown(report), end="")

    adaptive = report["routing"]["adaptive"]
    if args.check and (
        adaptive["route_accuracy"] < 0.85
        or adaptive["emergency_recall"] < 1.0
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
