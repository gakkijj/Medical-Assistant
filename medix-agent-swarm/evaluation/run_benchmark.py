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
    evaluate_retriever,
    evaluate_router,
    evaluate_static_baseline,
    load_cases,
    load_predictions,
    load_retrieval_cases,
)
from knowledge.hybrid_retriever import HybridRetriever  # noqa: E402


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
        f"- Route Macro-F1: {adaptive['route_macro_f1']:.1%}",
        f"- Emergency-signal recall: {adaptive['emergency_recall']:.1%}",
        f"- Mean routing confidence: {adaptive['mean_confidence']:.1%}",
        f"- LLM fallback rate: {adaptive['fallback_rate']:.1%}",
        f"- Routing failures: {len(adaptive['failures'])}",
    ])
    if "retrieval" in report:
        lines.extend([
            "",
            "## Retrieval ablation",
            "",
            "| Strategy | Recall@5 | MRR | Hit rate@5 |",
            "|---|---:|---:|---:|",
        ])
        for name in ("bm25", "vector", "hybrid"):
            metrics = report["retrieval"][name]
            lines.append(
                f"| {name} | {metrics['recall_at_k']:.1%} | "
                f"{metrics['mrr']:.1%} | {metrics['hit_rate_at_k']:.1%} |"
            )
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
    parser.add_argument(
        "--retrieval-cases",
        type=Path,
        default=Path(__file__).with_name("retrieval_cases.jsonl"),
    )
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
    retrieval_cases = load_retrieval_cases(args.retrieval_cases)
    retriever = HybridRetriever.from_directory(PROJECT_ROOT / "knowledge/data/documents")
    report["retrieval"] = {
        strategy: evaluate_retriever(
            retrieval_cases,
            retriever,
            top_k=5,
            strategy=strategy,
        )
        for strategy in ("bm25", "vector", "hybrid")
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
    hybrid = report["retrieval"]["hybrid"]
    if args.check and (
        adaptive["route_accuracy"] < 0.90
        or adaptive["route_macro_f1"] < 0.90
        or adaptive["primary_agent_accuracy"] < 0.90
        or adaptive["emergency_recall"] < 1.0
        or hybrid["recall_at_k"] < 0.85
        or hybrid["mrr"] < 0.75
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
