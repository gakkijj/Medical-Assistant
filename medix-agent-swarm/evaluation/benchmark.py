"""Dependency-free evaluation helpers for routing and saved model outputs."""
import json
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from core.routing import AdaptiveRouter


def load_cases(path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL benchmark and reject malformed or duplicate cases."""
    cases: List[Dict[str, Any]] = []
    seen_ids = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            case = json.loads(line)
            required = {"id", "question", "expected_mode", "expected_agent", "emergency"}
            missing = required - case.keys()
            if missing:
                raise ValueError(f"{path}:{line_number} missing fields: {sorted(missing)}")
            if case["id"] in seen_ids:
                raise ValueError(f"duplicate case id: {case['id']}")
            if case["expected_mode"] not in {"single", "swarm"}:
                raise ValueError(f"invalid expected_mode for {case['id']}")
            seen_ids.add(case["id"])
            cases.append(case)
    if not cases:
        raise ValueError(f"benchmark is empty: {path}")
    return cases


def load_retrieval_cases(path: Path) -> List[Dict[str, Any]]:
    cases = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            case = json.loads(line)
            missing = {"id", "query", "relevant_doc_ids"} - case.keys()
            if missing:
                raise ValueError(f"{path}:{line_number} missing fields: {sorted(missing)}")
            if not case["relevant_doc_ids"]:
                raise ValueError(f"{path}:{line_number} has no relevant documents")
            cases.append(case)
    if not cases:
        raise ValueError(f"retrieval benchmark is empty: {path}")
    return cases


def _safe_div(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _macro_f1(rows: List[Dict[str, Any]], expected_key: str, predicted_key: str) -> float:
    labels = sorted({str(row[expected_key]) for row in rows})
    scores = []
    for label in labels:
        true_positive = sum(
            row[expected_key] == label and row[predicted_key] == label for row in rows
        )
        false_positive = sum(
            row[expected_key] != label and row[predicted_key] == label for row in rows
        )
        false_negative = sum(
            row[expected_key] == label and row[predicted_key] != label for row in rows
        )
        precision = _safe_div(true_positive, true_positive + false_positive)
        recall = _safe_div(true_positive, true_positive + false_negative)
        scores.append(_safe_div(2 * precision * recall, precision + recall))
    return round(statistics.fmean(scores), 4) if scores else 0.0


def evaluate_router(
    cases: Iterable[Mapping[str, Any]],
    router: Optional[AdaptiveRouter] = None,
) -> Dict[str, Any]:
    """Evaluate the adaptive router against human-authored routing labels."""
    router = router or AdaptiveRouter()
    rows = []
    for case in cases:
        decision = router.decide(str(case["question"]))
        rows.append({
            "id": case["id"],
            "expected_mode": case["expected_mode"],
            "predicted_mode": decision.mode,
            "expected_agent": case["expected_agent"],
            "predicted_agent": decision.primary_agent,
            "emergency": bool(case["emergency"]),
            "predicted_emergency": decision.risk_level == "emergency",
            "complexity_score": decision.complexity_score,
            "confidence": decision.confidence,
            "router_stage": decision.router_stage,
            "fallback_required": decision.fallback_required,
            "reason_codes": list(decision.reason_codes),
        })

    emergency_rows = [row for row in rows if row["emergency"]]
    return {
        "case_count": len(rows),
        "route_accuracy": round(_safe_div(
            sum(row["expected_mode"] == row["predicted_mode"] for row in rows),
            len(rows),
        ), 4),
        "primary_agent_accuracy": round(_safe_div(
            sum(row["expected_agent"] == row["predicted_agent"] for row in rows),
            len(rows),
        ), 4),
        "route_macro_f1": _macro_f1(rows, "expected_mode", "predicted_mode"),
        "emergency_recall": round(_safe_div(
            sum(row["predicted_emergency"] for row in emergency_rows),
            len(emergency_rows),
        ), 4),
        "swarm_rate": round(_safe_div(
            sum(row["predicted_mode"] == "swarm" for row in rows),
            len(rows),
        ), 4),
        "lead_planning_rate": round(_safe_div(
            sum(row["predicted_mode"] == "swarm" for row in rows),
            len(rows),
        ), 4),
        "fallback_rate": round(_safe_div(
            sum(row["fallback_required"] for row in rows),
            len(rows),
        ), 4),
        "mean_confidence": round(statistics.fmean(
            row["confidence"] for row in rows
        ), 4),
        "failures": [
            row for row in rows
            if row["expected_mode"] != row["predicted_mode"]
            or row["expected_agent"] != row["predicted_agent"]
            or (row["emergency"] and not row["predicted_emergency"])
        ],
    }


def evaluate_static_baseline(
    cases: Iterable[Mapping[str, Any]],
    mode: str,
) -> Dict[str, Any]:
    """Evaluate an always-single or always-swarm routing baseline."""
    rows = list(cases)
    if mode not in {"single", "swarm"}:
        raise ValueError("mode must be single or swarm")
    return {
        "case_count": len(rows),
        "route_accuracy": round(_safe_div(
            sum(case["expected_mode"] == mode for case in rows),
            len(rows),
        ), 4),
        "swarm_rate": 1.0 if mode == "swarm" else 0.0,
        "lead_planning_rate": 1.0 if mode == "swarm" else 0.0,
    }


def evaluate_retriever(
    cases: Iterable[Mapping[str, Any]],
    retriever: Any,
    *,
    top_k: int = 5,
    strategy: str = "hybrid",
) -> Dict[str, Any]:
    """Evaluate document-level retrieval recall and reciprocal rank."""
    rows = []
    for case in cases:
        results = retriever.search(str(case["query"]), top_k=top_k, strategy=strategy)
        retrieved = [str(item.get("doc_id") or item.get("metadata", {}).get("doc_id")) for item in results]
        relevant = {str(doc_id) for doc_id in case["relevant_doc_ids"]}
        relevant_ranks = [rank for rank, doc_id in enumerate(retrieved, 1) if doc_id in relevant]
        rows.append({
            "id": case["id"],
            "retrieved_doc_ids": retrieved,
            "relevant_doc_ids": sorted(relevant),
            "recall": _safe_div(len(set(retrieved) & relevant), len(relevant)),
            "reciprocal_rank": 1 / min(relevant_ranks) if relevant_ranks else 0.0,
        })
    return {
        "case_count": len(rows),
        "strategy": strategy,
        "top_k": top_k,
        "hit_rate_at_k": round(_safe_div(sum(row["reciprocal_rank"] > 0 for row in rows), len(rows)), 4),
        "recall_at_k": round(statistics.fmean(row["recall"] for row in rows), 4),
        "mrr": round(statistics.fmean(row["reciprocal_rank"] for row in rows), 4),
        "failures": [row for row in rows if row["reciprocal_rank"] == 0],
    }


def _percentile(values: List[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(round((len(ordered) - 1) * percentile), len(ordered) - 1)
    return float(ordered[index])


def evaluate_predictions(
    cases: Iterable[Mapping[str, Any]],
    predictions: Iterable[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Score saved end-to-end outputs without invoking an LLM.

    Prediction records may contain answer, citations, latency_seconds,
    total_tokens, llm_call_count and route. Missing fields are treated as zero.
    """
    case_by_id = {str(case["id"]): case for case in cases}
    prediction_by_id = {str(item["id"]): item for item in predictions}
    matched = []
    for case_id, case in case_by_id.items():
        if case_id not in prediction_by_id:
            continue
        prediction = prediction_by_id[case_id]
        answer = str(prediction.get("answer", ""))
        required_terms = [str(term) for term in case.get("required_terms", [])]
        citations = prediction.get("citations") or []
        retrieved_doc_ids = prediction.get("retrieved_doc_ids") or [
            citation.get("doc_id") for citation in citations if citation.get("doc_id")
        ]
        relevant_doc_ids = [str(item) for item in case.get("relevant_doc_ids", [])]
        relevant_ranks = [
            index for index, doc_id in enumerate(retrieved_doc_ids, 1)
            if str(doc_id) in relevant_doc_ids
        ]
        matched.append({
            "emergency": bool(case["emergency"]),
            "emergency_advice": any(
                term in answer for term in ("立即就医", "拨打120", "急诊", "急救")
            ),
            "keyword_coverage": _safe_div(
                sum(term in answer for term in required_terms),
                len(required_terms),
            ) if required_terms else 1.0,
            "citation_present": bool(citations),
            "citation_required": bool(case.get("citation_required", False)),
            "retrieval_required": bool(relevant_doc_ids),
            "retrieval_recall": _safe_div(
                len(set(map(str, retrieved_doc_ids)) & set(relevant_doc_ids)),
                len(set(relevant_doc_ids)),
            ) if relevant_doc_ids else 1.0,
            "reciprocal_rank": 1 / min(relevant_ranks) if relevant_ranks else 0.0,
            "latency": float(prediction.get("latency_seconds", 0.0) or 0.0),
            "tokens": int(prediction.get("total_tokens", 0) or 0),
            "llm_calls": int(prediction.get("llm_call_count", 0) or 0),
        })

    if not matched:
        raise ValueError("no prediction ids matched the benchmark")
    emergency_rows = [row for row in matched if row["emergency"]]
    citation_rows = [row for row in matched if row["citation_required"]]
    retrieval_rows = [row for row in matched if row["retrieval_required"]]
    latencies = [row["latency"] for row in matched]
    return {
        "matched_cases": len(matched),
        "coverage": round(_safe_div(len(matched), len(case_by_id)), 4),
        "keyword_coverage": round(statistics.fmean(
            row["keyword_coverage"] for row in matched
        ), 4),
        "emergency_advice_recall": round(_safe_div(
            sum(row["emergency_advice"] for row in emergency_rows),
            len(emergency_rows),
        ), 4),
        "citation_coverage": round(_safe_div(
            sum(row["citation_present"] for row in citation_rows),
            len(citation_rows),
        ), 4),
        "retrieval_recall_at_k": round(statistics.fmean(
            row["retrieval_recall"] for row in retrieval_rows
        ), 4) if retrieval_rows else 0.0,
        "retrieval_mrr": round(statistics.fmean(
            row["reciprocal_rank"] for row in retrieval_rows
        ), 4) if retrieval_rows else 0.0,
        "latency_p50_seconds": round(_percentile(latencies, 0.50), 4),
        "latency_p95_seconds": round(_percentile(latencies, 0.95), 4),
        "mean_total_tokens": round(statistics.fmean(row["tokens"] for row in matched), 2),
        "mean_llm_calls": round(statistics.fmean(row["llm_calls"] for row in matched), 2),
    }


def load_predictions(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
