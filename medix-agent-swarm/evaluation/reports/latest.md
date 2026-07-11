# MediX Offline Benchmark

Generated: 2026-07-11T14:02:11.030149+00:00

> This is a routing regression benchmark, not a clinical-accuracy claim.

| Strategy | Route accuracy | Swarm rate | Lead planning rate |
|---|---:|---:|---:|
| always_single | 45.0% | 0.0% | 0.0% |
| always_swarm | 55.0% | 100.0% | 100.0% |
| adaptive | 100.0% | 55.0% | 55.0% |

- Cases: 40
- Primary-agent accuracy: 100.0%
- Route Macro-F1: 100.0%
- Emergency-signal recall: 100.0%
- Mean routing confidence: 94.8%
- LLM fallback rate: 5.0%
- Routing failures: 0

## Retrieval ablation

| Strategy | Recall@5 | MRR | Hit rate@5 |
|---|---:|---:|---:|
| bm25 | 100.0% | 91.7% | 100.0% |
| vector | 100.0% | 86.1% | 100.0% |
| hybrid | 100.0% | 87.5% | 100.0% |
