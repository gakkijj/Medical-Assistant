# 实验设计与结果

## 可复现命令

```bash
make test-offline
make benchmark
```

完整 API 测试需要轻量 CI 依赖：

```bash
cd medix-agent-swarm
python -m pip install -r requirements-ci.txt
python -m pytest tests -q
ruff check api core knowledge/hybrid_retriever.py evaluation tests
```

## 路由实验

40条人工编写回归用例覆盖普通咨询、简单诊断、简单研究、复杂症状、循证请求、急症、否定句和口语表达。

| 策略 | Accuracy | Swarm率 | Lead规划率 |
|---|---:|---:|---:|
| Always Single | 45.0% | 0.0% | 0.0% |
| Always Swarm | 55.0% | 100.0% | 100.0% |
| Adaptive | 100.0% | 55.0% | 55.0% |

Adaptive 在该集合上的 Macro-F1、主 Agent 准确率和急症召回均为100%，平均置信度94.8%。这是透明的小型工程回归集，不能外推为真实流量效果，更不能代表临床正确率。

## 检索消融

12条检索用例使用相同演示语料和相同 Top-5 条件：

| 策略 | Recall@5 | MRR | Hit Rate@5 |
|---|---:|---:|---:|
| BM25 | 100.0% | 91.7% | 100.0% |
| 字符向量 | 100.0% | 86.1% | 100.0% |
| 加权融合 | 100.0% | 87.5% | 100.0% |

结论：当前语料规模小、查询和文档关键词重合较高，BM25 排序最好；现阶段不能宣称混合检索带来提升。后续需要增加口语改写、缩写、跨语言和长文档数据，再评估 Dense Retrieval 与 Cross-Encoder 的收益。

## CI 门禁

- 路由 Accuracy、Macro-F1、主 Agent 准确率不得低于90%。
- 急症信号召回必须为100%。
- 混合检索 Recall@5 不得低于85%，MRR不得低于75%。
- Python 3.10和3.12均运行全部测试与静态检查。

最新机器生成报告位于 `medix-agent-swarm/evaluation/reports/latest.md`。

## 尚未完成的外部实验

下列结果需要有效模型 Key、授权语料或公网环境，本仓库不伪造数字：

- 单 Agent/RAG/Swarm/Adaptive 的回答质量与费用对比。
- 真实 LLM 请求下的P50/P95/P99延迟与吞吐。
- 经医学专业人员双标的安全性和事实性评估。
- 公开云环境的长期稳定性、告警和故障恢复。

