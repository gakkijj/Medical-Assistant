# 压测方法

仓库提供 `evaluation/load_test.py`，用于对已配置模型 Key 的运行实例做并发压测。脚本不会把问题正文写入报告，只输出状态码、吞吐和延迟分位数。

```bash
cd medix-agent-swarm
python evaluation/load_test.py \
  --url http://127.0.0.1:8000/api/chat \
  --requests 50 \
  --concurrency 5 \
  --routing-mode auto \
  --output evaluation/reports/load-test.json
```

建议分别测试 `single`、`swarm` 和 `auto`，并记录：

- 成功率与超时率
- 吞吐量（requests/s）
- P50/P95/P99延迟
- API 响应中的平均 Token 与 LLM 调用次数
- `/api/metrics` 中的路由分布

当前仓库没有提交虚构的压测结果。实际结果与模型供应商、网络、限流和问题复杂度高度相关。

