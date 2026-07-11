# 演示知识库说明

`documents/` 中的文本用于演示切片、检索、引用和回归测试，不是经过临床审核、持续更新或可直接用于生产医疗建议的权威知识库。

- 检索层将来源统一标记为“项目演示资料”，避免把文件标题包装成真实证据。
- 文档进入生产环境前必须补充原始 URL、发布机构、发布日期、版本、授权协议和人工审核状态。
- 任何模型回答都不能仅凭这些演示文本替代医生诊断或治疗。
- 包含提示词注入模式的知识块会被标记为不可信并排除。

推荐的生产元数据结构：

```json
{
  "doc_id": "stable-id",
  "title": "document title",
  "source_url": "https://...",
  "publisher": "publisher",
  "published_at": "YYYY-MM-DD",
  "license": "license identifier",
  "review_status": "approved",
  "content_hash": "sha256"
}
```

