# 检索质量评测

本目录提供一套可复现的检索评测流程,用于量化对比不同检索配置的效果,
避免凭少量人工 query 主观判断。

## 评测方法

1. **标注集构造**(`build_eval_set.py`):从 `validity == "有效"` 的法规 chunk
   中按文档分层抽样,用 LLM(DeepSeek)为每个 chunk 生成一个用户视角的问题,
   自动建立 `query → golden chunk` 配对。生成后对空/过短/截断的 query 做校验重试。
2. **评测**(`run_eval.py`):单正例标注,统计 Recall@K(单正例下等价 Hit@K)、
   MRR@10、NDCG@10;所有检索均加 `validity == "有效"` 过滤,只评现行有效库。
3. **对比四种配置**:dense / sparse / hybrid / hybrid+rerank,
   召回池 50 → rerank 取 top-10。

## 评测结果(100 条标注集)

| 配置              | Recall@1  | Recall@5  | Recall@10 | MRR@10    | NDCG@10   |
| ----------------- | --------- | --------- | --------- | --------- | --------- |
| dense             | 0.560     | 0.860     | 0.910     | 0.694     | 0.747     |
| sparse (BM25)     | 0.540     | 0.830     | 0.880     | 0.665     | 0.718     |
| hybrid (RRF)      | 0.580     | 0.860     | 0.910     | 0.708     | 0.758     |
| **hybrid+rerank** | **0.760** | **0.970** | **0.970** | **0.856** | **0.885** |

**关键结论**:三种召回方式 Recall@10 均达 0.88~0.91,召回阶段不是瓶颈;
真正的增益来自 Cross-Encoder 精排,将 Recall@1 从 0.58 拉到 0.76、
MRR@10 从 0.71 拉到 0.86,印证"召回宽、精排严"的两阶段设计。

## 指标说明

- **Recall@K**:正确答案是否出现在 top-K(单正例下即命中率 Hit@K)。
- **MRR@10**:正确答案排名的倒数(排第 r 位记 1/r,未进 top-10 记 0),衡量排序靠前程度。
- **NDCG@10**:排名的对数折损增益(单正例下 = 1/log2(rank+1)),衡量排序质量。

## 已知偏差(诚实声明)

- LLM 基于 chunk 生成 query,可能存在词汇泄露(query 与 golden 词汇重合),
  使 sparse 分数偏乐观;prompt 已要求同义改写以缓解,但无法根除。
- 单正例标注会低估 Recall(同一问题可能有多个相关 chunk 被判为未命中)。
- 标注集规模 100、覆盖 28 份现行有效文档,适合方向性对比,
  尚不足以支撑跨数据集的强统计结论。

## 复现说明

- `eval_set_sample.jsonl` 为 20 条标注集样本,用于展示格式。
- 完整 100 条标注集基于全量数据生成;全量法规数据仅供个人学习,未公开,
  因此样本中的 `golden_chunk_id` 无法直接在公开 20 条 chunk 上检索复现。
- 在自有全量库上,可用 `build_eval_set.py` 重新生成标注集、`run_eval.py` 复现评测。

## 文件说明

- `build_eval_set.py` —— 标注集构造(需 DeepSeek API key,读环境变量 `DEEPSEEK_API_KEY`)
- `run_eval.py` —— 四配置评测,输出指标表
- `eval_set_sample.jsonl` —— 20 条标注集样本(query + golden_chunk_id + doc 元信息)