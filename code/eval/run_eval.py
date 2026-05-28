"""
检索评测:4 配置 × 3 指标。

配置:
  - dense        : 纯 bge-m3 dense
  - sparse       : 纯 BM25 sparse
  - hybrid       : dense + sparse RRF 融合
  - hybrid+rerank: hybrid 召回 50 → bge-reranker 精排

指标(单正例标注集):
  - Recall@K  : golden 是否出现在 top-K(单正例下等价 Hit@K)。算 @1 @5 @10
  - MRR@10    : golden 的倒数排名(没进 top-10 记 0)
  - NDCG@10   : 单正例下 = 1/log2(rank+1),没进 top-10 记 0

所有检索都加 filter:validity == "有效"(只检索有效库)。

依赖:本项目的 hybrid_retriever.py / reranker.py / bm25_encoder.py
     需在能 import 它们的环境跑(把本脚本放到 code/ 下,或调整 sys.path)

用法:
  python run_eval.py <eval_set.jsonl>
"""
import json
import math
import sys
from pathlib import Path

# ---- 让脚本能 import 到你的检索/精排模块(按你的目录结构调整)----
CODE_ROOT = Path(__file__).parent.parent  # 假设本脚本在 code/test/ 下
sys.path.insert(0, str(CODE_ROOT / "retrieval"))
sys.path.insert(0, str(CODE_ROOT / "rerank"))

from hybrid_retriever import HybridRetriever
from reranker import Reranker

VALID_FILTER = 'validity == "有效"'
RECALL_KS = [1, 5, 10]
RECALL_POOL = 50   # 召回多少条(hybrid/rerank 用)
RERANK_TOPK = 10


def load_eval_set(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def rank_of_golden(hits: list[dict], golden_id: str) -> int:
    """golden 在 hits 里的排名(1-based),没找到返回 0。"""
    for i, h in enumerate(hits, 1):
        if h.get("chunk_id") == golden_id:
            return i
    return 0


def metrics_from_rank(rank: int) -> dict:
    """从单个 golden 的排名算各指标的贡献。"""
    out = {}
    for k in RECALL_KS:
        out[f"recall@{k}"] = 1.0 if (0 < rank <= k) else 0.0
    out["mrr@10"] = (1.0 / rank) if (0 < rank <= 10) else 0.0
    out["ndcg@10"] = (1.0 / math.log2(rank + 1)) if (0 < rank <= 10) else 0.0
    return out


def eval_config(name, search_fn, eval_set, reranker=None):
    """对一种配置跑完整评测。search_fn(query) -> list[dict]"""
    agg = defaultdict(float)
    n = len(eval_set)
    for row in eval_set:
        q = row["query"]
        golden = row["golden_chunk_id"]
        hits = search_fn(q)
        if reranker is not None:
            hits = reranker.rerank(q, hits, top_k=RERANK_TOPK)
        rank = rank_of_golden(hits, golden)
        for k, v in metrics_from_rank(rank).items():
            agg[k] += v
    return {k: v / n for k, v in agg.items()}


from collections import defaultdict


def main(eval_path: str):
    eval_set = load_eval_set(eval_path)
    print(f"[load] 评测集: {len(eval_set)} 条 query\n")

    retriever = HybridRetriever().load()
    reranker = Reranker().load()

    configs = {
        "dense": lambda q: retriever.search_dense(q, top_k=max(RECALL_KS), filter_expr=VALID_FILTER),
        "sparse": lambda q: retriever.search_sparse(q, top_k=max(RECALL_KS), filter_expr=VALID_FILTER),
        "hybrid": lambda q: retriever.search_hybrid(q, top_k=max(RECALL_KS), recall_k=RECALL_POOL, filter_expr=VALID_FILTER),
    }

    results = {}
    for name, fn in configs.items():
        print(f"[eval] {name} ...")
        results[name] = eval_config(name, fn, eval_set)

    # hybrid + rerank:召回 50 条再精排
    print(f"[eval] hybrid+rerank ...")
    hybrid_pool_fn = lambda q: retriever.search_hybrid(
        q, top_k=RECALL_POOL, recall_k=RECALL_POOL, filter_expr=VALID_FILTER
    )
    results["hybrid+rerank"] = eval_config("hybrid+rerank", hybrid_pool_fn, eval_set, reranker=reranker)

    # ---- 打印结果表 ----
    metrics = [f"recall@{k}" for k in RECALL_KS] + ["mrr@10", "ndcg@10"]
    print("\n" + "=" * 70)
    header = f"{'config':<16}" + "".join(f"{m:>12}" for m in metrics)
    print(header)
    print("-" * 70)
    for name in ["dense", "sparse", "hybrid", "hybrid+rerank"]:
        row = results[name]
        line = f"{name:<16}" + "".join(f"{row[m]:>12.3f}" for m in metrics)
        print(line)
    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python run_eval.py <eval_set.jsonl>")
        sys.exit(1)
    main(sys.argv[1])