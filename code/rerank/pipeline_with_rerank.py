"""
完整管线演示:HybridRetriever 召回 → Reranker 精排

工业界 RAG 标准 pattern:
    召回:hybrid_search(top_k=50, recall_k=50)  # 看 recall,宁多勿少
    精排:rerank(top_k=5)                        # 看 precision,严挑细选

每个 query 输出:
    1. 召回 top-5(看混合检索结果)
    2. 重排 top-5(看精排后结果)
    3. 排名变化分析(召回 → 重排有哪些新进入/被踢出)
    4. Rerank 耗时
"""
import sys
import time
from pathlib import Path

# 把兄弟目录加入 path,以 import HybridRetriever
HYBRID_DIR = Path(__file__).parent.parent / "retrieval"
sys.path.insert(0, str(HYBRID_DIR))

from hybrid_retriever import HybridRetriever  # noqa: E402
from reranker import Reranker  # noqa: E402


# ---------- 配置 ----------
RECALL_K = 50      # 召回数量(给 Rerank 提供候选池)
TOP_K = 5          # 最终展示数量

# 复用 06 demo 的 query 集,便于对比
TEST_QUERIES = [
    ("运输类飞机的适航标准",
     "基线:关键词+语义都对得上"),
    ("CCAR-25 关于飞机结构强度的要求",
     "强标识符:Rerank 能否救回 CCAR-25?"),
    ("飞机零件出问题了怎么报告",
     "口语化:Rerank 能否进一步聚焦'使用困难报告'?"),
    ("飞行员的视力要求",
     "同义词:Rerank 应能识别'视觉/视力/裸视'相关性"),
    ("运营人证书 有效期",
     "多关键词:Rerank 应排序更精准"),
    ("无人机驾驶员管理",
     "概念明确:Rerank 应锦上添花"),
    ("航油的危险品分类",
     "数据缺失场景:Rerank 是否敢给低分(no-answer signal)?"),
]


# ---------- 打印工具 ----------
def print_results(label: str, results: list[dict], score_key: str) -> None:
    """紧凑打印一路结果"""
    print(f"\n  [{label}] (共 {len(results)} 条)")
    if not results:
        print("    (无命中)")
        return
    for i, r in enumerate(results, 1):
        ccar = r.get("ccar_no") or "—"
        title = (r.get("doc_title") or "")[:32]
        text_preview = (r.get("chunk_text") or "")[:60].replace("\n", " ")
        score = r.get(score_key)
        print(f"    [{i}] {score_key}={score:.4f} | {ccar:<12} | {title}")
        print(f"        {text_preview}...")


def print_rank_change(
    recall_topk: list[dict],
    rerank_topk: list[dict],
) -> None:
    """对比召回 top-5 和重排 top-5 的差异"""
    recall_ids = [r["chunk_id"] for r in recall_topk]
    rerank_ids = [r["chunk_id"] for r in rerank_topk]

    kept = [cid for cid in rerank_ids if cid in recall_ids]
    promoted = [cid for cid in rerank_ids if cid not in recall_ids]
    demoted = [cid for cid in recall_ids if cid not in rerank_ids]

    print(f"\n  [排名变化]")
    print(f"    召回 top-5 保留进重排 top-5 : {len(kept)} / {TOP_K}")
    print(f"    重排从更深候选拉上来       : {len(promoted)} / {TOP_K}  (说明 hybrid 召回 top-5 漏了真答案)")
    print(f"    召回 top-5 被重排踢出       : {len(demoted)} / {TOP_K}")

    # 看 Top-1 的变化
    if recall_topk and rerank_topk:
        if recall_ids[0] == rerank_ids[0]:
            print(f"    Top-1 一致:重排确认了召回的头名")
        else:
            print(f"    Top-1 变化:召回头名 → 重排头名(reranker 改判)")


# ---------- 主流程 ----------
def main():
    print("初始化检索器和重排器...")
    retriever = HybridRetriever().load()
    reranker = Reranker().load()

    print(f"\n{'#'*72}")
    print(f"# 召回 + Rerank 管线演示")
    print(f"# 召回 recall_k={RECALL_K},重排 top_k={TOP_K}")
    print(f"# 共 {len(TEST_QUERIES)} 个测试 query")
    print(f"{'#'*72}")

    # 统计 Rerank 耗时(用于面试讲点)
    rerank_latencies_ms = []

    for idx, (query, intent) in enumerate(TEST_QUERIES, 1):
        print(f"\n\n{'='*72}")
        print(f"Query {idx}/{len(TEST_QUERIES)}: 「{query}」")
        print(f"考察点: {intent}")
        print(f"{'='*72}")

        # 1. 召回 50 条(用 hybrid)
        recall_results = retriever.search_hybrid(
            query, top_k=RECALL_K, recall_k=RECALL_K
        )
        recall_topk = recall_results[:TOP_K]  # 召回的 top-5,作为基线对照

        # 2. Rerank 50 条 → top-5
        t0 = time.perf_counter()
        rerank_topk = reranker.rerank(query, recall_results, top_k=TOP_K)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        rerank_latencies_ms.append(elapsed_ms)

        # 3. 输出对比
        print_results("Hybrid 召回 top-5", recall_topk, score_key="score")
        # 注意:rerank 后 'score' 字段已被覆盖为 rerank_score,这里直接展示
        print_results("Rerank 后 top-5  ", rerank_topk, score_key="rerank_score")
        print_rank_change(recall_topk, rerank_topk)

        print(f"\n  [性能] Rerank {RECALL_K} 条耗时: {elapsed_ms:.1f} ms")

    # 汇总
    print(f"\n\n{'#'*72}")
    print(f"# 性能汇总")
    print(f"{'#'*72}")
    print(f"Rerank 平均耗时: {sum(rerank_latencies_ms)/len(rerank_latencies_ms):.1f} ms")
    print(f"Rerank 最大耗时: {max(rerank_latencies_ms):.1f} ms")
    print(f"Rerank 最小耗时: {min(rerank_latencies_ms):.1f} ms")


if __name__ == "__main__":
    main()