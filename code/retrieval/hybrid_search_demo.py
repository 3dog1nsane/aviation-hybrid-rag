"""
混合检索对比演示:7 个典型 query,并列展示 dense / sparse / hybrid 三路结果。

目的:
1. 直观验证三种检索的差异和互补性
2. 为面试/简历准备讲点(每个 query 都对应一个场景)

7 个 query 的设计意图(见 main 函数注释)
"""
from hybrid_retriever import HybridRetriever


# ---------- 测试 query 设计 ----------
# (query, 设计意图/预期更强的方法)
TEST_QUERIES = [
    ("运输类飞机的适航标准",
     "基线:关键词+语义都对得上,三路应都能命中"),
    ("CCAR-25 关于飞机结构强度的要求",
     "强标识符:法规号 CCAR-25,BM25 应能精确匹配"),
    ("飞机零件出问题了怎么报告",
     "口语化:无法规术语,dense 应更强(理解'出问题→故障/困难')"),
    ("飞行员的视力要求",
     "同义词:'视力'在法规里可能写成'视觉敏锐度'等,dense 更强"),
    ("运营人证书 有效期",
     "多关键词精确匹配:每个词都重要,sparse 应擅长"),
    ("无人机驾驶员管理",
     "较新概念:措辞可能不一致,hybrid 应最稳"),
    ("航油的危险品分类",
     "跨域术语:'航油'='航空燃料',BM25 不识别同义,dense 强"),
]

TOP_K = 5
RECALL_K = 20


# ---------- 打印工具 ----------
def print_results(label: str, results: list[dict]) -> None:
    """紧凑展示一路检索结果"""
    print(f"\n  [{label}] (返回 {len(results)} 条)")
    if not results:
        print("    (无命中)")
        return
    for i, r in enumerate(results, 1):
        ccar = r["ccar_no"] or "—"
        title = (r["doc_title"] or "")[:32]
        text_preview = r["chunk_text"][:60].replace("\n", " ")
        print(f"    [{i}] score={r['score']:.4f} | {ccar:<12} | {title}")
        print(f"        {text_preview}...")


def print_overlap_summary(
    dense_results: list[dict],
    sparse_results: list[dict],
    hybrid_results: list[dict],
) -> None:
    """统计三路 top-K 的 chunk_id 重叠情况,直观看互补性"""
    d_ids = {r["chunk_id"] for r in dense_results}
    s_ids = {r["chunk_id"] for r in sparse_results}
    h_ids = {r["chunk_id"] for r in hybrid_results}

    overlap_ds = d_ids & s_ids
    only_d = d_ids - s_ids
    only_s = s_ids - d_ids

    print(f"\n  [重叠分析]")
    print(f"    dense ∩ sparse  : {len(overlap_ds)} / {TOP_K}  "
          f"(dense 独有 {len(only_d)}, sparse 独有 {len(only_s)})")
    print(f"    hybrid 来源     : 来自 dense top-{TOP_K} {len(h_ids & d_ids)} 个,"
          f"来自 sparse top-{TOP_K} {len(h_ids & s_ids)} 个,"
          f"来自更深召回 {len(h_ids - d_ids - s_ids)} 个")


# ---------- 主流程 ----------
def main():
    retriever = HybridRetriever().load()
    print(f"\n{'#'*70}")
    print(f"# 混合检索三路对比 (top_k={TOP_K}, recall_k={RECALL_K})")
    print(f"# 共 {len(TEST_QUERIES)} 个测试 query")
    print(f"{'#'*70}")

    for idx, (query, intent) in enumerate(TEST_QUERIES, 1):
        print(f"\n\n{'='*70}")
        print(f"Query {idx}/{len(TEST_QUERIES)}: 「{query}」")
        print(f"设计意图: {intent}")
        print(f"{'='*70}")

        dense_r = retriever.search_dense(query, top_k=TOP_K)
        sparse_r = retriever.search_sparse(query, top_k=TOP_K)
        hybrid_r = retriever.search_hybrid(
            query, top_k=TOP_K, recall_k=RECALL_K
        )

        print_results("Dense  ", dense_r)
        print_results("Sparse ", sparse_r)
        print_results("Hybrid ", hybrid_r)
        print_overlap_summary(dense_r, sparse_r, hybrid_r)


if __name__ == "__main__":
    main()