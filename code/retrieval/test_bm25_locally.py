"""
本地验证 BM25Encoder:不依赖 Milvus,用 numpy 算内积做检索。

目的:
1. 确认编码器正确(IDF 合理、稀疏向量长度合理)
2. 跑几个真实 query,肉眼判断检索结果是否相关
3. 与全文 grep 对比(关键词命中是否排前)
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from bm25_encoder import BM25Encoder

# ---------- 配置 ----------
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# 默认用小样本数据(20 条),想用全量数据请按 README 说明准备
CHUNKS_PATH = DATA_DIR / "samples" / "ccar_chunks_sample.jsonl"
USERDICT_PATH = DATA_DIR / "aviation_dict.txt"
ENCODER_SAVE_PATH = DATA_DIR / "bm25_encoder.pkl"


# ---------- 加载数据 ----------
def load_chunks() -> list[dict]:
    chunks = []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    print(f"[load_chunks] 加载 {len(chunks)} 条 chunk")
    return chunks


# ---------- BM25 检索(纯 Python 内积) ----------
def bm25_search(
    encoder: BM25Encoder,
    doc_sparse_vecs: list[dict[int, float]],
    chunks: list[dict],
    query: str,
    top_k: int = 10,
) -> list[tuple[float, dict]]:
    """用 query 向量和所有 doc 向量做稀疏内积,返回 top_k"""
    q_vec = encoder.encode_query(query)
    if not q_vec:
        print(f"[search] query='{query}' 分词后无有效 token")
        return []

    scores = []
    for i, d_vec in enumerate(doc_sparse_vecs):
        # 稀疏内积:只遍历 query 的 token
        score = 0.0
        for tok_id, q_w in q_vec.items():
            if tok_id in d_vec:
                score += q_w * d_vec[tok_id]
        if score > 0:
            scores.append((score, chunks[i]))

    scores.sort(key=lambda x: -x[0])
    return scores[:top_k]


# ---------- 主流程 ----------
def main():
    chunks = load_chunks()
    corpus = [c["chunk_text"] for c in chunks]

    # 训练
    encoder = BM25Encoder(
        k1=1.5,
        b=0.75,
        userdict_path=str(USERDICT_PATH) if USERDICT_PATH.exists() else None,
    )
    encoder.fit(corpus)
    encoder.save(str(ENCODER_SAVE_PATH))

    # 全量编码文档
    print("[main] 编码所有文档...")
    doc_sparse_vecs = [encoder.encode_doc(text) for text in corpus]

    # 编码器自检
    sample_lens = [len(v) for v in doc_sparse_vecs[:100]]
    print(
        f"[main] 前 100 chunk 稀疏向量非零数: "
        f"min={min(sample_lens)}, max={max(sample_lens)}, "
        f"avg={sum(sample_lens)/len(sample_lens):.1f}"
    )

    # 看几个高/低 IDF 词,直观感受
    print("\n[main] IDF 抽样(token 频次降序):")
    id2tok = {v: k for k, v in encoder.vocab.items()}
    top_freq_ids = sorted(encoder.idf.keys())[:20]   # token_id 小 = 频次高
    bot_freq_ids = sorted(encoder.idf.keys())[-20:]  # token_id 大 = 频次低
    print("  高频词(低 IDF):", [(id2tok[i], round(encoder.idf[i], 2)) for i in top_freq_ids])
    print("  低频词(高 IDF):", [(id2tok[i], round(encoder.idf[i], 2)) for i in bot_freq_ids[:10]])

    # 跑几个真实 query
    queries = [
        "运输类飞机的适航标准",
        "航空器维修人员执照管理规定",
        "民用航空器事故调查",
        "CCAR-25 关于飞机结构强度的要求",
        "飞行员体检合格证有效期",
    ]

    for q in queries:
        print(f"\n{'='*60}")
        print(f"[Query] {q}")
        print(f"{'='*60}")
        results = bm25_search(encoder, doc_sparse_vecs, chunks, q, top_k=5)
        if not results:
            print("  无命中")
            continue
        for rank, (score, chunk) in enumerate(results, 1):
            print(f"  [{rank}] score={score:.3f} | {chunk.get('ccar_no', '')} | {chunk.get('doc_title', '')[:40]}")
            print(f"      {chunk['chunk_text'][:100].replace(chr(10), ' ')}...")


if __name__ == "__main__":
    main()