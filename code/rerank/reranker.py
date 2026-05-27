"""
Cross-Encoder 重排器:基于 bge-reranker-v2-m3。

工业界 RAG 标准 pattern:
    召回阶段:HybridRetriever 召回 50 条(粗排,看 recall)
    精排阶段:Reranker 重排取 top-5(精排,看 precision)

输入是 HybridRetriever 的输出格式(list[dict]),
输出也是同样的 list[dict],只是顺序变了 + 多一个 'rerank_score' 字段。
"""
from pathlib import Path
from typing import Optional

import torch
from sentence_transformers import CrossEncoder


DEFAULT_RERANKER_PATH = str(Path.home() / "models" / "bge-reranker-v2-m3")


class Reranker:
    def __init__(
        self,
        model_path: str = DEFAULT_RERANKER_PATH,
        batch_size: int = 32,
        device: Optional[str] = None,
        max_length: int = 512,
    ):
        """
        Args:
            model_path: 本地 bge-reranker-v2-m3 路径
            batch_size: 推理 batch 大小
            device: 'cuda' / 'cpu' / None(自动)
            max_length: query+doc 拼接后的 token 上限,512 是 BERT 系标准
        """
        self.model_path = model_path
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.model: Optional[CrossEncoder] = None

    # ---------- 显式加载 ----------
    def load(self) -> "Reranker":
        self.model = CrossEncoder(
            self.model_path,
            device=self.device,
            max_length=self.max_length,
        )
        print(f"[reranker] 模型已加载: {self.model_path}")
        print(f"[reranker] device={self.device}, max_length={self.max_length}")
        return self

    # ---------- 核心:重排 ----------
    def rerank(
        self,
        query: str,
        docs: list[dict],
        top_k: int = 5,
        use_sigmoid: bool = True,
    ) -> list[dict]:
        """
        对 docs 列表按 (query, chunk_text) 重新打分排序。

        Args:
            query: 用户查询
            docs: HybridRetriever 输出的 list[dict],必须含 'chunk_text'
            top_k: 重排后返回前 K 条
            use_sigmoid: True 返回 0-1 概率,False 返回原始 logit
        Returns:
            排序后的 list[dict],每条增加 'rerank_score' 字段,
            原 'score'(召回分)保留为 'recall_score'
        """
        if not docs:
            return []

        assert self.model is not None, "请先调用 load()"

        # 构造 (query, doc_text) pair list
        pairs = [(query, d["chunk_text"]) for d in docs]

        # CrossEncoder.predict 内部已批处理
        activation_fn = torch.nn.Sigmoid() if use_sigmoid else None
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            activation_fn=activation_fn,
            convert_to_numpy=True,
        )

        # 组合 + 排序
        scored = []
        for d, s in zip(docs, scores):
            new_d = dict(d)
            new_d["recall_score"] = new_d.get("score")  # 留个底,知道召回阶段分数
            new_d["rerank_score"] = float(s)
            new_d["score"] = float(s)  # 主分数字段也更新,方便下游统一处理
            scored.append(new_d)

        scored.sort(key=lambda x: -x["rerank_score"])
        return scored[:top_k]


# ---------- 自检(冒烟测试) ----------
if __name__ == "__main__":
    reranker = Reranker().load()

    # 构造一个简单测试:1 个正例 + 3 个负例
    query = "飞行员体检合格证有效期"
    fake_docs = [
        {"chunk_text": "申请人具有第一级或第二级体检合格证,有效期24个月。", "score": 0.0},  # 正例
        {"chunk_text": "飞机结构必须能够承受极限载荷至少三秒钟。", "score": 0.0},          # 负例
        {"chunk_text": "运输类旋翼航空器适航规定与CCAR-29-R1相关。", "score": 0.0},        # 负例
        {"chunk_text": "体检合格证的颁发流程包括眼科、内科等专科检查。", "score": 0.0},   # 弱正例
    ]

    print(f"\n[smoke test] query: {query}")
    results = reranker.rerank(query, fake_docs, top_k=4)

    print("\n--- 重排结果(按 rerank_score 降序)---")
    for i, r in enumerate(results, 1):
        print(f"  [{i}] rerank_score={r['rerank_score']:.4f}")
        print(f"      {r['chunk_text']}")