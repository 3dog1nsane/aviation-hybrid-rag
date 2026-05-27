"""
混合检索器:封装 dense + sparse + hybrid 三路检索。

依赖 Milvus collection 'ccar_hybrid'(05_milvus_insert.py 灌好的数据)
+ bge-m3 dense 模型(/home/pyb/models/bge-m3)
+ BM25Encoder(bm25_encoder.pkl)

使用方式:
    retriever = HybridRetriever()
    retriever.load()                        # 加载模型 + 连 Milvus
    results = retriever.search_hybrid("查询文本", top_k=5)
"""
from pathlib import Path
from typing import Literal, Optional

import torch
from pymilvus import AnnSearchRequest, Collection, RRFRanker, connections
from sentence_transformers import SentenceTransformer

from bm25_encoder import BM25Encoder


# ---------- 默认配置(允许实例化时覆盖) ----------
DEFAULT_MILVUS_HOST = "localhost"
DEFAULT_MILVUS_PORT = "19530"
DEFAULT_COLLECTION_NAME = "ccar_hybrid"
DEFAULT_DENSE_MODEL_PATH = str(Path.home() / "models" / "bge-m3")
_PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_BM25_PATH = str(_PROJECT_ROOT / "data" / "bm25_encoder.pkl")

# 检索结果输出哪些字段(从 Milvus 拉回来)
OUTPUT_FIELDS = ["chunk_id", "chunk_text", "doc_title", "ccar_no", "validity"]


class HybridRetriever:
    def __init__(
        self,
        milvus_host: str = DEFAULT_MILVUS_HOST,
        milvus_port: str = DEFAULT_MILVUS_PORT,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        dense_model_path: str = DEFAULT_DENSE_MODEL_PATH,
        bm25_path: str = DEFAULT_BM25_PATH,
    ):
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.collection_name = collection_name
        self.dense_model_path = dense_model_path
        self.bm25_path = bm25_path

        # 延迟加载,避免 import 这个模块就连 Milvus
        self.collection: Optional[Collection] = None
        self.dense_model: Optional[SentenceTransformer] = None
        self.bm25: Optional[BM25Encoder] = None

    # ---------- 初始化(显式调用) ----------
    def load(self) -> "HybridRetriever":
        """连 Milvus + 加载模型。分开是为了让上层控制时机。"""
        # 1. 连 Milvus
        connections.connect(
            alias="default",
            host=self.milvus_host,
            port=self.milvus_port,
        )
        self.collection = Collection(self.collection_name)
        self.collection.load()
        print(f"[retriever] Milvus 已连接,collection={self.collection_name},"
              f"entity 数={self.collection.num_entities}")

        # 2. 加载 dense 模型
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dense_model = SentenceTransformer(self.dense_model_path, device=device)
        self.dense_model.max_seq_length = 1024
        print(f"[retriever] dense 模型已加载({device}, max_seq_length=1024)")

        # 3. 加载 BM25
        self.bm25 = BM25Encoder()
        self.bm25.load(self.bm25_path)
        print(f"[retriever] BM25 已加载,词表={len(self.bm25.vocab)}")

        return self

    # ---------- query 编码 ----------
    def _encode_dense_query(self, query: str) -> list[float]:
        """bge-m3 编码 query 成 1024 维 dense 向量"""
        vec = self.dense_model.encode(
            query,
            normalize_embeddings=True,    # ★ 配合 IP 度量
            convert_to_numpy=True,
        )
        return vec.tolist()

    def _encode_sparse_query(self, query: str) -> dict[int, float]:
        """BM25 编码 query 成稀疏向量"""
        return self.bm25.encode_query(query)

    # ---------- 三种检索 ----------
    def search_dense(
        self,
        query: str,
        top_k: int = 5,
        filter_expr: Optional[str] = None,
    ) -> list[dict]:
        """纯 dense 检索"""
        q_vec = self._encode_dense_query(query)
        results = self.collection.search(
            data=[q_vec],
            anns_field="dense_vec",
            param={"metric_type": "IP", "params": {"ef": 64}},  # HNSW 查询参数
            limit=top_k,
            expr=filter_expr,
            output_fields=OUTPUT_FIELDS,
        )
        return self._format_results(results[0])

    def search_sparse(
        self,
        query: str,
        top_k: int = 5,
        filter_expr: Optional[str] = None,
    ) -> list[dict]:
        """纯 sparse(BM25) 检索"""
        q_vec = self._encode_sparse_query(query)
        if not q_vec:
            # query 分词后所有 token 都 OOV,返回空
            return []
        results = self.collection.search(
            data=[q_vec],
            anns_field="sparse_vec",
            param={"metric_type": "IP", "params": {}},
            limit=top_k,
            expr=filter_expr,
            output_fields=OUTPUT_FIELDS,
        )
        return self._format_results(results[0])

    def search_hybrid(
        self,
        query: str,
        top_k: int = 5,
        recall_k: int = 20,
        rrf_k: int = 60,
        filter_expr: Optional[str] = None,
    ) -> list[dict]:
        """混合检索:dense + sparse,RRF 融合

        Args:
            top_k: 最终返回数量
            recall_k: 每路召回数量(通常远大于 top_k)
            rrf_k: RRF 算法的平滑参数,默认 60
        """
        dense_vec = self._encode_dense_query(query)
        sparse_vec = self._encode_sparse_query(query)

        # 构造两个 ANN 请求
        dense_req = AnnSearchRequest(
            data=[dense_vec],
            anns_field="dense_vec",
            param={"metric_type": "IP", "params": {"ef": 64}},
            limit=recall_k,
            expr=filter_expr,
        )

        if sparse_vec:
            sparse_req = AnnSearchRequest(
                data=[sparse_vec],
                anns_field="sparse_vec",
                param={"metric_type": "IP", "params": {}},
                limit=recall_k,
                expr=filter_expr,
            )
            reqs = [sparse_req, dense_req]
        else:
            # 全 OOV,只能走 dense
            reqs = [dense_req]

        results = self.collection.hybrid_search(
            reqs=reqs,
            rerank=RRFRanker(k=rrf_k),
            limit=top_k,
            output_fields=OUTPUT_FIELDS,
        )
        return self._format_results(results[0])

    # ---------- 结果格式化 ----------
    @staticmethod
    def _format_results(milvus_hits) -> list[dict]:
        """把 Milvus Hits 转成 list[dict],便于打印/序列化"""
        out = []
        for hit in milvus_hits:
            row = {
                "score": float(hit.distance),  # Milvus 用 distance 字段存分数
                "chunk_id": hit.entity.get("chunk_id"),
                "doc_title": hit.entity.get("doc_title"),
                "ccar_no": hit.entity.get("ccar_no"),
                "validity": hit.entity.get("validity"),
                "chunk_text": hit.entity.get("chunk_text"),
            }
            out.append(row)
        return out


# ---------- 自检 ----------
if __name__ == "__main__":
    # 简单冒烟测试
    retriever = HybridRetriever().load()
    query = "运输类飞机的适航标准"
    print(f"\n[smoke test] query: {query}")
    print("\n--- dense ---")
    for r in retriever.search_dense(query, top_k=3):
        print(f"  score={r['score']:.4f} | {r['ccar_no']} | {r['doc_title'][:30]}")
    print("\n--- sparse ---")
    for r in retriever.search_sparse(query, top_k=3):
        print(f"  score={r['score']:.4f} | {r['ccar_no']} | {r['doc_title'][:30]}")
    print("\n--- hybrid (RRF) ---")
    for r in retriever.search_hybrid(query, top_k=3):
        print(f"  score={r['score']:.4f} | {r['ccar_no']} | {r['doc_title'][:30]}")