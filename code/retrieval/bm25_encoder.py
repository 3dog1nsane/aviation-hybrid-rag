"""
BM25 编码器(手写实现)

用途:把文本编码成稀疏向量 {token_id: weight},供 Milvus 灌库 / 检索使用。

设计要点:
1. encode_doc 用 BM25 文档侧公式(tf 饱和 + 长度归一化)
2. encode_query 用 IDF 权重
3. query · doc(内积) ≈ BM25 打分
4. jieba 分词,加载航空术语自定义词典
5. 不做停用词过滤,靠 IDF 自动降权(停用词在所有文档出现,IDF→0)

依赖: jieba, numpy
"""
import json
import math
import pickle
from collections import Counter
from pathlib import Path
from typing import Iterable

import jieba
import numpy as np


class BM25Encoder:
    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        userdict_path: str | None = None,
    ):
        """
        Args:
            k1: 词频饱和参数,经典默认 1.5
            b:  长度归一化强度,经典默认 0.75
            userdict_path: jieba 自定义词典路径(航空术语)
        """
        self.k1 = k1
        self.b = b

        # 加载航空术语词典(很重要,否则"航空器适航审定"会被切碎)
        if userdict_path:
            jieba.load_userdict(userdict_path)
            print(f"[BM25Encoder] 已加载自定义词典: {userdict_path}")

        # fit 后填充
        self.vocab: dict[str, int] = {}        # token -> token_id
        self.idf: dict[int, float] = {}        # token_id -> IDF
        self.avgdl: float = 0.0                # 平均文档长度
        self.doc_count: int = 0                # 文档总数
        self._is_fitted: bool = False

    # ---------- 分词 ----------
    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """jieba 精确模式分词,去掉纯空白 token"""
        return [tok for tok in jieba.lcut(text) if tok.strip()]

    # ---------- 训练 ----------
    def fit(self, corpus: Iterable[str]) -> "BM25Encoder":
        """
        在全语料上训练:建词表 + 算 IDF + 算 avgdl

        BM25 的 IDF 公式(Robertson-Sparck Jones 平滑版):
            IDF(t) = log( (N - df + 0.5) / (df + 0.5) + 1 )
        其中 N 是总文档数,df 是包含 token t 的文档数。
        加 1 是为了保证 IDF 非负(常见停用词 df≈N 时,值接近 0 而不是负数)。
        """
        corpus = list(corpus)
        self.doc_count = len(corpus)

        # 第一遍:分词 + 统计 df(每个 token 在多少文档出现)
        df_counter: Counter[str] = Counter()
        doc_lengths: list[int] = []
        tokenized_docs: list[list[str]] = []

        for doc in corpus:
            tokens = self._tokenize(doc)
            tokenized_docs.append(tokens)
            doc_lengths.append(len(tokens))

            # df 用 set,避免同一文档多次计数
            for tok in set(tokens):
                df_counter[tok] += 1

        self.avgdl = sum(doc_lengths) / max(self.doc_count, 1)

        # 建词表:按出现频次排序,频次高的 token_id 小(无功能影响,纯习惯)
        sorted_tokens = sorted(df_counter.keys(), key=lambda t: -df_counter[t])
        self.vocab = {tok: idx for idx, tok in enumerate(sorted_tokens)}

        # 算 IDF
        N = self.doc_count
        for tok, df in df_counter.items():
            tok_id = self.vocab[tok]
            self.idf[tok_id] = math.log((N - df + 0.5) / (df + 0.5) + 1)

        self._is_fitted = True
        print(
            f"[BM25Encoder] fit 完成: 文档数={N}, 词表大小={len(self.vocab)}, "
            f"平均文档长度={self.avgdl:.1f}"
        )
        return self

    # ---------- 文档编码 ----------
    def encode_doc(self, text: str) -> dict[int, float]:
        """
        文档侧编码(BM25 文档项),输出 {token_id: weight}

        文档项 = (k1 + 1) * tf / (tf + k1 * (1 - b + b * |D| / avgdl))

        注意:这里只算文档项,不乘 IDF。
        IDF 留给 query 侧,这样 query · doc 内积 = 完整 BM25 score。
        """
        assert self._is_fitted, "请先 fit"
        tokens = self._tokenize(text)
        dl = len(tokens)
        tf_counter = Counter(tokens)

        sparse: dict[int, float] = {}
        # 长度归一化项
        norm = 1 - self.b + self.b * (dl / self.avgdl) if self.avgdl > 0 else 1.0

        for tok, tf in tf_counter.items():
            tok_id = self.vocab.get(tok)
            if tok_id is None:
                continue  # OOV(未登录词)直接丢弃
            weight = (self.k1 + 1) * tf / (tf + self.k1 * norm)
            sparse[tok_id] = float(weight)

        return sparse

    # ---------- 查询编码 ----------
    def encode_query(self, text: str) -> dict[int, float]:
        """
        查询侧编码(IDF),输出 {token_id: idf_weight}

        查询里同一个词出现多次只算一次(BM25 标准做法,简化版本)。
        """
        assert self._is_fitted, "请先 fit"
        tokens = set(self._tokenize(text))  # 去重

        sparse: dict[int, float] = {}
        for tok in tokens:
            tok_id = self.vocab.get(tok)
            if tok_id is None:
                continue
            sparse[tok_id] = float(self.idf[tok_id])

        return sparse

    # ---------- 持久化 ----------
    def save(self, path: str) -> None:
        """保存训练好的状态(词表 + IDF + avgdl + 超参)"""
        state = {
            "k1": self.k1,
            "b": self.b,
            "vocab": self.vocab,
            "idf": self.idf,
            "avgdl": self.avgdl,
            "doc_count": self.doc_count,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(state, f)
        print(f"[BM25Encoder] 已保存到 {path}")

    def load(self, path: str) -> "BM25Encoder":
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.k1 = state["k1"]
        self.b = state["b"]
        self.vocab = state["vocab"]
        self.idf = state["idf"]
        self.avgdl = state["avgdl"]
        self.doc_count = state["doc_count"]
        self._is_fitted = True
        print(f"[BM25Encoder] 已从 {path} 加载,词表={len(self.vocab)}")
        return self