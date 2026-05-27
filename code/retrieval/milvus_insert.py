"""
把 5372 条 chunks 灌入 Milvus collection 'ccar_hybrid'。

流程:
1. 加载 chunks JSONL
2. 加载训练好的 BM25Encoder(从 pkl)
3. 加载 bge-m3 模型(首次会下载 ~2.3GB)
4. 批量算 dense + sparse 向量
5. 批量 insert 到 Milvus

依赖: pymilvus, sentence-transformers, torch
"""
import os
# 先设环境变量,再 import 其他东西(尤其 HF 相关)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import json
from pathlib import Path

import torch
from pymilvus import Collection, connections
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# ---------- 配置 ----------
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

CHUNKS_PATH = DATA_DIR / "samples" / "ccar_chunks_sample.jsonl"
BM25_PATH = DATA_DIR / "bm25_encoder.pkl"

MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "ccar_hybrid"

DENSE_MODEL_NAME = str(Path.home() / "models" / "bge-m3")
DENSE_BATCH_SIZE = 32        # GPU batch
INSERT_BATCH_SIZE = 500      # Milvus insert batch

MAX_TEXT_LEN = 4096   # 字节数,与 schema 对齐;不再用字符数截断

# 字段长度上限(对应 schema)
MAX_LEN = {
    "chunk_id": 64,
    "doc_id": 64,
    "doc_title": 256,
    "ccar_no": 32,
    "validity": 16,
}


# ---------- 加载 BM25Encoder(动态 import 02_bm25_encoder.py) ----------
def load_bm25_encoder():
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from bm25_encoder import BM25Encoder
    encoder = BM25Encoder()
    encoder.load(str(BM25_PATH))
    return encoder


# ---------- 加载 chunks ----------
def load_chunks() -> list[dict]:
    chunks = []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    print(f"[load_chunks] 加载 {len(chunks)} 条 chunk")
    return chunks


# ---------- 工具:安全截断字符串字段 ----------
def safe_truncate(value, max_len: int) -> str:
    """处理 None、超长、非字符串"""
    if value is None:
        return ""
    s = str(value)
    if len(s) > max_len:
        return s[:max_len]
    return s


def safe_truncate_bytes(value, max_bytes: int) -> str:
    """按 UTF-8 字节数安全截断,不会破坏多字节字符。
    Milvus VARCHAR 的 max_length 单位是字节,中文 1 字符=3 字节,
    所以不能用 Python 的字符切片。
    """
    if value is None:
        return ""
    s = str(value)
    encoded = s.encode("utf-8")
    if len(encoded) <= max_bytes:
        return s
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


# ---------- 主流程 ----------
def main():
    # 1. 连接 Milvus
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
    collection = Collection(COLLECTION_NAME)
    print(f"[milvus] 连接 collection '{COLLECTION_NAME}',当前 entity 数: {collection.num_entities}")

    if collection.num_entities > 0:
        print(f"⚠️  collection 已有数据 ({collection.num_entities} 条)")
        ans = input("继续会重复插入,是否继续? [y/N]: ").strip().lower()
        if ans != "y":
            print("已取消")
            return

    # 2. 加载数据
    chunks = load_chunks()
    bm25 = load_bm25_encoder()

    # 3. 加载 dense 模型
    print(f"[dense] 加载模型 {DENSE_MODEL_NAME} ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[dense] device = {device}")
    model = SentenceTransformer(DENSE_MODEL_NAME, device=device)
    # bge-m3 默认最大 8192 tokens,我们 chunk 最长 800 字,够用
    # 显式设置一下,防止默认值变化
    model.max_seq_length = 1024
    print(f"[dense] max_seq_length = {model.max_seq_length}")

    # 4. 批量算 dense 向量
    texts = [safe_truncate_bytes(c["chunk_text"], MAX_TEXT_LEN) for c in chunks]
    print(f"[dense] 开始编码 {len(texts)} 条文本...")
    dense_vecs = model.encode(
        texts,
        batch_size=DENSE_BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,   # ★ bge-m3 配 IP 度量必须 True
        convert_to_numpy=True,
    )
    print(f"[dense] 输出 shape: {dense_vecs.shape}")  # 期望 (5372, 1024)

    # 5. 批量算 sparse 向量
    print(f"[sparse] 编码 {len(texts)} 条文本...")
    sparse_vecs = []
    for t in tqdm(texts):
        sparse_vecs.append(bm25.encode_doc(t))
    # 自检
    nonzero = [len(v) for v in sparse_vecs[:100]]
    print(f"[sparse] 前 100 条非零数: min={min(nonzero)}, max={max(nonzero)}, avg={sum(nonzero)/len(nonzero):.1f}")

    # 6. 批量 insert
    print(f"[insert] 开始灌库,batch_size={INSERT_BATCH_SIZE}")
    total = len(chunks)
    for start in tqdm(range(0, total, INSERT_BATCH_SIZE)):
        end = min(start + INSERT_BATCH_SIZE, total)
        batch = chunks[start:end]

        # 按 schema 字段顺序构造数据(不含 auto_id 的 pk)
        data = [
            [safe_truncate(c.get("chunk_id"), MAX_LEN["chunk_id"]) for c in batch],            # chunk_id
            [safe_truncate_bytes(c["chunk_text"], MAX_TEXT_LEN) for c in batch],                # chunk_text
            [safe_truncate(c.get("doc_id"), MAX_LEN["doc_id"]) for c in batch],                # doc_id
            [safe_truncate(c.get("doc_title"), MAX_LEN["doc_title"]) for c in batch],          # doc_title
            [safe_truncate(c.get("ccar_no"), MAX_LEN["ccar_no"]) for c in batch],              # ccar_no
            [safe_truncate(c.get("validity"), MAX_LEN["validity"]) for c in batch],            # validity
            dense_vecs[start:end].tolist(),                                                     # dense_vec
            sparse_vecs[start:end],                                                             # sparse_vec
        ]
        collection.insert(data)

    # 7. flush 确保数据落盘
    print("[flush] 持久化数据...")
    collection.flush()
    print(f"[done] collection entity 数: {collection.num_entities}")

    # 8. 重新 load(数据变化后)
    collection.load()
    print(f"\n✅ 灌库完成,共 {collection.num_entities} 条")


if __name__ == "__main__":
    main()