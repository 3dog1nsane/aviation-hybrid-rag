"""
建立 Milvus collection,定义 schema + 索引。

设计要点:
1. dense_vec(1024 维, bge-m3) + sparse_vec(BM25 自实现)
2. validity 字段保留,后面 metadata filter 演示用
3. 索引:dense 用 HNSW,sparse 用 SPARSE_INVERTED_INDEX,都用 IP 度量
4. 这个脚本只建结构,不灌数据。重跑会先 drop 再建(开发期方便)。

依赖: pymilvus
"""
from pymilvus import (
    connections,
    utility,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
)


# ---------- 配置 ----------
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "ccar_hybrid"
DENSE_DIM = 1024  # bge-m3


# ---------- Schema 定义 ----------
def build_schema() -> CollectionSchema:
    fields = [
        FieldSchema(
            name="pk",
            dtype=DataType.INT64,
            is_primary=True,
            auto_id=True,
        ),
        FieldSchema(
            name="chunk_id",
            dtype=DataType.VARCHAR,
            max_length=64,
        ),
        FieldSchema(
            name="chunk_text",
            dtype=DataType.VARCHAR,
            max_length=4096,
        ),
        FieldSchema(
            name="doc_id",
            dtype=DataType.VARCHAR,
            max_length=64,
        ),
        FieldSchema(
            name="doc_title",
            dtype=DataType.VARCHAR,
            max_length=256,
        ),
        FieldSchema(
            name="ccar_no",
            dtype=DataType.VARCHAR,
            max_length=32,
        ),
        FieldSchema(
            name="validity",
            dtype=DataType.VARCHAR,
            max_length=16,
        ),
        FieldSchema(
            name="dense_vec",
            dtype=DataType.FLOAT_VECTOR,
            dim=DENSE_DIM,
        ),
        FieldSchema(
            name="sparse_vec",
            dtype=DataType.SPARSE_FLOAT_VECTOR,
        ),
    ]

    schema = CollectionSchema(
        fields=fields,
        description="CCAR 法规混合检索 collection",
        enable_dynamic_field=False,  # 不开动态字段,schema 固定
    )
    return schema


# ---------- 创建索引 ----------
def create_indexes(collection: Collection) -> None:
    # dense 向量索引: HNSW + IP
    dense_index = {
        "index_type": "HNSW",
        "metric_type": "IP",
        "params": {
            "M": 16,
            "efConstruction": 64,
        },
    }
    collection.create_index(
        field_name="dense_vec",
        index_params=dense_index,
        index_name="dense_idx",
    )
    print("[index] dense_vec 索引创建完成 (HNSW, IP)")

    # sparse 向量索引: SPARSE_INVERTED_INDEX + IP
    sparse_index = {
        "index_type": "SPARSE_INVERTED_INDEX",
        "metric_type": "IP",
        "params": {},
    }
    collection.create_index(
        field_name="sparse_vec",
        index_params=sparse_index,
        index_name="sparse_idx",
    )
    print("[index] sparse_vec 索引创建完成 (SPARSE_INVERTED_INDEX, IP)")


# ---------- 主流程 ----------
def main():
    # 1. 连接
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
    print(f"[milvus] 已连接 {MILVUS_HOST}:{MILVUS_PORT}")

    # 2. 已存在则删除(开发期方便,生产环境绝对不这么干)
    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)
        print(f"[milvus] 已删除旧 collection: {COLLECTION_NAME}")

    # 3. 建 collection
    schema = build_schema()
    collection = Collection(
        name=COLLECTION_NAME,
        schema=schema,
        consistency_level="Strong",
    )
    print(f"[milvus] collection '{COLLECTION_NAME}' 创建成功")
    print(f"         字段: {[f.name for f in schema.fields]}")

    # 4. 建索引
    create_indexes(collection)

    # 5. 加载到内存(否则不能搜索;灌数据前 load 也没数据,
    #    但这里 load 是为了让 Attu 立刻显示 Loaded 状态)
    collection.load()
    print(f"[milvus] collection 已 load")

    print(f"\n✅ 完成。下一步运行 05_milvus_insert.py 灌数据。")


if __name__ == "__main__":
    main()