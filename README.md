# Aviation Hybrid RAG

[![Python](https://img.shields.io/badge/Python-3.12-blue)]()
[![Milvus](https://img.shields.io/badge/Milvus-2.6-orange)]()
[![Models](https://img.shields.io/badge/Models-bge--m3%20%7C%20bge--reranker--v2--m3-green)]()

> 面向中国民航 CCAR 法规的高精度检索系统。
> **Hybrid search (dense + sparse BM25) + Cross-Encoder rerank**,
> 工业级 RAG 检索管线的完整实现。

---

## 🎬 Demo

[![asciicast](https://asciinema.org/a/d40tPJGNwsFCnUwT.svg)](https://asciinema.org/a/d40tPJGNwsFCnUwT)

> 端到端问答:用户提问 → 混合检索 + Rerank → DeepSeek 生成回答 + 引用追溯 (citation)

---

## 🎯 项目特点

- **真实领域数据**:114 部 CCAR 民航法规、5372 个语义切片(repo 仅含 20 条样本)
- **手写 BM25 编码器**:jieba 中文分词 + 3296 词航空术语词典 + 自实现 IDF / 稀疏向量输出
- **混合检索**:Milvus 原生 `hybrid_search` + RRF 融合 (k=60),dense (bge-m3, 1024d) + sparse 双路
- **Cross-Encoder 精排**:bge-reranker-v2-m3,实测 Top-1 改判率 71%
- **No-answer detection**:利用 reranker 分数分布发现数据缺失场景(实测 0.05 vs 0.86~0.99,差一个数量级)
- **端到端 RAG 问答**:接入 DeepSeek API,实现 query → 答案 + **citation 引用追溯**(降低幻觉、可审计)

---

## 📊 关键指标

| 指标                           | 数值                                     |
| ------------------------------ | ---------------------------------------- |
| 文档规模                       | 114 法规 × 5372 chunks                   |
| BM25 词表大小                  | 35,117                                   |
| Dense 向量维度                 | 1024 (bge-m3)                            |
| Sparse 向量平均非零数          | 96                                       |
| Dense ∩ Sparse top-5 重叠率    | 0.4 / 5(7 个 query 中 6 个 0% 重叠)      |
| **Rerank Top-1 改判率**        | **71% (5/7)**                            |
| **真答案在召回 top-6~50 占比** | **80%**(验证"召回宽,精排严"必要性)       |
| **No-answer 阈值信号**         | rerank_score < 0.3(实测异常 query: 0.05) |
| Rerank 50 条耗时               | 1.3s (RTX 4080)                          |

---

## 🏗️ 架构

```text
┌─────────────┐    ┌──────────────────────────┐    ┌────────────────┐    ┌──────┐
│ User Query  │ -> │ Hybrid Retrieval (top-50)│ -> │ Rerank (top-5) │ -> │ LLM  │
└─────────────┘    │                          │    │                │    └──────┘
                   │  Dense  (bge-m3, 1024d)  │    │ bge-reranker   │
                   │  Sparse (BM25, 35K vocab)│    │     -v2-m3     │
                   │  Fusion (Milvus + RRF)   │    │                │
                   └──────────────────────────┘    └────────────────┘
```

### 模块组成

```text
code/
├── crawler/       # 数据获取与预处理:CCAR 网页爬虫 + PDF 解析 + chunk 切分
├── retrieval/     # 召回层:BM25Encoder + bge-m3 + Milvus hybrid_search
└── rerank/        # 精排层:Cross-Encoder + 召回-精排管线
```

## 🚀 快速开始

### 1. 环境准备

```bash
# Clone + 进入项目
git clone https://github.com/3dog1nsane/aviation-hybrid-rag.git
cd aviation-hybrid-rag

# 创建虚拟环境(推荐 uv,也可以用 venv)
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
```

### 2. 启动 Milvus

```bash
# 复用 Milvus 官方 standalone docker-compose
# (本仓库不包含 docker-compose,请参考 https://milvus.io/docs/install_standalone-docker.md)
docker compose up -d
```

### 3. 下载模型

```bash
# 从 ModelScope 下载(国内速度稳定,~5 分钟)
python code/retrieval/download_bge_m3.py
python code/rerank/download_reranker.py
```

模型默认存到 `~/models/bge-m3/` 和 `~/models/bge-reranker-v2-m3/`。

### 4. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 填入你的 DeepSeek API key、模型路径等
```

### 5. 跑通 demo

```bash
# 用 20 条样本数据建库
python code/retrieval/milvus_setup.py
python code/retrieval/milvus_insert.py

# 三路对比 demo(dense vs sparse vs hybrid)
python code/retrieval/hybrid_search_demo.py

# 召回 + 重排管线
python code/rerank/pipeline_with_rerank.py
```

---

## 🔧 关键设计决策

### 1. 为什么手写 BM25 而不用 `rank_bm25` 库?

`rank_bm25` 是 in-memory 打分库,输出"分数列表"而非"稀疏向量",**无法灌入 Milvus 的 SPARSE_FLOAT_VECTOR 字段**。本项目实现的 `BM25Encoder` 输出 `{token_id: weight}` 字典,直接对接 Milvus 稀疏索引,实现"BM25 → 稀疏向量 → ANN 检索"的统一管线。

### 2. 为什么 BM25 拆 query 侧 + doc 侧?

完整 BM25 公式可分解为 `query · doc` 内积:

- doc 侧:`(k1+1)·tf / (tf + k1·(1-b+b·|D|/avgdl))`
- query 侧:`IDF(t)`

这样所有向量库的内积运算就能复用,不用为 BM25 单独实现打分逻辑。Milvus / Qdrant 的稀疏检索本质就是这么干的。

### 3. 为什么 RRF 不需要归一化分数?

RRF 只看 rank,不看 score:`RRF(d) = Σ 1/(rank_i(d) + k)`。BM25 输出几十量级、dense 输出 0-1 量级,直接相加完全没意义。RRF 通过排名做到 zero-shot 融合,**不需要训练、不需要标注、不需要调权重**。k=60 是 Cormack 2009 原论文的经验值。

### 4. Cross-Encoder vs Bi-Encoder

|      | Bi-Encoder (embedding)  | Cross-Encoder (rerank) |
| ---- | ----------------------- | ---------------------- |
| 输入 | query 和 doc 分别编码   | query 和 doc 拼接      |
| 输出 | 向量                    | 单个分数               |
| 优点 | 向量可缓存,检索快 (ANN) | 全文 attention,精度高  |
| 缺点 | 信息有损                | 每次现算,慢            |
| 用途 | 召回 (recall)           | 精排 (precision)       |

**工业标准 pattern**:bi-encoder 召回 50~100 条 → cross-encoder 精排 5~10 条 → 喂 LLM。

---

## 🔬 实验发现

### 发现 1:Dense / Sparse 几乎 0% 重叠

7 个 query 中 6 个 dense top-5 与 sparse top-5 **完全不相交**。两路从根本上用不同方式看待文本:

- Dense:语义相似度("运输类飞机" ≈ "民航客机")
- Sparse:词项匹配度(必须有"运输类"这个 token)

**这就是混合检索的价值**:互补性极强。

### 发现 2:Rerank 的"no-answer detection"能力

故意设计一个数据库里没答案的 query("航油的危险品分类"),所有 7 个 query 的 Rerank Top-1 分数对比:
正常 query Top-1: 0.86 ~ 0.99
异常 query Top-1: 0.0503
**差一个数量级**。可以用 `rerank_score < 0.3` 作为阈值实现 no-answer detection,直接拒答而不返回错误内容。**这是 bi-encoder 做不到的**——dense 在不知道时也会给 0.5~0.7 的分数。

### 发现 3:BM25 + jieba 对强标识符的盲区

Query "CCAR-25 关于飞机结构强度的要求" 中,jieba 把 "CCAR-25" 切成 `["CCAR", "-", "25"]`,无法作为整体 token 匹配。Dense 也未找到 CCAR-25 主条款。**最后 Rerank 把真正的 CCAR-25-R4 条款从召回深处挖出来**——再次印证 Cross-Encoder 不依赖分词、做全文 attention 的价值。

工业级解决方案(本项目未实现,识别到问题):

- 加 ngram 索引(catch 强标识符)
- 命名实体识别预处理
- LLM 重写 query(展开"CCAR-25" → "CCAR-25 运输类飞机适航标准")

---

## ⚠️ 已知限制 / 未来工作

1. **自动评测缺失**:目前仅 7 个 query 人工对比,生产环境需要 200+ 标注 query 计算 Recall@5、MRR、NDCG
2. **强标识符问题**:见"发现 3",需要 ngram 索引或 query rewriting
3. **No-answer 阈值未严格调优**:仅观察到 0.05 vs 0.86 的差距,实际阈值需要更多负样本验证
4. **Chunk 策略粗糙**:目前按固定字符切分,更精细应该按法规条款语义切分(如按"第 X 条"切)
5. **缓存层缺失**:dense query 编码、热点 query 结果可以缓存到 Redis

### 计划中的下一步:Graph RAG

法规之间存在大量引用关系(CCAR-25 → CCAR-21 等),计划:

1. 用 LLM 抽取法规条款间的引用关系
2. 用 Neo4j 构建知识图谱
3. 实现"向量检索 + 图查询"的双路推理

---

## 📁 项目结构

```text
aviation-hybrid-rag/
├── code/
│   ├── crawler/                      # 数据获取
│   │   ├── crawl_ccar.py             # CCAR 网页爬虫
│   │   ├── parse_ccar.py             # HTML 解析
│   │   ├── parse_pdf.py              # PDF 解析
│   │   └── chunk_regulations.py      # 文本切分
│   ├── retrieval/                    # 召回层
│   │   ├── bm25_encoder.py           # 自实现 BM25 (核心模块)
│   │   ├── hybrid_retriever.py       # 混合检索器 (核心模块)
│   │   ├── milvus_setup.py           # Milvus collection 建表
│   │   ├── milvus_insert.py          # 灌库脚本
│   │   ├── test_bm25_locally.py     # BM25 本地验证
│   │   └── hybrid_search_demo.py     # 三路对比 demo
│   └── rerank/                       # 精排层
│       ├── reranker.py               # CrossEncoder 封装 (核心模块)
│       └── pipeline_with_rerank.py   # 召回+精排完整管线
├── data/
│   ├── aviation_dict.txt             # 3296 词航空术语 jieba 词典
│   ├── bm25_encoder.pkl              # 训练好的 BM25 状态 (词表+IDF)
│   └── samples/
│       └── ccar_chunks_sample.jsonl  # 20 条样本数据
├── docs/
│   └── C4_summary.md                 # 项目学习总结(中文)
└── requirements.txt
```

## 🙏 致谢与说明

- 跟随 [Datawhale all-in-rag](https://github.com/datawhalechina/all-in-rag) 教程的学习路径,但实现采用工业化方案
- CCAR 法规数据来源于民航局官网公开页面,**仅用于个人学习**
- 模型来自 BAAI 的 [bge-m3](https://huggingface.co/BAAI/bge-m3) 和 [bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- 详细学习总结见 [`docs/C4_summary.md`](docs/C4_summary.md)
