# Aviation Hybrid RAG

[![Python](https://img.shields.io/badge/Python-3.12-blue)]()
[![Milvus](https://img.shields.io/badge/Milvus-2.6-orange)]()
[![Models](https://img.shields.io/badge/Models-bge--m3%20%7C%20bge--reranker--v2--m3-green)]()
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

> 面向中国民航 CCAR 法规的高精度检索系统。
> **Hybrid search (dense + sparse BM25) + Cross-Encoder rerank**,
> 工业级 RAG 检索管线的完整实现。

---

## 🎬 Demo

[![asciicast](https://asciinema.org/a/d40tPJGNwsFCnUwT.svg)](https://asciinema.org/a/d40tPJGNwsFCnUwT)

> 端到端问答:用户提问 → 混合检索 + Rerank → DeepSeek 生成回答 + 引用追溯 (citation)

---

## 🎯 项目特点

- **真实领域数据**:114 份民航法规/规范性文件、5372 个语义切片(repo 仅含 20 条样本)
- **手写 BM25 编码器**:jieba 中文分词 + 3296 词航空术语词典 + 自实现 IDF / 稀疏向量输出
- **混合检索**:Milvus 原生 `hybrid_search` + RRF 融合 (k=60),dense (bge-m3, 1024d) + sparse 双路
- **Cross-Encoder 精排**:bge-reranker-v2-m3,100 条评测集上将 Recall@1 从 0.58 提升至 0.76
- **No-answer detection**:利用 reranker 分数分布识别库中无答案场景(异常 query 0.05 vs 正常 0.86~0.99,差一个数量级)
- **时效性过滤**:每个 chunk 保留 validity 状态,默认只检索现行有效法规,支持历史版本按需查询
- **端到端 RAG 问答**:接入 DeepSeek API,实现 query → 答案 + **citation 引用追溯**(降低幻觉、可审计)

---

## 📊 关键指标

| 指标             | 数值                                |
| ---------------- | ----------------------------------- |
| 入库文档规模     | 114 份文件 × 5372 chunks            |
| 现行有效法规库   | 28 份 × 1992 chunks(validity 过滤) |
| BM25 词表大小    | 35,117                              |
| Dense 向量维度   | 1024 (bge-m3)                       |
| 评测标注集       | 100 条 LLM 自动构造 query           |
| Rerank 50 条耗时 | 1.3s (RTX 4080)                     |

### 检索质量评测(100 条标注集,只检索现行有效库)

| 配置              | Recall@1  | Recall@5  | Recall@10 | MRR@10    | NDCG@10   |
| ----------------- | --------- | --------- | --------- | --------- | --------- |
| dense             | 0.560     | 0.860     | 0.910     | 0.694     | 0.747     |
| sparse (BM25)     | 0.540     | 0.830     | 0.880     | 0.665     | 0.718     |
| hybrid (RRF)      | 0.580     | 0.860     | 0.910     | 0.708     | 0.758     |
| **hybrid+rerank** | **0.760** | **0.970** | **0.970** | **0.856** | **0.885** |

> **关键结论**:三种召回方式 Recall@10 均达 0.88~0.91,说明召回阶段
> "把正确答案捞进候选池"基本不是瓶颈;真正的增益来自 **Cross-Encoder
> 精排**——它把 Recall@1 从 0.58 拉到 0.76、MRR@10 从 0.71 拉到 0.86。
> 这印证了"召回宽、精排严"的两阶段设计。

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
├── rerank/        # 精排层:Cross-Encoder + 召回-精排管线
└── llm/           # 生成层:DeepSeek 端到端问答 + citation
```

## 🚀 快速开始

### 1. 环境准备

```bash
git clone https://github.com/3dog1nsane/aviation-hybrid-rag.git
cd aviation-hybrid-rag

# 创建虚拟环境(推荐 uv,也可用标准 venv)
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
```

### 2. 启动 Milvus (Standalone)

本项目使用 Milvus 2.6 standalone 部署,仓库未包含 docker-compose.yml,请按官方文档下载:

```bash
wget https://github.com/milvus-io/milvus/releases/download/v2.6.0/milvus-standalone-docker-compose.yml -O docker-compose.yml
docker compose up -d
docker compose ps   # 验证容器状态
```

> 详见 [Milvus 官方文档](https://milvus.io/docs/install_standalone-docker.md)。

### 3. 下载模型

```bash
python code/retrieval/download_bge_m3.py
python code/rerank/download_reranker.py
```

模型默认存到 `~/models/bge-m3/` 和 `~/models/bge-reranker-v2-m3/`。

### 4. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 填入 DeepSeek API key、模型路径等
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

# 端到端 RAG 问答(混合检索 + Rerank + DeepSeek + citation)
python code/llm/demo.py
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

### 5. 为什么全量入库 + validity 过滤,而非只存有效法规?

法规系统中,失效/历史版本并非垃圾数据——历史版本查询、版本对比、合规追溯都需要它们。因此采用"软删除"思路:全量入库,用 `validity` 标量字段控制展示,默认只返回现行有效条款,需要时一行 filter 即可放开历史查询。

---

## 🔬 实验发现

### 发现 1:瓶颈在排序,不在召回 —— Rerank 是最大增益来源

对比表里最显著的一跳是 hybrid → hybrid+rerank:Recall@1 +18 个点、MRR@10 +0.15。而三种召回方式之间差异很小(Recall@1 都在 0.54~0.58)。

解读:召回阶段已经能把正确 chunk 放进 top-50(Recall@10≈0.9),瓶颈是"top-50 里怎么把对的排到最前",这正是 cross-encoder 做全文 token-level attention 的强项。**这是"召回宽、精排严"管线有效性的直接证据。**

### 发现 2:混合检索增益温和,但 dense / sparse 机理互补

hybrid 相比纯 dense 只高约 2 个点。原因:评测 query 语义较明确,bge-m3 dense 本身召回已强,sparse 的补充空间有限。但两路的**召回机理**仍是互补的:

- Dense:语义相似度("运输类飞机" ≈ "民航客机")
- Sparse:词项精确匹配(必须命中"运输类"这个 token)

在词项精确匹配更关键的场景(如强标识符、专有术语),sparse 的价值会更突出(见发现 4)。

### 发现 3:Rerank 分数可做 no-answer 拒答信号

对一个库中无答案的 query("航油的危险品分类"),其 Rerank Top-1 得分仅 0.05,而正常 query 普遍在 0.86~0.99,**差一个数量级**。可用 `rerank_score < 阈值` 作为低成本拒答信号,无相关上下文时主动拒答而非强行生成,降低幻觉。这是 bi-encoder 难以做到的(dense 在不确定时仍会给中等分数)。

> 注:此为定性观察,拒答阈值尚未在大规模负样本上严格标定,见"未来工作"。

### 发现 4:BM25 + jieba 对强标识符的盲区

Query "CCAR-25 关于飞机结构强度的要求" 中,jieba 把 "CCAR-25" 切成 `["CCAR", "-", "25"]`,无法整体匹配;dense 也未直接命中主条款。最终是 Rerank 把正确条款从召回深处提了上来——再次印证 cross-encoder 不依赖分词、做全文 attention 的价值。工业级补救方案(已识别,留待后续):ngram 索引、命名实体识别预处理、LLM query 改写。

---

## 🧪 评测方法

为避免凭少量人工 query 下结论,构建了一套可复现的检索评测流程:

1. **标注集构造**:从 `validity == "有效"` 的法规 chunk 中按文档分层抽样,用 LLM(DeepSeek)为每个 chunk 生成一个用户视角的问题,自动建立 `query → golden chunk` 配对,得到 100 条标注集。
2. **评测口径**:单正例标注,统计 Recall@K(单正例下等价 Hit@K)、MRR@10、NDCG@10;所有检索均加 `validity == "有效"` 过滤,只评现行有效库。
3. **对比四种配置**:dense / sparse / hybrid / hybrid+rerank,召回池 50 → rerank 取 top-10。

> 评测脚本与样本见 [`code/eval/`](code/eval/)。

---

## 🛠️ 未来工作

1. **扩展评测集**:从 100 条扩到数百条,引入多正例标注,提升统计可靠性;评估集覆盖在域、超域、强标识符三类场景。
2. **强标识符检索**:针对 "CCAR-25" 等标识符,加 ngram 索引或 LLM query 改写,解决发现 4 的分词盲区。
3. **No-answer 阈值标定**:在更多负样本上严格标定拒答阈值,而非经验值。
4. **更精细的 chunk 策略**:从固定字符切分改为按法规条款语义切分(如按"第 X 条")。
5. **缓存层**:dense query 编码、热点 query 结果缓存到 Redis,降低线上延迟。

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
│   │   ├── download_bge_m3.py        # bge-m3 模型下载
│   │   ├── test_bm25_locally.py      # BM25 本地验证
│   │   └── hybrid_search_demo.py     # 三路对比 demo
│   ├── rerank/                       # 精排层
│   │   ├── reranker.py               # CrossEncoder 封装 (核心模块)
│   │   ├── download_reranker.py      # reranker 模型下载
│   │   └── pipeline_with_rerank.py   # 召回+精排完整管线
│   └── llm/                          # 生成层
│       ├── rag_qa.py                 # 端到端 RAG 问答 (核心模块)
│       └── demo.py                   # 交互式 demo (asciinema 录制入口)
├── data/
│   ├── aviation_dict.txt             # 3296 词航空术语 jieba 词典
│   ├── bm25_encoder.pkl              # 训练好的 BM25 状态 (词表+IDF)
│   └── samples/
│       └── ccar_chunks_sample.jsonl  # 20 条样本数据
├── LICENSE
└── requirements.txt
```

## 🙏 致谢与说明

- 参考 [Datawhale all-in-rag](https://github.com/datawhalechina/all-in-rag) 教程的学习路径,实现采用工业化方案
- 法规数据来源于民航局官网公开页面,**仅用于个人学习**;法规有效性以 2026 年初爬取时为准
- 模型来自 BAAI 的 [bge-m3](https://huggingface.co/BAAI/bge-m3) 和 [bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)

## 📄 License

[MIT](LICENSE)