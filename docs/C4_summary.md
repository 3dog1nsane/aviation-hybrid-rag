# C4 检索优化 学习总结

> 项目:CCAR 民航法规智能检索系统  
> 学习周期:2026.05.21-2026.05.27  
> 跟随教程:Datawhale all-in-rag,但工业化方案优先  
> 用途:个人复盘 + 面试讲稿底稿

---

## 1. 项目一句话定位

针对中国民航 CCAR 法规(114 部规章、5372 chunks)构建的高精度检索系统,
采用 dense + sparse 混合召回 + Cross-Encoder 精排的工业级 RAG 检索管线。

---

## 2. 我做了什么(按时间顺序)

### Phase 1:数据准备(此前已完成)
- 爬取 114 部 CCAR 法规(HTML + PDF 共 ~200 文件)
- 用 LangChain RecursiveCharacterTextSplitter 切分:**800 字 / overlap 100 字**
- 产出:`ccar_chunks.jsonl`(5372 条,中位 658 字)
- **关键字段**:`validity`(有效/失效/废止/历史版本)、`ccar_no`、`doc_title` 等

### Phase 2:Sparse 检索(BM25 手写)
- **不依赖 rank_bm25 库,自己实现 BM25Encoder**
  - jieba 分词 + 加载 3296 词航空术语自定义词典
  - 训练时算 IDF(Robertson 平滑版本) + 平均文档长度
  - 文档侧编码:tf 饱和 + 长度归一化(k1=1.5, b=0.75)
  - 查询侧编码:IDF 权重
  - 输出格式:`{token_id: weight}` 字典,可直接灌 Milvus sparse 字段

### Phase 3:Dense 检索
- 用 BAAI/bge-m3(1024 维)做 dense 编码
- 从 ModelScope 下载到本地(避开 HF 镜像不稳)
- GPU 推理:batch_size=32,5372 条耗时 2 分 23 秒(RTX 4080)

### Phase 4:Milvus 混合检索
- Schema 设计:9 字段(7 标量 + 2 向量)
- 索引:dense 用 HNSW(M=16, efConstruction=64),sparse 用 SPARSE_INVERTED_INDEX
- 度量统一用 IP(配 bge-m3 默认 L2 归一化输出)
- 用 Milvus 原生 `hybrid_search` + `RRFRanker(k=60)` 做融合
- 灌库 5372 条,dense + sparse 双向量

### Phase 5:Cross-Encoder 精排
- 用 BAAI/bge-reranker-v2-m3 做 Rerank
- 模式:召回 top-50 → Rerank → top-5
- sigmoid 激活输出 0-1 分数,便于阈值判断

---

## 3. 关键数据(面试可直接讲)

| 指标 | 数字 |
|---|---|
| 数据规模 | 114 法规 × 5372 chunks |
| BM25 词表大小 | 35117 |
| Dense 向量维度 | 1024(bge-m3) |
| Sparse 向量平均非零数 | 96 |
| Dense ∩ Sparse top-5 重叠率 | **0.4 / 5**(7 query 中 6 个完全 0% 重叠) |
| Rerank Top-1 改判率 | **5/7 = 71%** |
| 80% 真答案在召回 top-6~50 区间 | 验证"召回宽,精排严"必要性 |
| Rerank no-answer 信号 | **数据缺失 query 得分 0.05 vs 正常 0.86~0.99**,差一个数量级 |
| Rerank 50 条耗时 | 1.3 秒(RTX 4080) |

---

## 4. 关键设计决策(面试金句库)

### 决策 1:为什么手写 BM25 不用 rank_bm25 库?

> rank_bm25 是 in-memory 打分库,输出"分数列表"而非"稀疏向量",
> 不能直接灌 Milvus 的 SPARSE_FLOAT_VECTOR 字段。  
> 我手写 BM25Encoder 输出 `{token_id: weight}` 字典,
> 实现了"BM25 → 稀疏向量 → ANN 检索"的统一管线。

### 决策 2:为什么 BM25 拆 query 侧 + doc 侧?

> 完整 BM25 公式可分解为 query · doc 内积:  
> - doc 侧:`(k1+1)*tf / (tf + k1*(1-b+b*|D|/avgdl))`  
> - query 侧:`IDF(t)`  
> 这样所有向量库的内积运算就能复用,不用为 BM25 单独实现打分逻辑。
> Milvus / Qdrant 的 sparse 检索本质就是这么干的。

### 决策 3:为什么 RRF 不需要归一化分数?

> RRF 只看 rank,不看 score:`RRF(d) = Σ 1/(rank_i(d) + k)`。
> BM25 输出几十量级、dense 输出 0-1 量级,直接相加完全没意义。
> RRF 通过"排名"这个统一标尺,做到 zero-shot 融合,
> 不需要训练、不需要标注、不需要调权重。k=60 是 Cormack 2009 原论文经验值。

### 决策 4:为什么用 bge-m3 不用 bge-small-zh-v1.5?

> bge-m3 是 2024 后中文 RAG 的主流选择:  
> 1) 维度 1024 vs 512,表达能力更强  
> 2) 长文本支持 8192 vs 512,我的 chunk 800 字完全 cover  
> 3) 同时输出 dense/sparse/colbert,虽然我只用 dense,但保留了扩展性  
> 显存约 4GB,4080 12G 充足。

### 决策 5:Cross-Encoder vs 双编码器有什么本质区别?

> - 双编码器(bi-encoder):query 和 doc 分别过模型 → 向量 → ANN  
>   优点:向量可缓存、检索快。缺点:信息压缩成单向量,有损。  
> - 交叉编码器(cross-encoder):query 和 doc 拼接过模型 → 分数  
>   优点:query-doc 间做 token-level attention,精度高。缺点:每次现算,慢。  
> 工业界标准:bi-encoder 召回(看 recall),cross-encoder 精排(看 precision)。

### 决策 6:Milvus VARCHAR max_length 单位是字节,不是字符

> 这是个工程坑。Milvus VARCHAR `max_length` 配置的是 UTF-8 字节数,
> 不是 Python 字符数。中文 1 字符 = 3 字节,所以 max_length 配置时
> 要按 "估计最大字符数 × 4 + 安全余量"。我吃过这个坑,
> 配 max_length=2048,实际 chunk 2058 字节就炸了,后来改成 4096。

---

## 5. 实验观察与发现

### 观察 1:dense 和 sparse 在我的数据上 0% 重叠

7 个 query 中 6 个的 dense top-5 与 sparse top-5 **完全不相交**。
说明两路从根本上是用**不同方式**看待文本:

- dense:语义相似度("运输类飞机" ≈ "民航客机")
- sparse:词项匹配度(必须有"运输类"这个 token)

**这就是混合检索的价值**:两路互补,任何一路单独都不够。

### 观察 2:Rerank 的 "no-answer detection" 能力

我设计了一个故意找不到答案的 query("航油的危险品分类"——CCAR 法规库里没这内容),
所有 7 个 query 的 Rerank Top-1 分数:
正常 query Top-1: 0.86 ~ 0.99
数据缺失 query Top-1: 0.0503

**差了一个数量级**。这意味着可以用 `rerank_score < 0.3` 做阈值,
直接拒答"我不知道",而不是返回 top-K 错误答案。

这是双编码器做不到的——dense 在不知道时也会给 0.5~0.7 的分数。

### 观察 3:80% 真答案在召回 top-6~50

7 个 query 中有 5 个,**Rerank top-5 里 4/5 都从召回 top-5 之外提上来**。
说明如果跳过 Rerank 直接用召回 top-5,大量真答案会被错过。

**结论**:Recall 50 → Rerank 5 是工业 RAG 的标准管线,不是过度设计。

### 观察 4:BM25 在强标识符上的盲区

我设计的 query 2 "CCAR-25 关于飞机结构强度的要求",
预期 BM25 应该精确匹配"CCAR-25",**结果失败**——
因为 jieba 把 "CCAR-25" 切成 `["CCAR", "-", "25"]`,
没法作为整体 token 匹配。Dense 也没找到真正的 CCAR-25 条款。

**最后是 Rerank 把真正的 CCAR-25-R4 条款从召回深处挖出来**,
反过来证明了 Cross-Encoder 不依赖分词、做全文 attention 的价值。

**工业界补救方案**(我没做,但识别到了问题):
- 加 ngram 索引(catch 强标识符)
- 命名实体识别预处理
- prompt LLM 重写 query(展开"CCAR-25"为"CCAR-25 运输类飞机适航标准")

---

## 6. 踩过的坑(实战经验)

### 坑 1:Milvus VARCHAR max_length 单位是字节
- **现象**:配 2048,中文 chunk 2058 字节炸
- **修复**:按 UTF-8 字节数截断 + max_length 配 4096
- **教训**:配置容量时考虑编码

### 坑 2:hf-mirror 下载 bge-m3 时 cas-bridge CDN 不稳
- **现象**:5% 卡住,Resume 后到 24% 又卡
- **修复**:改用 ModelScope(阿里魔搭),国内稳定 14 MB/s
- **教训**:大模型下载首选国内官方镜像,不要依赖社区镜像

### 坑 3:ModelScope `ignore_file_pattern` 乱写导致缺文件
- **现象**:为了省空间过度过滤,把 `pytorch_model.bin` 和 `1_Pooling/` 都跳过
- **修复**:除明显大文件(onnx),其他全下,出错代价更高
- **教训**:不熟悉的目录结构不要瞎过滤

### 坑 4:Python 文件名以数字开头不能 import
- **现象**:`02_bm25_encoder.py` 无法 `import`
- **修复**:模块文件用普通命名(`bm25_encoder.py`),
  数字编号只用于实验/演示脚本
- **教训**:"可被 import 的"和"一次性脚本"用不同命名规则

---

## 7. 没做但应该知道的事(诚实清单)

> 面试时如果被问"还有什么可以做",可以主动提这些:

1. **自动评测**:目前只用 7 个 query 人工对比,
   规范做法是构建 200+ 标注 query,算 Recall@5、MRR、NDCG
2. **查询重写**:用 LLM 把 "CCAR-25" 这种标识符做语义展开
3. **chunk 策略**:目前用固定字符切分,
   更精细应该按法规条款语义切分(如按 "第 X 条" 切)
4. **缓存层**:dense query 编码、热点 query 结果可以缓存到 Redis
5. **Graph RAG**:法规之间有引用关系(CCAR-25 → CCAR-21),
   建图后能做关联推理,这是我下一步计划

---

## 8. 代码组织
code/C4_retrieval/
├── 01_hybrid_search/
│   ├── bm25_encoder.py              # BM25Encoder 类(可复用模块)
│   ├── hybrid_retriever.py          # HybridRetriever 类(可复用模块)
│   ├── 03_test_bm25_locally.py      # BM25 本地验证
│   ├── 04_milvus_setup.py           # 建表 + 索引
│   ├── 05_milvus_insert.py          # 灌库
│   ├── 06_hybrid_search_demo.py     # 三路对比演示
│   └── download_bge_m3.py           # 模型下载
└── 02_rerank/
├── reranker.py                  # Reranker 类(可复用模块)
├── 01_test_rerank_locally.py    # Rerank 自检(冒烟测试)
├── 02_pipeline_with_rerank.py   # 召回+精排管线
└── download_reranker.py

---

## 9. 面试可能问题预演

### Q: 为什么不直接用 Milvus 内置的 BM25?
A: Milvus 内置 BM25 是黑盒,我希望对中文分词层有完全控制
   (jieba 分词、自定义航空术语词典),手写实现更可控可调试。
   并且过程中我学到了 BM25 拆 query/doc 侧的核心思想,
   这是用黑盒学不到的。

### Q: RRF 的 k 值为什么是 60?
A: Cormack 2009 论文的经验值,Elastic / Milvus 默认也是 60。
   它控制排名靠前文档的相对权重——k 越小,top-rank 权重越突出;
   k 越大,融合越平滑。60 在大多数场景下表现稳健。
   我没调参,生产环境通过 A/B 测试调。

### Q: Rerank 这么慢(1.3 秒),线上怎么用?
A: 这是 50 条全过 cross-encoder 的最坏情况。优化手段:
   1) 降低召回数到 20-30,速度可降至 500ms
   2) batch 推理 + GPU 利用率优化
   3) 用更小模型(bge-reranker-base, 0.5GB,速度快 3 倍)
   4) 异步并行召回 + 流式 LLM 输出,用户感知延迟可降到 1 秒内

### Q: 你的项目和市面上其他 RAG 项目有什么不同?
A: 1) 数据领域:民航法规(专业、有 validity 字段、强标识符多)
   2) BM25 手写:可控、能讲清楚原理
   3) 发现并验证了 Rerank 的 no-answer 信号(0.05 vs 0.99)
   4) 量化了 dense/sparse 互补性(0% 重叠)

### Q: 项目最大的挑战是什么?
A: jieba 分词对强标识符(如 "CCAR-25")的盲区。
   这暴露了"BM25 + 中文分词"的固有局限,
   Rerank 部分救回但不彻底。
   工业级解法需要叠加 ngram 索引或 LLM 重写 query。

---

## 10. 下一步规划

- 把现有代码整理推 GitHub(本周内)
- 加 LLM 端到端(DeepSeek API + Citation)
- 启动 Graph RAG 主项目(2026.6 前完成):
  在现有混合检索基础上,引入 Neo4j 构建法规引用图谱,
  实现 "向量检索 + 图查询" 的双路推理
