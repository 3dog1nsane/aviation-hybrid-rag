"""
端到端 RAG 问答：召回 + 重排 + LLM 生成（带 Citation）。

用法:
    from code.llm.rag_qa import RAGQA
    qa = RAGQA()
    qa.load()
    answer, sources = qa.ask("运输类飞机的适航标准是什么？")
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional
from dotenv import load_dotenv
from openai import OpenAI
# 把 code/retrieval 和 code/rerank 加入 sys.path，沿用现有模块的裸导入风格
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "code" / "retrieval"))
sys.path.insert(0, str(PROJECT_ROOT / "code" / "rerank"))
from hybrid_retriever import HybridRetriever  # noqa: E402
from reranker import Reranker  # noqa: E402
# 加载 .env（项目根目录下）
load_dotenv(PROJECT_ROOT / ".env")
DEFAULT_MODEL = "deepseek-chat"  # DeepSeek 模型名称
DEFAULT_TEMPERATURE = 0.1        # 生成温度，低一点保证引用准确
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TOP_K_RETRIEVAL = 50     # 混合检索召回数
DEFAULT_RERANK_TOP_K = 5         # 重排后保留 top-5


class RAGQA:
    """完整 RAG 问答 pipeline"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        top_k_retrieval: int = DEFAULT_TOP_K_RETRIEVAL,
        rerank_top_k: int = DEFAULT_RERANK_TOP_K,
    ):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("请设置 DEEPSEEK_API_KEY 环境变量或在初始化时传入 api_key")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_k_retrieval = top_k_retrieval
        self.rerank_top_k = rerank_top_k

        # 延迟加载
        self.client: Optional[OpenAI] = None
        self.retriever: Optional[HybridRetriever] = None
        self.reranker: Optional[Reranker] = None

    def load(self) -> "RAGQA":
        """初始化各个组件（连接 Milvus，加载模型等）"""
        # 初始化 LLM 客户端
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com/v1",
        )
        # 检索器
        self.retriever = HybridRetriever().load()
        # 重排器
        self.reranker = Reranker().load()
        return self

    # ---------- 核心接口 ----------
    def ask(self, query: str, verbose: bool = True) -> Tuple[str, List[dict]]:
        """
        输入 query，返回 (答案文本, 引用来源列表)
        来源列表即 reranker 输出的前 top_k 条，每条含 rerank_score 和元数据
        """
        # 1. 召回
        candidates = self.retriever.search_hybrid(
            query,
            top_k=self.rerank_top_k,
            recall_k=self.top_k_retrieval,
            rrf_k=60,
        )
        if not candidates:
            return ("未找到相关文档。", [])

        # 2. 重排
        top_docs = self.reranker.rerank(query, candidates, top_k=self.rerank_top_k)

        # 3. 构建 Prompt（包含上下文和引用格式要求）
        prompt = self._build_prompt(query, top_docs)

        # 4. 调用 LLM
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        answer = completion.choices[0].message.content.strip()

        if verbose:
            self._print_info(query, top_docs, answer)

        return answer, top_docs

    # ---------- Prompt 构建 ----------
    def _build_prompt(self, query: str, docs: list[dict]) -> str:
        """把 top-k chunk 组装成带编号的上下文，并在末尾要求 Citation"""
        context_parts = []
        for i, doc in enumerate(docs, 1):
            # 可读的引用标签
            source_label = self._source_label(doc)
            context_parts.append(f"[{i}] ({source_label})\n{doc['chunk_text']}")

        context = "\n\n".join(context_parts)

        prompt = f"""你是一个航空规章问答助手。请根据以下提供的参考文档片段回答用户的问题。

参考文档：
{context}

用户问题：{query}

要求：
1. 答案必须基于参考文档，如果文档中没有足够信息，请明确回答“文档中未提及”。
2. 2. 在答案的末尾，附上"引用来源"列表，格式为：[来源X] 《文档名称》（编号） 第X条，例如 [来源1] 《运输类飞机适航标准》（CCAR-25-R4） 第25.301条。若文档没有编号或条目号，相应部分省略。
3. 保持答案简洁专业。"""
        return prompt
    
    @staticmethod
    def _source_label(doc: dict) -> str:
        """格式化来源标签，用于 Prompt 和引用"""
        title = doc.get("doc_title", "未知文档")
        ccar = doc.get("ccar_no", "")
        if ccar:
            return f"《{title}》（{ccar}）"
        return f"《{title}》"

    # ---------- 可选的辅助方法 ----------
    def _print_info(self, query: str, top_docs: list[dict], answer: str) -> None:
        """打印调试信息"""
        print("\n" + "="*60)
        print(f"Query: {query}")
        print(f"\nTop-{len(top_docs)} 重排结果:")
        for i, doc in enumerate(top_docs, 1):
            print(f"  [{i}] score={doc['rerank_score']:.4f} | "
                  f"{doc.get('doc_title','')} | "
                  f"{doc.get('ccar_no','')}")
        print("\n--- 答案 ---")
        print(answer)
        print("="*60 + "\n")


# ---------- 自检（需要先设置好 .env 并确保 Milvus 运行）----------
if __name__ == "__main__":
    qa = RAGQA()
    qa.load()

    test_queries = [
        "运输类飞机的适航标准是什么？",
        "飞行学员体检合格证有效期多久？",
    ]
    for q in test_queries:
        ans, sources = qa.ask(q)
        # 检查引用格式
        if "[来源" in ans:
            print("✓ 检测到引用来源")
        else:
            print("✗ 未检测到引用来源，请检查 prompt")