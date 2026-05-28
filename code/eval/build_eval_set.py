"""
造检索评测标注集。

流程:
  1. 读 fixed jsonl,只保留 validity=="有效" 的 chunk
  2. 按 doc_id 均衡分层抽样,选 N 个 chunk
  3. 对每个 chunk,用 DeepSeek 生成一个「用户可能会问的问题」
  4. 输出标注集:每行 {query, golden_chunk_id, doc_id, doc_title}

注意:
  - LLM 生成的 query 可能词汇贴近原文(词汇泄露),prompt 已要求改写,但无法根除。
    生成后建议人工扫一遍,删掉明显不合理的。
  - 单正例标注:每个 query 只认 1 个 golden chunk。法规里可能有多个相关 chunk,
    这会让 Recall 偏保守(低估),属已知局限。

依赖:openai (pip install openai),DeepSeek 用 OpenAI 兼容接口
环境变量:DEEPSEEK_API_KEY(或在 .env 里,自己加载)

用法:
  python build_eval_set.py <fixed_jsonl> <输出_eval_set.jsonl> [样本数,默认100]
"""
import json
import os
import random
import sys
import time
from collections import defaultdict

from openai import OpenAI

# ---------- 配置 ----------
random.seed(42)  # 可复现

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-v4-pro"

# 抽样时跳过太短的 chunk(目录/碎片不适合生成 query)
MIN_CHUNK_LEN = 200

PROMPT_TEMPLATE = """你是民航法规问答系统的测试集构造助手。

下面是一段民航法规/规章的正文片段。请你站在「用户提问」的角度,
生成一个该片段能够回答的、自然的中文问题。

要求:
1. 问题必须能被这段正文直接回答
2. 用你自己的话提问,尽量不要直接照抄原文里的词句(用同义表达)
3. 问题要具体、口语化,像真实用户会问的,不要太书面
4. 只输出问题本身,不要任何解释、前缀、引号

正文片段:
{chunk_text}

问题:"""


def load_valid_chunks(path: str) -> list[dict]:
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            if o.get("validity") == "有效" and o.get("chunk_len", 0) >= MIN_CHUNK_LEN:
                chunks.append(o)
    return chunks


def stratified_sample(chunks: list[dict], n: int) -> list[dict]:
    """按 doc_id 均衡分层抽样,尽量让每个文档都有代表。"""
    by_doc = defaultdict(list)
    for c in chunks:
        by_doc[c["doc_id"]].append(c)

    docs = list(by_doc.keys())
    random.shuffle(docs)

    selected = []
    # 轮询:每轮从每个文档取 1 个,直到凑够 n
    round_idx = 0
    while len(selected) < n:
        added_this_round = 0
        for doc in docs:
            pool = by_doc[doc]
            if round_idx < len(pool):
                selected.append(pool[round_idx])
                added_this_round += 1
                if len(selected) >= n:
                    break
        if added_this_round == 0:
            break  # 所有文档都取完了
        round_idx += 1

    return selected[:n]


def gen_query(client: OpenAI, chunk_text: str) -> str:
    resp = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[{"role": "user",
                   "content": PROMPT_TEMPLATE.format(chunk_text=chunk_text[:1500])}],
        temperature=0.7,
        max_tokens=100,
    )
    return resp.choices[0].message.content.strip()


def main(in_path: str, out_path: str, n: int):
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("错误:未设置环境变量 DEEPSEEK_API_KEY")
        print("可以 export DEEPSEEK_API_KEY=xxx 或在脚本里加载 .env")
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)

    chunks = load_valid_chunks(in_path)
    print(f"[load] 有效且够长的 chunk: {len(chunks)}")

    sampled = stratified_sample(chunks, n)
    print(f"[sample] 分层抽样选出: {len(sampled)} 个 chunk")
    doc_dist = defaultdict(int)
    for c in sampled:
        doc_dist[c["doc_title"]] += 1
    print(f"[sample] 覆盖 {len(doc_dist)} 个文档")

    results = []
    for i, c in enumerate(sampled, 1):
        try:
            q = gen_query(client, c["chunk_text"])
        except Exception as e:
            print(f"  [{i}] 生成失败,跳过: {e}")
            time.sleep(2)
            continue
        results.append({
            "query": q,
            "golden_chunk_id": c["chunk_id"],
            "doc_id": c["doc_id"],
            "doc_title": c["doc_title"],
        })
        if i % 10 == 0:
            print(f"  进度 {i}/{len(sampled)}  最新 query: {q[:40]}")
        time.sleep(0.3)  # 轻微限速

    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n[done] 标注集已写出: {out_path} (共 {len(results)} 条)")
    print(f"[提示] 建议人工扫一遍,删掉明显不合理的 query 再用于评测。")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python build_eval_set.py <fixed_jsonl> <out_eval_set.jsonl> [样本数=100]")
        sys.exit(1)
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    main(sys.argv[1], sys.argv[2], n)