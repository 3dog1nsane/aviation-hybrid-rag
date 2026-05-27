"""
01_chunk_regulations.py

输入: ../../../data/C4/processed/ccar_regulations.jsonl  (爬虫产出)
输出:
  - ../../../data/C4/processed/ccar_chunks.jsonl         (切好的 chunk)
  - ../../../data/C4/aviation_dict.txt                   (航空术语词典,供 BM25 jieba 分词用)

流程:
  1. 加载 raw jsonl
  2. 过滤垃圾数据 (扫描件/数据表怪兽)
  3. 抽取航空术语,生成词典文件
  4. 用 LangChain RecursiveCharacterTextSplitter 切 chunk (800字/overlap 100字)
  5. 每个 chunk 继承父文档 metadata + 自己的位置信息
  6. 写 chunks.jsonl

依赖:
  uv pip install langchain langchain-text-splitters
"""

import json
import re
import argparse
from pathlib import Path
from collections import Counter
from typing import List, Dict, Set

from langchain_text_splitters import RecursiveCharacterTextSplitter


# ---- 数据质量过滤阈值 ----
MIN_TEXT_LEN = 200            # < 200 字废数据(扫描件/空文档)
MAX_TEXT_LEN = 1_500_000      # 极端兜底:> 150 万字几乎必是数据表

# 标题关键词黑名单(数据表/索引/型号清单等)
# 这些 title 包含特征词的文档,即使有正文也是数据表性质,不适合做 RAG
TITLE_BLACKLIST_KEYWORDS = [
    '目录',      # "...产品和零部件目录"
    '清单',      # "...型号清单"
    '型号表',
    '产品列表',
    '名录',
]

# ---- chunk 参数 ----
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# ---- 航空术语抽取正则 ----
# 1) CCAR 部号: CCAR-91 / CCAR-91-R2 / CCAR-25-R4
PATTERN_CCAR = re.compile(r'CCAR-\d+(?:-R\d+)?', re.IGNORECASE)
# 2) AC 咨询通告编号: AC-91-FS-2009-12 / AC-21-AA-2012-10R11
PATTERN_AC = re.compile(r'AC-\d+(?:-[A-Z]+)?(?:-\d{4})?(?:-\d+(?:R\d+)?)?', re.IGNORECASE)
# 3) 常用英文航空缩写
#    收紧:至少 3 个字母,避免抓到 AA / TO 这种垃圾
PATTERN_ABBR = re.compile(r'\b[A-Z]{3,5}(?:-?[A-Z\d]{1,3})?\b')
# 4) 机型号: B737 / B737-800 / A320 / C919 / ARJ21
PATTERN_AIRCRAFT = re.compile(r'\b[A-Z]\d{3}(?:-\d{3})?\b')
# 5) 条款编号: 第91.305条 / 第25.1条
PATTERN_ARTICLE = re.compile(r'第\d+(?:\.\d+)?条')

# 英文缩写黑名单(常见英文单词被误识别)
ABBR_BLACKLIST = {
    'DOOR', 'OPEN', 'CLOSE', 'NEXT', 'PREV', 'NOTE', 'AND', 'THE',
    'FOR', 'NOT', 'ALL', 'ANY', 'NEW', 'OLD', 'TOP', 'BOX',
    'SEE', 'USE', 'GET', 'PUT',
}


def is_record_valid(record: Dict) -> tuple[bool, str]:
    """
    判断一条 record 是否符合质量标准。
    
    过滤策略(按性质,不只按字数):
    1. 过短 → 扫描件 / 空文档
    2. title 含数据表关键词 → "目录"、"清单"等
    3. 极端长度 → 兜底防御
    4. extraction_status 失败 → PDF 解析没拿到内容

    Returns:
        (is_valid, reason): reason 用于统计
    """
    text = record.get('text', '')
    title = record.get('title', '') or ''
    
    if not text:
        return False, "empty_text"
    if len(text) < MIN_TEXT_LEN:
        return False, "too_short"
    # title 含数据表关键词,无论字数多少都过滤
    for kw in TITLE_BLACKLIST_KEYWORDS:
        if kw in title:
            return False, f"title_blacklist_{kw}"
    if len(text) > MAX_TEXT_LEN:
        return False, "too_long_monster"
    status = record.get('extraction_status')
    if status in ('needs_ocr', 'failed', 'pdf_download_failed'):
        return False, f"status_{status}"
    return True, "ok"


def extract_aviation_terms(records: List[Dict]) -> List[str]:
    """
    从所有有效 record 的正文里抽取航空术语,作为 jieba 自定义词典。
    
    抽取策略:多个正则匹配 + 词频排序 + 阈值过滤。
    """
    term_counter = Counter()
    
    for r in records:
        text = r.get('text', '')
        if not text:
            continue
        
        # 1) CCAR 部号
        for m in PATTERN_CCAR.findall(text):
            term_counter[m.upper()] += 1
        
        # 2) AC 编号  
        for m in PATTERN_AC.findall(text):
            term_counter[m.upper()] += 1
        
        # 3) 英文缩写
        for m in PATTERN_ABBR.findall(text):
            if m in ABBR_BLACKLIST:
                continue
            if len(m) < 2:
                continue
            term_counter[m] += 1
        
        # 4) 机型号
        for m in PATTERN_AIRCRAFT.findall(text):
            term_counter[m.upper()] += 1
        
        # 5) 条款编号
        for m in PATTERN_ARTICLE.findall(text):
            term_counter[m] += 1
    
    # 阈值过滤:至少出现 2 次才算有用术语
    terms = [t for t, c in term_counter.most_common() if c >= 2]
    return terms


def chunk_documents(records: List[Dict], chunk_size: int, chunk_overlap: int) -> List[Dict]:
    """
    把每条 record 的正文切成 chunk。
    
    每个 chunk 继承父文档的所有 metadata,再加上 chunk 位置信息。
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # 切分点优先级: 段落 → 行 → 中文句号 → 中文逗号 → 字符
        separators=["\n\n", "\n", "。", "!", "?", "、", ",", " ", ""],
        length_function=len,  # 中文按字符算
        is_separator_regex=False,
    )
    
    all_chunks = []
    for r in records:
        text = r.get('text', '')
        if not text:
            continue
        
        # 切分
        chunks = splitter.split_text(text)
        total = len(chunks)
        
        for idx, chunk_text in enumerate(chunks):
            chunk_record = {
                # ---- chunk 自己的字段 ----
                'chunk_id': f"{r['id']}_chunk_{idx:04d}",
                'chunk_index': idx,
                'total_chunks': total,
                'chunk_text': chunk_text,
                'chunk_len': len(chunk_text),
                
                # ---- 继承父文档 metadata (RAG 时做过滤和引用) ----
                'doc_id': r['id'],
                'doc_url': r['url'],
                'doc_title': r['title'],
                'ccar_no': r.get('ccar_no'),
                'doc_no': r.get('doc_no'),
                'issued_date': r.get('issued_date'),
                'validity': r.get('validity'),
                'issuing_dept': r.get('issuing_dept'),
                'category': r.get('category'),
            }
            all_chunks.append(chunk_record)
    
    return all_chunks


def main():
    parser = argparse.ArgumentParser(description="数据清洗 + 词典抽取 + chunk 切分")
    parser.add_argument('--input', type=Path,
                        default=Path('../../../data/C4/processed/ccar_regulations.jsonl'))
    parser.add_argument('--output-chunks', type=Path,
                        default=Path('../../../data/C4/processed/ccar_chunks.jsonl'))
    parser.add_argument('--output-dict', type=Path,
                        default=Path('../../../data/C4/aviation_dict.txt'))
    parser.add_argument('--chunk-size', type=int, default=CHUNK_SIZE)
    parser.add_argument('--chunk-overlap', type=int, default=CHUNK_OVERLAP)
    args = parser.parse_args()
    
    # ---- Step 1: 加载 + 过滤 ----
    print(f"[1] 加载 {args.input}")
    raw_records = []
    with args.input.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                raw_records.append(json.loads(line))
    print(f"    原始记录: {len(raw_records)}")
    
    valid_records = []
    filter_stats = Counter()
    for r in raw_records:
        is_valid, reason = is_record_valid(r)
        filter_stats[reason] += 1
        if is_valid:
            valid_records.append(r)
    
    print(f"\n[2] 过滤统计:")
    for reason, count in filter_stats.most_common():
        marker = '✅' if reason == 'ok' else '❌'
        print(f"    {marker} {reason}: {count}")
    print(f"    有效记录: {len(valid_records)} / {len(raw_records)} "
          f"({len(valid_records)/len(raw_records)*100:.0f}%)")
    
    # ---- Step 3: 抽取航空术语词典 ----
    print(f"\n[3] 从 {len(valid_records)} 条记录抽取航空术语...")
    terms = extract_aviation_terms(valid_records)
    print(f"    抽取到 {len(terms)} 个术语")
    print(f"    前 20 个高频术语示例:")
    for t in terms[:20]:
        print(f"      - {t}")
    
    args.output_dict.parent.mkdir(parents=True, exist_ok=True)
    args.output_dict.write_text('\n'.join(terms), encoding='utf-8')
    print(f"    已写入: {args.output_dict}")
    
    # ---- Step 4: chunk 切分 ----
    print(f"\n[4] 切分 chunk (size={args.chunk_size}, overlap={args.chunk_overlap})...")
    chunks = chunk_documents(valid_records, args.chunk_size, args.chunk_overlap)
    print(f"    共产出 chunk: {len(chunks)}")
    
    # 长度分布
    chunk_lens = [c['chunk_len'] for c in chunks]
    print(f"    chunk 长度: 最小 {min(chunk_lens)}, 中位 {sorted(chunk_lens)[len(chunk_lens)//2]}, "
          f"最大 {max(chunk_lens)}, 平均 {sum(chunk_lens)//len(chunk_lens)}")
    
    # 每篇文档的 chunk 数分布
    chunks_per_doc = Counter()
    for c in chunks:
        chunks_per_doc[c['doc_id']] += 1
    counts = list(chunks_per_doc.values())
    counts.sort()
    print(f"    每文档 chunk 数: 最小 {counts[0]}, 中位 {counts[len(counts)//2]}, "
          f"最大 {counts[-1]}")
    
    # ---- Step 5: 落盘 ----
    print(f"\n[5] 写入 {args.output_chunks}")
    args.output_chunks.parent.mkdir(parents=True, exist_ok=True)
    with args.output_chunks.open('w', encoding='utf-8') as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + '\n')
    
    print(f"\n[完成] {len(chunks)} 个 chunk 已就绪,可以进入第一节实验")


if __name__ == '__main__':
    main()