"""
CCAR 法规爬虫主程序 (v2)

新增能力:
1. 下载 HTML 后,如果发现 PDF 附件,继续下载 PDF
2. PDF 解析为纯文本(parse_pdf 模块)
3. 输出字段新增:
     - pdf_text:        PDF 解析的文本(可能为 "")
     - extraction_status: ok / fallback_pdftotext / needs_ocr / failed / no_pdf
     - text:            最终用于检索的文本(优先 PDF 文本,否则 HTML 正文)
"""

import argparse
import json
import time
import re
import hashlib
from pathlib import Path
from collections import deque
from typing import Set, Optional, Tuple

import requests
from tqdm import tqdm

from parse_ccar import parse_ccar_page
from parse_pdf import extract_pdf_text


DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/pdf",
    "Accept-Language": "zh-CN,zh;q=0.9",
}
REQUEST_TIMEOUT = 30           # PDF 比 HTML 大,超时拉长
SLEEP_BETWEEN_REQUESTS = 1.0
MAX_RETRIES = 2

# 如果 HTML 正文长度 < 这个阈值,认为正文实际在 PDF 里
HTML_TEXT_THRESHOLD = 500


def url_to_html_filename(url: str) -> str:
    """HTML URL → 缓存文件名,如 t20151102_8444.html"""
    match = re.search(r'(t\d+_\d+\.html)', url)
    if match:
        return match.group(1)
    h = hashlib.md5(url.encode()).hexdigest()[:12]
    return f"page_{h}.html"


def url_to_pdf_filename(url: str) -> str:
    """PDF URL → 缓存文件名,如 P020151103346669706346.pdf"""
    match = re.search(r'(P\d+\.pdf)', url)
    if match:
        return match.group(1)
    # fallback: 哈希
    h = hashlib.md5(url.encode()).hexdigest()[:12]
    return f"file_{h}.pdf"


def fetch_html(url: str, raw_dir: Path) -> Optional[str]:
    """获取 HTML,缓存优先,失败重试。"""
    cache_path = raw_dir / url_to_html_filename(url)

    if cache_path.exists():
        try:
            return cache_path.read_text(encoding='utf-8')
        except Exception as e:
            tqdm.write(f"  [HTML 缓存读取失败] {url}: {e}")

    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            resp.encoding = resp.apparent_encoding or 'utf-8'
            html = resp.text
            cache_path.write_text(html, encoding='utf-8')
            return html
        except requests.RequestException as e:
            if attempt < MAX_RETRIES:
                tqdm.write(f"  [HTML 重试 {attempt+1}] {url}: {e}")
                time.sleep(2 ** attempt)
            else:
                tqdm.write(f"  [HTML 失败] {url}: {e}")
                return None


def fetch_pdf(url: str, pdf_dir: Path) -> Optional[Path]:
    """
    下载 PDF,缓存优先。
    
    Returns: 本地 PDF 路径(成功)或 None(失败)
    """
    cache_path = pdf_dir / url_to_pdf_filename(url)

    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path

    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=DEFAULT_HEADERS, 
                                timeout=REQUEST_TIMEOUT, stream=True)
            resp.raise_for_status()
            # 简单校验: Content-Type 应该是 PDF
            ctype = resp.headers.get('Content-Type', '')
            if 'pdf' not in ctype.lower() and 'octet-stream' not in ctype.lower():
                tqdm.write(f"  [PDF 类型异常] {url}: Content-Type={ctype}")
            # 流式写盘
            with cache_path.open('wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return cache_path
        except requests.RequestException as e:
            if attempt < MAX_RETRIES:
                tqdm.write(f"  [PDF 重试 {attempt+1}] {url}: {e}")
                time.sleep(2 ** attempt)
            else:
                tqdm.write(f"  [PDF 失败] {url}: {e}")
                # 清掉残留半成品
                if cache_path.exists():
                    try:
                        cache_path.unlink()
                    except Exception:
                        pass
                return None


def enrich_with_pdf(record: dict, pdf_dir: Path) -> dict:
    """
    给 record 补充 PDF 提取的文本。
    
    逻辑:
    - 如果 HTML 正文已经足够长(>= 阈值),不需要 PDF
    - 如果 HTML 正文短 且 有 pdf_urls,下载第一个 PDF 并解析
    - 其他情况标记 no_pdf
    
    设置最终 text 字段:优先 PDF 文本,否则 HTML 正文
    """
    html_text_len = len(record.get('raw_text', ''))
    pdf_urls = record.get('pdf_urls', [])

    # 情况 1: HTML 正文足够长,不需要 PDF
    if html_text_len >= HTML_TEXT_THRESHOLD:
        record['pdf_text'] = ""
        record['extraction_status'] = "html_only"
        record['text'] = record['raw_text']
        return record

    # 情况 2: HTML 正文短,但没 PDF 可补
    if not pdf_urls:
        record['pdf_text'] = ""
        record['extraction_status'] = "no_pdf"
        record['text'] = record['raw_text']  # 凑合用 HTML 的(其实就一句"附件:..."),后续可以过滤掉
        return record

    # 情况 3: 下载并解析第一个 PDF
    pdf_url = pdf_urls[0]
    pdf_path = fetch_pdf(pdf_url, pdf_dir)

    if pdf_path is None:
        record['pdf_text'] = ""
        record['extraction_status'] = "pdf_download_failed"
        record['text'] = record['raw_text']
        return record

    text, status = extract_pdf_text(pdf_path)
    record['pdf_text'] = text
    record['extraction_status'] = status
    # PDF 文本作为主 text(如果非空),否则回退 HTML 正文
    record['text'] = text if text else record['raw_text']

    return record


def load_seeds(seed_file: Path) -> list:
    seeds = []
    for line in seed_file.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line and not line.startswith('#'):
            seeds.append(line)
    return seeds


def load_visited(jsonl_path: Path) -> Set[str]:
    visited = set()
    if jsonl_path.exists():
        for line in jsonl_path.read_text(encoding='utf-8').splitlines():
            if line.strip():
                try:
                    record = json.loads(line)
                    visited.add(record['url'])
                except (json.JSONDecodeError, KeyError):
                    continue
    return visited


def crawl(seed_file: Path, raw_html_dir: Path, raw_pdf_dir: Path,
          output_jsonl: Path, failed_log: Path, max_pages: int):
    raw_html_dir.mkdir(parents=True, exist_ok=True)
    raw_pdf_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    seeds = load_seeds(seed_file)
    visited = load_visited(output_jsonl)
    failed = []

    print(f"[初始化] 种子数: {len(seeds)}")
    print(f"[初始化] 已爬取: {len(visited)} 条")
    print(f"[初始化] 目标爬取: {max_pages}")

    queue = deque()
    for seed in seeds:
        if seed not in visited:
            queue.append(seed)

    pbar = tqdm(total=max_pages, initial=len(visited), desc="爬取中")
    with output_jsonl.open('a', encoding='utf-8') as fout:
        while queue and len(visited) < max_pages:
            url = queue.popleft()
            if url in visited:
                continue

            html = fetch_html(url, raw_html_dir)
            if html is None:
                failed.append(url)
                visited.add(url)
                continue

            try:
                record = parse_ccar_page(html, url)
            except Exception as e:
                tqdm.write(f"  [解析异常] {url}: {e}")
                failed.append(url)
                visited.add(url)
                continue

            if record is None:
                visited.add(url)
                continue

            # 关键步: 用 PDF 补正文
            try:
                record = enrich_with_pdf(record, raw_pdf_dir)
            except Exception as e:
                tqdm.write(f"  [PDF 处理异常] {url}: {e}")
                record['pdf_text'] = ""
                record['extraction_status'] = "pdf_exception"
                record['text'] = record.get('raw_text', '')

            fout.write(json.dumps(record, ensure_ascii=False) + '\n')
            fout.flush()
            visited.add(url)
            pbar.update(1)

            for next_url in record['related_urls']:
                if next_url not in visited:
                    queue.append(next_url)

            time.sleep(SLEEP_BETWEEN_REQUESTS)

    pbar.close()

    if failed:
        failed_log.write_text('\n'.join(failed), encoding='utf-8')
        print(f"[完成] 失败 {len(failed)} 条,见 {failed_log}")

    print(f"[完成] 共收录 {len(visited)} 个 URL")
    print(f"[完成] jsonl 输出: {output_jsonl}")


def main():
    parser = argparse.ArgumentParser(description="CCAR 法规爬虫 v2(支持 PDF)")
    parser.add_argument('--seed', type=Path, default=Path('seed_urls.txt'))
    parser.add_argument('--raw-html-dir', type=Path,
                        default=Path('../../../data/C4/raw/html'))
    parser.add_argument('--raw-pdf-dir', type=Path,
                        default=Path('../../../data/C4/raw/pdf'))
    parser.add_argument('--output', type=Path,
                        default=Path('../../../data/C4/processed/ccar_regulations.jsonl'))
    parser.add_argument('--failed-log', type=Path, default=Path('failed_urls.txt'))
    parser.add_argument('--max-pages', type=int, default=300)
    args = parser.parse_args()

    crawl(args.seed, args.raw_html_dir, args.raw_pdf_dir,
          args.output, args.failed_log, args.max_pages)


if __name__ == '__main__':
    main()
