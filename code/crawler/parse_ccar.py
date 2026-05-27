"""
CCAR 法规详情页 HTML 解析器 (v2)

相对 v1 的主要修复:
1. 元数据改用结构化抓取(<li><b>字段:</b>值</li>),不再依赖 get_text() 误抓 display:none 内容
2. ccar_no 专门用 CCAR-数字 正则匹配,不匹配返回 None(而不是降级抓到标题)
3. 正文专门从 class=content 提取,不再"找最长 div",避免误抓元数据列表
4. 新增 PDF 附件链接抓取(pdf_urls 字段)

输出字段(相比 v1 新增 pdf_urls):
    id, url, title, ccar_no, doc_no, issued_date, validity,
    issuing_dept, category, raw_text, related_urls,
    pdf_urls       <- 新增
"""

from typing import Optional, List
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import re


# 关心的元数据字段(中文标签 → 输出字段名)
META_FIELD_MAP = {
    '主题分类': 'category',
    '办文单位': 'issuing_dept',
    '发文日期': 'issued_date',
    '有效性': 'validity',
    '文号': 'doc_no',
    '名称': 'name',
}

# CCAR 部号专用正则: CCAR-91 / CCAR-91-R2 / CCAR-121-R5 等
CCAR_NO_PATTERN = re.compile(r'CCAR-\d+(?:-R\d+)?', re.IGNORECASE)

# 相关链接 URL 模式
RELATED_URL_PATTERN = re.compile(r'/XXGK/XXGK/[^/]+/\d+/t\d+_\d+\.html')


def parse_ccar_page(html: str, page_url: str) -> Optional[dict]:
    """
    解析一个 CCAR 法规详情页。
    
    Returns: dict 或 None(不是详情页时)
    """
    soup = BeautifulSoup(html, 'lxml')

    metadata = _extract_metadata_structured(soup)
    title = _extract_title(soup, metadata.get('name'))
    if not title:
        return None

    ccar_no = _extract_ccar_no(html)
    raw_text = _extract_main_text(soup)
    related_urls = _extract_related_urls(soup, page_url)
    pdf_urls = _extract_pdf_urls(soup, page_url)

    match = re.search(r't(\d+_\d+)\.html', page_url)
    page_id = f"ccar_{match.group(1)}" if match else page_url

    return {
        "id": page_id,
        "url": page_url,
        "title": title,
        "ccar_no": ccar_no,
        "doc_no": metadata.get('doc_no'),
        "issued_date": metadata.get('issued_date'),
        "validity": metadata.get('validity'),
        "issuing_dept": metadata.get('issuing_dept'),
        "category": metadata.get('category'),
        "raw_text": raw_text,
        "related_urls": related_urls,
        "pdf_urls": pdf_urls,
    }


def _extract_metadata_structured(soup: BeautifulSoup) -> dict:
    """
    从 <li><b>字段名:</b>值</li> 结构中提取元数据。
    
    v2 的核心修复:精确定位 <li> 和 <b>,不再用 get_text() 全文匹配。
    """
    metadata = {}

    for li in soup.find_all('li'):
        b_tag = li.find('b')
        if not b_tag:
            continue

        # 字段名(去掉中英文冒号 -- 半角 ':' 和全角 '：')
        label = b_tag.get_text(strip=True).rstrip(':').rstrip('：').strip()
        if label not in META_FIELD_MAP:
            continue

        # 字段值: <li> 内 <b> 后的文本
        full_text = li.get_text(strip=True)
        b_text = b_tag.get_text(strip=True)
        value = full_text[len(b_text):].strip()

        if value and value not in ('', '无', '/'):
            output_key = META_FIELD_MAP[label]
            if output_key not in metadata:
                metadata[output_key] = value

    # 兜底:"名称"字段可能是跨 li 结构
    # <li class="content_nav_left" style="..."><b>名称:</b></li><li style="...">实际名称</li>
    if 'name' not in metadata:
        for b in soup.find_all('b'):
            label = b.get_text(strip=True).rstrip(':').rstrip('：').strip()
            if label == '名称':
                parent_li = b.find_parent('li')
                if parent_li:
                    sibling = parent_li.find_next_sibling('li')
                    if sibling:
                        text = sibling.get_text(strip=True)
                        if text:
                            metadata['name'] = text
                break

    return metadata


def _extract_ccar_no(html: str) -> Optional[str]:
    """匹配 CCAR-XX 或 CCAR-XX-RX。不匹配返回 None。"""
    match = CCAR_NO_PATTERN.search(html)
    return match.group(0).upper() if match else None


def _extract_title(soup: BeautifulSoup, fallback: Optional[str]) -> Optional[str]:
    """
    标题提取,优先级:
    1. <meta name="ArticleTitle" content="...">  (最可靠)
    2. <div class="content_t">
    3. <title>
    4. 元数据"名称"
    """
    meta = soup.find('meta', attrs={'name': 'ArticleTitle'})
    if meta and meta.get('content'):
        return meta['content'].strip()

    title_div = soup.find('div', class_='content_t')
    if title_div:
        text = title_div.get_text(strip=True)
        if text:
            return text

    title_tag = soup.find('title')
    if title_tag and title_tag.text.strip():
        title = title_tag.text.strip()
        for sep in [' - ', ' | ', '_', '-']:
            if sep in title:
                title = title.split(sep)[0].strip()
                break
        return title

    return fallback


def _extract_main_text(soup: BeautifulSoup) -> str:
    """
    抽取正文。
    
    v2 改动:精确定位 <div class="content" data-role="n_content">。
    
    注意:
    - 这个 div 可能只含"附件:XXX.pdf"链接(典型,需要靠 PDF 补正文)
    - 也可能含真正的法规正文(主体规章 HTML 内嵌)
    - 不管哪种,返回它,下游判断长度决定是否要 PDF
    """
    content_div = soup.find('div', class_='content', attrs={'data-role': 'n_content'})
    
    if not content_div:
        content_div = soup.find('div', class_=lambda c: c == 'content' if c else False)
    
    if not content_div:
        return ""
    
    # 去脚本/样式标签
    for tag in content_div.find_all(['script', 'style']):
        tag.decompose()
    
    text = content_div.get_text('\n', strip=True)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text


def _extract_related_urls(soup: BeautifulSoup, page_url: str) -> List[str]:
    """相关链接(同 v1)。"""
    related = set()
    for a in soup.find_all('a', href=True):
        href = a['href']
        abs_url = urljoin(page_url, href)
        if RELATED_URL_PATTERN.search(abs_url):
            abs_url = abs_url.split('#')[0].split('?')[0]
            related.add(abs_url)
    related.discard(page_url)
    return sorted(related)


def _extract_pdf_urls(soup: BeautifulSoup, page_url: str) -> List[str]:
    """
    抽取页面里的 PDF 附件链接。
    扫描所有 .pdf 后缀的 a 标签,返回去重的绝对路径列表。
    """
    pdfs = set()
    for a in soup.find_all('a', href=True):
        href = a['href']
        if href.lower().endswith('.pdf'):
            abs_url = urljoin(page_url, href)
            abs_url = abs_url.split('#')[0].split('?')[0]
            pdfs.add(abs_url)
    return sorted(pdfs)


# ---- 自测 ----
if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) != 3:
        print("Usage: python parse_ccar.py <html_file> <page_url>")
        sys.exit(1)

    with open(sys.argv[1], 'r', encoding='utf-8') as f:
        html = f.read()
    result = parse_ccar_page(html, sys.argv[2])
    if result:
        print_dict = {k: v for k, v in result.items() if k != 'raw_text'}
        print_dict['raw_text_len'] = len(result['raw_text'])
        print_dict['raw_text_preview'] = result['raw_text'][:300]
        print(json.dumps(print_dict, ensure_ascii=False, indent=2))
    else:
        print("Not a regulation detail page")