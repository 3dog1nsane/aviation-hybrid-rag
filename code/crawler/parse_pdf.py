"""
PDF 文本提取模块。

降级策略:
1. pdfplumber 优先(中文支持好,layout 准确)
2. pdftotext (poppler-utils) 兜底
3. 扫描件直接放弃,标记为 needs_ocr

输出: (text, status) 元组
    text:   提取的纯文本,失败时为 ""
    status: "ok" / "fallback_pdftotext" / "needs_ocr" / "failed"
"""

import subprocess
from pathlib import Path
from typing import Tuple

import pdfplumber


# 经验阈值:抽到的文本如果少于这个字符数,认为是扫描件或解析失败
MIN_VALID_TEXT_LEN = 200


def extract_pdf_text(pdf_path: Path) -> Tuple[str, str]:
    """
    从 PDF 文件提取纯文本。
    
    Args:
        pdf_path: PDF 文件路径
    
    Returns:
        (text, status):
            text: 提取的文本,失败时为 ""
            status: 
                "ok"                 - pdfplumber 成功
                "fallback_pdftotext" - pdfplumber 拿不到,pdftotext 救回来
                "needs_ocr"          - 都拿不到,可能是扫描件
                "failed"             - PDF 文件本身坏了
    """
    if not pdf_path.exists():
        return "", "failed"

    # ---- 第一级: pdfplumber ----
    try:
        text = _extract_with_pdfplumber(pdf_path)
        if len(text) >= MIN_VALID_TEXT_LEN:
            return text, "ok"
        # 文本太短,继续降级
    except Exception as e:
        print(f"  [pdfplumber 异常] {pdf_path.name}: {e}")
        text = ""

    # ---- 第二级: pdftotext ----
    try:
        text2 = _extract_with_pdftotext(pdf_path)
        if len(text2) >= MIN_VALID_TEXT_LEN:
            return text2, "fallback_pdftotext"
    except Exception as e:
        print(f"  [pdftotext 异常] {pdf_path.name}: {e}")

    # ---- 都失败 ----
    # 区分"扫描件"和"完全坏掉":
    # 如果 pdfplumber 至少打开了 PDF(没抛异常),那是扫描件
    # 否则是文件本身有问题
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if len(pdf.pages) > 0:
                return "", "needs_ocr"
    except Exception:
        pass

    return "", "failed"


def _extract_with_pdfplumber(pdf_path: Path) -> str:
    """用 pdfplumber 抽取文本,逐页合并。"""
    pages_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            if t.strip():
                pages_text.append(t)
    return "\n\n".join(pages_text)


def _extract_with_pdftotext(pdf_path: Path) -> str:
    """
    用 pdftotext 命令行工具抽取(poppler-utils 提供)。
    -layout 保留布局,对多栏排版更友好。
    """
    result = subprocess.run(
        ["pdftotext", "-layout", str(pdf_path), "-"],
        capture_output=True,
        timeout=30,
    )
    if result.returncode != 0:
        return ""
    # pdftotext 输出 UTF-8
    return result.stdout.decode("utf-8", errors="replace")


# ---- 自测 ----
if __name__ == "__main__":
    """用法: python parse_pdf.py <pdf_file>"""
    import sys
    if len(sys.argv) != 2:
        print("Usage: python parse_pdf.py <pdf_file>")
        sys.exit(1)

    pdf = Path(sys.argv[1])
    text, status = extract_pdf_text(pdf)
    print(f"Status: {status}")
    print(f"Text length: {len(text)}")
    print(f"First 500 chars:\n{text[:500]}")
