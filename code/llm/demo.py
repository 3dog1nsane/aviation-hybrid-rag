"""
命令行交互式 RAG 问答 Demo

用法（在项目根目录或任意目录均可）:
    python code/llm/demo.py
"""
import sys
from pathlib import Path

# 让本文件可以 import 同目录下的 rag_qa
sys.path.insert(0, str(Path(__file__).parent))

from rag_qa import RAGQA  # noqa: E402


def main():
    print("初始化 RAG 问答系统（首次加载需要几十秒）...")
    qa = RAGQA().load()
    print("\n初始化完成！输入问题开始对话，输入 'quit' / 'exit' / 'q' 退出。\n")

    while True:
        try:
            query = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if query.lower() in ("quit", "exit", "q"):
            print("再见！")
            break

        if not query:
            continue

        try:
            qa.ask(query)  # ask() 内部 verbose=True 已经把答案打印出来
            print()
        except Exception as e:
            print(f"[ERROR] {type(e).__name__}: {e}\n")


if __name__ == "__main__":
    main()