"""
从 ModelScope 下载 bge-reranker-v2-m3 到本地。
和 bge-m3 在 ~/models/ 下并列存放。
"""
from modelscope import snapshot_download
from pathlib import Path

LOCAL_DIR = Path.home() / "models" / "bge-reranker-v2-m3"
LOCAL_DIR.mkdir(parents=True, exist_ok=True)

print(f"开始下载 bge-reranker-v2-m3 到 {LOCAL_DIR}")
print("(模型约 2.3GB,首次下载需几分钟)")

# 教训:上次 ignore 文件踩坑了,这次全部下载,不挑剔
# 仅排除明显用不到的 onnx(节省 1GB+)和文档大图
model_dir = snapshot_download(
    model_id="BAAI/bge-reranker-v2-m3",
    local_dir=str(LOCAL_DIR),
    ignore_file_pattern=[
        "*.onnx",
        "*.onnx_data",
        "onnx/*",
    ],
)

print(f"\n✅ 下载完成: {model_dir}")
print("\n文件列表:")
for f in sorted(LOCAL_DIR.iterdir()):
    if f.is_file():
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  {f.name}: {size_mb:.1f} MB")
    elif f.is_dir():
        # 简单展示子目录
        sub_files = list(f.iterdir())
        print(f"  {f.name}/ ({len(sub_files)} files)")