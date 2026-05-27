"""
补下 1_Pooling 目录(sentence-transformers 必需)
"""
from modelscope import snapshot_download
from pathlib import Path

LOCAL_DIR = Path.home() / "models" / "bge-m3"

print(f"补下 1_Pooling/ 到 {LOCAL_DIR}")

model_dir = snapshot_download(
    model_id="BAAI/bge-m3",
    local_dir=str(LOCAL_DIR),
    allow_file_pattern=["1_Pooling/*"],
)

print(f"\n✅ 完成: {model_dir}")
print("\n1_Pooling 目录内容:")
pooling_dir = LOCAL_DIR / "1_Pooling"
if pooling_dir.exists():
    for f in pooling_dir.iterdir():
        print(f"  {f.name}: {f.stat().st_size} bytes")
else:
    print("  ❌ 目录没建出来")