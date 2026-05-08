"""
ASR 模型工具模块（CLI / Notebook 独立使用）
═══════════════════════════════════════════════════════
委托到 src.services.transcriber 的 DashScope API 实现。

不再依赖本地 Qwen3-ASR 模型权重。

用法：
    python model/model_utils.py <音频文件路径>
"""

import os
import sys

# 确保项目根目录在 Python path 中
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.services.transcriber import transcribe as _api_transcribe
from src.services.transcriber import transcribe_stream as _api_transcribe_stream


def transcribe_audio(file_path: str, language: str = "Chinese") -> str:
    """转写音频文件 → 返回纯文本（通过 DashScope API）。"""
    return _api_transcribe(file_path, language)


def transcribe_audio_stream(file_path: str):
    """流式转写音频文件，逐块产出文本（通过 DashScope API）。"""
    yield from _api_transcribe_stream(file_path)


def save_transcription_markdown(file_path: str, output_dir: str = "interviews") -> str:
    """一站式：转写音频 + 保存为 Markdown 文件。"""
    text = transcribe_audio(file_path)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{base_name}.md")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# Transcription\n\n{text}\n")

    return output_path


# ═══════════════════════════════════════════════════════════
#  CLI 入口
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else None
    if not target:
        print("Usage: python model/model_utils.py <audio_file>")
        print()
        print("环境变量：")
        print("  DASHSCOPE_API_KEY    阿里云百炼 API Key（必填）")
        print("  DASHSCOPE_BASE_URL   API 端点（默认 dashscope.aliyuncs.com）")
        sys.exit(1)

    try:
        base = os.path.splitext(os.path.basename(target))[0]
        out_path = os.path.join("interviews", f"{base}.md")
        os.makedirs("interviews", exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write("# Transcription\n\n")
            for chunk in transcribe_audio_stream(target):
                f.write(f"{chunk}\n")
                print(chunk, end="", flush=True)

        print(f"\n\nSaved to {out_path}")
    except RuntimeError as e:
        print(f"配置错误: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
