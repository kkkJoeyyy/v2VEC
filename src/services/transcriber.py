"""
ASR 语音转写服务（阿里云 DashScope API）
═══════════════════════════════════════════════════════
通过阿里云百炼平台 qwen3-asr-flash 模型进行语音转写。

性能优化：
  - 分段并行转写：多个音频片段同时调用 API（max_workers=3）
  - 失败重试：指数退避重试，最多 3 次
  - 音频先切分为 WAV（PCM16LE 16kHz 单声道），压缩后体积更小，上传更快

前置条件：
  - 设置 DASHSCOPE_API_KEY（.env 或环境变量）
  - 安装 ffmpeg / ffprobe
"""

import base64
import logging
import os
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from dashscope import MultiModalConversation

from config import DASHSCOPE_API_KEY, ASR_MODEL, ASR_LANGUAGE

logger = logging.getLogger(__name__)

_CHUNK_DURATION = 10        # 每段时长（秒），短段更快返回
_OVERLAP = 1.5              # 段间重叠（秒）
_PARALLEL_WORKERS = 3       # 并行调用 API 的线程数
_MAX_RETRIES = 3            # 单段最大重试次数


# ═══════════════════════════════════════════════════════════
#  公开接口
# ═══════════════════════════════════════════════════════════

def transcribe(file_path: str, language: str = ASR_LANGUAGE) -> str:
    """同步转写：将音频文件发送到 DashScope API，返回完整文本。"""
    _check_api_key()
    audio_uri = _encode_audio_data_uri(file_path)
    response = _call_asr_with_retry(audio_uri, language)
    return _parse_response(response)


def transcribe_stream(file_path: str, language: str = ASR_LANGUAGE):
    """流式转写：音频切片后并行调用 API，按序产出文本。

    策略：
      1. ffmpeg 将音频切成 10s 片段（WAV 格式，体积小）
      2. 用线程池并行提交所有片段（最多 3 个并发）
      3. 每个片段最多重试 3 次（指数退避）
      4. 按原始顺序 yield 结果（保证文本连续性）
    """
    _check_api_key()

    duration = _get_audio_duration(file_path)
    if duration is None or duration <= _CHUNK_DURATION + 5:
        text = transcribe(file_path, language)
        if text.strip():
            yield text.strip()
        return

    logger.info("Audio duration: %.1fs, splitting into %ds chunks.", duration, _CHUNK_DURATION)
    chunk_paths = _split_audio(file_path, duration)
    total = len(chunk_paths)
    logger.info("Split into %d chunks, processing with %d parallel workers.", total, _PARALLEL_WORKERS)

    # ── 并行处理所有片段，收集结果 ──
    results: dict[int, str] = {}

    with ThreadPoolExecutor(max_workers=_PARALLEL_WORKERS, thread_name_prefix="asr_chunk") as pool:
        futures = {}
        for i, chunk_path in enumerate(chunk_paths):
            future = pool.submit(_process_one_chunk, chunk_path, language, i, total)
            futures[future] = i

        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                logger.error("Chunk %d/%d permanently failed: %s", idx + 1, total, exc)
                results[idx] = ""

    # ── 按序 yield，执行去重 ──
    prev_text = ""
    for i in range(total):
        chunk_text = results.get(i, "").strip()
        if chunk_text:
            deduped = _deduplicate_overlap(prev_text, chunk_text)
            if deduped:
                yield deduped
            prev_text = chunk_text

        # 清理临时片段
        if i < len(chunk_paths) and os.path.exists(chunk_paths[i]):
            os.remove(chunk_paths[i])


# ═══════════════════════════════════════════════════════════
#  单段处理（编码 + API 调用 + 重试）
# ═══════════════════════════════════════════════════════════

def _process_one_chunk(chunk_path: str, language: str, idx: int, total: int) -> str:
    """处理单个音频片段：编码 → 调 API（含重试） → 返回文本。

    此函数在 ThreadPoolExecutor 线程中运行，多个片段可并行执行。
    """
    logger.info("Chunk %d/%d: encoding...", idx + 1, total)
    audio_uri = _encode_audio_data_uri(chunk_path)

    logger.info("Chunk %d/%d: calling API (%d bytes)...", idx + 1, total, len(audio_uri))
    response = _call_asr_with_retry(audio_uri, language)
    text = _parse_response(response)
    logger.info("Chunk %d/%d: done (%d chars)", idx + 1, total, len(text))
    return text


# ═══════════════════════════════════════════════════════════
#  DashScope API 调用（含重试）
# ═══════════════════════════════════════════════════════════

def _check_api_key():
    if not DASHSCOPE_API_KEY:
        raise RuntimeError(
            "DASHSCOPE_API_KEY 未配置。请在 .env 文件中设置或通过环境变量导出。"
        )


def _encode_audio_data_uri(file_path: str) -> str:
    """读取音频文件，编码为 base64 data URI。

    ffmpeg 已经将片段转为 16kHz WAV（PCM16LE 单声道），
    体积已经较小，base64 编码后约为原始 WAV 的 1.33 倍。
    """
    ext = os.path.splitext(file_path)[1].lower().lstrip(".")
    mime_map = {
        "wav": "audio/wav", "mp3": "audio/mpeg", "m4a": "audio/mp4",
        "mp4": "audio/mp4", "aac": "audio/aac", "ogg": "audio/ogg",
        "flac": "audio/flac",
    }
    mime = mime_map.get(ext, "audio/wav")
    with open(file_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _call_asr_with_retry(audio_uri: str, language: str, max_retries: int = _MAX_RETRIES):
    """调用 DashScope ASR API，失败时指数退避重试。

    Args:
        audio_uri: base64 data URI 格式的音频
        language: 语言代码
        max_retries: 最大重试次数

    Returns:
        MultiModalConversation 响应对象

    Raises:
        RuntimeError: 重试耗尽后仍失败
    """
    last_error = None
    for attempt in range(max_retries):
        try:
            return _call_asr(audio_uri, language)
        except Exception as exc:
            last_error = exc
            if attempt < max_retries - 1:
                wait = 2 ** attempt  # 1s, 2s, 4s
                logger.warning(
                    "API call failed (attempt %d/%d), retrying in %ds: %s",
                    attempt + 1, max_retries, wait, exc,
                )
                time.sleep(wait)
            else:
                logger.error("API call failed after %d attempts: %s", max_retries, exc)

    raise RuntimeError(f"API call failed after {max_retries} retries") from last_error


def _call_asr(audio_uri: str, language: str):
    """单次调用 DashScope MultiModalConversation API。

    参考阿里官方示例：
      - system 角色消息（可配置上下文）
      - user 角色消息包含音频 data URI
      - result_format="message"
      - asr_options 配置语言检测
    """
    lang_code = "zh" if language in ("Chinese", "chinese", "zh", "cn") else language

    messages = [
        {"role": "system", "content": [{"text": ""}]},
        {"role": "user", "content": [{"audio": audio_uri}]},
    ]

    return MultiModalConversation.call(
        api_key=DASHSCOPE_API_KEY,
        model=ASR_MODEL,
        messages=messages,
        result_format="message",
        asr_options={
            "language": lang_code,
            "enable_lid": True,
            "enable_itn": False,
        },
    )


def _parse_response(response) -> str:
    """从 MultiModalConversation 响应中提取转写文本。"""
    try:
        output = response.output
        if output and output.choices:
            choice = output.choices[0]
            content = choice.message.content
            if isinstance(content, list) and len(content) > 0:
                return content[0].get("text", "") or ""
            if isinstance(content, str):
                return content
            if hasattr(choice.message, "text"):
                return choice.message.text or ""
    except (AttributeError, KeyError, IndexError, TypeError) as exc:
        logger.warning("Failed to parse ASR response: %s", exc)

    try:
        output = response.get("output", {})
        choices = output.get("choices", [])
        if choices:
            msg = choices[0].get("message", {})
            content = msg.get("content", [])
            if isinstance(content, list) and content:
                return content[0].get("text", "")
            if isinstance(content, str):
                return content
    except Exception:
        pass

    logger.warning("Could not extract text from ASR response.")
    return ""


# ═══════════════════════════════════════════════════════════
#  ffmpeg 音频分段
# ═══════════════════════════════════════════════════════════

def _get_audio_duration(file_path: str) -> float | None:
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                file_path,
            ],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            return float(result.stdout.strip())
    except (subprocess.TimeoutExpired, ValueError, FileNotFoundError) as exc:
        logger.warning("ffprobe failed: %s", exc)
    return None


def _split_audio(file_path: str, duration: float) -> list[str]:
    chunk_dir = tempfile.mkdtemp(prefix="asr_chunks_")
    chunks = []
    start = 0.0
    while start < duration:
        chunk_path = os.path.join(chunk_dir, f"chunk_{int(start):04d}.wav")
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", file_path,
                "-ss", str(start),
                "-t", str(_CHUNK_DURATION + _OVERLAP),
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                chunk_path,
            ],
            check=True, timeout=30,
        )
        chunks.append(chunk_path)
        start += _CHUNK_DURATION
    return chunks


def _deduplicate_overlap(prev: str, current: str) -> str:
    if not prev or not current:
        return current
    max_overlap = min(len(prev), len(current), 20)
    for n in range(max_overlap, 2, -1):
        if prev[-n:] == current[:n]:
            return current[n:].strip()
    return current
