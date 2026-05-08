"""
上传与转换 API 路由
═══════════════════════════════════════════════════════
POST /upload       — 同步模式：完整转写后一次性返回 Q&A Markdown
POST /upload/stream — 流式模式：通过 SSE 实时推送转写文本，完成后返回 Q&A

SSE 事件类型（新增 phase 事件展示全流程进度）：
  - phase       阶段切换，字段 { stage, message }
                stage 取值：uploaded / asr_started / asr_complete / qa_started / qa_complete
  - chunk       转写文本片段，字段 { text, accumulated_chars }
  - qa_done     Q&A 提取完成，字段 { markdown, qa_count, file }
  - error       错误信息，字段 { message }
"""

import asyncio
import json
import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from config import INTERVIEWS_DIR, TEMP_DIR
from src.services.qa_extractor import extract_qa_pairs, format_qa_markdown
from src.services.transcriber import transcribe, transcribe_stream

logger = logging.getLogger(__name__)

router = APIRouter()

_stream_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="asr_stream")

ALLOWED_TYPES = {"audio/m4a", "audio/mp3", "audio/mp4", "audio/wav", "audio/x-m4a", "audio/x-wav"}


# ═══════════════════════════════════════════════════════════
#  POST /upload —— 同步模式
# ═══════════════════════════════════════════════════════════

@router.post("/upload")
async def upload_audio(file: UploadFile):
    logger.info("Received: %s (type=%s)", file.filename, file.content_type)

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    suffix = os.path.splitext(file.filename)[1] or ".m4a"
    temp_name = f"{uuid.uuid4().hex}{suffix}"
    temp_path = os.path.join(TEMP_DIR, temp_name)

    try:
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)

        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # ── 阶段 1/2：ASR 语音识别 ──
        logger.info("══════════════════════════════════════════")
        logger.info("  [1/2] ASR 语音识别 — 开始")
        logger.info("══════════════════════════════════════════")

        raw_text = transcribe(temp_path)

        logger.info("  [1/2] ASR 语音识别 — 完成 (%d 字符)", len(raw_text))
        logger.info("══════════════════════════════════════════")

        if not raw_text or not raw_text.strip():
            return JSONResponse({
                "status": "warning",
                "message": "音频已处理，但未识别到有效语音内容。",
                "markdown": "# 访谈记录\n\n（未识别到有效语音内容）",
            })

        # ── 阶段 2/2：Q&A 提取 ──
        logger.info("  [2/2] Q&A 提取 — 开始")
        logger.info("══════════════════════════════════════════")

        pairs = extract_qa_pairs(raw_text)
        markdown = format_qa_markdown(pairs)

        logger.info("  [2/2] Q&A 提取 — 完成 (%d 组问答)", len(pairs))
        logger.info("══════════════════════════════════════════")

        date_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        base = os.path.splitext(file.filename)[0]
        output_name = f"{date_str}_{base}.md"
        output_path = os.path.join(INTERVIEWS_DIR, output_name)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown)

        logger.info("  结果已保存: %s", output_path)

        return JSONResponse({
            "status": "success",
            "message": "转换完成！",
            "markdown": markdown,
            "file": output_name,
            "qa_count": len(pairs),
        })

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Processing failed for %s", file.filename)
        raise HTTPException(status_code=500, detail=f"处理失败：{exc}") from exc

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ═══════════════════════════════════════════════════════════
#  POST /upload/stream —— SSE 流式模式
# ═══════════════════════════════════════════════════════════

@router.post("/upload/stream")
async def upload_audio_stream(file: UploadFile):
    logger.info("══════════════════════════════════════════")
    logger.info("  收到流式请求: %s (%s)", file.filename, file.content_type)

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    suffix = os.path.splitext(file.filename)[1] or ".m4a"
    temp_name = f"{uuid.uuid4().hex}{suffix}"
    temp_path = os.path.join(TEMP_DIR, temp_name)

    with open(temp_path, "wb") as f:
        content = await file.read()
        f.write(content)

    if len(content) == 0:
        os.remove(temp_path)
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    async def event_generator():
        collected_chunks: list[str] = []
        chunk_count = 0
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue(maxsize=256)

        # ── 转写线程 ──
        def _sync_transcribe():
            try:
                for chunk in transcribe_stream(temp_path):
                    chunk_count_val = len(collected_chunks) + 1
                    loop.call_soon_threadsafe(
                        lambda c=chunk, n=chunk_count_val: queue.put_nowait(
                            {"type": "chunk", "text": c, "index": n}
                        )
                    )
                loop.call_soon_threadsafe(
                    lambda: queue.put_nowait({"type": "stream_end"})
                )
            except Exception as exc:
                logger.exception("Stream transcription failed")
                loop.call_soon_threadsafe(
                    lambda e=exc: queue.put_nowait({"type": "error", "message": str(e)})
                )

        _stream_executor.submit(_sync_transcribe)

        # ── 阶段 1/2：ASR 语音识别 ──
        logger.info("  [1/2] ASR 语音识别 — 开始")

        # 告诉前端进入转写阶段
        yield _sse_event("phase", {
            "stage": "asr_started",
            "message": "ASR 语音识别中...",
        })

        while True:
            msg = await queue.get()
            msg_type = msg["type"]

            if msg_type == "chunk":
                text = msg["text"]
                collected_chunks.append(text)
                chunk_count = len(collected_chunks)
                yield _sse_event("chunk", {
                    "text": text,
                    "accumulated_chars": sum(len(c) for c in collected_chunks),
                })
                # 终端日志：每 3 个 chunk 输出一次进度
                if chunk_count % 3 == 1:
                    logger.info("    片段 %d，累计 %d 字符",
                                chunk_count, sum(len(c) for c in collected_chunks))

            elif msg_type == "stream_end":
                raw_text = "".join(collected_chunks)
                logger.info("  [1/2] ASR 语音识别 — 完成 (%d 字符, %d 片段)",
                            len(raw_text), chunk_count)
                logger.info("══════════════════════════════════════════")

                # 告诉前端转写完成
                yield _sse_event("phase", {
                    "stage": "asr_complete",
                    "message": f"识别完成，共 {len(raw_text)} 字符",
                })

                if not raw_text or not raw_text.strip():
                    yield _sse_event("qa_done", {
                        "markdown": "# 访谈记录\n\n（未识别到有效语音内容）",
                        "qa_count": 0,
                    })
                    break

                # ── 阶段 2/2：Q&A 提取 ──
                logger.info("  [2/2] Q&A 提取 — 开始")
                logger.info("══════════════════════════════════════════")

                yield _sse_event("phase", {
                    "stage": "qa_started",
                    "message": "正在提取问答...",
                })

                try:
                    pairs = extract_qa_pairs(raw_text)
                    markdown = format_qa_markdown(pairs)

                    date_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
                    base = os.path.splitext(file.filename)[0]
                    output_name = f"{date_str}_{base}.md"
                    output_path = os.path.join(INTERVIEWS_DIR, output_name)
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(markdown)

                    logger.info("  [2/2] Q&A 提取 — 完成 (%d 组问答)", len(pairs))
                    logger.info("══════════════════════════════════════════")
                    logger.info("  结果已保存: %s", output_path)

                    yield _sse_event("phase", {
                        "stage": "qa_complete",
                        "message": f"提取完成，{len(pairs)} 组问答",
                    })
                    yield _sse_event("qa_done", {
                        "markdown": markdown,
                        "qa_count": len(pairs),
                        "file": output_name,
                    })
                except Exception as exc:
                    logger.exception("Q&A extraction failed")
                    yield _sse_event("error", {"message": f"Q&A 提取失败：{exc}"})

                break

            elif msg_type == "error":
                yield _sse_event("error", {"message": msg["message"]})
                break

        # 清理
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


def _sse_event(event: str, data: dict) -> str:
    """格式化一条 SSE 消息。

    SSE 协议格式：
        event: <事件类型>\n
        data: <JSON 数据>\n
        \n
    """
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"
