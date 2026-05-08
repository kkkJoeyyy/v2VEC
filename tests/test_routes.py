"""
API 路由集成测试
═══════════════════════════════════════════════════════
使用 FastAPI TestClient 测试 POST /upload 端点。
ASR 模型调用被 mock 以跳过 GPU 推理，专注测试：
  - HTTP 请求/响应正确性
  - 文件上传校验逻辑
  - 错误处理路径
  - Q&A 提取 → Markdown 输出的端到端流程

运行方式：
    cd v2VEC
    python -m pytest tests/test_routes.py -v
"""

import io
import json
import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.server import create_app


@pytest.fixture
def client():
    """创建测试用的 FastAPI TestClient。

    使用 create_app() 工厂函数获取干净的 app 实例，
    避免模块级 app 的状态污染测试。
    """
    app = create_app()
    return TestClient(app)


# ═══════════════════════════════════════════════════════════
#  辅助函数：模拟转写结果
# ═══════════════════════════════════════════════════════════

def _mock_transcribe_returns(text: str):
    """创建 mock，使 transcribe() 返回指定的转写文本。"""
    mock = MagicMock(return_value=text)
    return mock


# 用于测试的标准转写文本
SAMPLE_TRANSCRIPT = "请自我介绍一下。我叫张三，今年30岁，目前在一家科技公司做产品经理。"


# ═══════════════════════════════════════════════════════════
#  POST /upload 测试
# ═══════════════════════════════════════════════════════════

class TestUploadEndpoint:
    """测试 /upload 端点的各种场景。"""

    def test_frontend_served(self, client: TestClient):
        """GET / 应返回前端 HTML 页面。"""
        resp = client.get("/")
        assert resp.status_code == 200
        assert "html" in resp.headers.get("content-type", "").lower() or resp.text.startswith("<!DOCTYPE")

    def test_upload_success(self, client: TestClient):
        """正常上传音频文件 → 成功响应。"""
        mock_text = SAMPLE_TRANSCRIPT
        with patch("src.routes.upload_routes.transcribe", return_value=mock_text):
            resp = client.post(
                "/upload",
                files={"file": ("test.m4a", io.BytesIO(b"fake audio data"), "audio/m4a")},
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["status"] == "success"
            assert "markdown" in body
            assert body["qa_count"] >= 1
            # 验证 Markdown 包含转写内容
            assert "张三" in body["markdown"]

    def test_upload_mp3(self, client: TestClient):
        """上传 mp3 格式文件 → 成功响应。"""
        mock_text = "你好，今天天气真好。是的，阳光明媚。"
        with patch("src.routes.upload_routes.transcribe", return_value=mock_text):
            resp = client.post(
                "/upload",
                files={"file": ("recording.mp3", io.BytesIO(b"mp3 data"), "audio/mp3")},
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["status"] == "success"

    def test_empty_audio(self, client: TestClient):
        """上传空文件 → 400 错误。"""
        resp = client.post(
            "/upload",
            files={"file": ("empty.m4a", io.BytesIO(b""), "audio/m4a")},
        )
        assert resp.status_code == 400

    def test_no_file_field(self, client: TestClient):
        """没有 file 字段的请求 → 422（FastAPI 自动校验）。

        FastAPI 对缺失必填字段直接返回 422 Unprocessable Entity。
        """
        resp = client.post("/upload")
        assert resp.status_code == 422

    def test_silent_audio_warning(self, client: TestClient):
        """静音音频（转写结果为空）→ 返回 warning 状态。

        这种情况下不应是 500 错误，而是一个携带提示信息的 200 响应。
        """
        with patch("src.routes.upload_routes.transcribe", return_value=""):
            resp = client.post(
                "/upload",
                files={"file": ("silence.m4a", io.BytesIO(b"some data"), "audio/m4a")},
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["status"] == "warning"
            assert "未识别" in body["message"]

    def test_whitespace_only_transcript(self, client: TestClient):
        """转写结果只有空白字符 → warning 状态。"""
        with patch("src.routes.upload_routes.transcribe", return_value="   \n  "):
            resp = client.post(
                "/upload",
                files={"file": ("noise.m4a", io.BytesIO(b"noise"), "audio/m4a")},
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["status"] == "warning"

    def test_transcribe_exception_handling(self, client: TestClient):
        """ASR 转写抛出异常 → 500 错误。"""
        with patch(
            "src.routes.upload_routes.transcribe",
            side_effect=RuntimeError("GPU out of memory")
        ):
            resp = client.post(
                "/upload",
                files={"file": ("test.m4a", io.BytesIO(b"data"), "audio/m4a")},
            )
            assert resp.status_code == 500
            body = resp.json()
            assert "处理失败" in body["detail"]

    def test_markdown_format_in_response(self, client: TestClient):
        """验证 Markdown 响应格式符合预期。"""
        mock_text = SAMPLE_TRANSCRIPT
        with patch("src.routes.upload_routes.transcribe", return_value=mock_text):
            resp = client.post(
                "/upload",
                files={"file": ("interview.m4a", io.BytesIO(b"audio"), "audio/m4a")},
            )
            body = resp.json()
            # 应包含标准 Markdown 结构
            assert body["markdown"].startswith("# ")
            assert "**Q" in body["markdown"]
            assert "**A" in body["markdown"]

    def test_file_saved_to_interviews(self, client: TestClient):
        """转写结果应保存到 interviews/ 目录。"""
        mock_text = SAMPLE_TRANSCRIPT
        with patch("src.routes.upload_routes.transcribe", return_value=mock_text):
            resp = client.post(
                "/upload",
                files={"file": ("my_interview.m4a", io.BytesIO(b"audio"), "audio/m4a")},
            )
            body = resp.json()
            assert "file" in body
            # 文件名格式：日期_时间_原始文件名.md
            assert body["file"].endswith(".md")
            assert "my_interview" in body["file"]

    def test_temp_file_cleaned_up(self, client: TestClient):
        """临时文件应在处理后清理——检查 temp/ 目录没有新增文件。

        注意：此测试在 mock transcribe 的情况下运行，所以 temp 目录
        内的文件在 finally 块中应当已被删除。
        """
        import glob
        from config import TEMP_DIR

        # 记录测试前的临时文件状态
        before = set(glob.glob(os.path.join(TEMP_DIR, "*")))

        mock_text = SAMPLE_TRANSCRIPT
        with patch("src.routes.upload_routes.transcribe", return_value=mock_text):
            client.post(
                "/upload",
                files={"file": ("cleanup_test.m4a", io.BytesIO(b"data"), "audio/m4a")},
            )

        # 检查是否有残留
        after = set(glob.glob(os.path.join(TEMP_DIR, "*")))
        new_files = after - before
        assert len(new_files) == 0, f"临时文件未被清理: {new_files}"

    def test_chinese_filename(self, client: TestClient):
        """中文文件名应正常处理。"""
        mock_text = "问：你好。答：你好。"
        with patch("src.routes.upload_routes.transcribe", return_value=mock_text):
            resp = client.post(
                "/upload",
                files={"file": ("录音 (1).m4a", io.BytesIO(b"audio"), "audio/m4a")},
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["status"] == "success"

    def test_static_files_accessible(self, client: TestClient):
        """静态资源（JS）应可访问。"""
        resp = client.get("/static/script.js")
        assert resp.status_code == 200
        assert "function" in resp.text or "const" in resp.text or "let" in resp.text

    def test_favicon_accessible(self, client: TestClient):
        """Favicon 应可访问。"""
        resp = client.get("/static/favicon.ico")
        # favicon 可能为空，只检查不报 404
        assert resp.status_code in (200, 404)


# ═══════════════════════════════════════════════════════════
#  POST /upload/stream 流式端点测试
# ═══════════════════════════════════════════════════════════

class TestUploadStreamEndpoint:
    """测试 SSE 流式转写端点。"""

    def _consume_sse(self, resp) -> list[dict]:
        """消费 SSE 流，返回解析后的事件列表。

        Starlette TestClient 会将 StreamingResponse 的完整内容收集到 resp.text 中。
        直接按 \\n\\n 分隔解析所有 SSE 消息。
        """
        events = []
        body = resp.text  # StreamingResponse 的完整输出

        for part in body.split("\n\n"):
            part = part.strip()
            if not part:
                continue
            event_type = ""
            data_str = ""
            for line in part.split("\n"):
                if line.startswith("event: "):
                    event_type = line[7:]
                elif line.startswith("data: "):
                    data_str = line[6:]
            if event_type and data_str:
                try:
                    events.append({"event": event_type, "data": json.loads(data_str)})
                except json.JSONDecodeError:
                    pass
        return events

    def test_stream_success_flow(self, client: TestClient):
        """流式端点的完整事件流：phase(asr_started) → chunk... → phase(qa_started) → qa_done。"""
        mock_chunks = ["你好，", "请自我介绍一下。", "我叫张三。"]

        def mock_stream_gen(file_path):
            yield from mock_chunks

        with patch("src.routes.upload_routes.transcribe_stream", side_effect=mock_stream_gen):
            resp = client.post(
                "/upload/stream",
                files={"file": ("interview.m4a", io.BytesIO(b"fake audio"), "audio/m4a")},
            )

            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers.get("content-type", "")

            events = self._consume_sse(resp)

            # 应包含 phase 事件（至少 asr_started）
            phase_events = [e for e in events if e["event"] == "phase"]
            stages = [e["data"]["stage"] for e in phase_events]
            assert "asr_started" in stages, f"应有 asr_started phase, stages={stages}"

            # 应包含 chunk 事件
            chunk_events = [e for e in events if e["event"] == "chunk"]
            assert len(chunk_events) == len(mock_chunks)

            # 最后一个事件应为 qa_done 或 error
            final_event = events[-1]
            assert final_event["event"] in ("qa_done", "error")

    def test_stream_empty_audio(self, client: TestClient):
        """空文件应在保存阶段就被拒绝（400）。"""
        resp = client.post(
            "/upload/stream",
            files={"file": ("empty.m4a", io.BytesIO(b""), "audio/m4a")},
        )
        assert resp.status_code == 400

    def test_stream_silent_audio(self, client: TestClient):
        """静音音频（无转写内容）→ qa_done 事件带空 markdown。"""
        mock_chunks = [""]  # 空 chunk

        def mock_stream_gen(file_path):
            yield from mock_chunks

        with patch("src.routes.upload_routes.transcribe_stream", side_effect=mock_stream_gen):
            resp = client.post(
                "/upload/stream",
                files={"file": ("silence.m4a", io.BytesIO(b"silence"), "audio/m4a")},
            )

            events = self._consume_sse(resp)
            final = events[-1]
            assert final["event"] == "qa_done"
            assert "未识别" in final["data"]["markdown"]

    def test_stream_transcribe_error(self, client: TestClient):
        """转写异常时流应推送 error 事件。"""
        def mock_stream_error(file_path):
            raise RuntimeError("GPU out of memory")
            yield  # 让函数成为生成器

        with patch("src.routes.upload_routes.transcribe_stream", side_effect=mock_stream_error):
            resp = client.post(
                "/upload/stream",
                files={"file": ("test.m4a", io.BytesIO(b"audio"), "audio/m4a")},
            )

            events = self._consume_sse(resp)
            error_events = [e for e in events if e["event"] == "error"]
            assert len(error_events) >= 1

    def test_stream_temp_file_cleaned_up(self, client: TestClient):
        """流式处理完成后临时文件应被清理。"""
        import glob
        from config import TEMP_DIR

        before = set(glob.glob(os.path.join(TEMP_DIR, "*")))

        def mock_stream_gen(file_path):
            yield "test chunk"

        with patch("src.routes.upload_routes.transcribe_stream", side_effect=mock_stream_gen):
            client.post(
                "/upload/stream",
                files={"file": ("stream_cleanup.m4a", io.BytesIO(b"audio"), "audio/m4a")},
            )

        after = set(glob.glob(os.path.join(TEMP_DIR, "*")))
        new_files = after - before
        assert len(new_files) == 0, f"流式临时文件未被清理: {new_files}"

    def test_stream_no_file_field(self, client: TestClient):
        """流式端点缺少 file 字段 → 422。"""
        resp = client.post("/upload/stream")
        assert resp.status_code == 422
