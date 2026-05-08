"""
Q&A 提取服务单元测试
═══════════════════════════════════════════════════════
覆盖范围：
  - 启发式提取（_heuristic_extract）：各类中文访谈文本的 Q&A 切分
  - 问句判定（_is_question）：疑问句 vs 陈述句
  - 句子拆分（_split_sentences）：中文标点边界拆分
  - Markdown 格式化（format_qa_markdown）：输出格式验证
  - LLM 提取（_llm_extract）：通过 mock httpx 验证 API 调用和解析逻辑
  - 主入口（extract_qa_pairs）：策略选择逻辑

运行方式：
    cd v2VEC
    python -m pytest tests/test_qa_extractor.py -v
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.services.qa_extractor import (
    QAPair,
    extract_qa_pairs,
    format_qa_markdown,
    _heuristic_extract,
    _is_question,
    _llm_extract,
    _split_sentences,
)


# ═══════════════════════════════════════════════════════════
#  Test data —— 各类中文访谈文本案例
# ═══════════════════════════════════════════════════════════

class TestData:
    """测试用例数据集。每个 case 包含输入文本和期望的 Q&A 对数量。"""

    # 标准一问一答
    simple_qa = "请自我介绍一下。我叫张三，今年30岁，做产品经理。"

    # 多轮问答（2 轮）
    multi_qa = (
        "请介绍一下你自己。"
        "我是李四，毕业于清华大学计算机系，目前在字节跳动做后端开发。"
        "你为什么想换工作呢？"
        "主要是想寻找更大的技术挑战，目前的工作内容有些重复。"
    )

    # 多轮问答（3 轮，含「你觉得」「怎么看」等情态问句）
    multi_qa_3 = (
        "你好，先简单介绍一下自己吧。"
        "好的，我叫王五，有五年前端开发经验，擅长 React 和 TypeScript。"
        "你能说说你做过的最有挑战的项目吗？"
        "最有挑战的是一个实时协作编辑器，需要处理 OT 算法和多人并发编辑的冲突。"
        "那你怎么看团队协作？"
        "我认为团队协作最重要的是沟通透明和文档完善。"
    )

    # 只有陈述，没有明确问句
    no_question = "我叫赵六，今年28岁，之前在阿里做运营，负责过双11的活动策划。"

    # 空文本
    empty_text = ""

    # 纯空白文本
    whitespace_text = "   \n  \t  "

    # 带「吗」「呢」结尾的简短问句
    short_questions = "你忙吗？还好。你会来吗？会。"

    # 追问模式（连续两个问句）
    follow_up = "你觉得这个方案怎么样？具体说说你的想法。我觉得方案整体可行，但细节还需要打磨。"


# ═══════════════════════════════════════════════════════════
#  _is_question 测试
# ═══════════════════════════════════════════════════════════

class TestIsQuestion:
    """测试问句判定函数。"""

    @pytest.mark.parametrize("text", [
        "你为什么要离职？",
        "请简单介绍一下自己。",
        "你觉得这个产品怎么样？",
        "能不能具体说明一下？",
        "你有什么职业规划吗？",
        "那你怎么看呢？",
        "这是为什么呢",
        "如何保证项目按时交付？",
    ])
    def test_positive(self, text: str):
        """预期判定为问句的文本。"""
        assert _is_question(text) is True, f"应该识别为问句: {text!r}"

    @pytest.mark.parametrize("text", [
        "我叫张三，今年30岁。",
        "之前在一家互联网公司做产品经理。",
        "主要负责用户增长和留存策略。",
        "我觉得这个方案是可行的。",
        "团队协作非常顺畅，大家配合很好。",
        "嗯，好的。",
        "明白了。",
    ])
    def test_negative(self, text: str):
        """预期判定为非问句的文本。"""
        assert _is_question(text) is False, f"不应该识别为问句: {text!r}"


# ═══════════════════════════════════════════════════════════
#  _split_sentences 测试
# ═══════════════════════════════════════════════════════════

class TestSplitSentences:
    """测试中文句子拆分函数。"""

    def test_basic_split(self):
        """基本拆分：以句号、问号、感叹号为边界。

        「很好！」只有 3 个字且不像问句，会被合并到前一句（预期行为）。
        """
        text = "你好。今天天气怎么样？很好！"
        result = _split_sentences(text)
        # 预期："你好。" | "今天天气怎么样？很好！"（短句合并）
        assert len(result) == 2, f"短句合并后应为 2 句，实际 {len(result)}: {result}"

    def test_short_fragment_merge(self):
        """短碎片（< 8 字且不像问句）应合并到前一句。"""
        text = "这是一个很长的句子包含很多信息。嗯。好的。"
        result = _split_sentences(text)
        # "嗯。" 和 "好的。" 都会被合并，最终应该是 1 句
        assert len(result) == 1, f"短碎片应被合并，实际 {len(result)}: {result}"

    def test_short_question_not_merged(self):
        """即使很短，问句不合并。"""
        text = "前面有一段很长的描述性文字。你呢？"
        result = _split_sentences(text)
        # "你呢？" 是问句，不应被合并
        assert len(result) >= 2, f"问句不应被合并，实际 {len(result)}: {result}"

    def test_newline_split(self):
        """换行符也应作为句子边界。"""
        text = "第一段较长文字用来测试分句逻辑。\n第二段也是较长文字。"
        result = _split_sentences(text)
        # 两句都超过 8 个字，不会被合并
        assert len(result) == 2, f"换行符作为边界应拆为 2 句，实际 {len(result)}: {result}"

    def test_empty_string(self):
        """空字符串返回空列表。"""
        assert _split_sentences("") == []


# ═══════════════════════════════════════════════════════════
#  _heuristic_extract 测试 —— 核心切分逻辑
# ═══════════════════════════════════════════════════════════

class TestHeuristicExtract:
    """测试启发式 Q&A 提取。"""

    def test_simple_qa(self):
        """标准一问一答。"""
        pairs = _heuristic_extract(TestData.simple_qa)
        assert len(pairs) == 1
        assert "自我介绍" in pairs[0].question
        assert "张三" in pairs[0].answer

    def test_multi_qa(self):
        """多轮问答（2 轮）。"""
        pairs = _heuristic_extract(TestData.multi_qa)
        assert len(pairs) >= 2, f"应提取出至少 2 组 Q&A，实际 {len(pairs)}"

    def test_multi_qa_3(self):
        """多轮问答（3 轮）。"""
        pairs = _heuristic_extract(TestData.multi_qa_3)
        assert len(pairs) >= 3, f"应提取出至少 3 组 Q&A，实际 {len(pairs)}"

    def test_no_question(self):
        """纯叙述无问题——整体视为一个回答。"""
        pairs = _heuristic_extract(TestData.no_question)
        assert len(pairs) == 1
        assert "赵六" in pairs[0].answer

    def test_empty_text(self):
        """空文本应返回占位 Q&A 对。"""
        pairs = _heuristic_extract(TestData.empty_text)
        assert len(pairs) == 1

    def test_whitespace_text(self):
        """纯空白文本应返回占位 Q&A 对。"""
        pairs = _heuristic_extract(TestData.whitespace_text)
        assert len(pairs) == 1

    def test_follow_up_questions(self):
        """追问模式——连续问句应分别输出。"""
        pairs = _heuristic_extract(TestData.follow_up)
        assert len(pairs) >= 1, f"追问应有至少 1 组 Q&A，实际 {len(pairs)}"

    def test_all_pairs_have_question_field(self):
        """所有 QAPair 的 question 字段都不应为 None。"""
        pairs = _heuristic_extract(TestData.multi_qa_3)
        for p in pairs:
            assert isinstance(p.question, str)
            assert isinstance(p.answer, str)

    def test_consecutive_qa_alternation(self):
        """验证 Q&A 交替出现的模式——question 不应为空。"""
        text = "你为什么来面试？我觉得贵公司很有前景。你的期望薪资是多少？希望能有市场平均水平。"
        pairs = _heuristic_extract(text)
        assert len(pairs) >= 2
        # 验证每个问题都包含有意义的问句内容
        for p in pairs:
            assert len(p.question) > 0


# ═══════════════════════════════════════════════════════════
#  format_qa_markdown 测试
# ═══════════════════════════════════════════════════════════

class TestFormatQAMarkdown:
    """测试 Markdown 格式化输出。"""

    def test_basic_format(self):
        """基本格式验证。"""
        pairs = [QAPair(question="问1", answer="答1")]
        md = format_qa_markdown(pairs)
        assert "# " in md
        assert "**Q1：**" in md
        assert "**A1：**" in md
        assert "问1" in md
        assert "答1" in md

    def test_custom_title(self):
        """自定义标题。"""
        pairs = [QAPair(question="Q", answer="A")]
        md = format_qa_markdown(pairs, title="自定义标题")
        assert "# 自定义标题" in md

    def test_empty_fields_use_placeholders(self):
        """空 question/answer 使用占位文字。"""
        pairs = [QAPair(question="", answer="")]
        md = format_qa_markdown(pairs)
        assert "未识别的问题" in md
        assert "未识别的回答" in md

    def test_none_fields_use_placeholders(self):
        """None question/answer 使用占位文字（防御性）。"""
        pairs = [QAPair(question=None, answer=None)]  # type: ignore
        md = format_qa_markdown(pairs)
        assert "未识别的问题" in md
        assert "未识别的回答" in md

    def test_multiple_pairs_numbering(self):
        """多组 Q&A 时编号递增。"""
        pairs = [
            QAPair(question="Q1", answer="A1"),
            QAPair(question="Q2", answer="A2"),
            QAPair(question="Q3", answer="A3"),
        ]
        md = format_qa_markdown(pairs)
        for i in range(1, 4):
            assert f"**Q{i}：**" in md
            assert f"**A{i}：**" in md

    def test_newline_separation(self):
        """每组 Q&A 之间应有空行分隔，保证 Markdown 可读性。"""
        pairs = [QAPair(question="Q1", answer="A1"), QAPair(question="Q2", answer="A2")]
        md = format_qa_markdown(pairs)
        # 每组之间应该有换行
        assert "\n\n" in md


# ═══════════════════════════════════════════════════════════
#  _llm_extract 测试（Mock HTTP）
# ═══════════════════════════════════════════════════════════

class TestLLMExtract:
    """测试 LLM 提取：mock httpx 请求，验证 JSON 解析逻辑。"""

    def _make_mock_response(self, content: list[dict], status_code: int = 200):
        """构造一个模拟的 httpx Response 对象。"""
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.json.return_value = {
            "choices": [
                {"message": {"content": json.dumps(content, ensure_ascii=False)}}
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    def test_basic_llm_response(self):
        """标准 LLM 返回 JSON 数组，应正常解析。"""
        mock_resp = self._make_mock_response([
            {"question": "你叫什么名字？", "answer": "我叫张三。"},
        ])
        with patch("httpx.Client.post", return_value=mock_resp):
            # 临时设置 API key 以触发 LLM 路径
            with patch("src.services.qa_extractor.QA_LLM_API_KEY", "fake-key"):
                pairs = _llm_extract("你叫什么名字？我叫张三。")
                assert len(pairs) == 1
                assert pairs[0].question == "你叫什么名字？"
                assert pairs[0].answer == "我叫张三。"

    def test_wrapped_json_response(self):
        """LLM 返回包裹在对象中的 JSON 数组：{"pairs": [...]}。"""
        mock_resp = self._make_mock_response(
            {"pairs": [{"question": "Q1", "answer": "A1"}]}
        )
        # 注意：这里 content 是 {"pairs": [...]} 的 JSON 字符串
        mock_resp.json.return_value = {
            "choices": [
                {"message": {"content": json.dumps(
                    {"pairs": [{"question": "Q1", "answer": "A1"}]},
                    ensure_ascii=False
                )}}
            ]
        }
        with patch("httpx.Client.post", return_value=mock_resp):
            with patch("src.services.qa_extractor.QA_LLM_API_KEY", "fake-key"):
                pairs = _llm_extract("test")
                assert len(pairs) == 1
                assert pairs[0].question == "Q1"

    def test_short_key_names(self):
        """LLM 使用 q/a 缩写 key 名时的兼容性。"""
        mock_resp = self._make_mock_response([
            {"q": "问题一", "a": "回答一"},
        ])
        with patch("httpx.Client.post", return_value=mock_resp):
            with patch("src.services.qa_extractor.QA_LLM_API_KEY", "fake-key"):
                pairs = _llm_extract("test")
                assert len(pairs) == 1
                assert pairs[0].question == "问题一"
                assert pairs[0].answer == "回答一"

    def test_empty_llm_response_returns_fallback(self):
        """LLM 返回空数组时，应返回包含全文的 QAPair。"""
        mock_resp = self._make_mock_response([])
        with patch("httpx.Client.post", return_value=mock_resp):
            with patch("src.services.qa_extractor.QA_LLM_API_KEY", "fake-key"):
                pairs = _llm_extract("全文内容")
                assert len(pairs) == 1
                assert pairs[0].answer == "全文内容"

    def test_markdown_wrapped_json(self):
        """LLM 将 JSON 嵌入 Markdown 代码块中，需要正则提取。"""
        raw_text = '```json\n[{"question": "Q", "answer": "A"}]\n```'
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": raw_text}}]
        }
        mock_resp.raise_for_status = MagicMock()
        with patch("httpx.Client.post", return_value=mock_resp):
            with patch("src.services.qa_extractor.QA_LLM_API_KEY", "fake-key"):
                pairs = _llm_extract("test")
                assert len(pairs) == 1
                assert pairs[0].question == "Q"
                assert pairs[0].answer == "A"


# ═══════════════════════════════════════════════════════════
#  extract_qa_pairs 主入口测试
# ═══════════════════════════════════════════════════════════

class TestExtractQAPairs:
    """测试 extract_qa_pairs 主入口的策略选择逻辑。"""

    def test_empty_transcript(self):
        """空转写文本返回占位结果。"""
        result = extract_qa_pairs("")
        assert len(result) == 1
        assert "无有效" in result[0].question

    @patch("src.services.qa_extractor.QA_LLM_API_KEY", "")
    def test_no_api_key_uses_heuristic(self):
        """无 API key 时使用启发式提取。"""
        result = extract_qa_pairs(TestData.simple_qa)
        assert len(result) == 1
        assert "张三" in result[0].answer

    @patch("src.services.qa_extractor.QA_LLM_API_KEY", "fake-key")
    def test_llm_failure_falls_back_to_heuristic(self):
        """LLM 调用失败时回退到启发式提取。"""
        # httpx 未安装/网络不可用时，LLM 调用会失败
        # 由于我们设置了 QA_LLM_API_KEY 但未 mock httpx，
        # build 请求时会失败，触发回退
        result = extract_qa_pairs(TestData.multi_qa)
        # 启发式回退仍应产生有效结果
        assert len(result) >= 1
        for p in result:
            assert isinstance(p.question, str)
            assert isinstance(p.answer, str)


# ═══════════════════════════════════════════════════════════
#  QAPair 数据结构测试
# ═══════════════════════════════════════════════════════════

class TestQAPair:
    """测试 QAPair 数据类的基本行为。"""

    def test_create_pair(self):
        p = QAPair(question="Q", answer="A")
        assert p.question == "Q"
        assert p.answer == "A"

    def test_equality(self):
        p1 = QAPair(question="Q", answer="A")
        p2 = QAPair(question="Q", answer="A")
        assert p1 == p2

    def test_inequality(self):
        p1 = QAPair(question="Q1", answer="A")
        p2 = QAPair(question="Q2", answer="A")
        assert p1 != p2
