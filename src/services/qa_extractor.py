"""
问答提取服务
═══════════════════════════════════════════════════════
将访谈录音的转写文本，自动切分为「问题 → 回答」结构化数据。

提取策略（按优先级下降）：
  1. LLM 提取 —— 调用 OpenAI 兼容 API，使用 prompt 指导模型解析 Q&A。
     准确度最高，能处理隐含问题、多轮对话等复杂场景。
  2. 启发式回退 —— 基于中文问句特征的正则匹配，无需联网/API key。
     适用于简单的一问一答式访谈。

使用方式：
    from src.services.qa_extractor import extract_qa_pairs, format_qa_markdown

    pairs = extract_qa_pairs("你好，请自我介绍一下。我是张三...")
    md = format_qa_markdown(pairs)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

import httpx

from config import QA_LLM_API_KEY, QA_LLM_BASE_URL, QA_LLM_MODEL

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════
#  LLM System Prompt —— 控制模型输出格式
# ═══════════════════════════════════════════════════════════
# 通过详细的 system prompt 指导 LLM 如何切分访谈内容。
# 核心约束：
#   - JSON 数组输出，包含 question / answer 字段
#   - 不缩写或改写原始回答
#   - 缺少明确问题时允许合理推断
SYSTEM_PROMPT = """你是一个专业的访谈记录整理助手。

你的任务：将一段访谈语音的转写文本，整理成结构化的「问题-回答」对。

规则：
1. 识别访谈者和受访者的对话，将每次提问和对应的回答配对。
2. 如果文本中只有回答没有明确的问题，根据回答内容推断合理的问题。
3. 输出格式为 JSON 数组，每个元素包含 "question" 和 "answer" 字段。
4. 问题以「问」开头、回答以「答」开头（但在 JSON 中只存纯文本）。
5. 保留回答的原始内容，不要缩写或改写。

示例输入：
"你好，请简单介绍一下你自己。我是张三，今年30岁，目前在一家科技公司做产品经理。那你为什么想换工作呢？主要是想寻找更大的发展空间。"

示例输出：
[
  {"question": "请简单介绍一下你自己。", "answer": "我是张三，今年30岁，目前在一家科技公司做产品经理。"},
  {"question": "那你为什么想换工作呢？", "answer": "主要是想寻找更大的发展空间。"}
]"""


# ═══════════════════════════════════════════════════════════
#  数据结构
# ═══════════════════════════════════════════════════════════

@dataclass
class QAPair:
    """一个问答对。

    question / answer 均为纯文本（不含 Markdown 标记），
    由外层的 format_qa_markdown() 负责格式化输出。
    """
    question: str
    answer: str


# ═══════════════════════════════════════════════════════════
#  公开接口
# ═══════════════════════════════════════════════════════════

def extract_qa_pairs(transcript: str) -> list[QAPair]:
    """从转写文本中提取 Q&A 对（主入口）。

    根据是否配置了 QA_LLM_API_KEY 自动选择策略：
      - 有 API key → 调用 LLM 提取（_llm_extract）
      - 无 API key 或 LLM 调用失败 → 回退到启发式提取（_heuristic_extract）

    Args:
        transcript: ASR 转写后的原始文本

    Returns:
        QAPair 列表。空文本或无效输入返回单元素列表。
    """
    if not transcript or not transcript.strip():
        return [QAPair(question="（无有效语音内容）", answer="")]

    if QA_LLM_API_KEY:
        try:
            return _llm_extract(transcript)
        except Exception as exc:
            # LLM 调用失败时只记录警告，不中断流程
            logger.warning("LLM extraction failed, falling back to heuristic: %s", exc)

    return _heuristic_extract(transcript)


# ═══════════════════════════════════════════════════════════
#  LLM 提取策略
# ═══════════════════════════════════════════════════════════

def _llm_extract(transcript: str) -> list[QAPair]:
    """调用 OpenAI 兼容 API，由 LLM 完成 Q&A 分割。

    流程：
      1. 构造 Chat Completion 请求（system prompt + user transcript）
      2. 使用 response_format={"type": "json_object"} 确保 LLM 输出合法 JSON
      3. 解析 JSON 响应，处理可能的嵌套/包裹格式
      4. 转为 QAPair 列表返回

    容错设计：
      - API 返回的 JSON 可能被包在 {"questions": [...]} 等结构中，
        自动探测并提取列表
      - 如果 LLM 返回纯文本 JSON 数组字符串，尝试正则提取
    """
    # 构造 API 端点 URL（去掉尾部斜杠防止双斜杠）
    url = f"{QA_LLM_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {QA_LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": QA_LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"请整理以下访谈转写文本：\n\n{transcript}"},
        ],
        "temperature": 0.1,  # 低温度以获得确定性的结构化输出
        "response_format": {"type": "json_object"},
    }

    # 设置 120 秒超时——LLM 推理可能较慢，尤其长文本
    with httpx.Client(timeout=httpx.Timeout(120)) as client:
        resp = client.post(url, headers=headers, json=payload)
        resp.raise_for_status()

    data = resp.json()
    raw = data["choices"][0]["message"]["content"]

    # ── 解析 LLM 响应 ──
    # LLM 可能返回：
    #   1. 直接的 JSON 数组  →  json.loads() 即可
    #   2. 包裹在对象中       →  提取第一个列表值
    #   3. 嵌入在 Markdown 中 →  正则提纯 JSON 数组
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # 尝试从非纯 JSON 文本中提取 JSON 数组片段
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
        else:
            raise  # 完全无法解析，向上抛出触发回退

    # 如果最外层是 dict（例如 {"pairs": [...]}），提取第一个 list 值
    if isinstance(parsed, dict):
        for val in parsed.values():
            if isinstance(val, list):
                parsed = val
                break
        else:
            # dict 中没有 list，说明 LLM 返回了无法处理的格式
            raise ValueError(f"LLM returned unexpected structure: {raw}")

    # 转为 QAPair 列表
    pairs = []
    for item in parsed:
        # 兼容 "question"/"answer" 和 "q"/"a" 两种 key 名
        q = item.get("question", item.get("q", ""))
        a = item.get("answer", item.get("a", ""))
        if q or a:  # 只要有一个非空就保留
            pairs.append(QAPair(question=q, answer=a))

    # 若 LLM 返回了空数组，退而返回原始文本作为单个回答
    return pairs or [QAPair(question="", answer=transcript)]


# ═══════════════════════════════════════════════════════════
#  启发式提取策略（离线可用，无需 API）
# ═══════════════════════════════════════════════════════════

# 中文访谈中常见的提问起始模式（正则列表）
# 设计思路：
#   - 覆盖疑问词（为什么/怎么/如何/是什么/有什么）
#   - 覆盖动作词（介绍/说说/谈谈/聊聊/描述/说明/评价）
#   - 覆盖情态词（觉得/认为/怎么看）
#   - 覆盖一般问句（能不能/是否可以/有没有/会不会）
#   - 覆盖句末语气词（吗/呢）
_QUESTION_MARKERS = [
    # 模式 1：第二人称 + 疑问/动作词 → "你为什么离职"、"你介绍一下自己"
    r"那?你.{0,6}?(?:为什么|怎么|如何|是什么|有什么|介绍|说说|谈谈|聊聊|讲[一一下]|描述|说明|评价|觉得|认为|怎么看)",
    # 模式 2：显式提问前缀 → "请问你的职业规划"、"能不能具体说明"
    r"(?:请问|我想问|想问一下|问一下|能不能|可以不可以|是否|有没有|会不会).{2,30}",
    # 模式 3：祈使式提问 → "请自我介绍一下"、"请说说你的看法"
    #     「请 + 动作词」在访谈语境中功能等同于一个问题
    r"请.{0,6}?(?:介绍|说说|谈谈|聊聊|讲[一一下]|描述|说明|评价|分享|列举|举例)",
    # 模式 4：以问号结尾的短句
    r".{0,10}[？?]$",
    # 模式 5：以「吗」「呢」结尾的中等长度句子
    r"^.{2,20}(?:吗|呢)[？?]?$",
    # 模式 6：「那...呢」模式 → "那你呢"
    r"^那.{2,15}呢[？?]?$",
]

# 问句尾部特征词（编译为正则，提高匹配速度）
# 如果句子以以下模式结尾，大概率是问句
_QUESTION_ENDERS = re.compile(
    r"(?:什么|怎么|如何|为什么|吗|呢|吧|能否|可否|是否|能不能|可以吗|行不行|对不对|是不是)[？?]?"
)

# 问答之间的过渡词模式
# 用于识别 "嗯，好的。那接下来..." 这种 Q→A 切换边界
_QA_SEPARATOR = re.compile(
    r"(?:(?:嗯|哦|好|好的|明白|了解|知道了|行|可以)[,，。]*)"
    r"(?=那|那么|接下来|下[一一个]|还有|另外|再|请|我|你|这|这[个些])"
)


def _heuristic_extract(transcript: str) -> list[QAPair]:
    """基于规则的中文访谈 Q&A 切分。

    核心思路：
      将转写文本按标点边界拆分为句子序列，然后遍历检测每个句子
      是否「像是一个问题」。交替地将问句和答句分组输出。

    切分逻辑：
      - 遍历句子列表
      - 遇到「像问句」的句子 → 开始一个新的 QA 对
      - 之后遇到「不像问句」的句子 → 追加到当前答案缓冲区
      - 再遇到「像问句」→ 之前的问答对完成，开始下一个

    边界情况处理：
      - 开头的非问句 → 视作开场白/自我介绍，作为第一个回答
      - 连续问句 → 视为追问，前面的 Q 如果有缓冲区则先输出
      - 尾部只有问句没有回答 → 标记为「回答为空或未识别」
    """
    text = transcript.strip()
    sentences = _split_sentences(text)

    # 只有一句话 → 无法分割，整体视为回答
    if len(sentences) <= 1:
        return [QAPair(question="（完整回答）", answer=text)]

    pairs: list[QAPair] = []
    buffer_q: list[str] = []   # 当前问题的句子缓冲区
    buffer_a: list[str] = []   # 当前回答的句子缓冲区
    expecting_answer = False    # 状态标记：是否在等待回答

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        is_question_like = _is_question(sent)

        if is_question_like and not expecting_answer:
            # ── 场景 A：遇到新问题，且不在等待回答状态 ──
            # 先清空已有的缓冲（如果有问题+回答则成对输出）
            if buffer_q and buffer_a:
                pairs.append(
                    QAPair(question="".join(buffer_q).strip(), answer="".join(buffer_a).strip())
                )
            elif buffer_a:
                # 有回答没有问题 → 用占位问题
                pairs.append(
                    QAPair(question="（前文承接）", answer="".join(buffer_a).strip())
                )
            # 开始记录新问题
            buffer_q = [sent]
            buffer_a = []
            expecting_answer = True

        elif is_question_like and expecting_answer:
            # ── 场景 B：已经在等回答，又遇到问句（追问/打断） ──
            # 先保存前一个问答对
            if buffer_a:
                pairs.append(
                    QAPair(question="".join(buffer_q).strip(), answer="".join(buffer_a).strip())
                )
            # 开始新问题
            buffer_q = [sent]
            buffer_a = []

        else:
            # ── 场景 C：非问句 ──
            if expecting_answer:
                # 当前在等待回答 → 追加到答案缓冲区
                buffer_a.append(sent)
            else:
                # 还没遇到第一个问题 → 这可能是开场白/自我介绍
                buffer_a.append(sent)

    # ── 处理末尾残留缓冲 ──
    if buffer_q:
        answer = "".join(buffer_a).strip()
        if not answer:
            answer = "（回答为空或未识别）"
        pairs.append(QAPair(question="".join(buffer_q).strip(), answer=answer))
    elif buffer_a:
        # 纯叙述，无明确问题 → 整体作为一个回答
        pairs.append(QAPair(question="（开场白 / 自我介绍）", answer="".join(buffer_a).strip()))

    return pairs


def _split_sentences(text: str) -> list[str]:
    """将中文文本按标点边界拆分为句子序列。

    拆分策略：
      1. 以 [。！？.!?\n] 为边界切分（保留标点在句末）
      2. 过短的碎片（< 8 字且不像问句）合并到前一句
         —— 这是为了避免 "嗯。" "好的。" 这类应答词被单独分离
    """
    # 正向后顾（lookbehind）：在标点符号之后切分，但保留标点
    parts = re.split(r"(?<=[。！？\.\!\?\n])", text)

    merged: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # 过短的碎片合并到前一句，避免破坏语义完整性
        if merged and len(part) < 8 and not _is_question(part):
            merged[-1] += part
        else:
            merged.append(part)
    return merged


def _is_question(text: str) -> bool:
    """判定一个句子是否「像是一个问题」。

    判断依据（任一满足即为 True）：
      1. 以疑问词结尾（什么/怎么/如何/为什么/吗/呢/吧/能否...）
      2. 匹配 _QUESTION_MARKERS 中的任一正则模式

    注意：
      这是一个启发式判定，存在误判可能：
      - False positive：「我觉得这样很好。」→ 不包含问题特征 ✓
      - False negative：反问句「难道不是吗」→ 结尾有「吗」✓
      - 对于长句中的嵌入疑问可能漏判，但中文访谈中问题通常
        是以独立句形式出现的，影响不大
    """
    if _QUESTION_ENDERS.search(text):
        return True
    for pattern in _QUESTION_MARKERS:
        if re.search(pattern, text):
            return True
    return False


# ═══════════════════════════════════════════════════════════
#  Markdown 格式化输出
# ═══════════════════════════════════════════════════════════

def format_qa_markdown(pairs: list[QAPair], title: str = "访谈记录") -> str:
    """将 QAPair 列表渲染为可读的 Markdown 文本。

    输出格式：
        # 访谈记录

        **Q1：** 请自我介绍一下。
        **A1：** 我叫张三，今年30岁...

        **Q2：** 你为什么想换工作？
        **A2：** 主要是想寻找更大的发展空间。

    Args:
        pairs: QAPair 列表
        title: 一级标题文字

    Returns:
        格式化的 Markdown 字符串（UTF-8，可直接写入 .md 文件）
    """
    lines = [f"# {title}\n"]
    for i, pair in enumerate(pairs, 1):
        # 空值保护：避免 None 或空字符串导致输出不完整
        q = pair.question or "（未识别的问题）"
        a = pair.answer or "（未识别的回答）"
        lines.append(f"**Q{i}：** {q}\n")
        lines.append(f"**A{i}：** {a}\n")
    return "\n".join(lines)
