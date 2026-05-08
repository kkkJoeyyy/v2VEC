# V2VEC — 访谈语音转结构化问答

将面试/访谈录音自动转换为结构化的「问题—回答」Markdown 文本。

**核心流程：** 上传音频 → 阿里云 DashScope qwen3-asr-flash API 转写 → Q&A 智能切分 → Markdown 输出

---

## 项目结构

```
v2VEC/
├── config.py                       # 全局配置（路径/模型参数/LLM）
├── requirements.txt                # Python 依赖
├── README.md
│
├── app/
│   ├── main.py                     # 兼容性入口 → 委托到 src.server
│   └── static/
│       ├── index.html              # 前端 SPA 页面
│       └── script.js               # 前端交互逻辑
│
├── src/
│   ├── server.py                   # FastAPI 应用工厂 + 入口
│   ├── routes/
│   │   └── upload_routes.py        # POST /upload 路由
│   └── services/
│       ├── transcriber.py          # ASR 转写服务（懒加载单例）
│       └── qa_extractor.py         # Q&A 提取服务（LLM + 启发式）
│
├── model/
│   └── model_utils.py              # 独立使用的模型工具（CLI / Notebook）
│
├── interviews/                     # 输出：转换后的 QA Markdown 文件
├── temp/                           # 临时上传文件（自动清理）
│
└── tests/
    ├── test_qa_extractor.py        # Q&A 提取单元测试（46 用例）
    └── test_routes.py              # API 路由集成测试（14 用例）
```

## 架构设计

| 层 | 文件 | 职责 |
|----|------|------|
| **配置** | `config.py` | 所有可调参数集中管理，支持环境变量覆盖 |
| **ASR 服务** | `src/services/transcriber.py` | 调用阿里云 DashScope qwen3-asr-flash API，提供 `transcribe()` 和 `transcribe_stream()` |
| **Q&A 提取** | `src/services/qa_extractor.py` | 将转写文本切分为 Q&A 对，支持 LLM / 启发式两种策略 |
| **路由层** | `src/routes/upload_routes.py` | HTTP 请求处理、文件管理、编排调用 |
| **应用入口** | `src/server.py` | FastAPI 工厂函数，挂载静态文件、注册路由 |

**设计原则：**
- **云端 ASR** —— 使用阿里云 DashScope API，无需 GPU / 本地模型
- **策略回退** —— LLM 提取失败自动降级到启发式规则，不影响可用性
- **临时文件清理** —— `try/finally` 确保上传的临时音频一定会被删除
- **并发安全** —— UUID 生成唯一临时文件名，避免冲突

## 环境准备

### 1. 系统要求

- Python 3.12+
- macOS / Linux / Windows
- **ffmpeg**（用于流式分段转写）
- **阿里云百炼 API Key**（[控制台获取](https://bailian.console.aliyun.com/)）

### 2. 安装依赖

```bash
# 系统依赖：ffmpeg（用于流式分段转写）
brew install ffmpeg            # macOS
# sudo apt install ffmpeg      # Ubuntu / Debian

# Python 依赖
cd v2VEC
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 3. 配置环境变量

在[阿里云百炼控制台](https://bailian.console.aliyun.com/)获取 API Key，然后配置 `.env` 文件：

```bash
# 复制配置模板
cp .env.example .env

# 编辑 .env，填入实际值
# DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

`.env` 中可配置的变量（`.env.example` 有完整说明）：

| 变量 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `DASHSCOPE_API_KEY` | **是** | — | 阿里云百炼 API Key |
| `ASR_MODEL` | 否 | `qwen3-asr-flash` | ASR 模型名 |
| `ASR_LANGUAGE` | 否 | `Chinese` | 转写语言 |
| `QA_LLM_API_KEY` | 否 | — | Q&A 提取 LLM Key（不填用启发式） |
| `QA_LLM_BASE_URL` | 否 | `https://api.openai.com/v1` | LLM API 端点 |
| `PORT` | 否 | `8000` | 服务端口 |

## 启动项目

### 快速启动（四步）

```bash
# 1. 进入项目目录
cd v2VEC

# 2. 激活虚拟环境
source venv/bin/activate

# 3. 确认模型文件存在
ls Qwen3-ASR-1.7B/model.safetensors.index.json

# 4. 启动服务
uvicorn src.server:app --host 0.0.0.0 --port 8000
```

启动成功后会看到：
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

打开浏览器访问 **http://localhost:8000** ，拖拽或点击上传音频文件即可。

### 后台运行

```bash
# 后台启动（日志写入文件）
nohup uvicorn src.server:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &

# 查看日志
tail -f server.log

# 停止服务
pkill -f "uvicorn src.server:app"
```

### 自定义端口

```bash
# 方式 A：命令行参数
uvicorn src.server:app --host 0.0.0.0 --port 9000

# 方式 B：环境变量
PORT=9000 uvicorn src.server:app --host 0.0.0.0
```

### 首次启动注意事项

- **API Key**：确保 `DASHSCOPE_API_KEY` 已正确设置，否则服务会返回配置错误。
- **网络**：服务需要访问 `dashscope.aliyuncs.com`，请确保网络连通。
- **计费**：qwen3-asr-flash 按音频时长计费（~0.000035 元/秒），短音频成本极低。

## Q&A 提取策略

### 策略 1：LLM 提取（推荐）

配置 API key 以启用 GPT-4o-mini / 兼容 API 进行智能 Q&A 切分：

```bash
export QA_LLM_API_KEY="sk-xxxxxxxx"
export QA_LLM_BASE_URL="https://api.openai.com/v1"    # 可选，默认 OpenAI
export QA_LLM_MODEL="gpt-4o-mini"                      # 可选
```

**兼容任何 OpenAI 格式的 API：**
- Ollama 本地模型：`QA_LLM_BASE_URL=http://localhost:11434/v1`
- vLLM 自部署：`QA_LLM_BASE_URL=http://localhost:8000/v1`
- Azure OpenAI / 国内中转 API

### 策略 2：启发式提取（离线可用）

未配置 API key 时自动使用正则规则切分，覆盖中文访谈常见的提问模式：

- 疑问词：「为什么」「怎么」「如何」...
- 祈使式：「请介绍」「请说说」「请描述」...
- 句末语气词：「吗」「呢」
- 情态问句：「你觉得」「你怎么看」...

## API 接口

### POST /upload/stream（推荐）

**流式转写**：通过 SSE 实时推送转写文本。transformers backend 下采用分段转写策略（ffmpeg 将音频切为 15s 片段，逐段转写、逐段推送），前端每 5-10 秒可看到新文本。

**请求：** `multipart/form-data`，字段 `file`

**响应：** `text/event-stream`（SSE 协议），事件类型：

| 事件 | 触发时机 | 字段 |
|------|---------|------|
| `start` | 转写开始 | `message` |
| `chunk` | 每识别一个文本片段 | `text`（文本）, `accumulated_chars`（累计字数） |
| `qa_done` | Q&A 提取完成 | `markdown`, `qa_count`, `file` |
| `error` | 错误 | `message` |

**示例（原始 SSE 流）：**
```
event: start
data: {"message":"转写已开始"}

event: chunk
data: {"text":"你好，","accumulated_chars":3}

event: chunk
data: {"text":"请自我介绍一下。","accumulated_chars":11}

event: qa_done
data: {"markdown":"# 访谈记录\n\n**Q1：** ...","qa_count":2,"file":"2026-03-26_interview.md"}
```

### POST /upload

**同步转写**：完整转写后一次性返回 JSON 结果。适合短音频（< 1 分钟）。长音频推荐使用 `/upload/stream`。

**请求：** `multipart/form-data`，字段 `file`

```bash
curl -F "file=@/path/to/interview.m4a" http://localhost:8000/upload
```

**响应：**
```json
{
  "status": "success",
  "message": "转换完成！",
  "markdown": "# 访谈记录\n\n**Q1：** 请自我介绍一下。\n**A1：** 我叫张三...\n",
  "file": "2026-03-26_143022_interview.md",
  "qa_count": 5
}
```

### GET /

返回前端 SPA 页面。

### GET /static/*

静态资源（JS / CSS / 图标）。

## 运行测试

```bash
cd v2VEC
source venv/bin/activate

# 运行所有测试
python -m pytest tests/ -v

# 仅 Q&A 提取测试
python -m pytest tests/test_qa_extractor.py -v

# 仅 API 路由测试
python -m pytest tests/test_routes.py -v
```

测试覆盖：

| 模块 | 用例数 | 内容 |
|------|--------|------|
| `test_qa_extractor.py` | 46 | 问句判定、句子拆分、启发式提取、LLM mock、Markdown 格式化、策略选择 |
| `test_routes.py` | 14 | 文件上传、空文件拒绝、静音警告、异常处理、临时文件清理、中文文件名 |

## CLI 独立使用

不启动 Web 服务，直接命令行转写音频：

```bash
python model/model_utils.py /path/to/audio.m4a
```

输出保存到 `interviews/` 目录。

## 配置参考

所有可配置项见 [config.py](config.py)，支持环境变量覆盖：

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `QA_LLM_API_KEY` | （空） | LLM API Key，为空则使用启发式提取 |
| `QA_LLM_BASE_URL` | `https://api.openai.com/v1` | LLM API 端点 |
| `QA_LLM_MODEL` | `gpt-4o-mini` | LLM 模型名 |
| `HOST` | `0.0.0.0` | 服务监听地址 |
| `PORT` | `8000` | 服务监听端口 |

## License

MIT
