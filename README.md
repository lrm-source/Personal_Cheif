# Personal Chief 🧑‍🍳

> AI 私人厨师 — 拍一张食材照片，秒获专属食谱推荐

## 功能

- 📷 **食材识别** — 上传食材照片，AI 自动识别并评估新鲜度
- 🍽️ **食谱推荐** — 基于可用食材智能搜索，按营养+难度综合评分排序
- 🗣️ **多轮对话** — 支持连续追问，偏好记忆、忌口记录
- 📦 **上下文压缩** — 长对话自动摘要，节省 Token 同时保持连贯
- 🛡️ **输入护栏** — 自动拦截非饮食相关请求，只服务"吃"这件事

## 技术栈

| 层级 | 技术 |
|------|------|
| 框架 | FastAPI + LangGraph |
| 大模型 | Qwen 3.5 Plus (DashScope / OpenAI 兼容接口) |
| 搜索 | Tavily Search API |
| 存储 | Alibaba Cloud OSS（图片上传）+ SQLite（对话记忆） |
| 前端 | Next.js 静态站点（SPA） |

## 项目结构

```
personal_cheif/
├── app/
│   ├── agents/          # LangGraph Agent — 食谱推荐核心
│   │   ├── init_model.py      # Agent 定义、流式对话
│   │   ├── context_manager.py # 上下文压缩器
│   │   ├── guard.py           # 输入护栏
│   │   └── config.py          # 模型配置
│   ├── api/v1/          # FastAPI 路由
│   │   ├── chat.py            # 对话接口（流式/历史/清除）
│   │   └── oss.py             # OSS 上传预签名
│   ├── models/          # Pydantic 数据模型
│   ├── common/          # 日志等公共模块
│   ├── static/          # Next.js 构建产物（前端）
│   └── main.py          # 应用入口
├── resources/           # 本地数据库文件
├── langgraph.json       # LangGraph 配置
├── pyproject.toml       # 项目依赖
└── .python-version      # Python 3.13+
```

## 快速开始

### 1. 环境准备

```bash
# Python 3.13+
# 安装 uv（推荐）
pip install uv

# 安装依赖
uv sync
```

### 2. 配置环境变量

在 `.env` 中填入你的 API Key：

```env
DASHSCOPE_API_KEY=你的阿里云百炼API_KEY
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
TAVILY_API_KEY=你的Tavily搜索API_KEY
OSS_ACCESS_KEY_ID=你的阿里云OSS_AK
OSS_ACCESS_KEY_SECRET=你的阿里云OSS_SK
OSS_BUCKET=你的OSS_Bucket名
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=你的LangSmith_KEY（可选）
```

### 3. 启动

```bash
uv run python -m app.main
# 访问 http://127.0.0.1:8001
```

## API

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/chat/stream` | 流式对话（支持文本+图片） |
| GET  | `/api/v1/chat/messages?thread_id=xxx` | 获取对话历史 |
| DELETE | `/api/v1/chat/messages?thread_id=xxx` | 清除对话 |
| GET  | `/api/v1/oss/presign?filename=xxx` | 获取 OSS 上传预签名 URL |
