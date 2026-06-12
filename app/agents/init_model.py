import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from dotenv import load_dotenv
load_dotenv(override=True)
from langchain.chat_models import init_chat_model
import os
from langchain.agents import create_agent
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain.messages import HumanMessage, AIMessage, AIMessageChunk
from app.common.logger import logger

raw_model = init_chat_model(
    model = "qwen3.5-plus",
    model_provider = "openai",
    base_url = os.getenv("DASHSCOPE_BASE_URL"),
    api_key = os.getenv("DASHSCOPE_API_KEY"),
)

# 绑定 JSON 模式：模型强制输出合法 JSON
model = raw_model.bind(response_format={"type": "json_object"})

from langchain_tavily import TavilySearch
web_search = TavilySearch(
    max_results = 5,
    topic = "general",
)

system_prompt = """
你是一名私人厨师，只处理与食材识别、食谱推荐、烹饪建议、饮食咨询相关的问题。

**边界限制（最高优先级）**：
如果用户输入与食物、食材、烹饪、饮食完全无关（如：风景照、人物照、动物照、电影推荐、科技问题等），你必须直接回复以下 JSON：
{{"recipes": [], "summary": "请上传食材照片或告诉我你手头有哪些食材，我帮你推荐食谱~"}}
不得对非食物输入发挥想象力编造食谱。

只有确认输入与食物相关时，才执行以下流程：

操作流程：

1. 识别和评估食材：若用户提供照片，首先辨识所有可见食材。基于食材的外观状态，评估其新鲜度与可用量，整理出一份"当前可用食材清单"。

2. 智能食谱检索：优先调用 web_search 工具，以"可用食材清单"为核心关键词，查找可行食谱。

3. 多维度评估与排序：从营养价值和制作难度两个维度对候选食谱量化打分，制作简单且营养丰富的排名前列。

4. 结构化输出：你必须只输出一个 JSON 对象，不要加任何 markdown 标记、解释文字或代码块。JSON 结构如下：

{{
  "recipes": [
    {{
      "name": "食谱名称",
      "nutrition_score": 8,
      "difficulty_score": 7,
      "total_score": 8,
      "ingredients": ["食材1", "食材2"],
      "reason": "推荐理由",
      "image_url": "参考图片URL（如有）"
    }}
  ],
  "summary": "一句话总结推荐理由"
}}

评分规则：
- nutrition_score (1-10)：营养均衡度，越高越好
- difficulty_score (1-10)：制作容易度，越高越简单
- total_score (1-10)：综合评分，(nutrition_score + difficulty_score) / 2，四舍五入
- 输出 3-5 道候选食谱，按 total_score 降序排列
- image_url 如搜索不到可为 null

请优先使用 web_search 工具，实在搜索不到就自行发挥。
"""

import sqlite3
conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)

agent = create_agent(
    model = model,
    tools = [web_search],
    system_prompt = system_prompt,
    checkpointer = checkpointer,
)

# 流式对话
async def search_recipes(prompt: str, image: str, thread_id: str):
    """调用agent搜索食谱，流式输出 JSON 片段给前端做格式化"""
    logger.info(f"[用户]: {prompt}, image: {image}, thread_id: {thread_id}")
    try:
        # --- 上下文压缩 ---
        from app.agents.context_manager import maybe_compress
        summary = await maybe_compress(raw_model, checkpointer, thread_id)
        if summary:
            prompt = f"[历史摘要]\n{summary}\n---\n[当前]\n{prompt}"

        # --- 输入护栏：拦截非私厨领域的请求 ---
        from app.agents.guard import check_guard
        if not await check_guard(raw_model, prompt, image):
            import json
            yield json.dumps({"recipes": [], "summary": "请上传食材照片或告诉我你手头有哪些食材，我帮你推荐食谱~"}, ensure_ascii=False)
            return

        # 判断是否有图片，封装不同格式的消息
        if not image or image.strip() == "":
            message = HumanMessage(content=prompt)
        else:
            message = HumanMessage(content=[
                {"type": "image", "url": image},
                {"type": "text", "text": prompt}
            ])

        # 流式透传 JSON 片段，前端负责累积后格式化
        for chunk, metadata in agent.stream(
            {"messages": [message]},
            {"configurable": {"thread_id": thread_id}},
            stream_mode="messages"
        ):
            if isinstance(chunk, AIMessageChunk) and chunk.content:
                yield chunk.content

    except Exception as e:
        logger.error(f"\n[错误]: {str(e)}")
        import json
        yield json.dumps({"recipes": [], "summary": "信息检索失败，试试看手动输入食物列表？"}, ensure_ascii=False)

# 清空会话
def clear_messages(thread_id: str):
    """清空会话"""
    logger.info(f"清空历史消息，thread_id: {thread_id}")
    checkpointer.delete_thread(thread_id)

# 查询会话历史
def get_messages(thread_id: str) -> list[dict[str, str]]:
    """获取会话历史"""
    logger.info(f"获取历史消息，thread_id: {thread_id}")

    # 根据 thread_id 查询 checkpoint
    checkpoint = checkpointer.get({"configurable": {"thread_id": thread_id}})

    # 如果不存在，返回空列表
    if not checkpoint:
        return []

    # 安全获取 messages
    channel_values = checkpoint.get("channel_values")
    if not channel_values:
        return []

    messages = channel_values.get("messages", [])
    if not messages:
        return []

    # 转换消息格式
    result = []
    for msg in messages:
        if not msg.content:
            continue

        if isinstance(msg, HumanMessage):
            result.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            result.append({"role": "assistant", "content": msg.content})

    return result

if __name__ == "__main__":
    from langchain.messages import HumanMessage
    multimodel_message = HumanMessage([
        {"type": "text", "text": "帮我看看能做什么。"},
        {"type": "image_url", "image_url":{"url": "https://aisearch.cdn.bcebos.com/pic_create/2026-04-10/10/74d52055e4947f8c.jpg"}},
    ])
    config = {"configurable": {"thread_id": "1"}}

    response1 = agent.invoke({"messages":[multimodel_message]}, config)
    for message in response1["messages"]:
        if message.type == "ai":
            message.pretty_print()

    response2 = agent.invoke({"messages":[HumanMessage(content="我比较喜欢C 能不能给我推荐一些类似的")]}, config)
    ai_msgs = [m for m in response2["messages"] if m.type == "ai"]
    if ai_msgs:
        ai_msgs[-1].pretty_print()
