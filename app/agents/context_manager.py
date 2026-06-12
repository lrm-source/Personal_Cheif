"""
上下文压缩管理：当对话轮次超过阈值时，将旧消息压缩为摘要，
释放 token 预算的同时保留历史上下文。
"""
from langchain.messages import HumanMessage, AIMessage
from app.common.logger import logger

# 阈值：超过 6 轮对话（12条消息）触发压缩
MAX_MESSAGES = 12
# 压缩后保留最近 2 轮（4条消息），其余压缩为摘要
KEEP_RECENT = 4

SUMMARIZE_PROMPT = """请用一段简短中文（200字以内）总结以下对话的关键信息：

1. 用户提到了哪些食材、偏好或忌口？
2. 之前推荐过哪些食谱，用户的反馈如何？
3. 用户有哪些明确的偏好（如：喜欢辣、不喜欢油炸等）？

对话内容：
{messages}

请直接输出摘要，不要加任何前缀或标记。"""


async def generate_summary(model, messages: list) -> str:
    """调用模型对历史消息做摘要"""
    # 过滤工具调用等非对话消息，只保留 Human/AI 文本
    dialog_lines = []
    for m in messages:
        if isinstance(m, HumanMessage):
            content = m.content
            if isinstance(content, str):
                dialog_lines.append(f"用户：{content}")
        elif isinstance(m, AIMessage):
            if m.content and isinstance(m.content, str):
                dialog_lines.append(f"助手：{m.content}")

    if not dialog_lines:
        return ""

    dialog_text = "\n".join(dialog_lines)
    prompt = SUMMARIZE_PROMPT.format(messages=dialog_text)

    try:
        response = await model.ainvoke([HumanMessage(content=prompt)])
        summary = response.content.strip()
        logger.info(f"[上下文压缩] 摘要完成，长度：{len(summary)} 字符")
        return summary
    except Exception as e:
        logger.error(f"[上下文压缩] 摘要失败：{e}")
        # 降级：返回最后几条原始消息的拼接
        return "最近讨论：" + dialog_lines[-3:]


async def maybe_compress(model, checkpointer, thread_id: str) -> str | None:
    """
    检查当前会话是否超过阈值，超过则压缩历史。

    返回：
        None  — 不需要压缩
        str   — 压缩后的摘要字符串（调用方应清空线程并重新注入）
    """
    checkpoint = checkpointer.get({"configurable": {"thread_id": thread_id}})
    if not checkpoint:
        return None

    channel_values = checkpoint.get("channel_values")
    if not channel_values:
        return None

    messages = list(channel_values.get("messages", []))
    if len(messages) <= MAX_MESSAGES:
        return None

    logger.info(
        f"[上下文压缩] 触发 — 消息数 {len(messages)} > {MAX_MESSAGES}"
    )

    # 旧消息（压缩）、最近消息（保留）
    old_messages = messages[:-KEEP_RECENT]

    summary = await generate_summary(model, old_messages)

    # 清空线程，摘要将由调用方重新注入
    checkpointer.delete_thread(thread_id)
    logger.info("[上下文压缩] 线程已重置，旧消息已清除")

    return summary
