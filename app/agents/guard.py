"""
输入护栏：在请求进入 Agent 之前做前置分类，拦截非私厨领域的输入。
"""
from langchain.messages import HumanMessage, SystemMessage
from app.common.logger import logger

GUARD_PROMPT = """你是一个内容审核助手。判断用户输入是否与"食材识别、食谱推荐、烹饪建议、饮食咨询"相关。

请只回复一个JSON，不要加任何其他文字：
{"is_food_related": true, "reason": "简短理由"}

必须返回 false 的情况：
- 风景照、人物照、动物照、建筑物等非食材图片
- 与食物、烹饪、食谱、饮食完全无关的文本问题
- 询问其他领域知识（电影、音乐、科技等）

必须返回 true 的情况：
- 包含食材、菜品、厨房场景的照片
- 询问食谱、烹饪方法、营养搭配
- 询问"这些食材能做什么"类问题
- 饮食偏好、忌口等与吃相关的问题

注意：如果用户输入中包含食材名称或食物相关词汇，即使同时有其他内容，也应返回 true。"""

REJECTION_RESPONSE = {
    "recipes": [],
    "summary": "请上传食材照片或告诉我你手头有哪些食材，我帮你推荐食谱~"
}


async def check_guard(model, prompt: str, image_url: str | None) -> bool:
    """
    前置护栏检查：判断输入是否属于私厨领域。

    返回 True 表示通过，False 表示拦截。
    """
    # 构造审核消息
    if image_url:
        guard_message = HumanMessage(content=[
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": f"请判断：以下用户输入是否与食物/食材/烹饪/食谱相关？\n\n用户文本：{prompt}\n\n回复JSON："}
        ])
    else:
        guard_message = HumanMessage(content=f"请判断：以下用户输入是否与食物/食材/烹饪/食谱相关？\n\n用户文本：{prompt}\n\n回复JSON：")

    try:
        response = await model.ainvoke([
            SystemMessage(content=GUARD_PROMPT),
            guard_message
        ])
        raw = response.content.strip()

        # 容错解析：模型可能包裹 markdown 代码块
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        import json
        result = json.loads(raw)
        is_food = result.get("is_food_related", False)

        if is_food:
            logger.info(f"[护栏] 通过 — {result.get('reason', '')}")
        else:
            logger.info(f"[护栏] 拦截 — {result.get('reason', '')}")

        return is_food

    except Exception as e:
        logger.error(f"[护栏] 检查失败，默认放行：{e}")
        # 降级策略：解析失败时放行，由后续 Prompt 护栏兜底
        return True
