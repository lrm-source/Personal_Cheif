from fastapi import APIRouter, Query
from app.models.schemas import ChatRequest
from app.agents.init_model import search_recipes, get_messages, clear_messages
from fastapi.responses import StreamingResponse

router = APIRouter()


@router.post("/chat/stream")
async def chat_endpoint(request: ChatRequest):
    """流式对话"""
    return StreamingResponse(
        search_recipes(request.message, request.image_url, request.thread_id),
        media_type="text/plain; charset=utf-8"
    )


@router.get("/chat/messages")
async def get_chat_messages(thread_id: str = Query(...)):
    """获取历史消息"""
    messages = get_messages(thread_id)
    return {"messages": messages}


@router.delete("/chat/messages")
async def clear_chat_messages(thread_id: str = Query(...)):
    """清空历史消息"""
    clear_messages(thread_id)
    return {"success": True}