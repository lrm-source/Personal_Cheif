from typing import Optional, List

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str
    image_url: Optional[str] = None
    thread_id: str


class RecipeItem(BaseModel):
    """单道食谱"""
    name: str = Field(description="食谱名称")
    nutrition_score: int = Field(ge=1, le=10, description="营养评分 1-10")
    difficulty_score: int = Field(ge=1, le=10, description="制作难度 1-10（越高越简单）")
    total_score: int = Field(ge=1, le=10, description="综合评分 1-10")
    ingredients: List[str] = Field(description="所用食材列表")
    reason: str = Field(description="推荐理由")
    image_url: Optional[str] = Field(default=None, description="参考图片URL")


class RecipeResponse(BaseModel):
    """Agent 返回的结构化食谱列表"""
    recipes: List[RecipeItem] = Field(description="排序后的食谱列表")
    summary: Optional[str] = Field(default=None, description="总体建议摘要")