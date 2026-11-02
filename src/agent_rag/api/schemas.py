"""请求/响应模型。用 Pydantic 约束入参，别让脏数据进到检索和工具层。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    thread_id: str | None = Field(default=None, description="会话 id，留空则新建")


class ConfirmRequest(BaseModel):
    thread_id: str
    approved: bool


class ConfirmResponse(BaseModel):
    status: str
    reply: str


class DocumentInfo(BaseModel):
    source: str
    chunks: int
    indexed_at: str


class UploadResponse(BaseModel):
    source: str
    chunks: int
    message: str
