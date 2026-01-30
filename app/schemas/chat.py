from enum import Enum
from uuid import UUID

from pydantic import BaseModel, Field


class ChatRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatMessage(BaseModel):
    """チャット履歴の1メッセージ"""
    role: ChatRole
    content: str


class ChatRequest(BaseModel):
    """RAGチャットリクエストスキーマ"""
    question: str = Field(..., min_length=1, description="質問テキスト")
    chat_history: list[ChatMessage] | None = Field(
        None,
        description="過去の会話履歴",
    )
    top_k: int = Field(5, ge=1, le=20, description="参照するチャンク数")
    score_threshold: float = Field(0.0, ge=0.0, le=1.0, description="スコア閾値")


class ChatSource(BaseModel):
    """回答の参照元情報"""
    document_id: UUID
    chunk_id: UUID
    filename: str | None
    title: str | None
    chunk_content: str
    page_number: int | None
    score: float


class ChatResponse(BaseModel):
    """RAGチャットレスポンススキーマ"""
    answer: str
    sources: list[ChatSource]
    question: str


class ChatStreamChunk(BaseModel):
    """ストリーミングレスポンスの1チャンク"""
    content: str
    done: bool = False
    sources: list[ChatSource] | None = None
