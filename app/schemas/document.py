from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class DocumentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


class DocumentCreate(BaseModel):
    """ドキュメント作成時のスキーマ（ファイルアップロード時）"""
    title: str | None = Field(None, max_length=500, description="ドキュメントタイトル")
    tags: list[str] | None = Field(None, description="タグリスト")


class DocumentResponse(BaseModel):
    """ドキュメントレスポンススキーマ"""
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    filename: str
    title: str | None
    file_path: str | None
    file_size: int | None
    mime_type: str | None
    tags: list[str] | None
    status: DocumentStatus
    chunk_count: int
    created_at: datetime
    updated_at: datetime


class DocumentListResponse(BaseModel):
    """ドキュメント一覧レスポンススキーマ"""
    items: list[DocumentResponse]
    total: int
    page: int
    per_page: int


class ChunkResponse(BaseModel):
    """チャンクレスポンススキーマ"""
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    document_id: UUID
    content: str
    chunk_index: int
    page_number: int | None
    chunk_metadata: dict | None
    created_at: datetime


class DocumentDetailResponse(DocumentResponse):
    """ドキュメント詳細レスポンススキーマ（チャンク情報含む）"""
    chunks: list[ChunkResponse] = Field(default_factory=list)
