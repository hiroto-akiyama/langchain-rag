from uuid import UUID

from pydantic import BaseModel, Field


class SearchFilter(BaseModel):
    """検索フィルタ"""
    tags: list[str] | None = Field(None, description="タグでフィルタ")
    document_ids: list[UUID] | None = Field(None, description="ドキュメントIDでフィルタ")


class SearchRequest(BaseModel):
    """検索リクエストスキーマ"""
    query: str = Field(..., min_length=1, description="検索クエリテキスト")
    top_k: int = Field(5, ge=1, le=100, description="取得件数")
    score_threshold: float = Field(0.0, ge=0.0, le=1.0, description="スコア閾値")
    filter: SearchFilter | None = Field(None, description="検索フィルタ")


class SearchResultMetadata(BaseModel):
    """検索結果メタデータ"""
    filename: str | None = None
    title: str | None = None
    page_number: int | None = None
    tags: list[str] | None = None


class SearchResult(BaseModel):
    """検索結果の1件"""
    document_id: UUID
    chunk_id: UUID
    content: str
    score: float
    metadata: SearchResultMetadata


class SearchResponse(BaseModel):
    """検索レスポンススキーマ"""
    results: list[SearchResult]
    total: int
    query: str
