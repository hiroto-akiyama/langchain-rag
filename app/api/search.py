"""検索API"""
from typing import Annotated

from fastapi import APIRouter, Depends

from app.schemas.search import SearchRequest, SearchResponse
from app.services.embedding_service import EmbeddingService
from app.services.search_service import SearchService

router = APIRouter(prefix="/search", tags=["search"])


def get_search_service() -> SearchService:
    """SearchServiceの依存性注入"""
    embedding_service = EmbeddingService()
    return SearchService(embedding_service=embedding_service)


@router.post(
    "",
    response_model=SearchResponse,
    summary="類似ドキュメント検索",
    description="クエリテキストに類似するドキュメントチャンクを検索します",
)
async def search(
    request: SearchRequest,
    service: Annotated[SearchService, Depends(get_search_service)],
) -> SearchResponse:
    """類似ドキュメントを検索"""
    return await service.search(request)
