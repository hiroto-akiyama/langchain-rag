"""RAG質問応答API"""
import json
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.schemas.chat import ChatRequest, ChatResponse, ChatStreamChunk
from app.services.embedding_service import EmbeddingService
from app.services.rag_service import RAGService
from app.services.search_service import SearchService

router = APIRouter(prefix="/chat", tags=["chat"])


def get_rag_service() -> RAGService:
    """RAGServiceの依存性注入"""
    embedding_service = EmbeddingService()
    search_service = SearchService(embedding_service=embedding_service)
    return RAGService(search_service=search_service)


@router.post(
    "",
    response_model=ChatResponse,
    summary="RAG質問応答",
    description="RAGによる質問応答を行います",
)
async def chat(
    request: ChatRequest,
    service: Annotated[RAGService, Depends(get_rag_service)],
) -> ChatResponse:
    """RAGによる質問応答"""
    return await service.chat(request)


@router.post(
    "/stream",
    summary="ストリーミング質問応答",
    description="ストリーミング形式でRAG質問応答を行います（Server-Sent Events）",
)
async def chat_stream(
    request: ChatRequest,
    service: Annotated[RAGService, Depends(get_rag_service)],
) -> StreamingResponse:
    """ストリーミング形式でRAG質問応答"""

    async def generate():
        async for chunk in service.chat_stream(request):
            data = chunk.model_dump_json()
            yield f"data: {data}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
