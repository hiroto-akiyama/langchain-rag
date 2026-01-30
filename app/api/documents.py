"""ドキュメント管理API"""
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy.orm import Session

from app.models.database import get_db
from app.schemas.document import (
    DocumentDetailResponse,
    DocumentListResponse,
    DocumentResponse,
)
from app.services.document_service import DocumentService
from app.services.embedding_service import EmbeddingService
from app.services.search_service import SearchService

router = APIRouter(prefix="/documents", tags=["documents"])


def get_document_service(
    db: Annotated[Session, Depends(get_db)],
) -> DocumentService:
    """DocumentServiceの依存性注入"""
    embedding_service = EmbeddingService()
    search_service = SearchService(embedding_service=embedding_service)
    return DocumentService(
        db=db,
        embedding_service=embedding_service,
        search_service=search_service,
    )


@router.post(
    "/upload",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="ドキュメントアップロード",
    description="ドキュメントをアップロードし、インデックスを作成します",
)
async def upload_document(
    file: Annotated[UploadFile, File(description="アップロードするファイル")],
    service: Annotated[DocumentService, Depends(get_document_service)],
    title: Annotated[str | None, Form(description="ドキュメントタイトル")] = None,
    tags: Annotated[list[str] | None, Form(description="タグリスト")] = None,
) -> DocumentResponse:
    """ドキュメントをアップロードしてインデックスを作成"""
    try:
        document = await service.upload_document(
            file=file,
            title=title,
            tags=tags,
        )
        return DocumentResponse.model_validate(document)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="ドキュメント一覧取得",
    description="ドキュメントの一覧を取得します",
)
def list_documents(
    service: Annotated[DocumentService, Depends(get_document_service)],
    page: Annotated[int, Query(ge=1, description="ページ番号")] = 1,
    per_page: Annotated[int, Query(ge=1, le=100, description="1ページあたりの件数")] = 20,
    tag: Annotated[str | None, Query(description="タグでフィルタ")] = None,
) -> DocumentListResponse:
    """ドキュメント一覧を取得"""
    return service.list_documents(page=page, per_page=per_page, tag=tag)


@router.get(
    "/{document_id}",
    response_model=DocumentDetailResponse,
    summary="ドキュメント詳細取得",
    description="ドキュメントの詳細情報を取得します",
)
def get_document(
    document_id: uuid.UUID,
    service: Annotated[DocumentService, Depends(get_document_service)],
) -> DocumentDetailResponse:
    """ドキュメント詳細を取得"""
    document = service.get_document_with_chunks(document_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="ドキュメントが見つかりません",
        )
    return DocumentDetailResponse.model_validate(document)


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="ドキュメント削除",
    description="ドキュメントを削除します（ベクトルインデックスも削除）",
)
async def delete_document(
    document_id: uuid.UUID,
    service: Annotated[DocumentService, Depends(get_document_service)],
) -> None:
    """ドキュメントを削除"""
    deleted = await service.delete_document(document_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="ドキュメントが見つかりません",
        )
