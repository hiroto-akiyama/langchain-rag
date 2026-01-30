import uuid
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from starlette.datastructures import Headers, UploadFile
from sqlalchemy.orm import Session

from app.models.document import Document, DocumentStatus
from app.services.document_service import DocumentService


def create_upload_file(
    filename: str,
    content: bytes,
    content_type: str = "text/plain",
) -> UploadFile:
    """テスト用UploadFileを作成"""
    return UploadFile(
        filename=filename,
        file=BytesIO(content),
        headers=Headers({"content-type": content_type}),
    )


class TestDocumentService:
    """DocumentServiceのテスト"""

    def test_get_document(
        self,
        db_session: Session,
        sample_document: Document,
    ):
        """ドキュメント取得"""
        service = DocumentService(db_session)
        result = service.get_document(sample_document.id)

        assert result is not None
        assert result.id == sample_document.id
        assert result.filename == "test.pdf"

    def test_get_document_not_found(self, db_session: Session):
        """存在しないドキュメントの取得"""
        service = DocumentService(db_session)
        result = service.get_document(uuid.uuid4())

        assert result is None

    def test_list_documents(
        self,
        db_session: Session,
        sample_document: Document,
    ):
        """ドキュメント一覧取得"""
        service = DocumentService(db_session)
        result = service.list_documents(page=1, per_page=10)

        assert result.total >= 1
        assert len(result.items) >= 1
        assert result.page == 1
        assert result.per_page == 10

    def test_list_documents_pagination(self, db_session: Session):
        """ドキュメント一覧のページネーション"""
        for i in range(15):
            doc = Document(
                filename=f"test{i}.pdf",
                title=f"テスト{i}",
                status=DocumentStatus.INDEXED.value,
            )
            db_session.add(doc)
        db_session.commit()

        service = DocumentService(db_session)

        page1 = service.list_documents(page=1, per_page=10)
        assert len(page1.items) == 10
        assert page1.total == 15

        page2 = service.list_documents(page=2, per_page=10)
        assert len(page2.items) == 5

    @pytest.mark.skip(reason="JSONB contains operator not supported in SQLite")
    def test_list_documents_filter_by_tag(
        self,
        db_session: Session,
        sample_document: Document,
    ):
        """タグによるフィルタリング"""
        service = DocumentService(db_session)

        result = service.list_documents(tag="test")
        assert result.total >= 1

        result = service.list_documents(tag="nonexistent")
        assert result.total == 0

    @pytest.mark.asyncio
    async def test_upload_document(
        self,
        db_session: Session,
        mock_embedding_service: MagicMock,
        mock_search_service: MagicMock,
    ):
        """ドキュメントアップロード"""
        content = b"This is test content for the document."
        file = create_upload_file("test.txt", content, "text/plain")

        with patch("app.services.document_service.settings") as mock_settings:
            mock_settings.chunk_size = 100
            mock_settings.chunk_overlap = 20
            mock_settings.upload_dir = "/tmp/test_uploads"
            mock_settings.max_file_size = 50 * 1024 * 1024

            service = DocumentService(
                db_session,
                embedding_service=mock_embedding_service,
                search_service=mock_search_service,
            )

            result = await service.upload_document(
                file=file,
                title="テストドキュメント",
                tags=["test"],
            )

        assert result is not None
        assert result.filename == "test.txt"
        assert result.title == "テストドキュメント"
        assert result.status == DocumentStatus.INDEXED.value

    @pytest.mark.asyncio
    async def test_upload_document_unsupported_format(
        self,
        db_session: Session,
    ):
        """サポートされていない形式のアップロード"""
        file = create_upload_file("test.exe", b"test", "application/octet-stream")

        service = DocumentService(db_session)

        with pytest.raises(ValueError, match="サポートされていないファイル形式"):
            await service.upload_document(file=file)

    @pytest.mark.asyncio
    async def test_upload_document_file_too_large(
        self,
        db_session: Session,
    ):
        """ファイルサイズ超過"""
        large_content = b"x" * (60 * 1024 * 1024)
        file = create_upload_file("large.txt", large_content, "text/plain")

        with patch("app.services.document_service.settings") as mock_settings:
            mock_settings.max_file_size = 50 * 1024 * 1024

            service = DocumentService(db_session)

            with pytest.raises(ValueError, match="ファイルサイズが制限を超えています"):
                await service.upload_document(file=file)

    @pytest.mark.asyncio
    async def test_delete_document(
        self,
        db_session: Session,
        sample_document: Document,
        mock_search_service: MagicMock,
    ):
        """ドキュメント削除"""
        doc_id = sample_document.id

        service = DocumentService(
            db_session,
            search_service=mock_search_service,
        )
        result = await service.delete_document(doc_id)

        assert result is True
        assert service.get_document(doc_id) is None
        mock_search_service.delete_document_vectors.assert_called_once_with(doc_id)

    @pytest.mark.asyncio
    async def test_delete_document_not_found(
        self,
        db_session: Session,
        mock_search_service: MagicMock,
    ):
        """存在しないドキュメントの削除"""
        service = DocumentService(
            db_session,
            search_service=mock_search_service,
        )
        result = await service.delete_document(uuid.uuid4())

        assert result is False
