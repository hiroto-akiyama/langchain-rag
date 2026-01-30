import os
import uuid
from pathlib import Path

from fastapi import UploadFile
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.models.chunk import Chunk
from app.models.document import Document, DocumentStatus
from app.schemas.document import DocumentListResponse, DocumentResponse
from app.services.embedding_service import EmbeddingService
from app.services.search_service import SearchService
from app.utils.file_parser import FileParser
from app.utils.text_splitter import TextSplitter


class DocumentService:
    """ドキュメント管理サービス"""

    def __init__(
        self,
        db: Session,
        embedding_service: EmbeddingService | None = None,
        search_service: SearchService | None = None,
    ):
        self.db = db
        self.embedding_service = embedding_service or EmbeddingService()
        self.search_service = search_service or SearchService()
        self.file_parser = FileParser()
        self.text_splitter = TextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    async def upload_document(
        self,
        file: UploadFile,
        title: str | None = None,
        tags: list[str] | None = None,
    ) -> Document:
        """
        ドキュメントをアップロードしてインデックスを作成

        Args:
            file: アップロードされたファイル
            title: ドキュメントタイトル（オプション）
            tags: タグリスト（オプション）

        Returns:
            作成されたドキュメント
        """
        filename = file.filename or "unknown"
        mime_type = file.content_type

        if not FileParser.is_supported(filename, mime_type):
            raise ValueError(f"サポートされていないファイル形式: {filename}")

        content = await file.read()
        file_size = len(content)

        if file_size > settings.max_file_size:
            raise ValueError(
                f"ファイルサイズが制限を超えています: {file_size} > {settings.max_file_size}"
            )

        document = Document(
            filename=filename,
            title=title or filename,
            file_size=file_size,
            mime_type=mime_type or FileParser.get_mime_type(filename),
            tags=tags or [],
            status=DocumentStatus.PROCESSING.value,
        )
        self.db.add(document)
        self.db.flush()

        file_path = await self._save_file(document.id, filename, content)
        document.file_path = file_path

        try:
            await self._process_document(document, content, filename)
            document.status = DocumentStatus.INDEXED.value
        except Exception as e:
            document.status = DocumentStatus.FAILED.value
            raise e
        finally:
            self.db.commit()

        return document

    async def _save_file(
        self,
        document_id: uuid.UUID,
        filename: str,
        content: bytes,
    ) -> str:
        """ファイルを保存"""
        upload_dir = Path(settings.upload_dir)
        upload_dir.mkdir(parents=True, exist_ok=True)

        safe_filename = f"{document_id}_{filename}"
        file_path = upload_dir / safe_filename

        file_path.write_bytes(content)
        return str(file_path)

    async def _process_document(
        self,
        document: Document,
        content: bytes,
        filename: str,
    ) -> None:
        """ドキュメントを処理してインデックスを作成"""
        parsed = self.file_parser.parse_bytes(content, filename)
        page_numbers = (
            list(range(1, parsed.page_count + 1)) if parsed.page_count else None
        )
        text_chunks = self.text_splitter.split_text(parsed.text, page_numbers)

        if not text_chunks:
            document.chunk_count = 0
            return

        chunk_texts = [chunk.content for chunk in text_chunks]
        embeddings = await self.embedding_service.create_embeddings(chunk_texts)

        chunks: list[Chunk] = []
        point_ids: list[uuid.UUID] = []

        for idx, (text_chunk, embedding) in enumerate(zip(text_chunks, embeddings)):
            chunk_id = uuid.uuid4()
            point_id = uuid.uuid4()
            chunk = Chunk(
                id=chunk_id,
                document_id=document.id,
                content=text_chunk.content,
                chunk_index=idx,
                page_number=text_chunk.page_number,
                qdrant_point_id=point_id,
                chunk_metadata=text_chunk.metadata,
            )
            chunks.append(chunk)
            point_ids.append(point_id)

        await self.search_service.index_chunks(
            point_ids=point_ids,
            embeddings=embeddings,
            chunks=chunks,
            document=document,
        )

        self.db.add_all(chunks)
        document.chunk_count = len(chunks)

    def get_document(self, document_id: uuid.UUID) -> Document | None:
        """ドキュメントを取得"""
        return self.db.get(Document, document_id)

    def get_document_with_chunks(self, document_id: uuid.UUID) -> Document | None:
        """チャンクを含むドキュメントを取得"""
        stmt = select(Document).where(Document.id == document_id)
        return self.db.execute(stmt).scalar_one_or_none()

    def list_documents(
        self,
        page: int = 1,
        per_page: int = 20,
        tag: str | None = None,
    ) -> DocumentListResponse:
        """ドキュメント一覧を取得"""
        query = select(Document)

        if tag:
            query = query.where(Document.tags.contains([tag]))

        total_query = select(func.count()).select_from(query.subquery())
        total = self.db.execute(total_query).scalar() or 0

        query = (
            query.order_by(Document.created_at.desc())
            .offset((page - 1) * per_page)
            .limit(per_page)
        )
        documents = list(self.db.execute(query).scalars().all())

        return DocumentListResponse(
            items=[DocumentResponse.model_validate(doc) for doc in documents],
            total=total,
            page=page,
            per_page=per_page,
        )

    async def delete_document(self, document_id: uuid.UUID) -> bool:
        """ドキュメントを削除"""
        document = self.get_document(document_id)
        if not document:
            return False

        await self.search_service.delete_document_vectors(document_id)

        if document.file_path and os.path.exists(document.file_path):
            os.remove(document.file_path)

        self.db.delete(document)
        self.db.commit()
        return True
