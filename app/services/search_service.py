import logging
import uuid
from typing import TYPE_CHECKING

from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)
from qdrant_client.http import models as qdrant_models

from app.config import settings
from app.schemas.search import (
    SearchFilter,
    SearchRequest,
    SearchResponse,
    SearchResult,
    SearchResultMetadata,
)
from app.services.embedding_service import EmbeddingService

if TYPE_CHECKING:
    from app.models.chunk import Chunk
    from app.models.document import Document


class SearchService:
    """ベクトル検索サービス"""

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        qdrant_url: str | None = None,
        collection_name: str | None = None,
    ):
        self.embedding_service = embedding_service or EmbeddingService()
        self.qdrant_url = qdrant_url or settings.qdrant_url
        self.collection_name = collection_name or settings.qdrant_collection
        self._client: QdrantClient | None = None

    @property
    def client(self) -> QdrantClient:
        """Qdrantクライアントを取得（遅延初期化）"""
        if self._client is None:
            self._client = QdrantClient(url=self.qdrant_url)
        return self._client

    async def ensure_collection(self) -> None:
        """コレクションが存在することを確認し、なければ作成"""
        collections = self.client.get_collections()
        collection_names = [c.name for c in collections.collections]

        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qdrant_models.VectorParams(
                    size=settings.qdrant_vector_size,
                    distance=qdrant_models.Distance.COSINE,
                ),
            )

    async def index_chunks(
        self,
        point_ids: list[uuid.UUID],
        embeddings: list[list[float]],
        chunks: list["Chunk"],
        document: "Document",
    ) -> None:
        """
        チャンクをQdrantにインデックス

        Args:
            point_ids: ポイントIDのリスト
            embeddings: 埋め込みベクトルのリスト
            chunks: チャンクのリスト
            document: 親ドキュメント
        """
        await self.ensure_collection()

        points = []
        for point_id, embedding, chunk in zip(point_ids, embeddings, chunks):
            payload = {
                "document_id": str(document.id),
                "chunk_id": str(chunk.id),
                "filename": document.filename,
                "title": document.title,
                "tags": document.tags or [],
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
            }
            points.append(
                qdrant_models.PointStruct(
                    id=str(point_id),
                    vector=embedding,
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    async def search(
        self,
        request: SearchRequest,
    ) -> SearchResponse:
        """
        類似ドキュメントを検索

        Args:
            request: 検索リクエスト

        Returns:
            検索結果
        """
        query_embedding = await self.embedding_service.create_embedding(request.query)

        filter_conditions = self._build_filter(request.filter)

        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=request.top_k,
            score_threshold=request.score_threshold if request.score_threshold > 0 else None,
            query_filter=filter_conditions,
        )

        logger.info(f"Qdrant検索結果: {len(search_result)}件")

        results = []
        skipped_count = 0
        for point in search_result:
            payload = point.payload or {}
            document_id_str = payload.get("document_id")
            chunk_id_str = payload.get("chunk_id")

            # 必須フィールドが欠けているレコードはスキップ
            if not document_id_str or not chunk_id_str:
                logger.warning(f"必須フィールド欠落でスキップ: document_id={document_id_str}, chunk_id={chunk_id_str}")
                skipped_count += 1
                continue

            try:
                results.append(
                    SearchResult(
                        document_id=uuid.UUID(document_id_str),
                        chunk_id=uuid.UUID(chunk_id_str),
                        content=payload.get("content", ""),
                        score=point.score,
                        metadata=SearchResultMetadata(
                            filename=payload.get("filename"),
                            title=payload.get("title"),
                            page_number=payload.get("page_number"),
                            tags=payload.get("tags"),
                        ),
                    )
                )
            except ValueError as e:
                # 不正なUUID形式のレコードはスキップ
                logger.warning(f"不正なUUID形式でスキップ: {e}, document_id={document_id_str!r}, chunk_id={chunk_id_str!r}")
                skipped_count += 1
                continue

        if skipped_count > 0:
            logger.warning(f"合計 {skipped_count} 件のレコードをスキップしました")

        return SearchResponse(
            results=results,
            total=len(results),
            query=request.query,
        )

    def _build_filter(
        self,
        search_filter: SearchFilter | None,
    ) -> qdrant_models.Filter | None:
        """検索フィルタを構築"""
        if not search_filter:
            return None

        conditions = []

        if search_filter.tags:
            conditions.append(
                qdrant_models.FieldCondition(
                    key="tags",
                    match=qdrant_models.MatchAny(any=search_filter.tags),
                )
            )

        if search_filter.document_ids:
            document_id_strs = [str(doc_id) for doc_id in search_filter.document_ids]
            conditions.append(
                qdrant_models.FieldCondition(
                    key="document_id",
                    match=qdrant_models.MatchAny(any=document_id_strs),
                )
            )

        if not conditions:
            return None

        return qdrant_models.Filter(must=conditions)

    async def delete_document_vectors(self, document_id: uuid.UUID) -> None:
        """ドキュメントのベクトルを削除"""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=qdrant_models.FilterSelector(
                filter=qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="document_id",
                            match=qdrant_models.MatchValue(value=str(document_id)),
                        )
                    ]
                )
            ),
        )

    async def get_chunks_by_ids(
        self,
        chunk_ids: list[uuid.UUID],
    ) -> list[dict]:
        """チャンクIDからチャンク情報を取得"""
        chunk_id_strs = [str(cid) for cid in chunk_ids]

        results = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="chunk_id",
                        match=qdrant_models.MatchAny(any=chunk_id_strs),
                    )
                ]
            ),
            limit=len(chunk_ids),
            with_payload=True,
        )

        return [point.payload for point in results[0] if point.payload]
