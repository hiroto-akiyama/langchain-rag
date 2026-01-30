import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.schemas.search import (
    SearchFilter,
    SearchRequest,
    SearchResponse,
    SearchResult,
    SearchResultMetadata,
)
from app.services.search_service import SearchService


class TestSearchService:
    """SearchServiceのテスト"""

    @pytest.fixture
    def search_service(
        self,
        mock_embedding_service: MagicMock,
        mock_qdrant_client: MagicMock,
    ) -> SearchService:
        """テスト用SearchService"""
        with patch("app.services.search_service.QdrantClient") as mock_client_cls:
            mock_client_cls.return_value = mock_qdrant_client
            service = SearchService(embedding_service=mock_embedding_service)
            service._client = mock_qdrant_client
            return service

    @pytest.mark.asyncio
    async def test_ensure_collection_creates_new(
        self,
        search_service: SearchService,
        mock_qdrant_client: MagicMock,
    ):
        """コレクションが存在しない場合に作成"""
        mock_qdrant_client.get_collections.return_value = MagicMock(collections=[])

        await search_service.ensure_collection()

        mock_qdrant_client.create_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_collection_already_exists(
        self,
        search_service: SearchService,
        mock_qdrant_client: MagicMock,
    ):
        """コレクションが既に存在する場合"""
        mock_collection = MagicMock()
        mock_collection.name = "documents"
        mock_qdrant_client.get_collections.return_value = MagicMock(
            collections=[mock_collection]
        )

        await search_service.ensure_collection()

        mock_qdrant_client.create_collection.assert_not_called()

    @pytest.mark.asyncio
    async def test_search_basic(
        self,
        search_service: SearchService,
        mock_qdrant_client: MagicMock,
        mock_embedding_service: MagicMock,
    ):
        """基本的な検索"""
        doc_id = uuid.uuid4()
        chunk_id = uuid.uuid4()

        mock_point = MagicMock()
        mock_point.score = 0.95
        mock_point.payload = {
            "document_id": str(doc_id),
            "chunk_id": str(chunk_id),
            "content": "テストコンテンツ",
            "filename": "test.pdf",
            "title": "テストドキュメント",
            "page_number": 1,
            "tags": ["test"],
        }
        mock_qdrant_client.search.return_value = [mock_point]

        request = SearchRequest(query="テスト", top_k=5)
        result = await search_service.search(request)

        assert isinstance(result, SearchResponse)
        assert result.total == 1
        assert result.query == "テスト"
        assert len(result.results) == 1
        assert result.results[0].document_id == doc_id
        assert result.results[0].score == 0.95

    @pytest.mark.asyncio
    async def test_search_with_filter(
        self,
        search_service: SearchService,
        mock_qdrant_client: MagicMock,
        mock_embedding_service: MagicMock,
    ):
        """フィルタ付き検索"""
        mock_qdrant_client.search.return_value = []

        request = SearchRequest(
            query="テスト",
            top_k=5,
            filter=SearchFilter(tags=["important"]),
        )
        await search_service.search(request)

        call_args = mock_qdrant_client.search.call_args
        assert call_args.kwargs.get("query_filter") is not None

    @pytest.mark.asyncio
    async def test_search_with_score_threshold(
        self,
        search_service: SearchService,
        mock_qdrant_client: MagicMock,
        mock_embedding_service: MagicMock,
    ):
        """スコア閾値付き検索"""
        mock_qdrant_client.search.return_value = []

        request = SearchRequest(
            query="テスト",
            top_k=5,
            score_threshold=0.7,
        )
        await search_service.search(request)

        call_args = mock_qdrant_client.search.call_args
        assert call_args.kwargs.get("score_threshold") == 0.7

    @pytest.mark.asyncio
    async def test_search_empty_results(
        self,
        search_service: SearchService,
        mock_qdrant_client: MagicMock,
        mock_embedding_service: MagicMock,
    ):
        """検索結果が空の場合"""
        mock_qdrant_client.search.return_value = []

        request = SearchRequest(query="存在しないクエリ", top_k=5)
        result = await search_service.search(request)

        assert result.total == 0
        assert len(result.results) == 0

    @pytest.mark.asyncio
    async def test_index_chunks(
        self,
        search_service: SearchService,
        mock_qdrant_client: MagicMock,
        sample_document,
        sample_chunks,
    ):
        """チャンクのインデックス作成"""
        point_ids = [uuid.uuid4() for _ in range(3)]
        embeddings = [[0.1] * 1536 for _ in range(3)]

        await search_service.index_chunks(
            point_ids=point_ids,
            embeddings=embeddings,
            chunks=sample_chunks,
            document=sample_document,
        )

        mock_qdrant_client.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_document_vectors(
        self,
        search_service: SearchService,
        mock_qdrant_client: MagicMock,
    ):
        """ドキュメントベクトルの削除"""
        doc_id = uuid.uuid4()

        await search_service.delete_document_vectors(doc_id)

        mock_qdrant_client.delete.assert_called_once()

    def test_build_filter_with_tags(self, search_service: SearchService):
        """タグフィルタの構築"""
        filter_obj = SearchFilter(tags=["tag1", "tag2"])
        result = search_service._build_filter(filter_obj)

        assert result is not None
        assert len(result.must) == 1

    def test_build_filter_with_document_ids(self, search_service: SearchService):
        """ドキュメントIDフィルタの構築"""
        doc_ids = [uuid.uuid4(), uuid.uuid4()]
        filter_obj = SearchFilter(document_ids=doc_ids)
        result = search_service._build_filter(filter_obj)

        assert result is not None
        assert len(result.must) == 1

    def test_build_filter_empty(self, search_service: SearchService):
        """空フィルタの構築"""
        result = search_service._build_filter(None)
        assert result is None

        result = search_service._build_filter(SearchFilter())
        assert result is None

    @pytest.mark.asyncio
    async def test_search_skips_missing_document_id(
        self,
        search_service: SearchService,
        mock_qdrant_client: MagicMock,
        mock_embedding_service: MagicMock,
    ):
        """document_idがないレコードはスキップ"""
        chunk_id = uuid.uuid4()

        mock_point = MagicMock()
        mock_point.score = 0.95
        mock_point.payload = {
            "chunk_id": str(chunk_id),
            "content": "テストコンテンツ",
        }
        mock_qdrant_client.search.return_value = [mock_point]

        request = SearchRequest(query="テスト", top_k=5)
        result = await search_service.search(request)

        assert result.total == 0
        assert len(result.results) == 0

    @pytest.mark.asyncio
    async def test_search_skips_missing_chunk_id(
        self,
        search_service: SearchService,
        mock_qdrant_client: MagicMock,
        mock_embedding_service: MagicMock,
    ):
        """chunk_idがないレコードはスキップ"""
        doc_id = uuid.uuid4()

        mock_point = MagicMock()
        mock_point.score = 0.95
        mock_point.payload = {
            "document_id": str(doc_id),
            "content": "テストコンテンツ",
        }
        mock_qdrant_client.search.return_value = [mock_point]

        request = SearchRequest(query="テスト", top_k=5)
        result = await search_service.search(request)

        assert result.total == 0
        assert len(result.results) == 0

    @pytest.mark.asyncio
    async def test_search_skips_invalid_uuid(
        self,
        search_service: SearchService,
        mock_qdrant_client: MagicMock,
        mock_embedding_service: MagicMock,
    ):
        """不正なUUID形式のレコードはスキップ"""
        mock_point = MagicMock()
        mock_point.score = 0.95
        mock_point.payload = {
            "document_id": "invalid-uuid",
            "chunk_id": "also-invalid",
            "content": "テストコンテンツ",
        }
        mock_qdrant_client.search.return_value = [mock_point]

        request = SearchRequest(query="テスト", top_k=5)
        result = await search_service.search(request)

        assert result.total == 0
        assert len(result.results) == 0

    @pytest.mark.asyncio
    async def test_search_mixed_valid_invalid_records(
        self,
        search_service: SearchService,
        mock_qdrant_client: MagicMock,
        mock_embedding_service: MagicMock,
    ):
        """有効なレコードと無効なレコードが混在する場合、有効なもののみ返す"""
        valid_doc_id = uuid.uuid4()
        valid_chunk_id = uuid.uuid4()

        # 有効なレコード
        valid_point = MagicMock()
        valid_point.score = 0.95
        valid_point.payload = {
            "document_id": str(valid_doc_id),
            "chunk_id": str(valid_chunk_id),
            "content": "有効なコンテンツ",
            "filename": "valid.pdf",
        }

        # document_idがないレコード
        invalid_point1 = MagicMock()
        invalid_point1.score = 0.90
        invalid_point1.payload = {
            "chunk_id": str(uuid.uuid4()),
            "content": "無効なコンテンツ1",
        }

        # 不正なUUID形式のレコード
        invalid_point2 = MagicMock()
        invalid_point2.score = 0.85
        invalid_point2.payload = {
            "document_id": "not-a-uuid",
            "chunk_id": "also-not-a-uuid",
            "content": "無効なコンテンツ2",
        }

        mock_qdrant_client.search.return_value = [
            valid_point,
            invalid_point1,
            invalid_point2,
        ]

        request = SearchRequest(query="テスト", top_k=5)
        result = await search_service.search(request)

        assert result.total == 1
        assert len(result.results) == 1
        assert result.results[0].document_id == valid_doc_id
        assert result.results[0].chunk_id == valid_chunk_id
        assert result.results[0].content == "有効なコンテンツ"

    @pytest.mark.asyncio
    async def test_search_skips_empty_payload(
        self,
        search_service: SearchService,
        mock_qdrant_client: MagicMock,
        mock_embedding_service: MagicMock,
    ):
        """payloadが空またはNoneのレコードはスキップ"""
        mock_point1 = MagicMock()
        mock_point1.score = 0.95
        mock_point1.payload = None

        mock_point2 = MagicMock()
        mock_point2.score = 0.90
        mock_point2.payload = {}

        mock_qdrant_client.search.return_value = [mock_point1, mock_point2]

        request = SearchRequest(query="テスト", top_k=5)
        result = await search_service.search(request)

        assert result.total == 0
        assert len(result.results) == 0
