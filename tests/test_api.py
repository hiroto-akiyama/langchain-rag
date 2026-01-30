"""APIエンドポイントのテスト"""
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.models.document import DocumentStatus


@pytest.fixture
def mock_search_service_class():
    """SearchServiceクラスのモック"""
    with patch("app.main.SearchService") as mock:
        mock.return_value.ensure_collection = AsyncMock()
        yield mock


@pytest.fixture
def app(mock_search_service_class):
    """テスト用FastAPIアプリ"""
    from app.main import app
    return app


@pytest.fixture
def client(app):
    """テストクライアント"""
    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    """ヘルスチェックエンドポイントのテスト"""

    def test_health_check(self, client: TestClient):
        """ヘルスチェックが正常に動作する"""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "components" in data


class TestRootEndpoint:
    """ルートエンドポイントのテスト"""

    def test_root(self, client: TestClient):
        """ルートエンドポイントが正常に動作する（HTML または JSON）"""
        response = client.get("/")

        assert response.status_code == 200
        # HTMLまたはJSONのいずれかを返す
        content_type = response.headers.get("content-type", "")
        assert "text/html" in content_type or "application/json" in content_type

    def test_api_info(self, client: TestClient):
        """API情報エンドポイントが正常に動作する"""
        response = client.get("/api")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data


class TestDocumentsAPI:
    """ドキュメントAPIのテスト"""

    def test_list_documents(self, app, client: TestClient):
        """ドキュメント一覧取得"""
        from app.api.documents import get_document_service

        mock_service = MagicMock()
        mock_service.list_documents.return_value = MagicMock(
            items=[],
            total=0,
            page=1,
            per_page=20,
        )
        app.dependency_overrides[get_document_service] = lambda: mock_service

        response = client.get("/api/v1/documents")

        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "total" in data

        app.dependency_overrides.clear()

    def test_list_documents_with_pagination(self, app, client: TestClient):
        """ページネーション付きドキュメント一覧取得"""
        from app.api.documents import get_document_service

        mock_service = MagicMock()
        mock_service.list_documents.return_value = MagicMock(
            items=[],
            total=50,
            page=2,
            per_page=10,
        )
        app.dependency_overrides[get_document_service] = lambda: mock_service

        response = client.get("/api/v1/documents?page=2&per_page=10")

        assert response.status_code == 200
        mock_service.list_documents.assert_called_once_with(page=2, per_page=10, tag=None)

        app.dependency_overrides.clear()

    def test_get_document(self, app, client: TestClient):
        """ドキュメント詳細取得"""
        from app.api.documents import get_document_service

        doc_id = uuid.uuid4()
        mock_service = MagicMock()
        mock_doc = MagicMock()
        mock_doc.id = doc_id
        mock_doc.filename = "test.pdf"
        mock_doc.title = "テスト"
        mock_doc.file_path = "/tmp/test.pdf"
        mock_doc.file_size = 1024
        mock_doc.mime_type = "application/pdf"
        mock_doc.tags = []
        mock_doc.status = DocumentStatus.INDEXED
        mock_doc.chunk_count = 5
        mock_doc.created_at = "2025-01-01T00:00:00"
        mock_doc.updated_at = "2025-01-01T00:00:00"
        mock_doc.chunks = []
        mock_service.get_document_with_chunks.return_value = mock_doc
        app.dependency_overrides[get_document_service] = lambda: mock_service

        response = client.get(f"/api/v1/documents/{doc_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(doc_id)

        app.dependency_overrides.clear()

    def test_get_document_not_found(self, app, client: TestClient):
        """存在しないドキュメントの取得"""
        from app.api.documents import get_document_service

        mock_service = MagicMock()
        mock_service.get_document_with_chunks.return_value = None
        app.dependency_overrides[get_document_service] = lambda: mock_service

        response = client.get(f"/api/v1/documents/{uuid.uuid4()}")

        assert response.status_code == 404

        app.dependency_overrides.clear()

    def test_upload_document(self, app, client: TestClient):
        """ドキュメントアップロード"""
        from app.api.documents import get_document_service

        doc_id = uuid.uuid4()
        mock_service = MagicMock()
        mock_doc = MagicMock()
        mock_doc.id = doc_id
        mock_doc.filename = "test.txt"
        mock_doc.title = "テスト"
        mock_doc.file_path = "/tmp/test.txt"
        mock_doc.file_size = 100
        mock_doc.mime_type = "text/plain"
        mock_doc.tags = ["test"]
        mock_doc.status = DocumentStatus.INDEXED
        mock_doc.chunk_count = 1
        mock_doc.created_at = "2025-01-01T00:00:00"
        mock_doc.updated_at = "2025-01-01T00:00:00"
        mock_service.upload_document = AsyncMock(return_value=mock_doc)
        app.dependency_overrides[get_document_service] = lambda: mock_service

        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("test.txt", b"test content", "text/plain")},
            data={"title": "テスト"},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["filename"] == "test.txt"

        app.dependency_overrides.clear()

    def test_upload_document_invalid_format(self, app, client: TestClient):
        """無効なファイル形式のアップロード"""
        from app.api.documents import get_document_service

        mock_service = MagicMock()
        mock_service.upload_document = AsyncMock(
            side_effect=ValueError("サポートされていないファイル形式")
        )
        app.dependency_overrides[get_document_service] = lambda: mock_service

        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("test.exe", b"test", "application/octet-stream")},
        )

        assert response.status_code == 400

        app.dependency_overrides.clear()

    def test_delete_document(self, app, client: TestClient):
        """ドキュメント削除"""
        from app.api.documents import get_document_service

        doc_id = uuid.uuid4()
        mock_service = MagicMock()
        mock_service.delete_document = AsyncMock(return_value=True)
        app.dependency_overrides[get_document_service] = lambda: mock_service

        response = client.delete(f"/api/v1/documents/{doc_id}")

        assert response.status_code == 204

        app.dependency_overrides.clear()

    def test_delete_document_not_found(self, app, client: TestClient):
        """存在しないドキュメントの削除"""
        from app.api.documents import get_document_service

        mock_service = MagicMock()
        mock_service.delete_document = AsyncMock(return_value=False)
        app.dependency_overrides[get_document_service] = lambda: mock_service

        response = client.delete(f"/api/v1/documents/{uuid.uuid4()}")

        assert response.status_code == 404

        app.dependency_overrides.clear()


class TestSearchAPI:
    """検索APIのテスト"""

    def test_search(self, app, client: TestClient):
        """検索実行"""
        from app.api.search import get_search_service

        mock_service = MagicMock()
        mock_service.search = AsyncMock(
            return_value=MagicMock(
                results=[],
                total=0,
                query="テスト検索",
            )
        )
        app.dependency_overrides[get_search_service] = lambda: mock_service

        response = client.post(
            "/api/v1/search",
            json={"query": "テスト検索", "top_k": 5},
        )

        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert data["query"] == "テスト検索"

        app.dependency_overrides.clear()

    def test_search_with_filter(self, app, client: TestClient):
        """フィルタ付き検索"""
        from app.api.search import get_search_service

        mock_service = MagicMock()
        mock_service.search = AsyncMock(
            return_value=MagicMock(
                results=[],
                total=0,
                query="テスト",
            )
        )
        app.dependency_overrides[get_search_service] = lambda: mock_service

        response = client.post(
            "/api/v1/search",
            json={
                "query": "テスト",
                "top_k": 10,
                "score_threshold": 0.5,
                "filter": {"tags": ["important"]},
            },
        )

        assert response.status_code == 200

        app.dependency_overrides.clear()

    def test_search_empty_query(self, client: TestClient):
        """空クエリでの検索"""
        response = client.post(
            "/api/v1/search",
            json={"query": "", "top_k": 5},
        )

        assert response.status_code == 422  # Validation Error


class TestChatAPI:
    """チャットAPIのテスト"""

    def test_chat(self, app, client: TestClient):
        """チャット実行"""
        from app.api.chat import get_rag_service

        mock_service = MagicMock()
        mock_service.chat = AsyncMock(
            return_value=MagicMock(
                answer="テスト回答です。",
                sources=[],
                question="テスト質問",
            )
        )
        app.dependency_overrides[get_rag_service] = lambda: mock_service

        response = client.post(
            "/api/v1/chat",
            json={"question": "テスト質問"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data

        app.dependency_overrides.clear()

    def test_chat_with_history(self, app, client: TestClient):
        """会話履歴付きチャット"""
        from app.api.chat import get_rag_service

        mock_service = MagicMock()
        mock_service.chat = AsyncMock(
            return_value=MagicMock(
                answer="テスト回答です。",
                sources=[],
                question="続きの質問",
            )
        )
        app.dependency_overrides[get_rag_service] = lambda: mock_service

        response = client.post(
            "/api/v1/chat",
            json={
                "question": "続きの質問",
                "chat_history": [
                    {"role": "user", "content": "前の質問"},
                    {"role": "assistant", "content": "前の回答"},
                ],
            },
        )

        assert response.status_code == 200

        app.dependency_overrides.clear()

    def test_chat_stream(self, app, client: TestClient):
        """ストリーミングチャット"""
        from app.api.chat import get_rag_service
        from app.schemas.chat import ChatStreamChunk

        mock_service = MagicMock()

        async def mock_stream(request):
            yield ChatStreamChunk(content="テスト", done=False, sources=None)
            yield ChatStreamChunk(content="", done=True, sources=[])

        mock_service.chat_stream = mock_stream
        app.dependency_overrides[get_rag_service] = lambda: mock_service

        response = client.post(
            "/api/v1/chat/stream",
            json={"question": "テスト質問"},
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

        app.dependency_overrides.clear()

    def test_chat_empty_question(self, client: TestClient):
        """空の質問でのチャット"""
        response = client.post(
            "/api/v1/chat",
            json={"question": ""},
        )

        assert response.status_code == 422  # Validation Error


class TestOpenAPISchema:
    """OpenAPIスキーマのテスト"""

    def test_openapi_schema_accessible(self, client: TestClient):
        """OpenAPIスキーマにアクセスできる"""
        response = client.get("/openapi.json")

        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "paths" in data

    def test_docs_accessible(self, client: TestClient):
        """Swagger UIにアクセスできる"""
        response = client.get("/docs")

        assert response.status_code == 200
