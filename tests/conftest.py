import uuid
from collections.abc import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import JSON, create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.sqlite.base import SQLiteTypeCompiler
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.models.database import Base
from app.models.document import Document, DocumentStatus
from app.models.chunk import Chunk

# SQLiteでJSONBをJSONとして扱う
SQLiteTypeCompiler.visit_JSONB = SQLiteTypeCompiler.visit_JSON


@pytest.fixture
def db_session() -> Generator[Session, None, None]:
    """テスト用のインメモリSQLiteセッション"""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)


@pytest.fixture
def sample_document(db_session: Session) -> Document:
    """テスト用サンプルドキュメント"""
    doc = Document(
        id=uuid.uuid4(),
        filename="test.pdf",
        title="テストドキュメント",
        file_path="/tmp/test.pdf",
        file_size=1024,
        mime_type="application/pdf",
        tags=["test", "sample"],
        status=DocumentStatus.INDEXED.value,
        chunk_count=3,
    )
    db_session.add(doc)
    db_session.commit()
    db_session.refresh(doc)
    return doc


@pytest.fixture
def sample_chunks(db_session: Session, sample_document: Document) -> list[Chunk]:
    """テスト用サンプルチャンク"""
    chunks = []
    for i in range(3):
        chunk = Chunk(
            id=uuid.uuid4(),
            document_id=sample_document.id,
            content=f"これはテストチャンク{i+1}の内容です。",
            chunk_index=i,
            page_number=i + 1,
            qdrant_point_id=uuid.uuid4(),
        )
        chunks.append(chunk)
    db_session.add_all(chunks)
    db_session.commit()
    for chunk in chunks:
        db_session.refresh(chunk)
    return chunks


@pytest.fixture
def mock_embedding_service() -> MagicMock:
    """モック埋め込みサービス"""
    mock = MagicMock()
    mock.create_embedding = AsyncMock(return_value=[0.1] * 1536)
    mock.create_embeddings = AsyncMock(return_value=[[0.1] * 1536] * 3)
    mock.vector_size = 1536
    return mock


@pytest.fixture
def mock_search_service() -> MagicMock:
    """モック検索サービス"""
    mock = MagicMock()
    mock.search = AsyncMock()
    mock.index_chunks = AsyncMock()
    mock.delete_document_vectors = AsyncMock()
    mock.ensure_collection = AsyncMock()
    return mock


@pytest.fixture
def mock_qdrant_client() -> MagicMock:
    """モックQdrantクライアント"""
    mock = MagicMock()
    mock.get_collections.return_value = MagicMock(collections=[])
    mock.create_collection = MagicMock()
    mock.upsert = MagicMock()
    mock.search = MagicMock(return_value=[])
    mock.delete = MagicMock()
    mock.scroll = MagicMock(return_value=([], None))
    return mock


@pytest.fixture
def mock_openai() -> Generator[MagicMock, None, None]:
    """モックOpenAI"""
    with patch("langchain_openai.ChatOpenAI") as mock_chat, \
         patch("langchain_openai.OpenAIEmbeddings") as mock_embed:
        mock_chat_instance = MagicMock()
        mock_chat_instance.ainvoke = AsyncMock(
            return_value=MagicMock(content="テスト回答です。")
        )
        mock_chat.return_value = mock_chat_instance

        mock_embed_instance = MagicMock()
        mock_embed_instance.aembed_query = AsyncMock(return_value=[0.1] * 1536)
        mock_embed_instance.aembed_documents = AsyncMock(
            return_value=[[0.1] * 1536] * 3
        )
        mock_embed.return_value = mock_embed_instance

        yield {"chat": mock_chat_instance, "embed": mock_embed_instance}
