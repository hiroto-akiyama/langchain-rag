import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.schemas.chat import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatRole,
    ChatSource,
)
from app.schemas.search import SearchResponse, SearchResult, SearchResultMetadata
from app.services.rag_service import RAGService


class TestRAGService:
    """RAGServiceのテスト"""

    @pytest.fixture
    def mock_search_results(self) -> SearchResponse:
        """モック検索結果"""
        doc_id = uuid.uuid4()
        chunk_id = uuid.uuid4()
        return SearchResponse(
            results=[
                SearchResult(
                    document_id=doc_id,
                    chunk_id=chunk_id,
                    content="これは検索結果のテストコンテンツです。休暇申請について記載しています。",
                    score=0.92,
                    metadata=SearchResultMetadata(
                        filename="規則.pdf",
                        title="就業規則",
                        page_number=15,
                        tags=["規則", "人事"],
                    ),
                )
            ],
            total=1,
            query="休暇申請",
        )

    @pytest.fixture
    def rag_service(self, mock_search_service: MagicMock) -> RAGService:
        """テスト用RAGService"""
        return RAGService(search_service=mock_search_service)

    @pytest.mark.asyncio
    async def test_chat_basic(
        self,
        rag_service: RAGService,
        mock_search_service: MagicMock,
        mock_search_results: SearchResponse,
    ):
        """基本的なチャット"""
        mock_search_service.search.return_value = mock_search_results

        with patch.object(rag_service, "_llm") as mock_llm:
            mock_response = MagicMock()
            mock_response.content = "休暇申請のルールは以下の通りです..."
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            rag_service._llm = mock_llm

            request = ChatRequest(
                question="休暇申請のルールを教えてください",
                top_k=5,
            )
            result = await rag_service.chat(request)

        assert isinstance(result, ChatResponse)
        assert result.answer == "休暇申請のルールは以下の通りです..."
        assert result.question == "休暇申請のルールを教えてください"
        assert len(result.sources) == 1
        assert result.sources[0].filename == "規則.pdf"

    @pytest.mark.asyncio
    async def test_chat_with_history(
        self,
        rag_service: RAGService,
        mock_search_service: MagicMock,
        mock_search_results: SearchResponse,
    ):
        """会話履歴付きチャット"""
        mock_search_service.search.return_value = mock_search_results

        with patch.object(rag_service, "_llm") as mock_llm:
            mock_response = MagicMock()
            mock_response.content = "回答内容"
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            rag_service._llm = mock_llm

            request = ChatRequest(
                question="それについて詳しく教えてください",
                chat_history=[
                    ChatMessage(role=ChatRole.USER, content="休暇について教えて"),
                    ChatMessage(role=ChatRole.ASSISTANT, content="休暇制度があります"),
                ],
                top_k=5,
            )
            result = await rag_service.chat(request)

        assert result is not None
        call_args = mock_llm.ainvoke.call_args[0][0]
        assert len(call_args) >= 3

    @pytest.mark.asyncio
    async def test_chat_no_results(
        self,
        rag_service: RAGService,
        mock_search_service: MagicMock,
    ):
        """検索結果がない場合のチャット"""
        mock_search_service.search.return_value = SearchResponse(
            results=[],
            total=0,
            query="不明なクエリ",
        )

        with patch.object(rag_service, "_llm") as mock_llm:
            mock_response = MagicMock()
            mock_response.content = "関連する情報が見つかりませんでした。"
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            rag_service._llm = mock_llm

            request = ChatRequest(question="存在しない情報", top_k=5)
            result = await rag_service.chat(request)

        assert result.sources == []

    @pytest.mark.asyncio
    async def test_chat_stream(
        self,
        rag_service: RAGService,
        mock_search_service: MagicMock,
        mock_search_results: SearchResponse,
    ):
        """ストリーミングチャット"""
        mock_search_service.search.return_value = mock_search_results

        async def mock_stream(*args, **kwargs):
            chunks = [
                MagicMock(content="回答"),
                MagicMock(content="の"),
                MagicMock(content="続き"),
            ]
            for chunk in chunks:
                yield chunk

        with patch.object(rag_service, "_llm") as mock_llm:
            mock_llm.astream = mock_stream
            rag_service._llm = mock_llm

            request = ChatRequest(question="テスト質問", top_k=5)
            chunks = []
            async for chunk in rag_service.chat_stream(request):
                chunks.append(chunk)

        assert len(chunks) >= 1
        assert chunks[-1].done is True
        assert chunks[-1].sources is not None

    def test_build_context(self, rag_service: RAGService):
        """コンテキスト構築"""
        doc_id = uuid.uuid4()
        chunk_id = uuid.uuid4()
        results = [
            SearchResult(
                document_id=doc_id,
                chunk_id=chunk_id,
                content="コンテンツ1",
                score=0.9,
                metadata=SearchResultMetadata(
                    filename="file1.pdf",
                    page_number=1,
                ),
            ),
            SearchResult(
                document_id=doc_id,
                chunk_id=uuid.uuid4(),
                content="コンテンツ2",
                score=0.8,
                metadata=SearchResultMetadata(
                    filename="file2.pdf",
                    page_number=5,
                ),
            ),
        ]

        context = rag_service._build_context(results)

        assert "コンテンツ1" in context
        assert "コンテンツ2" in context
        assert "file1.pdf" in context
        assert "ページ: 1" in context

    def test_build_context_empty(self, rag_service: RAGService):
        """空の検索結果でのコンテキスト構築"""
        context = rag_service._build_context([])

        assert "関連するドキュメントが見つかりませんでした" in context

    def test_build_messages(self, rag_service: RAGService):
        """メッセージ構築"""
        context = "テストコンテキスト"
        question = "テスト質問"

        messages = rag_service._build_messages(context, question)

        assert len(messages) >= 2
        assert "テストコンテキスト" in messages[0].content
        assert messages[-1].content == question

    def test_build_messages_with_history(self, rag_service: RAGService):
        """会話履歴付きメッセージ構築"""
        context = "コンテキスト"
        question = "新しい質問"
        history = [
            ChatMessage(role=ChatRole.USER, content="前の質問"),
            ChatMessage(role=ChatRole.ASSISTANT, content="前の回答"),
        ]

        messages = rag_service._build_messages(context, question, history)

        assert len(messages) == 4
        assert messages[1].content == "前の質問"
        assert messages[2].content == "前の回答"
        assert messages[3].content == question

    @pytest.mark.asyncio
    async def test_search_only(
        self,
        rag_service: RAGService,
        mock_search_service: MagicMock,
        mock_search_results: SearchResponse,
    ):
        """検索のみ（LLM不使用）"""
        mock_search_service.search.return_value = mock_search_results

        result = await rag_service.search_only(
            query="休暇申請",
            top_k=5,
            score_threshold=0.5,
        )

        assert len(result) == 1
        assert isinstance(result[0], ChatSource)
        assert result[0].filename == "規則.pdf"
