import logging
from collections.abc import AsyncGenerator
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

from app.config import settings
from app.schemas.chat import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatRole,
    ChatSource,
    ChatStreamChunk,
)
from app.schemas.search import SearchFilter, SearchRequest
from app.services.search_service import SearchService


SYSTEM_PROMPT = """あなたは社内ドキュメントに基づいて質問に回答するAIアシスタントです。

以下のルールに従って回答してください：
1. 提供されたコンテキスト（検索結果）に基づいて回答してください
2. コンテキストに含まれない情報は回答しないでください
3. 不明な点や情報が不足している場合は、その旨を正直に伝えてください
4. 回答は簡潔で分かりやすい日本語で行ってください
5. 必要に応じて箇条書きや見出しを使って整理してください

コンテキスト（検索結果）:
{context}
"""


class RAGService:
    """RAG質問応答サービス"""

    def __init__(
        self,
        search_service: SearchService | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ):
        self.search_service = search_service or SearchService()
        self.api_key = api_key or settings.openai_api_key
        self.model = model or settings.openai_chat_model
        self._llm: ChatOpenAI | None = None

    @property
    def llm(self) -> ChatOpenAI:
        """ChatOpenAIインスタンスを取得（遅延初期化）"""
        if self._llm is None:
            self._llm = ChatOpenAI(
                api_key=self.api_key,
                model=self.model,
                temperature=0.7,
            )
        return self._llm

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """
        RAGによる質問応答

        Args:
            request: チャットリクエスト

        Returns:
            チャットレスポンス
        """
        search_request = SearchRequest(
            query=request.question,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
        )
        search_response = await self.search_service.search(search_request)

        logger.info(f"検索クエリ: {request.question}")
        logger.info(f"検索結果件数: {len(search_response.results)}")
        for i, result in enumerate(search_response.results):
            logger.info(f"結果[{i}] スコア: {result.score:.4f}, コンテンツ: {result.content[:100]}...")

        context = self._build_context(search_response.results)
        messages = self._build_messages(
            context=context,
            question=request.question,
            chat_history=request.chat_history,
        )

        response = await self.llm.ainvoke(messages)
        answer = response.content if isinstance(response.content, str) else str(response.content)

        sources = [
            ChatSource(
                document_id=result.document_id,
                chunk_id=result.chunk_id,
                filename=result.metadata.filename,
                title=result.metadata.title,
                chunk_content=result.content[:500],
                page_number=result.metadata.page_number,
                score=result.score,
            )
            for result in search_response.results
        ]

        return ChatResponse(
            answer=answer,
            sources=sources,
            question=request.question,
        )

    async def chat_stream(
        self,
        request: ChatRequest,
    ) -> AsyncGenerator[ChatStreamChunk, None]:
        """
        ストリーミング形式でRAG質問応答

        Args:
            request: チャットリクエスト

        Yields:
            ストリーミングチャンク
        """
        search_request = SearchRequest(
            query=request.question,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
        )
        search_response = await self.search_service.search(search_request)

        context = self._build_context(search_response.results)
        messages = self._build_messages(
            context=context,
            question=request.question,
            chat_history=request.chat_history,
        )

        sources = [
            ChatSource(
                document_id=result.document_id,
                chunk_id=result.chunk_id,
                filename=result.metadata.filename,
                title=result.metadata.title,
                chunk_content=result.content[:500],
                page_number=result.metadata.page_number,
                score=result.score,
            )
            for result in search_response.results
        ]

        async for chunk in self.llm.astream(messages):
            content = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
            if content:
                yield ChatStreamChunk(content=content, done=False)

        yield ChatStreamChunk(content="", done=True, sources=sources)

    def _build_context(self, results: list[Any]) -> str:
        """検索結果からコンテキストを構築"""
        if not results:
            return "（関連するドキュメントが見つかりませんでした）"

        context_parts = []
        for i, result in enumerate(results, 1):
            source_info = []
            if result.metadata.filename:
                source_info.append(f"ファイル: {result.metadata.filename}")
            if result.metadata.page_number:
                source_info.append(f"ページ: {result.metadata.page_number}")

            source_str = " | ".join(source_info) if source_info else "不明"

            context_parts.append(
                f"[{i}] ({source_str})\n{result.content}"
            )

        return "\n\n---\n\n".join(context_parts)

    def _build_messages(
        self,
        context: str,
        question: str,
        chat_history: list[ChatMessage] | None = None,
    ) -> list[SystemMessage | HumanMessage | AIMessage]:
        """LLMに送信するメッセージを構築"""
        messages: list[SystemMessage | HumanMessage | AIMessage] = [
            SystemMessage(content=SYSTEM_PROMPT.format(context=context))
        ]

        if chat_history:
            for msg in chat_history:
                if msg.role == ChatRole.USER:
                    messages.append(HumanMessage(content=msg.content))
                elif msg.role == ChatRole.ASSISTANT:
                    messages.append(AIMessage(content=msg.content))

        messages.append(HumanMessage(content=question))
        return messages

    async def search_only(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
        filter: SearchFilter | None = None,
    ) -> list[ChatSource]:
        """
        検索のみを実行（LLMは使用しない）

        Args:
            query: 検索クエリ
            top_k: 取得件数
            score_threshold: スコア閾値
            filter: 検索フィルタ

        Returns:
            検索結果のソースリスト
        """
        search_request = SearchRequest(
            query=query,
            top_k=top_k,
            score_threshold=score_threshold,
            filter=filter,
        )
        search_response = await self.search_service.search(search_request)

        return [
            ChatSource(
                document_id=result.document_id,
                chunk_id=result.chunk_id,
                filename=result.metadata.filename,
                title=result.metadata.title,
                chunk_content=result.content[:500],
                page_number=result.metadata.page_number,
                score=result.score,
            )
            for result in search_response.results
        ]
