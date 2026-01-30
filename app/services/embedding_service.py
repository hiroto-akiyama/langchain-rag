from langchain_openai import OpenAIEmbeddings

from app.config import settings


class EmbeddingService:
    """埋め込みベクトル生成サービス"""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ):
        self.api_key = api_key or settings.openai_api_key
        self.model = model or settings.openai_embedding_model
        self._embeddings: OpenAIEmbeddings | None = None

    @property
    def embeddings(self) -> OpenAIEmbeddings:
        """OpenAIEmbeddingsインスタンスを取得（遅延初期化）"""
        if self._embeddings is None:
            self._embeddings = OpenAIEmbeddings(
                api_key=self.api_key,
                model=self.model,
            )
        return self._embeddings

    async def create_embedding(self, text: str) -> list[float]:
        """
        単一テキストの埋め込みベクトルを生成

        Args:
            text: 埋め込みを生成するテキスト

        Returns:
            埋め込みベクトル
        """
        if not text or not text.strip():
            raise ValueError("テキストが空です")

        return await self.embeddings.aembed_query(text)

    async def create_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        複数テキストの埋め込みベクトルを一括生成

        Args:
            texts: 埋め込みを生成するテキストのリスト

        Returns:
            埋め込みベクトルのリスト
        """
        if not texts:
            return []

        valid_texts = [t if t and t.strip() else " " for t in texts]
        return await self.embeddings.aembed_documents(valid_texts)

    def create_embedding_sync(self, text: str) -> list[float]:
        """
        単一テキストの埋め込みベクトルを同期的に生成

        Args:
            text: 埋め込みを生成するテキスト

        Returns:
            埋め込みベクトル
        """
        if not text or not text.strip():
            raise ValueError("テキストが空です")

        return self.embeddings.embed_query(text)

    def create_embeddings_sync(self, texts: list[str]) -> list[list[float]]:
        """
        複数テキストの埋め込みベクトルを同期的に一括生成

        Args:
            texts: 埋め込みを生成するテキストのリスト

        Returns:
            埋め込みベクトルのリスト
        """
        if not texts:
            return []

        valid_texts = [t if t and t.strip() else " " for t in texts]
        return self.embeddings.embed_documents(valid_texts)

    @property
    def vector_size(self) -> int:
        """ベクトルサイズを取得"""
        return settings.qdrant_vector_size
