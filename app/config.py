import os
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """アプリケーション設定"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # アプリケーション設定
    app_env: str = Field("local", description="環境（local/staging/production）")
    log_level: str = Field("DEBUG", description="ログレベル")
    debug: bool = Field(True, description="デバッグモード")

    # OpenAI設定
    openai_api_key: str = Field("", description="OpenAI APIキー")
    openai_embedding_model: str = Field(
        "text-embedding-3-small",
        description="埋め込みモデル",
    )
    openai_chat_model: str = Field("gpt-4o-mini", description="チャットモデル")

    # Qdrant設定
    qdrant_url: str = Field("http://qdrant:6333", description="Qdrant接続URL")
    qdrant_collection: str = Field("documents", description="Qdrantコレクション名")
    qdrant_vector_size: int = Field(1536, description="ベクトルサイズ")

    # データベース設定
    database_url: str = Field(
        "postgresql+psycopg://rag:rag@postgres:5432/rag",
        description="PostgreSQL接続URL",
    )

    # Redis設定（オプション）
    redis_url: str = Field("redis://redis:6379/0", description="Redis接続URL")
    use_redis: bool = Field(False, description="Redisを使用するか")

    # ログ設定
    log_dir: str = Field("./logs", description="ログディレクトリ")

    # ファイル設定
    upload_dir: str = Field("./uploads", description="アップロードディレクトリ")
    max_file_size: int = Field(50 * 1024 * 1024, description="最大ファイルサイズ（50MB）")
    allowed_extensions: list[str] = Field(
        default_factory=lambda: [".pdf", ".docx", ".doc", ".txt", ".md"],
        description="許可する拡張子",
    )

    # チャンキング設定
    chunk_size: int = Field(1000, description="チャンクサイズ（文字数）")
    chunk_overlap: int = Field(200, description="チャンクオーバーラップ（文字数）")


@lru_cache
def get_settings() -> Settings:
    """設定をキャッシュして取得"""
    return Settings()


settings = get_settings()
