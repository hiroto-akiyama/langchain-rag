"""FastAPIアプリケーション エントリーポイント"""
import logging
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.router import api_router
from app.config import settings
from app.models.database import Base, engine
from app.services.search_service import SearchService

# 静的ファイルのパス
STATIC_DIR = Path(__file__).parent.parent / "static"


def setup_logging() -> None:
    """ロギングを設定"""
    # ログディレクトリを作成
    log_dir = Path(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # ログフォーマット
    log_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ルートロガーを取得
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

    # 既存のハンドラをクリア
    root_logger.handlers.clear()

    # コンソールハンドラ
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    root_logger.addHandler(console_handler)

    # ファイルハンドラ（日次ローテーション）
    log_file = log_dir / "app.log"
    file_handler = TimedRotatingFileHandler(
        filename=str(log_file),
        when="midnight",
        interval=1,
        backupCount=30,
        encoding="utf-8",
    )
    file_handler.setFormatter(log_format)
    file_handler.suffix = "%Y-%m-%d"
    root_logger.addHandler(file_handler)

    # uvicornのロガーも設定
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error"]:
        uvicorn_logger = logging.getLogger(logger_name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.addHandler(console_handler)
        uvicorn_logger.addHandler(file_handler)


# ロギングを初期化
setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """アプリケーションのライフサイクル管理"""
    # 起動時の処理
    Base.metadata.create_all(bind=engine)

    # Qdrantコレクションの初期化
    search_service = SearchService()
    await search_service.ensure_collection()

    yield

    # 終了時の処理（必要に応じて追加）


app = FastAPI(
    title="社内ドキュメント検索AI",
    description="社内ドキュメントのRAG（Retrieval-Augmented Generation）システム",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORSミドルウェア
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番環境では適切に制限する
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# APIルーターを登録
app.include_router(api_router)

# 静的ファイルのマウント（存在する場合）
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/health", tags=["health"])
async def health_check() -> dict:
    """ヘルスチェックエンドポイント"""
    return {
        "status": "healthy",
        "components": {
            "database": "ok",
            "qdrant": "ok",
        },
    }


@app.get("/", tags=["root"])
async def root() -> FileResponse:
    """ルートエンドポイント - UIを返す"""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    # フォールバック: JSONレスポンス
    from fastapi.responses import JSONResponse
    return JSONResponse({
        "message": "社内ドキュメント検索AI API",
        "version": "1.0.0",
        "docs": "/docs",
    })


@app.get("/api", tags=["root"])
async def api_info() -> dict:
    """API情報エンドポイント"""
    return {
        "message": "社内ドキュメント検索AI API",
        "version": "1.0.0",
        "docs": "/docs",
    }
