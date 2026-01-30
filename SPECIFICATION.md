# 社内ドキュメント検索AI・RAGシステム仕様書

## 1. システム概要

### 1.1 目的
社内ドキュメント（PDF、Word、テキスト等）をベクトル化し、自然言語による質問応答（RAG: Retrieval-Augmented Generation）を実現するシステム。

### 1.2 主要機能
- ドキュメントのアップロード・インデックス作成
- 自然言語による類似ドキュメント検索
- RAGによる質問応答（LLMを活用した回答生成）
- ドキュメントメタデータ管理

### 1.3 実装状況

| 機能 | 状況 |
|------|------|
| ドキュメントアップロード | ✅ 実装済み |
| ドキュメント一覧・詳細・削除 | ✅ 実装済み |
| ベクトル検索 | ✅ 実装済み |
| RAG質問応答 | ✅ 実装済み |
| ストリーミング応答 | ✅ 実装済み |
| ヘルスチェック | ✅ 実装済み |
| APIドキュメント（Swagger UI） | ✅ 実装済み |
| フロントエンドUI | ✅ 実装済み |
| ユニットテスト | ✅ 実装済み（68テスト） |

---

## 2. アーキテクチャ構成

```
┌─────────────────────────────────────────────────────────────────┐
│                        クライアント                              │
│                   (ブラウザ / API Client)                        │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI Application                         │
│                        (app:8000)                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ Document    │  │ Search      │  │ RAG                     │  │
│  │ Management  │  │ Service     │  │ Service                 │  │
│  │ API         │  │             │  │ (LangChain + OpenAI)    │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
         │                  │                      │
         ▼                  ▼                      ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐
│  PostgreSQL │    │   Qdrant    │    │     OpenAI API          │
│  (postgres) │    │  (qdrant)   │    │  (text-embedding / GPT) │
│  :5432      │    │  :6333      │    │                         │
│             │    │             │    │                         │
│ メタデータ   │    │ ベクトル     │    │ Embedding + Chat        │
│ 管理        │    │ インデックス  │    │ Completion              │
└─────────────┘    └─────────────┘    └─────────────────────────┘

オプション:
┌─────────────┐    ┌─────────────┐
│   Redis     │    │   MinIO     │
│  (redis)    │    │  (minio)    │
│  :6379      │    │  :9000/9001 │
│             │    │             │
│ キャッシュ   │    │ ファイル     │
│ ジョブキュー │    │ ストレージ   │
└─────────────┘    └─────────────┘
```

### 2.1 コンポーネント一覧

| コンポーネント | 役割 | ポート |
|--------------|------|-------|
| app (FastAPI) | APIサーバー、RAG処理 | 8000 |
| Qdrant | ベクトルDB（埋め込みベクトル保存・検索） | 6333 |
| PostgreSQL | メタデータDB（文書情報、チャンク管理） | 5432 |
| Redis (任意) | キャッシュ、非同期ジョブキュー | 6379 |
| MinIO (任意) | ファイル原本保管（S3互換） | 9000/9001 |

---

## 3. ディレクトリ構造

```
workspace/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPIエントリーポイント ✅
│   ├── config.py               # 設定管理（pydantic-settings） ✅
│   │
│   ├── api/                    # APIエンドポイント ✅
│   │   ├── __init__.py
│   │   ├── router.py           # ルーター集約（/api/v1プレフィックス）
│   │   ├── documents.py        # ドキュメント管理API
│   │   ├── search.py           # 検索API
│   │   └── chat.py             # RAG質問応答API
│   │
│   ├── models/                 # データモデル ✅
│   │   ├── __init__.py
│   │   ├── database.py         # SQLAlchemy設定
│   │   ├── document.py         # Documentモデル
│   │   └── chunk.py            # Chunkモデル
│   │
│   ├── schemas/                # Pydanticスキーマ ✅
│   │   ├── __init__.py
│   │   ├── document.py         # ドキュメントスキーマ
│   │   ├── search.py           # 検索スキーマ
│   │   └── chat.py             # チャットスキーマ
│   │
│   ├── services/               # ビジネスロジック ✅
│   │   ├── __init__.py
│   │   ├── document_service.py # ドキュメント処理
│   │   ├── embedding_service.py# 埋め込み生成
│   │   ├── search_service.py   # ベクトル検索
│   │   └── rag_service.py      # RAG処理
│   │
│   └── utils/                  # ユーティリティ ✅
│       ├── __init__.py
│       ├── text_splitter.py    # テキスト分割
│       └── file_parser.py      # ファイルパーサー
│
├── static/                     # フロントエンドUI ✅
│   └── index.html              # メインページ（Bootstrap 5）
│
├── tests/                      # テスト ✅
│   ├── __init__.py
│   ├── conftest.py             # pytest共通フィクスチャ
│   ├── test_api.py             # APIエンドポイントテスト
│   ├── test_documents.py       # DocumentServiceテスト
│   ├── test_search.py          # SearchServiceテスト
│   ├── test_chat.py            # RAGServiceテスト
│   └── test_utils.py           # ユーティリティテスト
│
├── .devcontainer/
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── .env                        # 環境変数
├── .env.sample
├── requirements.txt
├── README.md
└── SPECIFICATION.md            # 本ファイル
```

---

## 4. API仕様

### 4.0 共通情報

**ベースURL:** `http://localhost:8000`

**APIドキュメント:**
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- OpenAPI JSON: `http://localhost:8000/openapi.json`

#### GET /
ルートエンドポイント

**Response:**
```json
{
  "message": "社内ドキュメント検索AI API",
  "version": "1.0.0",
  "docs": "/docs"
}
```

---

### 4.1 ドキュメント管理API

#### POST /api/v1/documents/upload
ドキュメントをアップロードし、インデックスを作成

**Request:**
```
Content-Type: multipart/form-data

file: <ファイル>
title: string (任意)
tags: string[] (任意)
```

**Response (201 Created):**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "sample.pdf",
  "title": "サンプル文書",
  "file_path": "./uploads/550e8400-e29b-41d4-a716-446655440000_sample.pdf",
  "file_size": 102400,
  "mime_type": "application/pdf",
  "tags": ["sample", "test"],
  "status": "indexed",
  "chunk_count": 15,
  "created_at": "2026-01-22T10:00:00Z",
  "updated_at": "2026-01-22T10:00:00Z"
}
```

**Error Response (400 Bad Request):**
```json
{
  "detail": "サポートされていないファイル形式: sample.exe"
}
```

#### GET /api/v1/documents
ドキュメント一覧取得

**Query Parameters:**
- `page`: int (default: 1)
- `per_page`: int (default: 20)
- `tag`: string (任意)

**Response:**
```json
{
  "items": [...],
  "total": 100,
  "page": 1,
  "per_page": 20
}
```

#### GET /api/v1/documents/{document_id}
ドキュメント詳細取得（チャンク情報を含む）

**Response (200 OK):**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "sample.pdf",
  "title": "サンプル文書",
  "file_path": "./uploads/550e8400-e29b-41d4-a716-446655440000_sample.pdf",
  "file_size": 102400,
  "mime_type": "application/pdf",
  "tags": ["sample"],
  "status": "indexed",
  "chunk_count": 3,
  "created_at": "2026-01-22T10:00:00Z",
  "updated_at": "2026-01-22T10:00:00Z",
  "chunks": [
    {
      "id": "660e8400-e29b-41d4-a716-446655440001",
      "document_id": "550e8400-e29b-41d4-a716-446655440000",
      "content": "チャンク1の内容...",
      "chunk_index": 0,
      "page_number": 1,
      "chunk_metadata": null,
      "created_at": "2026-01-22T10:00:00Z"
    }
  ]
}
```

**Error Response (404 Not Found):**
```json
{
  "detail": "ドキュメントが見つかりません"
}
```

#### DELETE /api/v1/documents/{document_id}
ドキュメント削除（ベクトルインデックスも削除）

**Response:** 204 No Content

**Error Response (404 Not Found):**
```json
{
  "detail": "ドキュメントが見つかりません"
}
```

---

### 4.2 検索API

#### POST /api/v1/search
類似ドキュメント検索

**Request:**
```json
{
  "query": "検索クエリテキスト",
  "top_k": 5,
  "score_threshold": 0.7,
  "filter": {
    "tags": ["tag1", "tag2"],
    "document_ids": ["550e8400-e29b-41d4-a716-446655440000"]
  }
}
```

| パラメータ | 型 | 必須 | 説明 |
|-----------|-----|------|------|
| query | string | ✅ | 検索クエリ（1文字以上） |
| top_k | int | - | 取得件数（1-100、デフォルト: 5） |
| score_threshold | float | - | スコア閾値（0.0-1.0、デフォルト: 0.0） |
| filter | object | - | 検索フィルタ |

**Response (200 OK):**
```json
{
  "results": [
    {
      "document_id": "550e8400-e29b-41d4-a716-446655440000",
      "chunk_id": "660e8400-e29b-41d4-a716-446655440001",
      "content": "マッチしたテキストチャンク...",
      "score": 0.92,
      "metadata": {
        "filename": "sample.pdf",
        "title": "サンプル文書",
        "page_number": 3,
        "tags": ["sample"]
      }
    }
  ],
  "total": 5,
  "query": "検索クエリテキスト"
}
```

---

### 4.3 RAG質問応答API

#### POST /api/v1/chat
RAGによる質問応答

**Request:**
```json
{
  "question": "社内の休暇申請のルールは？",
  "chat_history": [
    {"role": "user", "content": "前の質問"},
    {"role": "assistant", "content": "前の回答"}
  ],
  "top_k": 5,
  "score_threshold": 0.0
}
```

| パラメータ | 型 | 必須 | 説明 |
|-----------|-----|------|------|
| question | string | ✅ | 質問テキスト（1文字以上） |
| chat_history | array | - | 過去の会話履歴 |
| top_k | int | - | 参照するチャンク数（1-20、デフォルト: 5） |
| score_threshold | float | - | スコア閾値（0.0-1.0、デフォルト: 0.0） |

**Response (200 OK):**
```json
{
  "answer": "社内の休暇申請ルールは以下の通りです...",
  "sources": [
    {
      "document_id": "550e8400-e29b-41d4-a716-446655440000",
      "chunk_id": "660e8400-e29b-41d4-a716-446655440001",
      "filename": "就業規則.pdf",
      "title": "就業規則",
      "chunk_content": "参照されたテキスト...",
      "page_number": 15,
      "score": 0.89
    }
  ],
  "question": "社内の休暇申請のルールは？"
}
```

#### POST /api/v1/chat/stream
ストリーミング形式での質問応答（Server-Sent Events）

**Request:** POST /api/v1/chat と同じ

**Response:** `text/event-stream`
```
data: {"content": "社内の", "done": false, "sources": null}

data: {"content": "休暇申請", "done": false, "sources": null}

data: {"content": "ルールは...", "done": false, "sources": null}

data: {"content": "", "done": true, "sources": [...]}
```

---

### 4.4 ヘルスチェック

#### GET /health
```json
{
  "status": "healthy",
  "components": {
    "database": "ok",
    "qdrant": "ok"
  }
}
```

---

## 5. データモデル

### 5.1 PostgreSQL テーブル設計

#### documents テーブル
| カラム | 型 | 説明 |
|-------|-----|------|
| id | UUID | 主キー |
| filename | VARCHAR(255) | ファイル名 |
| title | VARCHAR(500) | タイトル |
| file_path | VARCHAR(1000) | 保存パス |
| file_size | BIGINT | ファイルサイズ |
| mime_type | VARCHAR(100) | MIMEタイプ |
| tags | JSONB | タグ配列 |
| status | VARCHAR(50) | 状態(pending/indexed/failed) |
| chunk_count | INTEGER | チャンク数 |
| created_at | TIMESTAMP | 作成日時 |
| updated_at | TIMESTAMP | 更新日時 |

#### chunks テーブル
| カラム | 型 | 説明 |
|-------|-----|------|
| id | UUID | 主キー |
| document_id | UUID | 外部キー（documents） |
| content | TEXT | チャンクテキスト |
| chunk_index | INTEGER | チャンク順序 |
| page_number | INTEGER | ページ番号 |
| qdrant_point_id | UUID | Qdrant上のID |
| metadata | JSONB | その他メタ情報 |
| created_at | TIMESTAMP | 作成日時 |

### 5.2 Qdrant コレクション設計

**コレクション名:** `documents`

```json
{
  "vectors": {
    "size": 1536,
    "distance": "Cosine"
  },
  "payload_schema": {
    "document_id": "keyword",
    "chunk_id": "keyword",
    "filename": "keyword",
    "tags": "keyword[]",
    "page_number": "integer"
  }
}
```

---

## 6. 処理フロー

### 6.1 ドキュメントインデックス作成フロー

```
1. ファイルアップロード受付
      │
      ▼
2. ファイル保存（ローカル or MinIO）
      │
      ▼
3. テキスト抽出（PDF/Word/テキスト）
      │
      ▼
4. テキスト分割（チャンキング）
   - chunk_size: 1000文字
   - chunk_overlap: 200文字
      │
      ▼
5. 埋め込みベクトル生成
   - OpenAI text-embedding-3-small
      │
      ▼
6. Qdrantへベクトル登録
      │
      ▼
7. PostgreSQLへメタデータ登録
      │
      ▼
8. 完了レスポンス返却
```

### 6.2 RAG質問応答フロー

```
1. 質問受付
      │
      ▼
2. 質問文の埋め込みベクトル生成
      │
      ▼
3. Qdrantで類似チャンク検索
   - top_k件取得
   - スコア閾値でフィルタ
      │
      ▼
4. コンテキスト構築
   - 検索結果をプロンプトに組み込み
      │
      ▼
5. LLMで回答生成
   - GPT-4o / GPT-4o-mini
   - システムプロンプト + コンテキスト + 質問
      │
      ▼
6. 回答 + 参照元情報を返却
```

---

## 7. 環境変数

| 変数名 | 説明 | デフォルト |
|-------|------|-----------|
| APP_ENV | 環境（local/staging/production） | local |
| LOG_LEVEL | ログレベル | DEBUG |
| OPENAI_API_KEY | OpenAI APIキー | - |
| QDRANT_URL | Qdrant接続URL | http://qdrant:6333 |
| QDRANT_COLLECTION | Qdrantコレクション名 | documents |
| DATABASE_URL | PostgreSQL接続URL | postgresql+psycopg://... |
| REDIS_URL | Redis接続URL | redis://redis:6379/0 |
| USE_REDIS | Redisを使用するか | false |

---

## 8. 使用ライブラリ

### 8.1 メインライブラリ

| ライブラリ | バージョン | 用途 |
|-----------|-----------|------|
| FastAPI | 0.115.5 | Webフレームワーク |
| Uvicorn | 0.34.0 | ASGIサーバー |
| LangChain | 0.3.0 | LLM連携フレームワーク |
| langchain-openai | 0.2.0 | OpenAI連携 |
| langchain-community | 0.3.0 | コミュニティ統合 |
| qdrant-client | 1.12.0 | Qdrantクライアント |
| SQLAlchemy | 2.0.36 | ORM |
| psycopg | 3.2.3 | PostgreSQLドライバ |
| Pydantic | 2.10.3 | データバリデーション |

### 8.2 追加推奨ライブラリ

| ライブラリ | 用途 |
|-----------|------|
| pypdf | PDF読み込み |
| python-docx | Word読み込み |
| unstructured | 各種ドキュメント解析 |
| tiktoken | トークンカウント |
| tenacity | リトライ処理 |

---

## 9. セキュリティ考慮事項

- APIキーは環境変数で管理し、コードに含めない
- ファイルアップロードはMIMEタイプ・拡張子のバリデーション必須
- 本番環境ではHTTPS必須
- 必要に応じてAPI認証（JWT等）を追加
- アップロードファイルサイズの上限設定

---

## 10. フロントエンドUI

### 10.1 概要

`static/index.html` にBootstrap 5を使用したシングルページアプリケーションを実装。

**URL:** `http://localhost:8000/`

### 10.2 画面構成

| タブ | 機能 |
|------|------|
| アップロード | ドキュメントのアップロード（ファイル、タイトル、タグ） |
| ドキュメント一覧 | アップロード済みドキュメントの一覧表示・削除 |
| 検索 | ベクトル類似検索（件数・閾値指定可） |
| チャット | RAG質問応答（会話履歴対応、参照元表示） |

### 10.3 使用技術

- Bootstrap 5.3.2（CDN）
- Bootstrap Icons 1.11.1（CDN）
- Vanilla JavaScript（フレームワークなし）

---

## 11. 起動方法

### 11.1 開発サーバー起動

```bash
# アプリケーション起動
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# または
python -m uvicorn app.main:app --reload
```

### 11.2 テスト実行

```bash
# 全テスト実行
pytest tests/ -v

# 特定のテストファイルを実行
pytest tests/test_api.py -v

# カバレッジ付きで実行
pytest tests/ -v --cov=app --cov-report=html
```

### 11.3 必要な外部サービス

アプリケーション起動前に以下のサービスが必要です：

1. **PostgreSQL** - メタデータ保存
2. **Qdrant** - ベクトルインデックス
3. **OpenAI API** - 埋め込み生成・LLM推論

---

## 12. 今後の拡張ポイント

- [ ] ユーザー認証・権限管理
- [ ] ドキュメントごとのアクセス制御
- [ ] バッチ処理（大量ドキュメント一括インデックス）
- [ ] 検索結果のキャッシュ（Redis利用）
- [ ] 会話履歴の永続化
- [ ] 複数LLMプロバイダー対応（Azure OpenAI、Anthropic等）
- [ ] UI（管理画面、チャットUI）
- [ ] ヘルスチェックの詳細化（実際の接続確認）
- [ ] ログ出力の構造化
- [ ] レート制限
