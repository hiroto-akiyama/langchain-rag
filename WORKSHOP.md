# LangChain 勉強会資料

## RAGアプリケーション ハンズオン

---

## 目次

1. [はじめに](#1-はじめに)
2. [アーキテクチャ概要](#2-アーキテクチャ概要)
3. [環境構築](#3-環境構築)
4. [LangChainの基礎](#4-langchainの基礎)
5. [本プロジェクトでのLangChain活用](#5-本プロジェクトでのlangchain活用)
6. [ハンズオン演習](#6-ハンズオン演習)
7. [発展的なトピック](#7-発展的なトピック)

---

## 1. はじめに

### 1.1 勉強会の目的

本勉強会では、LangChainを使用した**RAG（Retrieval-Augmented Generation）アプリケーション**の実装を通じて、以下を学びます：

- LangChainの基本概念と主要コンポーネント
- ベクトル検索を活用した文書検索の仕組み
- LLMを使った質問応答システムの構築
- 実務で使えるRAGパターンの理解

### 1.2 RAGとは

RAG（Retrieval-Augmented Generation）は、LLMの回答生成を外部知識で拡張する手法です。

```
┌─────────────────────────────────────────────────────────────┐
│                    RAG のワークフロー                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ユーザーの質問                                              │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────────┐    類似検索     ┌──────────────────┐      　│
│  │  Embedding  │ ──────────────▶ │  Vector Database │       │
│  │  (ベクトル化)│                 │  (Qdrant)        │       │
│  └─────────────┘                 └────────┬─────────┘       │
│                                           │                 │
│                                  関連ドキュメント            │
│                                           │                 │
│                                           ▼                 │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  プロンプト = システム指示 + コンテキスト + 質問        │   │
│  └─────────────────────────────────────────────────────┘    │
│                           │                                 │
│                           ▼                                 │
│                   ┌───────────────┐                         │
│                   │  LLM (GPT-4o) │                         │
│                   └───────┬───────┘                         │
│                           │                                 │
│                           ▼                                 │
│                      回答を生成                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 本アプリケーションの機能

| 機能                     | 説明                                             |
| ------------------------ | ------------------------------------------------ |
| ドキュメントアップロード | PDF/Word/テキスト/Markdownファイルをアップロード |
| 自動チャンキング         | ドキュメントを適切なサイズに分割                 |
| ベクトル検索             | 自然言語クエリによる類似検索                     |
| RAG質問応答              | 文書を参照した回答生成                           |
| ストリーミング応答       | リアルタイムでの回答出力                         |

---

## 2. アーキテクチャ概要

### 2.1 システム構成図

```
┌──────────────────────────────────────────────────────────────────────┐
│                           クライアント層                              │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  フロントエンド (static/index.html)                             │  │
│  │  - Bootstrap 5 UI                                              │  │
│  │  - Fetch API でバックエンドと通信                                │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│                           API層 (FastAPI)                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │ /documents   │  │ /search      │  │ /chat        │               │
│  │ ドキュメント  │  │ ベクトル検索  │  │ RAG質問応答   │               │
│  └──────────────┘  └──────────────┘  └──────────────┘               │
└──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│                       サービス層 (Business Logic)                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐   │
│  │ DocumentService │  │ EmbeddingService│  │ RAGService          │   │
│  │ ドキュメント処理  │  │ ベクトル生成     │  │ LLM推論             │   │
│  └─────────────────┘  └─────────────────┘  └─────────────────────┘   │
│  ┌─────────────────┐  ┌─────────────────────────────────────────┐   │
│  │ SearchService   │  │ Utilities (TextSplitter, FileParser)    │   │
│  │ Qdrant検索      │  │ テキスト分割、ファイル解析                 │   │
│  └─────────────────┘  └─────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│                          外部サービス層                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │ PostgreSQL   │  │ Qdrant       │  │ OpenAI API   │               │
│  │ メタデータ    │  │ ベクトルDB    │  │ Embedding/LLM│               │
│  └──────────────┘  └──────────────┘  └──────────────┘               │
└──────────────────────────────────────────────────────────────────────┘
```

### 2.2 ディレクトリ構造

```
app/
├── api/                    # APIエンドポイント
│   ├── router.py          # ルーター集約
│   ├── documents.py       # ドキュメント管理
│   ├── search.py          # ベクトル検索
│   └── chat.py            # RAG質問応答
├── models/                # データモデル (SQLAlchemy)
│   ├── database.py        # DB接続設定
│   ├── document.py        # Document モデル
│   └── chunk.py           # Chunk モデル
├── schemas/               # リクエスト/レスポンス (Pydantic)
│   ├── document.py
│   ├── search.py
│   └── chat.py
├── services/              # ビジネスロジック ★LangChain使用
│   ├── document_service.py
│   ├── embedding_service.py  ← OpenAIEmbeddings
│   ├── search_service.py
│   └── rag_service.py        ← ChatOpenAI
├── utils/                 # ユーティリティ
│   ├── text_splitter.py   # チャンキング
│   └── file_parser.py     # ファイル解析
├── config.py              # 設定管理
└── main.py                # エントリーポイント
```

---

## 3. 環境構築

### 3.1 前提条件

- Docker / Docker Compose
- OpenAI API キー

### 3.2 セットアップ手順

```bash
# 1. リポジトリのクローン
git clone https://github.com/hiroto-akiyama/langchain-rag
cd langchain-rag

# 2. 環境変数の設定
cp .env.sample .env

# 3. .envファイルを編集してOpenAI APIキーを設定
OPENAI_API_KEY=sk-xxxxx

# 4. Docker Composeで起動
docker compose up -d

# 5. アプリケーションにアクセス
# http://localhost:8000
```

### 3.3 主要な環境変数

| 変数名                   | 説明                 | デフォルト値             |
| ------------------------ | -------------------- | ------------------------ |
| `OPENAI_API_KEY`         | OpenAI APIキー       | (必須)                   |
| `OPENAI_EMBEDDING_MODEL` | 埋め込みモデル       | `text-embedding-3-small` |
| `OPENAI_CHAT_MODEL`      | チャットモデル       | `gpt-4o-mini`            |
| `QDRANT_URL`             | QdrantのURL          | `http://qdrant:6333`     |
| `DATABASE_URL`           | PostgreSQL接続文字列 | (Docker Compose設定)     |

---

## 4. LangChainの基礎

### 4.1 LangChainとは

LangChainは、LLM（大規模言語モデル）を活用したアプリケーション開発のためのフレームワークです。

**主要コンポーネント:**

| コンポーネント         | 説明                          |
| ---------------------- | ----------------------------- |
| **LLMs / Chat Models** | GPT-4、Claude等のLLMとの対話  |
| **Embeddings**         | テキストをベクトルに変換      |
| **Prompts**            | プロンプトテンプレートの管理  |
| **Chains**             | 複数の処理をつなげる          |
| **Agents**             | LLMが動的にツールを選択・実行 |

### 4.2 本プロジェクトで使用するLangChain機能

```python
# 使用するパッケージ
langchain==0.3.0
langchain-openai==0.2.0
langchain-community==0.3.0
```

| 機能           | クラス                                       | 用途                 |
| -------------- | -------------------------------------------- | -------------------- |
| チャットモデル | `ChatOpenAI`                                 | 質問応答の生成       |
| 埋め込み       | `OpenAIEmbeddings`                           | テキストのベクトル化 |
| メッセージ     | `SystemMessage`, `HumanMessage`, `AIMessage` | 会話履歴の管理       |
| プロンプト     | `ChatPromptTemplate`                         | プロンプトの構築     |

---

## 5. 本プロジェクトでのLangChain活用

### 5.1 埋め込みサービス (EmbeddingService)

**ファイル:** `app/services/embedding_service.py`

埋め込み（Embedding）は、テキストを数値ベクトルに変換する処理です。意味的に近いテキストは、ベクトル空間上でも近い位置になります。

```python
from langchain_openai import OpenAIEmbeddings

class EmbeddingService:
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        self.api_key = api_key
        self.model = model
        self._embeddings: OpenAIEmbeddings | None = None

    @property
    def embeddings(self) -> OpenAIEmbeddings:
        """遅延初期化パターン"""
        if self._embeddings is None:
            self._embeddings = OpenAIEmbeddings(
                api_key=self.api_key,
                model=self.model,
            )
        return self._embeddings

    async def embed_text(self, text: str) -> list[float]:
        """単一テキストをベクトル化"""
        return await self.embeddings.aembed_query(text)

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """複数テキストを一括ベクトル化"""
        return await self.embeddings.aembed_documents(texts)
```

**ポイント:**

- `OpenAIEmbeddings`: LangChainのOpenAI埋め込みクラス
- `aembed_query()`: 検索クエリのベクトル化（非同期）
- `aembed_documents()`: 複数ドキュメントの一括ベクトル化（非同期）
- 遅延初期化で初期化コストを削減

### 5.2 RAGサービス (RAGService)

**ファイル:** `app/services/rag_service.py`

RAGサービスは、検索結果をコンテキストとしてLLMに渡し、回答を生成します。

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# システムプロンプト
SYSTEM_PROMPT = """あなたは社内ドキュメントに基づいて質問に回答するアシスタントです。

以下のコンテキスト情報のみを使用して回答してください。
コンテキストに含まれない情報については「その情報は見つかりませんでした」と回答してください。

## コンテキスト
{context}

## 回答のガイドライン
- 簡潔かつ正確に回答する
- 参照した情報源があれば明示する
- 不明な点は推測せず、わからないと伝える
"""

class RAGService:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.model = model
        self._llm: ChatOpenAI | None = None

    @property
    def llm(self) -> ChatOpenAI:
        """ChatOpenAIの遅延初期化"""
        if self._llm is None:
            self._llm = ChatOpenAI(
                api_key=self.api_key,
                model=self.model,
                temperature=0.7,
            )
        return self._llm

    def build_context(self, search_results: list[SearchResult]) -> str:
        """検索結果からコンテキストを構築"""
        context_parts = []
        for i, result in enumerate(search_results, 1):
            context_parts.append(
                f"[{i}] {result.document_title}\n{result.content}"
            )
        return "\n\n---\n\n".join(context_parts)

    async def chat(
        self,
        question: str,
        search_results: list[SearchResult],
        history: list[ChatMessage] | None = None,
    ) -> ChatResponse:
        """質問に対する回答を生成"""
        # コンテキストを構築
        context = self.build_context(search_results)

        # メッセージリストを構築
        messages: list[SystemMessage | HumanMessage | AIMessage] = [
            SystemMessage(content=SYSTEM_PROMPT.format(context=context))
        ]

        # 会話履歴があれば追加
        if history:
            for msg in history:
                if msg.role == "user":
                    messages.append(HumanMessage(content=msg.content))
                else:
                    messages.append(AIMessage(content=msg.content))

        # 現在の質問を追加
        messages.append(HumanMessage(content=question))

        # LLMで回答を生成
        response = await self.llm.ainvoke(messages)

        return ChatResponse(
            answer=response.content,
            sources=[r.document_title for r in search_results],
        )
```

**ポイント:**

- `ChatOpenAI`: LangChainのOpenAIチャットモデルクラス
- `SystemMessage`: システム指示（役割設定、コンテキスト）
- `HumanMessage`: ユーザーの発言
- `AIMessage`: AIの応答（会話履歴用）
- `ainvoke()`: 非同期でLLMを実行

### 5.3 ストリーミング応答

ストリーミングにより、回答をリアルタイムで出力できます。

```python
async def chat_stream(
    self,
    question: str,
    search_results: list[SearchResult],
    history: list[ChatMessage] | None = None,
) -> AsyncGenerator[ChatStreamChunk, None]:
    """ストリーミングで回答を生成"""
    context = self.build_context(search_results)
    messages = self._build_messages(context, question, history)

    # astream()でストリーミング
    async for chunk in self.llm.astream(messages):
        content = chunk.content
        if isinstance(content, str) and content:
            yield ChatStreamChunk(content=content, done=False)

    yield ChatStreamChunk(content="", done=True)
```

**ポイント:**

- `astream()`: ストリーミング出力（AsyncGenerator）
- Server-Sent Events (SSE) でフロントエンドに配信

### 5.4 処理フローの全体像

```
┌───────────────────────────────────────────────────────────────────────┐
│                    ドキュメントインデックス処理                         │
├───────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  PDF/Word/TXT ──▶ FileParser ──▶ TextSplitter ──▶ EmbeddingService   │
│                    (テキスト抽出)   (チャンキング)    (ベクトル化)       │
│                                                           │           │
│                                                           ▼           │
│                                       PostgreSQL ◀── SearchService   │
│                                       (メタデータ)     (Qdrant登録)    │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────┐
│                       RAG質問応答処理                                  │
├───────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  質問 ──▶ EmbeddingService ──▶ SearchService ──▶ RAGService          │
│           (クエリベクトル化)     (類似検索)        (回答生成)           │
│                                     │                │                │
│                                     ▼                ▼                │
│                                  Qdrant          ChatOpenAI           │
│                                (ベクトル検索)      (LLM推論)           │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

---

## 6. ハンズオン演習

### 演習1: アプリケーションを動かしてみよう

**目標:** システムの基本動作を理解する

1. Docker Composeでアプリケーションを起動
2. ブラウザで `http://localhost:8000` にアクセス
3. サンプルドキュメント（PDF/テキスト）をアップロード
4. 検索機能を試す
5. チャット機能で質問する

### 演習2: Embeddingを理解しよう

**目標:** テキストのベクトル化を体験する

```python
# Python REPLまたはJupyter Notebookで実行

from langchain_openai import OpenAIEmbeddings
import os

# APIキー設定
embeddings = OpenAIEmbeddings(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="text-embedding-3-small"
)

# テキストをベクトル化
texts = [
    "Pythonは人気のプログラミング言語です",
    "JavaScriptはWeb開発で使われます",
    "機械学習はAIの一分野です",
]

vectors = embeddings.embed_documents(texts)

# ベクトルの次元数を確認
print(f"ベクトル次元数: {len(vectors[0])}")  # 1536次元

# 類似度計算（コサイン類似度）
import numpy as np

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# "Python"と"JavaScript"の類似度
sim_01 = cosine_similarity(vectors[0], vectors[1])
print(f"Python vs JavaScript: {sim_01:.4f}")

# "Python"と"機械学習"の類似度
sim_02 = cosine_similarity(vectors[0], vectors[2])
print(f"Python vs 機械学習: {sim_02:.4f}")
```

### 演習3: RAGの仕組みを理解しよう

**目標:** コンテキストを使った回答生成を体験する

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# LLMを初期化
llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o-mini",
    temperature=0.7
)

# コンテキストなしの質問
messages_no_context = [
    HumanMessage(content="ABC株式会社の設立年はいつですか？")
]

response1 = llm.invoke(messages_no_context)
print("コンテキストなし:")
print(response1.content)

# コンテキストありの質問（RAG）
context = """
## 会社概要
ABC株式会社は2015年4月に東京都渋谷区で設立されました。
主な事業内容はソフトウェア開発とITコンサルティングです。
従業員数は2024年現在で150名です。
"""

messages_with_context = [
    SystemMessage(content=f"""以下のコンテキストに基づいて回答してください。

{context}
"""),
    HumanMessage(content="ABC株式会社の設立年はいつですか？")
]

response2 = llm.invoke(messages_with_context)
print("\nコンテキストあり (RAG):")
print(response2.content)
```

### 演習4: ストリーミング出力を試そう

**目標:** リアルタイム出力を体験する

```python
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

async def stream_example():
    llm = ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini"
    )

    messages = [HumanMessage(content="Pythonの特徴を3つ教えてください")]

    print("ストリーミング出力:")
    async for chunk in llm.astream(messages):
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print()  # 改行

# 実行
asyncio.run(stream_example())
```

### 演習5: APIエンドポイントを叩いてみよう

**目標:** REST APIの動作を確認する

```bash
# ドキュメント一覧を取得
curl http://localhost:8000/api/documents

# 検索を実行
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "会社概要", "limit": 5}'

# RAG質問応答
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "このシステムの主な機能は何ですか？"}'

# ストリーミング（curlで確認）
curl -N http://localhost:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "Pythonについて教えてください"}'
```

---

## 7. 発展的なトピック

### 7.1 チャンキング戦略

テキストの分割方法は検索精度に大きく影響します。

| 戦略               | 説明                           | 適用場面                 |
| ------------------ | ------------------------------ | ------------------------ |
| 固定長分割         | 一定の文字数で分割             | シンプルなテキスト       |
| オーバーラップ     | 前後のチャンクと重複を持たせる | 文脈の保持が必要な場合   |
| セマンティック分割 | 意味のまとまりで分割           | 構造化されたドキュメント |
| 階層的分割         | 見出しごとに分割               | マニュアル、仕様書       |

**本プロジェクトの実装:** `app/utils/text_splitter.py`

```python
class TextSplitter:
    def __init__(
        self,
        chunk_size: int = 1000,      # チャンクサイズ
        chunk_overlap: int = 200,     # オーバーラップ
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
```

### 7.2 Embeddingモデルの選択

| モデル                   | 次元数 | 特徴                 |
| ------------------------ | ------ | -------------------- |
| `text-embedding-3-small` | 1536   | コスト効率が良い     |
| `text-embedding-3-large` | 3072   | より高精度           |
| `text-embedding-ada-002` | 1536   | 旧モデル（互換性用） |

### 7.3 プロンプトエンジニアリング

RAGの品質はプロンプト設計に大きく依存します。

**良いプロンプトの要素:**

- 明確な役割定義
- コンテキストの使用方法の指示
- 回答形式のガイドライン
- 不明な場合の対処法

```python
SYSTEM_PROMPT = """あなたは社内ドキュメントに基づいて質問に回答するアシスタントです。

## ルール
1. コンテキストに含まれる情報のみを使用する
2. 情報がない場合は「見つかりませんでした」と回答
3. 出典を明示する

## コンテキスト
{context}
"""
```

### 7.4 今後学ぶべきトピック

1. **LangChain Expression Language (LCEL)**
   - より宣言的な方法でチェーンを構築

2. **Agents**
   - LLMが動的にツールを選択・実行

3. **LangSmith**
   - LLMアプリケーションのデバッグ・監視

4. **LangGraph**
   - 複雑なワークフローの構築

5. **ハイブリッド検索**
   - ベクトル検索 + キーワード検索の組み合わせ

---

## 参考リソース

### 公式ドキュメント

- [LangChain Documentation](https://python.langchain.com/docs/)
- [LangChain API Reference](https://api.python.langchain.com/)
- [OpenAI API Documentation](https://platform.openai.com/docs/)

### 本プロジェクト関連

- `SPECIFICATION.md` - システム仕様書
- `README.md` - セットアップ手順
- `tests/` - テストコードを読んで動作を理解

### 推奨学習パス

1. 本ハンズオンでRAGの基礎を理解
2. LangChain公式チュートリアルを実施
3. 自分のユースケースでRAGアプリを構築
4. Agents、LangGraphへ進む

---

## 付録: 用語集

| 用語                | 説明                                                                      |
| ------------------- | ------------------------------------------------------------------------- |
| **RAG**             | Retrieval-Augmented Generation。外部知識を検索してLLMの回答を拡張する手法 |
| **Embedding**       | テキストを数値ベクトルに変換すること                                      |
| **Vector Database** | ベクトルを保存し、類似検索ができるデータベース（例: Qdrant）              |
| **Chunk**           | ドキュメントを分割した単位                                                |
| **Context**         | LLMに渡す参照情報                                                         |
| **Token**           | LLMが処理するテキストの最小単位                                           |
| **Prompt**          | LLMへの指示文                                                             |
| **Temperature**     | 出力のランダム性を制御するパラメータ（0-2）                               |

---

**Happy Coding!**
