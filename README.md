# LangChain and Ragアプリケーション

## 開発環境の構成（docker-compose）

- app(python)：資料の取り込み・検索API・RAG推論の実行
- qdrant: ベクトル検索（埋め込みベクトルの保存・近傍探索）
- postgres: ドキュメントのメタデータ（権限・出典・ファイルパス・チャンク対応表など）
- （任意）redis: キャッシュ、ジョブキュー、セッション、レート制御
- （任意）minio: ファイル原本（PDFなど）保管をS3互換でローカル実現

## 起動コマンド

※コマンドラインからdocker-composeでサーバーを起動する場合のコマンドです。
通常開発時はVSCodeからDevContainerで起動するので使用しません

- 基本（api + qdrant + postgres）

```
docker compose up -d --build
```

- Redisも使う（プロファイル指定）

```
docker compose --profile with-redis up -d --build
```

- MinIOも使う

```
docker compose --profile with-minio up -d --build
```

## 初回インストール

コンテナを起動したら必要なPythonライブラリをインストールする

```
pip install --no-cache-dir -r requirements.txt
```

## OpenAI APIキー取得

https://platform.openai.com/api-keys
にアクセスしてログインします。
左のメニューから「API keys」を選択し、右上の「＋Create new secret key」をクリックします。
NameにAPIキーの名前を入れてCreate secret keyボタンを押下するとAPIキーが発行されるのでコピーして.envに貼り付けてください。

## メモ

本番環境を作る場合にはDockerfleに以下の設定を追加する必要があります。
※PATHに/home/python/.local/binを追加する。
appコンテナを起動するDockerfielにてuvicornの起動までする行う場合に必要となります。

```
ENV PATH="/home/python/.local/bin:${PATH}"
```
