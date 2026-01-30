FROM python:3.12.12-slim

# -----------------------------
# 基本設定
# -----------------------------
ENV DEBIAN_FRONTEND=noninteractive

# -----------------------------
# OSパッケージ
#  - Node.js (Claude Code用)
#  - build-essential（将来のpip native拡張対策）
# -----------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    gnupg \
    build-essential \
    git \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# 非 root ユーザー作成
# -----------------------------
RUN useradd -m -s /bin/bash python
USER python
ENV HOME=/home/python
WORKDIR ${HOME}/workspace

# pip / npm の bin を PATH に追加
ENV PATH="/home/python/.local/bin:/home/python/.npm-global/bin:${PATH}"

# npm のグローバルインストール先をユーザー領域に
RUN npm config set prefix "/home/python/.npm-global"

# -----------------------------
# Claude Code インストール
# -----------------------------
RUN npm install -g @anthropic-ai/claude-code

# 確認（失敗するとビルドが止まる）
RUN claude --version
