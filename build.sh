#!/bin/bash
set -e # エラーが発生したら即停止

# Rust環境(rustup)をインストール
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# 環境変数を読み込んで cargo コマンドを使えるようにする
source "$HOME/.cargo/env"

# Rust (Wasm) のビルド
cd rust-shader-tools
wasm-pack build --target web
cd ..

# フロントエンドのビルド
pnpm run build
