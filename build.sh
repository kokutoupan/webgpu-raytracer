#!/bin/bash
set -e

# 1. Rust環境(rustup)のインストール
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# 2. Rust (Wasm) のビルド
# cd せずに、ルートからパス(rust-shader-tools)を指定して実行します
pnpm exec wasm-pack build rust-shader-tools --target web

# 3. フロントエンドのビルド
pnpm run build
