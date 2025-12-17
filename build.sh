#!/bin/bash
set -e

# 1. Rust環境(rustup)のインストール
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# 2. Rust (Wasm) のビルド
# "cd rust-shader-tools" は不要です。pnpm exec はルートから実行できます。
# --out-dir を指定して出力先を固定するとより確実ですが、
# 元の構成(rust-shader-tools/pkg)に合わせるため、ディレクトリ移動は維持しつつ
# コマンド実行方法を変えます。

cd rust-shader-tools
# pnpm exec を使うと node_modules/.bin のパスを自動解決してくれます
pnpm exec wasm-pack build --target web
cd ..

# 3. フロントエンドのビルド
pnpm run build
