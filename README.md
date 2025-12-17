# WebGPU Ray Tracer

![WebGPU](https://img.shields.io/badge/WebGPU-Enabled-green)
![TypeScript](https://img.shields.io/badge/TypeScript-5.0-blue)
![Rust](https://img.shields.io/badge/Rust-WASM-orange)
![Vite](https://img.shields.io/badge/Bundler-Vite-purple)

**WebGPU + TypeScript + WASM (Rust)** で実装された、ブラウザ上で動作する高性能なパストレーシング・レンダラーです。
計算負荷の高い BVH（Bounding Volume Hierarchy）構築やシーン解析を Rust (WASM) で行い、レンダリング自体は WebGPU の Compute Shader を用いて GPU 上で並列実行します。
また、分散レンダリングに対応しており、複数のブラウザを同時に使用することで、より高速なレンダリングが可能です。

🔗 **Demo(分散処理は未対応):** [https://kokutoupan.github.io/webgpu-raytracer/](https://kokutoupan.github.io/webgpu-raytracer/)

## ✨ 主な機能 (Features)

- **リアルタイム・パストレーシング**: WebGPU Compute Shader によるハードウェアアクセラレーション。
- **高度な加速構造**: 2 段階 BVH (TLAS / BLAS) を採用し、動的なシーンや多数のオブジェクトを高速に処理。
- **異種フォーマット対応**: `.obj`, `.glb` (glTF), `.vrm` ファイルのドラッグ＆ドロップ読み込みに対応。
- **マテリアル表現**:
  - Lambertian (拡散反射)
  - Metallic (金属)
  - Dielectric (ガラス/屈折)
  - Emissive (発光体)
- **テクスチャマッピング**: `texture_2d_array` を使用した効率的なテクスチャサンプリング。
- **カメラ制御**: 被写界深度 (Depth of Field) 対応のカメラ。
- **プログレッシブ・レンダリング**: フレームを蓄積してノイズを低減 (Accumulation Buffer)。

## 🏗 アーキテクチャ (Architecture)

このプロジェクトは、パフォーマンスを最大化するために以下の役割分担を行っています。

1.  **Frontend (TypeScript)**:
    - UI 制御、メインループ、WebGPU API の管理 (`src/main.ts`, `src/renderer.ts`)。
    - ユーザー入力に基づく設定（解像度、SPP、深度など）の動的更新。
      - SPP,DPS の更新は、shader のコンパイルが必要。
2.  **Core Logic (Rust -> WASM)**:
    - `rust-shader-tools/` ディレクトリ配下。
    - 3D モデルのパース (gltf, obj)。
    - BVH (TLAS/BLAS) の構築と平坦化。
    - シーンデータの GPU バッファ用レイアウトへの変換。
3.  **Rendering (WGSL)**:
    - `src/shader.wgsl`。
    - 交差判定 (Ray-Box, Ray-Triangle)。
    - パストレーシングのロジック (モンテカルロ積分)。

### データフロー

処理の流れは以下の通りです：

1.  **User Input / File** → **Main (TypeScript)**
    - ユーザーによるファイルドロップや設定変更を受け付けます。
2.  **Main** → **Rust Core (WASM)**
    - 読み込んだ 3D データの解析と BVH の構築を依頼します。
3.  **Rust Core (WASM)** → **Main**
    - GPU 用に最適化されたフラットなバッファデータ（頂点、インデックス、ノード情報など）を返します。
4.  **Main** → **WebGPU Buffers**
    - 受け取ったデータを GPU メモリ（Storage Buffer）に転送します。
5.  **WebGPU Buffers** → **Compute Shader (WGSL)**
    - シェーダーがバッファとテクスチャをバインドして計算を実行します。
6.  **Compute Shader** → **Canvas**
    - 計算結果を蓄積し、トーンマッピングを行って画面に描画します。

## 🚀 開発セットアップ (Development)

開発には [Node.js](https://nodejs.org/) と [Rust](https://www.rust-lang.org/) (および `wasm-pack`) が必要です。

### 1. 前提ツールのインストール

```bash
# Rustのwasmビルドツール
cargo install wasm-pack
```

### 2. リポジトリのクローンと依存関係のインストール

```Bash
git clone https://github.com/kokutoupan/webgpu-raytracer.git
cd webgpu-raytracer

# パッケージのインストール
pnpm install
# または npm install
```

### 3. Rust (WASM) のビルド

Rust 側のコードを変更した場合は、WASM の再ビルドが必要です。

```Bash
cd rust-shader-tools
wasm-pack build --target web
cd ..
```

### 4. 開発サーバーの起動

```Bash
pnpm dev
```

ブラウザで http://localhost:5173 (または表示される URL) にアクセスしてください。WebGPU 対応ブラウザ（Chrome, Edge, Firefox Nightly など）が必要です。

### 🎮 操作方法 (Controls)

- Scene Select: プリセットシーンの切り替え。
- File Input: .obj や .glb ファイルを選択してローカルモデルを表示。
- Resolution: レンダリング解像度。変更するとバッファが再確保されます。
- Max Depth: レイの最大反射回数。
- SPP / Frame: 1 フレームあたりのサンプル数（Samples Per Pixel）。
- Update Interval: アニメーション等の更新間隔。
- Animation: glTF アニメーションが含まれる場合、インデックスで選択可能。

### 📂 ディレクトリ構成 (File Structure)

新規開発者向けの主要ファイル解説です。

- src/

  - main.ts: エントリーポイント。DOM 操作とレンダリングループの制御。
  - renderer.ts: WebGPURenderer クラス。WebGPU デバイスの初期化、パイプライン作成、バッファ管 理、描画コマンドの発行。
  - world-bridge.ts: TypeScript と WASM のブリッジ。WASM メモリからのデータ取得を担当。
  - shader.wgsl: レイトレーシングの全ロジックを含む Compute Shader。

- rust-shader-tools/
  - src/bvh/: BVH 構築ロジック (tlas.rs, blas.rs)。
  - src/scene/: シーン、マテリアル、カメラの定義。
  - src/lib.rs: WASM として公開される API のエントリポイント。

### 🛠 技術スタック詳細

- Languages: TypeScript, Rust, WGSL

- Build Tool: Vite

- Rust Crates: wasm-bindgen, glam (数学), gltf (ローダー), rand

- Web APIs: WebGPU

### License

This project is licensed under the MIT License.
