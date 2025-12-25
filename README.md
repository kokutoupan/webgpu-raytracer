# WebGPU Ray Tracer (Next-Gen Path Tracer)

![WebGPU](https://img.shields.io/badge/WebGPU-Enabled-green)
![TypeScript](https://img.shields.io/badge/TypeScript-5.0-blue)
![Rust](https://img.shields.io/badge/Rust-WASM-orange)
![Vite](https://img.shields.io/badge/Bundler-Vite-purple)

**WebGPU + TypeScript + WASM (Rust)** で実装された、ブラウザ上で動作する高性能なパストレーシング・レンダラーです。
計算負荷の高い BVH（Bounding Volume Hierarchy）構築やシーン解析を Rust (WASM) で行い、レンダリング自体は WebGPU の Compute Shader を用いて GPU 上で並列実行します。
また、分散レンダリングに対応しており、複数のブラウザを同時に使用することで、より高速なレンダリングが可能です。

🔗 **Demo(分散処理は未対応):** [https://kokutoupan.github.io/webgpu-raytracer/](https://kokutoupan.github.io/webgpu-raytracer/)

## ✨ 主な機能 (Features)

### 🚀 先進機能

- **ReSTIR (Reservoir-based Spatiotemporal Importance Resampling)**: 数千万ポリゴンや多数の光源を効率的にサンプリングし、ノイズを劇的に低減。
- **TAA (Temporal Anti-Aliasing)**: サブピクセル・ジッタリングと時間軸の蓄積により、滑らかなエッジと安定した静止画を実現。
- **分散レンダリング (Distributed Rendering)**: WebRTC を使用。複数のブラウザ、複数のデバイスをクラスター化して並列レンダリング。
- **アダプティブ・プログレッシブ・リファインメント**: 放置するほど画像が美しくなり、最大 10,000 フレーム以上の極めて高精度な収束に対応。

### 🎨 レンダリング表現

- **フル PBR パストレーシング**: Lambertian (拡散), Metallic (金属), Dielectric (屈折/ガラス), Emissive (発光) をサポート。
- **動的加速構造**: 高速な 2 段階 BVH (TLAS / BLAS) により、数十万トライアングルのシーンをリアルタイム処理。
- **拡張モデルサポート**: `.obj`, `.glb` (glTF), `.vrm` ファイルの読み込み。

### 🎬 ポストプロセス & ツール

- **高度なデノイジング**: バイリニア補間、バイラテラルフィルタ、強力な Firefly 除去ロジックを搭載。
- **ビデオ・レコーダー**: 収束したフレームをネイティブ WebM 形式（VP9 互換）で直接録画・ダウンロード可能。
- **トーンマッピング**: ACES フィルムライクなトーンマッピングとガンマ補正。

---

## 🏗 アーキテクチャ (Architecture)

1.  **TypeScript (Frontend)**:
    - WebGPU API の管理、レンダリングループ、PostProcess 連携。
    - WebRTC / Signaling による分散処理のオーケストレーション。
2.  **Rust (WASM Core)**:
    - 超高速な BVH 構築、ジオメトリ平坦化、アニメーション計算。
    - 光源データの事前計算と最適化。
3.  **WGSL (Compute Shaders)**:
    - `Raytracer.wgsl`: コアとなる交差判定とパストレーシング。
    - `PostProcess.wgsl`: TAA、デノイズ、トーンマッピング。

---

## 🔧 開発 (Development)

### インストール & ビルド

```bash
# 1. 依存関係のインストール
pnpm install

# 2. Rust (WASM) のビルド (rust/-shader-tools内)
cd rust-shader-tools
wasm-pack build --target web --out-dir ../src/wasm
cd ..

# 3. 開発サーバー起動
pnpm dev
```

---

## 🎮 画面の見方

- **Scene**: プリセットシーンの切り替え。
- **SPP / Frame**: 1 フレームあたりの光線サンプリング数。高いほど 1 フレームの品質が上がります（重くなります）。
- **Samples**: 現在までに累積されたトータルのサンプル数。放置すると増え続け、画像がクリアになります。
- **Role**: `Host` で他クライアントを繋いで分散レンダリングを開始。

---

## 🛠 技術スタック

- **Graphics API**: [WebGPU](https://gpuweb.github.io/gpuweb/)
- **Programming**: [TypeScript](https://www.typescriptlang.org/), [Rust](https://www.rust-lang.org/) (WASM)
- **Math**: GLAM (Rust), gl-matrix (TS)
- **Bundler**: Vite
- **Networking**: WebRTC (SimplePeer), Socket.io (Signaling)

## License

This project is licensed under the MIT License.
