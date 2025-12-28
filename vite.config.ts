import { defineConfig } from "vite";
import wasm from "vite-plugin-wasm";
import topLevelAwait from "vite-plugin-top-level-await";
import dotenv from "dotenv";
import path from "path";
import basicSsl from "@vitejs/plugin-basic-ssl"; // 追加

dotenv.config({ path: path.resolve(__dirname, ".env") });

export default defineConfig(({ mode }) => {
  const isCloudflare = !(process.env.CF_PAGES_N === "1");

  // 'vite dev' (開発モード) かどうかを判定
  const isDev = mode === "development";

  return {
    base: isCloudflare ? "/" : "webgpu-raytracer/",
    build: {
      outDir: isCloudflare ? "dist" : "docs",
    },
    plugins: [
      wasm(),
      topLevelAwait(),
      // 開発モードの時だけSSLプラグインを有効化
      // 本番ビルドには影響しません
      ...(isDev ? [basicSsl()] : []),
    ],
    // 開発サーバーの設定（ここがWindows接続のキモです）
    server: {
      host: true, // LAN内の他のPC（Windows）からのアクセスを許可
      https: true, // SSL有効化
      // WebGPU/WASM(SharedArrayBuffer)用にヘッダーも念のため追加推奨
      headers: {
        "Cross-Origin-Opener-Policy": "same-origin",
        "Cross-Origin-Embedder-Policy": "require-corp",
      },
    },
  };
});
