// vite.config.ts
import { defineConfig } from 'vite';
import wasm from "vite-plugin-wasm";
import topLevelAwait from "vite-plugin-top-level-await";

export default defineConfig({
  // リポジトリ名を設定（これは必須）
  base: '/webgpu-raytracer/',
  build: {
    // 出力先を 'docs' フォルダに変更
    outDir: 'docs',
  },
  plugins: [
    wasm(),
    topLevelAwait()
  ]
});
