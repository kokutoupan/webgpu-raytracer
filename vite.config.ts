import { defineConfig } from 'vite';
import wasm from "vite-plugin-wasm";
import topLevelAwait from "vite-plugin-top-level-await";

export default defineConfig(({ mode }) => {
  const isCloudflare = process.env.CF_PAGES === '1';

  return {
    base: isCloudflare ? '/' :'webgpu-raytracer/', // Cloudflare用（GitHub Pages用にはローカルビルド時に別途調整が必要かもですが一旦ルートで）
    build: {
      // Cloudflareなら 'dist' (デフォルト)、それ以外(ローカル)なら 'docs'
      outDir: isCloudflare ? 'dist' : 'docs',
    },
    plugins: [
      wasm(),
      topLevelAwait()
    ]
  };
});
