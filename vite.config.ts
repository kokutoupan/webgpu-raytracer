import { defineConfig } from 'vite';
import wasm from "vite-plugin-wasm";
import topLevelAwait from "vite-plugin-top-level-await";
import dotenv from "dotenv";
import path from "path";


dotenv.config({ path: path.resolve(__dirname, ".env") });


export default defineConfig(({ mode }) => {
  const isCloudflare = !(process.env.CF_PAGES_N === '1');

  return {
    base: isCloudflare ? '/' : 'webgpu-raytracer/', // Cloudflare用（GitHub Pages用にはローカルビルド時に別途調整が必要かもですが一旦ルートで）
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
