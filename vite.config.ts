// vite.config.ts
import { defineConfig } from 'vite';

export default defineConfig({
  // リポジトリ名を設定（これは必須）
  base: '/webgpu-raytracer/', 
  build: {
    // 出力先を 'docs' フォルダに変更
    outDir: 'docs',
  }
});
