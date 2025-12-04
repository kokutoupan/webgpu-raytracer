// src/main.ts
import shaderCode from './shader.wgsl?raw';

// DOM取得
const canvas = document.getElementById('gpu-canvas') as HTMLCanvasElement;
const btn = document.getElementById('render-btn') as HTMLButtonElement;

async function initAndRender() {
  // 1. WebGPU初期化
  if (!navigator.gpu) {
    alert("WebGPUがサポートされていません。Chrome Canary等で試してください。");
    return;
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) { throw new Error("No adapter found"); }

  const device = await adapter.requestDevice();

  // 2. キャンバスコンテキスト設定
  const context = canvas.getContext("webgpu");
  if (!context) { throw new Error("WebGPU context not found"); }

  // const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
  //
  // context.configure({
  //   device,
  //   format: presentationFormat,
  //   usage: GPUTextureUsage.STORAGE_BINDING // Compute Shaderから書き込むため
  // });
  context.configure({
    device,
    format: 'rgba8unorm', // ★ここを強制的に rgba8unorm に変更
    usage: GPUTextureUsage.STORAGE_BINDING
  });
  // 3. パイプライン作成
  const shaderModule = device.createShaderModule({
    label: "RayTracing Shader",
    code: shaderCode,
  });

  const pipeline = device.createComputePipeline({
    label: "Main Pipeline",
    layout: "auto",
    compute: {
      module: shaderModule,
      entryPoint: "main",
    },
  });

  // 4. レンダリング関数（ボタンクリックで呼ばれる）
  const render = () => {
    console.log("Rendering...");
    const texture = context.getCurrentTexture();

    // BindGroup作成 (シェーダーのリソース定義と紐付け)
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: texture.createView(),
        },
      ],
    });

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();

    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);

    // スレッド数の計算 (8x8のワークグループ)
    const workgroupCountX = Math.ceil(canvas.width / 8);
    const workgroupCountY = Math.ceil(canvas.height / 8);
    passEncoder.dispatchWorkgroups(workgroupCountX, workgroupCountY);

    passEncoder.end();

    device.queue.submit([commandEncoder.finish()]);
    console.log("Done!");
  };

  // ボタンにイベントリスナー登録
  btn.addEventListener("click", render);

  // 起動確認用ログ
  console.log("WebGPU Ready. Click button to render.");
}

initAndRender().catch(console.error);
