// main.ts
import shaderCode from './shader.wgsl?raw';
import { createCameraData, getSceneData, type SceneData } from './scene';
import { BVHBuilder } from "./bvh";

// --- Config ---
const IS_RETINA = false;
const DPR = IS_RETINA ? (window.devicePixelRatio || 1) : 1;

// --- DOM ---
const canvas = document.getElementById('gpu-canvas') as HTMLCanvasElement;
const btn = document.getElementById('render-btn') as HTMLButtonElement;
const sceneSelect = document.getElementById('scene-select') as HTMLSelectElement;

// --- FPS UI ---
const statsDiv = document.createElement("div");
Object.assign(statsDiv.style, {
  position: "fixed", top: "10px", left: "10px", color: "#0f0",
  background: "rgba(0,0,0,0.7)", padding: "8px", fontFamily: "monospace",
  fontSize: "14px", pointerEvents: "none", zIndex: "9999"
});
document.body.appendChild(statsDiv);

// --- Global State ---
let frameCount = 0;
let isRendering = false; // 初期状態は false (停止)
let currentSceneData: SceneData | null = null;

async function initAndRender() {
  if (!navigator.gpu) { alert("WebGPU not supported."); return; }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("No adapter");
  const device = await adapter.requestDevice();
  const context = canvas.getContext("webgpu");
  if (!context) throw new Error("No context");

  // Resize
  const resizeCanvas = () => {
    canvas.width = canvas.clientWidth * DPR;
    canvas.height = canvas.clientHeight * DPR;
  };
  resizeCanvas();

  context.configure({
    device, format: 'rgba8unorm', usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT
  });

  // --- Static Resources ---
  const renderTarget = device.createTexture({
    size: [canvas.width, canvas.height], format: 'rgba8unorm', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC
  });
  const renderTargetView = renderTarget.createView();

  const bufferSize = canvas.width * canvas.height * 16;
  const accumulateBuffer = device.createBuffer({ size: bufferSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const frameUniformBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  // Pipeline
  const shaderModule = device.createShaderModule({ label: "RayTracing", code: shaderCode });
  const pipeline = device.createComputePipeline({
    label: "Main Pipeline", layout: "auto", compute: { module: shaderModule, entryPoint: "main" }
  });
  const bindGroupLayout = pipeline.getBindGroupLayout(0);

  // --- Dynamic Resources ---
  let bindGroup: GPUBindGroup;
  let cameraUniformBuffer: GPUBuffer;

  // ★修正1: 第2引数に autoStart を追加
  const loadScene = (sceneName: string, autoStart: boolean = true) => {
    console.log(`Loading Scene: ${sceneName}...`);

    isRendering = false;

    // 1. Get Data
    const scene = getSceneData(sceneName);
    currentSceneData = scene;
    console.log(currentSceneData);

    // 2. Build BVH
    const bvhBuilder = new BVHBuilder();
    const bvhResult = bvhBuilder.build(scene.primitives);

    // バッファ作成
    // ★統合プリミティブバッファ (Binding 4)
    const primBuffer = device.createBuffer({
      size: bvhResult.unifiedPrimitives.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(primBuffer, 0, bvhResult.unifiedPrimitives);

    // BVHノード (Binding 5)
    const bvhBuffer = device.createBuffer({
      size: bvhResult.bvhNodes.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(bvhBuffer, 0, bvhResult.bvhNodes);

    if (!cameraUniformBuffer) {
      cameraUniformBuffer = device.createBuffer({ size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    }
    const camData = createCameraData(scene.camera, canvas.width / canvas.height);
    device.queue.writeBuffer(cameraUniformBuffer, 0, camData);

    // 4. Create BindGroup
    bindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: renderTargetView },
        { binding: 1, resource: { buffer: accumulateBuffer } },
        { binding: 2, resource: { buffer: frameUniformBuffer } },
        { binding: 3, resource: { buffer: cameraUniformBuffer } },
        // ★修正: 統合されたのでこれだけ
        { binding: 4, resource: { buffer: primBuffer } },
        { binding: 5, resource: { buffer: bvhBuffer } },
      ],
    });

    resetAccumulation();

    // ★修正2: autoStart フラグに従って開始・停止を決める
    if (autoStart) {
      isRendering = true;
      btn.textContent = "Stop Rendering";
    } else {
      isRendering = false;
      btn.textContent = "Render Start";
    }

    console.log("Scene Loaded.");
  };

  const resetAccumulation = () => {
    const zeroData = new Float32Array(bufferSize / 4);
    device.queue.writeBuffer(accumulateBuffer, 0, zeroData);
    frameCount = 0;
  };

  // --- Render Loop ---
  const frameData = new Uint32Array(1);
  const dispatchX = Math.ceil(canvas.width / 8);
  const dispatchY = Math.ceil(canvas.height / 8);

  const copySize: GPUExtent3DStrict = { width: canvas.width, height: canvas.height, depthOrArrayLayers: 1 };
  const copySrc = { texture: renderTarget };
  const copyDst = { texture: null as unknown as GPUTexture };

  let lastTime = performance.now();
  let frameTimer = 0;

  const renderFrame = () => {
    // レンダリング中じゃなければ何もしない（ループは回し続ける）
    if (!isRendering || !bindGroup) {
      requestAnimationFrame(renderFrame);
      return;
    }

    const now = performance.now();
    frameCount++;
    frameTimer++;

    frameData[0] = frameCount;
    device.queue.writeBuffer(frameUniformBuffer, 0, frameData);

    copyDst.texture = context.getCurrentTexture();
    const commandEncoder = device.createCommandEncoder();
    const pass = commandEncoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(dispatchX, dispatchY);
    pass.end();

    commandEncoder.copyTextureToTexture(copySrc, copyDst, copySize);
    device.queue.submit([commandEncoder.finish()]);

    if (now - lastTime >= 1000) {
      statsDiv.textContent = `FPS: ${frameTimer} | ${(1000 / frameTimer).toFixed(2)}ms | Frame: ${frameCount}`;
      frameTimer = 0;
      lastTime = now;
    }

    requestAnimationFrame(renderFrame);
  };

  // --- Events ---
  btn.addEventListener("click", () => {
    if (isRendering) {
      // 停止処理
      isRendering = false;
      btn.textContent = "Resume Rendering";
    } else {
      // 開始処理
      isRendering = true;
      btn.textContent = "Stop Rendering";

      // もし完全にリセットして始めたい場合はここコメントアウトを外す
      // resetAccumulation(); 
    }
  });

  sceneSelect.addEventListener("change", (e) => {
    const target = e.target as HTMLSelectElement;
    // プルダウン変更時は自動スタートする
    loadScene(target.value, false);
  });

  // ★修正3: 初回ロード時は自動スタートしない (false)
  loadScene("cornell", false);

  // ループ自体は開始しておく（isRendering=falseなので待機状態になる）
  requestAnimationFrame(renderFrame);
}

initAndRender().catch(console.error);
