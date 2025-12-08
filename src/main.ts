// main.ts
import shaderCodeRaw from './shader.wgsl?raw'; // 生データを読み込む
import { createCameraData, getSceneData, type SceneData } from './scene';
import { BVHBuilder } from "./bvh";

// --- DOM ---
const canvas = document.getElementById('gpu-canvas') as HTMLCanvasElement;
const btn = document.getElementById('render-btn') as HTMLButtonElement;
const sceneSelect = document.getElementById('scene-select') as HTMLSelectElement;
const inputWidth = document.getElementById('res-width') as HTMLInputElement;
const inputHeight = document.getElementById('res-height') as HTMLInputElement;

// ★追加
const inputDepth = document.getElementById('max-depth') as HTMLInputElement;
const inputSPP = document.getElementById('spp-frame') as HTMLInputElement;
const btnRecompile = document.getElementById('recompile-btn') as HTMLButtonElement;

// --- FPS UI ---
const statsDiv = document.createElement("div");
Object.assign(statsDiv.style, {
  position: "fixed", bottom: "10px", left: "10px", color: "#0f0",
  background: "rgba(0,0,0,0.7)", padding: "8px", fontFamily: "monospace",
  fontSize: "14px", pointerEvents: "none", zIndex: "9999", borderRadius: "4px"
});
document.body.appendChild(statsDiv);

// --- Global State ---
let frameCount = 0;
let isRendering = false;
let currentSceneData: SceneData | null = null;

async function initAndRender() {
  if (!navigator.gpu) { alert("WebGPU not supported."); return; }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("No adapter");
  const device = await adapter.requestDevice();
  const context = canvas.getContext("webgpu");
  if (!context) throw new Error("No context");

  context.configure({
    device, format: 'rgba8unorm', usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT
  });

  // --- Dynamic Pipeline & Shader ---
  let pipeline: GPUComputePipeline;
  let bindGroupLayout: GPUBindGroupLayout;

  // ★関数: シェーダーをコンパイルしてパイプラインを作成
  const buildPipeline = () => {
    // UIから値を取得
    const depthVal = parseInt(inputDepth.value, 10) || 10;
    const sppVal = parseInt(inputSPP.value, 10) || 1;

    console.log(`Recompiling Shader... Depth:${depthVal}, SPP:${sppVal}`);

    // 文字列置換で定数を埋め込む
    let code = shaderCodeRaw;
    code = code.replace(/const\s+MAX_DEPTH\s*=\s*\d+u;/, `const MAX_DEPTH = ${depthVal}u;`);
    code = code.replace(/const\s+SPP\s*=\s*\d+u;/, `const SPP = ${sppVal}u;`);

    const shaderModule = device.createShaderModule({ label: "RayTracing", code });
    pipeline = device.createComputePipeline({
      label: "Main Pipeline", layout: "auto", compute: { module: shaderModule, entryPoint: "main" }
    });
    bindGroupLayout = pipeline.getBindGroupLayout(0);
  };

  // 初回ビルド
  buildPipeline();

  // --- Resources ---
  let renderTarget: GPUTexture;
  let renderTargetView: GPUTextureView;
  let accumulateBuffer: GPUBuffer;
  let frameUniformBuffer: GPUBuffer;
  let cameraUniformBuffer: GPUBuffer;
  let primBuffer: GPUBuffer;
  let bvhBuffer: GPUBuffer;
  let bindGroup: GPUBindGroup;

  let bufferSize = 0;

  // --- Functions ---
  const resetAccumulation = () => {
    if (!accumulateBuffer) return;
    const zeroData = new Float32Array(bufferSize / 4);
    device.queue.writeBuffer(accumulateBuffer, 0, zeroData);
    frameCount = 0;
  };

  const updateScreenResources = () => {
    let w = parseInt(inputWidth.value, 10);
    let h = parseInt(inputHeight.value, 10);
    if (isNaN(w) || w < 1) w = 720;
    if (isNaN(h) || h < 1) h = 480;

    canvas.width = w;
    canvas.height = h;

    if (renderTarget) renderTarget.destroy();
    renderTarget = device.createTexture({
      size: [canvas.width, canvas.height],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC
    });
    renderTargetView = renderTarget.createView();

    bufferSize = canvas.width * canvas.height * 16;
    if (accumulateBuffer) accumulateBuffer.destroy();
    accumulateBuffer = device.createBuffer({
      size: bufferSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });

    if (!frameUniformBuffer) {
      frameUniformBuffer = device.createBuffer({
        size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
      });
    }
  };

  const updateBindGroup = () => {
    if (!renderTargetView || !accumulateBuffer || !frameUniformBuffer || !cameraUniformBuffer || !primBuffer || !bvhBuffer) return;

    bindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: renderTargetView },
        { binding: 1, resource: { buffer: accumulateBuffer } },
        { binding: 2, resource: { buffer: frameUniformBuffer } },
        { binding: 3, resource: { buffer: cameraUniformBuffer } },
        { binding: 4, resource: { buffer: primBuffer } },
        { binding: 5, resource: { buffer: bvhBuffer } },
      ],
    });
  };

  const loadScene = (sceneName: string, autoStart: boolean = true) => {
    console.log(`Loading Scene: ${sceneName}...`);
    isRendering = false;

    const scene = getSceneData(sceneName);
    currentSceneData = scene;

    const bvhBuilder = new BVHBuilder();
    const bvhResult = bvhBuilder.build(scene.primitives);

    if (primBuffer) primBuffer.destroy();
    primBuffer = device.createBuffer({
      size: bvhResult.unifiedPrimitives.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(primBuffer, 0, bvhResult.unifiedPrimitives);

    if (bvhBuffer) bvhBuffer.destroy();
    bvhBuffer = device.createBuffer({
      size: bvhResult.bvhNodes.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(bvhBuffer, 0, bvhResult.bvhNodes);

    if (!cameraUniformBuffer) {
      cameraUniformBuffer = device.createBuffer({ size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    }
    const aspect = canvas.width / canvas.height;
    const camData = createCameraData(scene.camera, aspect);
    device.queue.writeBuffer(cameraUniformBuffer, 0, camData);

    updateBindGroup();
    resetAccumulation();

    if (autoStart) {
      isRendering = true;
      btn.textContent = "Stop Rendering";
    } else {
      isRendering = false;
      btn.textContent = "Render Start";
    }
  };

  // --- Render Loop ---
  const frameData = new Uint32Array(1);
  const copyDst = { texture: null as unknown as GPUTexture };
  let lastTime = performance.now();
  let frameTimer = 0;

  const renderFrame = () => {
    requestAnimationFrame(renderFrame);
    if (!isRendering || !bindGroup) return;

    const dispatchX = Math.ceil(canvas.width / 8);
    const dispatchY = Math.ceil(canvas.height / 8);

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

    const copySize: GPUExtent3DStrict = { width: canvas.width, height: canvas.height, depthOrArrayLayers: 1 };
    const copySrc = { texture: renderTarget };
    commandEncoder.copyTextureToTexture(copySrc, copyDst, copySize);
    device.queue.submit([commandEncoder.finish()]);

    if (now - lastTime >= 1000) {
      statsDiv.textContent = `FPS: ${frameTimer} | ${(1000 / frameTimer).toFixed(2)}ms | Frame: ${frameCount} | Res: ${canvas.width}x${canvas.height}`;
      frameTimer = 0;
      lastTime = now;
    }
  };

  // --- Events ---
  btn.addEventListener("click", () => {
    isRendering = !isRendering;
    btn.textContent = isRendering ? "Stop Rendering" : "Resume Rendering";
  });

  sceneSelect.addEventListener("change", (e) => {
    const target = e.target as HTMLSelectElement;
    loadScene(target.value, false);
  });

  // 解像度変更
  const onResolutionChange = () => {
    updateScreenResources();
    if (currentSceneData && cameraUniformBuffer) {
      const aspect = canvas.width / canvas.height;
      const camData = createCameraData(currentSceneData.camera, aspect);
      device.queue.writeBuffer(cameraUniformBuffer, 0, camData);
    }
    updateBindGroup();
    resetAccumulation();
  };
  inputWidth.addEventListener("change", onResolutionChange);
  inputHeight.addEventListener("change", onResolutionChange);

  // ★シェーダー再コンパイル
  btnRecompile.addEventListener("click", () => {
    isRendering = false; // 一旦止める
    buildPipeline();     // パイプライン再構築
    updateBindGroup();   // BindGroupはパイプラインのLayoutに依存するので、一応更新（Layoutが変わらなければ不要だが安全のため）
    resetAccumulation(); // 絵が変わるのでリセット

    // 再開
    isRendering = true;
    btn.textContent = "Stop Rendering";
  });

  // --- Init ---
  updateScreenResources();
  loadScene("cornell", false);
  requestAnimationFrame(renderFrame);
}

initAndRender().catch(console.error);
