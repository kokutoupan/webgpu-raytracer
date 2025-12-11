// src/main.ts
import shaderCodeRaw from './shader.wgsl?raw';
// 古いTSモジュールのimportは全削除
import init, { World } from '../rust-shader-tools/pkg/rust_shader_tools';

// --- DOM ---
const canvas = document.getElementById('gpu-canvas') as HTMLCanvasElement;
const btn = document.getElementById('render-btn') as HTMLButtonElement;
const sceneSelect = document.getElementById('scene-select') as HTMLSelectElement;
const inputWidth = document.getElementById('res-width') as HTMLInputElement;
const inputHeight = document.getElementById('res-height') as HTMLInputElement;
const inputFile = document.getElementById('obj-file') as HTMLInputElement;
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

// RustのWorldインスタンスとWasmメモリ
let currentWorld: World | null = null;
let wasmMemory: WebAssembly.Memory | null = null;

// OBJテキストデータの一時保持
let currentObjText: string | undefined = undefined;

async function initAndRender() {
  // 1. WebGPU初期化
  if (!navigator.gpu) { alert("WebGPU not supported."); return; }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("No adapter");
  const device = await adapter.requestDevice();
  const context = canvas.getContext("webgpu");
  if (!context) throw new Error("No context");

  context.configure({
    device, format: 'rgba8unorm', usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT
  });

  // 2. Wasm初期化
  const wasmInstance = await init();
  wasmMemory = wasmInstance.memory;
  console.log("Wasm initialized");

  // --- Dynamic Pipeline & Shader ---
  let pipeline: GPUComputePipeline;
  let bindGroupLayout: GPUBindGroupLayout;

  const buildPipeline = () => {
    const depthVal = parseInt(inputDepth.value, 10) || 10;
    const sppVal = parseInt(inputSPP.value, 10) || 1;
    console.log(`Recompiling Shader... Depth:${depthVal}, SPP:${sppVal}`);

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

  // ★新しいバッファ群
  let vertexBuffer: GPUBuffer;
  let indexBuffer: GPUBuffer;
  let attrBuffer: GPUBuffer;
  let bvhBuffer: GPUBuffer;

  let bindGroup: GPUBindGroup;
  let bufferSize = 0;

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
      size: [canvas.width, canvas.height], format: 'rgba8unorm',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC
    });
    renderTargetView = renderTarget.createView();

    bufferSize = canvas.width * canvas.height * 16;
    if (accumulateBuffer) accumulateBuffer.destroy();
    accumulateBuffer = device.createBuffer({ size: bufferSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });

    if (!frameUniformBuffer) {
      frameUniformBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    }
  };

  const updateBindGroup = () => {
    // 全リソースが揃っているか確認
    if (!renderTargetView || !accumulateBuffer || !frameUniformBuffer || !cameraUniformBuffer ||
      !vertexBuffer || !indexBuffer || !attrBuffer || !bvhBuffer) return;

    bindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: renderTargetView },
        { binding: 1, resource: { buffer: accumulateBuffer } },
        { binding: 2, resource: { buffer: frameUniformBuffer } },
        { binding: 3, resource: { buffer: cameraUniformBuffer } },
        // ★Binding番号を変更・追加 (シェーダー側もこれに合わせる必要あり)
        { binding: 4, resource: { buffer: vertexBuffer } }, // Vertices
        { binding: 5, resource: { buffer: indexBuffer } },  // Indices
        { binding: 6, resource: { buffer: attrBuffer } },   // Attributes
        { binding: 7, resource: { buffer: bvhBuffer } },    // BVH Nodes
      ],
    });
  };

  const loadScene = (sceneName: string, autoStart: boolean = true) => {
    console.log(`Loading Scene: ${sceneName}... (Rust)`);
    isRendering = false;

    if (currentWorld) {
      currentWorld.free();
    }

    // Rust側でシーン構築
    // viewerシーンならロード済みのOBJテキストを渡す
    const meshArg = (sceneName === 'viewer' && currentObjText) ? currentObjText : undefined;

    console.time("Rust Build");
    currentWorld = new World(sceneName, meshArg);
    console.timeEnd("Rust Build");

    if (!wasmMemory) return;

    // --- Wasmメモリからデータを取得 (ゼロコピー) ---

    // 1. Vertices (f32)
    const vPtr = currentWorld.vertices_ptr();
    const vLen = currentWorld.vertices_len();
    const vView = new Float32Array(wasmMemory.buffer, vPtr, vLen);

    // 2. Indices (u32)
    const iPtr = currentWorld.indices_ptr();
    const iLen = currentWorld.indices_len();
    const iView = new Uint32Array(wasmMemory.buffer, iPtr, iLen);

    // 3. Attributes (f32)
    const aPtr = currentWorld.attributes_ptr();
    const aLen = currentWorld.attributes_len();
    const aView = new Float32Array(wasmMemory.buffer, aPtr, aLen);

    // 4. BVH Nodes (f32)
    const bPtr = currentWorld.bvh_ptr();
    const bLen = currentWorld.bvh_len();
    const bView = new Float32Array(wasmMemory.buffer, bPtr, bLen);

    // 5. Camera (f32)
    const cPtr = currentWorld.camera_ptr();
    const cView = new Float32Array(wasmMemory.buffer, cPtr, 24);

    console.log(`Scene Stats: Verts:${vLen / 4}, Tris:${iLen / 3}, Nodes:${bLen / 8}`);

    // --- WebGPUバッファ転送 ---

    if (vertexBuffer) vertexBuffer.destroy();
    vertexBuffer = device.createBuffer({ size: vView.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(vertexBuffer, 0, vView);

    if (indexBuffer) indexBuffer.destroy();
    // Indicesはアライメント(4byte)に注意が必要だがu32なら問題なし
    // ただしサイズが0バイトだとcreateBufferで落ちることがあるので安全策
    const iSize = Math.max(iView.byteLength, 4);
    indexBuffer = device.createBuffer({ size: iSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    if (iView.byteLength > 0) device.queue.writeBuffer(indexBuffer, 0, iView);

    if (attrBuffer) attrBuffer.destroy();
    const aSize = Math.max(aView.byteLength, 4);
    attrBuffer = device.createBuffer({ size: aSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    if (aView.byteLength > 0) device.queue.writeBuffer(attrBuffer, 0, aView);

    if (bvhBuffer) bvhBuffer.destroy();
    bvhBuffer = device.createBuffer({ size: bView.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(bvhBuffer, 0, bView);

    // カメラバッファ
    if (!cameraUniformBuffer) {
      cameraUniformBuffer = device.createBuffer({ size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    }
    // 初期化時はWasmが計算したデフォルトを転送
    // (解像度がWasm側の想定と違う場合があるので、本当はここで updateCamera を呼ぶべき)
    currentWorld.update_camera(canvas.width, canvas.height); // 正しいアスペクト比で再計算
    const updatedCamView = new Float32Array(wasmMemory.buffer, currentWorld.camera_ptr(), 24);
    device.queue.writeBuffer(cameraUniformBuffer, 0, updatedCamView);

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

  inputFile.addEventListener("change", async (e) => {
    const target = e.target as HTMLInputElement;
    const file = target.files?.[0];
    if (!file) return;
    console.log(`Reading ${file.name}...`);
    try {
      const text = await file.text();
      currentObjText = text;
      sceneSelect.value = "viewer";
      loadScene("viewer", false);
    } catch (err) {
      console.error("Failed to load OBJ:", err);
      alert("Failed to load OBJ file.");
    }
    target.value = "";
  });

  const onResolutionChange = () => {
    updateScreenResources();
    // 解像度変更時、Rust側でカメラバッファを再計算
    if (currentWorld && wasmMemory && cameraUniformBuffer) {
      currentWorld.update_camera(canvas.width, canvas.height);
      const cView = new Float32Array(wasmMemory.buffer, currentWorld.camera_ptr(), 24);
      device.queue.writeBuffer(cameraUniformBuffer, 0, cView);
    }
    updateBindGroup();
    resetAccumulation();
  };
  inputWidth.addEventListener("change", onResolutionChange);
  inputHeight.addEventListener("change", onResolutionChange);

  btnRecompile.addEventListener("click", () => {
    isRendering = false;
    buildPipeline();
    updateBindGroup();
    resetAccumulation();
    isRendering = true;
    btn.textContent = "Stop Rendering";
  });

  // --- Init ---
  updateScreenResources();
  loadScene("cornell", false);
  requestAnimationFrame(renderFrame);
}

initAndRender().catch(console.error);
