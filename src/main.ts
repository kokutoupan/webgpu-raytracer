// src/main.ts
import shaderCodeRaw from './shader.wgsl?raw';
import init, { World } from '../rust-shader-tools/pkg/rust_shader_tools';

// --- DOM Elements ---
const canvas = document.getElementById('gpu-canvas') as HTMLCanvasElement;
const btn = document.getElementById('render-btn') as HTMLButtonElement;
const sceneSelect = document.getElementById('scene-select') as HTMLSelectElement;
const inputWidth = document.getElementById('res-width') as HTMLInputElement;
const inputHeight = document.getElementById('res-height') as HTMLInputElement;
const inputFile = document.getElementById('obj-file') as HTMLInputElement;
if (inputFile) inputFile.accept = ".obj,.glb,.vrm";
const inputAnimFile = document.getElementById('anim-file') as HTMLInputElement;
if (inputAnimFile) inputAnimFile.accept = ".glb,.gltf";

const inputDepth = document.getElementById('max-depth') as HTMLInputElement;
const inputSPP = document.getElementById('spp-frame') as HTMLInputElement;
const btnRecompile = document.getElementById('recompile-btn') as HTMLButtonElement;
const inputUpdateInterval = document.getElementById('update-interval') as HTMLInputElement;

// --- Stats UI ---
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
let currentWorld: World | null = null;
let wasmMemory: WebAssembly.Memory | null = null;

async function initAndRender() {
  if (!navigator.gpu) { alert("WebGPU not supported."); return; }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("No adapter");
  const device = await adapter.requestDevice();
  const context = canvas.getContext("webgpu") as GPUCanvasContext;

  context.configure({
    device, format: 'rgba8unorm', usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT
  });

  const wasmInstance = await init();
  wasmMemory = wasmInstance.memory;
  console.log("Wasm initialized");

  // --- Pipeline ---
  let pipeline: GPUComputePipeline;
  let bindGroupLayout: GPUBindGroupLayout;

  const buildPipeline = () => {
    const depthVal = parseInt(inputDepth.value, 10) || 10;
    const sppVal = parseInt(inputSPP.value, 10) || 1;
    let code = shaderCodeRaw;
    code = code.replace(/const\s+MAX_DEPTH\s*=\s*\d+u;/, `const MAX_DEPTH = ${depthVal}u;`);
    code = code.replace(/const\s+SPP\s*=\s*\d+u;/, `const SPP = ${sppVal}u;`);
    const shaderModule = device.createShaderModule({ label: "RayTracing", code });
    pipeline = device.createComputePipeline({
      label: "Main Pipeline", layout: "auto", compute: { module: shaderModule, entryPoint: "main" }
    });
    bindGroupLayout = pipeline.getBindGroupLayout(0);
  };
  buildPipeline();

  // --- Buffers ---
  let renderTarget: GPUTexture;
  let renderTargetView: GPUTextureView;
  let accumulateBuffer: GPUBuffer;
  let frameUniformBuffer: GPUBuffer;
  let cameraUniformBuffer: GPUBuffer;

  let vertexBuffer: GPUBuffer;
  let normalBuffer: GPUBuffer;
  let indexBuffer: GPUBuffer;
  let attrBuffer: GPUBuffer;
  let tlasBuffer: GPUBuffer;
  let blasBuffer: GPUBuffer;
  let instanceBuffer: GPUBuffer;

  let bindGroup: GPUBindGroup;
  let bufferSize = 0;

  const resetAccumulation = () => {
    if (!accumulateBuffer) return;
    device.queue.writeBuffer(accumulateBuffer, 0, new Float32Array(bufferSize / 4));
    frameCount = 0;
  };

  const updateScreenResources = () => {
    let w = parseInt(inputWidth.value, 10) || 720;
    let h = parseInt(inputHeight.value, 10) || 480;
    canvas.width = w; canvas.height = h;

    if (renderTarget) renderTarget.destroy();
    renderTarget = device.createTexture({
      size: [canvas.width, canvas.height], format: 'rgba8unorm',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC
    });
    renderTargetView = renderTarget.createView();

    bufferSize = canvas.width * canvas.height * 16;
    if (accumulateBuffer) accumulateBuffer.destroy();
    accumulateBuffer = device.createBuffer({ size: bufferSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });

    if (!frameUniformBuffer) frameUniformBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  };

  const updateBindGroup = () => {
    if (!renderTargetView || !accumulateBuffer || !vertexBuffer || !tlasBuffer) return;
    bindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: renderTargetView },
        { binding: 1, resource: { buffer: accumulateBuffer } },
        { binding: 2, resource: { buffer: frameUniformBuffer } },
        { binding: 3, resource: { buffer: cameraUniformBuffer } },
        { binding: 4, resource: { buffer: vertexBuffer } },
        { binding: 5, resource: { buffer: indexBuffer } },
        { binding: 6, resource: { buffer: attrBuffer } },
        { binding: 7, resource: { buffer: tlasBuffer } },
        { binding: 8, resource: { buffer: normalBuffer } },
        { binding: 9, resource: { buffer: blasBuffer } },
        { binding: 10, resource: { buffer: instanceBuffer } },
      ],
    });
  };

  const createStorage = (view: Float32Array<ArrayBuffer> | Uint32Array<ArrayBuffer>) => {
    const size = Math.max(view.byteLength, 4);
    const buf = device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    if (view.byteLength > 0) device.queue.writeBuffer(buf, 0, view);
    return buf;
  };

  const getF32 = (ptr: number, len: number) => new Float32Array(wasmMemory!.buffer, ptr, len);
  const getU32 = (ptr: number, len: number) => new Uint32Array(wasmMemory!.buffer, ptr, len);

  const loadScene = (sceneName: string, autoStart: boolean = true) => {
    console.log(`Loading Scene: ${sceneName}...`);
    isRendering = false;
    if (currentWorld) currentWorld.free();

    let objSource: string | undefined;
    let glbData: Uint8Array | undefined;

    if (sceneName === 'viewer' && currentFileData) {
      if (currentFileType === 'obj') objSource = currentFileData as string;
      else if (currentFileType === 'glb') glbData = new Uint8Array(currentFileData as ArrayBuffer);
    }

    currentWorld = new World(sceneName, objSource, glbData);

    // Initial Buffers
    const vView = getF32(currentWorld.vertices_ptr(), currentWorld.vertices_len());
    const nView = getF32(currentWorld.normals_ptr(), currentWorld.normals_len());
    const iView = getU32(currentWorld.indices_ptr(), currentWorld.indices_len());
    const aView = getF32(currentWorld.attributes_ptr(), currentWorld.attributes_len());
    const tView = getF32(currentWorld.tlas_ptr(), currentWorld.tlas_len());
    const bView = getF32(currentWorld.blas_ptr(), currentWorld.blas_len());
    const instView = getF32(currentWorld.instances_ptr(), currentWorld.instances_len());

    console.log(`Scene Stats: V=${vView.length / 4}, Tri=${iView.length / 3}, TLAS=${tView.length / 8}`);

    if (vertexBuffer) vertexBuffer.destroy(); vertexBuffer = createStorage(vView);
    if (normalBuffer) normalBuffer.destroy(); normalBuffer = createStorage(nView);
    if (indexBuffer) indexBuffer.destroy(); indexBuffer = createStorage(iView);
    if (attrBuffer) attrBuffer.destroy(); attrBuffer = createStorage(aView);
    if (tlasBuffer) tlasBuffer.destroy(); tlasBuffer = createStorage(tView);
    if (blasBuffer) blasBuffer.destroy(); blasBuffer = createStorage(bView);
    if (instanceBuffer) instanceBuffer.destroy(); instanceBuffer = createStorage(instView);

    if (!cameraUniformBuffer) cameraUniformBuffer = device.createBuffer({ size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    currentWorld.update_camera(canvas.width, canvas.height);
    device.queue.writeBuffer(cameraUniformBuffer, 0, getF32(currentWorld.camera_ptr(), 24));

    updateBindGroup();
    resetAccumulation();

    if (autoStart) { isRendering = true; btn.textContent = "Stop Rendering"; }
  };

  // --- Render Loop ---
  const frameData = new Uint32Array(1);
  const copyDst = { texture: null as unknown as GPUTexture };
  let lastTime = performance.now();
  let frameTimer = 0;

  const renderFrame = () => {
    requestAnimationFrame(renderFrame);
    if (!isRendering || !bindGroup || !currentWorld) return;

    let updateInterval = parseInt(inputUpdateInterval.value, 10);
    if (isNaN(updateInterval) || updateInterval < 0) updateInterval = 0;

    if (updateInterval > 0 && frameCount >= updateInterval) {
      const now = performance.now();
      const time = now / 1000.0;

      currentWorld.update(time);

      // ★修正: バッファ再作成ロジック (サイズ不足時は作り直す)
      let needsBindGroupUpdate = false;

      const updateOrRecreate = (oldBuf: GPUBuffer, view: Float32Array<ArrayBuffer> | Uint32Array<ArrayBuffer>): GPUBuffer => {
        if (oldBuf.size >= view.byteLength) {
          // サイズが足りれば書き込み
          device.queue.writeBuffer(oldBuf, 0, view);
          return oldBuf;
        } else {
          // サイズ不足なら破棄して作り直し
          console.log(`Buffer resizing: ${oldBuf.size} -> ${view.byteLength}`);
          oldBuf.destroy();
          needsBindGroupUpdate = true;
          return createStorage(view);
        }
      };

      // 各バッファを更新
      tlasBuffer = updateOrRecreate(tlasBuffer, getF32(currentWorld.tlas_ptr(), currentWorld.tlas_len()));
      blasBuffer = updateOrRecreate(blasBuffer, getF32(currentWorld.blas_ptr(), currentWorld.blas_len()));
      instanceBuffer = updateOrRecreate(instanceBuffer, getF32(currentWorld.instances_ptr(), currentWorld.instances_len()));
      vertexBuffer = updateOrRecreate(vertexBuffer, getF32(currentWorld.vertices_ptr(), currentWorld.vertices_len()));
      normalBuffer = updateOrRecreate(normalBuffer, getF32(currentWorld.normals_ptr(), currentWorld.normals_len()));
      indexBuffer = updateOrRecreate(indexBuffer, getU32(currentWorld.indices_ptr(), currentWorld.indices_len()));
      attrBuffer = updateOrRecreate(attrBuffer, getF32(currentWorld.attributes_ptr(), currentWorld.attributes_len()));

      // バッファが変わったらBindGroupも作り直す
      if (needsBindGroupUpdate) {
        updateBindGroup();
      }

      resetAccumulation();
    }

    const dispatchX = Math.ceil(canvas.width / 8);
    const dispatchY = Math.ceil(canvas.height / 8);

    frameCount++; frameTimer++;
    frameData[0] = frameCount;
    device.queue.writeBuffer(frameUniformBuffer, 0, frameData);

    copyDst.texture = context.getCurrentTexture();
    const commandEncoder = device.createCommandEncoder();
    const pass = commandEncoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(dispatchX, dispatchY);
    pass.end();

    commandEncoder.copyTextureToTexture({ texture: renderTarget }, copyDst, { width: canvas.width, height: canvas.height, depthOrArrayLayers: 1 });
    device.queue.submit([commandEncoder.finish()]);

    const now = performance.now();
    if (now - lastTime >= 1000) {
      statsDiv.textContent = `FPS: ${frameTimer} | ${(1000 / frameTimer).toFixed(2)}ms | Frame: ${frameCount}`;
      frameTimer = 0; lastTime = now;
    }
  };

  // --- Events ---
  let currentFileData: string | ArrayBuffer | null = null;
  let currentFileType: 'obj' | 'glb' | null = null;

  btn.addEventListener("click", () => {
    isRendering = !isRendering;
    btn.textContent = isRendering ? "Stop Rendering" : "Resume Rendering";
  });
  sceneSelect.addEventListener("change", (e) => loadScene((e.target as HTMLSelectElement).value, false));

  inputFile.addEventListener("change", async (e) => {
    const f = (e.target as HTMLInputElement).files?.[0];
    if (!f) return;
    const ext = f.name.split('.').pop()?.toLowerCase();
    if (ext === 'obj') { currentFileData = await f.text(); currentFileType = 'obj'; }
    else { currentFileData = await f.arrayBuffer(); currentFileType = 'glb'; }
    sceneSelect.value = "viewer"; loadScene("viewer", false);
  });

  inputAnimFile.addEventListener("change", async (e) => {
    const f = (e.target as HTMLInputElement).files?.[0];
    if (!f || !currentWorld) return;

    console.log(`Loading Motion: ${f.name}...`);
    const buffer = await f.arrayBuffer();
    const data = new Uint8Array(buffer);
    currentWorld.load_animation_glb(data);
    console.log("Motion Loaded!");
    (e.target as HTMLInputElement).value = "";
  });

  const onResolutionChange = () => {
    updateScreenResources();
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
    isRendering = false; buildPipeline(); updateBindGroup(); resetAccumulation(); isRendering = true;
  });

  updateScreenResources();
  loadScene("cornell", false);
  requestAnimationFrame(renderFrame);
}

initAndRender().catch(console.error);
