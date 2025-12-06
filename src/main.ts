// main.ts
import shaderCode from './shader.wgsl?raw';
import { createCameraData, makeCornellBox } from './scene';
import { BVHBuilder } from "./bvh";

// --- Configuration ---
const IS_RETINA = false; // Set true to use devicePixelRatio
const DPR = IS_RETINA ? (window.devicePixelRatio || 1) : 1;

// --- DOM Elements ---
const canvas = document.getElementById('gpu-canvas') as HTMLCanvasElement;
const btn = document.getElementById('render-btn') as HTMLButtonElement;

// --- FPS Counter ---
const statsDiv = document.createElement("div");
Object.assign(statsDiv.style, {
  position: "fixed", top: "10px", left: "10px", color: "#0f0",
  background: "rgba(0, 0, 0, 0.7)", padding: "8px", fontFamily: "monospace",
  fontSize: "14px", pointerEvents: "none", zIndex: "9999"
});
document.body.appendChild(statsDiv);

// --- Main Application ---
async function initAndRender() {
  // 1. WebGPU Initialization
  if (!navigator.gpu) { alert("WebGPU not supported."); return; }

  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) { throw new Error("No adapter found"); }

  const device = await adapter.requestDevice();
  const context = canvas.getContext("webgpu");
  if (!context) { throw new Error("WebGPU context not found"); }

  // 2. Resolution Setup
  const resizeCanvas = () => {
    canvas.width = canvas.clientWidth * DPR;
    canvas.height = canvas.clientHeight * DPR;
  };
  resizeCanvas();

  console.log(`DPR: ${DPR}, Buffer: ${canvas.width}x${canvas.height}`);

  // 3. Texture Configuration
  context.configure({
    device,
    format: 'rgba8unorm',
    usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT
  });

  // Intermediate Render Target
  const renderTarget = device.createTexture({
    size: [canvas.width, canvas.height],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC
  });
  const renderTargetView = renderTarget.createView();

  // 4. Buffer Creation
  const bufferSize = canvas.width * canvas.height * 16; // 4 floats * 4 bytes
  const accumulateBuffer = device.createBuffer({
    size: bufferSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  const frameUniformBuffer = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // Camera Data
  const cameraUniformBuffer = device.createBuffer({
    size: 96,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  {
    const camData = createCameraData(
      { x: 0, y: 1.0, z: -2.4 }, // Look from
      { x: 0, y: 1.0, z: 0 },    // Look at
      { x: 0, y: 1, z: 0 },      // Up
      60.0,                      // FOV
      canvas.width / canvas.height,
      0.0,                       // Defocus Angle
      2.4                        // Focus Dist
    );
    device.queue.writeBuffer(cameraUniformBuffer, 0, camData);
  }

  // --- Scene Data Generation ---

  // A. Spheres
  // const rawSpheres = makeSpheres(); // 球を表示したい場合はこちら

  // 球を表示しない場合でも、空のバッファを送るとエラーになるためダミー球を入れる
  // (中心座標を画面外のはるか遠くにする)
  const rawSpheres = new Float32Array([
    0, -9999, 0, 0,  // center(xyz), radius
    0, 0, 0, 0,      // color(rgb), mat
    0, 0, 0, 0       // extra, pad...
  ]);

  const sphereBuffer = device.createBuffer({
    size: rawSpheres.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(sphereBuffer, 0, rawSpheres);

  // B. Triangles (Cornell Box)
  const rawTriangles = makeCornellBox();

  const triangleBuffer = device.createBuffer({
    size: rawTriangles.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(triangleBuffer, 0, rawTriangles);

  // C. BVH Construction (Mixed)
  const bvhBuilder = new BVHBuilder();
  const bvhResult = bvhBuilder.build(rawSpheres, rawTriangles);

  console.log(`BVH Nodes: ${bvhResult.bvhNodes.length / 8}, Refs: ${bvhResult.primitiveRefs.length / 2}`);

  const bvhBuffer = device.createBuffer({
    size: bvhResult.bvhNodes.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(bvhBuffer, 0, bvhResult.bvhNodes);

  const refsBuffer = device.createBuffer({
    size: bvhResult.primitiveRefs.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
  });
  device.queue.writeBuffer(refsBuffer, 0, bvhResult.primitiveRefs);


  // 5. Pipeline Setup
  const shaderModule = device.createShaderModule({ label: "RayTracing", code: shaderCode });
  const pipeline = device.createComputePipeline({
    label: "Main Pipeline",
    layout: "auto",
    compute: { module: shaderModule, entryPoint: "main" },
  });

  const bindGroupLayout = pipeline.getBindGroupLayout(0);
  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: renderTargetView },
      { binding: 1, resource: { buffer: accumulateBuffer } },
      { binding: 2, resource: { buffer: frameUniformBuffer } },
      { binding: 3, resource: { buffer: cameraUniformBuffer } },
      { binding: 4, resource: { buffer: sphereBuffer } },
      { binding: 5, resource: { buffer: triangleBuffer } },
      { binding: 6, resource: { buffer: bvhBuffer } },
      { binding: 7, resource: { buffer: refsBuffer } },
    ],
  });

  // --- Render Loop Variables ---
  const frameData = new Uint32Array(1);
  const dispatchX = Math.ceil(canvas.width / 8);
  const dispatchY = Math.ceil(canvas.height / 8);

  const copySize: GPUExtent3DStrict = { width: canvas.width, height: canvas.height, depthOrArrayLayers: 1 };
  const copySrc = { texture: renderTarget };
  const copyDst = { texture: null as unknown as GPUTexture };

  let frameCount = 0;
  let isRendering = false;
  let lastTime = performance.now();
  let frameCountTimer = 0;

  // --- 6. Render Loop ---
  const renderFrame = () => {
    if (!isRendering) return;

    const now = performance.now();
    frameCount++;
    frameCountTimer++;

    // A. Update Uniforms
    frameData[0] = frameCount;
    device.queue.writeBuffer(frameUniformBuffer, 0, frameData);

    // B. Encode Commands
    copyDst.texture = context.getCurrentTexture();

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(dispatchX, dispatchY);
    passEncoder.end();

    // C. Blit
    commandEncoder.copyTextureToTexture(copySrc, copyDst, copySize);
    device.queue.submit([commandEncoder.finish()]);

    // D. FPS Measurement
    if (now - lastTime >= 1000) {
      const fps = frameCountTimer;
      const ms = (1000 / fps).toFixed(2);
      statsDiv.textContent = `FPS: ${fps} | Frame Time: ${ms}ms`;
      frameCountTimer = 0;
      lastTime = now;
    }

    requestAnimationFrame(renderFrame);
  };

  // --- Event Handling ---
  btn.addEventListener("click", () => {
    if (isRendering) {
      console.log("Stop Rendering");
      isRendering = false;
      btn.textContent = "Restart Rendering";
    } else {
      console.log("Reset & Start Rendering");
      frameCount = 0;
      // Clear Accumulation Buffer
      const zeroData = new Float32Array(bufferSize / 4);
      device.queue.writeBuffer(accumulateBuffer, 0, zeroData);

      isRendering = true;
      renderFrame();
      btn.textContent = "Stop Rendering";
    }
  });
}

initAndRender().catch(console.error);
