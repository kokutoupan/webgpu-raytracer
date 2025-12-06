// main.ts
import shaderCode from './shader.wgsl?raw';
import { createCameraData, makeCornellBox } from './scene';
import { BVHBuilder } from "./bvh";

// --- 設定値 ---
const IS_RETINA = false; // DPRを下げるなという指示に従う
const DPR = IS_RETINA ? (window.devicePixelRatio || 1) : 1;

// --- DOM取得 ---
const canvas = document.getElementById('gpu-canvas') as HTMLCanvasElement;
const btn = document.getElementById('render-btn') as HTMLButtonElement;

// --- FPSカウンター UI ---
const statsDiv = document.createElement("div");
Object.assign(statsDiv.style, {
  position: "fixed", top: "10px", left: "10px", color: "#0f0",
  background: "rgba(0, 0, 0, 0.7)", padding: "8px", fontFamily: "monospace",
  fontSize: "14px", pointerEvents: "none", zIndex: "9999"
});
document.body.appendChild(statsDiv);

// --- メイン処理 ---
async function initAndRender() {
  // 1. WebGPU初期化
  if (!navigator.gpu) { alert("WebGPU not supported."); return; }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) { throw new Error("No adapter found"); }
  const device = await adapter.requestDevice();
  const context = canvas.getContext("webgpu");
  if (!context) { throw new Error("WebGPU context not found"); }

  // 2. 解像度設定 (CSSサイズとバッファサイズを同期)
  const resizeCanvas = () => {
    canvas.width = canvas.clientWidth * DPR;
    canvas.height = canvas.clientHeight * DPR;
  };
  resizeCanvas(); // 初回実行

  console.log(`DPR: ${DPR}, Buffer: ${canvas.width}x${canvas.height}`);

  // 3. テクスチャ設定
  context.configure({
    device,
    format: 'rgba8unorm',
    usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT
  });

  // 中間レンダリングターゲット (Compute Shader書き込み用)
  const renderTarget = device.createTexture({
    size: [canvas.width, canvas.height],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC
  });
  const renderTargetView = renderTarget.createView();

  // 4. バッファ作成
  const bufferSize = canvas.width * canvas.height * 16; // 4 float * 4 byte
  const accumulateBuffer = device.createBuffer({
    size: bufferSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  const frameUniformBuffer = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // カメラデータ送信
  const cameraUniformBuffer = device.createBuffer({
    size: 96,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  {
    const camData = createCameraData(
      { x: 0, y: 1.0, z: -2.4 }, // 手前から奥を見る
      { x: 0, y: 1.0, z: 0 },    // 中心の少し上を見る
      { x: 0, y: 1, z: 0 },
      60.0,                  // FOV広め
      canvas.width / canvas.height,
      0.0,                   // ピンホール (ボケなし)
      2.4                    // 焦点距離
    );
    device.queue.writeBuffer(cameraUniformBuffer, 0, camData);
  }

  // 球データ送信
  // const spheresArray = makeSpheres();
  // 空の配列ではなく、ダミーデータを1つ入れる
  // (サイズ0のバッファを作るとWebGPUがエラーを吐くため)
  const spheresArray: number[][] = [
    // Center(0,-9999,0), Radius(0), Color(0,0,0), Mat(0), Extra(0) ...
    // カメラから絶対に見えない位置に、半径0の球を置く
    [0, -9999, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  ];
  const sphereF32Array = new Float32Array(spheresArray.flat());
  const sphereBuffer = device.createBuffer({
    size: sphereF32Array.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(sphereBuffer, 0, sphereF32Array);

  // 三角形

  const triangleF32Array = makeCornellBox();
  console.log(triangleF32Array);
  let bvhBuilder = new BVHBuilder();
  const bvhResult = bvhBuilder.build(sphereF32Array, triangleF32Array);
  console.log(bvhResult);

  const triangleBuffer = device.createBuffer({
    size: triangleF32Array.byteLength, // 空でもエラーにならないよう注意
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(triangleBuffer, 0, triangleF32Array);

  const bvhBuffer = device.createBuffer({
    size: bvhResult.bvhNodes.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(bvhBuffer, 0, bvhResult.bvhNodes);

  // プリミティブ参照リスト (Uint32Array)
  const refsBuffer = device.createBuffer({
    size: bvhResult.primitiveRefs.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
  });
  device.queue.writeBuffer(refsBuffer, 0, bvhResult.primitiveRefs);

  // 5. パイプライン作成
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

  // --- ループ用変数 (GC対策: ループ内でのnewを排除) ---
  const frameData = new Uint32Array(1);
  const dispatchX = Math.ceil(canvas.width / 8);
  const dispatchY = Math.ceil(canvas.height / 8);

  const copySize: GPUExtent3DStrict = {
    width: canvas.width,
    height: canvas.height,
    depthOrArrayLayers: 1
  };
  const copySrc = { texture: renderTarget };
  // textureプロパティは毎フレーム書き換えるため型アサーションで初期化
  const copyDst = { texture: null as unknown as GPUTexture };

  let frameCount = 0;
  let isRendering = false;
  let lastTime = performance.now();
  let frameCountTimer = 0;

  // --- 6. レンダリングループ ---
  const renderFrame = () => {
    if (!isRendering) return;

    const now = performance.now();
    frameCount++;
    frameCountTimer++;

    // A. フレーム情報更新
    frameData[0] = frameCount;
    device.queue.writeBuffer(frameUniformBuffer, 0, frameData);

    // B. コマンドエンコード
    const canvasTexture = context.getCurrentTexture();
    copyDst.texture = canvasTexture; // Canvasテクスチャをセット

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(dispatchX, dispatchY);
    passEncoder.end();

    // C. Blit (中間バッファ -> Canvas)
    commandEncoder.copyTextureToTexture(copySrc, copyDst, copySize);
    device.queue.submit([commandEncoder.finish()]);

    // D. FPS計測
    if (now - lastTime >= 1000) {
      const fps = frameCountTimer;
      const ms = (1000 / fps).toFixed(2);
      statsDiv.textContent = `FPS: ${fps} | Frame Time: ${ms}ms`;
      frameCountTimer = 0;
      lastTime = now;
    }

    requestAnimationFrame(renderFrame);
  };

  // --- イベントハンドラ ---
  btn.addEventListener("click", () => {
    if (isRendering) {
      console.log("Stop Rendering");
      isRendering = false;
      btn.textContent = "Restart Rendering";
    } else {
      console.log("Reset & Start Rendering");
      frameCount = 0;
      // 蓄積バッファクリア
      const zeroData = new Float32Array(bufferSize / 4);
      device.queue.writeBuffer(accumulateBuffer, 0, zeroData);

      isRendering = true;
      renderFrame();
      btn.textContent = "Stop Rendering";
    }
  });
}

initAndRender().catch(console.error);
