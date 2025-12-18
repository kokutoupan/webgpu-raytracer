import { WebGPURenderer } from "./renderer";
import { WorldBridge } from "./world-bridge";
import { VideoRecorder } from "./recorder/VideoRecorder";

// --- Worker State ---
let renderer: WebGPURenderer | null = null;
let worldBridge: WorldBridge | null = null;
let recorder: VideoRecorder | null = null;
let canvas: OffscreenCanvas | null = null;
let animationId: number | null = null;

let isRendering = false;
let frameCount = 0;
let totalFrameCount = 0;
let frameTimer = 0;
let lastTime = 0; // performance.now() is available in worker
// ★追加: 二重実行防止用のロックフラグ
let isProcessing = false;
let needsRestoration = false;

// Configuration
let currentDepth = 10;
let currentSPP = 1;
let updateInterval = 4;
let renderConfig = { width: 400, height: 400 };
const TILE_SIZE = 1024;

// --- Initialization ---
async function init(newCanvas: OffscreenCanvas, width: number, height: number) {
  canvas = newCanvas;
  renderConfig.width = width;
  renderConfig.height = height;

  renderer = new WebGPURenderer(canvas);
  worldBridge = new WorldBridge();

  try {
    await renderer.init();
    await worldBridge.initWasm();
    recorder = new VideoRecorder(renderer, worldBridge, canvas);

    // Initial setup
    renderer.updateScreenSize(width, height);
    renderer.buildPipeline(currentDepth, currentSPP);

    postMessage({ type: "init-complete" });
  } catch (e: any) {
    postMessage({ type: "error", message: e.toString() });
  }
}

// --- Loading Scene ---
async function loadScene(
  name: string,
  objSource?: string,
  glbData?: Uint8Array
) {
  if (!worldBridge || !renderer) return;

  stopRenderLoop(); // Ensures isRendering = false and timer cleared

  postMessage({ type: "status", message: `Loading Scene: ${name}...` });

  try {
    worldBridge.loadScene(name, objSource, glbData);
    worldBridge.printStats();

    await renderer.loadTexturesFromWorld(worldBridge);
    await uploadSceneBuffers();

    // Update Camera defaults
    renderer.updateScreenSize(renderConfig.width, renderConfig.height);
    worldBridge.updateCamera(renderConfig.width, renderConfig.height);
    renderer.updateSceneUniforms(worldBridge.cameraData, 0);

    renderer.recreateBindGroup();
    renderer.resetAccumulation();
    frameCount = 0;

    postMessage({
      type: "scene-loaded",
      animList: worldBridge.getAnimationList(),
    });

    // startRenderLoop(); // Auto-start disabled by user request
  } catch (e: any) {
    postMessage({ type: "error", message: "Load Failed: " + e.toString() });
  }
}

async function uploadSceneBuffers() {
  // ... (unchanged)
  if (!renderer || !worldBridge) return;
  renderer.updateCombinedGeometry(
    worldBridge.vertices,
    worldBridge.normals,
    worldBridge.uvs
  );
  renderer.updateCombinedBVH(worldBridge.tlas, worldBridge.blas);
  renderer.updateBuffer("index", worldBridge.indices);
  renderer.updateBuffer("attr", worldBridge.attributes);
  renderer.updateBuffer("instance", worldBridge.instances);
}

// Resurrect resources after dispose()
async function restoreRenderer() {
  if (!renderer || !worldBridge) return;
  postMessage({ type: "status", message: "Restoring Renderer..." });

  // 1. Recreate Textures
  await renderer.loadTexturesFromWorld(worldBridge);

  // 2. Recreate Buffers & Accel Structs
  await uploadSceneBuffers();

  // 3. Screen Res (Target & Accumulator)
  renderer.updateScreenSize(renderConfig.width, renderConfig.height);

  // 4. Update Camera & Uniforms
  renderer.ensureSceneUniformBuffer();
  worldBridge.updateCamera(renderConfig.width, renderConfig.height);
  renderer.updateSceneUniforms(worldBridge.cameraData, 0);

  // 5. BindGroup & Reset
  renderer.recreateBindGroup();
  renderer.resetAccumulation();

  // Reset per-frame counters
  frameCount = 0;
  // totalFrameCount should persist or reset? Usually reset for visual consistency
  totalFrameCount = 0;
  frameTimer = 0;
}

// --- Render Loop ---
// ループ開始関数
function startRenderLoop() {
  if (isRendering) return;
  isRendering = true;
  lastTime = performance.now();
  // setTimeoutではなくrAFを使うことで、ブラウザの負荷状況に合わせる
  self.requestAnimationFrame(renderFrame);
}

// ループ停止関数
function stopRenderLoop() {
  isRendering = false;
  // requestAnimationFrameは自動で止まるが、念のためフラグを下げる
  isProcessing = false;
  if (animationId) {
    self.cancelAnimationFrame(animationId); // clearTimeoutから変更
    animationId = null;
  }
}

// ★最重要: 修正されたレンダリングループ
async function renderFrame() {
  // 1. 基本的な実行条件チェック
  if (!isRendering || !renderer || !worldBridge || !canvas) return;

  // 録画中はRecorder側がループを回すので、ここでは何もしない
  if (recorder && recorder.recording) return;

  // 2. 多重実行ガード (前のフレームが終わっていなければスキップ)
  if (isProcessing) {
    if (isRendering) self.requestAnimationFrame(renderFrame);
    return;
  }
  isProcessing = true;

  try {
    // --- 【重要】ここからコマンド発行終了まで await は禁止 (アトミック実行) ---
    // 途中で await すると、その隙に onmessage (リサイズ等) が走り、
    // テクスチャやバッファが破棄されてクラッシュの原因になります。

    // Update Logic
    if (updateInterval > 0 && frameCount >= updateInterval) {
      worldBridge.update(totalFrameCount / updateInterval / 60);

      let needsRebind = false;
      needsRebind ||= renderer.updateCombinedBVH(
        worldBridge.tlas,
        worldBridge.blas
      );
      needsRebind ||= renderer.updateBuffer("instance", worldBridge.instances);

      worldBridge.updateCamera(renderConfig.width, renderConfig.height);
      renderer.updateSceneUniforms(worldBridge.cameraData, 0);

      if (needsRebind) renderer.recreateBindGroup();
      renderer.resetAccumulation();
      frameCount = 0;
    }

    frameCount++;
    totalFrameCount++;
    frameTimer++;

    // Render Logic
    if (isRendering) {
      const width = renderConfig.width;
      const height = renderConfig.height;

      // タイルサイズは元の設定(1024等)でも、アトミック実行なら競合は起きない
      // 安全のため256-512程度を推奨するが、ここではユーザー設定に従う
      const tilesX = Math.ceil(width / TILE_SIZE);
      const tilesY = Math.ceil(height / TILE_SIZE);

      for (let ty = 0; ty < tilesY; ty++) {
        for (let tx = 0; tx < tilesX; tx++) {
          const offsetX = tx * TILE_SIZE;
          const offsetY = ty * TILE_SIZE;
          const cw = Math.min(TILE_SIZE, width - offsetX);
          const ch = Math.min(TILE_SIZE, height - offsetY);

          const encoder = renderer.device.createCommandEncoder();
          renderer.encodeTileCommand(
            encoder,
            offsetX,
            offsetY,
            cw,
            ch,
            frameCount
          );
          renderer.device.queue.submit([encoder.finish()]);
        }
      }

      // Final Present
      const finalEncoder = renderer.device.createCommandEncoder();
      renderer.present(finalEncoder);
      renderer.device.queue.submit([finalEncoder.finish()]);
    }
    // --- コマンド発行終了 (ここまでノンストップ) ---

    // 3. GPU完了待機 (ここで初めて yield する)
    // これにより、GPUキューの詰まり(TDR)を防ぎつつ、
    // 待機中に届いたメッセージ(リサイズ操作等)は次のフレームで安全に反映される
    await renderer.device.queue.onSubmittedWorkDone();

    // Stats Logic
    const now = performance.now();
    if (now - lastTime >= 1000) {
      const fps = (frameTimer * 1000) / (now - lastTime);
      postMessage({ type: "stats", fps: fps, spp: frameCount });
      frameTimer = 0;
      lastTime = now;
    }
  } catch (err) {
    console.error("Render Error:", err);
    // エラー時は停止
    isRendering = false;
  } finally {
    // 4. ロック解除と次フレーム予約
    isProcessing = false;
    if (isRendering) {
      self.requestAnimationFrame(renderFrame);
    }
  }
}

// --- Message Handler ---
self.onmessage = async (e) => {
  const { type, payload } = e.data;

  switch (type) {
    case "init":
      await init(payload.canvas, payload.width, payload.height);
      break;

    case "resize":
      if (renderer && worldBridge) {
        renderConfig.width = payload.width;
        renderConfig.height = payload.height;
        renderer.updateScreenSize(payload.width, payload.height);
        worldBridge.updateCamera(payload.width, payload.height);
        renderer.updateSceneUniforms(worldBridge.cameraData, 0);
        renderer.recreateBindGroup();
        renderer.resetAccumulation();
        frameCount = 0;
        totalFrameCount = 0;
      }
      break;

    case "load-scene":
      await loadScene(payload.name, payload.objSource, payload.glbData);
      break;

    case "set-anim":
      if (worldBridge) {
        worldBridge.setAnimation(payload.index);
      }
      break;

    case "update-config":
      // { depth, spp, updateInterval }
      if (payload.depth) currentDepth = payload.depth;
      if (payload.spp) currentSPP = payload.spp;
      if (payload.updateInterval !== undefined)
        updateInterval = payload.updateInterval;

      if (renderer && (payload.depth || payload.spp)) {
        stopRenderLoop();
        renderer.buildPipeline(currentDepth, currentSPP);
        renderer.recreateBindGroup();
        renderer.resetAccumulation();
        frameCount = 0;
        startRenderLoop();
      }
      break;

    case "start-render":
      if (needsRestoration) {
        await restoreRenderer();
        needsRestoration = false;
      }
      if (!isRendering) startRenderLoop();
      break;

    case "stop-render":
      stopRenderLoop();
      break;

    case "start-record":
      // Payload: config, role ("host" | "worker")
      if (!recorder) return;
      stopRenderLoop(); // Stop RAF

      if (needsRestoration) {
        await restoreRenderer();
        needsRestoration = false;
      }

      try {
        if (payload.role === "worker") {
          // Return chunks
          const chunks = await recorder.recordChunks(payload.config, (f, t) => {
            postMessage({
              type: "record-progress",
              current: f,
              total: t,
              stage: "recording",
            });
          });
          postMessage({ type: "record-result-chunks", chunks });
        } else {
          // Download directly
          await recorder.record(
            payload.config,
            (f, t) => {
              postMessage({
                type: "record-progress",
                current: f,
                total: t,
                stage: "recording",
              });
            },
            (buffer) => {
              (postMessage as any)({ type: "record-complete", buffer }, [
                buffer,
              ]);
            }
          );
        }
      } catch (e: any) {
        postMessage({
          type: "error",
          message: "Recording failed: " + e.toString(),
        });
      } finally {
        // Resume
        if (renderer) renderer.dispose(); // Free huge VRAM
        needsRestoration = true; // Mark as needing restore next time
        // await restoreRenderer(); // No immediate restore
        // startRenderLoop(); // No auto-resume
      }
      break;
  }
};
