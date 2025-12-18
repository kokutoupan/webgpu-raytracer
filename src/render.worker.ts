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

// Configuration
let currentDepth = 10;
let currentSPP = 1;
let updateInterval = 4;
let renderConfig = { width: 400, height: 400 };

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

  isRendering = false;
  if (animationId) self.clearTimeout(animationId);

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

    startRenderLoop();
  } catch (e: any) {
    postMessage({ type: "error", message: "Load Failed: " + e.toString() });
  }
}

async function uploadSceneBuffers() {
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

// --- Render Loop ---
function startRenderLoop() {
  isRendering = true;
  lastTime = performance.now();
  renderFrame();
}

function stopRenderLoop() {
  isRendering = false;
  if (animationId) {
    self.clearTimeout(animationId);
    animationId = null;
  }
}

async function renderFrame() {
  if (!isRendering || !renderer || !worldBridge || !canvas) return;

  if (recorder && recorder.recording) {
    // If recording, we don't use the standard RAF loop in the same way,
    // or we let the recorder drive it. Ideally, we shouldn't mix them.
    // If recorder handles the loop, we stop this one.
    return;
  }

  // Prevent freezing the browser by waiting for the GPU
  await renderer.device.queue.onSubmittedWorkDone();

  // Use setTimeout instead of requestAnimationFrame for better stability in Firefox
  // requestAnimationFrame in workers can sometimes cause instabilities with heavy WebGPU usage
  animationId = self.setTimeout(renderFrame, 0);

  if (!worldBridge.hasWorld) return;

  // Animation Update
  if (updateInterval > 0 && frameCount >= updateInterval) {
    worldBridge.update(totalFrameCount / updateInterval / 60);

    let needsRebind = false;
    needsRebind ||= renderer.updateCombinedBVH(
      worldBridge.tlas,
      worldBridge.blas
    );
    needsRebind ||= renderer.updateBuffer("instance", worldBridge.instances);

    // Geometry might change if we support deformation later, but for now assuming rigid
    needsRebind ||= renderer.updateCombinedGeometry(
      worldBridge.vertices,
      worldBridge.normals,
      worldBridge.uvs
    );
    needsRebind ||= renderer.updateBuffer("index", worldBridge.indices);
    needsRebind ||= renderer.updateBuffer("attr", worldBridge.attributes);

    worldBridge.updateCamera(renderConfig.width, renderConfig.height);
    renderer.updateSceneUniforms(worldBridge.cameraData, 0);

    if (needsRebind) renderer.recreateBindGroup();
    renderer.resetAccumulation();
    frameCount = 0;
  }

  frameCount++;
  totalFrameCount++;
  frameTimer++;

  renderer.render(frameCount);

  // Status Update
  const now = performance.now();
  if (now - lastTime >= 1000) {
    const fps = (frameTimer * 1000) / (now - lastTime);
    postMessage({
      type: "stats",
      fps: fps,
      spp: frameCount,
    });
    frameTimer = 0;
    lastTime = now;
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
      if (!isRendering) startRenderLoop();
      break;

    case "stop-render":
      stopRenderLoop();
      break;

    case "start-record":
      // Payload: config, role ("host" | "worker")
      if (!recorder) return;
      stopRenderLoop(); // Stop RAF

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
            (url) => {
              postMessage({ type: "record-complete", url });
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
        startRenderLoop();
      }
      break;
  }
};
