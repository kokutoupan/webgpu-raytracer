import { WebGPURenderer } from "./renderer";
import { WorldBridge } from "./world-bridge";
import { Config } from "./config";
import { UIManager } from "./ui/UIManager";
import { VideoRecorder } from "./recorder/VideoRecorder";
import { SignalingClient } from "./network/SignalingClient";

// --- Global State ---
let isRendering = false;
let currentFileData: string | ArrayBuffer | null = null;
let currentFileType: "obj" | "glb" | null = null;

// --- Modules ---
const ui = new UIManager();
const renderer = new WebGPURenderer(ui.canvas);
const worldBridge = new WorldBridge();
const recorder = new VideoRecorder(renderer, worldBridge, ui.canvas);
const signaling = new SignalingClient();

// --- Main Application Loop ---
let frameCount = 0;
let totalFrameCount = 0;
let frameTimer = 0;
let lastTime = performance.now();

// --- Functions ---
const rebuildPipeline = () => {
  const depth = parseInt(ui.inputDepth.value, 10) || Config.defaultDepth;
  const spp = parseInt(ui.inputSPP.value, 10) || Config.defaultSPP;
  renderer.buildPipeline(depth, spp);
};

const updateResolution = () => {
  const { width, height } = ui.getRenderConfig();
  renderer.updateScreenSize(width, height);

  if (worldBridge.hasWorld) {
    worldBridge.updateCamera(width, height);
    renderer.updateSceneUniforms(worldBridge.cameraData, 0);
  }
  renderer.recreateBindGroup();
  renderer.resetAccumulation();
  frameCount = 0;
  totalFrameCount = 0;
};

const loadScene = async (name: string, autoStart = true) => {
  isRendering = false;
  console.log(`Loading Scene: ${name}...`);

  let objSource: string | undefined;
  let glbData: Uint8Array | undefined;

  if (name === "viewer" && currentFileData) {
    if (currentFileType === "obj") objSource = currentFileData as string;
    else if (currentFileType === "glb")
      glbData = new Uint8Array(currentFileData as ArrayBuffer);
  }

  worldBridge.loadScene(name, objSource, glbData);
  worldBridge.printStats();

  await renderer.loadTexturesFromWorld(worldBridge);
  await uploadSceneBuffers();

  updateResolution();
  ui.updateAnimList(worldBridge.getAnimationList());

  if (autoStart) {
    isRendering = true;
    ui.updateRenderButton(true);
  }
};

const uploadSceneBuffers = async () => {
  renderer.updateCombinedGeometry(
    worldBridge.vertices,
    worldBridge.normals,
    worldBridge.uvs
  );
  renderer.updateCombinedBVH(worldBridge.tlas, worldBridge.blas);
  renderer.updateBuffer("index", worldBridge.indices);
  renderer.updateBuffer("attr", worldBridge.attributes);
  renderer.updateBuffer("instance", worldBridge.instances);
};

// --- Render Loop ---
const renderFrame = () => {
  if (recorder.recording) return;

  requestAnimationFrame(renderFrame);
  if (!isRendering || !worldBridge.hasWorld) return;

  let updateInterval = parseInt(ui.inputUpdateInterval.value, 10) || 0;
  if (updateInterval < 0) updateInterval = 0;

  if (updateInterval > 0 && frameCount >= updateInterval) {
    worldBridge.update(totalFrameCount / updateInterval / 60);

    let needsRebind = false;
    needsRebind ||= renderer.updateCombinedBVH(
      worldBridge.tlas,
      worldBridge.blas
    );
    needsRebind ||= renderer.updateBuffer("instance", worldBridge.instances);
    needsRebind ||= renderer.updateCombinedGeometry(
      worldBridge.vertices,
      worldBridge.normals,
      worldBridge.uvs
    );
    needsRebind ||= renderer.updateBuffer("index", worldBridge.indices);
    needsRebind ||= renderer.updateBuffer("attr", worldBridge.attributes);

    worldBridge.updateCamera(ui.canvas.width, ui.canvas.height);
    renderer.updateSceneUniforms(worldBridge.cameraData, 0);

    if (needsRebind) renderer.recreateBindGroup();
    renderer.resetAccumulation();
    frameCount = 0;
  }

  frameCount++;
  frameTimer++;
  totalFrameCount++;

  renderer.render(frameCount);

  const now = performance.now();
  if (now - lastTime >= 1000) {
    ui.updateStats(frameTimer, 1000 / frameTimer, frameCount);
    frameTimer = 0;
    lastTime = now;
  }
};

// --- Event Binding ---
const bindEvents = () => {
  ui.onRenderStart = () => {
    isRendering = true;
  };
  ui.onRenderStop = () => {
    isRendering = false;
  };
  ui.onSceneSelect = (name) => loadScene(name, false);
  ui.onResolutionChange = updateResolution;

  ui.onRecompile = (depth, spp) => {
    isRendering = false;
    renderer.buildPipeline(depth, spp);
    renderer.recreateBindGroup();
    renderer.resetAccumulation();
    frameCount = 0;
    isRendering = true;
  };

  ui.onFileSelect = async (f) => {
    const ext = f.name.split(".").pop()?.toLowerCase();
    if (ext === "obj") {
      currentFileData = await f.text();
      currentFileType = "obj";
    } else {
      currentFileData = await f.arrayBuffer();
      currentFileType = "glb";
    }
    ui.sceneSelect.value = "viewer";
    loadScene("viewer", false);
  };

  ui.onAnimSelect = (idx) => worldBridge.setAnimation(idx);

  ui.onRecordStart = async () => {
    if (recorder.recording) return;
    isRendering = false;
    ui.setRecordingState(true);

    const config = ui.getRenderConfig();
    try {
      await recorder.record(
        config,
        (f, t) =>
          ui.setRecordingState(
            true,
            `Rec: ${f}/${t} (${Math.round((f / t) * 100)}%)`
          ),
        (url) => {
          const a = document.createElement("a");
          a.href = url;
          a.download = `raytrace_${Date.now()}.webm`;
          a.click();
          URL.revokeObjectURL(url);
        }
      );
    } catch (e) {
      alert("Recording failed.");
    } finally {
      ui.setRecordingState(false);
      isRendering = true;
      ui.updateRenderButton(true);
      requestAnimationFrame(renderFrame);
    }
  };

  // Network Events
  ui.onConnectHost = () => signaling.connect("host");
  ui.onConnectWorker = () => signaling.connect("worker");

  ui.onSendScene = async () => {
    if (!currentFileData || !currentFileType) {
      alert("No scene loaded!");
      return;
    }
    ui.setSendSceneText("Sending...");
    ui.setSendSceneEnabled(false);

    const config = ui.getRenderConfig();
    await signaling.broadcastScene(currentFileData, currentFileType, config);

    ui.setSendSceneText("Send Scene");
    ui.setSendSceneEnabled(true);
  };

  // Signaling Callbacks
  signaling.onStatusChange = (msg) => ui.setStatus(`Status: ${msg}`);
  signaling.onWorkerJoined = (_id) => ui.setSendSceneEnabled(true);

  signaling.onSceneReceived = async (data, config) => {
    console.log("Scene received successfully.");

    // Update UI
    ui.setRenderConfig(config);

    // Load Scene
    currentFileType = config.fileType;
    if (config.fileType === "obj") currentFileData = data as string;
    else currentFileData = data as ArrayBuffer;

    ui.sceneSelect.value = "viewer";
    await loadScene("viewer", false);

    if (config.anim !== undefined) {
      ui.animSelect.value = config.anim.toString();
      worldBridge.setAnimation(config.anim);
    }
  };
};

// --- Entry Point ---
async function bootstrap() {
  try {
    await renderer.init();
    await worldBridge.initWasm();
  } catch (e) {
    alert("Init failed: " + e);
    return;
  }

  bindEvents();
  rebuildPipeline();
  updateResolution();

  loadScene("cornell", false);
  requestAnimationFrame(renderFrame);
}

bootstrap().catch(console.error);
