import { VideoRecorder } from "./recorder/VideoRecorder";
import { SignalingClient } from "./network/SignalingClient";
import { DistributedHost } from "./distributed/DistributedHost";
import { DistributedWorker } from "./distributed/DistributedWorker";
import { WebGPURenderer } from "./renderer";
import { WorldBridge } from "./world-bridge";
import { UIManager } from "./ui/UIManager";
import { Config } from "./config";
import type { RenderConfig } from "./network/Protocol";

// --- Global State ---
let isRendering = false;
let currentFileData: string | ArrayBuffer | null = null;
let currentFileType: "obj" | "glb" | null = null;
let currentRole: "host" | "worker" | null = null;

// --- Worker State ---
const BATCH_SIZE = 20;

// --- Modules ---
const ui = new UIManager();
const renderer = new WebGPURenderer(ui.canvas);
const worldBridge = new WorldBridge();
const recorder = new VideoRecorder(renderer, worldBridge, ui.canvas);
const signaling = new SignalingClient();

const dHost = new DistributedHost(signaling, ui);
const dWorker = new DistributedWorker(signaling, renderer, ui, recorder);

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
  // ★ 追加: 録画中はリサイズ処理（コンテキストの再構成）を行わないようにする
  if (recorder.recording) {
    console.warn(
      "Resize blocked during recording to prevent resource invalidation."
    );
    return;
  }
  const { width, height } = ui.getRenderConfig();
  renderer.updateScreenSize(width, height);

  if (worldBridge.hasWorld) {
    worldBridge.updateCamera(width, height);
    renderer.updateSceneUniforms(
      worldBridge.cameraData,
      0,
      worldBridge.lightCount
    );
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
    else if (currentFileType === "glb") {
      // SLICE to avoid detaching the original buffer, which is needed for distribution!
      glbData = new Uint8Array(currentFileData as ArrayBuffer).slice(0);
    }
  }

  await worldBridge.loadScene(name, objSource, glbData);
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
  renderer.updateBuffer("topology", worldBridge.mesh_topology);
  renderer.updateBuffer("instance", worldBridge.instances);
  renderer.updateBuffer("lights", worldBridge.lights); // Added
  renderer.updateSceneUniforms(
    worldBridge.cameraData,
    0,
    worldBridge.lightCount
  );
  await renderer.device.queue.onSubmittedWorkDone();
};

// --- Render Loop ---
const renderFrame = () => {
  if (recorder.recording) return;

  requestAnimationFrame(renderFrame);
  if (!isRendering || !worldBridge.hasWorld) return;

  let updateInterval = parseInt(ui.inputUpdateInterval.value, 10) || 0;

  // updateInterval <= 0 means disabled
  if (updateInterval > 0 && frameCount >= updateInterval) {
    worldBridge.update(totalFrameCount / (updateInterval || 1) / 60);
  }

  if (worldBridge.hasNewData) {
    let needsRebind = false;
    needsRebind ||= renderer.updateCombinedBVH(
      worldBridge.tlas,
      worldBridge.blas
    );
    needsRebind ||= renderer.updateBuffer("instance", worldBridge.instances);

    if (worldBridge.hasNewGeometry) {
      needsRebind ||= renderer.updateCombinedGeometry(
        worldBridge.vertices,
        worldBridge.normals,
        worldBridge.uvs
      );
      needsRebind ||= renderer.updateBuffer(
        "topology",
        worldBridge.mesh_topology
      );
      needsRebind ||= renderer.updateBuffer("lights", worldBridge.lights); // Added
      worldBridge.hasNewGeometry = false;
    }

    worldBridge.updateCamera(ui.canvas.width, ui.canvas.height);
    renderer.updateSceneUniforms(
      worldBridge.cameraData,
      0,
      worldBridge.lightCount
    );

    if (needsRebind) renderer.recreateBindGroup();
    renderer.resetAccumulation();
    frameCount = 0;
    worldBridge.hasNewData = false;
  }

  frameCount++;
  frameTimer++;
  totalFrameCount++;

  renderer.compute(frameCount);
  renderer.present();

  const now = performance.now();
  if (now - lastTime >= 1000) {
    ui.updateStats(frameTimer, 1000 / frameTimer, frameCount);
    frameTimer = 0;
    lastTime = now;
  }
};

// --- Signaling Callbacks (Coordination) ---

signaling.onStatusChange = (msg) => ui.setStatus(`Status: ${msg}`);

signaling.onWorkerStatus = async (id, hasScene, job) => {
  const result = await dHost.onWorkerStatus(id, hasScene, job);
  if (result === "NEED_SCENE") {
    console.log(`[Host] Worker ${id} needs scene. Syncing...`);
    await dHost.sendSceneHelper(currentFileData, currentFileType, id);
  }
};

signaling.onRenderResult = async (chunks, startFrame, workerId) => {
  const result = await dHost.onRenderResult(chunks, startFrame, workerId);
  if (result === "ALL_COMPLETE") {
    console.log("[Host] All jobs complete. Muxing and downloading...");
    ui.setStatus("Muxing...");
    await dHost.muxAndDownload();
  }
};

signaling.onSceneReceived = async (data, config) => {
  console.log("[Worker] Received Scene from Host.");
  await dWorker.onSceneReceived(data, config);

  currentFileType = config.fileType;
  if (config.fileType === "obj") currentFileData = data as string;
  else currentFileData = data as ArrayBuffer;

  ui.sceneSelect.value = config.sceneName || "viewer";
  await loadScene(config.sceneName || "viewer", false);

  if (config.anim !== undefined) {
    ui.animSelect.value = config.anim.toString();
    worldBridge.setAnimation(config.anim);
  }

  dWorker.isDistributedSceneLoaded = true;
  dWorker.isSceneLoading = false;

  console.log("[Worker] Distributed Scene Loaded. Signaling Host.");
  await signaling.sendSceneLoaded();

  dWorker.handlePendingRenderRequest();
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

    if (currentRole === "host") {
      const workers = signaling.getWorkerIds();
      dHost.distributedConfig = ui.getRenderConfig() as RenderConfig;
      const totalFrames = Math.ceil(
        dHost.distributedConfig.fps * dHost.distributedConfig.duration
      );

      if (
        !confirm(
          `Distribute recording? (Workers: ${workers.length})\nAuto Scene Sync enabled.`
        )
      )
        return;

      // Init Job Queue
      dHost.jobQueue = [];
      dHost.pendingChunks.clear();
      dHost.completedJobs = 0;
      dHost.activeJobs.clear();

      for (let f = 0; f < totalFrames; f += BATCH_SIZE) {
        const count = Math.min(BATCH_SIZE, totalFrames - f);
        dHost.jobQueue.push({ start: f, count });
      }
      dHost.totalJobs = dHost.jobQueue.length;

      // Init Status
      workers.forEach((w) => dHost.workerStatus.set(w, "idle"));

      ui.setStatus(
        `Distributed Progress: 0 / ${dHost.totalJobs} jobs (Waiting for workers...)`
      );

      // Broadcast Scene to EXISTING workers
      if (workers.length > 0) {
        ui.setStatus("Syncing Scene to Workers...");
        signaling.sendRenderStart();
        await dHost.sendSceneHelper(currentFileData, currentFileType);
      } else {
        console.log("No workers yet. Waiting...");
      }
    } else {
      // Local Recording
      isRendering = false;
      ui.setRecordingState(true);
      const config = ui.getRenderConfig();
      try {
        const timer = performance.now();
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
        console.log(`Recording took ${performance.now() - timer}[ms]`);
      } catch (e) {
        alert("Recording failed.");
      } finally {
        ui.setRecordingState(false);
        isRendering = false;
        ui.updateRenderButton(false);
        requestAnimationFrame(renderFrame);
      }
    }
  };

  ui.onConnectHost = () => {
    if (currentRole === "host") {
      signaling.disconnect();
      currentRole = null;
      ui.setConnectionState(null);
    } else {
      signaling.connect("host");
      currentRole = "host";
      ui.setConnectionState("host");
    }
  };

  ui.onConnectWorker = () => {
    if (currentRole === "worker") {
      signaling.disconnect();
      currentRole = null;
      ui.setConnectionState(null);
    } else {
      signaling.connect("worker");
      currentRole = "worker";
      ui.setConnectionState("worker");
    }
  };

  // Initial State
  ui.setConnectionState(null);
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
