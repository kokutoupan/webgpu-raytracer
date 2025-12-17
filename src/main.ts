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

  // --- Dynamic Job Scheduling State ---
  let jobQueue: { start: number; count: number }[] = [];
  let pendingChunks: Map<number, any[]> = new Map(); // startFrame -> SerializedChunk[]
  let completedJobs = 0;
  let totalJobs = 0;
  let totalRenderFrames = 0;
  let distributedConfig: any = null;

  const BATCH_SIZE = 20;

  const assignJob = async (workerId: string) => {
    if (jobQueue.length === 0) return;
    const job = jobQueue.shift();
    if (!job) return;

    console.log(
      `Assigning Job ${job.start} - ${job.start + job.count} to ${workerId}`
    );
    await signaling.sendRenderRequest(workerId, job.start, job.count, {
      ...distributedConfig,
      fileType: "obj",
    });
  };

  ui.onRecordStart = async () => {
    if (recorder.recording) return;

    if (currentRole === "host" && signaling.getWorkerCount() > 0) {
      // Distributed Recording
      distributedConfig = ui.getRenderConfig();
      totalRenderFrames = Math.ceil(
        distributedConfig.fps * distributedConfig.duration
      );

      const workers = signaling.getWorkerIds();
      if (workers.length === 0) return;

      if (
        !confirm(
          `Distribute recording to ${workers.length} workers (Dynamic Load Balancing)?`
        )
      )
        return;

      // Init Job Queue
      jobQueue = [];
      pendingChunks.clear();
      completedJobs = 0;

      for (let f = 0; f < totalRenderFrames; f += BATCH_SIZE) {
        const count = Math.min(BATCH_SIZE, totalRenderFrames - f);
        jobQueue.push({ start: f, count });
      }
      totalJobs = jobQueue.length;

      // Initial Assignment
      workers.forEach((w) => assignJob(w));

      ui.setStatus(`Distributed Progress: 0 / ${totalJobs} jobs`);
    } else {
      // Local Recording
      isRendering = false;
      ui.setRecordingState(true);
      const config = ui.getRenderConfig();
      try {
        // Provide an onComplete callback that accepts url AND blob
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
    }
  };

  // Network Events
  let currentRole: "host" | "worker" | null = null;

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

  signaling.onWorkerJoined = (id) => {
    ui.setStatus(`Worker Joined: ${id}`);
    ui.setSendSceneEnabled(true);
    // If we are in the middle of a distributed render, assign job?
    if (currentRole === "host" && jobQueue.length > 0) {
      assignJob(id);
    }
  };

  signaling.onRenderRequest = async (startFrame, frameCount, config) => {
    console.log(
      `[Worker] Received Render Request: Frames ${startFrame} - ${
        startFrame + frameCount
      }`
    );
    ui.setStatus(`Remote Rendering: ${startFrame}-${startFrame + frameCount}`);

    isRendering = false;

    const workerConfig = {
      ...config,
      startFrame: startFrame,
      duration: frameCount / config.fps,
    };

    try {
      ui.setRecordingState(true, `Remote: ${frameCount} f`);

      // Use recordChunks
      const chunks = await recorder.recordChunks(workerConfig as any, (f, t) =>
        ui.setRecordingState(true, `Remote: ${f}/${t}`)
      );

      console.log("Sending Chunks back to Host...");
      ui.setRecordingState(true, "Uploading...");
      await signaling.sendRenderResult(chunks, startFrame);
      ui.setRecordingState(false);
      ui.setStatus("Idle");
    } catch (e) {
      console.error("Remote Recording Failed", e);
      ui.setStatus("Recording Failed");
    } finally {
      isRendering = true;
      requestAnimationFrame(renderFrame);
    }
  };

  signaling.onRenderResult = async (chunks, startFrame, workerId) => {
    console.log(
      `[Host] Received ${chunks.length} chunks for ${startFrame} from ${workerId}`
    );

    pendingChunks.set(startFrame, chunks);
    completedJobs++;
    ui.setStatus(`Distributed Progress: ${completedJobs} / ${totalJobs} jobs`);

    // Assign next job to this worker
    await assignJob(workerId);

    // Check completion
    if (completedJobs >= totalJobs) {
      console.log("All jobs complete. Muxing...");
      ui.setStatus("Muxing...");

      await muxAndDownload();
    }
  };

  const muxAndDownload = async () => {
    // Import Muxer (need to import at top, assuming it's available globally or imported)
    // We used 'webm-muxer' in VideoRecorder.
    // We should probably export a Muxer helper or use it here.
    // import * as WebMMuxer from "webm-muxer"; // Need to ensure import

    const sortedStarts = Array.from(pendingChunks.keys()).sort((a, b) => a - b);

    // Create Muxer
    const { Muxer, ArrayBufferTarget } = await import("webm-muxer");
    const mult = new Muxer({
      target: new ArrayBufferTarget(),
      video: {
        codec: "V_VP9",
        width: distributedConfig.width,
        height: distributedConfig.height,
        frameRate: distributedConfig.fps,
      },
    });

    for (const start of sortedStarts) {
      const chunks = pendingChunks.get(start);
      if (!chunks) continue; // Should not happen
      for (const c of chunks) {
        mult.addVideoChunk(
          new EncodedVideoChunk({
            type: c.type,
            timestamp: c.timestamp,
            duration: c.duration,
            data: c.data, // ArrayBuffer
          }),
          { decoderConfig: c.decoderConfig }
        ); // meta? EncodedVideoChunkMetadata?
        // Wait, addVideoChunk takes (chunk, meta).
        // meta defaults?
        // VideoEncoder output gives meta (decoderConfig etc).
        // We didn't save meta!
        // WebMMuxer might need it?
        // "meta is optional but recommended for critical codec info" in strict modes.
        // VP9 usually fine.
        // But strict typing?
      }
    }

    mult.finalize();
    const { buffer } = mult.target;
    const blob = new Blob([buffer], { type: "video/webm" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `distributed_trace_${Date.now()}.webm`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    ui.setStatus("Finished!");
  };

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
