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
let currentRole: "host" | "worker" | null = null;

// --- Distributed State ---
let jobQueue: { start: number; count: number }[] = [];
let pendingChunks: Map<number, any[]> = new Map(); // startFrame -> SerializedChunk[]
let completedJobs = 0;
let totalJobs = 0;
let totalRenderFrames = 0;
let distributedConfig: any = null;
let workerStatus: Map<string, "idle" | "loading" | "busy"> = new Map();
let activeJobs: Map<string, { start: number; count: number }> = new Map();

// --- Worker State ---
let isSceneLoading = false;
let pendingRenderRequest: { start: number; count: number; config: any } | null =
  null;
const BATCH_SIZE = 20; // Moved to global

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
  renderer.updateBuffer("index", worldBridge.indices);
  renderer.updateBuffer("attr", worldBridge.attributes);
  renderer.updateBuffer("instance", worldBridge.instances);
  renderer.updateSceneUniforms(worldBridge.cameraData, 0);
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
    worldBridge.hasNewData = false;
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

// --- Distributed Helpers ---
const sendSceneHelper = async (workerId?: string) => {
  const currentScene = ui.sceneSelect.value;
  const isProcedural = currentScene !== "viewer";

  if (!isProcedural && (!currentFileData || !currentFileType)) return;

  const config = ui.getRenderConfig();
  const sceneName = isProcedural ? currentScene : undefined;

  // For procedural, we send dummy data but with sceneName in config
  const fileData = isProcedural ? "DUMMY" : currentFileData!;
  const fileType = isProcedural ? "obj" : currentFileType!;

  // Inject sceneName
  (config as any).sceneName = sceneName;

  if (workerId) {
    console.log(`Sending scene to specific worker: ${workerId}`);
    workerStatus.set(workerId, "loading");
    await signaling.sendSceneToWorker(workerId, fileData, fileType, config);
  } else {
    console.log(`Broadcasting scene to all workers...`);
    signaling.getWorkerIds().forEach((id) => workerStatus.set(id, "loading"));
    await signaling.broadcastScene(fileData, fileType, config);
  }
};

const assignJob = async (workerId: string) => {
  // Only assign if worker is IDLE
  if (workerStatus.get(workerId) !== "idle") {
    console.log(
      `Worker ${workerId} is ${workerStatus.get(
        workerId
      )}, skipping assignment.`
    );
    return;
  }

  if (jobQueue.length === 0) return;
  const job = jobQueue.shift();
  if (!job) return;

  workerStatus.set(workerId, "busy"); // Mark as busy
  activeJobs.set(workerId, job); // Track active job
  console.log(
    `Assigning Job ${job.start} - ${job.start + job.count} to ${workerId}`
  );
  await signaling.sendRenderRequest(workerId, job.start, job.count, {
    ...distributedConfig,
    fileType: "obj",
  });
};

const muxAndDownload = async () => {
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
    if (!chunks) continue;
    for (const c of chunks) {
      mult.addVideoChunk(
        new EncodedVideoChunk({
          type: c.type,
          timestamp: c.timestamp,
          duration: c.duration,
          data: c.data,
        }),
        { decoderConfig: c.decoderConfig }
      );
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

const executeWorkerRender = async (
  startFrame: number,
  frameCount: number,
  config: any
) => {
  console.log(
    `[Worker] Starting Render: Frames ${startFrame} - ${
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

const handlePendingRenderRequest = async () => {
  if (!pendingRenderRequest) return;
  const { start, count, config } = pendingRenderRequest;
  pendingRenderRequest = null;

  // Execute the delayed request
  await executeWorkerRender(start, count, config);
};

// --- Signaling Callbacks (Global) ---
signaling.onStatusChange = (msg) => ui.setStatus(`Status: ${msg}`);

signaling.onWorkerLeft = (id) => {
  console.log(`Worker Left: ${id}`);
  ui.setStatus(`Worker Left: ${id}`);

  workerStatus.delete(id);

  // Check if it had an active job
  const failedJob = activeJobs.get(id);
  if (failedJob) {
    console.warn(`Worker ${id} failed job ${failedJob.start}. Re-queueing.`);
    jobQueue.unshift(failedJob); // Put back at font
    activeJobs.delete(id);

    ui.setStatus(`Re-queued Job ${failedJob.start}`);
  }
};

signaling.onWorkerReady = (id: string) => {
  console.log(`Worker ${id} is READY`);
  ui.setStatus(`Worker ${id} Ready!`);
  workerStatus.set(id, "idle");

  if (currentRole === "host" && jobQueue.length > 0) {
    assignJob(id);
  }
};

signaling.onWorkerJoined = (id) => {
  ui.setStatus(`Worker Joined: ${id}`);

  workerStatus.set(id, "idle");

  if (currentRole === "host" && jobQueue.length > 0) {
    sendSceneHelper(id);
  }
};

signaling.onRenderRequest = async (startFrame, frameCount, config) => {
  console.log(
    `[Worker] Received Render Request: Frames ${startFrame} - ${
      startFrame + frameCount
    }`
  );

  if (isSceneLoading) {
    console.log(
      `[Worker] Scene loading in progress. Queueing Render Request for ${startFrame}`
    );
    pendingRenderRequest = { start: startFrame, count: frameCount, config };
    return;
  }

  await executeWorkerRender(startFrame, frameCount, config);
};

signaling.onRenderResult = async (chunks, startFrame, workerId) => {
  console.log(
    `[Host] Received ${chunks.length} chunks for ${startFrame} from ${workerId}`
  );

  pendingChunks.set(startFrame, chunks);
  completedJobs++;
  ui.setStatus(`Distributed Progress: ${completedJobs} / ${totalJobs} jobs`);

  workerStatus.set(workerId, "idle");
  activeJobs.delete(workerId);

  await assignJob(workerId);

  if (completedJobs >= totalJobs) {
    console.log("All jobs complete. Muxing...");
    ui.setStatus("Muxing...");
    await muxAndDownload();
  }
};

signaling.onSceneReceived = async (data, config) => {
  console.log("Scene received successfully.");
  isSceneLoading = true;

  ui.setRenderConfig(config);

  currentFileType = config.fileType;
  if (config.fileType === "obj") currentFileData = data as string;
  else currentFileData = data as ArrayBuffer;

  ui.sceneSelect.value = config.sceneName || "viewer";
  await loadScene(config.sceneName || "viewer", false);

  if (config.anim !== undefined) {
    ui.animSelect.value = config.anim.toString();
    worldBridge.setAnimation(config.anim);
  }

  isSceneLoading = false;

  console.log("Scene Loaded. Sending WORKER_READY.");
  await signaling.sendWorkerReady();

  handlePendingRenderRequest();
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
      distributedConfig = ui.getRenderConfig();
      totalRenderFrames = Math.ceil(
        distributedConfig.fps * distributedConfig.duration
      );

      if (
        !confirm(
          `Distribute recording? (Workers: ${workers.length})\nAuto Scene Sync enabled.`
        )
      )
        return;

      // Init Job Queue
      jobQueue = [];
      pendingChunks.clear();
      completedJobs = 0;
      activeJobs.clear();

      for (let f = 0; f < totalRenderFrames; f += BATCH_SIZE) {
        const count = Math.min(BATCH_SIZE, totalRenderFrames - f);
        jobQueue.push({ start: f, count });
      }
      totalJobs = jobQueue.length;

      // Init Status
      workers.forEach((w) => workerStatus.set(w, "idle"));

      ui.setStatus(
        `Distributed Progress: 0 / ${totalJobs} jobs (Waiting for workers...)`
      );

      // Broadcast Scene to EXISTING workers
      if (workers.length > 0) {
        ui.setStatus("Syncing Scene to Workers...");
        await sendSceneHelper();
      } else {
        console.log("No workers yet. Waiting...");
      }
    } else {
      // Local Recording
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
