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
let isDistributedSceneLoaded = false; // Added: True only after receiving a remote scene
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
  if (!distributedConfig) {
    console.warn(`[Host] No distributed config yet, skipping assignment.`);
    return;
  }

  const job = jobQueue.shift();
  if (!job) return;

  workerStatus.set(workerId, "busy"); // Mark as busy
  activeJobs.set(workerId, job); // Track active job
  console.log(
    `Assigning Job ${job.start} - ${job.start + job.count} to ${workerId}`
  );

  try {
    await signaling.sendRenderRequest(workerId, job.start, job.count, {
      ...distributedConfig,
      fileType: distributedConfig.fileType || "obj",
    });
  } catch (e) {
    console.error(`[Host] Failed to assign job to ${workerId}:`, e);
    // Re-queue the job
    jobQueue.unshift(job);
    activeJobs.delete(workerId);
    workerStatus.set(workerId, "idle");
    ui.setStatus(`Assignment failed for ${workerId}, re-queued.}`);

    // Try again for any idle worker after a short delay
    setTimeout(() => {
      for (const id of workerStatus.keys()) {
        if (workerStatus.get(id) === "idle") assignJob(id);
      }
    }, 1000);
  }
};

const triggerAssignments = () => {
  for (const [id, status] of workerStatus.entries()) {
    if (status === "idle") {
      assignJob(id);
    }
  }
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

// Worker Result Buffer
let bufferedResults: { chunks: any[]; startFrame: number }[] = [];

// Worker Abort Controller
let currentWorkerAbortController: AbortController | null = null;

const executeWorkerRender = async (
  startFrame: number,
  frameCount: number,
  config: any
) => {
  // Lock to prevent overlapping renders
  if (recorder.isRecording) {
    console.warn("[Worker] Already recording/rendering, skipping request.");
    return;
  }

  if (currentWorkerAbortController) {
    currentWorkerAbortController.abort();
  }
  currentWorkerAbortController = new AbortController();
  const signal = currentWorkerAbortController.signal;

  currentWorkerJob = { start: startFrame, count: frameCount };
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

    const chunks = await recorder.recordChunks(
      workerConfig as any,
      (f, t) => ui.setRecordingState(true, `Remote: ${f}/${t}`),
      signal
    );

    console.log("Rendering Complete. Attempting to send result...");
    ui.setRecordingState(true, "Uploading...");

    // Buffer the result first
    bufferedResults.push({ chunks, startFrame });
    await trySendBufferedResults();

    ui.setRecordingState(false);
    ui.setStatus("Idle");
  } catch (e: any) {
    if (e.message === "Aborted") {
      console.log("[Worker] Render Aborted");
    } else {
      console.error("Remote Recording Failed", e);
      ui.setStatus("Recording Failed");
    }
  } finally {
    currentWorkerJob = null;
    currentWorkerAbortController = null;
    isRendering = false;
    requestAnimationFrame(renderFrame);
  }
};

const trySendBufferedResults = async () => {
  if (bufferedResults.length === 0) return;

  console.log(
    `[Worker] Attempting to send ${bufferedResults.length} buffered results...`
  );

  const remaining: typeof bufferedResults = [];
  for (const res of bufferedResults) {
    try {
      await signaling.sendRenderResult(res.chunks, res.startFrame);
      console.log(
        `[Worker] Successfully sent buffered result for ${res.startFrame}`
      );
    } catch (e) {
      console.warn(
        `[Worker] Failed to send result for ${res.startFrame}, keeping in buffer.`
      );
      remaining.push(res);
    }
  }
  bufferedResults = remaining;
};

const handlePendingRenderRequest = async () => {
  if (!pendingRenderRequest) return;
  const { start, count, config } = pendingRenderRequest;
  pendingRenderRequest = null;

  // Execute the delayed request
  await executeWorkerRender(start, count, config);
};

// --- Host Grace Period State ---
const disconnectedWorkers = new Map<
  string,
  { job: { start: number; count: number }; timeoutId: number }
>();
const GRACE_PERIOD_MS = 30000; // 30 seconds

// --- Signaling Callbacks (Global) ---
signaling.onStatusChange = (msg) => ui.setStatus(`Status: ${msg}`);

signaling.onWorkerLeft = (id) => {
  console.log(`Worker Left: ${id}`);
  ui.setStatus(`Worker Left: ${id}`);

  // Check if it had an active job
  const activeJob = activeJobs.get(id);
  if (activeJob) {
    console.warn(
      `Worker ${id} disconnected with active job ${activeJob.start}. Starting grace period.`
    );

    // Clear old timeout if exists
    if (disconnectedWorkers.has(id)) {
      clearTimeout(disconnectedWorkers.get(id)!.timeoutId);
    }

    const timeoutId = setTimeout(() => {
      console.log(
        `Grace period expired for ${id}. Re-queueing job ${activeJob.start}.`
      );
      jobQueue.unshift(activeJob);
      activeJobs.delete(id);
      workerStatus.delete(id);
      disconnectedWorkers.delete(id);
      ui.setStatus(`Re-queued Job ${activeJob.start}`);
      triggerAssignments();
    }, GRACE_PERIOD_MS) as any;

    disconnectedWorkers.set(id, { job: activeJob, timeoutId });
  } else {
    workerStatus.delete(id);
  }
};

signaling.onWorkerReady = (id: string) => {
  console.log(`Worker ${id} is READY (Connection established)`);
  ui.setStatus(`Worker ${id} Connected!`);

  // Clear grace period if any
  if (disconnectedWorkers.has(id)) {
    clearTimeout(disconnectedWorkers.get(id)!.timeoutId);
    disconnectedWorkers.delete(id);
  }

  // We don't mark as idle or assign job here anymore.
  // We wait for SCENE_LOADED or sendSceneHelper to finish.
};

signaling.onWorkerJoined = (id) => {
  ui.setStatus(`Worker Joined: ${id}`);

  // Check if it was a returning worker
  if (disconnectedWorkers.has(id)) {
    console.log(`Worker ${id} returned during grace period. Resuming.`);
    clearTimeout(disconnectedWorkers.get(id)!.timeoutId);
    disconnectedWorkers.delete(id);
    // workerStatus and activeJobs are already set from before
    return;
  }

  workerStatus.set(id, "loading");

  if (currentRole === "host" && jobQueue.length > 0) {
    sendSceneHelper(id);
  }
};

signaling.onWorkerStatus = (id, hasScene, job) => {
  console.log(
    `[Host] Worker ${id} status: hasScene=${hasScene}, job=${job?.start}`
  );

  // Clear grace period if any
  if (disconnectedWorkers.has(id)) {
    clearTimeout(disconnectedWorkers.get(id)!.timeoutId);
    disconnectedWorkers.delete(id);
  }

  if (!hasScene) {
    if (workerStatus.get(id) === "loading") {
      console.log(
        `[Host] Worker ${id} has no scene but is already loading. Skipping redundant send.`
      );
      return;
    }
    if (workerStatus.get(id) === "busy") {
      console.warn(
        `[Host] Worker ${id} reports no scene while host thinks it is busy. Re-syncing.`
      );
    }

    console.log(`[Host] Worker ${id} has no scene. Syncing...`);
    workerStatus.set(id, "loading");
    sendSceneHelper(id);
    // If it HAD an active job, re-queue it because it definitely lost it
    const activeJob = activeJobs.get(id);
    if (activeJob) {
      console.warn(
        `[Host] Worker ${id} lost its job ${activeJob.start} due to refresh. Re-queuing.`
      );
      jobQueue.unshift(activeJob);
      activeJobs.delete(id);
    }
  } else {
    // Has scene. Check if the job matches.
    const expectedJob = activeJobs.get(id);
    if (expectedJob) {
      if (!job || job.start !== expectedJob.start) {
        console.warn(
          `[Host] Worker ${id} lost its job ${expectedJob.start}. Re-queuing.`
        );
        jobQueue.unshift(expectedJob);
        activeJobs.delete(id);
        workerStatus.set(id, "idle");
        triggerAssignments();
      } else {
        console.log(
          `[Host] Worker ${id} is correctly continuing job ${job.start}`
        );
        workerStatus.set(id, "busy");
      }
    } else {
      // Host thinks it's idle
      if (job) {
        // If the frame is already completed by someone else, stop it!
        if (pendingChunks.has(job.start)) {
          console.log(
            `[Host] Worker ${id} is working on already completed job ${job.start}. Stopping.`
          );
          signaling.sendStopRender(id);
          workerStatus.set(id, "idle");
          assignJob(id);
        } else {
          console.warn(
            `[Host] Worker ${id} is busy with ${job.start} but host thinks it is idle. Accepting anyway.`
          );
          activeJobs.set(id, job);
          workerStatus.set(id, "busy");
        }
      } else {
        workerStatus.set(id, "idle");
        assignJob(id);
      }
    }
  }
};

signaling.onHostHello = () => {
  console.log(
    "[Worker] Host Hello received. Syncing status and retrying results..."
  );
  trySendBufferedResults();

  // Send current status to host
  const hasScene = isDistributedSceneLoaded;
  // If we are currently "recording" (remote render), we have a job
  // We can track it via recorder state or main.ts state
  // Let's use recorder.recording as a hint, but we need the actual job info.
  // We can store the current job in a variable.
  const job = currentWorkerJob
    ? { start: currentWorkerJob.start, count: currentWorkerJob.count }
    : undefined;
  signaling.sendWorkerStatus(hasScene, job);
};

// Add this global to track current job on worker
let currentWorkerJob: { start: number; count: number } | null = null;

signaling.onRenderRequest = async (startFrame, frameCount, config) => {
  console.log(
    `[Worker] Received Render Request: Frames ${startFrame} - ${
      startFrame + frameCount
    }`
  );

  if (isSceneLoading || !isDistributedSceneLoaded) {
    console.log(
      `[Worker] Scene loading (or not synced) in progress. Queueing Render Request for ${startFrame}`
    );
    pendingRenderRequest = { start: startFrame, count: frameCount, config };
    return;
  }

  await executeWorkerRender(startFrame, frameCount, config);
};

signaling.onStopRender = () => {
  if (currentWorkerAbortController) {
    console.log("[Worker] Host requested STOP_RENDER. Aborting...");
    currentWorkerAbortController.abort();
  }
};

signaling.onSceneLoaded = async (id) => {
  if (workerStatus.get(id) !== "loading") {
    console.log(
      `[Host] Ignore redundant SCENE_LOADED from ${id} (Status: ${workerStatus.get(
        id
      )})`
    );
    return;
  }
  console.log(`[Host] Worker ${id} loaded the scene.`);
  workerStatus.set(id, "idle");
  await assignJob(id);
};

signaling.onRenderResult = async (chunks, startFrame, workerId) => {
  if (pendingChunks.has(startFrame)) {
    console.warn(
      `[Host] Ignore duplicate result for ${startFrame} from ${workerId}`
    );
    workerStatus.set(workerId, "idle");
    activeJobs.delete(workerId);
    await assignJob(workerId);
    return;
  }
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

  isDistributedSceneLoaded = true;
  isSceneLoading = false;

  console.log("Distributed Scene Loaded. Sending SCENE_LOADED.");
  await signaling.sendSceneLoaded();

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
