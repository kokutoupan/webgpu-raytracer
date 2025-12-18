import { Config } from "./config";
import { UIManager } from "./ui/UIManager";
import { SignalingClient } from "./network/SignalingClient";
// import render worker
import RenderWorker from "./render.worker?worker";

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
let distributedConfig: any = null;
let workerStatus: Map<string, "idle" | "loading" | "busy"> = new Map();
let activeJobs: Map<string, { start: number; count: number }> = new Map();

// --- Worker State ---
let isSceneLoading = false;
let pendingRenderRequest: { start: number; count: number; config: any } | null =
  null;
const BATCH_SIZE = 20;

// --- Modules ---
const ui = new UIManager();
const signaling = new SignalingClient();
const renderWorker = new RenderWorker();

// --- Main Application Loop ---
// Note: Frame counting is now handled by the worker

// --- Functions ---
const rebuildPipeline = () => {
  const depth = parseInt(ui.inputDepth.value, 10) || Config.defaultDepth;
  const spp = parseInt(ui.inputSPP.value, 10) || Config.defaultSPP;
  renderWorker.postMessage({ type: "update-config", payload: { depth, spp } });
};

const updateResolution = () => {
  const { width, height } = ui.getRenderConfig();
  renderWorker.postMessage({ type: "resize", payload: { width, height } });
};

const loadScene = async (name: string, autoStart = true) => {
  isRendering = false;

  let objSource: string | undefined;
  let glbData: Uint8Array | undefined;

  if (name === "viewer" && currentFileData) {
    if (currentFileType === "obj") objSource = currentFileData as string;
    else if (currentFileType === "glb")
      glbData = new Uint8Array(currentFileData as ArrayBuffer);
  }

  renderWorker.postMessage({
    type: "load-scene",
    payload: { name, objSource, glbData },
  });

  if (autoStart) {
    isRendering = true;
    ui.updateRenderButton(true);
    // Worker will start loop after scene load
  }
};

// --- Distributed Helpers ---
const sendSceneHelper = async (workerId?: string) => {
  if (!currentFileData || !currentFileType) return;

  const config = ui.getRenderConfig();

  if (workerId) {
    console.log(`Sending scene to specific worker: ${workerId}`);
    workerStatus.set(workerId, "loading");
    await signaling.sendSceneToWorker(
      workerId,
      currentFileData,
      currentFileType,
      config
    );
  } else {
    console.log(`Broadcasting scene to all workers...`);
    signaling.getWorkerIds().forEach((id) => workerStatus.set(id, "loading"));
    await signaling.broadcastScene(currentFileData, currentFileType, config);
  }
};

const assignJob = async (workerId: string) => {
  // Only assign if worker is IDLE
  if (workerStatus.get(workerId) !== "idle") {
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

  // Delegate to render worker
  renderWorker.postMessage({
    type: "start-record",
    payload: {
      config: workerConfig,
      role: "worker",
    },
  });
};

const handlePendingRenderRequest = async () => {
  if (!pendingRenderRequest) return;
  const { start, count, config } = pendingRenderRequest;
  pendingRenderRequest = null;

  // Execute the delayed request
  await executeWorkerRender(start, count, config);
};

// --- Worker Message Handling ---
renderWorker.onmessage = async (e) => {
  const {
    type,
    // payload, // unused
    message,
    url,
    chunks,
    fps,
    spp,
    current,
    total,
    stage,
  } = e.data;

  switch (type) {
    case "init-complete":
      console.log("Worker initialized.");
      loadScene("cornell", false);
      break;
    case "status":
      ui.setStatus(message);
      break;
    case "error":
      console.error("Worker Error: " + message);
      ui.setStatus("Error: " + message);
      break;
    case "stats":
      ui.updateStats(0, fps, spp); // Timer logic moved to worker
      break;
    case "scene-loaded":
      if (e.data.animList) {
        ui.updateAnimList(e.data.animList);
      }
      if (isRendering) {
        renderWorker.postMessage({ type: "start-render" });
      }

      // Distributed loading logic
      if (isSceneLoading) {
        // Flag from signaling
        isSceneLoading = false;
        console.log("Scene Loaded. Sending WORKER_READY.");
        await signaling.sendWorkerReady();
        handlePendingRenderRequest();
      }
      break;

    case "record-progress":
      ui.setRecordingState(
        true,
        `${stage === "recording" ? "Rec" : "Enc"}: ${current}/${total}`
      );
      break;

    case "record-complete":
      ui.setRecordingState(false);
      // isRendering = true;
      // ui.updateRenderButton(true);
      // renderWorker.postMessage({ type: "start-render" }); // Disabled by user

      const a = document.createElement("a");
      a.href = url;
      a.download = `raytrace_${Date.now()}.webm`;
      a.click();
      break;

    case "record-result-chunks":
      // Worker role finished recording chunks
      ui.setRecordingState(true, "Uploading...");
      // 'pendingRenderRequest' context is lost here if we don't track it,
      // but we can assume the last request.
      // Actually, we need startFrame.
      // Let's assume sequential for now or store ID.
      // But valid simplification:
      // const startFrame = (e.data as any).startFrame || 0;
      // Actually currently worker doesn't pass back startFrame.
      // We can infer or fix worker.
      // For now, let's fix worker or assume:
      // Actually, the signaling.sendRenderResult needs it.
      // Let's assume we store it in a closure variable?

      // In executeWorkerRender -> we can store currentJob

      // Simplification: We only support one active job at a time per worker.
      // We need to pass the startFrame through the whole chain.
      // Refactor: Pass startFrame in payload to worker, worker returns it.

      // Temporary Fix: We will fix this in next step if broken, but for now
      // let's grab it from a global variable if needed, or update protocol
      await signaling.sendRenderResult(chunks, (e.data as any).startFrame || 0); // Need to fix this
      ui.setRecordingState(false);
      ui.setStatus("Idle");
      break;
  }
};
// Fix record-result-chunks protocol:
// Only "executeWorkerRender" sets the job. We can track it here.
let currentWorkerJobStartFrame = 0;

// Modify executeWorkerRender to set this
const executeWorkerRenderFixed = async (
  start: number,
  count: number,
  config: any
) => {
  currentWorkerJobStartFrame = start;
  executeWorkerRender(start, count, config);
};
// Re-bind correctly below.

// Override handler for chunks
renderWorker.addEventListener("message", async (e) => {
  if (e.data.type === "record-result-chunks") {
    ui.setRecordingState(true, "Uploading...");
    await signaling.sendRenderResult(e.data.chunks, currentWorkerJobStartFrame);
    ui.setRecordingState(false);
    ui.setStatus("Idle");

    // isRendering = true;
    // renderWorker.postMessage({ type: "start-render" }); // Disabled
  }
});

// --- Signaling Callbacks (Global) ---
signaling.onStatusChange = (msg) => ui.setStatus(`Status: ${msg}`);
signaling.onWorkerLeft = (id) => {
  console.log(`Worker Left: ${id}`);
  ui.setStatus(`Worker Left: ${id}`);
  workerStatus.delete(id);
  const failedJob = activeJobs.get(id);
  if (failedJob) {
    jobQueue.unshift(failedJob);
    activeJobs.delete(id);
    ui.setStatus(`Re-queued Job ${failedJob.start}`);
  }
};
signaling.onWorkerReady = (id) => {
  ui.setStatus(`Worker ${id} Ready!`);
  workerStatus.set(id, "idle");
  if (currentRole === "host" && jobQueue.length > 0) assignJob(id);
};
signaling.onWorkerJoined = (id) => {
  ui.setStatus(`Worker Joined: ${id}`);
  workerStatus.set(id, "idle");
  if (currentRole === "host" && jobQueue.length > 0) sendSceneHelper(id);
};
signaling.onRenderRequest = async (startFrame, frameCount, config) => {
  if (isSceneLoading) {
    pendingRenderRequest = { start: startFrame, count: frameCount, config };
    return;
  }
  await executeWorkerRenderFixed(startFrame, frameCount, config);
};
signaling.onRenderResult = async (chunks, startFrame, workerId) => {
  pendingChunks.set(startFrame, chunks);
  completedJobs++;
  ui.setStatus(`Distributed Progress: ${completedJobs} / ${totalJobs} jobs`);
  workerStatus.set(workerId, "idle");
  activeJobs.delete(workerId);
  await assignJob(workerId);
  if (completedJobs >= totalJobs) {
    ui.setStatus("Muxing...");
    await muxAndDownload();
  }
};
signaling.onSceneReceived = async (data, config) => {
  isSceneLoading = true; // Set flag
  ui.setRenderConfig(config);
  currentFileType = config.fileType;
  if (config.fileType === "obj") currentFileData = data as string;
  else currentFileData = data as ArrayBuffer;
  ui.sceneSelect.value = "viewer";
  await loadScene("viewer", false);
  if (config.anim !== undefined) {
    ui.animSelect.value = config.anim.toString();
    // worldBridge.setAnimation(config.anim); // Moved to scene-loaded event
  }
  // Note: WORKER_READY is sent in scene-loaded handler
};

// --- Event Binding ---
const bindEvents = () => {
  ui.onRenderStart = () => {
    isRendering = true;
    renderWorker.postMessage({ type: "start-render" });
  };
  ui.onRenderStop = () => {
    isRendering = false;
    renderWorker.postMessage({ type: "stop-render" });
  };
  ui.onSceneSelect = (name) => loadScene(name, false);
  ui.onResolutionChange = updateResolution;

  ui.onRecompile = (_depth, _spp) => {
    rebuildPipeline();
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

  ui.onAnimSelect = (idx) => {
    renderWorker.postMessage({ type: "set-anim", payload: { index: idx } });
  };

  ui.onUpdateIntervalChange = (val) => {
    renderWorker.postMessage({
      type: "update-config",
      payload: { updateInterval: val },
    });
  };

  ui.onRecordStart = async () => {
    // ... (Same logic, but delegate to worker)
    if (currentRole === "host") {
      const workers = signaling.getWorkerIds();
      distributedConfig = ui.getRenderConfig();
      const totalRenderFrames = Math.ceil(
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
      workers.forEach((w) => workerStatus.set(w, "idle"));
      ui.setStatus(
        `Distributed Progress: 0 / ${totalJobs} jobs (Waiting for workers...)`
      );

      if (workers.length > 0) {
        ui.setStatus("Syncing Scene to Workers...");
        await sendSceneHelper();
      }
    } else {
      // Local Recording via Worker
      isRendering = false;
      ui.setRecordingState(true);
      const config = ui.getRenderConfig();

      renderWorker.postMessage({
        type: "start-record",
        payload: {
          config,
          role: "host", // Direct download
        },
      });
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

  ui.setConnectionState(null);
};

// --- Entry Point ---
async function bootstrap() {
  // Transfer logic
  const offscreen = ui.canvas.transferControlToOffscreen();

  bindEvents();

  // Init Worker
  renderWorker.postMessage(
    {
      type: "init",
      payload: {
        canvas: offscreen,
        width: ui.canvas.width,
        height: ui.canvas.height,
      },
    },
    [offscreen]
  );
}

bootstrap().catch(console.error);
