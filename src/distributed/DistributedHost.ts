import { SignalingClient } from "../network/SignalingClient";
import { UIManager } from "../ui/UIManager";
import type { RenderConfig } from "../network/Protocol";

export class DistributedHost {
  public jobQueue: { start: number; count: number }[] = [];
  public pendingChunks: Map<number, any[]> = new Map(); // startFrame -> SerializedChunk[]
  public completedJobs = 0;
  public totalJobs = 0;
  public totalRenderFrames = 0;
  public distributedConfig: RenderConfig | null = null;
  public workerStatus: Map<string, "idle" | "loading" | "busy"> = new Map();
  public activeJobs: Map<string, { start: number; count: number }> = new Map();

  private signaling: SignalingClient;
  private ui: UIManager;

  private disconnectedWorkers = new Map<
    string,
    { job: { start: number; count: number }; timeoutId: number }
  >();
  private readonly GRACE_PERIOD_MS = 30000;

  constructor(signaling: SignalingClient, ui: UIManager) {
    this.signaling = signaling;
    this.ui = ui;
    this.setupSignaling();
  }

  private setupSignaling() {
    this.signaling.onWorkerLeft = (id) => this.onWorkerLeft(id);
    this.signaling.onWorkerReady = (id) => this.onWorkerReady(id);
    this.signaling.onWorkerJoined = (id) => this.onWorkerJoined(id);
    this.signaling.onWorkerStatus = (id, hasScene, job) =>
      this.onWorkerStatus(id, hasScene, job);
    this.signaling.onSceneLoaded = (id) => this.onSceneLoaded(id);
    this.signaling.onRenderResult = (chunks, startFrame, workerId) =>
      this.onRenderResult(chunks, startFrame, workerId);
  }

  public async sendSceneHelper(
    currentFileData: string | ArrayBuffer | null,
    currentFileType: "obj" | "glb" | null,
    workerId?: string
  ) {
    const currentScene = this.ui.sceneSelect.value;
    const isProcedural = currentScene !== "viewer";

    if (!isProcedural && (!currentFileData || !currentFileType)) return;

    const config = this.ui.getRenderConfig() as RenderConfig;
    const sceneName = isProcedural ? currentScene : undefined;
    const fileData = isProcedural ? "DUMMY" : currentFileData!;
    const fileType = isProcedural ? "obj" : currentFileType!;

    config.sceneName = sceneName;
    config.fileType = fileType;

    if (workerId) {
      console.log(`[Host] Sending scene to specific worker: ${workerId}`);
      this.workerStatus.set(workerId, "loading");
      await this.signaling.sendSceneToWorker(
        workerId,
        fileData,
        fileType,
        config
      );
    } else {
      console.log(`[Host] Broadcasting scene to all workers...`);
      this.signaling
        .getWorkerIds()
        .forEach((id) => this.workerStatus.set(id, "loading"));
      await this.signaling.broadcastScene(fileData, fileType, config);
    }
  }

  public async assignJob(workerId: string) {
    if (this.workerStatus.get(workerId) !== "idle") return;
    if (this.jobQueue.length === 0) return;

    if (!this.distributedConfig) {
      console.warn("[Host] Distributed config is missing. Cannot assign job.");
      return;
    }

    const job = this.jobQueue.shift()!;
    this.workerStatus.set(workerId, "busy");
    this.activeJobs.set(workerId, job);

    console.log(
      `[Host] Assigning job to ${workerId}: Frames ${job.start} - ${
        job.start + job.count
      }`
    );

    try {
      await this.signaling.sendRenderRequest(
        workerId,
        job.start,
        job.count,
        this.distributedConfig
      );
    } catch (e) {
      console.error(`[Host] Failed to send job to ${workerId}, re-queuing`, e);
      this.jobQueue.push(job);
      this.workerStatus.set(workerId, "idle");
      this.activeJobs.delete(workerId);

      setTimeout(() => this.assignJob(workerId), 2000);
    }
  }

  public triggerAssignments() {
    for (const [id, status] of this.workerStatus.entries()) {
      if (status === "idle") {
        this.assignJob(id);
      }
    }
  }

  public onWorkerLeft(id: string) {
    console.log(`[Host] Worker ${id} left.`);
    const activeJob = this.activeJobs.get(id);

    if (activeJob) {
      console.log(`[Host] Worker ${id} had active job. Starting grace period.`);
      const timeoutId = window.setTimeout(() => {
        console.log(`[Host] Grace period expired for ${id}. Re-queuing job.`);
        this.jobQueue.push(activeJob);
        this.disconnectedWorkers.delete(id);
        this.activeJobs.delete(id);
        this.workerStatus.delete(id);
        this.triggerAssignments();
      }, this.GRACE_PERIOD_MS);

      this.disconnectedWorkers.set(id, { job: activeJob, timeoutId });
    } else {
      this.workerStatus.delete(id);
      this.activeJobs.delete(id);
    }
  }

  public onWorkerReady(id: string) {
    console.log(`[Host] Worker ${id} is ready (Manual Signal).`);
    // This is a legacy signal, we now rely on SCENE_LOADED
  }

  public onWorkerJoined(id: string) {
    console.log(`[Host] Worker ${id} joined.`);
    this.workerStatus.set(id, "loading");

    // If it's a reconnection, resume its job
    const disconnected = this.disconnectedWorkers.get(id);
    if (disconnected) {
      console.log(`[Host] Worker ${id} re-joined. Resuming job.`);
      clearTimeout(disconnected.timeoutId);
      this.activeJobs.set(id, disconnected.job);
      this.disconnectedWorkers.delete(id);

      // Re-send the scene if needed, or just re-assign
      // If the worker joined, it might need the scene again
    }
  }

  public async onWorkerStatus(
    id: string,
    hasScene: boolean,
    job?: { start: number; count: number }
  ) {
    console.log(`[Host] Worker ${id} status update: hasScene=${hasScene}`, job);

    if (!hasScene) {
      if (this.workerStatus.get(id) === "loading") {
        console.log(
          `[Host] Worker ${id} has no scene but is already loading. Skipping redundant send.`
        );
        return;
      }
      if (this.workerStatus.get(id) === "busy") {
        console.warn(
          `[Host] Worker ${id} reports no scene while host thinks it is busy. Re-syncing.`
        );
      }

      console.log(`[Host] Worker ${id} has no scene. Syncing...`);
      // Note: main.ts will need to call back into this for file data
      return "NEED_SCENE";
    }

    if (!job && this.workerStatus.get(id) !== "busy") {
      if (this.workerStatus.get(id) === "loading") {
        console.log(`[Host] Worker ${id} is still loading scene.`);
      } else {
        this.workerStatus.set(id, "idle");
        await this.assignJob(id);
      }
    } else if (job) {
      this.workerStatus.set(id, "busy");
      this.activeJobs.set(id, job);
    }
  }

  public async onSceneLoaded(id: string) {
    if (this.workerStatus.get(id) !== "loading") {
      console.log(
        `[Host] Ignore redundant SCENE_LOADED from ${id} (Status: ${this.workerStatus.get(
          id
        )})`
      );
      return;
    }
    console.log(`[Host] Worker ${id} loaded the scene.`);
    this.workerStatus.set(id, "idle");
    await this.assignJob(id);
  }

  public async onRenderResult(
    chunks: any[],
    startFrame: number,
    workerId: string
  ) {
    if (this.pendingChunks.has(startFrame)) {
      console.warn(
        `[Host] Ignore duplicate result for ${startFrame} from ${workerId}`
      );
      this.workerStatus.set(workerId, "idle");
      this.activeJobs.delete(workerId);
      await this.assignJob(workerId);
      return;
    }
    console.log(
      `[Host] Received ${chunks.length} chunks for ${startFrame} from ${workerId}`
    );

    this.pendingChunks.set(startFrame, chunks);
    this.completedJobs++;
    this.ui.setStatus(
      `Distributed Progress: ${this.completedJobs} / ${this.totalJobs} jobs`
    );

    this.workerStatus.set(workerId, "idle");
    this.activeJobs.delete(workerId);

    await this.assignJob(workerId);

    if (this.completedJobs >= this.totalJobs) {
      console.log("[Host] All jobs complete. Triggering Muxing Callback.");
      return "ALL_COMPLETE";
    }
  }

  public async muxAndDownload() {
    const sortedStarts = Array.from(this.pendingChunks.keys()).sort(
      (a, b) => a - b
    );

    // Create Muxer
    const { Muxer, ArrayBufferTarget } = await import("webm-muxer");
    const mult = new Muxer({
      target: new ArrayBufferTarget(),
      video: {
        codec: "V_VP9",
        width: this.distributedConfig!.width,
        height: this.distributedConfig!.height,
        frameRate: this.distributedConfig!.fps,
      },
    });

    for (const start of sortedStarts) {
      const chunks = this.pendingChunks.get(start);
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
    a.download = `distributed_render_${Date.now()}.webm`;
    a.click();
    URL.revokeObjectURL(url);
    this.ui.setStatus("Distributed Render Complete.");
  }
}
