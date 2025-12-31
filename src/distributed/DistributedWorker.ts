import { SignalingClient } from "../network/SignalingClient";
import { WebGPURenderer } from "../renderer";
import { UIManager } from "../ui/UIManager";
import { VideoRecorder } from "../recorder/VideoRecorder";
import type { RenderConfig } from "../network/Protocol";

export class DistributedWorker {
  public isSceneLoading = false;
  public isDistributedSceneLoaded = false;
  public pendingRenderRequest: {
    start: number;
    count: number;
    config: RenderConfig;
  } | null = null;
  public currentWorkerJob: { start: number; count: number } | null = null;
  public onRemoteSceneLoad:
    | ((data: string | ArrayBuffer, type: "obj" | "glb") => Promise<void>)
    | null = null;

  private signaling: SignalingClient;
  private renderer: WebGPURenderer;
  private ui: UIManager;
  private recorder: VideoRecorder;

  private bufferedResults: { chunks: any[]; startFrame: number }[] = [];
  private currentWorkerAbortController: AbortController | null = null;

  constructor(
    signaling: SignalingClient,
    renderer: WebGPURenderer,
    ui: UIManager,
    recorder: VideoRecorder
  ) {
    this.signaling = signaling;
    this.renderer = renderer;
    this.ui = ui;
    this.recorder = recorder;
    this.setupSignaling();
  }

  private setupSignaling() {
    this.signaling.onHostHello = () => this.onHostHello();
    this.signaling.onRenderRequest = (start, count, config) =>
      this.onRenderRequest(start, count, config);
    this.signaling.onStopRender = () => this.onStopRender();
    this.signaling.onSceneReceived = (data, config) =>
      this.onSceneReceived(data, config);
  }

  public async executeWorkerRender(
    startFrame: number,
    frameCount: number,
    config: RenderConfig
  ) {
    if (this.recorder.isRecording) {
      console.warn("[Worker] Already recording/rendering, skipping request.");
      return;
    }

    if (this.currentWorkerAbortController) {
      this.currentWorkerAbortController.abort();
    }
    this.currentWorkerAbortController = new AbortController();
    const signal = this.currentWorkerAbortController.signal;

    if (this.isSceneLoading || !this.isDistributedSceneLoaded) {
      console.log(
        `[Worker] Scene loading (or not synced) in progress. Queueing Render Request for ${startFrame}`
      );
      this.pendingRenderRequest = {
        start: startFrame,
        count: frameCount,
        config,
      };
      return;
    }

    this.currentWorkerJob = { start: startFrame, count: frameCount };
    console.log(
      `[Worker] Starting Render: Frames ${startFrame} - ${
        startFrame + frameCount
      }`
    );
    this.ui.setStatus(
      `Remote Rendering: ${startFrame}-${startFrame + frameCount}`
    );

    // Sync shader settings before recording
    if (config.maxDepth !== undefined && config.shaderSpp !== undefined) {
      console.log(
        `[Worker] Updating Shader Pipeline: Depth=${config.maxDepth}, SPP=${config.shaderSpp}`
      );
      this.renderer.buildPipeline(config.maxDepth, config.shaderSpp);
    }

    const workerConfig = {
      ...config,
      startFrame: startFrame,
      duration: frameCount / config.fps,
    };

    try {
      this.ui.setRecordingState(true, `Remote: ${frameCount} f`);

      const chunks = await this.recorder.recordChunks(
        workerConfig as any,
        (f, t) => this.ui.setRecordingState(true, `Remote: ${f}/${t}`),
        signal
      );

      console.log(
        `[Worker] Render Finished for ${startFrame}. Sending results.`
      );
      await this.signaling.sendRenderResult(chunks, startFrame);
      this.currentWorkerJob = null;
    } catch (e: any) {
      if (e.name === "AbortError") {
        console.log(`[Worker] Render Aborted for ${startFrame}`);
      } else {
        console.error("[Worker] Remote Recording Failed", e);
        this.ui.setStatus("Recording Failed");
      }
    } finally {
      this.currentWorkerJob = null;
      this.currentWorkerAbortController = null;
      this.ui.updateRenderButton(false);
      this.ui.setRecordingState(false);
    }
  }

  public async trySendBufferedResults() {
    if (this.bufferedResults.length === 0) return;

    console.log(
      `[Worker] Retrying to send ${this.bufferedResults.length} buffered results...`
    );
    const stillFailed = [];
    for (const res of this.bufferedResults) {
      try {
        await this.signaling.sendRenderResult(res.chunks, res.startFrame);
      } catch (e) {
        stillFailed.push(res);
      }
    }
    this.bufferedResults = stillFailed;
  }

  public handlePendingRenderRequest() {
    if (this.pendingRenderRequest) {
      console.log(
        `[Worker] Processing Pending Render Request: ${this.pendingRenderRequest.start}`
      );
      const req = this.pendingRenderRequest;
      this.pendingRenderRequest = null;
      this.executeWorkerRender(req.start, req.count, req.config);
    }
  }

  public onHostHello() {
    console.log("[Worker] Host Hello received.");
    this.signaling.sendWorkerStatus(
      this.isDistributedSceneLoaded,
      this.currentWorkerJob || undefined
    );
  }

  public onRenderRequest(
    startFrame: number,
    frameCount: number,
    config: RenderConfig
  ) {
    this.executeWorkerRender(startFrame, frameCount, config);
  }

  public onStopRender() {
    console.log("[Worker] Stop Render received.");
    if (this.currentWorkerAbortController) {
      this.currentWorkerAbortController.abort();
    }
  }

  public async onSceneReceived(
    data: string | ArrayBuffer,
    config: RenderConfig
  ) {
    console.log("[Worker] Scene received successfully.");
    // Force stop any rogue recording state to prevent "Resize blocked" or resource errors
    this.recorder.cancel();

    this.isSceneLoading = true;

    this.ui.setRenderConfig(config);

    // Sync shader settings
    if (config.maxDepth !== undefined && config.shaderSpp !== undefined) {
      console.log(
        `[Worker] Syncing Shader settings: Depth=${config.maxDepth}, SPP=${config.shaderSpp}`
      );
      this.renderer.buildPipeline(config.maxDepth, config.shaderSpp);
    }

    // Call main loader logic
    if (this.onRemoteSceneLoad) {
      // Cast fileType because config is RenderConfig which has "obj"|"glb"
      await this.onRemoteSceneLoad(data, config.fileType || "obj");

      // Sync Wait: Prevent "black frame" by giving WASM memory time to settle / message loop to clear
      await new Promise((resolve) => setTimeout(resolve, 500));
    }

    this.isSceneLoading = false;
    this.isDistributedSceneLoaded = true;
    this.signaling.sendSceneLoaded();

    // Now that scene is loaded, check for pending jobs
    // REMOVED: handlePendingRenderRequest() is called by main.ts AFTER setting animation/camera.
    // Calling it here causes "First frame default animation" issue because SetAnimation hasn't run yet.
    /*
    if (this.pendingRenderRequest) {
      console.log("[Worker] Found pending render request. Executing now.");
      this.handlePendingRenderRequest();
    }
    */

    return { data, config };
  }
}
