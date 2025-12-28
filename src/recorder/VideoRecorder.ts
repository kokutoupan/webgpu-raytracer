import { WebGPURenderer } from "../renderer";
import { WorldBridge } from "../world-bridge";

export class VideoRecorder {
  public isRecording = false;
  private renderer: WebGPURenderer;
  private worldBridge: WorldBridge;
  private canvas: HTMLCanvasElement;
  private currentBatchSize: number = 0;

  constructor(
    renderer: WebGPURenderer,
    worldBridge: WorldBridge,
    canvas: HTMLCanvasElement
  ) {
    this.renderer = renderer;
    this.worldBridge = worldBridge;
    this.canvas = canvas;
  }

  public get recording() {
    return this.isRecording;
  }

  public async record(
    config: { fps: number; duration: number; spp: number; batch: number },
    onProgress: (frame: number, total: number) => void,
    onComplete: (blobUrl: string, blob?: Blob) => void
  ) {
    if (this.isRecording) return;
    this.isRecording = true;
    const { Muxer, ArrayBufferTarget } = await import("webm-muxer");

    const totalFrames = Math.ceil(config.fps * config.duration);
    console.log(
      `Starting recording: ${totalFrames} frames @ ${config.fps}fps (VP9)`
    );

    const muxer = new Muxer({
      target: new ArrayBufferTarget(),
      video: {
        codec: "V_VP9",
        width: this.canvas.width,
        height: this.canvas.height,
        frameRate: config.fps,
      },
    });

    const videoEncoder = new VideoEncoder({
      output: (chunk, meta) => muxer.addVideoChunk(chunk, meta),
      error: (e) => console.error("VideoEncoder Error:", e),
    });

    videoEncoder.configure({
      codec: "vp09.00.10.08",
      width: this.canvas.width,
      height: this.canvas.height,
      bitrate: 12_000_000,
    });

    try {
      await this.renderAndEncode(
        totalFrames,
        config,
        videoEncoder,
        onProgress,
        (config as any).startFrame || 0
      );

      await videoEncoder.flush();
      muxer.finalize();

      const { buffer } = muxer.target;
      const blob = new Blob([buffer], { type: "video/webm" });
      const url = URL.createObjectURL(blob);
      onComplete(url, blob);
    } catch (e) {
      console.error("Recording failed:", e);
      throw e;
    } finally {
      this.isRecording = false;
    }
  }

  public async recordChunks(
    config: { fps: number; duration: number; spp: number; batch: number },
    onProgress: (frame: number, total: number) => void,
    abortSignal?: AbortSignal
  ): Promise<any[]> {
    // Return SerializedChunk[] but avoid circular dep on Protocol for now or import it
    if (this.isRecording) throw new Error("Already recording");
    this.isRecording = true;

    const chunks: any[] = []; // SerializedChunk[]
    const totalFrames = Math.ceil(config.fps * config.duration);

    const videoEncoder = new VideoEncoder({
      output: (chunk, meta) => {
        const buffer = new Uint8Array(chunk.byteLength);
        chunk.copyTo(buffer);
        chunks.push({
          type: chunk.type,
          timestamp: chunk.timestamp,
          duration: chunk.duration,
          data: buffer.buffer,
          decoderConfig: meta?.decoderConfig,
        });
      },
      error: (e) => console.error("VideoEncoder Error:", e),
    });

    videoEncoder.configure({
      codec: "vp09.00.10.08",
      width: this.canvas.width,
      height: this.canvas.height,
      bitrate: 12_000_000, // Maybe lower bitrate for chunks?
    });

    try {
      await this.renderAndEncode(
        totalFrames,
        config,
        videoEncoder,
        onProgress,
        (config as any).startFrame || 0,
        abortSignal
      );
      await videoEncoder.flush();
      return chunks;
    } finally {
      this.isRecording = false;
    }
  }

  // Extracted render loop for easier maintenance & potential future partial-rendering
  private async renderAndEncode(
    totalFrames: number,
    config: { fps: number; spp: number; batch: number },
    encoder: VideoEncoder,
    onProgress: (f: number, t: number) => void,
    startFrameOffset: number = 0,
    abortSignal?: AbortSignal
  ) {
    if (abortSignal?.aborted) throw new Error("Aborted");
    this.currentBatchSize = config.batch; // Initialize with suggested batch
    // 1. Bootstrap: Update for first frame and wait for it
    const startFrame = startFrameOffset;
    this.worldBridge.update(startFrame / config.fps);
    await this.worldBridge.waitForNextUpdate();

    for (let i = 0; i < totalFrames; i++) {
      if (abortSignal?.aborted) throw new Error("Aborted");
      onProgress(i, totalFrames);

      // Allow UI to breathe
      await new Promise((r) => setTimeout(r, 0));

      // 2. Upload buffers for CURRENT frame (data available from bootstrap or previous loop wait)
      await this.updateSceneBuffers();

      // 3. Kick off NEXT frame update asynchronously (overlap with rendering)
      let nextUpdatePromise: Promise<void> | null = null;
      if (i < totalFrames - 1) {
        const nextFrame = startFrameOffset + i + 1;
        this.worldBridge.update(nextFrame / config.fps);
        nextUpdatePromise = this.worldBridge.waitForNextUpdate();
      }

      // 4. Render CURRENT frame
      await this.renderFrame(config.spp);

      // 5. Encode
      if (encoder.encodeQueueSize > 5) {
        await encoder.flush();
      }

      // Use GPU Readback to safely capture frame data, avoiding Swapchain/Canvas invalidation issues
      // caused by popups or backgrounding.
      try {
        const { data, width, height } = await this.renderer.captureFrame();

        // Explicitly use RGBA because renderTarget is hardcoded to "rgba8unorm" in renderer.ts.
        // Doing navigator.gpu.getPreferredCanvasFormat() is incorrect because we are reading
        // from the intermediate render buffer, not the swapchain itself.
        const format = "RGBA";

        const frame = new VideoFrame(data, {
          codedWidth: width,
          codedHeight: height,
          format: format,
          timestamp: ((startFrameOffset + i) * 1000000) / config.fps,
          duration: 1000000 / config.fps,
        });

        encoder.encode(frame, { keyFrame: i % config.fps === 0 });
        frame.close();
      } catch (err) {
        console.warn(
          `[VideoRecorder] Frame ${i} skipped: Readback failed.`,
          err
        );
      }
      // 6. Wait for NEXT frame data before entering next iteration
      if (nextUpdatePromise) {
        await nextUpdatePromise;
      }
    }
  }

  private async updateSceneBuffers() {
    let needsRebind = false;
    needsRebind ||= this.renderer.updateCombinedBVH(
      this.worldBridge.tlas,
      this.worldBridge.blas
    );
    needsRebind ||= this.renderer.updateBuffer(
      "instance",
      this.worldBridge.instances
    );
    needsRebind ||= this.renderer.updateCombinedGeometry(
      this.worldBridge.vertices,
      this.worldBridge.normals,
      this.worldBridge.uvs
    );
    needsRebind ||= this.renderer.updateBuffer(
      "topology",
      this.worldBridge.mesh_topology
    );
    needsRebind ||= this.renderer.updateBuffer(
      "lights",
      this.worldBridge.lights
    );

    this.worldBridge.updateCamera(this.canvas.width, this.canvas.height);
    this.renderer.updateSceneUniforms(
      this.worldBridge.cameraData,
      0,
      this.worldBridge.lightCount
    );

    if (needsRebind) this.renderer.recreateBindGroup();
    this.renderer.resetAccumulation();
  }

  private async renderFrame(totalSpp: number) {
    let samplesDone = 0;
    let lastPresentTime = performance.now();

    while (samplesDone < totalSpp) {
      const batch = Math.min(this.currentBatchSize, totalSpp - samplesDone);
      const t0 = performance.now();

      for (let k = 0; k < batch; k++) {
        this.renderer.compute(samplesDone + k);
      }
      samplesDone += batch;

      // THROTTLE PRESENT: Only present if >100ms passed OR finished
      // This prevents massive BindGroup churn (which happens in present())
      // and fixes the "valid no external instance reference" crash on fast GPUs.
      const now = performance.now();
      const isFinished = samplesDone >= totalSpp;
      if (isFinished || now - lastPresentTime > 100) {
        this.renderer.present();
        lastPresentTime = now;
      }

      await this.renderer.device.queue.onSubmittedWorkDone();

      const t1 = performance.now();
      const elapsed = t1 - t0;

      // Auto-adjust batch size to target ~100ms (10 FPS)
      // Cap the raw multiplier to prevent explosive growth (max 1.5x per step)
      let rawMultiplier = elapsed > 0 ? 100 / elapsed : 1.5;
      rawMultiplier = Math.min(rawMultiplier, 1.5);

      // Adjust with more damping to prevent oscillation
      const nextBatch = Math.round(
        this.currentBatchSize * (0.8 + 0.2 * rawMultiplier)
      );

      // HARD CAP: 50. Doing >50 passes per JS frame is likely too much CPU/Driver overhead.
      // If the GPU is that fast, we are just limited by JS/Driver dispatch.
      this.currentBatchSize = Math.max(
        1,
        Math.min(totalSpp, Math.min(nextBatch, 50))
      );

      // Removed logging to reduce console noise during high-speed recording
    }
  }
}
