import { WebGPURenderer } from "../renderer";
import { WorldBridge } from "../world-bridge";

export class VideoRecorder {
  private isRecording = false;
  private renderer: WebGPURenderer;
  private worldBridge: WorldBridge;
  private canvas: HTMLCanvasElement;

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
    onProgress: (frame: number, total: number) => void
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
        (config as any).startFrame || 0
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
    startFrameOffset: number = 0
  ) {
    for (let i = 0; i < totalFrames; i++) {
      onProgress(i, totalFrames);

      // Allow UI to breathe
      await new Promise((r) => setTimeout(r, 0));

      const currentFrame = startFrameOffset + i;
      const time = currentFrame / config.fps;
      this.worldBridge.update(time);
      await this.worldBridge.waitForNextUpdate();

      // Re-upload Geometry/BVH as animation might have changed them
      await this.updateSceneBuffers();

      // Render Frame
      await this.renderFrame(config.spp, config.batch);

      // Encode
      if (encoder.encodeQueueSize > 5) {
        await encoder.flush();
      }

      const frame = new VideoFrame(this.canvas, {
        timestamp: (currentFrame * 1000000) / config.fps,
        duration: 1000000 / config.fps,
      });

      encoder.encode(frame, { keyFrame: i % config.fps === 0 });
      frame.close();
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
      "index",
      this.worldBridge.indices
    );
    needsRebind ||= this.renderer.updateBuffer(
      "attr",
      this.worldBridge.attributes
    );

    this.worldBridge.updateCamera(this.canvas.width, this.canvas.height);
    this.renderer.updateSceneUniforms(this.worldBridge.cameraData, 0);

    if (needsRebind) this.renderer.recreateBindGroup();
    this.renderer.resetAccumulation();
  }

  private async renderFrame(totalSpp: number, batchSize: number) {
    let samplesDone = 0;
    while (samplesDone < totalSpp) {
      const batch = Math.min(batchSize, totalSpp - samplesDone);
      for (let k = 0; k < batch; k++) {
        this.renderer.render(samplesDone + k);
      }
      samplesDone += batch;
      await this.renderer.device.queue.onSubmittedWorkDone();
      if (samplesDone < totalSpp) {
        await new Promise((r) => setTimeout(r, 0));
      }
    }
  }
}
