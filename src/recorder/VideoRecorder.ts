import { WebGPURenderer } from "../renderer";
import { WorldBridge } from "../world-bridge";

export class VideoRecorder {
  private isRecording = false;
  private renderer: WebGPURenderer;
  private worldBridge: WorldBridge;
  private canvas: HTMLCanvasElement | OffscreenCanvas;
  private TILE_SIZE = 512;

  constructor(
    renderer: WebGPURenderer,
    worldBridge: WorldBridge,
    canvas: HTMLCanvasElement | OffscreenCanvas
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
    onComplete: (buffer: ArrayBuffer) => void
  ) {
    if (this.isRecording) return;
    this.isRecording = true;
    const { Muxer, ArrayBufferTarget } = await import("webm-muxer");

    const totalFrames = Math.ceil(config.fps * config.duration);
    console.log(
      `Starting recording: ${totalFrames} frames @ ${config.fps}fps (VP9)`
    );

    // ★修正1: 解像度を偶数に強制する (VP9クラッシュ対策)
    const width = this.canvas.width & ~1;
    const height = this.canvas.height & ~1;

    const muxer = new Muxer({
      target: new ArrayBufferTarget(),
      video: {
        codec: "V_VP9",
        width: width,
        height: height,
        frameRate: config.fps,
      },
    });

    const videoEncoder = new VideoEncoder({
      output: (chunk, meta) => muxer.addVideoChunk(chunk, meta),
      error: (e) => console.error("VideoEncoder Error:", e),
    });

    videoEncoder.configure({
      codec: "vp09.00.10.08",
      width: width,
      height: height,
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

      // ★修正2: エンコーダーを明示的に閉じてVRAMを解放する
      videoEncoder.close();

      const { buffer } = muxer.target;
      onComplete(buffer);
    } catch (e) {
      try {
        videoEncoder.close();
      } catch (e) {} // エラー時も閉じる
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
        alpha: "discard",
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

      const width = this.canvas.width;
      const height = this.canvas.height;

      // We process a batch of samples. For each sample, we do a full tiled pass.
      // Or optimization: encode N samples inside the tile loop?
      // Since uniform logic is per-dispatch (or per-tile in our case),
      // strict replication of old behavior means loop K outside, allowing uniforms update.
      // But tiled offset is also uniform.

      // Correct approach:
      // For each sample index k:
      //   For each tile:
      //     Update Uniform (Frame + TileOffset)
      //     Dispatch
      //   Wait?
      const TILE_SIZE = this.TILE_SIZE;
      // Ideally we loop tiles inside k loop.
      const tilesX = Math.ceil(width / TILE_SIZE);
      const tilesY = Math.ceil(height / TILE_SIZE);

      for (let k = 0; k < batch; k++) {
        const frameCount = samplesDone + k;

        for (let ty = 0; ty < tilesY; ty++) {
          for (let tx = 0; tx < tilesX; tx++) {
            const offsetX = tx * TILE_SIZE;
            const offsetY = ty * TILE_SIZE;
            const cw = Math.min(TILE_SIZE, width - offsetX);
            const ch = Math.min(TILE_SIZE, height - offsetY);

            const encoder = this.renderer.device.createCommandEncoder();
            this.renderer.encodeTileCommand(
              encoder,
              offsetX,
              offsetY,
              cw,
              ch,
              frameCount
            );
            this.renderer.device.queue.submit([encoder.finish()]);

            // Wait for tile
            // await this.renderer.device.queue.onSubmittedWorkDone();
          }
        }

        // Wait for batch to complete to prevent TDR
        await this.renderer.device.queue.onSubmittedWorkDone();

        samplesDone += batch;

        if (samplesDone < totalSpp) {
          // Yield for UI
          await new Promise((r) => setTimeout(r, 16));
        }
      }

      // Final present to canvas for capture
      const finalEncoder = this.renderer.device.createCommandEncoder();
      this.renderer.present(finalEncoder);
      this.renderer.device.queue.submit([finalEncoder.finish()]);
      await this.renderer.device.queue.onSubmittedWorkDone();
    }
  }
}
