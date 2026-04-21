export class WebGPUContext {
  device!: GPUDevice;
  context!: GPUCanvasContext;
  canvas: HTMLCanvasElement;

  private readbackBuffer: GPUBuffer | null = null;
  private readbackBufferSize = 0;
  private readbackResultBuffer: Uint8Array | null = null;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
  }

  async init() {
    if (!navigator.gpu) throw new Error("WebGPU not supported.");
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: "high-performance",
    });
    if (!adapter) throw new Error("No adapter");
    console.log(
      "Max Storage Buffers Per Shader Stage:",
      adapter.limits.maxStorageBuffersPerShaderStage
    );
    this.device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBuffersPerShaderStage: 10,
      },
    });
    this.context = this.canvas.getContext("webgpu") as GPUCanvasContext;

    this.context.configure({
      device: this.device,
      format: "rgba8unorm",
      usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });
  }

  async captureFrame(renderTarget: GPUTexture): Promise<{
    data: ArrayBufferLike;
    width: number;
    height: number;
  }> {
    if (!renderTarget) throw new Error("No render target");

    const w = this.canvas.width;
    const h = this.canvas.height;

    // Bytes per row must be multiple of 256
    const bytesPerPixel = 4;
    const unpaddedBytesPerRow = w * bytesPerPixel;
    const align = 256;
    const paddedBytesPerRow = Math.ceil(unpaddedBytesPerRow / align) * align;
    const totalSize = paddedBytesPerRow * h;

    if (!this.readbackBuffer || this.readbackBufferSize < totalSize) {
      if (this.readbackBuffer) this.readbackBuffer.destroy();
      this.readbackBuffer = this.device.createBuffer({
        size: totalSize,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      this.readbackBufferSize = totalSize;
    }

    const commandEncoder = this.device.createCommandEncoder();
    commandEncoder.copyTextureToBuffer(
      { texture: renderTarget },
      {
        buffer: this.readbackBuffer,
        bytesPerRow: paddedBytesPerRow,
        rowsPerImage: h,
      },
      { width: w, height: h, depthOrArrayLayers: 1 }
    );

    this.device.queue.submit([commandEncoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();

    await this.readbackBuffer.mapAsync(GPUMapMode.READ);
    const srcArray = new Uint8Array(this.readbackBuffer.getMappedRange());

    const desiredSize = w * h * 4;
    // Reuse CPU buffer
    if (
      !this.readbackResultBuffer ||
      this.readbackResultBuffer.byteLength !== desiredSize
    ) {
      this.readbackResultBuffer = new Uint8Array(desiredSize);
    }
    const result = this.readbackResultBuffer;

    if (paddedBytesPerRow === unpaddedBytesPerRow) {
      result.set(srcArray.subarray(0, desiredSize));
    } else {
      for (let y = 0; y < h; y++) {
        const srcOffset = y * paddedBytesPerRow;
        const dstOffset = y * unpaddedBytesPerRow;
        result.set(
          srcArray.subarray(srcOffset, srcOffset + unpaddedBytesPerRow),
          dstOffset
        );
      }
    }

    this.readbackBuffer.unmap();

    return { data: result.buffer, width: w, height: h };
  }
}
