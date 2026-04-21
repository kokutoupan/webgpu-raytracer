import { WebGPUContext } from "./WebGPUContext";

export class ResourceManager {
  // Screen Resources
  renderTarget!: GPUTexture;
  renderTargetView!: GPUTextureView;
  accumulateBuffer!: GPUBuffer;

  // Consolidated Uniforms
  sceneUniformBuffer!: GPUBuffer;

  // Geometry Buffers
  geometryBuffer!: GPUBuffer; // Merged (Pos + Normal + UV)
  nodesBuffer!: GPUBuffer; // Merged (TLAS + BLAS)
  topologyBuffer!: GPUBuffer; // Merged (Indices + Attributes)

  // Standalone Buffers
  instanceBuffer!: GPUBuffer;
  lightsBuffer!: GPUBuffer;

  // Texture Support
  texture!: GPUTexture;
  defaultTexture!: GPUTexture;
  sampler!: GPUSampler;

  // TAA Resources
  historyTextures: GPUTexture[] = [];
  historyTextureViews: GPUTextureView[] = [];
  historyIndex = 0;
  prevCameraData: Float32Array = new Float32Array(24);
  jitter = { x: 0, y: 0 };
  prevJitter = { x: 0, y: 0 };

  private bufferSize = 0;

  // Cached for updating
  blasOffset = 0;
  vertexCount = 0;
  normOffset = 0;
  uvOffset = 0;
  lightCount = 0;

  seed = Math.floor(Math.random() * 0xffffff);

  // Reuse to avoid allocation
  private uniformMixedData = new Uint32Array(12);

  ctx: WebGPUContext;

  constructor(ctx: WebGPUContext) {
    this.ctx = ctx;
  }

  init() {
    // Uniform Buffer: Current Camera(96) + Prev Camera(96) + Mixed(48) = 240
    this.sceneUniformBuffer = this.ctx.device.createBuffer({
      size: 256,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.sampler = this.ctx.device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
      mipmapFilter: "linear",
      addressModeU: "repeat",
      addressModeV: "repeat",
    });

    this.createDefaultTexture();
    this.texture = this.defaultTexture;
  }

  private createDefaultTexture() {
    const data = new Uint8Array([255, 255, 255, 255]);
    this.defaultTexture = this.ctx.device.createTexture({
      size: [1, 1, 1],
      format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });

    this.ctx.device.queue.writeTexture(
      { texture: this.defaultTexture, origin: [0, 0, 0] },
      data,
      { bytesPerRow: 256, rowsPerImage: 1 },
      [1, 1]
    );
  }

  updateScreenSize(width: number, height: number) {
    if (this.renderTarget) this.renderTarget.destroy();
    this.renderTarget = this.ctx.device.createTexture({
      size: [width, height],
      format: "rgba8unorm",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.renderTargetView = this.renderTarget.createView();

    this.bufferSize = width * height * 16;
    if (this.accumulateBuffer) this.accumulateBuffer.destroy();
    this.accumulateBuffer = this.ctx.device.createBuffer({
      size: this.bufferSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Create history textures for TAA (Stored in Linear HDR)
    for (let i = 0; i < 2; i++) {
      if (this.historyTextures[i]) this.historyTextures[i].destroy();
      this.historyTextures[i] = this.ctx.device.createTexture({
        size: [width, height],
        format: "rgba16float",
        usage:
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.STORAGE_BINDING |
          GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
      });
      this.historyTextureViews[i] = this.historyTextures[i].createView();
    }
  }

  resetAccumulation() {
    if (!this.accumulateBuffer) return;
    this.ctx.device.queue.writeBuffer(
      this.accumulateBuffer,
      0,
      new Float32Array(this.bufferSize / 4)
    );
  }

  async loadTexturesFromWorld(bridge: any) {
    const count = bridge.textureCount;
    if (count === 0) {
      this.createDefaultTexture();
      return;
    }
    console.log(`Loading ${count} textures...`);
    const bitmaps: ImageBitmap[] = [];
    for (let i = 0; i < count; i++) {
      const data = bridge.getTexture(i);
      if (data) {
        try {
          const blob = new Blob([data]);
          const bmp = await createImageBitmap(blob, {
            resizeWidth: 1024,
            resizeHeight: 1024,
          });
          bitmaps.push(bmp);
        } catch (e) {
          console.warn(`Failed tex ${i}`, e);
          bitmaps.push(await this.createFallbackBitmap());
        }
      } else {
        bitmaps.push(await this.createFallbackBitmap());
      }
    }

    if (this.texture) this.texture.destroy();
    this.texture = this.ctx.device.createTexture({
      size: [1024, 1024, bitmaps.length],
      format: "rgba8unorm",
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.RENDER_ATTACHMENT,
    });

    for (let i = 0; i < bitmaps.length; i++) {
      this.ctx.device.queue.copyExternalImageToTexture(
        { source: bitmaps[i] },
        { texture: this.texture, origin: [0, 0, i] },
        [1024, 1024]
      );
    }
    await this.ctx.device.queue.onSubmittedWorkDone();
  }

  private async createFallbackBitmap() {
    const canvas = document.createElement("canvas");
    canvas.width = 1024;
    canvas.height = 1024;
    const canvasCtx = canvas.getContext("2d")!;
    canvasCtx.fillStyle = "white";
    canvasCtx.fillRect(0, 0, 1024, 1024);
    return await createImageBitmap(canvas);
  }

  private ensureBuffer(
    currentBuf: GPUBuffer | undefined,
    size: number,
    label: string
  ): GPUBuffer {
    if (currentBuf && currentBuf.size >= size) return currentBuf;
    if (currentBuf) currentBuf.destroy();

    // 1.5x scaling policy
    let newSize = Math.ceil(size * 1.5);
    newSize = (newSize + 3) & ~3;
    newSize = Math.max(newSize, 16);

    return this.ctx.device.createBuffer({
      label,
      size: newSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
  }

  updateBuffer(
    type: "topology" | "instance" | "lights",
    data: Uint32Array | Float32Array
  ): boolean {
    const byteLen = data.byteLength;
    let needsRebind = false;
    let buf: GPUBuffer | undefined;

    if (type === "topology") {
      if (!this.topologyBuffer || this.topologyBuffer.size < byteLen)
        needsRebind = true;
      this.topologyBuffer = this.ensureBuffer(
        this.topologyBuffer,
        byteLen,
        "TopologyBuffer"
      );
      buf = this.topologyBuffer;
    } else if (type === "instance") {
      if (!this.instanceBuffer || this.instanceBuffer.size < byteLen)
        needsRebind = true;
      this.instanceBuffer = this.ensureBuffer(
        this.instanceBuffer,
        byteLen,
        "InstanceBuffer"
      );
      buf = this.instanceBuffer;
    } else {
      if (!this.lightsBuffer || this.lightsBuffer.size < byteLen)
        needsRebind = true;
      this.lightsBuffer = this.ensureBuffer(
        this.lightsBuffer,
        byteLen,
        "LightsBuffer"
      );
      buf = this.lightsBuffer;
    }

    this.ctx.device.queue.writeBuffer(buf, 0, data as any, 0, data.length);
    return needsRebind;
  }

  updateCombinedGeometry(
    v: Float32Array,
    n: Float32Array,
    uv: Float32Array
  ): boolean {
    const align = 256;
    const posLen = v.byteLength;
    this.normOffset = Math.ceil(posLen / align) * align;
    const normLen = n.byteLength;
    this.uvOffset = Math.ceil((this.normOffset + normLen) / align) * align;
    const totalBytes = this.uvOffset + uv.byteLength;

    let needsRebind = false;
    if (!this.geometryBuffer || this.geometryBuffer.size < totalBytes)
      needsRebind = true;

    const vertexCount = v.length / 4;
    this.vertexCount = vertexCount;

    this.geometryBuffer = this.ensureBuffer(
      this.geometryBuffer,
      totalBytes,
      "GeometryBuffer"
    );

    const hasUV = uv.length >= vertexCount * 2;
    if (!hasUV && vertexCount > 0) {
      console.warn(
        `UV buffer mismatch: V=${vertexCount}, UV=${uv.length / 2}. Filling 0.`
      );
    }

    this.ctx.device.queue.writeBuffer(this.geometryBuffer, 0, v as any);
    this.ctx.device.queue.writeBuffer(this.geometryBuffer, this.normOffset, n as any);
    this.ctx.device.queue.writeBuffer(this.geometryBuffer, this.uvOffset, uv as any);

    return needsRebind;
  }

  updateCombinedBVH(tlas: Float32Array, blas: Float32Array): boolean {
    const tlasBytes = tlas.byteLength;
    const blasBytes = blas.byteLength;
    const totalBytes = tlasBytes + blasBytes;

    let needsRebind = false;
    if (!this.nodesBuffer || this.nodesBuffer.size < totalBytes)
      needsRebind = true;

    this.nodesBuffer = this.ensureBuffer(
      this.nodesBuffer,
      totalBytes,
      "NodesBuffer"
    );

    this.ctx.device.queue.writeBuffer(this.nodesBuffer, 0, tlas as any);
    this.ctx.device.queue.writeBuffer(this.nodesBuffer, tlasBytes, blas as any);

    this.blasOffset = tlas.length / 8;

    return needsRebind;
  }

  getHalton(index: number, base: number): number {
    let f = 1;
    let r = 0;
    while (index > 0) {
      f = f / base;
      r = r + f * (index % base);
      index = Math.floor(index / base);
    }
    return r;
  }

  updateSceneUniforms(
    cameraData: Float32Array,
    frameCount: number,
    lightCount: number
  ) {
    this.lightCount = lightCount;
    if (!this.sceneUniformBuffer) return;

    const jitterX = this.getHalton((frameCount % 16) + 1, 2) - 0.5;
    const jitterY = this.getHalton((frameCount % 16) + 1, 3) - 0.5;
    this.jitter = {
      x: jitterX / this.ctx.canvas.width,
      y: jitterY / this.ctx.canvas.height,
    };

    this.ctx.device.queue.writeBuffer(this.sceneUniformBuffer, 0, cameraData as any);
    this.ctx.device.queue.writeBuffer(this.sceneUniformBuffer, 96, this.prevCameraData as any);

    this.uniformMixedData[0] = frameCount;
    this.uniformMixedData[1] = this.blasOffset;
    this.uniformMixedData[2] = this.vertexCount;
    this.uniformMixedData[3] = this.seed;
    this.uniformMixedData[4] = lightCount;
    this.uniformMixedData[5] = this.ctx.canvas.width;
    this.uniformMixedData[6] = this.ctx.canvas.height;
    this.uniformMixedData[7] = 0; // Padding

    const floatView = new Float32Array(this.uniformMixedData.buffer);
    floatView[8] = this.jitter.x;
    floatView[9] = this.jitter.y;

    this.ctx.device.queue.writeBuffer(this.sceneUniformBuffer, 192, this.uniformMixedData as any);
    this.prevCameraData.set(cameraData);
  }

  updateFrameUniforms(frameCount: number, totalFrames: number) {
    const jitterX = this.getHalton((totalFrames % 16) + 1, 2) - 0.5;
    const jitterY = this.getHalton((totalFrames % 16) + 1, 3) - 0.5;

    this.prevJitter.x = this.jitter.x;
    this.prevJitter.y = this.jitter.y;

    this.jitter = {
      x: jitterX / this.ctx.canvas.width,
      y: jitterY / this.ctx.canvas.height,
    };

    this.uniformMixedData[0] = frameCount;
    this.uniformMixedData[1] = this.blasOffset;
    this.uniformMixedData[2] = this.vertexCount;
    this.uniformMixedData[3] = this.seed;
    this.uniformMixedData[4] = this.lightCount;
    this.uniformMixedData[5] = this.ctx.canvas.width;
    this.uniformMixedData[6] = this.ctx.canvas.height;
    this.uniformMixedData[7] = 0; // Padding

    const floatView = new Float32Array(this.uniformMixedData.buffer);
    floatView[8] = this.jitter.x;
    floatView[9] = this.jitter.y;
    floatView[10] = this.prevJitter.x;
    floatView[11] = this.prevJitter.y;

    this.ctx.device.queue.writeBuffer(
      this.sceneUniformBuffer,
      192,
      this.uniformMixedData as any
    );
  }
}
