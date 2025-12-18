// src/renderer.ts
import shaderCodeRaw from "./shader.wgsl?raw";

export class WebGPURenderer {
  public adapter!: GPUAdapter;
  public device!: GPUDevice;
  public context!: GPUCanvasContext;
  public pipeline!: GPUComputePipeline;
  bindGroupLayout!: GPUBindGroupLayout;
  bindGroup!: GPUBindGroup;

  // Screen Resources
  renderTarget!: GPUTexture;
  renderTargetView!: GPUTextureView;
  accumulateBuffer!: GPUBuffer;

  // Consolidated Uniforms
  sceneUniformBuffer!: GPUBuffer;

  // Geometry Buffers
  geometryBuffer!: GPUBuffer; // Merged (Pos + Normal + UV)
  nodesBuffer!: GPUBuffer; // Merged (TLAS + BLAS)

  // Standalone Buffers
  indexBuffer!: GPUBuffer;
  attrBuffer!: GPUBuffer;
  instanceBuffer!: GPUBuffer;

  // Texture Support
  texture!: GPUTexture;
  defaultTexture!: GPUTexture;
  sampler!: GPUSampler;

  private bufferSize = 0;
  private canvas: HTMLCanvasElement | OffscreenCanvas;

  // Cached for updating
  private blasOffset = 0;
  private vertexCount = 0;

  constructor(canvas: HTMLCanvasElement | OffscreenCanvas) {
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

    this.device.lost.then((info) => {
      console.error(`WebGPU Device Lost: ${info.reason}, ${info.message}`);
    });

    this.context = this.canvas.getContext("webgpu") as GPUCanvasContext;

    this.context.configure({
      device: this.device,
      format: "rgba8unorm",
      usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });

    // Uniform Buffer: Camera(96) + Frame(4) + BLAS_Idx(4) + Pad(8) = 112
    // Aligned to 16 bytes.
    this.sceneUniformBuffer = this.device.createBuffer({
      size: 128, // Round up to be safe
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.sampler = this.device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
      mipmapFilter: "linear",
      addressModeU: "repeat",
      addressModeV: "repeat",
    });

    this.createDefaultTexture();
    this.texture = this.defaultTexture;
  }

  createDefaultTexture() {
    const data = new Uint8Array([255, 255, 255, 255]);
    this.defaultTexture = this.device.createTexture({
      size: [1, 1, 1],
      format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });

    this.device.queue.writeTexture(
      { texture: this.defaultTexture, origin: [0, 0, 0] },
      data,
      { bytesPerRow: 256, rowsPerImage: 1 },
      [1, 1]
    );
  }

  buildPipeline(depth: number, spp: number) {
    let code = shaderCodeRaw;
    code = code.replace(
      /const\s+MAX_DEPTH\s*=\s*\d+u;/,
      `const MAX_DEPTH = ${depth}u;`
    );
    code = code.replace(/const\s+SPP\s*=\s*\d+u;/, `const SPP = ${spp}u;`);

    const shaderModule = this.device.createShaderModule({
      label: "RayTracing",
      code,
    });
    this.pipeline = this.device.createComputePipeline({
      label: "Main Pipeline",
      layout: "auto",
      compute: { module: shaderModule, entryPoint: "main" },
    });
    this.bindGroupLayout = this.pipeline.getBindGroupLayout(0);
  }

  updateScreenSize(width: number, height: number) {
    this.canvas.width = width;
    this.canvas.height = height;

    if (this.renderTarget) this.renderTarget.destroy();
    this.renderTarget = this.device.createTexture({
      size: [width, height],
      format: "rgba8unorm",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC,
    });
    this.renderTargetView = this.renderTarget.createView();

    this.bufferSize = width * height * 16;
    if (this.accumulateBuffer) this.accumulateBuffer.destroy();
    this.accumulateBuffer = this.device.createBuffer({
      size: this.bufferSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
  }

  resetAccumulation() {
    if (!this.accumulateBuffer) return;
    this.device.queue.writeBuffer(
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
    this.texture = this.device.createTexture({
      size: [1024, 1024, bitmaps.length],
      format: "rgba8unorm",
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.RENDER_ATTACHMENT,
    });

    for (let i = 0; i < bitmaps.length; i++) {
      this.device.queue.copyExternalImageToTexture(
        { source: bitmaps[i] },
        { texture: this.texture, origin: [0, 0, i] },
        [1024, 1024]
      );
    }
  }

  async createFallbackBitmap() {
    const canvas = document.createElement("canvas");
    canvas.width = 1024;
    canvas.height = 1024;
    const ctx = canvas.getContext("2d")!;
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, 1024, 1024);
    return await createImageBitmap(canvas);
  }

  // --- Buffer Management Shortcuts ---

  ensureBuffer(
    currentBuf: GPUBuffer | undefined,
    size: number,
    label: string
  ): GPUBuffer {
    if (currentBuf && currentBuf.size >= size) return currentBuf;
    if (currentBuf) currentBuf.destroy();

    // 1.5x scaling policy
    let newSize = Math.ceil(size * 1.5);
    newSize = (newSize + 3) & ~3;
    newSize = Math.max(newSize, 4);

    return this.device.createBuffer({
      label,
      size: newSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
  }

  // Handle generic buffers (indices, attributes, instances)
  updateBuffer(
    type: "index" | "attr" | "instance",
    data: Uint32Array | Float32Array
  ): boolean {
    const byteLen = data.byteLength;
    // Check if rebind needed (new buffer created)
    let needsRebind = false;
    let buf: GPUBuffer | undefined;

    if (type === "index") {
      if (!this.indexBuffer || this.indexBuffer.size < byteLen)
        needsRebind = true;
      this.indexBuffer = this.ensureBuffer(
        this.indexBuffer,
        byteLen,
        "IndexBuffer"
      );
      buf = this.indexBuffer;
    } else if (type === "attr") {
      if (!this.attrBuffer || this.attrBuffer.size < byteLen)
        needsRebind = true;
      this.attrBuffer = this.ensureBuffer(
        this.attrBuffer,
        byteLen,
        "AttrBuffer"
      );
      buf = this.attrBuffer;
    } else {
      if (!this.instanceBuffer || this.instanceBuffer.size < byteLen)
        needsRebind = true;
      this.instanceBuffer = this.ensureBuffer(
        this.instanceBuffer,
        byteLen,
        "InstanceBuffer"
      );
      buf = this.instanceBuffer;
    }

    this.device.queue.writeBuffer(buf, 0, data as any, 0, data.length);
    return needsRebind;
  }

  // Merge V, N, UV -> Geometry
  updateCombinedGeometry(
    v: Float32Array,
    n: Float32Array,
    uv: Float32Array
  ): boolean {
    const totalBytes = v.byteLength + n.byteLength + uv.byteLength;

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

    // Fully Separated Layout helping buffer.set():
    // [Positions (vec4)... | Normals (vec4)... | UVs (vec2)...]
    // Note: Positions and Normals are assumed to be stride-4 input arrays (x,y,z,w)
    // UVs are assumed to be stride-2 input arrays (u,v)

    const posCount = v.length;
    const normCount = n.length;
    const uvCount = uv.length;

    const sizeFloats = posCount + normCount + uvCount;
    const bufferData = new Float32Array(sizeFloats);

    // 1. Fill Positions
    bufferData.set(v, 0);

    // 2. Fill Normals
    bufferData.set(n, posCount);

    // 3. Fill UVs
    bufferData.set(uv, posCount + normCount);

    this.device.queue.writeBuffer(this.geometryBuffer, 0, bufferData);
    return needsRebind;
  }

  // Merge TLAS, BLAS -> Nodes
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

    // Write TLAS at 0
    this.device.queue.writeBuffer(this.nodesBuffer, 0, tlas as any);
    // Write BLAS after TLAS
    this.device.queue.writeBuffer(this.nodesBuffer, tlasBytes, blas as any);

    this.blasOffset = tlas.length / 8;

    // console.log(`BVH Updated: TLAS=${tlas.length/8} nodes, BLAS=${blas.length/8} nodes. Offset=${this.blasOffset}`);

    return needsRebind;
  }

  updateSceneUniforms(cameraData: Float32Array, frameCount: number) {
    if (!this.sceneUniformBuffer) return;
    this.device.queue.writeBuffer(
      this.sceneUniformBuffer,
      0,
      cameraData as any
    );

    const mixed = new Uint32Array([
      frameCount,
      this.blasOffset,
      this.vertexCount,
      0,
    ]);
    this.device.queue.writeBuffer(this.sceneUniformBuffer, 96, mixed);
  }

  recreateBindGroup() {
    if (
      !this.renderTargetView ||
      !this.accumulateBuffer ||
      !this.geometryBuffer ||
      !this.nodesBuffer ||
      !this.sceneUniformBuffer
    )
      return;

    this.bindGroup = this.device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: this.renderTargetView },
        { binding: 1, resource: { buffer: this.accumulateBuffer } },
        { binding: 2, resource: { buffer: this.sceneUniformBuffer } },

        { binding: 3, resource: { buffer: this.geometryBuffer } },
        { binding: 4, resource: { buffer: this.indexBuffer } },
        { binding: 5, resource: { buffer: this.attrBuffer } },
        { binding: 6, resource: { buffer: this.nodesBuffer } },
        { binding: 7, resource: { buffer: this.instanceBuffer } },

        {
          binding: 8,
          resource: this.texture.createView({ dimension: "2d-array" }),
        },
        { binding: 9, resource: this.sampler },
      ],
    });
  }

  encodeTileCommand(
    commandEncoder: GPUCommandEncoder,
    offsetX: number,
    offsetY: number,
    width: number,
    height: number,
    frameCount: number
  ) {
    if (!this.bindGroup) return;

    // 1. Uniform Update (Writes to buffer immediately)
    const params = new Uint32Array([
      frameCount,
      this.blasOffset,
      this.vertexCount,
      offsetX,
      offsetY,
      0,
      0,
    ]);
    this.device.queue.writeBuffer(this.sceneUniformBuffer, 96, params);

    // 2. Compute Pass
    const dispatchX = Math.ceil(width / 8);
    const dispatchY = Math.ceil(height / 8);

    const pass = commandEncoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.dispatchWorkgroups(dispatchX, dispatchY);
    pass.end();
  }

  present(commandEncoder: GPUCommandEncoder) {
    commandEncoder.copyTextureToTexture(
      { texture: this.renderTarget },
      { texture: this.context.getCurrentTexture() },
      {
        width: this.canvas.width,
        height: this.canvas.height,
        depthOrArrayLayers: 1,
      }
    );
  }
}
