import raytracerCodeRaw from "./Raytracer.wgsl?raw";
import postprocessCodeRaw from "./PostProcess.wgsl?raw";

export class WebGPURenderer {
  device!: GPUDevice;
  context!: GPUCanvasContext;
  pipeline!: GPUComputePipeline;
  postprocessPipeline!: GPUComputePipeline;
  bindGroupLayout!: GPUBindGroupLayout;
  postprocessBindGroupLayout!: GPUBindGroupLayout;
  bindGroup!: GPUBindGroup;
  postprocessBindGroup!: GPUBindGroup;

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

  reservoirBuffer!: GPUBuffer;

  private bufferSize = 0;
  private canvas: HTMLCanvasElement;

  // Cached for updating
  private blasOffset = 0;
  private vertexCount = 0;

  private seed = Math.floor(Math.random() * 0xffffff);

  // TAA Resources
  historyTextures: GPUTexture[] = [];
  historyTextureViews: GPUTextureView[] = [];
  historyIndex = 0;
  prevCameraData: Float32Array = new Float32Array(24);
  jitter = { x: 0, y: 0 };
  prevJitter = { x: 0, y: 0 };

  // Reuse to avoid allocation
  private uniformMixedData = new Uint32Array(12);

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

    // Uniform Buffer: Current Camera(96) + Prev Camera(96) + Mixed(48) = 240
    // Mixed: frame(4), blas(4), vertex(4), seed(4), light(4), w(4), h(4), jitterX(4), jitterY(4), pad(12)
    this.sceneUniformBuffer = this.device.createBuffer({
      size: 256,
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
    let raytraceCode = raytracerCodeRaw;
    raytraceCode = raytraceCode.replace(
      /const\s+MAX_DEPTH\s*=\s*\d+u;/,
      `const MAX_DEPTH = ${depth}u;`
    );
    raytraceCode = raytraceCode.replace(
      /const\s+SPP\s*=\s*\d+u;/,
      `const SPP = ${spp}u;`
    );

    const raytraceModule = this.device.createShaderModule({
      label: "RayTracing",
      code: raytraceCode,
    });
    this.pipeline = this.device.createComputePipeline({
      label: "Main Pipeline",
      layout: "auto",
      compute: { module: raytraceModule, entryPoint: "main" },
    });
    this.bindGroupLayout = this.pipeline.getBindGroupLayout(0);

    const postprocessModule = this.device.createShaderModule({
      label: "PostProcess",
      code: postprocessCodeRaw,
    });
    this.postprocessPipeline = this.device.createComputePipeline({
      label: "PostProcess Pipeline",
      layout: "auto",
      compute: { module: postprocessModule, entryPoint: "main" },
    });
    this.postprocessBindGroupLayout =
      this.postprocessPipeline.getBindGroupLayout(0);
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

    // Create history textures for TAA (Stored in Linear HDR)
    for (let i = 0; i < 2; i++) {
      if (this.historyTextures[i]) this.historyTextures[i].destroy();
      this.historyTextures[i] = this.device.createTexture({
        size: [width, height],
        format: "rgba16float",
        usage:
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.STORAGE_BINDING |
          GPUTextureUsage.COPY_DST,
      });
      this.historyTextureViews[i] = this.historyTextures[i].createView();
    }

    // 画面解像度分のリザーバを用意
    // 構造体サイズに合わせて調整 (今回は余裕を見て 32 bytes * pixel数 * 2)
    const reservoirSize = width * height * 32 * 2;
    if (this.reservoirBuffer) this.reservoirBuffer.destroy();
    this.reservoirBuffer = this.device.createBuffer({
      label: "ReservoirBuffer",
      size: reservoirSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
  }

  resetAccumulation() {
    if (!this.accumulateBuffer || !this.reservoirBuffer) return;
    this.device.queue.writeBuffer(
      this.accumulateBuffer,
      0,
      new Float32Array(this.bufferSize / 4)
    );

    this.device.queue.writeBuffer(
      this.reservoirBuffer,
      0,
      new Float32Array((this.canvas.width * this.canvas.height * 32 * 2) / 4) // サイズ注意
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
    newSize = Math.max(newSize, 16);

    return this.device.createBuffer({
      label,
      size: newSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
  }

  // Handle generic buffers (topology, instances, lights)
  updateBuffer(
    type: "topology" | "instance" | "lights",
    data: Uint32Array | Float32Array
  ): boolean {
    const byteLen = data.byteLength;
    // Check if rebind needed (new buffer created)
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

    // Direct write to buffer to save CPU memory
    let offset = 0;

    // 1. Write Positions
    this.device.queue.writeBuffer(this.geometryBuffer, offset, v as any);
    offset += v.byteLength;

    // 2. Write Normals
    this.device.queue.writeBuffer(this.geometryBuffer, offset, n as any);
    offset += n.byteLength;

    // 3. Write UVs
    this.device.queue.writeBuffer(this.geometryBuffer, offset, uv as any);

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

  updateSceneUniforms(
    cameraData: Float32Array,
    frameCount: number,
    lightCount: number
  ) {
    this.lightCount = lightCount;
    if (!this.sceneUniformBuffer) return;

    // Halton jitter for TAA
    const getHalton = (index: number, base: number) => {
      let f = 1;
      let r = 0;
      while (index > 0) {
        f = f / base;
        r = r + f * (index % base);
        index = Math.floor(index / base);
      }
      return r;
    };

    // Low SPP (1-4) often benefits from jitter.
    // If SPP is high, jitter might be less critical but still good for temporal stability.
    const jitterX = getHalton((frameCount % 16) + 1, 2) - 0.5;
    const jitterY = getHalton((frameCount % 16) + 1, 3) - 0.5;
    this.jitter = {
      x: jitterX / this.canvas.width,
      y: jitterY / this.canvas.height,
    };

    // Write Current Camera
    this.device.queue.writeBuffer(
      this.sceneUniformBuffer,
      0,
      cameraData as any
    );
    // Write Previous Camera
    this.device.queue.writeBuffer(
      this.sceneUniformBuffer,
      96,
      this.prevCameraData as any
    );

    this.uniformMixedData[0] = frameCount;
    this.uniformMixedData[1] = this.blasOffset;
    this.uniformMixedData[2] = this.vertexCount;
    this.uniformMixedData[3] = this.seed;
    this.uniformMixedData[4] = lightCount;
    this.uniformMixedData[5] = this.canvas.width;
    this.uniformMixedData[6] = this.canvas.height;
    this.uniformMixedData[7] = 0; // Padding for 8-byte alignment of jitter

    const floatView = new Float32Array(this.uniformMixedData.buffer);
    floatView[8] = this.jitter.x;
    floatView[9] = this.jitter.y;

    this.device.queue.writeBuffer(
      this.sceneUniformBuffer,
      192, // After Current(96) and Prev(96)
      this.uniformMixedData as any
    );

    // Save current camera for next frame
    this.prevCameraData.set(cameraData);
  }

  recreateBindGroup() {
    if (
      !this.renderTargetView ||
      !this.accumulateBuffer ||
      !this.geometryBuffer ||
      !this.nodesBuffer ||
      !this.sceneUniformBuffer ||
      !this.lightsBuffer ||
      !this.reservoirBuffer
    )
      return;

    this.bindGroup = this.device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 1, resource: { buffer: this.accumulateBuffer } },
        { binding: 2, resource: { buffer: this.sceneUniformBuffer } },
        { binding: 3, resource: { buffer: this.geometryBuffer } },
        { binding: 4, resource: { buffer: this.topologyBuffer } },
        { binding: 5, resource: { buffer: this.nodesBuffer } },
        { binding: 6, resource: { buffer: this.instanceBuffer } },
        {
          binding: 7,
          resource: this.texture.createView({ dimension: "2d-array" }),
        },
        { binding: 8, resource: this.sampler },
        { binding: 9, resource: { buffer: this.lightsBuffer } },
        { binding: 10, resource: { buffer: this.reservoirBuffer } },
      ],
    });

    this.postprocessBindGroup = this.device.createBindGroup({
      layout: this.postprocessBindGroupLayout,
      entries: [
        { binding: 0, resource: this.renderTargetView },
        { binding: 1, resource: { buffer: this.accumulateBuffer } },
        { binding: 2, resource: { buffer: this.sceneUniformBuffer } },
        {
          binding: 3,
          resource: this.historyTextureViews[1 - this.historyIndex],
        }, // Previous frame (Read)
        { binding: 4, resource: this.sampler },
        {
          binding: 5,
          resource: this.historyTextureViews[this.historyIndex],
        }, // Current frame (Write)
      ],
    });
  }

  private totalFrames = 0;

  private lightCount = 0;

  // ★ 名前を変更: render -> compute
  compute(frameCount: number) {
    if (!this.bindGroup || !this.postprocessBindGroup) return;

    this.totalFrames++;

    // Halton jitter for TAA (Persistent counter)
    const getHalton = (index: number, base: number) => {
      let f = 1;
      let r = 0;
      while (index > 0) {
        f = f / base;
        r = r + f * (index % base);
        index = Math.floor(index / base);
      }
      return r;
    };

    const jitterX = getHalton((this.totalFrames % 16) + 1, 2) - 0.5;
    const jitterY = getHalton((this.totalFrames % 16) + 1, 3) - 0.5;

    // Store current as previous
    this.prevJitter.x = this.jitter.x;
    this.prevJitter.y = this.jitter.y;

    this.jitter = {
      x: jitterX / this.canvas.width,
      y: jitterY / this.canvas.height,
    };

    // Uniformの更新
    this.uniformMixedData[0] = frameCount;
    this.uniformMixedData[1] = this.blasOffset;
    this.uniformMixedData[2] = this.vertexCount;
    this.uniformMixedData[3] = this.seed;
    this.uniformMixedData[4] = this.lightCount; // RESTORE: Use cached lightCount
    this.uniformMixedData[5] = this.canvas.width;
    this.uniformMixedData[6] = this.canvas.height;
    this.uniformMixedData[7] = 0; // Padding

    const floatView = new Float32Array(this.uniformMixedData.buffer);
    floatView[8] = this.jitter.x;
    floatView[9] = this.jitter.y;
    floatView[10] = this.prevJitter.x;
    floatView[11] = this.prevJitter.y;

    this.device.queue.writeBuffer(
      this.sceneUniformBuffer,
      192,
      this.uniformMixedData as any
    );

    const dispatchX = Math.ceil(this.canvas.width / 8);
    const dispatchY = Math.ceil(this.canvas.height / 8);

    const commandEncoder = this.device.createCommandEncoder();

    // 1. Raytrace Pass
    const rayPass = commandEncoder.beginComputePass();
    rayPass.setPipeline(this.pipeline);
    rayPass.setBindGroup(0, this.bindGroup);
    rayPass.dispatchWorkgroups(dispatchX, dispatchY);
    rayPass.end();

    this.device.queue.submit([commandEncoder.finish()]);
  }

  // ★ 新設: 画面への転送のみを行うメソッド
  present() {
    if (!this.renderTarget || !this.postprocessBindGroup) return;

    const dispatchX = Math.ceil(this.canvas.width / 8);
    const dispatchY = Math.ceil(this.canvas.height / 8);

    const commandEncoder = this.device.createCommandEncoder();

    // 1. Postprocess Pass (Normalizes and tone maps accumulateBuffer into renderTarget)
    const postPass = commandEncoder.beginComputePass();
    postPass.setPipeline(this.postprocessPipeline);
    postPass.setBindGroup(0, this.postprocessBindGroup);
    postPass.dispatchWorkgroups(dispatchX, dispatchY);
    postPass.end();

    // 2. Copy renderTarget to Canvas
    commandEncoder.copyTextureToTexture(
      { texture: this.renderTarget },
      { texture: this.context.getCurrentTexture() },
      {
        width: this.canvas.width,
        height: this.canvas.height,
        depthOrArrayLayers: 1,
      }
    );

    this.device.queue.submit([commandEncoder.finish()]);

    // Swap history index
    this.historyIndex = 1 - this.historyIndex;
    this.recreateBindGroup(); // To update history binding
  }
}
