// src/renderer.ts
import shaderCodeRaw from './shader.wgsl?raw';

export class WebGPURenderer {
  device!: GPUDevice;
  context!: GPUCanvasContext;
  pipeline!: GPUComputePipeline;
  bindGroupLayout!: GPUBindGroupLayout;
  bindGroup!: GPUBindGroup;

  // Screen Resources
  renderTarget!: GPUTexture;
  renderTargetView!: GPUTextureView;
  accumulateBuffer!: GPUBuffer;

  // Uniforms
  frameUniformBuffer!: GPUBuffer;
  cameraUniformBuffer!: GPUBuffer;

  // Geometry Buffers
  vertexBuffer!: GPUBuffer;
  normalBuffer!: GPUBuffer;
  indexBuffer!: GPUBuffer;
  attrBuffer!: GPUBuffer;
  tlasBuffer!: GPUBuffer;
  blasBuffer!: GPUBuffer;
  instanceBuffer!: GPUBuffer;

  private bufferSize = 0;
  private canvas: HTMLCanvasElement;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
  }

  async init() {
    if (!navigator.gpu) throw new Error("WebGPU not supported.");
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("No adapter");
    this.device = await adapter.requestDevice();
    this.context = this.canvas.getContext("webgpu") as GPUCanvasContext;

    this.context.configure({
      device: this.device,
      format: 'rgba8unorm',
      usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT
    });

    // 初期バッファ作成
    this.frameUniformBuffer = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.cameraUniformBuffer = this.device.createBuffer({ size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  }

  buildPipeline(depth: number, spp: number) {
    let code = shaderCodeRaw;
    // 正規表現で定数を書き換え
    code = code.replace(/const\s+MAX_DEPTH\s*=\s*\d+u;/, `const MAX_DEPTH = ${depth}u;`);
    code = code.replace(/const\s+SPP\s*=\s*\d+u;/, `const SPP = ${spp}u;`);

    const shaderModule = this.device.createShaderModule({ label: "RayTracing", code });
    this.pipeline = this.device.createComputePipeline({
      label: "Main Pipeline",
      layout: "auto",
      compute: { module: shaderModule, entryPoint: "main" }
    });
    this.bindGroupLayout = this.pipeline.getBindGroupLayout(0);
  }

  updateScreenSize(width: number, height: number) {
    this.canvas.width = width;
    this.canvas.height = height;

    if (this.renderTarget) this.renderTarget.destroy();
    this.renderTarget = this.device.createTexture({
      size: [width, height],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC
    });
    this.renderTargetView = this.renderTarget.createView();

    this.bufferSize = width * height * 16;
    if (this.accumulateBuffer) this.accumulateBuffer.destroy();
    this.accumulateBuffer = this.device.createBuffer({
      size: this.bufferSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
  }

  resetAccumulation() {
    if (!this.accumulateBuffer) return;
    this.device.queue.writeBuffer(this.accumulateBuffer, 0, new Float32Array(this.bufferSize / 4));
  }

  // バッファの更新・再生成ロジック
  updateGeometryBuffer(
    type: 'tlas' | 'blas' | 'instance' | 'vertex' | 'normal' | 'index' | 'attr',
    data: Float32Array<ArrayBuffer> | Uint32Array<ArrayBuffer>
  ): boolean {
    // マッピング
    const map: Record<string, GPUBuffer> = {
      tlas: this.tlasBuffer, blas: this.blasBuffer, instance: this.instanceBuffer,
      vertex: this.vertexBuffer, normal: this.normalBuffer, index: this.indexBuffer, attr: this.attrBuffer
    };
    let currentBuf = map[type];

    // 新規作成が必要かチェック
    if (!currentBuf || currentBuf.size < data.byteLength) {
      if (currentBuf) currentBuf.destroy();
      const size = Math.max(data.byteLength, 4); // 最低4バイト
      const newBuf = this.device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(newBuf, 0, data);

      // クラスプロパティを更新
      switch (type) {
        case 'tlas': this.tlasBuffer = newBuf; break;
        case 'blas': this.blasBuffer = newBuf; break;
        case 'instance': this.instanceBuffer = newBuf; break;
        case 'vertex': this.vertexBuffer = newBuf; break;
        case 'normal': this.normalBuffer = newBuf; break;
        case 'index': this.indexBuffer = newBuf; break;
        case 'attr': this.attrBuffer = newBuf; break;
      }
      return true; // BindGroupの再生成が必要
    } else {
      // サイズが足りれば書き込みのみ
      if (data.byteLength > 0) {
        this.device.queue.writeBuffer(currentBuf, 0, data);
      }
      return false;
    }
  }

  updateCameraBuffer(data: Float32Array<ArrayBuffer>) {
    this.device.queue.writeBuffer(this.cameraUniformBuffer, 0, data);
  }

  updateFrameBuffer(frameCount: number) {
    this.device.queue.writeBuffer(this.frameUniformBuffer, 0, new Uint32Array([frameCount]));
  }

  recreateBindGroup() {
    if (!this.renderTargetView || !this.accumulateBuffer || !this.vertexBuffer || !this.tlasBuffer) return;

    this.bindGroup = this.device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: this.renderTargetView },
        { binding: 1, resource: { buffer: this.accumulateBuffer } },
        { binding: 2, resource: { buffer: this.frameUniformBuffer } },
        { binding: 3, resource: { buffer: this.cameraUniformBuffer } },
        { binding: 4, resource: { buffer: this.vertexBuffer } },
        { binding: 5, resource: { buffer: this.indexBuffer } },
        { binding: 6, resource: { buffer: this.attrBuffer } },
        { binding: 7, resource: { buffer: this.tlasBuffer } },
        { binding: 8, resource: { buffer: this.normalBuffer } },
        { binding: 9, resource: { buffer: this.blasBuffer } },
        { binding: 10, resource: { buffer: this.instanceBuffer } },
      ],
    });
  }

  render(frameCount: number) {
    if (!this.bindGroup) return;

    this.updateFrameBuffer(frameCount);

    const dispatchX = Math.ceil(this.canvas.width / 8);
    const dispatchY = Math.ceil(this.canvas.height / 8);

    const commandEncoder = this.device.createCommandEncoder();
    const pass = commandEncoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.dispatchWorkgroups(dispatchX, dispatchY);
    pass.end();

    commandEncoder.copyTextureToTexture(
      { texture: this.renderTarget },
      { texture: this.context.getCurrentTexture() },
      { width: this.canvas.width, height: this.canvas.height, depthOrArrayLayers: 1 }
    );

    this.device.queue.submit([commandEncoder.finish()]);
  }
}
