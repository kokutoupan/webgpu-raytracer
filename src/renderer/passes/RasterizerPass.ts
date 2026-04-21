import { WebGPUContext } from "../WebGPUContext";
import { ResourceManager } from "../ResourceManager";
import rasterizerShaderCode from "../../shaders/Rasterizer.wgsl?raw";

export class RasterizerPass {
  pipeline!: GPURenderPipeline;
  bindGroupLayout!: GPUBindGroupLayout;
  bindGroup!: GPUBindGroup;
  ctx: WebGPUContext;

  constructor(ctx: WebGPUContext) {
    this.ctx = ctx;
  }

  buildPipeline() {
    const module = this.ctx.device.createShaderModule({
      label: "Rasterizer Shader",
      code: rasterizerShaderCode,
    });

    this.pipeline = this.ctx.device.createRenderPipeline({
      label: "Rasterizer Pipeline",
      layout: "auto",
      vertex: {
        module: module,
        entryPoint: "vs_main",
      },
      fragment: {
        module: module,
        entryPoint: "fs_main",
        targets: [
          {
            format: "rgba8unorm",
          },
        ],
      },
      primitive: {
        topology: "triangle-list",
        cullMode: "none",
      },
      depthStencil: {
        depthWriteEnabled: true,
        depthCompare: "less",
        format: "depth24plus",
      },
    });

    this.bindGroupLayout = this.pipeline.getBindGroupLayout(0);
  }

  updateBindGroup(res: ResourceManager) {
    if (!res.sceneUniformBuffer || !res.geometryBuffer || !res.topologyBuffer || !res.instanceBuffer) return;

    this.bindGroup = this.ctx.device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 2, resource: { buffer: res.sceneUniformBuffer } },
        {
          binding: 3,
          resource: {
            buffer: res.geometryBuffer,
            offset: 0,
            size: res.vertexCount * 16,
          },
        },
        { binding: 4, resource: { buffer: res.topologyBuffer } },
        { binding: 6, resource: { buffer: res.instanceBuffer } },
        {
          binding: 7,
          resource: res.texture.createView({ dimension: "2d-array" }),
        },
        { binding: 8, resource: res.sampler },
        {
          binding: 11,
          resource: {
            buffer: res.geometryBuffer,
            offset: res.normOffset,
            size: res.vertexCount * 16,
          },
        },
        {
          binding: 12,
          resource: {
            buffer: res.geometryBuffer,
            offset: res.uvOffset,
            size: res.vertexCount * 8,
          },
        },
      ],
    });
  }

  private depthTexture!: GPUTexture;
  private depthTextureView!: GPUTextureView;

  execute(commandEncoder: GPUCommandEncoder, res: ResourceManager) {
    if (!this.bindGroup || !res.drawCommandBuffer) return;

    // Create depth texture if needed
    if (!this.depthTexture || this.depthTexture.width !== this.ctx.canvas.width || this.depthTexture.height !== this.ctx.canvas.height) {
      if (this.depthTexture) this.depthTexture.destroy();
      this.depthTexture = this.ctx.device.createTexture({
        size: [this.ctx.canvas.width, this.ctx.canvas.height],
        format: "depth24plus",
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      });
      this.depthTextureView = this.depthTexture.createView();
    }

    const renderPassDescriptor: GPURenderPassDescriptor = {
      colorAttachments: [
        {
          view: res.renderTargetView,
          loadOp: 'clear', 
          storeOp: 'store',
          clearValue: { r: 0.1, g: 0.1, b: 0.2, a: 1.0 },
        },
      ],
      depthStencilAttachment: {
        view: this.depthTextureView,
        depthClearValue: 1.0,
        depthLoadOp: "clear",
        depthStoreOp: "store",
      },
    };

    const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
    passEncoder.setPipeline(this.pipeline);
    passEncoder.setBindGroup(0, this.bindGroup);
    
    // Draw directly using CPU-side commands
    if (res.drawCommandsArray) {
      const arr = res.drawCommandsArray;
      const instanceCount = res.instanceCount;
      for (let i = 0; i < instanceCount; i++) {
        const v_count = arr[i * 4 + 0];
        const i_count = arr[i * 4 + 1];
        const v_start = arr[i * 4 + 2];
        const i_start = arr[i * 4 + 3];
        if (v_count > 0 && i_count > 0) {
          passEncoder.draw(v_count, i_count, v_start, i_start);
        }
      }
    }
    
    passEncoder.end();
  }
}