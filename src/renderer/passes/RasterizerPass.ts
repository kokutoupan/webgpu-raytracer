import { WebGPUContext } from "../WebGPUContext";
import { ResourceManager } from "../ResourceManager";

/**
 * Skeleton for the future Rasterizer Pass.
 */
export class RasterizerPass {
  pipeline!: GPURenderPipeline;

  ctx: WebGPUContext;

  constructor(ctx: WebGPUContext) {
    this.ctx = ctx;
  }

  buildPipeline() {
    // TODO: Create GPURenderPipeline for rasterization
  }

  updateBindGroup(_res: ResourceManager) {
    // TODO: Create bind groups
  }

  execute(_commandEncoder: GPUCommandEncoder, _res: ResourceManager) {
    // TODO: Begin render pass and draw
    /*
    const renderPassDescriptor: GPURenderPassDescriptor = {
      colorAttachments: [
        {
          view: res.renderTargetView, // Or a dedicated G-Buffer
          loadOp: 'clear',
          storeOp: 'store',
          clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
        },
      ],
    };
    const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
    passEncoder.setPipeline(this.pipeline);
    passEncoder.draw(...);
    passEncoder.end();
    */
  }
}
