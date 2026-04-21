import { WebGPUContext } from "../WebGPUContext";
import { ResourceManager } from "../ResourceManager";
import postprocessCodeRaw from "../../shaders/PostProcess.wgsl?raw";

export class PostProcessPass {
  pipeline!: GPUComputePipeline;
  bindGroupLayout!: GPUBindGroupLayout;
  bindGroup!: GPUBindGroup;

  ctx: WebGPUContext;

  constructor(ctx: WebGPUContext) {
    this.ctx = ctx;
  }

  buildPipeline() {
    const postprocessModule = this.ctx.device.createShaderModule({
      label: "PostProcess",
      code: postprocessCodeRaw,
    });

    this.pipeline = this.ctx.device.createComputePipeline({
      label: "PostProcess Pipeline",
      layout: "auto",
      compute: { module: postprocessModule, entryPoint: "main" },
    });

    this.bindGroupLayout = this.pipeline.getBindGroupLayout(0);
  }

  updateBindGroup(res: ResourceManager) {
    if (
      !res.renderTargetView ||
      !res.accumulateBuffer ||
      !res.sceneUniformBuffer
    ) {
      return;
    }

    this.bindGroup = this.ctx.device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: res.renderTargetView },
        { binding: 1, resource: { buffer: res.accumulateBuffer } },
        { binding: 2, resource: { buffer: res.sceneUniformBuffer } },
        {
          binding: 3,
          resource: res.historyTextureViews[1 - res.historyIndex],
        }, // Previous frame (Read)
        { binding: 4, resource: res.sampler },
        {
          binding: 5,
          resource: res.historyTextureViews[res.historyIndex],
        }, // Current frame (Write)
      ],
    });
  }

  execute(commandEncoder: GPUCommandEncoder) {
    if (!this.bindGroup) return;

    const dispatchX = Math.ceil(this.ctx.canvas.width / 8);
    const dispatchY = Math.ceil(this.ctx.canvas.height / 8);

    const postPass = commandEncoder.beginComputePass();
    postPass.setPipeline(this.pipeline);
    postPass.setBindGroup(0, this.bindGroup);
    postPass.dispatchWorkgroups(dispatchX, dispatchY);
    postPass.end();
  }
}
