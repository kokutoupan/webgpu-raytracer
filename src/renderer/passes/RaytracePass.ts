import { WebGPUContext } from "../WebGPUContext";
import { ResourceManager } from "../ResourceManager";
import initialSamplingCodeRaw from "../../shaders/InitialSampling.wgsl?raw";
import finalShadingCodeRaw from "../../shaders/FinalShading.wgsl?raw";

export class RaytracePass {
  initialSamplingPipeline!: GPUComputePipeline;
  finalShadingPipeline!: GPUComputePipeline;
  initialBindGroupLayout!: GPUBindGroupLayout;
  finalBindGroupLayout!: GPUBindGroupLayout;
  initialBindGroup!: GPUBindGroup;
  finalBindGroup!: GPUBindGroup;

  ctx: WebGPUContext;

  constructor(ctx: WebGPUContext) {
    this.ctx = ctx;
  }

  buildPipeline(depth: number) {
    const initialModule = this.ctx.device.createShaderModule({
      label: "Initial Sampling Shader",
      code: initialSamplingCodeRaw,
    });

    const finalModule = this.ctx.device.createShaderModule({
      label: "Final Shading Shader",
      code: finalShadingCodeRaw,
    });

    this.initialSamplingPipeline = this.ctx.device.createComputePipeline({
      label: "Initial Sampling Pipeline",
      layout: "auto",
      compute: {
        module: initialModule,
        entryPoint: "initial_sampling",
        constants: {
          MAX_DEPTH: depth,
        },
      },
    });

    this.finalShadingPipeline = this.ctx.device.createComputePipeline({
      label: "Final Shading Pipeline",
      layout: "auto",
      compute: {
        module: finalModule,
        entryPoint: "final_shading",
        constants: {
          MAX_DEPTH: depth,
        },
      },
    });
    this.initialBindGroupLayout = this.initialSamplingPipeline.getBindGroupLayout(0);
    this.finalBindGroupLayout = this.finalShadingPipeline.getBindGroupLayout(0);
  }

  updateBindGroup(res: ResourceManager) {
    if (
      !res.accumulateBuffer ||
      !res.samplesBuffer ||
      !res.geometryBuffer ||
      !res.nodesBuffer ||
      !res.sceneUniformBuffer ||
      !res.lightsBuffer
    ) {
      return;
    }

    const commonEntries = [
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
      { binding: 5, resource: { buffer: res.nodesBuffer } },
      { binding: 6, resource: { buffer: res.instanceBuffer } },
      {
        binding: 7,
        resource: res.texture.createView({ dimension: "2d-array" }),
      },
      { binding: 8, resource: res.sampler },
      { binding: 9, resource: { buffer: res.lightsBuffer } },
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
      { binding: 13, resource: res.renderTargetView },
      { binding: 14, resource: res.gBufferNormalView },
      { binding: 15, resource: res.depthTextureView },
      { binding: 16, resource: { buffer: res.samplesBuffer } },
    ];

    this.initialBindGroup = this.ctx.device.createBindGroup({
      layout: this.initialBindGroupLayout,
      entries: commonEntries,
    });

    this.finalBindGroup = this.ctx.device.createBindGroup({
      layout: this.finalBindGroupLayout,
      entries: [
        { binding: 1, resource: { buffer: res.accumulateBuffer } },
        ...commonEntries.filter(e => e.binding !== 11), // final_shading doesn't use geometry_norm
      ],
    });
  }

  execute(commandEncoder: GPUCommandEncoder) {
    if (!this.initialBindGroup || !this.finalBindGroup) return;

    const dispatchX = Math.ceil(this.ctx.canvas.width / 8);
    const dispatchY = Math.ceil(this.ctx.canvas.height / 8);

    // Initial Sampling Pass
    const initialPass = commandEncoder.beginComputePass();
    initialPass.setPipeline(this.initialSamplingPipeline);
    initialPass.setBindGroup(0, this.initialBindGroup);
    initialPass.dispatchWorkgroups(dispatchX, dispatchY);
    initialPass.end();

    // Final Shading Pass
    const finalPass = commandEncoder.beginComputePass();
    finalPass.setPipeline(this.finalShadingPipeline);
    finalPass.setBindGroup(0, this.finalBindGroup);
    finalPass.dispatchWorkgroups(dispatchX, dispatchY);
    finalPass.end();
  }
}
