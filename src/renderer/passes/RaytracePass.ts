import { WebGPUContext } from "../WebGPUContext";
import { ResourceManager } from "../ResourceManager";
import raytracerCodeRaw from "../../shaders/Raytracer.wgsl?raw";

export class RaytracePass {
  pipeline!: GPUComputePipeline;
  bindGroupLayout!: GPUBindGroupLayout;
  bindGroup!: GPUBindGroup;

  ctx: WebGPUContext;

  constructor(ctx: WebGPUContext) {
    this.ctx = ctx;
  }

  buildPipeline(depth: number, spp: number) {
    const raytraceModule = this.ctx.device.createShaderModule({
      label: "RayTracing",
      code: raytracerCodeRaw,
    });

    this.pipeline = this.ctx.device.createComputePipeline({
      label: "Main Pipeline",
      layout: "auto",
      compute: {
        module: raytraceModule,
        entryPoint: "main",
        constants: {
          MAX_DEPTH: depth,
          SPP: spp,
        },
      },
    });
    this.bindGroupLayout = this.pipeline.getBindGroupLayout(0);
  }

  updateBindGroup(res: ResourceManager) {
    if (
      !res.accumulateBuffer ||
      !res.geometryBuffer ||
      !res.nodesBuffer ||
      !res.sceneUniformBuffer ||
      !res.lightsBuffer
    ) {
      return;
    }

    this.bindGroup = this.ctx.device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 1, resource: { buffer: res.accumulateBuffer } },
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
      ],
    });
  }

  execute(commandEncoder: GPUCommandEncoder) {
    if (!this.bindGroup) return;

    const dispatchX = Math.ceil(this.ctx.canvas.width / 8);
    const dispatchY = Math.ceil(this.ctx.canvas.height / 8);

    const rayPass = commandEncoder.beginComputePass();
    rayPass.setPipeline(this.pipeline);
    rayPass.setBindGroup(0, this.bindGroup);
    rayPass.dispatchWorkgroups(dispatchX, dispatchY);
    rayPass.end();
  }
}
