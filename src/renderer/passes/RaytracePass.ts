import { WebGPUContext } from "../WebGPUContext";
import { ResourceManager } from "../ResourceManager";
import initialSamplingCodeRaw from "../../shaders/InitialSampling.wgsl?raw";
import temporalReuseCodeRaw from "../../shaders/TemporalReuse.wgsl?raw";
import spatialReuseCodeRaw from "../../shaders/SpatialReuse.wgsl?raw";
import finalShadingCodeRaw from "../../shaders/FinalShading.wgsl?raw";

export class RaytracePass {
  initialSamplingPipeline!: GPUComputePipeline;
  temporalReusePipeline!: GPUComputePipeline;
  spatialReusePipeline!: GPUComputePipeline;
  finalShadingPipeline!: GPUComputePipeline;
  initialBindGroupLayout!: GPUBindGroupLayout;
  temporalBindGroupLayout!: GPUBindGroupLayout;
  spatialBindGroupLayout!: GPUBindGroupLayout;
  finalBindGroupLayout!: GPUBindGroupLayout;
  initialBindGroups: GPUBindGroup[] = [];
  temporalBindGroups: GPUBindGroup[] = [];
  spatialBindGroups: GPUBindGroup[] = [];
  finalBindGroups: GPUBindGroup[] = [];

  ctx: WebGPUContext;

  constructor(ctx: WebGPUContext) {
    this.ctx = ctx;
  }

  buildPipeline(depth: number) {
    const initialModule = this.ctx.device.createShaderModule({
      label: "Initial Sampling Shader",
      code: initialSamplingCodeRaw,
    });

    const temporalModule = this.ctx.device.createShaderModule({
      label: "Temporal Reuse Shader",
      code: temporalReuseCodeRaw,
    });

    const spatialModule = this.ctx.device.createShaderModule({
      label: "Spatial Reuse Shader",
      code: spatialReuseCodeRaw,
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
        constants: { MAX_DEPTH: depth },
      },
    });

    this.temporalReusePipeline = this.ctx.device.createComputePipeline({
      label: "Temporal Reuse Pipeline",
      layout: "auto",
      compute: {
        module: temporalModule,
        entryPoint: "temporal_reuse",
        constants: { MAX_DEPTH: depth },
      },
    });

    this.spatialReusePipeline = this.ctx.device.createComputePipeline({
      label: "Spatial Reuse Pipeline",
      layout: "auto",
      compute: {
        module: spatialModule,
        entryPoint: "spatial_reuse",
      },
    });

    this.finalShadingPipeline = this.ctx.device.createComputePipeline({
      label: "Final Shading Pipeline",
      layout: "auto",
      compute: {
        module: finalModule,
        entryPoint: "final_shading",
        constants: { MAX_DEPTH: depth },
      },
    });
    this.initialBindGroupLayout = this.initialSamplingPipeline.getBindGroupLayout(0);
    this.temporalBindGroupLayout = this.temporalReusePipeline.getBindGroupLayout(0);
    this.spatialBindGroupLayout = this.spatialReusePipeline.getBindGroupLayout(0);
    this.finalBindGroupLayout = this.finalShadingPipeline.getBindGroupLayout(0);
  }

  updateBindGroup(res: ResourceManager) {
    if (
      !res.accumulateBuffer ||
      !res.reservoirsBufferA ||
      !res.reservoirsBufferB ||
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
    ];

    for (let i = 0; i < 2; i++) {
      // i=0: curr=A, prev=B (Odd frame)
      // i=1: curr=B, prev=A (Even frame)
      const currBuffer = i === 0 ? res.reservoirsBufferA : res.reservoirsBufferB;
      const prevBuffer = i === 0 ? res.reservoirsBufferB : res.reservoirsBufferA;

      const entries = [
        ...commonEntries,
        { binding: 16, resource: { buffer: currBuffer } },
        { binding: 17, resource: { buffer: prevBuffer } },
      ];

      this.initialBindGroups[i] = this.ctx.device.createBindGroup({
        layout: this.initialBindGroupLayout,
        entries: entries.filter(e => e.binding !== 17),
      });

      this.temporalBindGroups[i] = this.ctx.device.createBindGroup({
        layout: this.temporalBindGroupLayout,
        entries: entries.filter(e => 
          e.binding === 2 || 
          e.binding === 16 || 
          e.binding === 17 || 
          e.binding === 4 || 
          e.binding === 13 || 
          e.binding === 14 || 
          e.binding === 15
        ),
      });

      this.spatialBindGroups[i] = this.ctx.device.createBindGroup({
        layout: this.spatialBindGroupLayout,
        entries: [
          ...commonEntries,
          { binding: 16, resource: { buffer: currBuffer } }, // Input from Temporal
          { binding: 17, resource: { buffer: res.spatialReservoirsBuffer } }, // Output
        ].filter(e => 
          e.binding === 2 || 
          e.binding === 16 || 
          e.binding === 17 || 
          e.binding === 14 || 
          e.binding === 15 ||
          e.binding === 4
        ),
      });

      this.finalBindGroups[i] = this.ctx.device.createBindGroup({
        layout: this.finalBindGroupLayout,
        entries: [
          { binding: 1, resource: { buffer: res.accumulateBuffer } },
          ...commonEntries,
          { binding: 16, resource: { buffer: res.spatialReservoirsBuffer } }, // Read from Spatial
        ].filter(e => e.binding !== 11 && e.binding !== 17),
      });
    }
  }

  execute(commandEncoder: GPUCommandEncoder, frameCount: number) {
    // 偶数フレーム (frame_count % 2 == 0) -> i=1 (Buffer B)
    // 奇数フレーム (frame_count % 2 == 1) -> i=0 (Buffer A)
    const i = frameCount % 2 === 0 ? 1 : 0;

    if (!this.initialBindGroups[i] || !this.temporalBindGroups[i] || !this.spatialBindGroups[i] || !this.finalBindGroups[i]) return;

    const dispatchX = Math.ceil(this.ctx.canvas.width / 8);
    const dispatchY = Math.ceil(this.ctx.canvas.height / 8);

    // Initial Sampling Pass
    const initialPass = commandEncoder.beginComputePass();
    initialPass.setPipeline(this.initialSamplingPipeline);
    initialPass.setBindGroup(0, this.initialBindGroups[i]);
    initialPass.dispatchWorkgroups(dispatchX, dispatchY);
    initialPass.end();
 
    // Temporal Reuse Pass
    const temporalPass = commandEncoder.beginComputePass();
    temporalPass.setPipeline(this.temporalReusePipeline);
    temporalPass.setBindGroup(0, this.temporalBindGroups[i]);
    temporalPass.dispatchWorkgroups(dispatchX, dispatchY);
    temporalPass.end();
 
    // Spatial Reuse Pass
    const spatialPass = commandEncoder.beginComputePass();
    spatialPass.setPipeline(this.spatialReusePipeline);
    spatialPass.setBindGroup(0, this.spatialBindGroups[i]);
    spatialPass.dispatchWorkgroups(dispatchX, dispatchY);
    spatialPass.end();
 
    // Final Shading Pass
    const finalPass = commandEncoder.beginComputePass();
    finalPass.setPipeline(this.finalShadingPipeline);
    finalPass.setBindGroup(0, this.finalBindGroups[i]);
    finalPass.dispatchWorkgroups(dispatchX, dispatchY);
    finalPass.end();
  }
}
