import { WebGPUContext } from "./WebGPUContext";
import { ResourceManager } from "./ResourceManager";
import { RaytracePass } from "./passes/RaytracePass";
import { PostProcessPass } from "./passes/PostProcessPass";
import { RasterizerPass } from "./passes/RasterizerPass";

export class WebGPURenderer {
  private ctx: WebGPUContext;
  private res: ResourceManager;

  private raytracePass: RaytracePass;
  private postProcessPass: PostProcessPass;
  private rasterizerPass: RasterizerPass;

  private totalFrames = 0;

  constructor(canvas: HTMLCanvasElement) {
    this.ctx = new WebGPUContext(canvas);
    this.res = new ResourceManager(this.ctx);
    this.raytracePass = new RaytracePass(this.ctx);
    this.postProcessPass = new PostProcessPass(this.ctx);
    this.rasterizerPass = new RasterizerPass(this.ctx);
  }

  get device(): GPUDevice {
    return this.ctx.device;
  }

  async init() {
    await this.ctx.init();
    this.res.init();
  }

  buildPipeline(depth: number, spp: number) {
    this.raytracePass.buildPipeline(depth, spp);
    this.postProcessPass.buildPipeline();
    this.rasterizerPass.buildPipeline();
    this.recreateBindGroup();
  }

  updateScreenSize(width: number, height: number) {
    this.ctx.canvas.width = width;
    this.ctx.canvas.height = height;
    this.res.updateScreenSize(width, height);
  }

  resetAccumulation() {
    this.res.resetAccumulation();
  }

  async loadTexturesFromWorld(bridge: any) {
    await this.res.loadTexturesFromWorld(bridge);
  }

  updateBuffer(
    type: "topology" | "instance" | "lights" | "draw_commands",
    data: Uint32Array | Float32Array
  ): boolean {
    return this.res.updateBuffer(type, data);
  }

  updateCombinedGeometry(
    v: Float32Array,
    n: Float32Array,
    uv: Float32Array
  ): boolean {
    return this.res.updateCombinedGeometry(v, n, uv);
  }

  updateCombinedBVH(tlas: Float32Array, blas: Float32Array): boolean {
    return this.res.updateCombinedBVH(tlas, blas);
  }

  updateSceneUniforms(
    cameraData: Float32Array,
    frameCount: number,
    lightCount: number
  ) {
    this.res.updateSceneUniforms(cameraData, frameCount, lightCount);
  }

  recreateBindGroup() {
    this.raytracePass.updateBindGroup(this.res);
    this.postProcessPass.updateBindGroup(this.res);
    this.rasterizerPass.updateBindGroup(this.res);
  }

  compute(frameCount: number) {
    this.totalFrames++;

    this.res.updateFrameUniforms(frameCount, this.totalFrames);

    const commandEncoder = this.ctx.device.createCommandEncoder();

    // 1. Rasterizer Pass
    this.rasterizerPass.execute(commandEncoder, this.res);

    // 2. Raytrace Pass
    this.raytracePass.execute(commandEncoder);

    this.ctx.device.queue.submit([commandEncoder.finish()]);
  }

  present() {
    const commandEncoder = this.ctx.device.createCommandEncoder();

    this.postProcessPass.execute(commandEncoder);

    try {
      const currentTexture = this.ctx.context.getCurrentTexture();
      commandEncoder.copyTextureToTexture(
        { texture: this.res.renderTarget },
        { texture: currentTexture },
        {
          width: this.ctx.canvas.width,
          height: this.ctx.canvas.height,
          depthOrArrayLayers: 1,
        }
      );
    } catch (e) {
      console.warn("Skipping present(): Swapchain unavailable or invalid.", e);
    }

    this.ctx.device.queue.submit([commandEncoder.finish()]);

    // Swap history index for TAA
    this.res.historyIndex = 1 - this.res.historyIndex;
    this.recreateBindGroup(); // To update history binding
  }

  async captureFrame(): Promise<{
    data: ArrayBufferLike;
    width: number;
    height: number;
  }> {
    return this.ctx.captureFrame(this.res.renderTarget);
  }
}
