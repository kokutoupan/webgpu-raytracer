// src/world-bridge.ts
import init, { World } from '../rust-shader-tools/pkg/rust_shader_tools';

export class WorldBridge {
  private world: World | null = null;
  private wasmMemory: WebAssembly.Memory | null = null;

  async initWasm() {
    const wasmInstance = await init();
    this.wasmMemory = wasmInstance.memory;
    console.log("Wasm initialized");
  }

  loadScene(sceneName: string, objSource?: string, glbData?: Uint8Array) {
    if (this.world) this.world.free();
    this.world = new World(sceneName, objSource, glbData);
  }

  update(time: number) {
    this.world?.update(time);
  }

  updateCamera(width: number, height: number) {
    this.world?.update_camera(width, height);
  }

  loadAnimation(data: Uint8Array) {
    this.world?.load_animation_glb(data);
  }

  // --- Data Accessors ---
  // ポインタから直接TypedArrayのビューを返す
  private getF32(ptr: number, len: number) {
    return new Float32Array(this.wasmMemory!.buffer, ptr, len);
  }
  private getU32(ptr: number, len: number) {
    return new Uint32Array(this.wasmMemory!.buffer, ptr, len);
  }

  get vertices() { return this.getF32(this.world!.vertices_ptr(), this.world!.vertices_len()); }
  get normals() { return this.getF32(this.world!.normals_ptr(), this.world!.normals_len()); }
  get indices() { return this.getU32(this.world!.indices_ptr(), this.world!.indices_len()); }
  get attributes() { return this.getF32(this.world!.attributes_ptr(), this.world!.attributes_len()); }
  get tlas() { return this.getF32(this.world!.tlas_ptr(), this.world!.tlas_len()); }
  get blas() { return this.getF32(this.world!.blas_ptr(), this.world!.blas_len()); }
  get instances() { return this.getF32(this.world!.instances_ptr(), this.world!.instances_len()); }
  get cameraData() { return this.getF32(this.world!.camera_ptr(), 24); } // Camera struct size

  get hasWorld() { return !!this.world; }

  // デバッグ用
  printStats() {
    if (!this.world) return;
    console.log(`Scene Stats: V=${this.vertices.length / 4}, Tri=${this.indices.length / 3}, BLAS=${this.blas.length / 8}, TLAS=${this.tlas.length / 8}`);
  }
}
