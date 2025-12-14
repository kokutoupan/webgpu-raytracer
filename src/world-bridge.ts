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

  getAnimationList(): string[] {
    if (!this.world) return [];
    const count = this.world.get_animation_count();
    const list: string[] = [];
    for (let i = 0; i < count; i++) {
      list.push(this.world.get_animation_name(i));
    }
    return list;
  }

  setAnimation(index: number) {
    this.world?.set_animation(index);
  }

  // --- Data Accessors ---
  private getF32(ptr: number, len: number) {
    return new Float32Array(this.wasmMemory!.buffer, ptr, len);
  }
  private getU32(ptr: number, len: number) {
    return new Uint32Array(this.wasmMemory!.buffer, ptr, len);
  }

  get vertices() { return this.getF32(this.world!.vertices_ptr(), this.world!.vertices_len()); }
  get normals() { return this.getF32(this.world!.normals_ptr(), this.world!.normals_len()); }
  get uvs() { return this.getF32(this.world!.uvs_ptr(), this.world!.uvs_len()); } // ★追加
  get indices() { return this.getU32(this.world!.indices_ptr(), this.world!.indices_len()); }
  get attributes() { return this.getF32(this.world!.attributes_ptr(), this.world!.attributes_len()); }
  get tlas() { return this.getF32(this.world!.tlas_ptr(), this.world!.tlas_len()); }
  get blas() { return this.getF32(this.world!.blas_ptr(), this.world!.blas_len()); }
  get instances() { return this.getF32(this.world!.instances_ptr(), this.world!.instances_len()); }
  get cameraData() { return this.getF32(this.world!.camera_ptr(), 24); }

  get textureCount() { return this.world?.get_texture_count() || 0; }

  getTexture(index: number): Uint8Array | null {
    if (!this.world) return null;
    const ptr = this.world.get_texture_ptr(index);
    const size = this.world.get_texture_size(index);
    if (!ptr || size === 0) return null;
    return new Uint8Array(this.wasmMemory!.buffer, ptr, size).slice();
  }

  get hasWorld() { return !!this.world; }

  printStats() {
    if (!this.world) return;
    console.log(`Scene Stats: V=${this.vertices.length / 4}, Tri=${this.indices.length / 3}, BLAS=${this.blas.length / 8}, TLAS=${this.tlas.length / 8}`);
  }
}
