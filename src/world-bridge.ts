import type { MainMessage, WorkerMessage } from "./worker/protocol";
import Worker from "./worker/wasm-worker?worker";

export class WorldBridge {
  private worker: Worker;
  private resolveReady: (() => void) | null = null;

  // Local Data Cache (Simulating simple memory access)
  private _vertices = new Float32Array(0);
  private _normals = new Float32Array(0);
  private _uvs = new Float32Array(0);
  private _mesh_topology = new Uint32Array(0); // Renamed
  private _tlas = new Float32Array(0);
  private _blas = new Float32Array(0);
  private _instances = new Float32Array(0);
  private _cameraData = new Float32Array(24);
  private _textureCount = 0;
  private _textures: Uint8Array[] = [];
  private _animations: string[] = [];
  public hasNewData = false;
  public hasNewGeometry = false;
  public pendingUpdate = false;

  // Scene load promise
  private resolveSceneLoad: (() => void) | null = null;

  constructor() {
    this.worker = new Worker();
    this.worker.onmessage = this.handleMessage.bind(this);
  }

  async initWasm() {
    return new Promise<void>((resolve) => {
      this.resolveReady = resolve;
      this.worker.postMessage({ type: "INIT" } as WorkerMessage);
    });
  }

  private handleMessage(e: MessageEvent<MainMessage>) {
    const msg = e.data;
    switch (msg.type) {
      case "READY":
        console.log("Main: Worker Ready");
        this.resolveReady?.();
        break;
      case "SCENE_LOADED":
        this._vertices = msg.vertices as any;
        this._normals = msg.normals as any;
        this._uvs = msg.uvs as any;
        this._mesh_topology = msg.mesh_topology as any;
        this._tlas = msg.tlas as any;
        this._blas = msg.blas as any;
        this._instances = msg.instances as any;
        this._cameraData = msg.camera as any;
        this._textureCount = msg.textureCount;
        this._textures = msg.textures || [];
        this._animations = msg.animations || [];
        this.hasNewData = true; // Ensure initial data is marked as new
        this.hasNewGeometry = true;
        this.resolveSceneLoad?.();
        break;
      // ... (rest of update result)

      case "UPDATE_RESULT":
        this._tlas = msg.tlas as any;
        this._blas = msg.blas as any;
        this._instances = msg.instances as any;
        this._cameraData = msg.camera as any;
        if (msg.vertices) {
          this._vertices = msg.vertices as any;
          this.hasNewGeometry = true;
        }
        if (msg.normals) this._normals = msg.normals as any;
        if (msg.uvs) this._uvs = msg.uvs as any;
        if (msg.mesh_topology) this._mesh_topology = msg.mesh_topology as any;
        this.hasNewData = true;
        this.pendingUpdate = false;
        this.updateResolvers.forEach((r) => r());
        this.updateResolvers = [];
        break;
    }
  }

  getAnimationList(): string[] {
    return this._animations;
  }

  getTexture(index: number): Uint8Array | null {
    if (index >= 0 && index < this._textures.length) {
      return this._textures[index];
    }
    return null;
  }

  loadScene(
    sceneName: string,
    objSource?: string,
    glbData?: Uint8Array
  ): Promise<void> {
    // Reset camera cache so we force an update for the new world
    this.lastWidth = -1;
    this.lastHeight = -1;

    return new Promise((resolve) => {
      this.resolveSceneLoad = resolve;
      this.worker.postMessage(
        {
          type: "LOAD_SCENE",
          sceneName,
          objSource,
          glbData,
        } as WorkerMessage,
        glbData ? [glbData.buffer] : []
      );
    });
  }

  private updateResolvers: (() => void)[] = [];

  waitForNextUpdate(): Promise<void> {
    return new Promise((resolve) => {
      this.updateResolvers.push(resolve);
    });
  }

  update(time: number) {
    if (this.pendingUpdate) return;
    this.pendingUpdate = true;
    this.worker.postMessage({ type: "UPDATE", time } as WorkerMessage);
  }

  private lastWidth = -1;
  private lastHeight = -1;

  updateCamera(width: number, height: number) {
    if (this.lastWidth === width && this.lastHeight === height) return;
    this.lastWidth = width;
    this.lastHeight = height;

    this.worker.postMessage({
      type: "UPDATE_CAMERA",
      width,
      height,
    } as WorkerMessage);
  }

  loadAnimation(data: Uint8Array) {
    this.worker.postMessage({ type: "LOAD_ANIMATION", data } as WorkerMessage, [
      data.buffer,
    ]);
  }

  setAnimation(index: number) {
    this.worker.postMessage({ type: "SET_ANIMATION", index } as WorkerMessage);
  }

  // --- Data Accessors ---
  get vertices() {
    return this._vertices;
  }
  get normals() {
    return this._normals;
  }
  get uvs() {
    return this._uvs;
  }
  get mesh_topology() {
    return this._mesh_topology;
  }
  get tlas() {
    return this._tlas;
  }
  get blas() {
    return this._blas;
  }
  get instances() {
    return this._instances;
  }
  get cameraData() {
    return this._cameraData;
  }
  get textureCount() {
    return this._textureCount;
  }
  get hasWorld() {
    return this._vertices.length > 0;
  }

  printStats() {
    console.log(
      `Scene Stats (Worker Proxy): V=${this.vertices.length / 4}, Topo=${
        this.mesh_topology.length / 12
      }, I=${this.instances.length / 16}, TLAS=${this.tlas.length / 8}, BLAS=${
        this.blas.length / 8
      }, Anim=${this._animations.length}`
    );
  }
}
