export type WorkerMessage =
  | { type: "INIT" }
  | {
      type: "LOAD_SCENE";
      sceneName: string;
      objSource?: string;
      glbData?: Uint8Array;
    }
  | { type: "UPDATE"; time: number }
  | { type: "UPDATE_CAMERA"; width: number; height: number }
  | { type: "SET_ANIMATION"; index: number }
  | { type: "LOAD_ANIMATION"; data: Uint8Array };

export type MainMessage =
  | { type: "READY" }
  | {
      type: "SCENE_LOADED";
      vertices: Float32Array;
      normals: Float32Array;
      uvs: Float32Array;
      mesh_topology: Uint32Array;
      lights: Uint32Array; // Added
      tlas: Float32Array;
      blas: Float32Array;
      instances: Float32Array;
      camera: Float32Array;
      textureCount: number;
      textures: Uint8Array[];
      animations: string[];
    }
  | {
      type: "UPDATE_RESULT";
      tlas: Float32Array;
      blas: Float32Array;
      instances: Float32Array;
      lights: Uint32Array; // Added
      camera: Float32Array;
      vertices?: Float32Array;
      normals?: Float32Array;
      uvs?: Float32Array;
      mesh_topology?: Uint32Array;
    }
  | { type: "TEXTURE_DATA"; index: number; data: Uint8Array };
