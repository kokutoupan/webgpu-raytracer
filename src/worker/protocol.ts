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
      indices: Uint32Array;
      attributes: Float32Array;
      tlas: Float32Array;
      blas: Float32Array;
      instances: Float32Array;
      camera: Float32Array;
      textureCount: number;
      textures: Uint8Array[]; // Added
      animations: string[]; // Added
    }
  | {
      type: "UPDATE_RESULT";
      tlas: Float32Array;
      blas: Float32Array;
      instances: Float32Array;
      camera: Float32Array;
      vertices?: Float32Array;
      normals?: Float32Array;
      uvs?: Float32Array;
      indices?: Uint32Array;
      attributes?: Float32Array; // Attributes might animate?
    }
  | { type: "TEXTURE_DATA"; index: number; data: Uint8Array };
