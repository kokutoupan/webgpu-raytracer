import init, { World } from "../../rust-shader-tools/pkg/rust_shader_tools";
import type { MainMessage, WorkerMessage } from "./protocol";

let world: World | null = null;
let wasmMemory: WebAssembly.Memory | null = null;

// Helpers to extract data from WASM memory
// IMPORTANT: We must slice() to create a copy, because the underlying WASM memory
// cannot be detached or transferred if it's SharedArrayBuffer (or simply to avoid race conditions).
// Even if it's normal ArrayBuffer, WASM memory grows and invalidates views.
// For transferables to work and be safe, copies are best here.

const getF32 = (ptr: number, len: number) => {
  return new Float32Array(wasmMemory!.buffer, ptr, len).slice();
};

const getU32 = (ptr: number, len: number) => {
  return new Uint32Array(wasmMemory!.buffer, ptr, len).slice();
};

const sendSceneData = () => {
  if (!world) return;

  // Full Geometry extraction
  const vertices = getF32(world.vertices_ptr(), world.vertices_len());
  const normals = getF32(world.normals_ptr(), world.normals_len());
  const uvs = getF32(world.uvs_ptr(), world.uvs_len());
  const indices = getU32(world.indices_ptr(), world.indices_len());
  const attributes = getF32(world.attributes_ptr(), world.attributes_len());

  // BVH & Instances
  const tlas = getF32(world.tlas_ptr(), world.tlas_len());
  const blas = getF32(world.blas_ptr(), world.blas_len());
  const instances = getF32(world.instances_ptr(), world.instances_len());
  const camera = getF32(world.camera_ptr(), 24);

  // Textures
  const textureCount = world.get_texture_count();
  const textures: Uint8Array[] = [];
  const transerableTextures: ArrayBuffer[] = [];
  for (let i = 0; i < textureCount; i++) {
    const ptr = world.get_texture_ptr(i);
    const size = world.get_texture_size(i);
    // Slice is critical here
    const texData = new Uint8Array(wasmMemory!.buffer, ptr, size).slice();
    textures.push(texData);
    transerableTextures.push(texData.buffer);
  }

  // Animations
  const animCount = world.get_animation_count();
  const animations: string[] = [];
  for (let i = 0; i < animCount; i++) {
    animations.push(world.get_animation_name(i));
  }

  const msg: MainMessage = {
    type: "SCENE_LOADED",
    vertices,
    normals,
    uvs,
    indices,
    attributes,
    tlas,
    blas,
    instances,
    camera,
    textureCount,
    textures,
    animations,
  };

  // Send with transferables to avoid copying the copy
  self.postMessage(msg, [
    vertices.buffer,
    normals.buffer,
    uvs.buffer,
    indices.buffer,
    attributes.buffer,
    tlas.buffer,
    blas.buffer,
    instances.buffer,
    camera.buffer,
    ...transerableTextures,
  ] as any);
};

const sendUpdateResult = (includeGeometry = true) => {
  if (!world) return;

  const tlas = getF32(world.tlas_ptr(), world.tlas_len());
  const blas = getF32(world.blas_ptr(), world.blas_len());
  const instances = getF32(world.instances_ptr(), world.instances_len());
  const camera = getF32(world.camera_ptr(), 24);

  let vertices, normals, uvs, indices, attributes;
  const transfers: any[] = [
    tlas.buffer,
    blas.buffer,
    instances.buffer,
    camera.buffer,
  ];

  if (includeGeometry) {
    vertices = getF32(world.vertices_ptr(), world.vertices_len());
    normals = getF32(world.normals_ptr(), world.normals_len());
    uvs = getF32(world.uvs_ptr(), world.uvs_len());
    indices = getU32(world.indices_ptr(), world.indices_len());
    attributes = getF32(world.attributes_ptr(), world.attributes_len());

    transfers.push(
      vertices.buffer,
      normals.buffer,
      uvs.buffer,
      indices.buffer,
      attributes.buffer
    );
  }

  const msg: MainMessage = {
    type: "UPDATE_RESULT",
    tlas,
    blas,
    instances,
    camera,
    vertices,
    normals,
    uvs,
    indices,
    attributes,
  };

  // Force cast to any because in Worker scope postMessage signature is different
  (self as any).postMessage(msg, transfers);
};

self.onmessage = async (e: MessageEvent<WorkerMessage>) => {
  const msg = e.data;

  switch (msg.type) {
    case "INIT":
      const instance = await init();
      wasmMemory = instance.memory;
      // init() returns InitOutput which has memory, but default export init is a function
      // that returns a promise of InitOutput.
      // wait, `import init` usually gives the default function.
      // Let's re-verify usage. `import init from ...` -> `await init()`.
      // The snippet in world-bridge.ts was:
      // const wasmInstance = await init();
      // this.wasmMemory = wasmInstance.memory;

      // So:
      // await init();
      // But we need the instance to get memory if it's not exported separately?
      // Actually `init` promise resolves to the exports.

      console.log("Worker: WASM Initialized");
      self.postMessage({ type: "READY" } as MainMessage);
      break;

    case "LOAD_SCENE":
      if (world) world.free();
      console.log(`Worker: Loading Scene ${msg.sceneName}`);
      world = new World(msg.sceneName, msg.objSource, msg.glbData);
      world.update(0); // Ensure initial state is calculated
      console.log("Worker: Scene Loaded, sending data...");
      sendSceneData();
      break;

    case "UPDATE":
      if (world) {
        world.update(msg.time);
        sendUpdateResult();
      }
      break;

    case "UPDATE_CAMERA":
      if (world) {
        world.update_camera(msg.width, msg.height);
        sendUpdateResult(false); // Only update camera/instances, keep geometry
      }
      break;

    case "SET_ANIMATION":
      world?.set_animation(msg.index);
      break;

    case "LOAD_ANIMATION":
      world?.load_animation_glb(msg.data);
      // We probably need to re-send animation list?
      // For now, let's assume UI handles it or we add a message.
      break;
  }
};
