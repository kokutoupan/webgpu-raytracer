var __defProp = Object.defineProperty;
var __defNormalProp = (obj, key, value) => key in obj ? __defProp(obj, key, { enumerable: true, configurable: true, writable: true, value }) : obj[key] = value;
var __publicField = (obj, key, value) => __defNormalProp(obj, typeof key !== "symbol" ? key + "" : key, value);
(async () => {
  (function() {
    const e = document.createElement("link").relList;
    if (e && e.supports && e.supports("modulepreload")) return;
    for (const r of document.querySelectorAll('link[rel="modulepreload"]')) t(r);
    new MutationObserver((r) => {
      for (const i of r) if (i.type === "childList") for (const s of i.addedNodes) s.tagName === "LINK" && s.rel === "modulepreload" && t(s);
    }).observe(document, {
      childList: true,
      subtree: true
    });
    function n(r) {
      const i = {};
      return r.integrity && (i.integrity = r.integrity), r.referrerPolicy && (i.referrerPolicy = r.referrerPolicy), r.crossOrigin === "use-credentials" ? i.credentials = "include" : r.crossOrigin === "anonymous" ? i.credentials = "omit" : i.credentials = "same-origin", i;
    }
    function t(r) {
      if (r.ep) return;
      r.ep = true;
      const i = n(r);
      fetch(r.href, i);
    }
  })();
  const I = "modulepreload", G = function(o) {
    return "/webgpu-raytracer/" + o;
  }, M = {}, U = function(e, n, t) {
    let r = Promise.resolve();
    if (n && n.length > 0) {
      let l = function(d) {
        return Promise.all(d.map((v) => Promise.resolve(v).then((p) => ({
          status: "fulfilled",
          value: p
        }), (p) => ({
          status: "rejected",
          reason: p
        }))));
      };
      var s = l;
      document.getElementsByTagName("link");
      const a = document.querySelector("meta[property=csp-nonce]"), c = (a == null ? void 0 : a.nonce) || (a == null ? void 0 : a.getAttribute("nonce"));
      r = l(n.map((d) => {
        if (d = G(d), d in M) return;
        M[d] = true;
        const v = d.endsWith(".css"), p = v ? '[rel="stylesheet"]' : "";
        if (document.querySelector(`link[href="${d}"]${p}`)) return;
        const m = document.createElement("link");
        if (m.rel = v ? "stylesheet" : I, v || (m.as = "script"), m.crossOrigin = "", m.href = d, c && m.setAttribute("nonce", c), document.head.appendChild(m), v) return new Promise((w, P) => {
          m.addEventListener("load", w), m.addEventListener("error", () => P(new Error(`Unable to preload CSS for ${d}`)));
        });
      }));
    }
    function i(a) {
      const c = new Event("vite:preloadError", {
        cancelable: true
      });
      if (c.payload = a, window.dispatchEvent(c), !c.defaultPrevented) throw a;
    }
    return r.then((a) => {
      for (const c of a || []) c.status === "rejected" && i(c.reason);
      return e().catch(i);
    });
  };
  class H {
    constructor(e) {
      __publicField(this, "device");
      __publicField(this, "context");
      __publicField(this, "canvas");
      __publicField(this, "readbackBuffer", null);
      __publicField(this, "readbackBufferSize", 0);
      __publicField(this, "readbackResultBuffer", null);
      this.canvas = e;
    }
    async init() {
      if (!navigator.gpu) throw new Error("WebGPU not supported.");
      const e = await navigator.gpu.requestAdapter({
        powerPreference: "high-performance"
      });
      if (!e) throw new Error("No adapter");
      console.log("Max Storage Buffers Per Shader Stage:", e.limits.maxStorageBuffersPerShaderStage), this.device = await e.requestDevice({
        requiredLimits: {
          maxStorageBuffersPerShaderStage: 10
        }
      }), this.context = this.canvas.getContext("webgpu"), this.context.configure({
        device: this.device,
        format: "rgba8unorm",
        usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT
      });
    }
    async captureFrame(e) {
      if (!e) throw new Error("No render target");
      const n = this.canvas.width, t = this.canvas.height, i = n * 4, s = 256, a = Math.ceil(i / s) * s, c = a * t;
      (!this.readbackBuffer || this.readbackBufferSize < c) && (this.readbackBuffer && this.readbackBuffer.destroy(), this.readbackBuffer = this.device.createBuffer({
        size: c,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
      }), this.readbackBufferSize = c);
      const l = this.device.createCommandEncoder();
      l.copyTextureToBuffer({
        texture: e
      }, {
        buffer: this.readbackBuffer,
        bytesPerRow: a,
        rowsPerImage: t
      }, {
        width: n,
        height: t,
        depthOrArrayLayers: 1
      }), this.device.queue.submit([
        l.finish()
      ]), await this.device.queue.onSubmittedWorkDone(), await this.readbackBuffer.mapAsync(GPUMapMode.READ);
      const d = new Uint8Array(this.readbackBuffer.getMappedRange()), v = n * t * 4;
      (!this.readbackResultBuffer || this.readbackResultBuffer.byteLength !== v) && (this.readbackResultBuffer = new Uint8Array(v));
      const p = this.readbackResultBuffer;
      if (a === i) p.set(d.subarray(0, v));
      else for (let m = 0; m < t; m++) {
        const w = m * a, P = m * i;
        p.set(d.subarray(w, w + i), P);
      }
      return this.readbackBuffer.unmap(), {
        data: p.buffer,
        width: n,
        height: t
      };
    }
  }
  class N {
    constructor(e) {
      __publicField(this, "renderTarget");
      __publicField(this, "renderTargetView");
      __publicField(this, "gBufferNormal");
      __publicField(this, "gBufferNormalView");
      __publicField(this, "depthTextures", []);
      __publicField(this, "depthTextureViews", []);
      __publicField(this, "accumulateBuffer");
      __publicField(this, "samplesBuffer");
      __publicField(this, "reservoirsBufferA");
      __publicField(this, "reservoirsBufferB");
      __publicField(this, "spatialReservoirsBuffer");
      __publicField(this, "sceneUniformBuffer");
      __publicField(this, "geometryBuffer");
      __publicField(this, "nodesBuffer");
      __publicField(this, "topologyBuffer");
      __publicField(this, "instanceBuffer");
      __publicField(this, "lightsBuffer");
      __publicField(this, "drawCommandBuffer");
      __publicField(this, "drawCommandsArray", null);
      __publicField(this, "texture");
      __publicField(this, "defaultTexture");
      __publicField(this, "sampler");
      __publicField(this, "historyTextures", []);
      __publicField(this, "historyTextureViews", []);
      __publicField(this, "historyIndex", 0);
      __publicField(this, "prevCameraData", new Float32Array(24));
      __publicField(this, "accumulatedJitter", {
        x: 0,
        y: 0
      });
      __publicField(this, "jitter", {
        x: 0,
        y: 0
      });
      __publicField(this, "prevJitter", {
        x: 0,
        y: 0
      });
      __publicField(this, "averageJitter", {
        x: 0,
        y: 0
      });
      __publicField(this, "bufferSize", 0);
      __publicField(this, "blasOffset", 0);
      __publicField(this, "vertexCount", 0);
      __publicField(this, "normOffset", 0);
      __publicField(this, "uvOffset", 0);
      __publicField(this, "lightCount", 0);
      __publicField(this, "instanceCount", 0);
      __publicField(this, "seed", Math.floor(Math.random() * 16777215));
      __publicField(this, "uniformMixedData", new Uint32Array(16));
      __publicField(this, "ctx");
      this.ctx = e;
    }
    init() {
      this.sceneUniformBuffer = this.ctx.device.createBuffer({
        size: 256,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
      }), this.sampler = this.ctx.device.createSampler({
        magFilter: "linear",
        minFilter: "linear",
        mipmapFilter: "linear",
        addressModeU: "repeat",
        addressModeV: "repeat"
      }), this.createDefaultTexture(), this.texture = this.defaultTexture;
    }
    createDefaultTexture() {
      const e = new Uint8Array([
        255,
        255,
        255,
        255
      ]);
      this.defaultTexture = this.ctx.device.createTexture({
        size: [
          1,
          1,
          1
        ],
        format: "rgba8unorm",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT
      }), this.ctx.device.queue.writeTexture({
        texture: this.defaultTexture,
        origin: [
          0,
          0,
          0
        ]
      }, e, {
        bytesPerRow: 256,
        rowsPerImage: 1
      }, [
        1,
        1
      ]);
    }
    updateScreenSize(e, n) {
      this.renderTarget && this.renderTarget.destroy(), this.renderTarget = this.ctx.device.createTexture({
        size: [
          e,
          n
        ],
        format: "rgba8unorm",
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT
      }), this.renderTargetView = this.renderTarget.createView(), this.gBufferNormal && this.gBufferNormal.destroy(), this.gBufferNormal = this.ctx.device.createTexture({
        size: [
          e,
          n
        ],
        format: "rgba32float",
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT
      }), this.gBufferNormalView = this.gBufferNormal.createView();
      for (let i = 0; i < 2; i++) this.depthTextures[i] && this.depthTextures[i].destroy(), this.depthTextures[i] = this.ctx.device.createTexture({
        size: [
          e,
          n
        ],
        format: "depth32float",
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
      }), this.depthTextureViews[i] = this.depthTextures[i].createView();
      this.bufferSize = e * n * 16, this.accumulateBuffer && this.accumulateBuffer.destroy(), this.accumulateBuffer = this.ctx.device.createBuffer({
        size: this.bufferSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
      });
      const t = e * n * 48;
      this.samplesBuffer && this.samplesBuffer.destroy(), this.samplesBuffer = this.ctx.device.createBuffer({
        size: t,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
      });
      const r = e * n * 64;
      this.reservoirsBufferA && this.reservoirsBufferA.destroy(), this.reservoirsBufferA = this.ctx.device.createBuffer({
        size: r,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
      }), this.reservoirsBufferB && this.reservoirsBufferB.destroy(), this.reservoirsBufferB = this.ctx.device.createBuffer({
        size: r,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
      }), this.spatialReservoirsBuffer && this.spatialReservoirsBuffer.destroy(), this.spatialReservoirsBuffer = this.ctx.device.createBuffer({
        size: r,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
      });
      for (let i = 0; i < 2; i++) this.historyTextures[i] && this.historyTextures[i].destroy(), this.historyTextures[i] = this.ctx.device.createTexture({
        size: [
          e,
          n
        ],
        format: "rgba16float",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT
      }), this.historyTextureViews[i] = this.historyTextures[i].createView();
    }
    resetAccumulation() {
      this.accumulateBuffer && this.ctx.device.queue.writeBuffer(this.accumulateBuffer, 0, new Float32Array(this.bufferSize / 4));
    }
    async loadTexturesFromWorld(e) {
      const n = e.textureCount;
      if (n === 0) {
        this.createDefaultTexture();
        return;
      }
      console.log(`Loading ${n} textures...`);
      const t = [];
      for (let r = 0; r < n; r++) {
        const i = e.getTexture(r);
        if (i) try {
          const s = new Blob([
            i
          ]), a = await createImageBitmap(s, {
            resizeWidth: 1024,
            resizeHeight: 1024
          });
          t.push(a);
        } catch (s) {
          console.warn(`Failed tex ${r}`, s), t.push(await this.createFallbackBitmap());
        }
        else t.push(await this.createFallbackBitmap());
      }
      this.texture && this.texture.destroy(), this.texture = this.ctx.device.createTexture({
        size: [
          1024,
          1024,
          t.length
        ],
        format: "rgba8unorm",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT
      });
      for (let r = 0; r < t.length; r++) this.ctx.device.queue.copyExternalImageToTexture({
        source: t[r]
      }, {
        texture: this.texture,
        origin: [
          0,
          0,
          r
        ]
      }, [
        1024,
        1024
      ]);
      await this.ctx.device.queue.onSubmittedWorkDone();
    }
    async createFallbackBitmap() {
      const e = document.createElement("canvas");
      e.width = 1024, e.height = 1024;
      const n = e.getContext("2d");
      return n.fillStyle = "white", n.fillRect(0, 0, 1024, 1024), await createImageBitmap(e);
    }
    ensureBuffer(e, n, t) {
      if (e && e.size >= n) return e;
      e && e.destroy();
      let r = Math.ceil(n * 1.5);
      return r = r + 3 & -4, r = Math.max(r, 16), this.ctx.device.createBuffer({
        label: t,
        size: r,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
      });
    }
    updateBuffer(e, n) {
      const t = n.byteLength;
      let r = false, i;
      return e === "topology" ? ((!this.topologyBuffer || this.topologyBuffer.size < t) && (r = true), this.topologyBuffer = this.ensureBuffer(this.topologyBuffer, t, "TopologyBuffer"), i = this.topologyBuffer) : e === "instance" ? (this.instanceCount = n.length / 36, (!this.instanceBuffer || this.instanceBuffer.size < t) && (r = true), this.instanceBuffer = this.ensureBuffer(this.instanceBuffer, t, "InstanceBuffer"), i = this.instanceBuffer) : e === "lights" ? ((!this.lightsBuffer || this.lightsBuffer.size < t) && (r = true), this.lightsBuffer = this.ensureBuffer(this.lightsBuffer, t, "LightsBuffer"), i = this.lightsBuffer) : (e === "draw_commands" && (this.drawCommandsArray = n), (!this.drawCommandBuffer || this.drawCommandBuffer.size < t) && (this.drawCommandBuffer && this.drawCommandBuffer.destroy(), this.drawCommandBuffer = this.ctx.device.createBuffer({
        label: "DrawCommandBuffer",
        size: Math.max(t, 16),
        usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST
      }), r = true), i = this.drawCommandBuffer), this.ctx.device.queue.writeBuffer(i, 0, n, 0, n.length), r;
    }
    updateCombinedGeometry(e, n, t) {
      const i = e.byteLength;
      this.normOffset = Math.ceil(i / 256) * 256;
      const s = n.byteLength;
      this.uvOffset = Math.ceil((this.normOffset + s) / 256) * 256;
      const a = this.uvOffset + t.byteLength;
      let c = false;
      (!this.geometryBuffer || this.geometryBuffer.size < a) && (c = true);
      const l = e.length / 4;
      return this.vertexCount = l, this.geometryBuffer = this.ensureBuffer(this.geometryBuffer, a, "GeometryBuffer"), !(t.length >= l * 2) && l > 0 && console.warn(`UV buffer mismatch: V=${l}, UV=${t.length / 2}. Filling 0.`), this.ctx.device.queue.writeBuffer(this.geometryBuffer, 0, e), this.ctx.device.queue.writeBuffer(this.geometryBuffer, this.normOffset, n), this.ctx.device.queue.writeBuffer(this.geometryBuffer, this.uvOffset, t), c;
    }
    updateCombinedBVH(e, n) {
      const t = e.byteLength, r = n.byteLength, i = t + r;
      let s = false;
      return (!this.nodesBuffer || this.nodesBuffer.size < i) && (s = true), this.nodesBuffer = this.ensureBuffer(this.nodesBuffer, i, "NodesBuffer"), this.ctx.device.queue.writeBuffer(this.nodesBuffer, 0, e), this.ctx.device.queue.writeBuffer(this.nodesBuffer, t, n), this.blasOffset = e.length / 8, s;
    }
    getHalton(e, n) {
      let t = 1, r = 0;
      for (; e > 0; ) t = t / n, r = r + t * (e % n), e = Math.floor(e / n);
      return r;
    }
    updateSceneUniforms(e, n, t) {
      if (this.lightCount = t, !this.sceneUniformBuffer) return;
      this.prevJitter.x = this.jitter.x, this.prevJitter.y = this.jitter.y;
      const r = this.getHalton(n % 16 + 1, 2) - 0.5, i = this.getHalton(n % 16 + 1, 3) - 0.5;
      this.jitter = {
        x: r / this.ctx.canvas.width,
        y: i / this.ctx.canvas.height
      }, this.ctx.device.queue.writeBuffer(this.sceneUniformBuffer, 0, e), this.ctx.device.queue.writeBuffer(this.sceneUniformBuffer, 96, this.prevCameraData), this.uniformMixedData[0] = n, this.uniformMixedData[1] = this.blasOffset, this.uniformMixedData[2] = this.vertexCount, this.uniformMixedData[3] = this.seed, this.uniformMixedData[4] = t, this.uniformMixedData[5] = this.ctx.canvas.width, this.uniformMixedData[6] = this.ctx.canvas.height, this.uniformMixedData[7] = 0, n === 1 ? (this.accumulatedJitter.x = this.jitter.x, this.accumulatedJitter.y = this.jitter.y) : (this.accumulatedJitter.x += this.jitter.x, this.accumulatedJitter.y += this.jitter.y), this.averageJitter.x = this.accumulatedJitter.x / n, this.averageJitter.y = this.accumulatedJitter.y / n;
      const s = new Float32Array(this.uniformMixedData.buffer);
      s[8] = this.jitter.x, s[9] = this.jitter.y, s[10] = this.averageJitter.x, s[11] = this.averageJitter.y, s[12] = this.prevJitter.x, s[13] = this.prevJitter.y, this.ctx.device.queue.writeBuffer(this.sceneUniformBuffer, 192, this.uniformMixedData), this.prevCameraData.set(e);
    }
    updateFrameUniforms(e, n) {
      this.prevJitter.x = this.jitter.x, this.prevJitter.y = this.jitter.y;
      const t = this.getHalton(n % 16 + 1, 2) - 0.5, r = this.getHalton(n % 16 + 1, 3) - 0.5;
      this.jitter = {
        x: t / this.ctx.canvas.width,
        y: r / this.ctx.canvas.height
      }, e === 1 ? (this.accumulatedJitter.x = this.jitter.x, this.accumulatedJitter.y = this.jitter.y) : (this.accumulatedJitter.x += this.jitter.x, this.accumulatedJitter.y += this.jitter.y), this.averageJitter.x = this.accumulatedJitter.x / e, this.averageJitter.y = this.accumulatedJitter.y / e, this.uniformMixedData[0] = e, this.uniformMixedData[1] = this.blasOffset, this.uniformMixedData[2] = this.vertexCount, this.uniformMixedData[3] = this.seed, this.uniformMixedData[4] = this.lightCount, this.uniformMixedData[5] = this.ctx.canvas.width, this.uniformMixedData[6] = this.ctx.canvas.height, this.uniformMixedData[7] = 0;
      const i = new Float32Array(this.uniformMixedData.buffer);
      i[8] = this.jitter.x, i[9] = this.jitter.y, i[10] = this.averageJitter.x, i[11] = this.averageJitter.y, i[12] = this.prevJitter.x, i[13] = this.prevJitter.y, this.ctx.device.queue.writeBuffer(this.sceneUniformBuffer, 192, this.uniformMixedData);
    }
  }
  const O = `// =========================================================
//   WebGPU Ray Tracer (Raytracer.wgsl)
// =========================================================

const PI = 3.141592653589793;
const T_MIN = 0.001;
const T_MAX = 1e30;
override MAX_DEPTH: u32;

// =========================================================
//   Structs
// =========================================================

struct Sample {
    hit_p: vec4<f32>,    // xyz: Position(secondary ray hit point), w: scatter.dir.x
    normal: vec4<f32>,   // xyz: Normal, w: scatter.dir.y
    radiance: vec4<f32>, // xyz: L_i, w: scatter.dir.z
}

struct Reservoir {
    sample: Sample,
    w_sum: f32,
    W: f32,
    M: u32,
    padding: f32,
}


struct Camera {
    origin: vec4<f32>, // w: lens_radius
    lower_left_corner: vec4<f32>,
    horizontal: vec4<f32>,
    vertical: vec4<f32>,
    u: vec4<f32>,
    v: vec4<f32>
}

struct SceneUniforms {
    camera: Camera,
    prev_camera: Camera,
    frame_count: u32,
    blas_base_idx: u32,
    vertex_count: u32,
    rand_seed: u32,
    light_count: u32,
    width: u32,
    height: u32,
    pad: u32,
    jitter: vec2<f32>,
    average_jitter: vec2<f32>,
    prev_jitter: vec2<f32>,
    pad2: vec2<f32>
}

struct MeshTopology {
    v0: u32,
    v1: u32,
    v2: u32,
    pad: u32,
    data0: vec4<f32>, // rgb: BaseColor, w: MaterialType (cast)
    data1: vec4<f32>, // x: Metallic, y: Roughness, z: IOR, w: 0.0
    data2: vec4<f32>, // x: BaseTex, y: MetRoughTex, z: NormalTex, w: EmissiveTex
    data3: vec4<f32>  // rgb: EmissiveColor, w: OcclusionTex
}

struct LightRef {
    inst_idx: u32,
    tri_idx: u32
}

struct BVHNode {
    min_b: vec4<f32>, // w: skip_pointer
    max_b: vec4<f32>, // w: data (internal: 0, leaf: (left_first << 3) | tri_count)
}

struct Instance {
    transform_0: vec4<f32>,
    transform_1: vec4<f32>,
    transform_2: vec4<f32>,
    transform_3: vec4<f32>,
    inv_0: vec4<f32>,
    inv_1: vec4<f32>,
    inv_2: vec4<f32>,
    inv_3: vec4<f32>,
    blas_node_offset: u32,
    attr_offset: u32,
    instance_id: u32,
    padding: u32,
}

struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
    inv_d: vec3<f32>,
    origin_inv_d: vec3<f32>
}

fn make_ray(origin: vec3<f32>, direction: vec3<f32>) -> Ray {
    let inv_d = 1.0 / direction;
    return Ray(origin, direction, inv_d, origin * inv_d);
}

struct HitResult {
    t: f32,
    tri_idx: f32,
    inst_idx: i32
}

struct ONB {
    u: vec3<f32>,
    v: vec3<f32>,
    w: vec3<f32>,
}

struct LightSample {
    L: vec3<f32>,       // Radiance
    dir: vec3<f32>,     // Direction to light
    dist: f32,          // Distance to light
    pdf: f32,           // PDF of sampling this point
}

struct ScatterResult {
    dir: vec3<f32>,
    pdf: f32,
    throughput: vec3<f32>,
    is_specular: bool
}


// \u516B\u9762\u4F53\u30A8\u30F3\u30B3\u30FC\u30C7\u30A3\u30F3\u30B0\u306B\u3088\u308B\u6CD5\u7DDA\u5727\u7E2E (vec3 -> vec2)
fn pack_normal(n: vec3<f32>) -> vec2<f32> {
    let p = n.xy * (1.0 / (abs(n.x) + abs(n.y) + abs(n.z)));
    return select(p, (1.0 - abs(p.yx)) * select(vec2(-1.0), vec2(1.0), p.xy >= vec2(0.0)), n.z < 0.0);
}

fn unpack_normal(p: vec2<f32>) -> vec3<f32> {
    var n = vec3(p, 1.0 - abs(p.x) - abs(p.y));
    let t = saturate(-n.z);
    n.x += select(t, -t, n.x >= 0.0);
    n.y += select(t, -t, n.y >= 0.0);
    return normalize(n);
}


// =========================================================
//   Bindings
// =========================================================

@group(0) @binding(1) var<storage, read_write> accumulateBuffer: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> scene: SceneUniforms;
@group(0) @binding(3) var<storage, read> geometry_pos: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> topology: array<MeshTopology>;
@group(0) @binding(5) var<storage, read> nodes: array<BVHNode>; 
@group(0) @binding(6) var<storage, read> instances: array<Instance>;
@group(0) @binding(7) var tex: texture_2d_array<f32>;
@group(0) @binding(8) var smp: sampler;
@group(0) @binding(9) var<storage, read> lights: array<LightRef>;
@group(0) @binding(11) var<storage, read> geometry_norm: array<vec4<f32>>;
@group(0) @binding(12) var<storage, read> geometry_uv: array<vec2<f32>>;
@group(0) @binding(13) var g_albedo: texture_2d<f32>;
@group(0) @binding(14) var g_normal: texture_2d<f32>;
@group(0) @binding(15) var g_depth: texture_depth_2d;
@group(0) @binding(16) var<storage, read_write> reservoirsBuffer: array<Reservoir>;
@group(0) @binding(17) var<storage, read> prevReservoirsBuffer: array<Reservoir>;

// =========================================================
//   Buffer Accessors
// =========================================================

fn get_pos(idx: u32) -> vec3<f32> {
    return geometry_pos[idx].xyz;
}

fn get_normal(idx: u32) -> vec3<f32> {
    return geometry_norm[idx].xyz;
}

fn get_uv(idx: u32) -> vec2<f32> {
    return geometry_uv[idx];
}

fn get_transform(inst: Instance) -> mat4x4<f32> {
    return mat4x4<f32>(inst.transform_0, inst.transform_1, inst.transform_2, inst.transform_3);
}

fn get_inv_transform(inst: Instance) -> mat4x4<f32> {
    return mat4x4<f32>(inst.inv_0, inst.inv_1, inst.inv_2, inst.inv_3);
}

// =========================================================
//   Math & RNG Helpers
// =========================================================

fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn init_rng(pixel_idx: u32, frame_count: u32) -> u32 {
    var seed = pixel_idx + frame_count * 719393u;
    seed ^= 2747636419u; seed *= 2654435769u; seed ^= (seed >> 16u);
    seed *= 2654435769u; seed ^= (seed >> 16u); seed *= 2654435769u;
    return seed;
}

fn rand_pcg(state: ptr<function, u32>) -> f32 {
    let old = *state; *state = old * 747796405u + 2891336453u;
    let word = ((*state) >> ((old >> 28u) + 4u)) ^ (*state);
    return f32((word >> 22u) ^ word) / 4294967295.0;
}

fn random_unit_vector(onb: ONB, rng: ptr<function, u32>) -> vec3<f32> {
    let r1 = rand_pcg(rng);
    let r2 = rand_pcg(rng);
    let phi = 2.0 * PI * r1;
    let cos_theta = sqrt(1.0 - r2);
    let sin_theta = sqrt(r2);
    let local_dir = vec3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
    return local_to_world(onb, local_dir);
}

fn random_in_unit_disk(rng: ptr<function, u32>) -> vec3<f32> {
    let r = sqrt(rand_pcg(rng));
    let theta = 2.0 * PI * rand_pcg(rng);
    return vec3<f32>(r * cos(theta), r * sin(theta), 0.0);
}

fn build_onb(n: vec3<f32>) -> ONB {
    let sign = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (sign + n.z);
    let b = n.x * n.y * a;
    let u = vec3(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x);
    let v = vec3(b, sign + n.y * n.y * a, -n.y);
    return ONB(u, v, n);
}

fn local_to_world(onb: ONB, a: vec3<f32>) -> vec3<f32> {
    return a.x * onb.u + a.y * onb.v + a.z * onb.w;
}

// =========================================================
//   BSDF Functions
// =========================================================

fn eval_diffuse(albedo: vec3<f32>) -> vec3<f32> {
    return albedo / PI;
}

fn sample_diffuse(normal: vec3<f32>, albedo: vec3<f32>, rng: ptr<function, u32>) -> ScatterResult {
    let onb = build_onb(normal);
    let dir = random_unit_vector(onb, rng);
    let cos_theta = max(dot(normal, dir), 0.0);
    return ScatterResult(dir, cos_theta / PI, albedo, false);
}

// GGX
fn ggx_d(n_dot_h: f32, a2: f32) -> f32 {
    let d = (n_dot_h * a2 - n_dot_h) * n_dot_h + 1.0;
    return a2 / (PI * d * d);
}

fn ggx_g(n_dot_v: f32, n_dot_l: f32, a2: f32) -> f32 {
    let g1_v = 2.0 * n_dot_v / (n_dot_v + sqrt(a2 + (1.0 - a2) * n_dot_v * n_dot_v));
    let g1_l = 2.0 * n_dot_l / (n_dot_l + sqrt(a2 + (1.0 - a2) * n_dot_l * n_dot_l));
    return g1_v * g1_l;
}

fn pow5(x: f32) -> f32 {
    let x2 = x * x;
    return x2 * x2 * x;
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (1.0 - f0) * pow5(clamp(1.0 - cos_theta, 0.0, 1.0));
}

fn eval_ggx(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, roughness: f32, f0: vec3<f32>) -> vec3<f32> {
    let h = normalize(v + l);
    let n_dot_v = max(dot(n, v), 1e-4);
    let n_dot_l = max(dot(n, l), 1e-4);
    let n_dot_h = max(dot(n, h), 1e-4);
    let v_dot_h = max(dot(v, h), 1e-4);

    let a2 = roughness * roughness;
    let d = ggx_d(n_dot_h, a2);
    let g = ggx_g(n_dot_v, n_dot_l, a2);
    let f = fresnel_schlick(v_dot_h, f0);

    return (d * g * f) / (4.0 * n_dot_v * n_dot_l);
}

fn sample_ggx(n: vec3<f32>, v: vec3<f32>, roughness: f32, f0: vec3<f32>, rng: ptr<function, u32>) -> ScatterResult {
    let a = roughness;
    let u = vec2(rand_pcg(rng), rand_pcg(rng));

    let phi = 2.0 * PI * u.x;
    let cos_theta = sqrt(max(0.0, (1.0 - u.y) / (1.0 + (a * a - 1.0) * u.y)));
    let sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));

    let h_local = vec3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
    let onb = build_onb(n);
    let h = local_to_world(onb, h_local);
    let l = reflect(-v, h);

    if dot(n, l) <= 0.0 {
        return ScatterResult(vec3(0.0), 0.0, vec3(0.0), false);
    }

    let n_dot_v = max(dot(n, v), 1e-4);
    let n_dot_l = max(dot(n, l), 1e-4);
    let n_dot_h = max(dot(n, h), 1e-4);
    let v_dot_h = max(dot(v, h), 1e-4);

    let a2 = a * a;
    let d = ggx_d(n_dot_h, a2);
    let g = ggx_g(n_dot_v, n_dot_l, a2);
    let f = fresnel_schlick(v_dot_h, f0);

    let pdf = (d * n_dot_h) / (4.0 * v_dot_h);
    var throughput = vec3(0.0);
    if pdf > 1e-6 {
        throughput = (g * f * v_dot_h) / (n_dot_v * n_dot_h);
    }
    let treat_as_specular = roughness < 0.01;

    return ScatterResult(l, pdf, throughput, treat_as_specular);
}

fn bsdf_to_throughput(d: f32, g: f32, f: vec3<f32>, n_dot_v: f32, n_dot_l: f32, n_dot_h: f32, v_dot_h: f32, pdf: f32) -> vec3<f32> {
    if pdf <= 0.0 { return vec3(0.0); }
    return (d * g * f) / (4.0 * n_dot_v * n_dot_l) * n_dot_l / pdf;
}



// Dielectric
fn reflectance_dielectric(cosine: f32, ref_idx: f32) -> f32 {
    var r0 = (1.0 - ref_idx) / (1.0 + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0 - r0) * pow5(1.0 - cosine);
}

fn sample_dielectric(dir: vec3<f32>, normal: vec3<f32>, ior: f32, albedo: vec3<f32>, rng: ptr<function, u32>) -> ScatterResult {
    let front_face = dot(dir, normal) < 0.0;
    let refraction_ratio = select(ior, 1.0 / ior, front_face);
    let n = select(-normal, normal, front_face);

    let unit_dir = normalize(dir);
    let cos_theta = min(dot(-unit_dir, n), 1.0);
    let sin_theta = sqrt(1.0 - cos_theta * cos_theta);

    let cannot_refract = refraction_ratio * sin_theta > 1.0;
    var direction: vec3<f32>;

    if cannot_refract || reflectance_dielectric(cos_theta, refraction_ratio) > rand_pcg(rng) {
        direction = reflect(unit_dir, n);
    } else {
        direction = refract(unit_dir, n, refraction_ratio);
    }

    return ScatterResult(direction, 1.0, albedo, true);
}

// =========================================================
//   Direct Light Sampling
// =========================================================

fn sample_light_source(hit_p: vec3<f32>, rng: ptr<function, u32>) -> LightSample {
    let light_count = scene.light_count;
    if light_count == 0u {
        return LightSample(vec3(0.0), vec3(0.0), 0.0, 0.0);
    }

    let light_pick_idx = u32(rand_pcg(rng) * f32(light_count));
    let l_ref = lights[light_pick_idx];

    let tri = topology[l_ref.tri_idx];
    let inst = instances[l_ref.inst_idx];
    let m = get_transform(inst);

    let v0 = (m * vec4(get_pos(tri.v0), 1.0)).xyz;
    let v1 = (m * vec4(get_pos(tri.v1), 1.0)).xyz;
    let v2 = (m * vec4(get_pos(tri.v2), 1.0)).xyz;

    let r1 = rand_pcg(rng);
    let r2 = rand_pcg(rng);
    let sqrt_r1 = sqrt(r1);
    let u = 1.0 - sqrt_r1;
    let v = r2 * sqrt_r1;
    let w = 1.0 - u - v;

    let p = v0 * u + v1 * v + v2 * w;
    let edge1 = v1 - v0;
    let edge2 = v2 - v0;
    let n_raw = normalize(cross(edge1, edge2));
    let area = length(cross(edge1, edge2)) * 0.5;

    let l_dir = p - hit_p;
    let dist_sq = dot(l_dir, l_dir);
    let dist = sqrt(dist_sq);
    let unit_l = l_dir / dist;

    let cos_theta_l = max(dot(n_raw, -unit_l), 0.0);
    if cos_theta_l < 1e-6 || area < 1e-6 {
        return LightSample(vec3(0.0), vec3(0.0), 0.0, 0.0);
    }

    // Albedo if light
    let uv0 = get_uv(tri.v0);
    let uv1 = get_uv(tri.v1);
    let uv2 = get_uv(tri.v2);
    let tex_uv = uv0 * u + uv1 * v + uv2 * w;
    var L = tri.data0.rgb;
    let base_tex = tri.data2.x;
    if base_tex > -0.5 {
        L *= textureSampleLevel(tex, smp, tex_uv, i32(base_tex), 0.0).rgb;
    }

    let pdf = (dist_sq / (cos_theta_l * area)) / f32(light_count);

    return LightSample(L, unit_l, dist, pdf);
}

fn get_light_pdf(origin: vec3<f32>, tri_idx: u32, inst_idx: u32, t: f32, l_dir: vec3<f32>) -> f32 {
    let tri = topology[tri_idx];
    let inst = instances[inst_idx];
    let m = get_transform(inst);

    let v0 = (m * vec4(get_pos(tri.v0), 1.0)).xyz;
    let v1 = (m * vec4(get_pos(tri.v1), 1.0)).xyz;
    let v2 = (m * vec4(get_pos(tri.v2), 1.0)).xyz;

    let edge1 = v1 - v0;
    let edge2 = v2 - v0;
    let area = length(cross(edge1, edge2)) * 0.5;
    let normal = normalize(cross(edge1, edge2));

    let cos_theta_l = max(dot(normal, -l_dir), 0.0);
    if cos_theta_l < 1e-4 { return 0.0; }

    let light_count = scene.light_count;
    let dist_sq = t * t;
    return (dist_sq / (cos_theta_l * area)) / f32(light_count);
}

fn power_heuristic(pdf_a: f32, pdf_b: f32) -> f32 {
    let a2 = pdf_a * pdf_a;
    let b2 = pdf_b * pdf_b;
    return a2 / (a2 + b2 + 1e-6);
}

// =========================================================
//   Intersection Functions
// =========================================================

fn intersect_aabb(min_b: vec3<f32>, max_b: vec3<f32>, r: Ray, t_min: f32, t_max: f32) -> f32 {
    let t1 = min_b * r.inv_d - r.origin_inv_d;
    let t2 = max_b * r.inv_d - r.origin_inv_d;
    let t_near = min(t1, t2);
    let t_far = max(t1, t2);
    let tm_near = max(t_min, max(t_near.x, max(t_near.y, t_near.z)));
    let tm_far = min(t_max, min(t_far.x, min(t_far.y, t_far.z)));
    return select(T_MAX, tm_near, tm_near <= tm_far);
}

fn hit_triangle_raw(v0: vec3<f32>, v1: vec3<f32>, v2: vec3<f32>, r: Ray, t_min: f32, t_max: f32) -> f32 {
    let e1 = v1 - v0; let e2 = v2 - v0;
    let h = cross(r.direction, e2); let a = dot(e1, h);
    if abs(a) < 1e-6 { return -1.0; } // Increased epsilon
    let f = 1.0 / a; let s = r.origin - v0; let u = f * dot(s, h);
    if u < 0.0 || u > 1.0 { return -1.0; }
    let q = cross(s, e1); let v = f * dot(r.direction, q);
    if v < 0.0 || u + v > 1.0 { return -1.0; }
    let t = f * dot(e2, q);
    return select(-1.0, t, t > t_min && t < t_max);
}

fn intersect_blas(r: Ray, t_min: f32, t_max: f32, node_start_idx: u32) -> vec2<f32> {
    var closest_t = t_max;
    var hit_idx = -1.0;
    
    let end_node = node_start_idx + bitcast<u32>(nodes[node_start_idx].min_b.w);
    var curr = node_start_idx;
    
    while curr < end_node {
        let node = nodes[curr];
        
        var hit_t = closest_t;
        let t_aabb = intersect_aabb(node.min_b.xyz, node.max_b.xyz, r, t_min, closest_t);
        
        if t_aabb < T_MAX {
            let data = bitcast<u32>(node.max_b.w);
            if data != 0u {
                // Leaf node
                let first = data >> 3u;
                let count = data & 7u;
                for (var i = 0u; i < count; i++) {
                    let tri_id = first + i;
                    let tr = topology[tri_id];
                    let t = hit_triangle_raw(get_pos(tr.v0), get_pos(tr.v1), get_pos(tr.v2), r, t_min, closest_t);
                    if t > 0.0 { 
                        closest_t = t; 
                        hit_idx = f32(tri_id); 
                    }
                }
                curr = node_start_idx + bitcast<u32>(node.min_b.w);
            } else {
                // Internal node
                curr = curr + 1u;
            }
        } else {
            // Missed AABB
            curr = node_start_idx + bitcast<u32>(node.min_b.w);
        }
    }
    return vec2<f32>(closest_t, hit_idx);
}

fn intersect_tlas(r: Ray, t_min: f32, t_max: f32) -> HitResult {
    var res: HitResult; res.t = t_max; res.tri_idx = -1.0; res.inst_idx = -1;
    if scene.blas_base_idx == 0u { return res; }

    var curr = 0u;
    let end_node = bitcast<u32>(nodes[0].min_b.w);

    while curr < end_node {
        let node = nodes[curr];
        
        if intersect_aabb(node.min_b.xyz, node.max_b.xyz, r, t_min, res.t) < T_MAX {
            let data = bitcast<u32>(node.max_b.w);
            if data != 0u {
                // Leaf
                let inst_idx = data >> 3u;
                let inst = instances[inst_idx];
                let r_local = make_ray((get_inv_transform(inst) * vec4(r.origin, 1.0)).xyz, (get_inv_transform(inst) * vec4(r.direction, 0.0)).xyz);
                let blas = intersect_blas(r_local, t_min, res.t, scene.blas_base_idx + inst.blas_node_offset);
                if blas.y > -0.5 { 
                    res.t = blas.x; 
                    res.tri_idx = blas.y; 
                    res.inst_idx = i32(inst_idx); 
                }
                curr = bitcast<u32>(node.min_b.w);
            } else {
                curr = curr + 1u;
            }
        } else {
            curr = bitcast<u32>(node.min_b.w);
        }
    }
    return res;
}

// shadow ray\u7248
// \u30B7\u30E3\u30C9\u30A6\u30EC\u30A4\u7528\u306EBLAS\u4EA4\u5DEE\u5224\u5B9A\uFF08\u30D2\u30C3\u30C8\u3057\u305F\u3089\u5373true\u3092\u8FD4\u3059\uFF09
fn intersect_blas_shadow(r: Ray, t_min: f32, t_max: f32, node_start_idx: u32) -> bool {
    let end_node = node_start_idx + bitcast<u32>(nodes[node_start_idx].min_b.w);
    var curr = node_start_idx;
    
    while curr < end_node {
        let node = nodes[curr];
        
        let t_aabb = intersect_aabb(node.min_b.xyz, node.max_b.xyz, r, t_min, t_max);
        if t_aabb < T_MAX {
            let data = bitcast<u32>(node.max_b.w);
            if data != 0u {
                // Leaf node
                let first = data >> 3u;
                let count = data & 7u;
                for (var i = 0u; i < count; i++) {
                    let tri_id = first + i;
                    let tr = topology[tri_id];
                    let t = hit_triangle_raw(get_pos(tr.v0), get_pos(tr.v1), get_pos(tr.v2), r, t_min, t_max);
                    if t > 0.0 { return true; }
                }
                curr = node_start_idx + bitcast<u32>(node.min_b.w);
            } else {
                // Internal node
                curr = curr + 1u;
            }
        } else {
            // Missed AABB
            curr = node_start_idx + bitcast<u32>(node.min_b.w);
        }
    }
    return false;
}

// \u30B7\u30E3\u30C9\u30A6\u30EC\u30A4\u7528\u306ETLAS\u4EA4\u5DEE\u5224\u5B9A
fn intersect_tlas_shadow(r: Ray, t_min: f32, t_max: f32) -> bool {
    if scene.blas_base_idx == 0u { return false; }

    var curr = 0u;
    let end_node = bitcast<u32>(nodes[0].min_b.w);

    while curr < end_node {
        let node = nodes[curr];
        
        let t_aabb = intersect_aabb(node.min_b.xyz, node.max_b.xyz, r, t_min, t_max);
        if t_aabb < T_MAX {
            let data = bitcast<u32>(node.max_b.w);
            if data != 0u {
                // Leaf
                let inst_idx = data >> 3u;
                let inst = instances[inst_idx];
                
                let r_local = make_ray(
                    (get_inv_transform(inst) * vec4(r.origin, 1.0)).xyz, 
                    (get_inv_transform(inst) * vec4(r.direction, 0.0)).xyz
                );
                
                if intersect_blas_shadow(r_local, t_min, t_max, scene.blas_base_idx + inst.blas_node_offset) {
                    return true;
                }
                curr = bitcast<u32>(node.min_b.w);
            } else {
                curr = curr + 1u;
            }
        } else {
            curr = bitcast<u32>(node.min_b.w);
        }
    }
    return false;
}

fn get_throughput(w_o: vec3<f32>, w_i: vec3<f32>, normal: vec3<f32>, mat_type: u32, roughness: f32, f0: vec3<f32>, albedo: vec3<f32>) -> vec3<f32> {
    if mat_type == 0u {
        return albedo;
    } else if mat_type == 1u {
        let h = normalize(w_o + w_i);
        let n_dot_v = max(dot(normal, w_o), 1e-4);
        let n_dot_l = max(dot(normal, w_i), 1e-4);
        let n_dot_h = max(dot(normal, h), 1e-4);
        let v_dot_h = max(dot(w_o, h), 1e-4);
        let a2 = roughness * roughness;
        let g = ggx_g(n_dot_v, n_dot_l, a2);
        let f = fresnel_schlick(v_dot_h, f0);
        return (g * f * v_dot_h) / (n_dot_v * n_dot_h);
    } else { // mat_type == 2u
        return albedo;
    }
}

fn eval_brdf_cos(w_o: vec3<f32>, w_i: vec3<f32>, normal: vec3<f32>, mat_type: u32, roughness: f32, f0: vec3<f32>, albedo: vec3<f32>) -> vec3<f32> {
    let n_dot_l = max(dot(normal, w_i), 1e-4);

    if mat_type == 0u {
        return (albedo / PI) * n_dot_l;
    } else if mat_type == 1u {
        let h = normalize(w_o + w_i);
        let n_dot_v = max(dot(normal, w_o), 1e-4);
        let n_dot_h = max(dot(normal, h), 1e-4);
        let v_dot_h = max(dot(w_o, h), 1e-4);
        let a2 = roughness * roughness;
        let d = ggx_d(n_dot_h, a2);
        let g = ggx_g(n_dot_v, n_dot_l, a2);
        let f = fresnel_schlick(v_dot_h, f0);
        return (d * g * f) / (4.0 * n_dot_v);
    } else { 
        return vec3<f32>(0.0);
    }
}

fn get_pdf(w_o: vec3<f32>, w_i: vec3<f32>, normal: vec3<f32>, mat_type: u32, roughness: f32) -> f32 {
    if mat_type == 0u {
        return max(dot(normal, w_i), 0.0) / PI;
    } else if mat_type == 1u {
        let h = normalize(w_o + w_i);
        let n_dot_h = max(dot(normal, h), 1e-4);
        let v_dot_h = max(dot(w_o, h), 1e-4);
        let a2 = roughness * roughness;
        let d = ggx_d(n_dot_h, a2);
        return (d * n_dot_h) / (4.0 * v_dot_h);
    } else {
        return 0.0;
    }
}

fn update_reservoir(r: ptr<function, Reservoir>, s: Sample, weight: f32, rng: ptr<function, u32>) {
    r.w_sum += weight;
    if rand_pcg(rng) < (weight / r.w_sum) {
        r.sample = s;
    }
}


@compute @workgroup_size(8, 8)
fn initial_sampling(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= scene.width || id.y >= scene.height { return; }
    let p_idx = id.y * scene.width + id.x;
    var rng = init_rng(p_idx, scene.frame_count);

    let depth_val = textureLoad(g_depth, id.xy, 0);
    if depth_val >= 1.0 { 
        var r: Reservoir;
        r.sample = Sample(vec4(0.0), vec4(0.0), vec4(0.0));
        r.M = 1u;
        r.w_sum = length(r.sample.radiance.xyz);
        r.W = 0.0;
        reservoirsBuffer[p_idx] = r;
        return; 
    }

    let g_normal_val = textureLoad(g_normal, id.xy, 0);
    var tri_idx: u32 = bitcast<u32>(g_normal_val.z);
    var inst_idx: i32 = i32(bitcast<u32>(g_normal_val.w));

    var tri = topology[tri_idx];
    var inst = instances[inst_idx];
    var inv = get_inv_transform(inst);
    var v0_pos = get_pos(tri.v0);
    var v1_pos = get_pos(tri.v1);
    var v2_pos = get_pos(tri.v2);

    var off = vec3(0.);
    if scene.camera.origin.w > 0. {
        let rd = scene.camera.origin.w * random_in_unit_disk(&rng);
        off = scene.camera.u.xyz * rd.x + scene.camera.v.xyz * rd.y;
    }

    let u_cam = (f32(id.x) + 0.5 + scene.jitter.x * f32(scene.width)) / f32(scene.width);
    let v_cam = 1. - (f32(id.y) + 0.5 + scene.jitter.y * f32(scene.height)) / f32(scene.height);
    let dir = scene.camera.lower_left_corner.xyz + u_cam * scene.camera.horizontal.xyz + v_cam * scene.camera.vertical.xyz - scene.camera.origin.xyz - off;
    var r_in = make_ray(scene.camera.origin.xyz + off, dir);

    var r_local = make_ray((inv * vec4(r_in.origin, 1.)).xyz, (inv * vec4(r_in.direction, 0.)).xyz);
    var s = r_local.origin - v0_pos;
    var e1 = v1_pos - v0_pos;
    var e2 = v2_pos - v0_pos;
    var h_val = cross(r_local.direction, e2);
    var f_val = 1.0 / dot(e1, h_val);
    var u_bar = f_val * dot(s, h_val);
    var q = cross(s, e1);
    var v_bar = f_val * dot(r_local.direction, q);
    var w_bar = 1.0 - u_bar - v_bar;
    var hit_t = f_val * dot(e2, q);
    
    var uv0 = get_uv(tri.v0);
    var uv1 = get_uv(tri.v1);
    var uv2 = get_uv(tri.v2);
    var tex_uv = uv0 * w_bar + uv1 * u_bar + uv2 * v_bar;

    var normal = unpack_normal(g_normal_val.xy);
    var albedo = textureLoad(g_albedo, id.xy, 0).rgb;

    var local_geom_n = normalize(cross(e1, e2));
    var world_geom_n = normalize((vec4(local_geom_n, 0.0) * inv).xyz);

    let mat_type = u32(tri.data0.w + 0.5);
    let primary_hit_p = r_in.origin + r_in.direction * hit_t;

    normal = select(-normal, normal, dot(r_in.direction, normal) < 0.0);
    world_geom_n = select(-world_geom_n, world_geom_n, dot(r_in.direction, world_geom_n) < 0.0);

    var metallic = tri.data1.x;
    var roughness = tri.data1.y;
    if tri.data2.y > -0.5 {
        let mr = textureSampleLevel(tex, smp, tex_uv, i32(tri.data2.y), 0.0).rgb;
        metallic *= mr.b; roughness *= mr.g;
    }
    roughness = max(roughness, 0.005);

    var emissive = tri.data3.rgb;
    if tri.data2.w > -0.5 { emissive *= textureSampleLevel(tex, smp, tex_uv, i32(tri.data2.w), 0.0).rgb; }

    let f0 = mix(vec3(0.04), albedo, metallic);
    
    if mat_type == 3u || length(emissive) > 1e-4 {
        var r: Reservoir;
        r.sample = Sample(vec4(primary_hit_p, 1.0), vec4(normal, 0.0), vec4(0.0));
        r.M = 1u;
        r.w_sum = length(r.sample.radiance.xyz);
        r.W = 0.0;
        reservoirsBuffer[p_idx] = r;
        return;
    }

    var scatter: ScatterResult;
    if mat_type == 0u { 
        scatter = sample_diffuse(normal, albedo, &rng); 
    } else if mat_type == 1u { 
        scatter = sample_ggx(normal, -r_in.direction, roughness, f0, &rng); 
    } else { 
        scatter = sample_dielectric(r_in.direction, normal, tri.data1.z, albedo, &rng); 
    }

    if mat_type != 2u && dot(scatter.dir, world_geom_n) <= 0.0 {
        var r: Reservoir;
        r.sample = Sample(vec4(primary_hit_p, 1.0), vec4(normal, 0.0), vec4(0.0));
        r.M = 1u;
        r.w_sum = length(r.sample.radiance.xyz);
        r.W = 0.0;
        reservoirsBuffer[p_idx] = r;
        return;
    }

    if scatter.pdf <= 0.0 || length(scatter.throughput) <= 0.0 {
        var r: Reservoir;
        r.sample = Sample(vec4(primary_hit_p, 1.0), vec4(normal, 0.0), vec4(0.0));
        r.M = 1u;
        r.w_sum = length(r.sample.radiance.xyz);
        r.W = 0.0;
        reservoirsBuffer[p_idx] = r;
        return;
    }

    let ray_offset_normal = select(-world_geom_n, world_geom_n, dot(scatter.dir, world_geom_n) > 0.0);
    var ray = make_ray(primary_hit_p + ray_offset_normal * 1e-4, scatter.dir);

    let is_delta = (mat_type == 2u) || (mat_type == 1u && metallic > 0.9 && roughness < 0.01);
    var throughput = select(vec3(1.0), scatter.throughput, is_delta);

    var radiance = vec3(0.0);
    var prev_bsdf_pdf = scatter.pdf;
    var specular_bounce = scatter.is_specular;

    var sec_hit_p = vec3(0.0);
    var sec_normal = vec3(0.0);
    var sec_hit_valid = false;

    for (var depth = 1u; depth < MAX_DEPTH; depth++) {
        let hit = intersect_tlas(ray, T_MIN, T_MAX);
        if hit.inst_idx < 0 { break; }
        
        hit_t = hit.t;
        tri_idx = u32(hit.tri_idx);
        inst_idx = hit.inst_idx;

        tri = topology[tri_idx];
        inst = instances[inst_idx];
        inv = get_inv_transform(inst);
        v0_pos = get_pos(tri.v0);
        v1_pos = get_pos(tri.v1);
        v2_pos = get_pos(tri.v2);

        r_local = make_ray((inv * vec4(ray.origin, 1.)).xyz, (inv * vec4(ray.direction, 0.)).xyz);
        s = r_local.origin - v0_pos;
        e1 = v1_pos - v0_pos;
        e2 = v2_pos - v0_pos;
        h_val = cross(r_local.direction, e2);
        f_val = 1.0 / dot(e1, h_val);
        u_bar = f_val * dot(s, h_val);
        q = cross(s, e1);
        v_bar = f_val * dot(r_local.direction, q);
        w_bar = 1.0 - u_bar - v_bar;

        uv0 = get_uv(tri.v0);
        uv1 = get_uv(tri.v1);
        uv2 = get_uv(tri.v2);
        tex_uv = uv0 * w_bar + uv1 * u_bar + uv2 * v_bar;

        let n0 = get_normal(tri.v0);
        let n1 = get_normal(tri.v1);
        let n2 = get_normal(tri.v2);
        let ln = normalize(n0 * w_bar + n1 * u_bar + n2 * v_bar);
        normal = normalize((vec4(ln, 0.0) * inv).xyz);

        albedo = tri.data0.rgb;
        if tri.data2.x > -0.5 { albedo *= textureSampleLevel(tex, smp, tex_uv, i32(tri.data2.x), 0.0).rgb; }

        if tri.data2.z > -0.5 {
            let n_map = textureSampleLevel(tex, smp, tex_uv, i32(tri.data2.z), 0.0).rgb * 2.0 - 1.0;
            let T = normalize(e1);
            let B = normalize(cross(ln, T));
            let ln_mapped = normalize(T * n_map.x + B * n_map.y + ln * n_map.z);
            normal = normalize((vec4(ln_mapped, 0.0) * inv).xyz);
        }

        local_geom_n = normalize(cross(e1, e2));
        world_geom_n = normalize((vec4(local_geom_n, 0.0) * inv).xyz);

        let curr_mat_type = u32(tri.data0.w + 0.5);
        let curr_hit_p = ray.origin + ray.direction * hit_t;

        if depth == 1u {
            sec_hit_p = curr_hit_p;
            sec_normal = normal;
            sec_hit_valid = true;
        }

        normal = select(-normal, normal, dot(ray.direction, normal) < 0.0);
        world_geom_n = select(-world_geom_n, world_geom_n, dot(ray.direction, world_geom_n) < 0.0);

        metallic = tri.data1.x;
        roughness = tri.data1.y;
        if tri.data2.y > -0.5 {
            let mr = textureSampleLevel(tex, smp, tex_uv, i32(tri.data2.y), 0.0).rgb;
            metallic *= mr.b; roughness *= mr.g;
        }
        roughness = max(roughness, 0.005);

        emissive = tri.data3.rgb;
        if tri.data2.w > -0.5 { emissive *= textureSampleLevel(tex, smp, tex_uv, i32(tri.data2.w), 0.0).rgb; }

        let curr_f0 = mix(vec3(0.04), albedo, metallic);

        if curr_mat_type == 3u || length(emissive) > 1e-4 {
            let em_val = select(emissive, albedo, curr_mat_type == 3u);
            if specular_bounce { 
                radiance += throughput * em_val; 
            } else { 
                radiance += throughput * em_val * power_heuristic(prev_bsdf_pdf, get_light_pdf(ray.origin, tri_idx, u32(inst_idx), hit_t, ray.direction)); 
            }
            if curr_mat_type == 3u { break; }
        }

        if curr_mat_type != 2u && curr_mat_type != 3u {
            let light_s = sample_light_source(curr_hit_p, &rng);
            if light_s.pdf > 1e-6 {
                if !intersect_tlas_shadow(make_ray(curr_hit_p + world_geom_n * 1e-4, light_s.dir), T_MIN, light_s.dist - 2e-4) {
                    let w_o = -ray.direction;
                    let tp = eval_brdf_cos(w_o, light_s.dir, normal, curr_mat_type, roughness, curr_f0, albedo);
                    let bsdf_pdf_val = get_pdf(w_o, light_s.dir, normal, curr_mat_type, roughness);
                    radiance += throughput * tp * light_s.L * power_heuristic(light_s.pdf, bsdf_pdf_val) / light_s.pdf;
                }
            }
        }

        var curr_scatter: ScatterResult;
        if curr_mat_type == 0u { 
            curr_scatter = sample_diffuse(normal, albedo, &rng); 
        } else if curr_mat_type == 1u { 
            curr_scatter = sample_ggx(normal, -ray.direction, roughness, curr_f0, &rng); 
        } else { 
            curr_scatter = sample_dielectric(ray.direction, normal, tri.data1.z, albedo, &rng); 
        }

        if curr_mat_type != 2u && dot(curr_scatter.dir, world_geom_n) <= 0.0 {
            break;
        }
        
        if curr_scatter.pdf <= 0.0 || length(curr_scatter.throughput) <= 0.0 { break; }
        
        throughput *= curr_scatter.throughput;
        
        let curr_ray_offset_normal = select(-world_geom_n, world_geom_n, dot(curr_scatter.dir, world_geom_n) > 0.0);
        ray = make_ray(curr_hit_p + curr_ray_offset_normal * 1e-4, curr_scatter.dir);
        
        prev_bsdf_pdf = curr_scatter.pdf;
        specular_bounce = curr_scatter.is_specular;

        if depth > 3u {
            let p = max(throughput.r, max(throughput.g, throughput.b));
            if rand_pcg(&rng) > p { break; }
            throughput /= p;
        }
    }

    var out_sample: Sample;
    if sec_hit_valid {
        out_sample = Sample(vec4(sec_hit_p, scatter.dir.x), vec4(sec_normal, scatter.dir.y), vec4(radiance, scatter.dir.z));
    } else {
        out_sample = Sample(vec4(primary_hit_p + scatter.dir * 1000.0, scatter.dir.x), vec4(vec3(0.0), scatter.dir.y), vec4(vec3(0.0), scatter.dir.z));
    }
    var r: Reservoir;
    r.sample = out_sample;
    r.M = 1u;

    var p_hat = 0.0;
    if is_delta {
        p_hat = luminance(radiance);
    } else {
        let w_i = scatter.dir;
        let w_o = -r_in.direction;
        let tp = eval_brdf_cos(w_o, w_i, normal, mat_type, roughness, f0, albedo);
        p_hat = luminance(radiance * tp);
    }

    r.w_sum = p_hat / max(scatter.pdf, 1e-6);
    if p_hat > 1e-6 {
        r.W = r.w_sum / (f32(r.M) * p_hat);
    } else {
        r.W = 0.0;
    }
    reservoirsBuffer[p_idx] = r;
}`, q = `// =========================================================
//   WebGPU Ray Tracer (Raytracer.wgsl)
// =========================================================

const PI = 3.141592653589793;
const T_MIN = 0.001;
const T_MAX = 1e30;
override MAX_DEPTH: u32;

// =========================================================
//   Structs
// =========================================================

struct Sample {
    hit_p: vec4<f32>,    // xyz: Position(secondary ray hit point), w: scatter.dir.x
    normal: vec4<f32>,   // xyz: Normal, w: scatter.dir.y
    radiance: vec4<f32>, // xyz: L_i, w: scatter.dir.z
}

struct Reservoir {
    sample: Sample,
    w_sum: f32,
    W: f32,
    M: u32,
    padding: f32,
}


struct Camera {
    origin: vec4<f32>, // w: lens_radius
    lower_left_corner: vec4<f32>,
    horizontal: vec4<f32>,
    vertical: vec4<f32>,
    u: vec4<f32>,
    v: vec4<f32>
}

struct SceneUniforms {
    camera: Camera,
    prev_camera: Camera,
    frame_count: u32,
    blas_base_idx: u32,
    vertex_count: u32,
    rand_seed: u32,
    light_count: u32,
    width: u32,
    height: u32,
    pad: u32,
    jitter: vec2<f32>,
    average_jitter: vec2<f32>,
    prev_jitter: vec2<f32>,
    pad2: vec2<f32>
}

struct MeshTopology {
    v0: u32,
    v1: u32,
    v2: u32,
    pad: u32,
    data0: vec4<f32>, // rgb: BaseColor, w: MaterialType (cast)
    data1: vec4<f32>, // x: Metallic, y: Roughness, z: IOR, w: 0.0
    data2: vec4<f32>, // x: BaseTex, y: MetRoughTex, z: NormalTex, w: EmissiveTex
    data3: vec4<f32>  // rgb: EmissiveColor, w: OcclusionTex
}

struct LightRef {
    inst_idx: u32,
    tri_idx: u32
}

struct BVHNode {
    min_b: vec4<f32>, // w: skip_pointer
    max_b: vec4<f32>, // w: data (internal: 0, leaf: (left_first << 3) | tri_count)
}

struct Instance {
    transform_0: vec4<f32>,
    transform_1: vec4<f32>,
    transform_2: vec4<f32>,
    transform_3: vec4<f32>,
    inv_0: vec4<f32>,
    inv_1: vec4<f32>,
    inv_2: vec4<f32>,
    inv_3: vec4<f32>,
    blas_node_offset: u32,
    attr_offset: u32,
    instance_id: u32,
    padding: u32,
}

struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
    inv_d: vec3<f32>,
    origin_inv_d: vec3<f32>
}

fn make_ray(origin: vec3<f32>, direction: vec3<f32>) -> Ray {
    let inv_d = 1.0 / direction;
    return Ray(origin, direction, inv_d, origin * inv_d);
}

struct HitResult {
    t: f32,
    tri_idx: f32,
    inst_idx: i32
}

struct ONB {
    u: vec3<f32>,
    v: vec3<f32>,
    w: vec3<f32>,
}

struct LightSample {
    L: vec3<f32>,       // Radiance
    dir: vec3<f32>,     // Direction to light
    dist: f32,          // Distance to light
    pdf: f32,           // PDF of sampling this point
}

struct ScatterResult {
    dir: vec3<f32>,
    pdf: f32,
    throughput: vec3<f32>,
    is_specular: bool
}


// \u516B\u9762\u4F53\u30A8\u30F3\u30B3\u30FC\u30C7\u30A3\u30F3\u30B0\u306B\u3088\u308B\u6CD5\u7DDA\u5727\u7E2E (vec3 -> vec2)
fn pack_normal(n: vec3<f32>) -> vec2<f32> {
    let p = n.xy * (1.0 / (abs(n.x) + abs(n.y) + abs(n.z)));
    return select(p, (1.0 - abs(p.yx)) * select(vec2(-1.0), vec2(1.0), p.xy >= vec2(0.0)), n.z < 0.0);
}

fn unpack_normal(p: vec2<f32>) -> vec3<f32> {
    var n = vec3(p, 1.0 - abs(p.x) - abs(p.y));
    let t = saturate(-n.z);
    n.x += select(t, -t, n.x >= 0.0);
    n.y += select(t, -t, n.y >= 0.0);
    return normalize(n);
}


// =========================================================
//   Bindings
// =========================================================

@group(0) @binding(1) var<storage, read_write> accumulateBuffer: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> scene: SceneUniforms;
@group(0) @binding(3) var<storage, read> geometry_pos: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> topology: array<MeshTopology>;
@group(0) @binding(5) var<storage, read> nodes: array<BVHNode>; 
@group(0) @binding(6) var<storage, read> instances: array<Instance>;
@group(0) @binding(7) var tex: texture_2d_array<f32>;
@group(0) @binding(8) var smp: sampler;
@group(0) @binding(9) var<storage, read> lights: array<LightRef>;
@group(0) @binding(11) var<storage, read> geometry_norm: array<vec4<f32>>;
@group(0) @binding(12) var<storage, read> geometry_uv: array<vec2<f32>>;
@group(0) @binding(13) var g_albedo: texture_2d<f32>;
@group(0) @binding(14) var g_normal: texture_2d<f32>;
@group(0) @binding(15) var g_depth: texture_depth_2d;
@group(0) @binding(16) var<storage, read_write> reservoirsBuffer: array<Reservoir>;
@group(0) @binding(17) var<storage, read> prevReservoirsBuffer: array<Reservoir>;
@group(0) @binding(18) var g_prev_depth: texture_depth_2d;

// =========================================================
//   Buffer Accessors
// =========================================================

fn get_pos(idx: u32) -> vec3<f32> {
    return geometry_pos[idx].xyz;
}

fn get_normal(idx: u32) -> vec3<f32> {
    return geometry_norm[idx].xyz;
}

fn get_uv(idx: u32) -> vec2<f32> {
    return geometry_uv[idx];
}

fn get_transform(inst: Instance) -> mat4x4<f32> {
    return mat4x4<f32>(inst.transform_0, inst.transform_1, inst.transform_2, inst.transform_3);
}

fn get_inv_transform(inst: Instance) -> mat4x4<f32> {
    return mat4x4<f32>(inst.inv_0, inst.inv_1, inst.inv_2, inst.inv_3);
}

// =========================================================
//   Math & RNG Helpers
// =========================================================

fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn init_rng(pixel_idx: u32, frame_count: u32) -> u32 {
    var seed = pixel_idx + frame_count * 719393u;
    seed ^= 2747636419u; seed *= 2654435769u; seed ^= (seed >> 16u);
    seed *= 2654435769u; seed ^= (seed >> 16u); seed *= 2654435769u;
    return seed;
}

fn rand_pcg(state: ptr<function, u32>) -> f32 {
    let old = *state; *state = old * 747796405u + 2891336453u;
    let word = ((*state) >> ((old >> 28u) + 4u)) ^ (*state);
    return f32((word >> 22u) ^ word) / 4294967295.0;
}

fn random_unit_vector(onb: ONB, rng: ptr<function, u32>) -> vec3<f32> {
    let r1 = rand_pcg(rng);
    let r2 = rand_pcg(rng);
    let phi = 2.0 * PI * r1;
    let cos_theta = sqrt(1.0 - r2);
    let sin_theta = sqrt(r2);
    let local_dir = vec3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
    return local_to_world(onb, local_dir);
}

fn random_in_unit_disk(rng: ptr<function, u32>) -> vec3<f32> {
    let r = sqrt(rand_pcg(rng));
    let theta = 2.0 * PI * rand_pcg(rng);
    return vec3<f32>(r * cos(theta), r * sin(theta), 0.0);
}

fn build_onb(n: vec3<f32>) -> ONB {
    let sign = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (sign + n.z);
    let b = n.x * n.y * a;
    let u = vec3(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x);
    let v = vec3(b, sign + n.y * n.y * a, -n.y);
    return ONB(u, v, n);
}

fn local_to_world(onb: ONB, a: vec3<f32>) -> vec3<f32> {
    return a.x * onb.u + a.y * onb.v + a.z * onb.w;
}

// =========================================================
//   BSDF Functions
// =========================================================

fn eval_diffuse(albedo: vec3<f32>) -> vec3<f32> {
    return albedo / PI;
}

fn sample_diffuse(normal: vec3<f32>, albedo: vec3<f32>, rng: ptr<function, u32>) -> ScatterResult {
    let onb = build_onb(normal);
    let dir = random_unit_vector(onb, rng);
    let cos_theta = max(dot(normal, dir), 0.0);
    return ScatterResult(dir, cos_theta / PI, albedo, false);
}

// GGX
fn ggx_d(n_dot_h: f32, a2: f32) -> f32 {
    let d = (n_dot_h * a2 - n_dot_h) * n_dot_h + 1.0;
    return a2 / (PI * d * d);
}

fn ggx_g(n_dot_v: f32, n_dot_l: f32, a2: f32) -> f32 {
    let g1_v = 2.0 * n_dot_v / (n_dot_v + sqrt(a2 + (1.0 - a2) * n_dot_v * n_dot_v));
    let g1_l = 2.0 * n_dot_l / (n_dot_l + sqrt(a2 + (1.0 - a2) * n_dot_l * n_dot_l));
    return g1_v * g1_l;
}

fn pow5(x: f32) -> f32 {
    let x2 = x * x;
    return x2 * x2 * x;
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (1.0 - f0) * pow5(clamp(1.0 - cos_theta, 0.0, 1.0));
}

fn eval_ggx(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, roughness: f32, f0: vec3<f32>) -> vec3<f32> {
    let h = normalize(v + l);
    let n_dot_v = max(dot(n, v), 1e-4);
    let n_dot_l = max(dot(n, l), 1e-4);
    let n_dot_h = max(dot(n, h), 1e-4);
    let v_dot_h = max(dot(v, h), 1e-4);

    let a2 = roughness * roughness;
    let d = ggx_d(n_dot_h, a2);
    let g = ggx_g(n_dot_v, n_dot_l, a2);
    let f = fresnel_schlick(v_dot_h, f0);

    return (d * g * f) / (4.0 * n_dot_v * n_dot_l);
}

fn sample_ggx(n: vec3<f32>, v: vec3<f32>, roughness: f32, f0: vec3<f32>, rng: ptr<function, u32>) -> ScatterResult {
    let a = roughness;
    let u = vec2(rand_pcg(rng), rand_pcg(rng));

    let phi = 2.0 * PI * u.x;
    let cos_theta = sqrt(max(0.0, (1.0 - u.y) / (1.0 + (a * a - 1.0) * u.y)));
    let sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));

    let h_local = vec3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
    let onb = build_onb(n);
    let h = local_to_world(onb, h_local);
    let l = reflect(-v, h);

    if dot(n, l) <= 0.0 {
        return ScatterResult(vec3(0.0), 0.0, vec3(0.0), false);
    }

    let n_dot_v = max(dot(n, v), 1e-4);
    let n_dot_l = max(dot(n, l), 1e-4);
    let n_dot_h = max(dot(n, h), 1e-4);
    let v_dot_h = max(dot(v, h), 1e-4);

    let a2 = a * a;
    let d = ggx_d(n_dot_h, a2);
    let g = ggx_g(n_dot_v, n_dot_l, a2);
    let f = fresnel_schlick(v_dot_h, f0);

    let pdf = (d * n_dot_h) / (4.0 * v_dot_h);
    var throughput = vec3(0.0);
    if pdf > 1e-6 {
        throughput = (g * f * v_dot_h) / (n_dot_v * n_dot_h);
    }
    let treat_as_specular = roughness < 0.01;

    return ScatterResult(l, pdf, throughput, treat_as_specular);
}

fn bsdf_to_throughput(d: f32, g: f32, f: vec3<f32>, n_dot_v: f32, n_dot_l: f32, n_dot_h: f32, v_dot_h: f32, pdf: f32) -> vec3<f32> {
    if pdf <= 0.0 { return vec3(0.0); }
    return (d * g * f) / (4.0 * n_dot_v * n_dot_l) * n_dot_l / pdf;
}



// Dielectric
fn reflectance_dielectric(cosine: f32, ref_idx: f32) -> f32 {
    var r0 = (1.0 - ref_idx) / (1.0 + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0 - r0) * pow5(1.0 - cosine);
}

fn sample_dielectric(dir: vec3<f32>, normal: vec3<f32>, ior: f32, albedo: vec3<f32>, rng: ptr<function, u32>) -> ScatterResult {
    let front_face = dot(dir, normal) < 0.0;
    let refraction_ratio = select(ior, 1.0 / ior, front_face);
    let n = select(-normal, normal, front_face);

    let unit_dir = normalize(dir);
    let cos_theta = min(dot(-unit_dir, n), 1.0);
    let sin_theta = sqrt(1.0 - cos_theta * cos_theta);

    let cannot_refract = refraction_ratio * sin_theta > 1.0;
    var direction: vec3<f32>;

    if cannot_refract || reflectance_dielectric(cos_theta, refraction_ratio) > rand_pcg(rng) {
        direction = reflect(unit_dir, n);
    } else {
        direction = refract(unit_dir, n, refraction_ratio);
    }

    return ScatterResult(direction, 1.0, albedo, true);
}

// =========================================================
//   Direct Light Sampling
// =========================================================

fn sample_light_source(hit_p: vec3<f32>, rng: ptr<function, u32>) -> LightSample {
    let light_count = scene.light_count;
    if light_count == 0u {
        return LightSample(vec3(0.0), vec3(0.0), 0.0, 0.0);
    }

    let light_pick_idx = u32(rand_pcg(rng) * f32(light_count));
    let l_ref = lights[light_pick_idx];

    let tri = topology[l_ref.tri_idx];
    let inst = instances[l_ref.inst_idx];
    let m = get_transform(inst);

    let v0 = (m * vec4(get_pos(tri.v0), 1.0)).xyz;
    let v1 = (m * vec4(get_pos(tri.v1), 1.0)).xyz;
    let v2 = (m * vec4(get_pos(tri.v2), 1.0)).xyz;

    let r1 = rand_pcg(rng);
    let r2 = rand_pcg(rng);
    let sqrt_r1 = sqrt(r1);
    let u = 1.0 - sqrt_r1;
    let v = r2 * sqrt_r1;
    let w = 1.0 - u - v;

    let p = v0 * u + v1 * v + v2 * w;
    let edge1 = v1 - v0;
    let edge2 = v2 - v0;
    let n_raw = normalize(cross(edge1, edge2));
    let area = length(cross(edge1, edge2)) * 0.5;

    let l_dir = p - hit_p;
    let dist_sq = dot(l_dir, l_dir);
    let dist = sqrt(dist_sq);
    let unit_l = l_dir / dist;

    let cos_theta_l = max(dot(n_raw, -unit_l), 0.0);
    if cos_theta_l < 1e-6 || area < 1e-6 {
        return LightSample(vec3(0.0), vec3(0.0), 0.0, 0.0);
    }

    // Albedo if light
    let uv0 = get_uv(tri.v0);
    let uv1 = get_uv(tri.v1);
    let uv2 = get_uv(tri.v2);
    let tex_uv = uv0 * u + uv1 * v + uv2 * w;
    var L = tri.data0.rgb;
    let base_tex = tri.data2.x;
    if base_tex > -0.5 {
        L *= textureSampleLevel(tex, smp, tex_uv, i32(base_tex), 0.0).rgb;
    }

    let pdf = (dist_sq / (cos_theta_l * area)) / f32(light_count);

    return LightSample(L, unit_l, dist, pdf);
}

fn get_light_pdf(origin: vec3<f32>, tri_idx: u32, inst_idx: u32, t: f32, l_dir: vec3<f32>) -> f32 {
    let tri = topology[tri_idx];
    let inst = instances[inst_idx];
    let m = get_transform(inst);

    let v0 = (m * vec4(get_pos(tri.v0), 1.0)).xyz;
    let v1 = (m * vec4(get_pos(tri.v1), 1.0)).xyz;
    let v2 = (m * vec4(get_pos(tri.v2), 1.0)).xyz;

    let edge1 = v1 - v0;
    let edge2 = v2 - v0;
    let area = length(cross(edge1, edge2)) * 0.5;
    let normal = normalize(cross(edge1, edge2));

    let cos_theta_l = max(dot(normal, -l_dir), 0.0);
    if cos_theta_l < 1e-4 || area < 1e-6 { return 0.0; }

    let light_count = scene.light_count;
    let dist_sq = t * t;
    return (dist_sq / (cos_theta_l * area)) / f32(light_count);
}

fn power_heuristic(pdf_a: f32, pdf_b: f32) -> f32 {
    let a2 = pdf_a * pdf_a;
    let b2 = pdf_b * pdf_b;
    return a2 / (a2 + b2 + 1e-6);
}

// =========================================================
//   Intersection Functions
// =========================================================

fn intersect_aabb(min_b: vec3<f32>, max_b: vec3<f32>, r: Ray, t_min: f32, t_max: f32) -> f32 {
    let t1 = min_b * r.inv_d - r.origin_inv_d;
    let t2 = max_b * r.inv_d - r.origin_inv_d;
    let t_near = min(t1, t2);
    let t_far = max(t1, t2);
    let tm_near = max(t_min, max(t_near.x, max(t_near.y, t_near.z)));
    let tm_far = min(t_max, min(t_far.x, min(t_far.y, t_far.z)));
    return select(T_MAX, tm_near, tm_near <= tm_far);
}

fn hit_triangle_raw(v0: vec3<f32>, v1: vec3<f32>, v2: vec3<f32>, r: Ray, t_min: f32, t_max: f32) -> f32 {
    let e1 = v1 - v0; let e2 = v2 - v0;
    let h = cross(r.direction, e2); let a = dot(e1, h);
    if abs(a) < 1e-6 { return -1.0; } // Increased epsilon
    let f = 1.0 / a; let s = r.origin - v0; let u = f * dot(s, h);
    if u < 0.0 || u > 1.0 { return -1.0; }
    let q = cross(s, e1); let v = f * dot(r.direction, q);
    if v < 0.0 || u + v > 1.0 { return -1.0; }
    let t = f * dot(e2, q);
    return select(-1.0, t, t > t_min && t < t_max);
}

fn intersect_blas(r: Ray, t_min: f32, t_max: f32, node_start_idx: u32) -> vec2<f32> {
    var closest_t = t_max;
    var hit_idx = -1.0;
    
    let end_node = node_start_idx + bitcast<u32>(nodes[node_start_idx].min_b.w);
    var curr = node_start_idx;
    
    while curr < end_node {
        let node = nodes[curr];
        
        var hit_t = closest_t;
        let t_aabb = intersect_aabb(node.min_b.xyz, node.max_b.xyz, r, t_min, closest_t);
        
        if t_aabb < T_MAX {
            let data = bitcast<u32>(node.max_b.w);
            if data != 0u {
                // Leaf node
                let first = data >> 3u;
                let count = data & 7u;
                for (var i = 0u; i < count; i++) {
                    let tri_id = first + i;
                    let tr = topology[tri_id];
                    let t = hit_triangle_raw(get_pos(tr.v0), get_pos(tr.v1), get_pos(tr.v2), r, t_min, closest_t);
                    if t > 0.0 { 
                        closest_t = t; 
                        hit_idx = f32(tri_id); 
                    }
                }
                curr = node_start_idx + bitcast<u32>(node.min_b.w);
            } else {
                // Internal node
                curr = curr + 1u;
            }
        } else {
            // Missed AABB
            curr = node_start_idx + bitcast<u32>(node.min_b.w);
        }
    }
    return vec2<f32>(closest_t, hit_idx);
}

fn intersect_tlas(r: Ray, t_min: f32, t_max: f32) -> HitResult {
    var res: HitResult; res.t = t_max; res.tri_idx = -1.0; res.inst_idx = -1;
    if scene.blas_base_idx == 0u { return res; }

    var curr = 0u;
    let end_node = bitcast<u32>(nodes[0].min_b.w);

    while curr < end_node {
        let node = nodes[curr];
        
        if intersect_aabb(node.min_b.xyz, node.max_b.xyz, r, t_min, res.t) < T_MAX {
            let data = bitcast<u32>(node.max_b.w);
            if data != 0u {
                // Leaf
                let inst_idx = data >> 3u;
                let inst = instances[inst_idx];
                let r_local = make_ray((get_inv_transform(inst) * vec4(r.origin, 1.0)).xyz, (get_inv_transform(inst) * vec4(r.direction, 0.0)).xyz);
                let blas = intersect_blas(r_local, t_min, res.t, scene.blas_base_idx + inst.blas_node_offset);
                if blas.y > -0.5 { 
                    res.t = blas.x; 
                    res.tri_idx = blas.y; 
                    res.inst_idx = i32(inst_idx); 
                }
                curr = bitcast<u32>(node.min_b.w);
            } else {
                curr = curr + 1u;
            }
        } else {
            curr = bitcast<u32>(node.min_b.w);
        }
    }
    return res;
}

// shadow ray\u7248
// \u30B7\u30E3\u30C9\u30A6\u30EC\u30A4\u7528\u306EBLAS\u4EA4\u5DEE\u5224\u5B9A\uFF08\u30D2\u30C3\u30C8\u3057\u305F\u3089\u5373true\u3092\u8FD4\u3059\uFF09
fn intersect_blas_shadow(r: Ray, t_min: f32, t_max: f32, node_start_idx: u32) -> bool {
    let end_node = node_start_idx + bitcast<u32>(nodes[node_start_idx].min_b.w);
    var curr = node_start_idx;
    
    while curr < end_node {
        let node = nodes[curr];
        
        let t_aabb = intersect_aabb(node.min_b.xyz, node.max_b.xyz, r, t_min, t_max);
        if t_aabb < T_MAX {
            let data = bitcast<u32>(node.max_b.w);
            if data != 0u {
                // Leaf node
                let first = data >> 3u;
                let count = data & 7u;
                for (var i = 0u; i < count; i++) {
                    let tri_id = first + i;
                    let tr = topology[tri_id];
                    let t = hit_triangle_raw(get_pos(tr.v0), get_pos(tr.v1), get_pos(tr.v2), r, t_min, t_max);
                    if t > 0.0 { return true; }
                }
                curr = node_start_idx + bitcast<u32>(node.min_b.w);
            } else {
                // Internal node
                curr = curr + 1u;
            }
        } else {
            // Missed AABB
            curr = node_start_idx + bitcast<u32>(node.min_b.w);
        }
    }
    return false;
}

// \u30B7\u30E3\u30C9\u30A6\u30EC\u30A4\u7528\u306ETLAS\u4EA4\u5DEE\u5224\u5B9A
fn intersect_tlas_shadow(r: Ray, t_min: f32, t_max: f32) -> bool {
    if scene.blas_base_idx == 0u { return false; }

    var curr = 0u;
    let end_node = bitcast<u32>(nodes[0].min_b.w);

    while curr < end_node {
        let node = nodes[curr];
        
        let t_aabb = intersect_aabb(node.min_b.xyz, node.max_b.xyz, r, t_min, t_max);
        if t_aabb < T_MAX {
            let data = bitcast<u32>(node.max_b.w);
            if data != 0u {
                // Leaf
                let inst_idx = data >> 3u;
                let inst = instances[inst_idx];
                
                let r_local = make_ray(
                    (get_inv_transform(inst) * vec4(r.origin, 1.0)).xyz, 
                    (get_inv_transform(inst) * vec4(r.direction, 0.0)).xyz
                );
                
                if intersect_blas_shadow(r_local, t_min, t_max, scene.blas_base_idx + inst.blas_node_offset) {
                    return true;
                }
                curr = bitcast<u32>(node.min_b.w);
            } else {
                curr = curr + 1u;
            }
        } else {
            curr = bitcast<u32>(node.min_b.w);
        }
    }
    return false;
}

fn get_throughput(w_o: vec3<f32>, w_i: vec3<f32>, normal: vec3<f32>, mat_type: u32, roughness: f32, f0: vec3<f32>, albedo: vec3<f32>) -> vec3<f32> {
    if mat_type == 0u {
        return albedo;
    } else if mat_type == 1u {
        let h = normalize(w_o + w_i);
        let n_dot_v = max(dot(normal, w_o), 1e-4);
        let n_dot_l = max(dot(normal, w_i), 1e-4);
        let n_dot_h = max(dot(normal, h), 1e-4);
        let v_dot_h = max(dot(w_o, h), 1e-4);
        let a2 = roughness * roughness;
        let g = ggx_g(n_dot_v, n_dot_l, a2);
        let f = fresnel_schlick(v_dot_h, f0);
        return (g * f * v_dot_h) / (n_dot_v * n_dot_h);
    } else { // mat_type == 2u
        return albedo;
    }
}

fn eval_brdf_cos(w_o: vec3<f32>, w_i: vec3<f32>, normal: vec3<f32>, mat_type: u32, roughness: f32, f0: vec3<f32>, albedo: vec3<f32>) -> vec3<f32> {
    let n_dot_l = max(dot(normal, w_i), 1e-4);

    if mat_type == 0u {
        return (albedo / PI) * n_dot_l;
    } else if mat_type == 1u {
        let h = normalize(w_o + w_i);
        let n_dot_v = max(dot(normal, w_o), 1e-4);
        let n_dot_h = max(dot(normal, h), 1e-4);
        let v_dot_h = max(dot(w_o, h), 1e-4);
        let a2 = roughness * roughness;
        let d = ggx_d(n_dot_h, a2);
        let g = ggx_g(n_dot_v, n_dot_l, a2);
        let f = fresnel_schlick(v_dot_h, f0);
        return (d * g * f) / (4.0 * n_dot_v);
    } else { 
        return vec3<f32>(0.0);
    }
}

fn update_reservoir(r: ptr<function, Reservoir>, s: Sample, weight: f32, rng: ptr<function, u32>) {
    r.w_sum += weight;
    if rand_pcg(rng) < (weight / r.w_sum) {
        r.sample = s;
    }
}

fn get_world_pos(id: vec2<u32>, depth_val: f32) -> vec3<f32> {
    let u_cam = (f32(id.x) + 0.5 + scene.jitter.x * f32(scene.width)) / f32(scene.width);
    let v_cam = 1.0 - (f32(id.y) + 0.5 + scene.jitter.y * f32(scene.height)) / f32(scene.height);
    let ray_dir = normalize(scene.camera.lower_left_corner.xyz + u_cam * scene.camera.horizontal.xyz + v_cam * scene.camera.vertical.xyz - scene.camera.origin.xyz);
    
    // Reverse non-linear Z to view-space Z
    let z_near = 0.001;
    let z_far = 10000.0;
    let z_view = (z_far * z_near) / (z_far - depth_val * (z_far - z_near));
    
    // View-space Z to ray distance t
    let eye = scene.camera.origin.xyz;
    let center = scene.camera.lower_left_corner.xyz + scene.camera.horizontal.xyz * 0.5 + scene.camera.vertical.xyz * 0.5;
    let forward = normalize(center - eye);
    let t = z_view / dot(ray_dir, forward);
    
    return eye + ray_dir * t;
}

fn get_prev_world_pos(id: vec2<u32>, depth_val: f32) -> vec3<f32> {
    let u_cam = (f32(id.x) + 0.5 + scene.prev_jitter.x * f32(scene.width)) / f32(scene.width);
    let v_cam = 1.0 - (f32(id.y) + 0.5 + scene.prev_jitter.y * f32(scene.height)) / f32(scene.height);
    let ray_dir = normalize(scene.prev_camera.lower_left_corner.xyz + u_cam * scene.prev_camera.horizontal.xyz + v_cam * scene.prev_camera.vertical.xyz - scene.prev_camera.origin.xyz);
    
    // Reverse non-linear Z to view-space Z
    let z_near = 0.001;
    let z_far = 10000.0;
    let z_view = (z_far * z_near) / (z_far - depth_val * (z_far - z_near));
    
    // View-space Z to ray distance t
    let eye = scene.prev_camera.origin.xyz;
    let center = scene.prev_camera.lower_left_corner.xyz + scene.prev_camera.horizontal.xyz * 0.5 + scene.prev_camera.vertical.xyz * 0.5;
    let forward = normalize(center - eye);
    let t = z_view / dot(ray_dir, forward);
    
    return eye + ray_dir * t;
}

@compute @workgroup_size(8, 8)
fn temporal_reuse(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= scene.width || id.y >= scene.height { return; }
    let p_idx = id.y * scene.width + id.x;
    var rng = init_rng(p_idx, scene.frame_count + 1000u);

    let g_normal_val = textureLoad(g_normal, id.xy, 0);
    let depth_val = textureLoad(g_depth, id.xy, 0);

    // Current candidate from InitialSampling (already has M=1, w_sum=p_hat/q)
    var r_curr = reservoirsBuffer[p_idx];
    
    if depth_val >= 1.0 {
        r_curr.W = 0.0;
        reservoirsBuffer[p_idx] = r_curr;
        return;
    }

    var tri_idx = bitcast<u32>(g_normal_val.z);
    var tri = topology[tri_idx];
    let mat_type = u32(tri.data0.w + 0.5);
    var normal = unpack_normal(g_normal_val.xy);
    
    let u_cam = (f32(id.x) + 0.5 + scene.jitter.x * f32(scene.width)) / f32(scene.width);
    let v_cam = 1. - (f32(id.y) + 0.5 + scene.jitter.y * f32(scene.height)) / f32(scene.height);
    let dir = normalize(scene.camera.lower_left_corner.xyz + u_cam * scene.camera.horizontal.xyz + v_cam * scene.camera.vertical.xyz - scene.camera.origin.xyz);
    let w_o = -dir;

    normal = select(-normal, normal, dot(w_o, normal) > 0.0);
    
    let albedo = textureLoad(g_albedo, id.xy, 0).rgb;
    let metallic = tri.data1.x;
    let roughness = max(tri.data1.y, 0.005);
    let f0 = mix(vec3(0.04), albedo, metallic);

    let is_delta = (mat_type == 2u) || (mat_type == 1u && metallic > 0.9 && roughness < 0.01);

    // Re-evaluate current reservoir p_hat using BRDF
    let w_i_curr = vec3(r_curr.sample.hit_p.w, r_curr.sample.normal.w, r_curr.sample.radiance.w);
    var p_hat_curr = 0.0;
    if is_delta {
        p_hat_curr = luminance(r_curr.sample.radiance.xyz);
    } else {
        let brdf_curr = eval_brdf_cos(w_o, w_i_curr, normal, mat_type, roughness, f0, albedo);
        p_hat_curr = luminance(r_curr.sample.radiance.xyz * brdf_curr);
    }
    r_curr.w_sum = r_curr.W * f32(r_curr.M) * p_hat_curr;

    // Reconstruct world position of current pixel
    let world_pos = get_world_pos(id.xy, depth_val);

    // Project world position to previous frame screen
    let W_vec = world_pos - scene.prev_camera.origin.xyz;
    let H_vec = scene.prev_camera.horizontal.xyz;
    let V_vec = scene.prev_camera.vertical.xyz;
    let L_vec = scene.prev_camera.lower_left_corner.xyz - scene.prev_camera.origin.xyz;
    let w_vec = cross(H_vec, V_vec);
    
    let T_val = dot(W_vec, w_vec) / dot(L_vec, w_vec);
    let u_prev = dot(W_vec - T_val * L_vec, H_vec) / (T_val * dot(H_vec, H_vec));
    let v_prev = dot(W_vec - T_val * L_vec, V_vec) / (T_val * dot(V_vec, V_vec));

    let prev_x = i32(u_prev * f32(scene.width));
    let prev_y = i32((1.0 - v_prev) * f32(scene.height));

    // Previous frame reservoir
    var r_prev: Reservoir;
    var disoccluded = false;
    if prev_x >= 0 && prev_x < i32(scene.width) && prev_y >= 0 && prev_y < i32(scene.height) {
        let prev_p_idx = u32(prev_y) * scene.width + u32(prev_x);
        r_prev = prevReservoirsBuffer[prev_p_idx];
        
        let prev_depth_val = textureLoad(g_prev_depth, vec2<i32>(prev_x, prev_y), 0);
        if prev_depth_val >= 1.0 {
            disoccluded = true;
        } else {
            let prev_world_pos = get_prev_world_pos(vec2<u32>(u32(prev_x), u32(prev_y)), prev_depth_val);
            if distance(world_pos, prev_world_pos) > 0.1 {
                disoccluded = true;
            }
        }
    } else {
        disoccluded = true;
    }

    if disoccluded {
        r_prev.M = 0u;
        r_prev.w_sum = 0.0;
    }

    // Limit history M to prevent excessive ghosting and bias.
    // Use a smaller cap for metals to keep them responsive.
    let max_M = select(20u, 20u, metallic > 0.5);
    if r_prev.M > max_M {
        let scale = f32(max_M) / f32(r_prev.M);
        r_prev.w_sum *= scale;
        r_prev.M = max_M;
    }

    // Re-evaluate previous sample's p_hat at current pixel
    var p_hat_prev = 0.0;
    let w_i_prev = vec3(r_prev.sample.hit_p.w, r_prev.sample.normal.w, r_prev.sample.radiance.w);
    if is_delta {
        p_hat_prev = luminance(r_prev.sample.radiance.xyz);
    } else {
        let brdf_prev = eval_brdf_cos(w_o, w_i_prev, normal, mat_type, roughness, f0, albedo);
        p_hat_prev = luminance(r_prev.sample.radiance.xyz * brdf_prev);
    }

    if p_hat_prev > 1e-6 {
        let weight_prev = p_hat_prev * r_prev.W * f32(r_prev.M);
        update_reservoir(&r_curr, r_prev.sample, weight_prev, &rng);
    }
    r_curr.M += r_prev.M;

    // Resolve final W weight
    var p_hat_final = 0.0;
    let w_i_final = vec3(r_curr.sample.hit_p.w, r_curr.sample.normal.w, r_curr.sample.radiance.w);
    if is_delta {
        p_hat_final = luminance(r_curr.sample.radiance.xyz);
    } else {
        let brdf_final = eval_brdf_cos(w_o, w_i_final, normal, mat_type, roughness, f0, albedo);
        p_hat_final = luminance(r_curr.sample.radiance.xyz * brdf_final);
    }

    if p_hat_final > 1e-6 {
        r_curr.W = r_curr.w_sum / (f32(r_curr.M) * p_hat_final);
    } else {
        r_curr.W = 0.0;
    }
    
    // Hard clamp W to suppress fireflies
    // r_curr.W = min(r_curr.W, 1000.0);

    reservoirsBuffer[p_idx] = r_curr;
}
`, j = `struct Camera {
    origin: vec4<f32>,
    lower_left_corner: vec4<f32>,
    horizontal: vec4<f32>,
    vertical: vec4<f32>,
    u: vec4<f32>,
    v: vec4<f32>,
}

struct SceneUniforms {
    camera: Camera,
    prev_camera: Camera,
    frame_count: u32,
    blas_base_idx: u32,
    vertex_count: u32,
    rand_seed: u32,
    light_count: u32,
    width: u32,
    height: u32,
    pad: u32,
    jitter: vec2<f32>,
    average_jitter: vec2<f32>,
    prev_jitter: vec2<f32>,
    pad2: vec2<f32>
}

struct Sample {
    hit_p: vec4<f32>,    // xyz: pos, w: dir.x
    normal: vec4<f32>,   // xyz: normal, w: dir.y
    radiance: vec4<f32>, // xyz: radiance, w: dir.z
}

struct Reservoir {
    sample: Sample,
    w_sum: f32,
    W: f32,
    M: u32,
    padding: f32,
}

struct MeshTopology {
    v0: u32,
    v1: u32,
    v2: u32,
    pad: u32,
    data0: vec4<f32>, // rgb: BaseColor, w: MaterialType (cast)
    data1: vec4<f32>, // x: Metallic, y: Roughness, z: IOR, w: 0.0
    data2: vec4<f32>, // x: BaseTex, y: MetRoughTex, z: NormalTex, w: EmissiveTex
    data3: vec4<f32>, // rgb: Emissive, w: 0.0
}

struct BVHNode {
    min_b: vec4<f32>, // w: skip_pointer
    max_b: vec4<f32>, // w: data (internal: 0, leaf: (left_first << 3) | tri_count)
}

struct Instance {
    transform_0: vec4<f32>,
    transform_1: vec4<f32>,
    transform_2: vec4<f32>,
    transform_3: vec4<f32>,
    inv_0: vec4<f32>,
    inv_1: vec4<f32>,
    inv_2: vec4<f32>,
    inv_3: vec4<f32>,
    blas_node_offset: u32,
    attr_offset: u32,
    instance_id: u32,
    padding: u32,
}

struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
    inv_d: vec3<f32>,
    origin_inv_d: vec3<f32>
}

fn make_ray(origin: vec3<f32>, direction: vec3<f32>) -> Ray {
    let inv_d = 1.0 / direction;
    return Ray(origin, direction, inv_d, origin * inv_d);
}

@group(0) @binding(2) var<uniform> scene : SceneUniforms;
@group(0) @binding(3) var<storage, read> geometry_pos : array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> topology : array<MeshTopology>;
@group(0) @binding(5) var<storage, read> nodes : array<BVHNode>; 
@group(0) @binding(6) var<storage, read> instances : array<Instance>;
@group(0) @binding(14) var g_normal : texture_2d<f32>;
@group(0) @binding(15) var g_depth : texture_depth_2d;
@group(0) @binding(16) var<storage, read_write> reservoirsBuffer : array<Reservoir>;
@group(0) @binding(17) var<storage, read_write> spatialReservoirsBuffer : array<Reservoir>;

fn get_pos(idx: u32) -> vec3<f32> {
    return geometry_pos[idx].xyz;
}

fn get_inv_transform(inst: Instance) -> mat4x4<f32> {
    return mat4x4<f32>(inst.inv_0, inst.inv_1, inst.inv_2, inst.inv_3);
}

fn intersect_aabb(min_b: vec3<f32>, max_b: vec3<f32>, r: Ray, t_min: f32, t_max: f32) -> f32 {
    let t1 = min_b * r.inv_d - r.origin_inv_d;
    let t2 = max_b * r.inv_d - r.origin_inv_d;
    let t_near = min(t1, t2);
    let t_far = max(t1, t2);
    let tm_near = max(t_min, max(t_near.x, max(t_near.y, t_near.z)));
    let tm_far = min(t_max, min(t_far.x, min(t_far.y, t_far.z)));
    return select(1e30, tm_near, tm_near <= tm_far);
}

fn hit_triangle_raw(v0: vec3<f32>, v1: vec3<f32>, v2: vec3<f32>, r: Ray, t_min: f32, t_max: f32) -> f32 {
    let e1 = v1 - v0; let e2 = v2 - v0;
    let h = cross(r.direction, e2); let a = dot(e1, h);
    if abs(a) < 1e-6 { return -1.0; } 
    let f = 1.0 / a; let s = r.origin - v0; let u = f * dot(s, h);
    if u < 0.0 || u > 1.0 { return -1.0; }
    let q = cross(s, e1); let v = f * dot(r.direction, q);
    if v < 0.0 || u + v > 1.0 { return -1.0; }
    let t = f * dot(e2, q);
    return select(-1.0, t, t > t_min && t < t_max);
}

fn intersect_blas_shadow(r: Ray, t_min: f32, t_max: f32, node_start_idx: u32) -> bool {
    let end_node = node_start_idx + bitcast<u32>(nodes[node_start_idx].min_b.w);
    var curr = node_start_idx;
    while curr < end_node {
        let node = nodes[curr];
        if intersect_aabb(node.min_b.xyz, node.max_b.xyz, r, t_min, t_max) < 1e30 {
            let data = bitcast<u32>(node.max_b.w);
            if data != 0u {
                let first = data >> 3u;
                let count = data & 7u;
                for (var i = 0u; i < count; i++) {
                    let tr = topology[first + i];
                    if hit_triangle_raw(get_pos(tr.v0), get_pos(tr.v1), get_pos(tr.v2), r, t_min, t_max) > 0.0 { return true; }
                }
                curr = node_start_idx + bitcast<u32>(node.min_b.w);
            } else { curr = curr + 1u; }
        } else { curr = node_start_idx + bitcast<u32>(node.min_b.w); }
    }
    return false;
}

fn intersect_tlas_shadow(r: Ray, t_min: f32, t_max: f32) -> bool {
    if scene.blas_base_idx == 0u { return false; }
    var curr = 0u;
    let end_node = bitcast<u32>(nodes[0].min_b.w);
    while curr < end_node {
        let node = nodes[curr];
        if intersect_aabb(node.min_b.xyz, node.max_b.xyz, r, t_min, t_max) < 1e30 {
            let data = bitcast<u32>(node.max_b.w);
            if data != 0u {
                let inst = instances[data >> 3u];
                let r_local = make_ray((get_inv_transform(inst) * vec4(r.origin, 1.0)).xyz, (get_inv_transform(inst) * vec4(r.direction, 0.0)).xyz);
                if intersect_blas_shadow(r_local, t_min, t_max, scene.blas_base_idx + inst.blas_node_offset) { return true; }
                curr = bitcast<u32>(node.min_b.w);
            } else { curr = curr + 1u; }
        } else { curr = bitcast<u32>(node.min_b.w); }
    }
    return false;
}

const PI: f32 = 3.14159265359;

// =========================================================
//   Math & Helpers
// =========================================================

fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn init_rng(pixel_idx: u32, frame_count: u32) -> u32 {
    var seed = pixel_idx + frame_count * 719393u;
    seed ^= 2747636419u; seed *= 2654435769u; seed ^= (seed >> 16u);
    seed *= 2654435769u; seed ^= (seed >> 16u); seed *= 2654435769u;
    return seed;
}

fn rand_pcg(rng: ptr<function, u32>) -> f32 {
    let state = *rng;
    *rng = state * 747796405u + 2891336453u;
    var word: u32 = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    word = (word >> 22u) ^ word;
    return f32(word) / 4294967296.0;
}

fn unpack_normal(p: vec2<f32>) -> vec3<f32> {
    var n = vec3(p, 1.0 - abs(p.x) - abs(p.y));
    let t = saturate(-n.z);
    n.x += select(t, -t, n.x >= 0.0);
    n.y += select(t, -t, n.y >= 0.0);
    return normalize(n);
}

// =========================================================
//   BRDF Helpers
// =========================================================

fn ggx_d(n_dot_h: f32, a2: f32) -> f32 {
    let d = (n_dot_h * a2 - n_dot_h) * n_dot_h + 1.0;
    return a2 / (PI * d * d);
}

fn ggx_g(n_dot_v: f32, n_dot_l: f32, a2: f32) -> f32 {
    let g_v = n_dot_v + sqrt((-n_dot_v * a2 + n_dot_v) * n_dot_v + a2);
    let g_l = n_dot_l + sqrt((-n_dot_l * a2 + n_dot_l) * n_dot_l + a2);
    return 2.0 * n_dot_v * n_dot_l / (g_v * g_l);
}

fn fresnel_schlick(v_dot_h: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (1.0 - f0) * pow(1.0 - v_dot_h, 5.0);
}

fn eval_brdf_cos(w_o: vec3<f32>, w_i: vec3<f32>, normal: vec3<f32>, mat_type: u32, roughness: f32, f0: vec3<f32>, albedo: vec3<f32>) -> vec3<f32> {
    let n_dot_l = max(dot(normal, w_i), 1e-4);
    let n_dot_v = max(dot(normal, w_o), 1e-4);

    if mat_type == 0u {
        return (albedo / PI) * n_dot_l;
    } else if mat_type == 1u {
        let h = normalize(w_o + w_i);
        let n_dot_h = max(dot(normal, h), 1e-4);
        let v_dot_h = max(dot(w_o, h), 1e-4);
        let a2 = roughness * roughness;
        let d = ggx_d(n_dot_h, a2);
        let g = ggx_g(n_dot_v, n_dot_l, a2);
        let f = fresnel_schlick(v_dot_h, f0);
        return (d * g * f) / (4.0 * n_dot_v);
    } else { 
        return vec3<f32>(0.0);
    }
}

// =========================================================
//   ReSTIR Logic
// =========================================================

fn update_reservoir(r: ptr<function, Reservoir>, s: Sample, weight: f32, rng: ptr<function, u32>) {
    if weight <= 0.0 { return; }
    r.w_sum += weight;
    if rand_pcg(rng) < (weight / r.w_sum) {
        r.sample = s;
    }
}

fn get_world_pos(id: vec2<u32>, depth_val: f32) -> vec3<f32> {
    let u_cam = (f32(id.x) + 0.5 + scene.jitter.x * f32(scene.width)) / f32(scene.width);
    let v_cam = 1.0 - (f32(id.y) + 0.5 + scene.jitter.y * f32(scene.height)) / f32(scene.height);
    let ray_dir = normalize(scene.camera.lower_left_corner.xyz + u_cam * scene.camera.horizontal.xyz + v_cam * scene.camera.vertical.xyz - scene.camera.origin.xyz);
    
    // Reverse non-linear Z to view-space Z
    let z_near = 0.001;
    let z_far = 10000.0;
    let z_view = (z_far * z_near) / (z_far - depth_val * (z_far - z_near));
    
    // View-space Z to ray distance t
    let eye = scene.camera.origin.xyz;
    let center = scene.camera.lower_left_corner.xyz + scene.camera.horizontal.xyz * 0.5 + scene.camera.vertical.xyz * 0.5;
    let forward = normalize(center - eye);
    let t = z_view / dot(ray_dir, forward);
    
    return eye + ray_dir * t;
}

@compute @workgroup_size(8, 8)
fn spatial_reuse(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= scene.width || id.y >= scene.height { return; }
    let p_idx = id.y * scene.width + id.x;
    var rng = init_rng(p_idx, scene.frame_count + 2000u);

    let g_normal_val = textureLoad(g_normal, id.xy, 0);
    let depth_val = textureLoad(g_depth, id.xy, 0);
    if depth_val >= 1.0 { 
        spatialReservoirsBuffer[p_idx] = reservoirsBuffer[p_idx];
        return; 
    }

    let tri_idx = bitcast<u32>(g_normal_val.z);
    let tri = topology[tri_idx];
    let mat_type = u32(tri.data0.w + 0.5);
    let albedo = tri.data0.rgb;
    let metallic = tri.data1.x;
    let roughness = max(tri.data1.y, 0.005);
    let f0 = mix(vec3(0.04), albedo, metallic);

    let u_cam = (f32(id.x) + 0.5 + scene.jitter.x * f32(scene.width)) / f32(scene.width);
    let v_cam = 1.0 - (f32(id.y) + 0.5 + scene.jitter.y * f32(scene.height)) / f32(scene.height);
    let ray_dir = normalize(scene.camera.lower_left_corner.xyz + u_cam * scene.camera.horizontal.xyz + v_cam * scene.camera.vertical.xyz - scene.camera.origin.xyz);
    let w_o = -ray_dir;

    var normal = unpack_normal(g_normal_val.xy);
    normal = select(-normal, normal, dot(w_o, normal) > 0.0);
    
    // Reconstruct world hit point
    let curr_hit_p = get_world_pos(id.xy, depth_val);

    let is_delta = (mat_type == 2u) || (mat_type == 1u && metallic > 0.9 && roughness < 0.01);

    if is_delta {
        spatialReservoirsBuffer[p_idx] = reservoirsBuffer[p_idx];
        return;
    }

    var r_curr = reservoirsBuffer[p_idx];
    
    // Initialize w_sum for the merge
    let w_i_curr = vec3(r_curr.sample.hit_p.w, r_curr.sample.normal.w, r_curr.sample.radiance.w);
    var p_hat_curr = 0.0;
    let brdf_curr = eval_brdf_cos(w_o, w_i_curr, normal, mat_type, roughness, f0, albedo);
    p_hat_curr = luminance(r_curr.sample.radiance.xyz * brdf_curr);
    r_curr.w_sum = r_curr.W * f32(r_curr.M) * p_hat_curr;

    // Spatial Reuse Parameters
    const num_neighbors: u32 = 4u;
    let base_radius = 20.0;
    var dynamic_radius = mix(1.0, 10.0, roughness);
    dynamic_radius = select(base_radius, dynamic_radius,  mat_type == 1u);


    for (var i = 0u; i < num_neighbors; i++) {
        let angle = rand_pcg(&rng) * 2.0 * PI;
        let dist = rand_pcg(&rng) * dynamic_radius;
        let offset = vec2<i32>(i32(cos(angle) * dist), i32(sin(angle) * dist));
        let neighbor_coord = vec2<i32>(id.xy) + offset;

        if neighbor_coord.x < 0 || neighbor_coord.x >= i32(scene.width) ||
           neighbor_coord.y < 0 || neighbor_coord.y >= i32(scene.height) {
            continue;
        }

        let n_idx = u32(neighbor_coord.y) * scene.width + u32(neighbor_coord.x);
        let neighbor_normal_val = textureLoad(g_normal, neighbor_coord, 0);
        let neighbor_depth = textureLoad(g_depth, neighbor_coord, 0);
        let neighbor_normal = unpack_normal(neighbor_normal_val.xy);
        
        if dot(normal, neighbor_normal) < 0.9 || abs(depth_val - neighbor_depth) > 0.1 * depth_val {
            continue;
        }

        let r_neighbor = reservoirsBuffer[n_idx];
        let light_p = r_neighbor.sample.hit_p.xyz;
        let light_n = r_neighbor.sample.normal.xyz;
        
        // Reconnection Shift: New direction from current point to neighbor's light sample
        let v_curr = light_p - curr_hit_p;
        let dist_curr2 = dot(v_curr, v_curr);
        let dist_curr = sqrt(dist_curr2);
        let w_i_new = v_curr / dist_curr;

        // Jacobian calculation
        // J = (|cos_theta_L'| * dist_neighbor^2) / (|cos_theta_L| * dist_curr^2)
        let neighbor_hit_p = get_world_pos(vec2<u32>(neighbor_coord), neighbor_depth);
        
        let v_neighbor = light_p - neighbor_hit_p;
        let dist_neighbor2 = dot(v_neighbor, v_neighbor);
        let w_i_old = v_neighbor / sqrt(dist_neighbor2);

        let cos_L_curr = max(dot(light_n, -w_i_new), 0.0);
        let cos_L_old = max(dot(light_n, -w_i_old), 0.0);
        
        var jacobian = 1.0;
        if cos_L_old > 1e-6 {
            jacobian = (cos_L_curr * dist_neighbor2) / (cos_L_old * dist_curr2);
        }
        // jacobian = clamp(jacobian, 0.1, 10.0);

        // Visibility Check: Shadow ray from current hit point to neighbor's light position
        let shadow_ray = make_ray(curr_hit_p + normal * 1e-4, w_i_new);
        if intersect_tlas_shadow(shadow_ray, 0.001, dist_curr - 2e-4) {
            continue;
        }

        var p_hat_new = 0.0;
        let brdf_new = eval_brdf_cos(w_o, w_i_new, normal, mat_type, roughness, f0, albedo);
        p_hat_new = luminance(r_neighbor.sample.radiance.xyz * brdf_new);

        if p_hat_new > 1e-6 {
            let weight = p_hat_new * r_neighbor.W * f32(r_neighbor.M) * jacobian;
            var shifted_sample = r_neighbor.sample;
            shifted_sample.hit_p.w = w_i_new.x;
            shifted_sample.normal.w = w_i_new.y;
            shifted_sample.radiance.w = w_i_new.z;
            
            update_reservoir(&r_curr, shifted_sample, weight, &rng);
            r_curr.M += r_neighbor.M;
        }
    }

    let w_i_final = vec3(r_curr.sample.hit_p.w, r_curr.sample.normal.w, r_curr.sample.radiance.w);
    var p_hat_final = 0.0;
    let brdf_final = eval_brdf_cos(w_o, w_i_final, normal, mat_type, roughness, f0, albedo);
    p_hat_final = luminance(r_curr.sample.radiance.xyz * brdf_final);

    if p_hat_final > 1e-6 {
        r_curr.W = r_curr.w_sum / (f32(r_curr.M) * p_hat_final);
    } else {
        r_curr.W = 0.0;
    }

    // r_curr.W = min(r_curr.W, 1000.0);
    spatialReservoirsBuffer[p_idx] = r_curr;
}
`, $ = `// =========================================================
//   WebGPU Ray Tracer (Raytracer.wgsl)
// =========================================================

const PI = 3.141592653589793;
const T_MIN = 0.001;
const T_MAX = 1e30;
override MAX_DEPTH: u32;

// =========================================================
//   Structs
// =========================================================

struct Sample {
    hit_p: vec4<f32>,    // xyz: Position(secondary ray hit point), w: scatter.dir.x
    normal: vec4<f32>,   // xyz: Normal, w: scatter.dir.y
    radiance: vec4<f32>, // xyz: L_i, w: scatter.dir.z
}

struct Reservoir {
    sample: Sample,
    w_sum: f32,
    W: f32,
    M: u32,
    padding: f32,
}


struct Camera {
    origin: vec4<f32>, // w: lens_radius
    lower_left_corner: vec4<f32>,
    horizontal: vec4<f32>,
    vertical: vec4<f32>,
    u: vec4<f32>,
    v: vec4<f32>
}

struct SceneUniforms {
    camera: Camera,
    prev_camera: Camera,
    frame_count: u32,
    blas_base_idx: u32,
    vertex_count: u32,
    rand_seed: u32,
    light_count: u32,
    width: u32,
    height: u32,
    pad: u32,
    jitter: vec2<f32>,
    average_jitter: vec2<f32>,
    prev_jitter: vec2<f32>,
    pad2: vec2<f32>
}

struct MeshTopology {
    v0: u32,
    v1: u32,
    v2: u32,
    pad: u32,
    data0: vec4<f32>, // rgb: BaseColor, w: MaterialType (cast)
    data1: vec4<f32>, // x: Metallic, y: Roughness, z: IOR, w: 0.0
    data2: vec4<f32>, // x: BaseTex, y: MetRoughTex, z: NormalTex, w: EmissiveTex
    data3: vec4<f32>  // rgb: EmissiveColor, w: OcclusionTex
}

struct LightRef {
    inst_idx: u32,
    tri_idx: u32
}

struct BVHNode {
    min_b: vec4<f32>, // w: skip_pointer
    max_b: vec4<f32>, // w: data (internal: 0, leaf: (left_first << 3) | tri_count)
}

struct Instance {
    transform_0: vec4<f32>,
    transform_1: vec4<f32>,
    transform_2: vec4<f32>,
    transform_3: vec4<f32>,
    inv_0: vec4<f32>,
    inv_1: vec4<f32>,
    inv_2: vec4<f32>,
    inv_3: vec4<f32>,
    blas_node_offset: u32,
    attr_offset: u32,
    instance_id: u32,
    padding: u32,
}

struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
    inv_d: vec3<f32>,
    origin_inv_d: vec3<f32>
}

fn make_ray(origin: vec3<f32>, direction: vec3<f32>) -> Ray {
    let inv_d = 1.0 / direction;
    return Ray(origin, direction, inv_d, origin * inv_d);
}

struct HitResult {
    t: f32,
    tri_idx: f32,
    inst_idx: i32
}

struct ONB {
    u: vec3<f32>,
    v: vec3<f32>,
    w: vec3<f32>,
}

struct LightSample {
    L: vec3<f32>,       // Radiance
    dir: vec3<f32>,     // Direction to light
    dist: f32,          // Distance to light
    pdf: f32,           // PDF of sampling this point
}

struct ScatterResult {
    dir: vec3<f32>,
    pdf: f32,
    throughput: vec3<f32>,
    is_specular: bool
}


// \u516B\u9762\u4F53\u30A8\u30F3\u30B3\u30FC\u30C7\u30A3\u30F3\u30B0\u306B\u3088\u308B\u6CD5\u7DDA\u5727\u7E2E (vec3 -> vec2)
fn pack_normal(n: vec3<f32>) -> vec2<f32> {
    let p = n.xy * (1.0 / (abs(n.x) + abs(n.y) + abs(n.z)));
    return select(p, (1.0 - abs(p.yx)) * select(vec2(-1.0), vec2(1.0), p.xy >= vec2(0.0)), n.z < 0.0);
}

fn unpack_normal(p: vec2<f32>) -> vec3<f32> {
    var n = vec3(p, 1.0 - abs(p.x) - abs(p.y));
    let t = saturate(-n.z);
    n.x += select(t, -t, n.x >= 0.0);
    n.y += select(t, -t, n.y >= 0.0);
    return normalize(n);
}


// =========================================================
//   Bindings
// =========================================================

@group(0) @binding(1) var<storage, read_write> accumulateBuffer: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> scene: SceneUniforms;
@group(0) @binding(3) var<storage, read> geometry_pos: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> topology: array<MeshTopology>;
@group(0) @binding(5) var<storage, read> nodes: array<BVHNode>; 
@group(0) @binding(6) var<storage, read> instances: array<Instance>;
@group(0) @binding(7) var tex: texture_2d_array<f32>;
@group(0) @binding(8) var smp: sampler;
@group(0) @binding(9) var<storage, read> lights: array<LightRef>;
@group(0) @binding(11) var<storage, read> geometry_norm: array<vec4<f32>>;
@group(0) @binding(12) var<storage, read> geometry_uv: array<vec2<f32>>;
@group(0) @binding(13) var g_albedo: texture_2d<f32>;
@group(0) @binding(14) var g_normal: texture_2d<f32>;
@group(0) @binding(15) var g_depth: texture_depth_2d;
@group(0) @binding(16) var<storage, read_write> reservoirsBuffer: array<Reservoir>;
@group(0) @binding(17) var<storage, read> prevReservoirsBuffer: array<Reservoir>;

// =========================================================
//   Buffer Accessors
// =========================================================

fn get_pos(idx: u32) -> vec3<f32> {
    return geometry_pos[idx].xyz;
}

fn get_normal(idx: u32) -> vec3<f32> {
    return geometry_norm[idx].xyz;
}

fn get_uv(idx: u32) -> vec2<f32> {
    return geometry_uv[idx];
}

fn get_transform(inst: Instance) -> mat4x4<f32> {
    return mat4x4<f32>(inst.transform_0, inst.transform_1, inst.transform_2, inst.transform_3);
}

fn get_inv_transform(inst: Instance) -> mat4x4<f32> {
    return mat4x4<f32>(inst.inv_0, inst.inv_1, inst.inv_2, inst.inv_3);
}

// =========================================================
//   Math & RNG Helpers
// =========================================================

fn init_rng(pixel_idx: u32, frame: u32) -> u32 {
    var seed = pixel_idx + frame * 719393u;
    seed ^= 2747636419u; seed *= 2654435769u; seed ^= (seed >> 16u);
    seed *= 2654435769u; seed ^= (seed >> 16u); seed *= 2654435769u;
    return seed;
}

fn rand_pcg(state: ptr<function, u32>) -> f32 {
    let old = *state; *state = old * 747796405u + 2891336453u;
    let word = ((*state) >> ((old >> 28u) + 4u)) ^ (*state);
    return f32((word >> 22u) ^ word) / 4294967295.0;
}

fn random_unit_vector(onb: ONB, rng: ptr<function, u32>) -> vec3<f32> {
    let r1 = rand_pcg(rng);
    let r2 = rand_pcg(rng);
    let phi = 2.0 * PI * r1;
    let cos_theta = sqrt(1.0 - r2);
    let sin_theta = sqrt(r2);
    let local_dir = vec3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
    return local_to_world(onb, local_dir);
}

fn random_in_unit_disk(rng: ptr<function, u32>) -> vec3<f32> {
    let r = sqrt(rand_pcg(rng));
    let theta = 2.0 * PI * rand_pcg(rng);
    return vec3<f32>(r * cos(theta), r * sin(theta), 0.0);
}

fn build_onb(n: vec3<f32>) -> ONB {
    let sign = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (sign + n.z);
    let b = n.x * n.y * a;
    let u = vec3(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x);
    let v = vec3(b, sign + n.y * n.y * a, -n.y);
    return ONB(u, v, n);
}

fn local_to_world(onb: ONB, a: vec3<f32>) -> vec3<f32> {
    return a.x * onb.u + a.y * onb.v + a.z * onb.w;
}

// =========================================================
//   BSDF Functions
// =========================================================

fn eval_diffuse(albedo: vec3<f32>) -> vec3<f32> {
    return albedo / PI;
}

fn sample_diffuse(normal: vec3<f32>, albedo: vec3<f32>, rng: ptr<function, u32>) -> ScatterResult {
    let onb = build_onb(normal);
    let dir = random_unit_vector(onb, rng);
    let cos_theta = max(dot(normal, dir), 0.0);
    return ScatterResult(dir, cos_theta / PI, albedo, false);
}

// GGX
fn ggx_d(n_dot_h: f32, a2: f32) -> f32 {
    let d = (n_dot_h * a2 - n_dot_h) * n_dot_h + 1.0;
    return a2 / (PI * d * d);
}

fn ggx_g(n_dot_v: f32, n_dot_l: f32, a2: f32) -> f32 {
    let g1_v = 2.0 * n_dot_v / (n_dot_v + sqrt(a2 + (1.0 - a2) * n_dot_v * n_dot_v));
    let g1_l = 2.0 * n_dot_l / (n_dot_l + sqrt(a2 + (1.0 - a2) * n_dot_l * n_dot_l));
    return g1_v * g1_l;
}

fn pow5(x: f32) -> f32 {
    let x2 = x * x;
    return x2 * x2 * x;
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (1.0 - f0) * pow5(clamp(1.0 - cos_theta, 0.0, 1.0));
}

fn eval_ggx(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, roughness: f32, f0: vec3<f32>) -> vec3<f32> {
    let h = normalize(v + l);
    let n_dot_v = max(dot(n, v), 1e-4);
    let n_dot_l = max(dot(n, l), 1e-4);
    let n_dot_h = max(dot(n, h), 1e-4);
    let v_dot_h = max(dot(v, h), 1e-4);

    let a2 = roughness * roughness;
    let d = ggx_d(n_dot_h, a2);
    let g = ggx_g(n_dot_v, n_dot_l, a2);
    let f = fresnel_schlick(v_dot_h, f0);

    return (d * g * f) / (4.0 * n_dot_v * n_dot_l);
}

fn sample_ggx(n: vec3<f32>, v: vec3<f32>, roughness: f32, f0: vec3<f32>, rng: ptr<function, u32>) -> ScatterResult {
    let a = roughness;
    let u = vec2(rand_pcg(rng), rand_pcg(rng));

    let phi = 2.0 * PI * u.x;
    let cos_theta = sqrt(max(0.0, (1.0 - u.y) / (1.0 + (a * a - 1.0) * u.y)));
    let sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));

    let h_local = vec3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
    let onb = build_onb(n);
    let h = local_to_world(onb, h_local);
    let l = reflect(-v, h);

    if dot(n, l) <= 0.0 {
        return ScatterResult(vec3(0.0), 0.0, vec3(0.0), false);
    }

    let n_dot_v = max(dot(n, v), 1e-4);
    let n_dot_l = max(dot(n, l), 1e-4);
    let n_dot_h = max(dot(n, h), 1e-4);
    let v_dot_h = max(dot(v, h), 1e-4);

    let a2 = a * a;
    let d = ggx_d(n_dot_h, a2);
    let g = ggx_g(n_dot_v, n_dot_l, a2);
    let f = fresnel_schlick(v_dot_h, f0);

    let pdf = (d * n_dot_h) / (4.0 * v_dot_h);
    var throughput = vec3(0.0);
    if pdf > 1e-6 {
        throughput = (g * f * v_dot_h) / (n_dot_v * n_dot_h);
    }
    let treat_as_specular = roughness < 0.01;

    return ScatterResult(l, pdf, throughput, treat_as_specular);
}

fn bsdf_to_throughput(d: f32, g: f32, f: vec3<f32>, n_dot_v: f32, n_dot_l: f32, n_dot_h: f32, v_dot_h: f32, pdf: f32) -> vec3<f32> {
    if pdf <= 0.0 { return vec3(0.0); }
    return (d * g * f) / (4.0 * n_dot_v * n_dot_l) * n_dot_l / pdf;
}



// Dielectric
fn reflectance_dielectric(cosine: f32, ref_idx: f32) -> f32 {
    var r0 = (1.0 - ref_idx) / (1.0 + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0 - r0) * pow5(1.0 - cosine);
}

fn sample_dielectric(dir: vec3<f32>, normal: vec3<f32>, ior: f32, albedo: vec3<f32>, rng: ptr<function, u32>) -> ScatterResult {
    let front_face = dot(dir, normal) < 0.0;
    let refraction_ratio = select(ior, 1.0 / ior, front_face);
    let n = select(-normal, normal, front_face);

    let unit_dir = normalize(dir);
    let cos_theta = min(dot(-unit_dir, n), 1.0);
    let sin_theta = sqrt(1.0 - cos_theta * cos_theta);

    let cannot_refract = refraction_ratio * sin_theta > 1.0;
    var direction: vec3<f32>;

    if cannot_refract || reflectance_dielectric(cos_theta, refraction_ratio) > rand_pcg(rng) {
        direction = reflect(unit_dir, n);
    } else {
        direction = refract(unit_dir, n, refraction_ratio);
    }

    return ScatterResult(direction, 1.0, albedo, true);
}

// =========================================================
//   Direct Light Sampling
// =========================================================

fn sample_light_source(hit_p: vec3<f32>, rng: ptr<function, u32>) -> LightSample {
    let light_count = scene.light_count;
    if light_count == 0u {
        return LightSample(vec3(0.0), vec3(0.0), 0.0, 0.0);
    }

    let light_pick_idx = u32(rand_pcg(rng) * f32(light_count));
    let l_ref = lights[light_pick_idx];

    let tri = topology[l_ref.tri_idx];
    let inst = instances[l_ref.inst_idx];
    let m = get_transform(inst);

    let v0 = (m * vec4(get_pos(tri.v0), 1.0)).xyz;
    let v1 = (m * vec4(get_pos(tri.v1), 1.0)).xyz;
    let v2 = (m * vec4(get_pos(tri.v2), 1.0)).xyz;

    let r1 = rand_pcg(rng);
    let r2 = rand_pcg(rng);
    let sqrt_r1 = sqrt(r1);
    let u = 1.0 - sqrt_r1;
    let v = r2 * sqrt_r1;
    let w = 1.0 - u - v;

    let p = v0 * u + v1 * v + v2 * w;
    let edge1 = v1 - v0;
    let edge2 = v2 - v0;
    let n_raw = normalize(cross(edge1, edge2));
    let area = length(cross(edge1, edge2)) * 0.5;

    let l_dir = p - hit_p;
    let dist_sq = dot(l_dir, l_dir);
    let dist = sqrt(dist_sq);
    let unit_l = l_dir / dist;

    let cos_theta_l = max(dot(n_raw, -unit_l), 0.0);
    if cos_theta_l < 1e-6 || area < 1e-6 {
        return LightSample(vec3(0.0), vec3(0.0), 0.0, 0.0);
    }

    // Albedo if light
    let uv0 = get_uv(tri.v0);
    let uv1 = get_uv(tri.v1);
    let uv2 = get_uv(tri.v2);
    let tex_uv = uv0 * u + uv1 * v + uv2 * w;
    var L = tri.data0.rgb;
    let base_tex = tri.data2.x;
    if base_tex > -0.5 {
        L *= textureSampleLevel(tex, smp, tex_uv, i32(base_tex), 0.0).rgb;
    }

    let pdf = (dist_sq / (cos_theta_l * area)) / f32(light_count);

    return LightSample(L, unit_l, dist, pdf);
}

fn get_light_pdf(origin: vec3<f32>, tri_idx: u32, inst_idx: u32, t: f32, l_dir: vec3<f32>) -> f32 {
    let tri = topology[tri_idx];
    let inst = instances[inst_idx];
    let m = get_transform(inst);

    let v0 = (m * vec4(get_pos(tri.v0), 1.0)).xyz;
    let v1 = (m * vec4(get_pos(tri.v1), 1.0)).xyz;
    let v2 = (m * vec4(get_pos(tri.v2), 1.0)).xyz;

    let edge1 = v1 - v0;
    let edge2 = v2 - v0;
    let area = length(cross(edge1, edge2)) * 0.5;
    let normal = normalize(cross(edge1, edge2));

    let cos_theta_l = max(dot(normal, -l_dir), 0.0);
    if cos_theta_l < 1e-4 || area < 1e-6 { return 0.0; }

    let light_count = scene.light_count;
    let dist_sq = t * t;
    return (dist_sq / (cos_theta_l * area)) / f32(light_count);
}

fn power_heuristic(pdf_a: f32, pdf_b: f32) -> f32 {
    let a2 = pdf_a * pdf_a;
    let b2 = pdf_b * pdf_b;
    return a2 / (a2 + b2 + 1e-6);
}

// =========================================================
//   Intersection Functions
// =========================================================

fn intersect_aabb(min_b: vec3<f32>, max_b: vec3<f32>, r: Ray, t_min: f32, t_max: f32) -> f32 {
    let t1 = min_b * r.inv_d - r.origin_inv_d;
    let t2 = max_b * r.inv_d - r.origin_inv_d;
    let t_near = min(t1, t2);
    let t_far = max(t1, t2);
    let tm_near = max(t_min, max(t_near.x, max(t_near.y, t_near.z)));
    let tm_far = min(t_max, min(t_far.x, min(t_far.y, t_far.z)));
    return select(T_MAX, tm_near, tm_near <= tm_far);
}

fn hit_triangle_raw(v0: vec3<f32>, v1: vec3<f32>, v2: vec3<f32>, r: Ray, t_min: f32, t_max: f32) -> f32 {
    let e1 = v1 - v0; let e2 = v2 - v0;
    let h = cross(r.direction, e2); let a = dot(e1, h);
    if abs(a) < 1e-6 { return -1.0; } // Increased epsilon
    let f = 1.0 / a; let s = r.origin - v0; let u = f * dot(s, h);
    if u < 0.0 || u > 1.0 { return -1.0; }
    let q = cross(s, e1); let v = f * dot(r.direction, q);
    if v < 0.0 || u + v > 1.0 { return -1.0; }
    let t = f * dot(e2, q);
    return select(-1.0, t, t > t_min && t < t_max);
}

fn intersect_blas(r: Ray, t_min: f32, t_max: f32, node_start_idx: u32) -> vec2<f32> {
    var closest_t = t_max;
    var hit_idx = -1.0;
    
    let end_node = node_start_idx + bitcast<u32>(nodes[node_start_idx].min_b.w);
    var curr = node_start_idx;
    
    while curr < end_node {
        let node = nodes[curr];
        
        var hit_t = closest_t;
        let t_aabb = intersect_aabb(node.min_b.xyz, node.max_b.xyz, r, t_min, closest_t);
        
        if t_aabb < T_MAX {
            let data = bitcast<u32>(node.max_b.w);
            if data != 0u {
                // Leaf node
                let first = data >> 3u;
                let count = data & 7u;
                for (var i = 0u; i < count; i++) {
                    let tri_id = first + i;
                    let tr = topology[tri_id];
                    let t = hit_triangle_raw(get_pos(tr.v0), get_pos(tr.v1), get_pos(tr.v2), r, t_min, closest_t);
                    if t > 0.0 { 
                        closest_t = t; 
                        hit_idx = f32(tri_id); 
                    }
                }
                curr = node_start_idx + bitcast<u32>(node.min_b.w);
            } else {
                // Internal node
                curr = curr + 1u;
            }
        } else {
            // Missed AABB
            curr = node_start_idx + bitcast<u32>(node.min_b.w);
        }
    }
    return vec2<f32>(closest_t, hit_idx);
}

fn intersect_tlas(r: Ray, t_min: f32, t_max: f32) -> HitResult {
    var res: HitResult; res.t = t_max; res.tri_idx = -1.0; res.inst_idx = -1;
    if scene.blas_base_idx == 0u { return res; }

    var curr = 0u;
    let end_node = bitcast<u32>(nodes[0].min_b.w);

    while curr < end_node {
        let node = nodes[curr];
        
        if intersect_aabb(node.min_b.xyz, node.max_b.xyz, r, t_min, res.t) < T_MAX {
            let data = bitcast<u32>(node.max_b.w);
            if data != 0u {
                // Leaf
                let inst_idx = data >> 3u;
                let inst = instances[inst_idx];
                let r_local = make_ray((get_inv_transform(inst) * vec4(r.origin, 1.0)).xyz, (get_inv_transform(inst) * vec4(r.direction, 0.0)).xyz);
                let blas = intersect_blas(r_local, t_min, res.t, scene.blas_base_idx + inst.blas_node_offset);
                if blas.y > -0.5 { 
                    res.t = blas.x; 
                    res.tri_idx = blas.y; 
                    res.inst_idx = i32(inst_idx); 
                }
                curr = bitcast<u32>(node.min_b.w);
            } else {
                curr = curr + 1u;
            }
        } else {
            curr = bitcast<u32>(node.min_b.w);
        }
    }
    return res;
}

// shadow ray\u7248
// \u30B7\u30E3\u30C9\u30A6\u30EC\u30A4\u7528\u306EBLAS\u4EA4\u5DEE\u5224\u5B9A\uFF08\u30D2\u30C3\u30C8\u3057\u305F\u3089\u5373true\u3092\u8FD4\u3059\uFF09
fn intersect_blas_shadow(r: Ray, t_min: f32, t_max: f32, node_start_idx: u32) -> bool {
    let end_node = node_start_idx + bitcast<u32>(nodes[node_start_idx].min_b.w);
    var curr = node_start_idx;
    
    while curr < end_node {
        let node = nodes[curr];
        
        let t_aabb = intersect_aabb(node.min_b.xyz, node.max_b.xyz, r, t_min, t_max);
        if t_aabb < T_MAX {
            let data = bitcast<u32>(node.max_b.w);
            if data != 0u {
                // Leaf node
                let first = data >> 3u;
                let count = data & 7u;
                for (var i = 0u; i < count; i++) {
                    let tri_id = first + i;
                    let tr = topology[tri_id];
                    let t = hit_triangle_raw(get_pos(tr.v0), get_pos(tr.v1), get_pos(tr.v2), r, t_min, t_max);
                    if t > 0.0 { return true; }
                }
                curr = node_start_idx + bitcast<u32>(node.min_b.w);
            } else {
                // Internal node
                curr = curr + 1u;
            }
        } else {
            // Missed AABB
            curr = node_start_idx + bitcast<u32>(node.min_b.w);
        }
    }
    return false;
}

// \u30B7\u30E3\u30C9\u30A6\u30EC\u30A4\u7528\u306ETLAS\u4EA4\u5DEE\u5224\u5B9A
fn intersect_tlas_shadow(r: Ray, t_min: f32, t_max: f32) -> bool {
    if scene.blas_base_idx == 0u { return false; }

    var curr = 0u;
    let end_node = bitcast<u32>(nodes[0].min_b.w);

    while curr < end_node {
        let node = nodes[curr];
        
        let t_aabb = intersect_aabb(node.min_b.xyz, node.max_b.xyz, r, t_min, t_max);
        if t_aabb < T_MAX {
            let data = bitcast<u32>(node.max_b.w);
            if data != 0u {
                // Leaf
                let inst_idx = data >> 3u;
                let inst = instances[inst_idx];
                
                let r_local = make_ray(
                    (get_inv_transform(inst) * vec4(r.origin, 1.0)).xyz, 
                    (get_inv_transform(inst) * vec4(r.direction, 0.0)).xyz
                );
                
                if intersect_blas_shadow(r_local, t_min, t_max, scene.blas_base_idx + inst.blas_node_offset) {
                    return true;
                }
                curr = bitcast<u32>(node.min_b.w);
            } else {
                curr = curr + 1u;
            }
        } else {
            curr = bitcast<u32>(node.min_b.w);
        }
    }
    return false;
}

fn get_throughput(w_o: vec3<f32>, w_i: vec3<f32>, normal: vec3<f32>, mat_type: u32, roughness: f32, f0: vec3<f32>, albedo: vec3<f32>) -> vec3<f32> {
    if mat_type == 0u {
        return albedo;
    } else if mat_type == 1u {
        let h = normalize(w_o + w_i);
        let n_dot_v = max(dot(normal, w_o), 1e-4);
        let n_dot_l = max(dot(normal, w_i), 1e-4);
        let n_dot_h = max(dot(normal, h), 1e-4);
        let v_dot_h = max(dot(w_o, h), 1e-4);
        let a2 = roughness * roughness;
        let g = ggx_g(n_dot_v, n_dot_l, a2);
        let f = fresnel_schlick(v_dot_h, f0);
        return (g * f * v_dot_h) / (n_dot_v * n_dot_h);
    } else { // mat_type == 2u
        return albedo;
    }
}

fn eval_brdf_cos(w_o: vec3<f32>, w_i: vec3<f32>, normal: vec3<f32>, mat_type: u32, roughness: f32, f0: vec3<f32>, albedo: vec3<f32>) -> vec3<f32> {
    let n_dot_l = max(dot(normal, w_i), 1e-4);

    if mat_type == 0u {
        return (albedo / PI) * n_dot_l;
    } else if mat_type == 1u {
        let h = normalize(w_o + w_i);
        let n_dot_v = max(dot(normal, w_o), 1e-4);
        let n_dot_h = max(dot(normal, h), 1e-4);
        let v_dot_h = max(dot(w_o, h), 1e-4);
        let a2 = roughness * roughness;
        let d = ggx_d(n_dot_h, a2);
        let g = ggx_g(n_dot_v, n_dot_l, a2);
        let f = fresnel_schlick(v_dot_h, f0);
        return (d * g * f) / (4.0 * n_dot_v);
    } else { 
        return vec3<f32>(0.0);
    }
}

fn get_pdf(w_o: vec3<f32>, w_i: vec3<f32>, normal: vec3<f32>, mat_type: u32, roughness: f32) -> f32 {
    if mat_type == 0u {
        return max(dot(normal, w_i), 0.0) / PI;
    } else if mat_type == 1u {
        let h = normalize(w_o + w_i);
        let n_dot_h = max(dot(normal, h), 1e-4);
        let v_dot_h = max(dot(w_o, h), 1e-4);
        let a2 = roughness * roughness;
        let d = ggx_d(n_dot_h, a2);
        return (d * n_dot_h) / (4.0 * v_dot_h);
    } else {
        return 0.0;
    }
}

fn update_reservoir(r: ptr<function, Reservoir>, s: Sample, weight: f32, rng: ptr<function, u32>) {
    r.w_sum += weight;
    if rand_pcg(rng) < (weight / r.w_sum) {
        r.sample = s;
    }
}


fn get_world_pos(id: vec2<u32>, depth_val: f32) -> vec3<f32> {
    let u_cam = (f32(id.x) + 0.5 + scene.jitter.x * f32(scene.width)) / f32(scene.width);
    let v_cam = 1.0 - (f32(id.y) + 0.5 + scene.jitter.y * f32(scene.height)) / f32(scene.height);
    let ray_dir = normalize(scene.camera.lower_left_corner.xyz + u_cam * scene.camera.horizontal.xyz + v_cam * scene.camera.vertical.xyz - scene.camera.origin.xyz);
    
    // Reverse non-linear Z to view-space Z
    let z_near = 0.001;
    let z_far = 10000.0;
    let z_view = (z_far * z_near) / (z_far - depth_val * (z_far - z_near));
    
    // View-space Z to ray distance t
    let eye = scene.camera.origin.xyz;
    let center = scene.camera.lower_left_corner.xyz + scene.camera.horizontal.xyz * 0.5 + scene.camera.vertical.xyz * 0.5;
    let forward = normalize(center - eye);
    let t = z_view / dot(ray_dir, forward);
    
    return eye + ray_dir * t;
}

@compute @workgroup_size(8, 8)
fn final_shading(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= scene.width || id.y >= scene.height { return; }
    let p_idx = id.y * scene.width + id.x;
    var rng = init_rng(p_idx, scene.frame_count);

    var off = vec3(0.);
    if scene.camera.origin.w > 0. {
        let rd = scene.camera.origin.w * random_in_unit_disk(&rng);
        off = scene.camera.u.xyz * rd.x + scene.camera.v.xyz * rd.y;
    }

    let depth_val = textureLoad(g_depth, id.xy, 0);
    if depth_val >= 1.0 { 
        var acc_val = vec4(0.0, 0.0, 0.0, 1.0);
        if scene.frame_count > 1u { acc_val = accumulateBuffer[p_idx]; }
        accumulateBuffer[p_idx] = acc_val;
        return; 
    }

    let r = reservoirsBuffer[p_idx];
    let sample = r.sample;
    let W = r.W;

    let g_normal_val = textureLoad(g_normal, id.xy, 0);
    var tri_idx: u32 = bitcast<u32>(g_normal_val.z);
    var inst_idx: i32 = i32(bitcast<u32>(g_normal_val.w));

    var tri = topology[tri_idx];
    var inst = instances[inst_idx];
    var inv = get_inv_transform(inst);
    var v0_pos = get_pos(tri.v0);
    var v1_pos = get_pos(tri.v1);
    var v2_pos = get_pos(tri.v2);

    let u_cam = (f32(id.x) + 0.5 + scene.jitter.x * f32(scene.width)) / f32(scene.width);
    let v_cam = 1. - (f32(id.y) + 0.5 + scene.jitter.y * f32(scene.height)) / f32(scene.height);
    let dir = scene.camera.lower_left_corner.xyz + u_cam * scene.camera.horizontal.xyz + v_cam * scene.camera.vertical.xyz - scene.camera.origin.xyz - off;
    var r_in = make_ray(scene.camera.origin.xyz + off, dir);

    var r_local = make_ray((inv * vec4(r_in.origin, 1.)).xyz, (inv * vec4(r_in.direction, 0.)).xyz);
    var s = r_local.origin - v0_pos;
    var e1 = v1_pos - v0_pos;
    var e2 = v2_pos - v0_pos;
    var h_val = cross(r_local.direction, e2);
    var f_val = 1.0 / dot(e1, h_val);
    var u_bar = f_val * dot(s, h_val);
    var q = cross(s, e1);
    var v_bar = f_val * dot(r_local.direction, q);
    var w_bar = 1.0 - u_bar - v_bar;
    var hit_t = f_val * dot(e2, q);
    
    var uv0 = get_uv(tri.v0);
    var uv1 = get_uv(tri.v1);
    var uv2 = get_uv(tri.v2);
    var tex_uv = uv0 * w_bar + uv1 * u_bar + uv2 * v_bar;

    var normal = unpack_normal(g_normal_val.xy);
    var albedo = textureLoad(g_albedo, id.xy, 0).rgb;

    var local_geom_n = normalize(cross(e1, e2));
    var world_geom_n = normalize((vec4(local_geom_n, 0.0) * inv).xyz);

    let mat_type = u32(tri.data0.w + 0.5);
    let primary_hit_p = r_in.origin + r_in.direction * hit_t;

    normal = select(-normal, normal, dot(r_in.direction, normal) < 0.0);
    world_geom_n = select(-world_geom_n, world_geom_n, dot(r_in.direction, world_geom_n) < 0.0);

    var metallic = tri.data1.x;
    var roughness = tri.data1.y;
    if tri.data2.y > -0.5 {
        let mr = textureSampleLevel(tex, smp, tex_uv, i32(tri.data2.y), 0.0).rgb;
        metallic *= mr.b; roughness *= mr.g;
    }
    roughness = max(roughness, 0.005);

    var emissive = tri.data3.rgb;
    if tri.data2.w > -0.5 { emissive *= textureSampleLevel(tex, smp, tex_uv, i32(tri.data2.w), 0.0).rgb; }

    let f0 = mix(vec3(0.04), albedo, metallic);

    var final_radiance = vec3(0.0);

    // 1. Emissive contribution
    if mat_type == 3u || length(emissive) > 1e-4 {
        let em_val = select(emissive, albedo, mat_type == 3u);
        final_radiance += em_val;
    }

    // 2. Direct Lighting (NEE)
    if mat_type != 2u && mat_type != 3u {
        let light_s = sample_light_source(primary_hit_p, &rng);
        if light_s.pdf > 1e-6 {
            if !intersect_tlas_shadow(make_ray(primary_hit_p + world_geom_n * 1e-4, light_s.dir), T_MIN, light_s.dist - 2e-4) {
                let w_o = -r_in.direction;
                let tp = eval_brdf_cos(w_o, light_s.dir, normal, mat_type, roughness, f0, albedo);
                let bsdf_pdf_val = get_pdf(w_o, light_s.dir, normal, mat_type, roughness);
                final_radiance += tp * light_s.L * power_heuristic(light_s.pdf, bsdf_pdf_val) / light_s.pdf;
            }
        }
    }

    // 3. Indirect Lighting (from sample)
    if length(sample.radiance.xyz) > 0.0 && mat_type != 3u {
        let is_delta = (mat_type == 2u) || (mat_type == 1u && metallic > 0.9 && roughness < 0.01);
        if is_delta {
            final_radiance += sample.radiance.xyz * W;
        } else {
            let w_i = vec3(sample.hit_p.w, sample.normal.w, sample.radiance.w);
            let w_o = -r_in.direction;
            let tp = eval_brdf_cos(w_o, w_i, normal, mat_type, roughness, f0, albedo);
            final_radiance += tp * sample.radiance.xyz * W;
        }
    }

    var acc_val = vec4<f32>(final_radiance, 1.0);
    if scene.frame_count > 1u {
        acc_val = accumulateBuffer[p_idx] + vec4<f32>(final_radiance, 1.0);
    }
    accumulateBuffer[p_idx] = acc_val;
}
`;
  class F {
    constructor(e) {
      __publicField(this, "initialSamplingPipeline");
      __publicField(this, "temporalReusePipeline");
      __publicField(this, "spatialReusePipeline");
      __publicField(this, "finalShadingPipeline");
      __publicField(this, "initialBindGroupLayout");
      __publicField(this, "temporalBindGroupLayout");
      __publicField(this, "spatialBindGroupLayout");
      __publicField(this, "finalBindGroupLayout");
      __publicField(this, "initialBindGroups", []);
      __publicField(this, "temporalBindGroups", []);
      __publicField(this, "spatialBindGroups", []);
      __publicField(this, "finalBindGroups", []);
      __publicField(this, "ctx");
      this.ctx = e;
    }
    buildPipeline(e) {
      const n = this.ctx.device.createShaderModule({
        label: "Initial Sampling Shader",
        code: O
      }), t = this.ctx.device.createShaderModule({
        label: "Temporal Reuse Shader",
        code: q
      }), r = this.ctx.device.createShaderModule({
        label: "Spatial Reuse Shader",
        code: j
      }), i = this.ctx.device.createShaderModule({
        label: "Final Shading Shader",
        code: $
      });
      this.initialSamplingPipeline = this.ctx.device.createComputePipeline({
        label: "Initial Sampling Pipeline",
        layout: "auto",
        compute: {
          module: n,
          entryPoint: "initial_sampling",
          constants: {
            MAX_DEPTH: e
          }
        }
      }), this.temporalReusePipeline = this.ctx.device.createComputePipeline({
        label: "Temporal Reuse Pipeline",
        layout: "auto",
        compute: {
          module: t,
          entryPoint: "temporal_reuse",
          constants: {
            MAX_DEPTH: e
          }
        }
      }), this.spatialReusePipeline = this.ctx.device.createComputePipeline({
        label: "Spatial Reuse Pipeline",
        layout: "auto",
        compute: {
          module: r,
          entryPoint: "spatial_reuse"
        }
      }), this.finalShadingPipeline = this.ctx.device.createComputePipeline({
        label: "Final Shading Pipeline",
        layout: "auto",
        compute: {
          module: i,
          entryPoint: "final_shading",
          constants: {
            MAX_DEPTH: e
          }
        }
      }), this.initialBindGroupLayout = this.initialSamplingPipeline.getBindGroupLayout(0), this.temporalBindGroupLayout = this.temporalReusePipeline.getBindGroupLayout(0), this.spatialBindGroupLayout = this.spatialReusePipeline.getBindGroupLayout(0), this.finalBindGroupLayout = this.finalShadingPipeline.getBindGroupLayout(0);
    }
    updateBindGroup(e) {
      if (!e.accumulateBuffer || !e.reservoirsBufferA || !e.reservoirsBufferB || !e.geometryBuffer || !e.nodesBuffer || !e.sceneUniformBuffer || !e.lightsBuffer) return;
      const n = [
        {
          binding: 2,
          resource: {
            buffer: e.sceneUniformBuffer
          }
        },
        {
          binding: 3,
          resource: {
            buffer: e.geometryBuffer,
            offset: 0,
            size: e.vertexCount * 16
          }
        },
        {
          binding: 4,
          resource: {
            buffer: e.topologyBuffer
          }
        },
        {
          binding: 5,
          resource: {
            buffer: e.nodesBuffer
          }
        },
        {
          binding: 6,
          resource: {
            buffer: e.instanceBuffer
          }
        },
        {
          binding: 7,
          resource: e.texture.createView({
            dimension: "2d-array"
          })
        },
        {
          binding: 8,
          resource: e.sampler
        },
        {
          binding: 9,
          resource: {
            buffer: e.lightsBuffer
          }
        },
        {
          binding: 11,
          resource: {
            buffer: e.geometryBuffer,
            offset: e.normOffset,
            size: e.vertexCount * 16
          }
        },
        {
          binding: 12,
          resource: {
            buffer: e.geometryBuffer,
            offset: e.uvOffset,
            size: e.vertexCount * 8
          }
        },
        {
          binding: 13,
          resource: e.renderTargetView
        },
        {
          binding: 14,
          resource: e.gBufferNormalView
        },
        {
          binding: 15,
          resource: e.depthTextureViews[e.historyIndex]
        }
      ];
      for (let t = 0; t < 2; t++) {
        const r = t === 0 ? e.reservoirsBufferA : e.reservoirsBufferB, i = t === 0 ? e.reservoirsBufferB : e.reservoirsBufferA, s = [
          ...n,
          {
            binding: 16,
            resource: {
              buffer: r
            }
          },
          {
            binding: 17,
            resource: {
              buffer: i
            }
          },
          {
            binding: 18,
            resource: e.depthTextureViews[1 - e.historyIndex]
          }
        ];
        this.initialBindGroups[t] = this.ctx.device.createBindGroup({
          layout: this.initialBindGroupLayout,
          entries: s.filter((a) => a.binding !== 17 && a.binding !== 18)
        }), this.temporalBindGroups[t] = this.ctx.device.createBindGroup({
          layout: this.temporalBindGroupLayout,
          entries: s.filter((a) => a.binding === 2 || a.binding === 16 || a.binding === 17 || a.binding === 4 || a.binding === 13 || a.binding === 14 || a.binding === 15 || a.binding === 18)
        }), this.spatialBindGroups[t] = this.ctx.device.createBindGroup({
          layout: this.spatialBindGroupLayout,
          entries: [
            ...n,
            {
              binding: 16,
              resource: {
                buffer: r
              }
            },
            {
              binding: 17,
              resource: {
                buffer: e.spatialReservoirsBuffer
              }
            }
          ].filter((a) => a.binding === 2 || a.binding === 16 || a.binding === 17 || a.binding === 14 || a.binding === 15 || a.binding === 4 || a.binding === 3 || a.binding === 5 || a.binding === 6)
        }), this.finalBindGroups[t] = this.ctx.device.createBindGroup({
          layout: this.finalBindGroupLayout,
          entries: [
            {
              binding: 1,
              resource: {
                buffer: e.accumulateBuffer
              }
            },
            ...n,
            {
              binding: 16,
              resource: {
                buffer: e.spatialReservoirsBuffer
              }
            }
          ].filter((a) => a.binding !== 11 && a.binding !== 17)
        });
      }
    }
    execute(e, n) {
      const t = n % 2 === 0 ? 1 : 0;
      if (!this.initialBindGroups[t] || !this.temporalBindGroups[t] || !this.spatialBindGroups[t] || !this.finalBindGroups[t]) return;
      const r = Math.ceil(this.ctx.canvas.width / 8), i = Math.ceil(this.ctx.canvas.height / 8), s = e.beginComputePass();
      s.setPipeline(this.initialSamplingPipeline), s.setBindGroup(0, this.initialBindGroups[t]), s.dispatchWorkgroups(r, i), s.end();
      const a = e.beginComputePass();
      a.setPipeline(this.temporalReusePipeline), a.setBindGroup(0, this.temporalBindGroups[t]), a.dispatchWorkgroups(r, i), a.end();
      const c = e.beginComputePass();
      c.setPipeline(this.spatialReusePipeline), c.setBindGroup(0, this.spatialBindGroups[t]), c.dispatchWorkgroups(r, i), c.end();
      const l = e.beginComputePass();
      l.setPipeline(this.finalShadingPipeline), l.setBindGroup(0, this.finalBindGroups[t]), l.dispatchWorkgroups(r, i), l.end();
    }
  }
  const J = `// =========================================================
//   Post Process (PostProcess.wgsl)
// =========================================================

struct Camera {
    origin: vec4<f32>, // w: lens_radius
    lower_left_corner: vec4<f32>,
    horizontal: vec4<f32>,
    vertical: vec4<f32>,
    u: vec4<f32>,
    v: vec4<f32>
}

struct SceneUniforms {
    camera: Camera,
    prev_camera: Camera,
    frame_count: u32,
    blas_base_idx: u32,
    vertex_count: u32,
    rand_seed: u32,
    light_count: u32,
    width: u32,
    height: u32,
    pad: u32,
    jitter: vec2<f32>,
    average_jitter: vec2<f32>,
    prev_jitter: vec2<f32>,
    pad2: vec2<f32>
}

@group(0) @binding(0) var outputTex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<storage, read> accumulateBuffer: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> scene: SceneUniforms;
@group(0) @binding(3) var historyTex: texture_2d<f32>;
@group(0) @binding(4) var smp: sampler;
@group(0) @binding(5) var historyOutput: texture_storage_2d<rgba16float, write>;

fn aces_tone_mapping(color: vec3<f32>) -> vec3<f32> {
    let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
    return clamp((color * (a * color + vec3<f32>(b))) / (color * (c * color + vec3<f32>(d)) + vec3<f32>(e)), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn get_radiance(coord: vec2<i32>) -> vec3<f32> {
    let c = clamp(coord, vec2<i32>(0), vec2<i32>(i32(scene.width) - 1, i32(scene.height) - 1));
    let p_idx = u32(c.y) * scene.width + u32(c.x);
    let acc = accumulateBuffer[p_idx];
    if acc.a <= 0.0 { return vec3(0.0); }
    return acc.rgb / acc.a;
}

fn get_radiance_clean(coord: vec2<i32>) -> vec3<f32> {
    let center = get_radiance(coord);

    var min_nb = vec3(1e6);
    var max_nb = vec3(-1e6);
    
    // 3x3 box (excluding center) for more stable suppression
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            if x == 0 && y == 0 { continue; }
            let nb = get_radiance(coord + vec2<i32>(x, y));
            min_nb = min(min_nb, nb);
            max_nb = max(max_nb, nb);
        }
    }
    
    // Firefly suppression: clamp center to neighborhood range with some headroom
    let threshold = 3.0;
    return clamp(center, vec3<f32>(0.0), max_nb * threshold + vec3<f32>(0.1));
}

// Bilinear sampling for un-jittering
fn get_radiance_bilinear(uv: vec2<f32>) -> vec3<f32> {
    let dims = vec2<f32>(f32(scene.width), f32(scene.height));
    let f_coord = uv * dims - vec2<f32>(0.5);
    let i_coord = vec2<i32>(i32(floor(f_coord.x)), i32(floor(f_coord.y)));
    let f = f_coord - vec2<f32>(floor(f_coord.x), floor(f_coord.y));

    let c00 = get_radiance_clean(i_coord + vec2<i32>(0, 0));
    let c10 = get_radiance_clean(i_coord + vec2<i32>(1, 0));
    let c01 = get_radiance_clean(i_coord + vec2<i32>(0, 1));
    let c11 = get_radiance_clean(i_coord + vec2<i32>(1, 1));

    return mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);
}

// \u2605 Enhanced un-jittering: un-jitter using the average jitter of all accumulated frames
// This perfectly stabilizes the image during the first few frames of accumulation.
fn get_radiance_nearest(coord: vec2<i32>) -> vec3<f32> {
    // If accumulated enough frames, average jitter is basically 0, skip bilinear filtering to guarantee sharpness.
    if scene.frame_count > 16u {
        return get_radiance_clean(coord);
    }

    let dims = vec2<f32>(f32(scene.width), f32(scene.height));
    let uv = (vec2<f32>(f32(coord.x), f32(coord.y)) + vec2<f32>(0.5)) / dims;
    
    return get_radiance_bilinear(uv - scene.average_jitter);
}

fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= scene.width || id.y >= scene.height { return; }

    let dims = vec2<f32>(f32(scene.width), f32(scene.height));
    let uv = (vec2<f32>(f32(id.x), f32(id.y)) + vec2<f32>(0.5)) / dims;

    // 1. Un-jittered Current Frame Radiance (Now cleaned internally)
    let center_color = get_radiance_nearest(vec2<i32>(i32(id.x), i32(id.y)));

    // 2. Bilateral Filter
    let SIGMA_S = 0.5;
    let SIGMA_R = 0.1;
    let RADIUS = 1;

    var filtered_sum = vec3(0.0);
    var total_weight = 0.0;
    for (var dy = -RADIUS; dy <= RADIUS; dy++) {
        for (var dx = -RADIUS; dx <= RADIUS; dx++) {
            let neighbor_pos = vec2<i32>(i32(id.x), i32(id.y)) + vec2<i32>(dx, dy);
            let neighbor_color = get_radiance_nearest(neighbor_pos);

            let w_s = exp(-f32(dx * dx + dy * dy) / (2.0 * SIGMA_S * SIGMA_S));
            let color_diff = neighbor_color - center_color;
            let w_r = exp(-dot(color_diff, color_diff) / (2.0 * SIGMA_R * f32(RADIUS) * f32(RADIUS)));
            let w = w_s * w_r;
            filtered_sum += neighbor_color * w;
            total_weight += w;
        }
    }
    let denoised_hdr = filtered_sum / max(total_weight, 1e-4);
    
    // 3. TAA Blend (HDR Feedback)
    let samples_history = textureSampleLevel(historyTex, smp, uv, 0.0).rgb;
    
    // Neighborhood Clamping
    var m1 = vec3<f32>(0.0);
    var m2 = vec3<f32>(0.0);
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let neighbor_pos = vec2<i32>(i32(id.x), i32(id.y)) + vec2<i32>(dx, dy);
            let neighbor_color = get_radiance_nearest(neighbor_pos);
            m1 += neighbor_color;
            m2 += neighbor_color * neighbor_color;
        }
    }
    let mean = m1 / 9.0;
    let stddev = sqrt(max(m2 / 9.0 - mean * mean, vec3<f32>(0.0)));
    
    // Adaptive clamping
    var k = 1.0; // Tighter clamping for better stability during animation
    if scene.frame_count > 16u { k = 60.0; } // Effectively disable clamping when static for full convergence
    let clamped_history = clamp(samples_history, mean - stddev * k, mean + stddev * k);
 
    // Adaptive alpha for progressive refinement
    var alpha = 1.0 / f32(scene.frame_count);
    if scene.frame_count == 1u {
        alpha = 0.1; // Enable TAA even on first frame after update to hide jitter
    }
    alpha = max(alpha, 0.0001); // Even deeper convergence (10000 frames)

    let final_hdr = mix(clamped_history, denoised_hdr, alpha);

    // Store un-jittered result
    textureStore(historyOutput, vec2<i32>(i32(id.x), i32(id.y)), vec4<f32>(final_hdr, 1.0));

    // 4. Output
    let mapped_center = aces_tone_mapping(center_color);
    let mapped_denoised = aces_tone_mapping(denoised_hdr);

    let ldr_edge = mapped_center - mapped_denoised; 

    let sharpened = mapped_center + ldr_edge * 0.3;

    let ldr_out = pow(clamp(sharpened, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / 2.2));
    textureStore(outputTex, vec2<i32>(i32(id.x), i32(id.y)), vec4<f32>(ldr_out, 1.0));
}`;
  class V {
    constructor(e) {
      __publicField(this, "pipeline");
      __publicField(this, "bindGroupLayout");
      __publicField(this, "bindGroup");
      __publicField(this, "ctx");
      this.ctx = e;
    }
    buildPipeline() {
      const e = this.ctx.device.createShaderModule({
        label: "PostProcess",
        code: J
      });
      this.pipeline = this.ctx.device.createComputePipeline({
        label: "PostProcess Pipeline",
        layout: "auto",
        compute: {
          module: e,
          entryPoint: "main"
        }
      }), this.bindGroupLayout = this.pipeline.getBindGroupLayout(0);
    }
    updateBindGroup(e) {
      !e.renderTargetView || !e.accumulateBuffer || !e.sceneUniformBuffer || (this.bindGroup = this.ctx.device.createBindGroup({
        layout: this.bindGroupLayout,
        entries: [
          {
            binding: 0,
            resource: e.renderTargetView
          },
          {
            binding: 1,
            resource: {
              buffer: e.accumulateBuffer
            }
          },
          {
            binding: 2,
            resource: {
              buffer: e.sceneUniformBuffer
            }
          },
          {
            binding: 3,
            resource: e.historyTextureViews[1 - e.historyIndex]
          },
          {
            binding: 4,
            resource: e.sampler
          },
          {
            binding: 5,
            resource: e.historyTextureViews[e.historyIndex]
          }
        ]
      }));
    }
    execute(e) {
      if (!this.bindGroup) return;
      const n = Math.ceil(this.ctx.canvas.width / 8), t = Math.ceil(this.ctx.canvas.height / 8), r = e.beginComputePass();
      r.setPipeline(this.pipeline), r.setBindGroup(0, this.bindGroup), r.dispatchWorkgroups(n, t), r.end();
    }
  }
  const X = `struct Camera {
    origin: vec4<f32>,
    lower_left_corner: vec4<f32>,
    horizontal: vec4<f32>,
    vertical: vec4<f32>,
    u: vec4<f32>,
    v: vec4<f32>
}

struct SceneUniforms {
    camera: Camera,
    prev_camera: Camera,
    frame_count: u32,
    blas_base_idx: u32,
    vertex_count: u32,
    rand_seed: u32,
    light_count: u32,
    width: u32,
    height: u32,
    pad: u32,
    jitter: vec2<f32>,
    average_jitter: vec2<f32>,
    prev_jitter: vec2<f32>,
    pad2: vec2<f32>
}

struct MeshTopology {
    v0: u32,
    v1: u32,
    v2: u32,
    pad: u32,
    data0: vec4<f32>,
    data1: vec4<f32>,
    data2: vec4<f32>,
    data3: vec4<f32>
}

struct Instance {
    transform_0: vec4<f32>,
    transform_1: vec4<f32>,
    transform_2: vec4<f32>,
    transform_3: vec4<f32>,
    inv_0: vec4<f32>,
    inv_1: vec4<f32>,
    inv_2: vec4<f32>,
    inv_3: vec4<f32>,
    blas_node_offset: u32,
    attr_offset: u32,
    instance_id: u32,
    padding: u32,
}

@group(0) @binding(2) var<uniform> scene: SceneUniforms;
@group(0) @binding(3) var<storage, read> geometry_pos: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> topology: array<MeshTopology>;
@group(0) @binding(6) var<storage, read> instances: array<Instance>;
@group(0) @binding(11) var<storage, read> geometry_norm: array<vec4<f32>>;
@group(0) @binding(12) var<storage, read> geometry_uv: array<vec2<f32>>;
@group(0) @binding(7) var tex: texture_2d_array<f32>;
@group(0) @binding(8) var smp: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) tex_id: f32,
    @location(4) @interpolate(flat) instance_id: u32,
    @location(5) @interpolate(flat) tri_idx: u32,
}

// \u516B\u9762\u4F53\u30A8\u30F3\u30B3\u30FC\u30C7\u30A3\u30F3\u30B0\u306B\u3088\u308B\u6CD5\u7DDA\u5727\u7E2E (vec3 -> vec2)
fn pack_normal(n: vec3<f32>) -> vec2<f32> {
    let p = n.xy * (1.0 / (abs(n.x) + abs(n.y) + abs(n.z)));
    return select(p, (1.0 - abs(p.yx)) * select(vec2(-1.0), vec2(1.0), p.xy >= vec2(0.0)), n.z < 0.0);
}

struct GBufferOutput {
    @location(0) albedo: vec4<f32>,
    @location(1) normal_and_id: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index : u32, @builtin(instance_index) instance_index : u32) -> VertexOutput {
    let tri_idx = vertex_index / 3u;
    let local_idx = vertex_index % 3u;

    // Out of bounds safety
    let tri = topology[tri_idx];
    let inst = instances[instance_index];

    var v_idx: u32;
    if (local_idx == 0u) {
        v_idx = tri.v0;
    } else if (local_idx == 1u) {
        v_idx = tri.v1;
    } else {
        v_idx = tri.v2;
    }

    let local_pos = geometry_pos[v_idx].xyz;
    let local_norm = geometry_norm[v_idx].xyz;
    let local_uv = geometry_uv[v_idx];

    // Apply instance transform
    let mat = mat4x4<f32>(inst.transform_0, inst.transform_1, inst.transform_2, inst.transform_3);
    let inv_mat = mat4x4<f32>(inst.inv_0, inst.inv_1, inst.inv_2, inst.inv_3);

    let pos = (mat * vec4<f32>(local_pos, 1.0)).xyz;
    let norm = normalize((vec4<f32>(local_norm, 0.0) * inv_mat).xyz);

    // Exact View-Projection matching the Raytracer's image plane
    let eye = scene.camera.origin.xyz;
    let horizontal = scene.camera.horizontal.xyz;
    let vertical = scene.camera.vertical.xyz;
    let lower_left = scene.camera.lower_left_corner.xyz;
    
    let center = lower_left + horizontal * 0.5 + vertical * 0.5;
    let forward_vec = center - eye;
    let focal_length = length(forward_vec);
    let forward = normalize(forward_vec);
    
    let plane_w = length(horizontal);
    let plane_h = length(vertical);
    
    let right = normalize(horizontal);
    let up = normalize(vertical);

    let view_dir = pos - eye;
    let z_view = dot(view_dir, normalize(center - eye));
    let x_view = dot(view_dir, right);
    let y_view = dot(view_dir, up);

    let z_near = 0.001;
    let z_far = 10000.0;
    
    // Depth mapping to [0, 1] for WebGPU
    let z_clip = z_view * (z_far / (z_far - z_near)) - (z_far * z_near) / (z_far - z_near);

    let proj_pos = vec4<f32>(
        x_view * (focal_length / (plane_w * 0.5)),
        y_view * (focal_length / (plane_h * 0.5)),
        z_clip,
        z_view
    );

    var out: VertexOutput;
    out.position = proj_pos;
    
    // Apply jitter in NDC space to exactly match Raytracer's primary ray
    out.position.x -= scene.jitter.x * 2.0 * out.position.w;
    out.position.y += scene.jitter.y * 2.0 * out.position.w;
    
    out.color = tri.data0.rgb; // BaseColor
    out.normal = norm;
    out.uv = local_uv;
    out.tex_id = tri.data2.x; // BaseTex ID
    out.instance_id = instance_index; // Pass array index, not application ID!
    out.tri_idx = tri_idx;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> GBufferOutput {
    var albedo = in.color;
    if (in.tex_id > -0.5) {
        let tex_col = textureSampleLevel(tex, smp, in.uv, i32(in.tex_id), 0.0).rgb;
        albedo *= tex_col;
    }

    var out: GBufferOutput;
    out.albedo = vec4<f32>(albedo, 1.0);
    out.normal_and_id = vec4<f32>(pack_normal(normalize(in.normal)), bitcast<f32>(in.tri_idx), bitcast<f32>(in.instance_id));
    return out;
}`;
  class Y {
    constructor(e) {
      __publicField(this, "pipeline");
      __publicField(this, "bindGroupLayout");
      __publicField(this, "bindGroup");
      __publicField(this, "ctx");
      this.ctx = e;
    }
    buildPipeline() {
      const e = this.ctx.device.createShaderModule({
        label: "Rasterizer Shader",
        code: X
      });
      this.pipeline = this.ctx.device.createRenderPipeline({
        label: "Rasterizer Pipeline",
        layout: "auto",
        vertex: {
          module: e,
          entryPoint: "vs_main"
        },
        fragment: {
          module: e,
          entryPoint: "fs_main",
          targets: [
            {
              format: "rgba8unorm"
            },
            {
              format: "rgba32float"
            }
          ]
        },
        primitive: {
          topology: "triangle-list",
          cullMode: "none"
        },
        depthStencil: {
          depthWriteEnabled: true,
          depthCompare: "less",
          format: "depth32float"
        }
      }), this.bindGroupLayout = this.pipeline.getBindGroupLayout(0);
    }
    updateBindGroup(e) {
      !e.sceneUniformBuffer || !e.geometryBuffer || !e.topologyBuffer || !e.instanceBuffer || (this.bindGroup = this.ctx.device.createBindGroup({
        layout: this.bindGroupLayout,
        entries: [
          {
            binding: 2,
            resource: {
              buffer: e.sceneUniformBuffer
            }
          },
          {
            binding: 3,
            resource: {
              buffer: e.geometryBuffer,
              offset: 0,
              size: e.vertexCount * 16
            }
          },
          {
            binding: 4,
            resource: {
              buffer: e.topologyBuffer
            }
          },
          {
            binding: 6,
            resource: {
              buffer: e.instanceBuffer
            }
          },
          {
            binding: 7,
            resource: e.texture.createView({
              dimension: "2d-array"
            })
          },
          {
            binding: 8,
            resource: e.sampler
          },
          {
            binding: 11,
            resource: {
              buffer: e.geometryBuffer,
              offset: e.normOffset,
              size: e.vertexCount * 16
            }
          },
          {
            binding: 12,
            resource: {
              buffer: e.geometryBuffer,
              offset: e.uvOffset,
              size: e.vertexCount * 8
            }
          }
        ]
      }));
    }
    execute(e, n) {
      if (!this.bindGroup || !n.drawCommandBuffer || !n.depthTextureViews[n.historyIndex]) return;
      const t = {
        colorAttachments: [
          {
            view: n.renderTargetView,
            loadOp: "clear",
            storeOp: "store",
            clearValue: {
              r: 0,
              g: 0,
              b: 0,
              a: 0
            }
          },
          {
            view: n.gBufferNormalView,
            loadOp: "clear",
            storeOp: "store",
            clearValue: {
              r: 0,
              g: 0,
              b: 0,
              a: 0
            }
          }
        ],
        depthStencilAttachment: {
          view: n.depthTextureViews[n.historyIndex],
          depthClearValue: 1,
          depthLoadOp: "clear",
          depthStoreOp: "store"
        }
      }, r = e.beginRenderPass(t);
      if (r.setPipeline(this.pipeline), r.setBindGroup(0, this.bindGroup), n.drawCommandsArray) {
        const i = n.drawCommandsArray, s = n.instanceCount;
        for (let a = 0; a < s; a++) {
          const c = i[a * 4 + 0], l = i[a * 4 + 1], d = i[a * 4 + 2], v = i[a * 4 + 3];
          c > 0 && l > 0 && r.draw(c, l, d, v);
        }
      }
      r.end();
    }
  }
  class K {
    constructor(e) {
      __publicField(this, "ctx");
      __publicField(this, "res");
      __publicField(this, "raytracePass");
      __publicField(this, "postProcessPass");
      __publicField(this, "rasterizerPass");
      __publicField(this, "totalFrames", 0);
      this.ctx = new H(e), this.res = new N(this.ctx), this.raytracePass = new F(this.ctx), this.postProcessPass = new V(this.ctx), this.rasterizerPass = new Y(this.ctx);
    }
    get device() {
      return this.ctx.device;
    }
    async init() {
      await this.ctx.init(), this.res.init();
    }
    buildPipeline(e) {
      this.raytracePass.buildPipeline(e), this.postProcessPass.buildPipeline(), this.rasterizerPass.buildPipeline(), this.recreateBindGroup();
    }
    updateScreenSize(e, n) {
      this.ctx.canvas.width = e, this.ctx.canvas.height = n, this.res.updateScreenSize(e, n);
    }
    resetAccumulation() {
      this.res.resetAccumulation();
    }
    async loadTexturesFromWorld(e) {
      await this.res.loadTexturesFromWorld(e);
    }
    updateBuffer(e, n) {
      return this.res.updateBuffer(e, n);
    }
    updateCombinedGeometry(e, n, t) {
      return this.res.updateCombinedGeometry(e, n, t);
    }
    updateCombinedBVH(e, n) {
      return this.res.updateCombinedBVH(e, n);
    }
    updateSceneUniforms(e, n, t) {
      this.res.updateSceneUniforms(e, n, t);
    }
    recreateBindGroup() {
      this.raytracePass.updateBindGroup(this.res), this.postProcessPass.updateBindGroup(this.res), this.rasterizerPass.updateBindGroup(this.res);
    }
    compute(e) {
      this.totalFrames++, this.res.updateFrameUniforms(e, this.totalFrames);
      const n = this.ctx.device.createCommandEncoder();
      this.rasterizerPass.execute(n, this.res), this.raytracePass.execute(n, e), this.ctx.device.queue.submit([
        n.finish()
      ]);
    }
    present() {
      const e = this.ctx.device.createCommandEncoder();
      this.postProcessPass.execute(e);
      try {
        const n = this.ctx.context.getCurrentTexture();
        e.copyTextureToTexture({
          texture: this.res.renderTarget
        }, {
          texture: n
        }, {
          width: this.ctx.canvas.width,
          height: this.ctx.canvas.height,
          depthOrArrayLayers: 1
        });
      } catch (n) {
        console.warn("Skipping present(): Swapchain unavailable or invalid.", n);
      }
      this.ctx.device.queue.submit([
        e.finish()
      ]), this.res.historyIndex = 1 - this.res.historyIndex, this.recreateBindGroup();
    }
    async captureFrame() {
      return this.ctx.captureFrame(this.res.renderTarget);
    }
  }
  function Z(o) {
    return new Worker("/webgpu-raytracer/assets/wasm-worker-BkIi53X6.js", {
      name: o == null ? void 0 : o.name
    });
  }
  class Q {
    constructor() {
      __publicField(this, "worker");
      __publicField(this, "resolveReady", null);
      __publicField(this, "_vertices", new Float32Array(0));
      __publicField(this, "_normals", new Float32Array(0));
      __publicField(this, "_uvs", new Float32Array(0));
      __publicField(this, "_mesh_topology", new Uint32Array(0));
      __publicField(this, "_lights", new Uint32Array(0));
      __publicField(this, "_draw_commands", new Uint32Array(0));
      __publicField(this, "_tlas", new Float32Array(0));
      __publicField(this, "_blas", new Float32Array(0));
      __publicField(this, "_instances", new Float32Array(0));
      __publicField(this, "_cameraData", new Float32Array(24));
      __publicField(this, "_textureCount", 0);
      __publicField(this, "_textures", []);
      __publicField(this, "_animations", []);
      __publicField(this, "hasNewData", false);
      __publicField(this, "hasNewGeometry", false);
      __publicField(this, "pendingUpdate", false);
      __publicField(this, "resolveSceneLoad", null);
      __publicField(this, "updateResolvers", []);
      __publicField(this, "lastWidth", -1);
      __publicField(this, "lastHeight", -1);
      this.worker = new Z(), this.worker.onmessage = this.handleMessage.bind(this);
    }
    get lights() {
      return this._lights;
    }
    get lightCount() {
      return this._lights.length / 2;
    }
    async initWasm() {
      return new Promise((e) => {
        this.resolveReady = e, this.worker.postMessage({
          type: "INIT"
        });
      });
    }
    handleMessage(e) {
      var _a, _b;
      const n = e.data;
      switch (n.type) {
        case "READY":
          console.log("Main: Worker Ready"), (_a = this.resolveReady) == null ? void 0 : _a.call(this);
          break;
        case "SCENE_LOADED":
          this._vertices = n.vertices, this._normals = n.normals, this._uvs = n.uvs, this._mesh_topology = n.mesh_topology, this._lights = n.lights, this._tlas = n.tlas, this._blas = n.blas, this._instances = n.instances, this._draw_commands = n.draw_commands, this._cameraData = n.camera, this._textureCount = n.textureCount, this._textures = n.textures || [], this._animations = n.animations || [], this.hasNewData = true, this.hasNewGeometry = true, (_b = this.resolveSceneLoad) == null ? void 0 : _b.call(this);
          break;
        case "UPDATE_RESULT":
          this._tlas = n.tlas, this._blas = n.blas, this._instances = n.instances, this._lights = n.lights, this._draw_commands = n.draw_commands, this._cameraData = n.camera, n.vertices && (this._vertices = n.vertices, this.hasNewGeometry = true), n.normals && (this._normals = n.normals), n.uvs && (this._uvs = n.uvs), n.mesh_topology && (this._mesh_topology = n.mesh_topology), this.hasNewData = true, this.pendingUpdate = false, this.updateResolvers.forEach((t) => t()), this.updateResolvers = [];
          break;
      }
    }
    getAnimationList() {
      return this._animations;
    }
    getTexture(e) {
      return e >= 0 && e < this._textures.length ? this._textures[e] : null;
    }
    loadScene(e, n, t) {
      return this.lastWidth = -1, this.lastHeight = -1, new Promise((r) => {
        this.resolveSceneLoad = r, this.worker.postMessage({
          type: "LOAD_SCENE",
          sceneName: e,
          objSource: n,
          glbData: t
        }, t ? [
          t.buffer
        ] : []);
      });
    }
    waitForNextUpdate() {
      return new Promise((e) => {
        this.updateResolvers.push(e);
      });
    }
    update(e) {
      this.pendingUpdate || (this.pendingUpdate = true, this.worker.postMessage({
        type: "UPDATE",
        time: e
      }));
    }
    updateCamera(e, n) {
      this.lastWidth === e && this.lastHeight === n || (this.lastWidth = e, this.lastHeight = n, this.worker.postMessage({
        type: "UPDATE_CAMERA",
        width: e,
        height: n
      }));
    }
    loadAnimation(e) {
      this.worker.postMessage({
        type: "LOAD_ANIMATION",
        data: e
      }, [
        e.buffer
      ]);
    }
    setAnimation(e) {
      this.worker.postMessage({
        type: "SET_ANIMATION",
        index: e
      });
    }
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
    get draw_commands() {
      return this._draw_commands;
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
      console.log(`Scene Stats (Worker Proxy): V=${this.vertices.length / 4}, Topo=${this.mesh_topology.length / 12}, I=${this.instances.length / 16}, TLAS=${this.tlas.length / 8}, BLAS=${this.blas.length / 8}, Anim=${this._animations.length}, Lights=${this._lights.length / 2}`);
    }
  }
  class ee {
    constructor(e, n, t) {
      __publicField(this, "isRecording", false);
      __publicField(this, "renderer");
      __publicField(this, "worldBridge");
      __publicField(this, "canvas");
      __publicField(this, "currentBatchSize", 0);
      this.renderer = e, this.worldBridge = n, this.canvas = t;
    }
    get recording() {
      return this.isRecording;
    }
    cancel() {
      this.isRecording && (console.warn("[VideoRecorder] Forces cancelling recording state."), this.isRecording = false);
    }
    async record(e, n, t) {
      if (this.isRecording) return;
      this.isRecording = true;
      const { Muxer: r, ArrayBufferTarget: i } = await U(async () => {
        const { Muxer: l, ArrayBufferTarget: d } = await import("./webm-muxer-MLtUgOCn.js");
        return {
          Muxer: l,
          ArrayBufferTarget: d
        };
      }, []), s = Math.ceil(e.fps * e.duration);
      console.log(`Starting recording: ${s} frames @ ${e.fps}fps (VP9)`);
      const a = new r({
        target: new i(),
        video: {
          codec: "V_VP9",
          width: this.canvas.width,
          height: this.canvas.height,
          frameRate: e.fps
        }
      }), c = new VideoEncoder({
        output: (l, d) => a.addVideoChunk(l, d),
        error: (l) => console.error("VideoEncoder Error:", l)
      });
      c.configure({
        codec: "vp09.00.10.08",
        width: this.canvas.width,
        height: this.canvas.height,
        bitrate: 12e6
      });
      try {
        await this.renderAndEncode(s, e, c, n, e.startFrame || 0), await c.flush(), a.finalize();
        const { buffer: l } = a.target, d = new Blob([
          l
        ], {
          type: "video/webm"
        }), v = URL.createObjectURL(d);
        t(v, d);
      } catch (l) {
        throw console.error("Recording failed:", l), l;
      } finally {
        this.isRecording = false;
      }
    }
    async recordChunks(e, n, t) {
      if (this.isRecording) throw new Error("Already recording");
      this.isRecording = true;
      const r = [], i = Math.ceil(e.fps * e.duration), s = new VideoEncoder({
        output: (a, c) => {
          const l = new Uint8Array(a.byteLength);
          a.copyTo(l), r.push({
            type: a.type,
            timestamp: a.timestamp,
            duration: a.duration,
            data: l.buffer,
            decoderConfig: c == null ? void 0 : c.decoderConfig
          });
        },
        error: (a) => console.error("VideoEncoder Error:", a)
      });
      s.configure({
        codec: "vp09.00.10.08",
        width: this.canvas.width,
        height: this.canvas.height,
        bitrate: 12e6
      });
      try {
        return await this.renderAndEncode(i, e, s, n, e.startFrame || 0, t), await s.flush(), r;
      } finally {
        this.isRecording = false;
      }
    }
    async renderAndEncode(e, n, t, r, i = 0, s) {
      if (s == null ? void 0 : s.aborted) throw new Error("Aborted");
      this.currentBatchSize = n.batch;
      const a = i;
      this.worldBridge.update(a / n.fps), await this.worldBridge.waitForNextUpdate(), await this.updateSceneBuffers();
      for (let c = 0; c < 5; c++) this.renderer.compute(c), this.renderer.present(), await this.renderer.device.queue.onSubmittedWorkDone(), this.renderer.resetAccumulation();
      for (let c = 0; c < e; c++) {
        if (s == null ? void 0 : s.aborted) throw new Error("Aborted");
        r(c, e), await new Promise((d) => setTimeout(d, 0)), await this.updateSceneBuffers();
        let l = null;
        if (c < e - 1) {
          const d = i + c + 1;
          this.worldBridge.update(d / n.fps), l = this.worldBridge.waitForNextUpdate();
        }
        await this.renderFrame(n.spp), t.encodeQueueSize > 5 && await t.flush();
        try {
          const { data: d, width: v, height: p } = await this.renderer.captureFrame(), m = "RGBA", w = new VideoFrame(d, {
            codedWidth: v,
            codedHeight: p,
            format: m,
            timestamp: (i + c) * 1e6 / n.fps,
            duration: 1e6 / n.fps
          });
          t.encode(w, {
            keyFrame: c % n.fps === 0
          }), w.close();
        } catch (d) {
          console.warn(`[VideoRecorder] Frame ${c} skipped: Readback failed.`, d);
        }
        l && await l;
      }
    }
    async updateSceneBuffers() {
      let e = false;
      e || (e = this.renderer.updateCombinedBVH(this.worldBridge.tlas, this.worldBridge.blas)), e || (e = this.renderer.updateBuffer("instance", this.worldBridge.instances)), e || (e = this.renderer.updateCombinedGeometry(this.worldBridge.vertices, this.worldBridge.normals, this.worldBridge.uvs)), e || (e = this.renderer.updateBuffer("topology", this.worldBridge.mesh_topology)), e || (e = this.renderer.updateBuffer("lights", this.worldBridge.lights)), e || (e = this.renderer.updateBuffer("draw_commands", this.worldBridge.draw_commands)), this.worldBridge.updateCamera(this.canvas.width, this.canvas.height), this.renderer.updateSceneUniforms(this.worldBridge.cameraData, 0, this.worldBridge.lightCount), e && this.renderer.recreateBindGroup(), this.renderer.resetAccumulation();
    }
    async renderFrame(e) {
      let n = 0, t = performance.now();
      for (; n < e; ) {
        const r = Math.min(this.currentBatchSize, e - n), i = performance.now();
        for (let p = 0; p < r; p++) this.renderer.compute(n + p);
        n += r;
        const s = performance.now();
        (n >= e || s - t > 100) && (this.renderer.present(), t = s), await this.renderer.device.queue.onSubmittedWorkDone();
        const l = performance.now() - i;
        let d = l > 0 ? 100 / l : 1.5;
        d = Math.min(d, 1.5);
        const v = Math.round(this.currentBatchSize * (0.8 + 0.2 * d));
        this.currentBatchSize = Math.max(1, Math.min(e, Math.min(v, 50)));
      }
    }
  }
  const h = {
    defaultWidth: 720,
    defaultHeight: 480,
    defaultDepth: 10,
    defaultSPP: 1,
    signalingServerUrl: "ws://localhost:8080",
    rtcConfig: {
      iceServers: JSON.parse('[{"urls": "stun:stun.l.google.com:19302"}]')
    },
    ids: {
      canvas: "gpu-canvas",
      renderBtn: "render-btn",
      sceneSelect: "scene-select",
      resWidth: "res-width",
      resHeight: "res-height",
      objFile: "obj-file",
      maxDepth: "max-depth",
      sppFrame: "spp-frame",
      recompileBtn: "recompile-btn",
      updateInterval: "update-interval",
      animSelect: "anim-select",
      recordBtn: "record-btn",
      recFps: "rec-fps",
      recDuration: "rec-duration",
      recSpp: "rec-spp",
      recBatch: "rec-batch",
      distJobBatch: "dist-batch",
      btnHost: "btn-host",
      btnWorker: "btn-worker",
      statusDiv: "status",
      uiToggleBtn: "ui-toggle-btn",
      controlsPanel: "controls-panel"
    }
  }, ne = {
    iceServers: [
      {
        urls: "stun:stun.l.google.com:19302"
      }
    ]
  };
  class E {
    constructor(e, n) {
      __publicField(this, "pc");
      __publicField(this, "dc", null);
      __publicField(this, "remoteId");
      __publicField(this, "sendSignal");
      __publicField(this, "transferLock", Promise.resolve());
      __publicField(this, "receiveBuffer", new Uint8Array(0));
      __publicField(this, "receivedBytes", 0);
      __publicField(this, "sceneMeta", null);
      __publicField(this, "resultMeta", null);
      __publicField(this, "onSceneReceived", null);
      __publicField(this, "onRenderRequest", null);
      __publicField(this, "onRenderResult", null);
      __publicField(this, "onDataChannelOpen", null);
      __publicField(this, "onAckReceived", null);
      __publicField(this, "onWorkerReady", null);
      __publicField(this, "onConnectionFailure", null);
      __publicField(this, "onWorkerStatus", null);
      __publicField(this, "onStopRender", null);
      __publicField(this, "onSceneLoaded", null);
      this.remoteId = e, this.sendSignal = n, this.pc = new RTCPeerConnection(ne), this.pc.onconnectionstatechange = () => {
        console.log(`[RTC] State Change: ${this.pc.connectionState}`), (this.pc.connectionState === "failed" || this.pc.connectionState === "disconnected") && this.onConnectionFailure && this.onConnectionFailure();
      }, this.pc.onicecandidate = (t) => {
        t.candidate && this.sendSignal({
          type: "candidate",
          candidate: t.candidate.toJSON(),
          targetId: this.remoteId
        });
      };
    }
    isDataChannelOpen() {
      var _a;
      return ((_a = this.dc) == null ? void 0 : _a.readyState) === "open";
    }
    async startAsHost() {
      this.dc = this.pc.createDataChannel("render-channel"), this.setupDataChannel();
      const e = await this.pc.createOffer();
      await this.pc.setLocalDescription(e), this.sendSignal({
        type: "offer",
        sdp: e,
        targetId: this.remoteId
      });
    }
    async handleOffer(e) {
      this.pc.ondatachannel = (t) => {
        this.dc = t.channel, this.setupDataChannel();
      }, await this.pc.setRemoteDescription(new RTCSessionDescription(e));
      const n = await this.pc.createAnswer();
      await this.pc.setLocalDescription(n), this.sendSignal({
        type: "answer",
        sdp: n,
        targetId: this.remoteId
      });
    }
    async handleAnswer(e) {
      await this.pc.setRemoteDescription(new RTCSessionDescription(e));
    }
    async handleCandidate(e) {
      await this.pc.addIceCandidate(new RTCIceCandidate(e));
    }
    async sendScene(e, n, t) {
      if (!this.dc || this.dc.readyState !== "open") throw new Error("DataChannel is not open");
      await (this.transferLock = this.transferLock.then(async () => {
        let r;
        typeof e == "string" ? r = new TextEncoder().encode(e) : r = new Uint8Array(e);
        const i = {
          type: "SCENE_INIT",
          totalBytes: r.byteLength,
          config: {
            ...t,
            fileType: n
          }
        };
        await this.sendData(i), await this.sendBinaryChunks(r);
      }).catch((r) => {
        throw console.error("[RTC] sendScene failed:", r), r;
      }));
    }
    async sendRenderResult(e, n) {
      if (!this.dc || this.dc.readyState !== "open") throw new Error("DataChannel is not open");
      await (this.transferLock = this.transferLock.then(async () => {
        let t = 0;
        const r = e.map((a) => {
          const c = a.data.byteLength;
          return t += c, {
            type: a.type,
            timestamp: a.timestamp,
            duration: a.duration,
            size: c,
            decoderConfig: a.decoderConfig
          };
        });
        console.log(`[RTC] Sending Render Result: ${t} bytes, ${e.length} chunks`), await this.sendData({
          type: "RENDER_RESULT",
          startFrame: n,
          totalBytes: t,
          chunksMeta: r
        });
        const i = new Uint8Array(t);
        let s = 0;
        for (const a of e) i.set(new Uint8Array(a.data), s), s += a.data.byteLength;
        await this.sendBinaryChunks(i);
      }).catch((t) => {
        throw console.error("[RTC] sendRenderResult failed:", t), t;
      }));
    }
    async sendBinaryChunks(e) {
      let t = 0;
      const r = () => new Promise((i) => {
        const s = setInterval(() => {
          (!this.dc || this.dc.bufferedAmount < 65536) && (clearInterval(s), i());
        }, 5);
      });
      for (; t < e.byteLength; ) {
        this.dc && this.dc.bufferedAmount > 256 * 1024 && await r();
        const i = Math.min(t + 16384, e.byteLength);
        if (this.dc) try {
          this.dc.send(e.subarray(t, i));
        } catch {
        }
        t = i, t % (16384 * 5) === 0 && await new Promise((s) => setTimeout(s, 0));
      }
      console.log("[RTC] Transfer Complete");
    }
    setupDataChannel() {
      this.dc && (this.dc.binaryType = "arraybuffer", this.dc.onopen = () => {
        console.log("[RTC] DataChannel Open"), this.onDataChannelOpen && this.onDataChannelOpen();
      }, this.dc.onmessage = (e) => {
        const n = e.data;
        if (typeof n == "string") try {
          const t = JSON.parse(n);
          this.handleControlMessage(t);
        } catch {
        }
        else n instanceof ArrayBuffer && this.handleBinaryChunk(n);
      });
    }
    handleControlMessage(e) {
      var _a, _b, _c, _d, _e, _f;
      e.type === "SCENE_INIT" ? (console.log(`[RTC] Receiving Scene: ${e.config.fileType}, ${e.totalBytes} bytes`), this.sceneMeta = {
        config: e.config,
        totalBytes: e.totalBytes
      }, this.receiveBuffer = new Uint8Array(e.totalBytes), this.receivedBytes = 0) : e.type === "SCENE_ACK" ? (console.log(`[RTC] Scene ACK: ${e.receivedBytes} bytes`), this.onAckReceived && this.onAckReceived(e.receivedBytes)) : e.type === "RENDER_REQUEST" ? (console.log(`[RTC] Render Request: Frame ${e.startFrame}, Count ${e.frameCount}`), (_a = this.onRenderRequest) == null ? void 0 : _a.call(this, e.startFrame, e.frameCount, e.config)) : e.type === "RENDER_RESULT" ? (console.log(`[RTC] Receiving Render Result: ${e.totalBytes} bytes`), this.resultMeta = {
        startFrame: e.startFrame,
        totalBytes: e.totalBytes,
        chunksMeta: e.chunksMeta
      }, this.receiveBuffer = new Uint8Array(e.totalBytes), this.receivedBytes = 0) : e.type === "WORKER_READY" ? (console.log("[RTC] Worker Ready Signal Received"), (_b = this.onWorkerReady) == null ? void 0 : _b.call(this)) : e.type === "WORKER_STATUS" ? (console.log(`[RTC] Worker Status Received: hasScene=${e.hasScene}, job=${(_c = e.currentJob) == null ? void 0 : _c.start}`), (_d = this.onWorkerStatus) == null ? void 0 : _d.call(this, e.hasScene, e.currentJob)) : e.type === "STOP_RENDER" ? (console.log("[RTC] Stop Render Signal Received"), (_e = this.onStopRender) == null ? void 0 : _e.call(this)) : e.type === "SCENE_LOADED" && (console.log("[RTC] Scene Loaded Signal Received"), (_f = this.onSceneLoaded) == null ? void 0 : _f.call(this));
    }
    handleBinaryChunk(e) {
      var _a, _b;
      try {
        const n = new Uint8Array(e);
        if (this.receivedBytes + n.byteLength > this.receiveBuffer.byteLength) {
          console.error("[RTC] Receive Buffer Overflow!");
          return;
        }
        this.receiveBuffer.set(n, this.receivedBytes), this.receivedBytes += n.byteLength;
      } catch (n) {
        console.error("[RTC] Error handling binary chunk", n);
        return;
      }
      if (this.sceneMeta) {
        if (this.receivedBytes >= this.sceneMeta.totalBytes) {
          console.log("[RTC] Scene Download Complete!");
          let n;
          this.sceneMeta.config.fileType === "obj" ? n = new TextDecoder().decode(this.receiveBuffer) : n = this.receiveBuffer.buffer, (_a = this.onSceneReceived) == null ? void 0 : _a.call(this, n, this.sceneMeta.config), this.sceneMeta = null;
        }
      } else if (this.resultMeta && this.receivedBytes >= this.resultMeta.totalBytes) {
        console.log("[RTC] Render Result Complete!");
        const n = [];
        let t = 0;
        for (const r of this.resultMeta.chunksMeta) {
          const i = this.receiveBuffer.slice(t, t + r.size);
          n.push({
            type: r.type,
            timestamp: r.timestamp,
            duration: r.duration,
            data: i.buffer,
            decoderConfig: r.decoderConfig
          }), t += r.size;
        }
        (_b = this.onRenderResult) == null ? void 0 : _b.call(this, n, this.resultMeta.startFrame), this.resultMeta = null;
      }
    }
    async sendData(e) {
      var _a;
      ((_a = this.dc) == null ? void 0 : _a.readyState) === "open" && (this.dc.bufferedAmount > 1024 * 1024 && await new Promise((n) => {
        const t = setInterval(() => {
          (!this.dc || this.dc.bufferedAmount < 524288) && (clearInterval(t), n());
        }, 10);
      }), this.dc.send(JSON.stringify(e)));
    }
    sendAck(e) {
      this.sendData({
        type: "SCENE_ACK",
        receivedBytes: e
      });
    }
    sendRenderRequest(e, n, t) {
      const r = {
        type: "RENDER_REQUEST",
        startFrame: e,
        frameCount: n,
        config: t
      };
      this.sendData(r);
    }
    sendWorkerReady() {
      this.sendData({
        type: "WORKER_READY"
      });
    }
    sendWorkerStatus(e, n) {
      this.sendData({
        type: "WORKER_STATUS",
        hasScene: e,
        currentJob: n
      });
    }
    sendStopRender() {
      this.sendData({
        type: "STOP_RENDER"
      });
    }
    sendSceneLoaded() {
      this.sendData({
        type: "SCENE_LOADED"
      });
    }
    close() {
      this.dc && (this.dc.close(), this.dc = null), this.pc && this.pc.close(), console.log(`[RTC] Connection closed: ${this.remoteId}`);
    }
  }
  class te {
    constructor() {
      __publicField(this, "ws", null);
      __publicField(this, "myRole", null);
      __publicField(this, "workers", /* @__PURE__ */ new Map());
      __publicField(this, "hostClient", null);
      __publicField(this, "onStatusChange", null);
      __publicField(this, "onWorkerJoined", null);
      __publicField(this, "onWorkerLeft", null);
      __publicField(this, "onHostConnected", null);
      __publicField(this, "onWorkerReady", null);
      __publicField(this, "onSceneReceived", null);
      __publicField(this, "onHostHello", null);
      __publicField(this, "onRenderResult", null);
      __publicField(this, "onRenderRequest", null);
      __publicField(this, "onWorkerStatus", null);
      __publicField(this, "onStopRender", null);
      __publicField(this, "onSceneLoaded", null);
    }
    connect(e) {
      var _a;
      if (this.ws) return;
      this.myRole = e, (_a = this.onStatusChange) == null ? void 0 : _a.call(this, `Connecting as ${e.toUpperCase()}...`);
      const n = "secretpassword";
      this.ws = new WebSocket(`${h.signalingServerUrl}?token=${n}`), this.ws.onopen = () => {
        var _a2;
        if (console.log("WS Connected"), (_a2 = this.onStatusChange) == null ? void 0 : _a2.call(this, `Waiting for Peer (${e.toUpperCase()})`), e === "worker") {
          const t = sessionStorage.getItem("raytracer_session_id"), r = sessionStorage.getItem("raytracer_session_token");
          this.sendSignal({
            type: "register_worker",
            sessionId: t || void 0,
            sessionToken: r || void 0
          });
        } else this.sendSignal({
          type: "register_host"
        });
      }, this.ws.onmessage = (t) => {
        const r = JSON.parse(t.data);
        this.handleMessage(r);
      }, this.ws.onclose = () => {
        var _a2;
        (_a2 = this.onStatusChange) == null ? void 0 : _a2.call(this, "Disconnected"), this.ws = null;
      };
    }
    disconnect() {
      var _a;
      this.ws && (this.ws.close(), this.ws = null), this.workers.forEach((e) => e.close()), this.workers.clear(), this.hostClient && (this.hostClient.close(), this.hostClient = null), (_a = this.onStatusChange) == null ? void 0 : _a.call(this, "Disconnected");
    }
    getWorkerCount() {
      return this.workers.size;
    }
    getWorkerIds() {
      return Array.from(this.workers.keys());
    }
    async sendRenderResult(e, n) {
      if (this.hostClient) await this.hostClient.sendRenderResult(e, n);
      else throw new Error("No Host Connection");
    }
    sendSignal(e) {
      var _a;
      ((_a = this.ws) == null ? void 0 : _a.readyState) === WebSocket.OPEN && this.ws.send(JSON.stringify(e));
    }
    async handleMessage(e) {
      this.myRole === "host" ? await this.handleHostMessage(e) : await this.handleWorkerMessage(e);
    }
    async handleHostMessage(e) {
      var _a, _b, _c, _d, _e;
      switch (e.type) {
        case "worker_joined":
          console.log(`Worker joined: ${e.workerId}`);
          const n = (r) => {
            const i = new E(r, (s) => this.sendSignal(s));
            return this.workers.set(r, i), i.onDataChannelOpen = () => {
              var _a2;
              console.log(`[Host] Open for ${r}`), i.sendData({
                type: "HELLO",
                msg: "Hello from Host!"
              }), (_a2 = this.onWorkerJoined) == null ? void 0 : _a2.call(this, r);
            }, i.onAckReceived = (s) => {
              console.log(`Worker ${r} ACK: ${s}`);
            }, i.onRenderResult = (s, a) => {
              var _a2;
              console.log(`Received Render Result from ${r}: ${s.length} chunks`), (_a2 = this.onRenderResult) == null ? void 0 : _a2.call(this, s, a, r);
            }, i.onWorkerReady = () => {
              var _a2;
              (_a2 = this.onWorkerReady) == null ? void 0 : _a2.call(this, r);
            }, i.onWorkerStatus = (s, a) => {
              var _a2;
              (_a2 = this.onWorkerStatus) == null ? void 0 : _a2.call(this, r, s, a);
            }, i.onStopRender = () => {
              var _a2;
              (_a2 = this.onStopRender) == null ? void 0 : _a2.call(this);
            }, i.onSceneLoaded = () => {
              var _a2;
              (_a2 = this.onSceneLoaded) == null ? void 0 : _a2.call(this, r);
            }, i.onConnectionFailure = () => {
              console.warn(`[Host] Connection failed for ${r}. Retrying...`), i.close(), setTimeout(() => {
                var _a2;
                this.workers.has(r) && (n(r), (_a2 = this.workers.get(r)) == null ? void 0 : _a2.startAsHost());
              }, 2e3);
            }, i;
          };
          await n(e.workerId).startAsHost();
          break;
        case "worker_left":
          console.log(`Worker left: ${e.workerId}`), (_a = this.workers.get(e.workerId)) == null ? void 0 : _a.close(), this.workers.delete(e.workerId), (_b = this.onWorkerLeft) == null ? void 0 : _b.call(this, e.workerId);
          break;
        case "answer":
          e.fromId && await ((_c = this.workers.get(e.fromId)) == null ? void 0 : _c.handleAnswer(e.sdp));
          break;
        case "candidate":
          e.fromId && await ((_d = this.workers.get(e.fromId)) == null ? void 0 : _d.handleCandidate(e.candidate));
          break;
        case "host_exists":
          alert("Host already exists!");
          break;
        case "WORKER_READY":
          e.workerId && ((_e = this.onWorkerReady) == null ? void 0 : _e.call(this, e.workerId));
          break;
      }
    }
    async sendWorkerReady() {
      this.hostClient && this.hostClient.sendWorkerReady();
    }
    async sendWorkerStatus(e, n) {
      this.hostClient && this.hostClient.sendWorkerStatus(e, n);
    }
    async handleWorkerMessage(e) {
      var _a;
      switch (e.type) {
        case "session_info":
          console.log(`[Worker] Session Info Received: ${e.sessionId}`), sessionStorage.setItem("raytracer_session_id", e.sessionId), sessionStorage.setItem("raytracer_session_token", e.sessionToken);
          break;
        case "offer":
          e.fromId && await ((r) => {
            var _a2, _b;
            return this.hostClient && this.hostClient.close(), this.hostClient = new E(r, (i) => this.sendSignal(i)), (_a2 = this.onStatusChange) == null ? void 0 : _a2.call(this, "Connected to Host!"), (_b = this.onHostConnected) == null ? void 0 : _b.call(this), this.hostClient.onDataChannelOpen = () => {
              var _a3, _b2;
              (_a3 = this.hostClient) == null ? void 0 : _a3.sendData({
                type: "HELLO",
                msg: "Hello from Worker!"
              }), (_b2 = this.onHostHello) == null ? void 0 : _b2.call(this);
            }, this.hostClient.onSceneReceived = (i, s) => {
              var _a3, _b2;
              (_a3 = this.onSceneReceived) == null ? void 0 : _a3.call(this, i, s);
              const a = typeof i == "string" ? i.length : i.byteLength;
              (_b2 = this.hostClient) == null ? void 0 : _b2.sendAck(a);
            }, this.hostClient.onRenderRequest = (i, s, a) => {
              var _a3;
              (_a3 = this.onRenderRequest) == null ? void 0 : _a3.call(this, i, s, a);
            }, this.hostClient.onStopRender = () => {
              var _a3;
              (_a3 = this.onStopRender) == null ? void 0 : _a3.call(this);
            }, this.hostClient.onSceneLoaded = () => {
              var _a3;
              (_a3 = this.onSceneLoaded) == null ? void 0 : _a3.call(this, "host");
            }, this.hostClient.onConnectionFailure = () => {
              var _a3;
              console.warn(`[Worker] Connection failed for host ${r}.`), this.hostClient && (this.hostClient.close(), this.hostClient = null), (_a3 = this.onStatusChange) == null ? void 0 : _a3.call(this, "Disconnected from Host (Reconnecting...)");
            }, this.hostClient;
          })(e.fromId).handleOffer(e.sdp);
          break;
        case "candidate":
          await ((_a = this.hostClient) == null ? void 0 : _a.handleCandidate(e.candidate));
          break;
      }
    }
    async broadcastScene(e, n, t) {
      const r = Array.from(this.workers.values()).map((i) => i.sendScene(e, n, t));
      await Promise.all(r);
    }
    async sendSceneToWorker(e, n, t, r) {
      const i = this.workers.get(e);
      if (!i) {
        console.error(`[Host] Cannot send scene to ${e}: Client not found.`);
        return;
      }
      i.isDataChannelOpen() || (console.log(`[Host] DataChannel for ${e} not ready, waiting...`), await new Promise((s) => {
        var _a;
        const a = () => {
          var _a2;
          (_a2 = i.dc) == null ? void 0 : _a2.removeEventListener("open", a), s();
        };
        (_a = i.dc) == null ? void 0 : _a.addEventListener("open", a), setTimeout(s, 3e3);
      }), i.isDataChannelOpen() || console.warn(`[Host] DataChannel for ${e} timed out. Attempting Send anyway...`)), i && await i.sendScene(n, t, r);
    }
    async sendRenderRequest(e, n, t, r) {
      const i = this.workers.get(e);
      i && await i.sendRenderRequest(n, t, r);
    }
    sendStopRender(e) {
      const n = this.workers.get(e);
      n && n.sendStopRender();
    }
    sendRenderStart() {
      this.sendSignal({
        type: "render_start"
      });
    }
    sendRenderStop() {
      this.sendSignal({
        type: "render_stop"
      });
    }
    sendSceneLoaded() {
      this.hostClient && this.hostClient.isDataChannelOpen() ? (console.log("[Signaling] Sending WORKER_READY via DataChannel"), this.hostClient.sendWorkerReady()) : (console.warn("[Signaling] DataChannel not ready, trying WebSocket for WORKER_READY (Server might not forward)"), this.sendSignal({
        type: "WORKER_READY"
      }));
    }
  }
  class re {
    constructor() {
      __publicField(this, "canvas");
      __publicField(this, "btnRender");
      __publicField(this, "sceneSelect");
      __publicField(this, "inputWidth");
      __publicField(this, "inputHeight");
      __publicField(this, "inputFile");
      __publicField(this, "inputDepth");
      __publicField(this, "inputSPP");
      __publicField(this, "btnRecompile");
      __publicField(this, "inputUpdateInterval");
      __publicField(this, "animSelect");
      __publicField(this, "btnRecord");
      __publicField(this, "inputRecFps");
      __publicField(this, "inputRecDur");
      __publicField(this, "inputRecSpp");
      __publicField(this, "inputRecBatch");
      __publicField(this, "inputDistJobBatch");
      __publicField(this, "btnHost");
      __publicField(this, "btnWorker");
      __publicField(this, "statusDiv");
      __publicField(this, "btnToggleUI");
      __publicField(this, "controlsPanel");
      __publicField(this, "statsDiv");
      __publicField(this, "onRenderStart", null);
      __publicField(this, "onRenderStop", null);
      __publicField(this, "onSceneSelect", null);
      __publicField(this, "onResolutionChange", null);
      __publicField(this, "onRecompile", null);
      __publicField(this, "onFileSelect", null);
      __publicField(this, "onAnimSelect", null);
      __publicField(this, "onRecordStart", null);
      __publicField(this, "onConnectHost", null);
      __publicField(this, "onConnectWorker", null);
      this.canvas = this.el(h.ids.canvas), this.btnRender = this.el(h.ids.renderBtn), this.sceneSelect = this.el(h.ids.sceneSelect), this.inputWidth = this.el(h.ids.resWidth), this.inputHeight = this.el(h.ids.resHeight), this.inputFile = this.setupFileInput(), this.inputDepth = this.el(h.ids.maxDepth), this.inputSPP = this.el(h.ids.sppFrame), this.btnRecompile = this.el(h.ids.recompileBtn), this.inputUpdateInterval = this.el(h.ids.updateInterval), this.animSelect = this.el(h.ids.animSelect), this.btnRecord = this.el(h.ids.recordBtn), this.inputRecFps = this.el(h.ids.recFps), this.inputRecDur = this.el(h.ids.recDuration), this.inputRecSpp = this.el(h.ids.recSpp), this.inputRecBatch = this.el(h.ids.recBatch), this.inputDistJobBatch = this.el(h.ids.distJobBatch), this.btnHost = this.el(h.ids.btnHost), this.btnWorker = this.el(h.ids.btnWorker), this.statusDiv = this.el(h.ids.statusDiv), this.btnToggleUI = this.el(h.ids.uiToggleBtn), this.controlsPanel = this.el(h.ids.controlsPanel), this.statsDiv = this.createStatsDiv(), this.bindEvents();
    }
    el(e) {
      const n = document.getElementById(e);
      if (!n) throw new Error(`Element not found: ${e}`);
      return n;
    }
    setupFileInput() {
      const e = this.el(h.ids.objFile);
      return e && (e.accept = ".obj,.glb,.vrm"), e;
    }
    createStatsDiv() {
      const e = document.createElement("div");
      return Object.assign(e.style, {
        position: "fixed",
        bottom: "10px",
        left: "10px",
        color: "#0f0",
        background: "rgba(0,0,0,0.7)",
        padding: "8px",
        fontFamily: "monospace",
        fontSize: "14px",
        pointerEvents: "none",
        zIndex: "9999",
        borderRadius: "4px"
      }), document.body.appendChild(e), e;
    }
    bindEvents() {
      this.btnRender.addEventListener("click", () => {
        var _a, _b;
        this.btnRender.textContent === "Render Start" || this.btnRender.textContent === "Resume Rendering" ? ((_a = this.onRenderStart) == null ? void 0 : _a.call(this), this.updateRenderButton(true)) : ((_b = this.onRenderStop) == null ? void 0 : _b.call(this), this.updateRenderButton(false));
      }), this.sceneSelect.addEventListener("change", () => {
        var _a;
        return (_a = this.onSceneSelect) == null ? void 0 : _a.call(this, this.sceneSelect.value);
      });
      const e = () => {
        var _a;
        return (_a = this.onResolutionChange) == null ? void 0 : _a.call(this, parseInt(this.inputWidth.value) || h.defaultWidth, parseInt(this.inputHeight.value) || h.defaultHeight);
      };
      this.inputWidth.addEventListener("change", e), this.inputHeight.addEventListener("change", e), this.btnRecompile.addEventListener("click", () => {
        var _a;
        return (_a = this.onRecompile) == null ? void 0 : _a.call(this, parseInt(this.inputDepth.value) || 10, parseInt(this.inputSPP.value) || 1);
      }), this.inputFile.addEventListener("change", (n) => {
        var _a, _b;
        const t = (_a = n.target.files) == null ? void 0 : _a[0];
        t && ((_b = this.onFileSelect) == null ? void 0 : _b.call(this, t));
      }), this.animSelect.addEventListener("change", () => {
        var _a;
        const n = parseInt(this.animSelect.value, 10);
        (_a = this.onAnimSelect) == null ? void 0 : _a.call(this, n);
      }), this.btnRecord.addEventListener("click", () => {
        var _a;
        return (_a = this.onRecordStart) == null ? void 0 : _a.call(this);
      }), this.btnHost.addEventListener("click", () => {
        var _a;
        return (_a = this.onConnectHost) == null ? void 0 : _a.call(this);
      }), this.btnWorker.addEventListener("click", () => {
        var _a;
        return (_a = this.onConnectWorker) == null ? void 0 : _a.call(this);
      }), this.btnToggleUI.addEventListener("click", () => {
        this.controlsPanel.classList.toggle("collapsed");
      });
    }
    updateRenderButton(e) {
      this.btnRender.textContent = e ? "Stop Rendering" : "Resume Rendering";
    }
    updateStats(e, n, t) {
      this.statsDiv.textContent = `FPS: ${e} | ${n.toFixed(2)}ms | Frame: ${t}`;
    }
    setStatus(e) {
      this.statusDiv.textContent = e;
    }
    setConnectionState(e) {
      e === "host" ? (this.btnHost.textContent = "Disconnect", this.btnHost.disabled = false, this.btnWorker.textContent = "Worker", this.btnWorker.disabled = true) : e === "worker" ? (this.btnHost.textContent = "Host", this.btnHost.disabled = true, this.btnWorker.textContent = "Disconnect", this.btnWorker.disabled = false) : (this.btnHost.textContent = "Host", this.btnHost.disabled = false, this.btnWorker.textContent = "Worker", this.btnWorker.disabled = false, this.statusDiv.textContent = "Offline");
    }
    setRecordingState(e, n) {
      e ? (this.btnRecord.disabled = true, this.btnRecord.textContent = n || "Recording...", this.btnRender.textContent = "Resume Rendering") : (this.btnRecord.disabled = false, this.btnRecord.textContent = "\u25CF Rec");
    }
    updateAnimList(e) {
      if (this.animSelect.innerHTML = "", e.length === 0) {
        const n = document.createElement("option");
        n.text = "No Anim", this.animSelect.add(n), this.animSelect.disabled = true;
        return;
      }
      this.animSelect.disabled = false, e.forEach((n, t) => {
        const r = document.createElement("option");
        r.text = `[${t}] ${n}`, r.value = t.toString(), this.animSelect.add(r);
      }), this.animSelect.value = "0";
    }
    getRenderConfig() {
      return {
        width: parseInt(this.inputWidth.value, 10) || h.defaultWidth,
        height: parseInt(this.inputHeight.value, 10) || h.defaultHeight,
        fps: parseInt(this.inputRecFps.value, 10) || 30,
        duration: parseFloat(this.inputRecDur.value) || 3,
        spp: parseInt(this.inputRecSpp.value, 10) || 64,
        batch: parseInt(this.inputRecBatch.value, 10) || 4,
        jobBatch: parseInt(this.inputDistJobBatch.value, 10) || 20,
        anim: parseInt(this.animSelect.value, 10) || 0,
        maxDepth: parseInt(this.inputDepth.value, 10) || h.defaultDepth,
        shaderSpp: parseInt(this.inputSPP.value, 10) || h.defaultSPP
      };
    }
    setRenderConfig(e) {
      this.inputWidth.value = e.width.toString(), this.inputHeight.value = e.height.toString(), this.inputRecFps.value = e.fps.toString(), this.inputRecDur.value = e.duration.toString(), this.inputRecSpp.value = e.spp.toString(), this.inputRecBatch.value = e.batch.toString(), e.jobBatch !== void 0 && (this.inputDistJobBatch.value = e.jobBatch.toString()), e.maxDepth !== void 0 && (this.inputDepth.value = e.maxDepth.toString()), e.shaderSpp !== void 0 && (this.inputSPP.value = e.shaderSpp.toString());
    }
  }
  class ie {
    constructor(e, n) {
      __publicField(this, "jobQueue", []);
      __publicField(this, "pendingChunks", /* @__PURE__ */ new Map());
      __publicField(this, "completedJobs", 0);
      __publicField(this, "totalJobs", 0);
      __publicField(this, "totalRenderFrames", 0);
      __publicField(this, "distributedConfig", null);
      __publicField(this, "workerStatus", /* @__PURE__ */ new Map());
      __publicField(this, "activeJobs", /* @__PURE__ */ new Map());
      __publicField(this, "signaling");
      __publicField(this, "ui");
      __publicField(this, "disconnectedWorkers", /* @__PURE__ */ new Map());
      __publicField(this, "GRACE_PERIOD_MS", 3e4);
      __publicField(this, "cachedSceneData", null);
      this.signaling = e, this.ui = n, this.setupSignaling();
    }
    setupSignaling() {
      this.signaling.onWorkerLeft = (e) => this.onWorkerLeft(e), this.signaling.onWorkerReady = (e) => this.onWorkerReady(e), this.signaling.onWorkerJoined = (e) => this.onWorkerJoined(e), this.signaling.onWorkerStatus = (e, n, t) => this.onWorkerStatus(e, n, t), this.signaling.onSceneLoaded = (e) => this.onSceneLoaded(e), this.signaling.onRenderResult = (e, n, t) => this.onRenderResult(e, n, t);
    }
    async sendSceneHelper(e, n, t) {
      const r = this.ui.sceneSelect.value, i = r !== "viewer";
      if (!i && (!e || !n)) return;
      const s = this.ui.getRenderConfig(), a = i ? r : void 0, c = i ? "DUMMY" : e, l = i ? "obj" : n;
      s.sceneName = a, s.fileType = l, this.cachedSceneData = {
        data: c,
        type: l,
        config: {
          ...s
        }
      }, t ? (console.log(`[Host] Sending scene to specific worker: ${t}`), this.workerStatus.set(t, "loading"), await this.signaling.sendSceneToWorker(t, c, l, s)) : (console.log("[Host] Broadcasting scene to all workers..."), this.signaling.getWorkerIds().forEach((d) => this.workerStatus.set(d, "loading")), await this.signaling.broadcastScene(c, l, s));
    }
    async assignJob(e) {
      const n = this.workerStatus.get(e);
      if (console.log(`[Host] Attempting to assign job to ${e} (Status: ${n})`), n !== "idle") {
        console.log(`[Host] Worker ${e} is not idle. Aborting assignment.`);
        return;
      }
      if (this.jobQueue.length === 0) {
        console.log(`[Host] Job queue is empty. No work to assign to ${e}.`);
        return;
      }
      if (!this.distributedConfig) {
        console.warn("[Host] Distributed config is missing. Cannot assign job.");
        return;
      }
      const t = this.jobQueue.shift();
      this.workerStatus.set(e, "busy"), this.activeJobs.set(e, t), console.log(`[Host] Assigning job to ${e}: Frames ${t.start} - ${t.start + t.count}`);
      try {
        await this.signaling.sendRenderRequest(e, t.start, t.count, this.distributedConfig);
      } catch (r) {
        console.error(`[Host] Failed to send job to ${e}, re-queuing`, r), this.jobQueue.push(t), this.workerStatus.set(e, "idle"), this.activeJobs.delete(e), setTimeout(() => this.assignJob(e), 2e3);
      }
    }
    triggerAssignments() {
      for (const [e, n] of this.workerStatus.entries()) n === "idle" && this.assignJob(e);
    }
    onWorkerLeft(e) {
      console.log(`[Host] Worker ${e} left.`);
      const n = this.activeJobs.get(e);
      if (n) {
        console.log(`[Host] Worker ${e} had active job. Starting grace period.`);
        const t = window.setTimeout(() => {
          console.log(`[Host] Grace period expired for ${e}. Re-queuing job.`), this.jobQueue.push(n), this.disconnectedWorkers.delete(e), this.activeJobs.delete(e), this.workerStatus.delete(e), this.triggerAssignments();
        }, this.GRACE_PERIOD_MS);
        this.disconnectedWorkers.set(e, {
          job: n,
          timeoutId: t
        });
      } else this.workerStatus.delete(e), this.activeJobs.delete(e);
    }
    onWorkerReady(e) {
      if (console.log(`[Host] Worker ${e} is ready (Manual Signal).`), this.activeJobs.has(e)) {
        console.log(`[Host] Worker ${e} sent READY but has active job. Assuming it is processing pending request.`), this.workerStatus.set(e, "busy");
        return;
      }
      this.workerStatus.set(e, "idle"), this.assignJob(e);
    }
    onWorkerJoined(e) {
      console.log(`[Host] Worker ${e} joined.`), this.workerStatus.set(e, "loading");
      const n = this.disconnectedWorkers.get(e);
      n ? (console.log(`[Host] Worker ${e} re-joined. Resuming job.`), clearTimeout(n.timeoutId), this.activeJobs.set(e, n.job), this.disconnectedWorkers.delete(e)) : this.cachedSceneData && (console.log(`[Host] Auto-sending cached scene to new worker ${e}`), this.signaling.sendSceneToWorker(e, this.cachedSceneData.data, this.cachedSceneData.type, this.cachedSceneData.config));
    }
    async onWorkerStatus(e, n, t) {
      if (console.log(`[Host] Worker ${e} status update: hasScene=${n}`, t), !n) return this.workerStatus.get(e) === "busy" && console.warn(`[Host] Worker ${e} reports no scene while host thinks it is busy. Re-syncing.`), console.log(`[Host] Worker ${e} has no scene. Syncing...`), "NEED_SCENE";
      !t && this.workerStatus.get(e) !== "busy" ? this.workerStatus.get(e) === "loading" ? console.log(`[Host] Worker ${e} is still loading scene.`) : (this.workerStatus.set(e, "idle"), await this.assignJob(e)) : t && (this.workerStatus.set(e, "busy"), this.activeJobs.set(e, t));
    }
    async onSceneLoaded(e) {
      if (this.workerStatus.get(e) !== "loading") {
        console.log(`[Host] Ignore redundant SCENE_LOADED from ${e} (Status: ${this.workerStatus.get(e)})`);
        return;
      }
      console.log(`[Host] Worker ${e} loaded the scene.`), this.workerStatus.set(e, "idle"), await this.assignJob(e);
    }
    async onRenderResult(e, n, t) {
      if (this.pendingChunks.has(n)) {
        console.warn(`[Host] Ignore duplicate result for ${n} from ${t}`), this.workerStatus.set(t, "idle"), this.activeJobs.delete(t), await this.assignJob(t);
        return;
      }
      if (console.log(`[Host] Received ${e.length} chunks for ${n} from ${t}`), this.pendingChunks.set(n, e), this.completedJobs++, this.ui.setStatus(`Distributed Progress: ${this.completedJobs} / ${this.totalJobs} jobs`), this.workerStatus.set(t, "idle"), this.activeJobs.delete(t), await this.assignJob(t), this.completedJobs >= this.totalJobs) return console.log("[Host] All jobs complete. Triggering Muxing Callback."), "ALL_COMPLETE";
    }
    async muxAndDownload() {
      const e = Array.from(this.pendingChunks.keys()).sort((l, d) => l - d), { Muxer: n, ArrayBufferTarget: t } = await U(async () => {
        const { Muxer: l, ArrayBufferTarget: d } = await import("./webm-muxer-MLtUgOCn.js");
        return {
          Muxer: l,
          ArrayBufferTarget: d
        };
      }, []), r = new n({
        target: new t(),
        video: {
          codec: "V_VP9",
          width: this.distributedConfig.width,
          height: this.distributedConfig.height,
          frameRate: this.distributedConfig.fps
        }
      });
      for (const l of e) {
        const d = this.pendingChunks.get(l);
        if (d) for (const v of d) r.addVideoChunk(new EncodedVideoChunk({
          type: v.type,
          timestamp: v.timestamp,
          duration: v.duration,
          data: v.data
        }), {
          decoderConfig: v.decoderConfig
        });
      }
      r.finalize();
      const { buffer: i } = r.target, s = new Blob([
        i
      ], {
        type: "video/webm"
      }), a = URL.createObjectURL(s), c = document.createElement("a");
      c.href = a, c.download = `distributed_render_${Date.now()}.webm`, c.click(), URL.revokeObjectURL(a), this.ui.setStatus("Distributed Render Complete."), this.signaling.sendRenderStop();
    }
  }
  class ae {
    constructor(e, n, t, r) {
      __publicField(this, "isSceneLoading", false);
      __publicField(this, "isDistributedSceneLoaded", false);
      __publicField(this, "pendingRenderRequest", null);
      __publicField(this, "currentWorkerJob", null);
      __publicField(this, "onRemoteSceneLoad", null);
      __publicField(this, "signaling");
      __publicField(this, "renderer");
      __publicField(this, "ui");
      __publicField(this, "recorder");
      __publicField(this, "bufferedResults", []);
      __publicField(this, "currentWorkerAbortController", null);
      this.signaling = e, this.renderer = n, this.ui = t, this.recorder = r, this.setupSignaling();
    }
    setupSignaling() {
      this.signaling.onHostHello = () => this.onHostHello(), this.signaling.onRenderRequest = (e, n, t) => this.onRenderRequest(e, n, t), this.signaling.onStopRender = () => this.onStopRender(), this.signaling.onSceneReceived = (e, n) => this.onSceneReceived(e, n);
    }
    async executeWorkerRender(e, n, t) {
      if (this.recorder.isRecording) {
        console.warn("[Worker] Already recording/rendering, skipping request.");
        return;
      }
      this.currentWorkerAbortController && this.currentWorkerAbortController.abort(), this.currentWorkerAbortController = new AbortController();
      const r = this.currentWorkerAbortController.signal;
      if (this.isSceneLoading || !this.isDistributedSceneLoaded) {
        console.log(`[Worker] Scene loading (or not synced) in progress. Queueing Render Request for ${e}`), this.pendingRenderRequest = {
          start: e,
          count: n,
          config: t
        };
        return;
      }
      this.currentWorkerJob = {
        start: e,
        count: n
      }, console.log(`[Worker] Starting Render: Frames ${e} - ${e + n}`), this.ui.setStatus(`Remote Rendering: ${e}-${e + n}`), t.maxDepth !== void 0 && t.shaderSpp !== void 0 && (console.log(`[Worker] Updating Shader Pipeline: Depth=${t.maxDepth}, SPP=${t.shaderSpp}`), this.renderer.buildPipeline(t.maxDepth));
      const i = {
        ...t,
        startFrame: e,
        duration: n / t.fps
      };
      try {
        this.ui.setRecordingState(true, `Remote: ${n} f`);
        const s = await this.recorder.recordChunks(i, (a, c) => this.ui.setRecordingState(true, `Remote: ${a}/${c}`), r);
        console.log(`[Worker] Render Finished for ${e}. Sending results.`), await this.signaling.sendRenderResult(s, e), this.currentWorkerJob = null;
      } catch (s) {
        s.name === "AbortError" ? console.log(`[Worker] Render Aborted for ${e}`) : (console.error("[Worker] Remote Recording Failed", s), this.ui.setStatus("Recording Failed"));
      } finally {
        this.currentWorkerJob = null, this.currentWorkerAbortController = null, this.ui.updateRenderButton(false), this.ui.setRecordingState(false);
      }
    }
    async trySendBufferedResults() {
      if (this.bufferedResults.length === 0) return;
      console.log(`[Worker] Retrying to send ${this.bufferedResults.length} buffered results...`);
      const e = [];
      for (const n of this.bufferedResults) try {
        await this.signaling.sendRenderResult(n.chunks, n.startFrame);
      } catch {
        e.push(n);
      }
      this.bufferedResults = e;
    }
    handlePendingRenderRequest() {
      if (this.pendingRenderRequest) {
        console.log(`[Worker] Processing Pending Render Request: ${this.pendingRenderRequest.start}`);
        const e = this.pendingRenderRequest;
        this.pendingRenderRequest = null, this.executeWorkerRender(e.start, e.count, e.config);
      }
    }
    onHostHello() {
      console.log("[Worker] Host Hello received."), this.signaling.sendWorkerStatus(this.isDistributedSceneLoaded, this.currentWorkerJob || void 0);
    }
    onRenderRequest(e, n, t) {
      this.executeWorkerRender(e, n, t);
    }
    onStopRender() {
      console.log("[Worker] Stop Render received."), this.currentWorkerAbortController && this.currentWorkerAbortController.abort();
    }
    async onSceneReceived(e, n) {
      return console.log("[Worker] Scene received successfully."), this.recorder.cancel(), this.isSceneLoading = true, this.ui.setRenderConfig(n), n.maxDepth !== void 0 && n.shaderSpp !== void 0 && (console.log(`[Worker] Syncing Shader settings: Depth=${n.maxDepth}, SPP=${n.shaderSpp}`), this.renderer.buildPipeline(n.maxDepth)), this.onRemoteSceneLoad && (await this.onRemoteSceneLoad(e, n.fileType || "obj"), await new Promise((t) => setTimeout(t, 500))), this.isSceneLoading = false, this.isDistributedSceneLoaded = true, this.signaling.sendSceneLoaded(), {
        data: e,
        config: n
      };
    }
  }
  let b = false, y = null, B = null, R = null;
  const u = new re(), f = new K(u.canvas), _ = new Q(), k = new ee(f, _, u.canvas), x = new te(), g = new ie(x, u), z = new ae(x, f, u, k);
  z.onRemoteSceneLoad = async (o, e) => {
    y = o, B = e, await T("viewer", false), console.log("[Main] Remote scene loaded via dWorker callback.");
  };
  let S = 0, D = 0, C = 0, W = performance.now();
  const se = () => {
    const o = parseInt(u.inputDepth.value, 10) || h.defaultDepth;
    f.buildPipeline(o);
  }, A = () => {
    const { width: o, height: e } = u.getRenderConfig();
    f.updateScreenSize(o, e), _.hasWorld && (_.updateCamera(o, e), f.updateSceneUniforms(_.cameraData, 0, _.lightCount)), f.recreateBindGroup(), f.resetAccumulation(), S = 0, D = 0;
  }, T = async (o, e = true) => {
    b = false, console.log(`Loading Scene: ${o}...`);
    let n, t;
    o === "viewer" && y && (B === "obj" ? n = y : B === "glb" && (t = new Uint8Array(y).slice(0))), await _.loadScene(o, n, t), _.printStats(), await f.loadTexturesFromWorld(_), await oe(), A(), u.updateAnimList(_.getAnimationList()), e && (b = true, u.updateRenderButton(true));
  }, oe = async () => {
    f.updateCombinedGeometry(_.vertices, _.normals, _.uvs), f.updateCombinedBVH(_.tlas, _.blas), f.updateBuffer("topology", _.mesh_topology), f.updateBuffer("instance", _.instances), f.updateBuffer("lights", _.lights), f.updateBuffer("draw_commands", _.draw_commands), f.updateSceneUniforms(_.cameraData, 0, _.lightCount), await f.device.queue.onSubmittedWorkDone();
  }, L = () => {
    if (k.recording || (requestAnimationFrame(L), !b || !_.hasWorld)) return;
    let o = parseInt(u.inputUpdateInterval.value, 10) || 0;
    if (o > 0 && S >= o && _.update(D / (o || 1) / 60), _.hasNewData) {
      let n = false;
      n || (n = f.updateCombinedBVH(_.tlas, _.blas)), n || (n = f.updateBuffer("instance", _.instances)), n || (n = f.updateBuffer("draw_commands", _.draw_commands)), _.hasNewGeometry && (n || (n = f.updateCombinedGeometry(_.vertices, _.normals, _.uvs)), n || (n = f.updateBuffer("topology", _.mesh_topology)), n || (n = f.updateBuffer("lights", _.lights)), _.hasNewGeometry = false), _.updateCamera(u.canvas.width, u.canvas.height), f.updateSceneUniforms(_.cameraData, 0, _.lightCount), n && f.recreateBindGroup(), f.resetAccumulation(), S = 0, _.hasNewData = false;
    }
    S++, C++, D++, f.compute(S), f.present();
    const e = performance.now();
    e - W >= 1e3 && (u.updateStats(C, 1e3 / C, S), C = 0, W = e);
  };
  x.onStatusChange = (o) => u.setStatus(`Status: ${o}`);
  x.onWorkerStatus = async (o, e, n) => {
    await g.onWorkerStatus(o, e, n) === "NEED_SCENE" && (console.log(`[Host] Worker ${o} needs scene. Syncing...`), await g.sendSceneHelper(y, B, o));
  };
  x.onRenderResult = async (o, e, n) => {
    await g.onRenderResult(o, e, n) === "ALL_COMPLETE" && (console.log("[Host] All jobs complete. Muxing and downloading..."), u.setStatus("Muxing..."), await g.muxAndDownload());
  };
  x.onSceneReceived = async (o, e) => {
    console.log("[Worker] Received Scene from Host."), await z.onSceneReceived(o, e), u.sceneSelect.value = e.sceneName || "viewer", e.anim !== void 0 && (u.animSelect.value = e.anim.toString(), _.setAnimation(e.anim)), z.isDistributedSceneLoaded = true, z.isSceneLoading = false, console.log("[Worker] Distributed Scene Loaded. Signaling Host."), await x.sendSceneLoaded(), z.handlePendingRenderRequest();
  };
  const ce = () => {
    u.onRenderStart = () => {
      b = true;
    }, u.onRenderStop = () => {
      b = false;
    }, u.onSceneSelect = (o) => T(o, false), u.onResolutionChange = A, u.onRecompile = (o, e) => {
      b = false, f.buildPipeline(o), f.recreateBindGroup(), f.resetAccumulation(), S = 0, b = true;
    }, u.onFileSelect = async (o) => {
      var _a;
      ((_a = o.name.split(".").pop()) == null ? void 0 : _a.toLowerCase()) === "obj" ? (y = await o.text(), B = "obj") : (y = await o.arrayBuffer(), B = "glb"), u.sceneSelect.value = "viewer", T("viewer", false);
    }, u.onAnimSelect = (o) => _.setAnimation(o), u.onRecordStart = async () => {
      if (!k.recording) if (R === "host") {
        const o = x.getWorkerIds();
        g.distributedConfig = u.getRenderConfig();
        const e = Math.ceil(g.distributedConfig.fps * g.distributedConfig.duration);
        if (!confirm(`Distribute recording? (Workers: ${o.length})
Auto Scene Sync enabled.`)) return;
        g.jobQueue = [], g.pendingChunks.clear(), g.completedJobs = 0, g.activeJobs.clear();
        const n = g.distributedConfig.jobBatch || 20;
        for (let t = 0; t < e; t += n) {
          const r = Math.min(n, e - t);
          g.jobQueue.push({
            start: t,
            count: r
          });
        }
        g.totalJobs = g.jobQueue.length, o.forEach((t) => g.workerStatus.set(t, "idle")), u.setStatus(`Distributed Progress: 0 / ${g.totalJobs} jobs (Waiting for workers...)`), o.length > 0 ? (u.setStatus("Syncing Scene to Workers..."), x.sendRenderStart(), await g.sendSceneHelper(y, B)) : console.log("No workers yet. Waiting...");
      } else {
        b = false, u.setRecordingState(true);
        const o = u.getRenderConfig();
        try {
          const e = performance.now();
          await k.record(o, (n, t) => u.setRecordingState(true, `Rec: ${n}/${t} (${Math.round(n / t * 100)}%)`), (n) => {
            const t = document.createElement("a");
            t.href = n, t.download = `raytrace_${Date.now()}.webm`, t.click(), URL.revokeObjectURL(n);
          }), console.log(`Recording took ${performance.now() - e}[ms]`);
        } catch {
          alert("Recording failed.");
        } finally {
          u.setRecordingState(false), b = false, u.updateRenderButton(false), requestAnimationFrame(L);
        }
      }
    }, u.onConnectHost = () => {
      R === "host" ? (x.disconnect(), R = null, u.setConnectionState(null)) : (x.connect("host"), R = "host", u.setConnectionState("host"));
    }, u.onConnectWorker = () => {
      R === "worker" ? (x.disconnect(), R = null, u.setConnectionState(null)) : (k.cancel(), x.connect("worker"), R = "worker", u.setConnectionState("worker"));
    }, u.setConnectionState(null);
  };
  async function le() {
    try {
      await f.init(), await _.initWasm();
    } catch (o) {
      alert("Init failed: " + o);
      return;
    }
    ce(), se(), A(), T("cornell", false), requestAnimationFrame(L);
  }
  le().catch(console.error);
})();
