var __defProp = Object.defineProperty;
var __defNormalProp = (obj, key, value) => key in obj ? __defProp(obj, key, { enumerable: true, configurable: true, writable: true, value }) : obj[key] = value;
var __publicField = (obj, key, value) => __defNormalProp(obj, typeof key !== "symbol" ? key + "" : key, value);
(function() {
  const e = document.createElement("link").relList;
  if (e && e.supports && e.supports("modulepreload")) return;
  for (const s of document.querySelectorAll('link[rel="modulepreload"]')) r(s);
  new MutationObserver((s) => {
    for (const a of s) if (a.type === "childList") for (const o of a.addedNodes) o.tagName === "LINK" && o.rel === "modulepreload" && r(o);
  }).observe(document, { childList: true, subtree: true });
  function i(s) {
    const a = {};
    return s.integrity && (a.integrity = s.integrity), s.referrerPolicy && (a.referrerPolicy = s.referrerPolicy), s.crossOrigin === "use-credentials" ? a.credentials = "include" : s.crossOrigin === "anonymous" ? a.credentials = "omit" : a.credentials = "same-origin", a;
  }
  function r(s) {
    if (s.ep) return;
    s.ep = true;
    const a = i(s);
    fetch(s.href, a);
  }
})();
const ki = "modulepreload", Ci = function(t) {
  return "/webgpu-raytracer/" + t;
}, $t = {}, xi = function(e, i, r) {
  let s = Promise.resolve();
  if (i && i.length > 0) {
    let m = function(C) {
      return Promise.all(C.map((f) => Promise.resolve(f).then((g) => ({ status: "fulfilled", value: g }), (g) => ({ status: "rejected", reason: g }))));
    };
    document.getElementsByTagName("link");
    const o = document.querySelector("meta[property=csp-nonce]"), l = (o == null ? void 0 : o.nonce) || (o == null ? void 0 : o.getAttribute("nonce"));
    s = m(i.map((C) => {
      if (C = Ci(C), C in $t) return;
      $t[C] = true;
      const f = C.endsWith(".css"), g = f ? '[rel="stylesheet"]' : "";
      if (document.querySelector(`link[href="${C}"]${g}`)) return;
      const v = document.createElement("link");
      if (v.rel = f ? "stylesheet" : ki, f || (v.as = "script"), v.crossOrigin = "", v.href = C, l && v.setAttribute("nonce", l), document.head.appendChild(v), f) return new Promise((N, V) => {
        v.addEventListener("load", N), v.addEventListener("error", () => V(new Error(`Unable to preload CSS for ${C}`)));
      });
    }));
  }
  function a(o) {
    const l = new Event("vite:preloadError", { cancelable: true });
    if (l.payload = o, window.dispatchEvent(l), !l.defaultPrevented) throw o;
  }
  return s.then((o) => {
    for (const l of o || []) l.status === "rejected" && a(l.reason);
    return e().catch(a);
  });
}, Ri = `// =========================================================
//   WebGPU Ray Tracer (TLAS & BLAS)
// =========================================================

const PI = 3.141592653589793;
const T_MIN = 0.001;
const T_MAX = 1e30;
// These are replaced by TypeScript
const MAX_DEPTH = 10u;
const SPP = 1u;

// --- Structs ---

struct Camera {
    origin: vec3<f32>,
    lens_radius: f32,
    lower_left_corner: vec3<f32>,
    horizontal: vec3<f32>,
    vertical: vec3<f32>,
    u: vec3<f32>,
    v: vec3<f32>
}

struct SceneUniforms {
    camera: Camera,
    frame_count: u32,
    blas_base_idx: u32, // Start index of BLAS nodes in 'nodes' array
    vertex_count: u32,
    pad2: u32
}

struct TriangleAttributes {
    data0: vec4<f32>,
    data1: vec4<f32>
}

struct BVHNode {
    min_b: vec3<f32>,
    left_first: f32, // [TLAS] Child/Inst Idx, [BLAS] Child/Tri Idx
    max_b: vec3<f32>,
    tri_count: f32   // [TLAS] Leaf(1)/Int(0), [BLAS] Count
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

// --- Bindings (Consolidated) ---

@group(0) @binding(0) var outputTex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<storage, read_write> accumulateBuffer: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> scene: SceneUniforms;

@group(0) @binding(3) var<storage, read> geometry: array<f32>; // [Pos(4)... | Norm(4)... | UV(2)...]
@group(0) @binding(4) var<storage, read> indices: array<u32>;
@group(0) @binding(5) var<storage, read> attributes: array<TriangleAttributes>;
@group(0) @binding(6) var<storage, read> nodes: array<BVHNode>; // Merged TLAS/BLAS
@group(0) @binding(7) var<storage, read> instances: array<Instance>;

@group(0) @binding(8) var tex: texture_2d_array<f32>;
@group(0) @binding(9) var smp: sampler;

// --- Helpers ---

// Accessors
fn get_pos(idx: u32) -> vec3<f32> {
    let i = idx * 4u;
    return vec3(geometry[i], geometry[i + 1u], geometry[i + 2u]);
}
fn get_normal(idx: u32) -> vec3<f32> {
    let i = (scene.vertex_count * 4u) + (idx * 4u);
    return vec3(geometry[i], geometry[i + 1u], geometry[i + 2u]);
}
fn get_uv(idx: u32) -> vec2<f32> {
    let i = (scene.vertex_count * 8u) + (idx * 2u);
    return vec2(geometry[i], geometry[i + 1u]);
}

struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>
}

struct HitResult {
    t: f32,
    tri_idx: f32,
    inst_idx: i32
}

fn get_inv_transform(inst: Instance) -> mat4x4<f32> {
    return mat4x4<f32>(inst.inv_0, inst.inv_1, inst.inv_2, inst.inv_3);
}

// --- Random ---
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
fn random_unit_vector(rng: ptr<function, u32>) -> vec3<f32> {
    let z = rand_pcg(rng) * 2.0 - 1.0; let a = rand_pcg(rng) * 2.0 * PI;
    let r = sqrt(max(0.0, 1.0 - z * z)); return vec3<f32>(r * cos(a), r * sin(a), z);
}
fn random_in_unit_disk(rng: ptr<function, u32>) -> vec3<f32> {
    let r = sqrt(rand_pcg(rng)); let theta = 2.0 * PI * rand_pcg(rng);
    return vec3<f32>(r * cos(theta), r * sin(theta), 0.0);
}
fn reflectance(cosine: f32, ref_idx: f32) -> f32 {
    var r0 = (1.0 - ref_idx) / (1.0 + ref_idx); r0 = r0 * r0;
    return r0 + (1.0 - r0) * pow((1.0 - cosine), 5.0);
}

// --- Intersection ---

fn intersect_aabb(min_b: vec3<f32>, max_b: vec3<f32>, r: Ray, inv_d: vec3<f32>, t_min: f32, t_max: f32) -> f32 {
    let t0s = (min_b - r.origin) * inv_d;
    let t1s = (max_b - r.origin) * inv_d;
    let t_small = min(t0s, t1s);
    let t_big = max(t0s, t1s);
    let tmin = max(t_min, max(t_small.x, max(t_small.y, t_small.z)));
    let tmax = min(t_max, min(t_big.x, min(t_big.y, t_big.z)));
    return select(1e30, tmin, tmin <= tmax);
}

fn hit_triangle_raw(v0: vec3<f32>, v1: vec3<f32>, v2: vec3<f32>, r: Ray, t_min: f32, t_max: f32) -> f32 {
    let e1 = v1 - v0; let e2 = v2 - v0;
    let h = cross(r.direction, e2); let a = dot(e1, h);
    if abs(a) < 1e-4 { return -1.0; }
    let f = 1.0 / a; let s = r.origin - v0; let u = f * dot(s, h);
    if u < 0.0 || u > 1.0 { return -1.0; }
    let q = cross(s, e1); let v = f * dot(r.direction, q);
    if v < 0.0 || u + v > 1.0 { return -1.0; }
    let t = f * dot(e2, q);
    if t > t_min && t < t_max { return t; }
    return -1.0;
}

fn intersect_blas(r: Ray, t_min: f32, t_max: f32, node_start_idx: u32) -> vec2<f32> {
    var closest_t = t_max;
    var hit_idx = -1.0;
    let inv_d = 1.0 / r.direction;
    var stack: array<u32, 48>;
    var stackptr = 0u;

    // Root Node (Global Index in 'nodes' array)
    let root_idx = node_start_idx;
    let root = nodes[root_idx];

    if intersect_aabb(root.min_b, root.max_b, r, inv_d, t_min, closest_t) < 1e30 {
        stack[stackptr] = root_idx;
        stackptr++;
    }

    while stackptr > 0u {
        stackptr--;
        let idx = stack[stackptr];
        let node = nodes[idx];
        let count = u32(node.tri_count);

        if count > 0u {
            // Leaf Node
            let first = u32(node.left_first); // Triangle Index (Sorted)
            for (var i = 0u; i < count; i++) {
                let tri_id = first + i;
                let b = tri_id * 3u;
                
                // Get vertices from geometry buffer
                let v0 = get_pos(indices[b]);
                let v1 = get_pos(indices[b + 1u]);
                let v2 = get_pos(indices[b + 2u]);

                let t = hit_triangle_raw(v0, v1, v2, r, t_min, closest_t);
                if t > 0.0 { closest_t = t; hit_idx = f32(tri_id); }
            }
        } else {
            // Internal Node
            // node.left_first is RELATIVE to the start of this BLAS.
            let l = u32(node.left_first) + node_start_idx;
            let r_node_idx = l + 1u;

            let nl = nodes[l];
            let nr = nodes[r_node_idx];

            let dl = intersect_aabb(nl.min_b, nl.max_b, r, inv_d, t_min, closest_t);
            let dr = intersect_aabb(nr.min_b, nr.max_b, r, inv_d, t_min, closest_t);

            let hl = dl < 1e30; let hr = dr < 1e30;
            if hl && hr {
                if dl < dr { stack[stackptr] = r_node_idx; stackptr++; stack[stackptr] = l; stackptr++; } else { stack[stackptr] = l; stackptr++; stack[stackptr] = r_node_idx; stackptr++; }
            } else if hl { stack[stackptr] = l; stackptr++; } else if hr { stack[stackptr] = r_node_idx; stackptr++; }
        }
    }
    return vec2<f32>(closest_t, hit_idx);
}

fn intersect_tlas(r: Ray, t_min: f32, t_max: f32) -> HitResult {
    var res: HitResult; res.t = t_max; res.tri_idx = -1.0; res.inst_idx = -1;
    // Need at least one node.
    // Ideally we pass tlas_count, but checking arrayLength of storage buffer 'nodes' is not precise for sub-allocations.
    // Assuming root is at 0.
    // If scene.blas_base_idx == 0, then tlas is empty or something weird.
    if scene.blas_base_idx == 0u { return res; }

    let inv_d = 1.0 / r.direction;
    var stack: array<u32, 24>;
    var stackptr = 0u;

    if intersect_aabb(nodes[0].min_b, nodes[0].max_b, r, inv_d, t_min, res.t) < 1e30 {
        stack[stackptr] = 0u; stackptr++;
    }

    while stackptr > 0u {
        stackptr--;
        let idx = stack[stackptr];
        let node = nodes[idx];

        if node.tri_count > 0.5 { // Leaf (Instance)
            let inst_idx = u32(node.left_first);
            let inst = instances[inst_idx];
            let inv = get_inv_transform(inst);

            let r_local = Ray((inv * vec4(r.origin, 1.0)).xyz, (inv * vec4(r.direction, 0.0)).xyz);
            
            // BLAS start = global BLAS base + instance specific offset
            let blas_start = scene.blas_base_idx + inst.blas_node_offset;
            let blas = intersect_blas(r_local, t_min, res.t, blas_start);

            if blas.y > -0.5 {
                res.t = blas.x;
                res.tri_idx = blas.y;
                res.inst_idx = i32(inst_idx);
            }
        } else {
            // Internal (TLAS)
            let l = u32(node.left_first);
            let r_idx = l + 1u;
            let nl = nodes[l];
            let nr = nodes[r_idx];
            let dl = intersect_aabb(nl.min_b, nl.max_b, r, inv_d, t_min, res.t);
            let dr = intersect_aabb(nr.min_b, nr.max_b, r, inv_d, t_min, res.t);
            if dl < 1e30 && dr < 1e30 {
                if dl < dr {
                    stack[stackptr] = r_idx; stackptr++; stack[stackptr] = l; stackptr++;
                } else {
                    stack[stackptr] = l; stackptr++; stack[stackptr] = r_idx; stackptr++;
                }
            } else if dl < 1e30 {
                stack[stackptr] = l; stackptr++;
            } else if dr < 1e30 {
                stack[stackptr] = r_idx; stackptr++;
            }
        }
    }
    return res;
}

fn ray_color(r_in: Ray, rng: ptr<function, u32>) -> vec3<f32> {
    var ray = r_in;
    var throughput = vec3<f32>(1.0);

    for (var depth = 0u; depth < MAX_DEPTH; depth++) {
        let hit = intersect_tlas(ray, T_MIN, T_MAX);
        if hit.inst_idx < 0 { return vec3<f32>(0.0); }

        let inst = instances[u32(hit.inst_idx)];
        let tri_idx = u32(hit.tri_idx);

        let i0 = indices[tri_idx * 3u];
        let i1 = indices[tri_idx * 3u + 1u];
        let i2 = indices[tri_idx * 3u + 2u];

        // Retrieve properties from separate geometry blocks
        let v0_pos = get_pos(i0);
        let v1_pos = get_pos(i1);
        let v2_pos = get_pos(i2);

        // Normal Interpolation
        let inv = get_inv_transform(inst);
        let r_local = Ray((inv * vec4(ray.origin, 1.)).xyz, (inv * vec4(ray.direction, 0.)).xyz);

        let e1 = v1_pos - v0_pos;
        let e2 = v2_pos - v0_pos;
        let h = cross(r_local.direction, e2);
        let a = dot(e1, h);
        let f = 1.0 / a;
        let s = r_local.origin - v0_pos;
        let u = f * dot(s, h);
        let q = cross(s, e1);
        let v = f * dot(r_local.direction, q);
        let w = 1.0 - u - v;

        // Load Normals
        let n0 = get_normal(i0);
        let n1 = get_normal(i1);
        let n2 = get_normal(i2);

        let ln = normalize(n0 * w + n1 * u + n2 * v);
        let wn = normalize((vec4(ln, 0.0) * inv).xyz);

        var n = wn;
        let front = dot(ray.direction, n) < 0.0;
        n = select(-n, n, front);

        // Interpolate UV
        let uv0 = get_uv(i0);
        let uv1 = get_uv(i1);
        let uv2 = get_uv(i2);

        let tex_uv = uv0 * w + uv1 * u + uv2 * v;

        // Attributes
        let attr = attributes[tri_idx];
        let albedo = attr.data0.rgb;
        let mat_type = bitcast<u32>(attr.data0.w);

        if mat_type == 3u { return select(vec3(0.), throughput * albedo, front); }

        var scat = vec3(0.);
        if mat_type == 0u {
            scat = n + random_unit_vector(rng);
            if length(scat) < 0.001 { scat = n; }
        } else if mat_type == 1u {
            scat = reflect(ray.direction, n) + attr.data1.x * random_unit_vector(rng);
            if dot(scat, n) <= 0. { return vec3(0.); }
        } else {
            let ir = attr.data1.x;
            let ratio = select(ir, 1.0 / ir, front);
            let unit = normalize(ray.direction);
            let cos_t = min(dot(-unit, n), 1.0);
            let sin_t = sqrt(1.0 - cos_t * cos_t);
            if ratio * sin_t > 1.0 || reflectance(cos_t, ratio) > rand_pcg(rng) {
                scat = reflect(unit, n);
            } else {
                scat = ratio * (unit + cos_t * n) - sqrt(abs(1.0 - (1.0 - cos_t * cos_t) * ratio * ratio)) * n;
            }
        }

        ray = Ray(ray.origin + hit.t * ray.direction + scat * 1e-4, scat);
        let tex_idx = attr.data1.y;
        var tex_color = vec3(1.0);
        if tex_idx > -0.5 {
            tex_color = textureSampleLevel(tex, smp, tex_uv, i32(tex_idx), 0.0).rgb;
        }
        let final_albedo = albedo * tex_color;

        throughput *= final_albedo;

        if depth > 2u {
            let p = max(throughput.r, max(throughput.g, throughput.b));
            if rand_pcg(rng) > p { break; }
            throughput /= p;
        }
    }
    return vec3(0.);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(outputTex);
    if id.x >= dims.x || id.y >= dims.y { return; }
    let p_idx = id.y * dims.x + id.x;
    var rng = init_rng(p_idx, scene.frame_count);

    var col = vec3(0.);
    for (var s = 0u; s < SPP; s++) {
        var off = vec3(0.);
        if scene.camera.lens_radius > 0. {
            let rd = scene.camera.lens_radius * random_in_unit_disk(&rng);
            off = scene.camera.u * rd.x + scene.camera.v * rd.y;
        }
        let u = (f32(id.x) + rand_pcg(&rng)) / f32(dims.x);
        let v = 1. - (f32(id.y) + rand_pcg(&rng)) / f32(dims.y);
        let d = scene.camera.lower_left_corner + u * scene.camera.horizontal + v * scene.camera.vertical - scene.camera.origin - off;
        col += ray_color(Ray(scene.camera.origin + off, d), &rng);
    }
    col /= f32(SPP);

    var acc = vec4(0.);
    if scene.frame_count > 1u { acc = accumulateBuffer[p_idx]; }
    let new_acc = acc + vec4(col, 1.0);
    accumulateBuffer[p_idx] = new_acc;

    let out = sqrt(clamp(new_acc.rgb / new_acc.a, vec3(0.), vec3(1.)));
    textureStore(outputTex, vec2<i32>(id.xy), vec4(out, 1.));
}
`;
class Bi {
  constructor(e) {
    __publicField(this, "device");
    __publicField(this, "context");
    __publicField(this, "pipeline");
    __publicField(this, "bindGroupLayout");
    __publicField(this, "bindGroup");
    __publicField(this, "renderTarget");
    __publicField(this, "renderTargetView");
    __publicField(this, "accumulateBuffer");
    __publicField(this, "sceneUniformBuffer");
    __publicField(this, "geometryBuffer");
    __publicField(this, "nodesBuffer");
    __publicField(this, "indexBuffer");
    __publicField(this, "attrBuffer");
    __publicField(this, "instanceBuffer");
    __publicField(this, "texture");
    __publicField(this, "defaultTexture");
    __publicField(this, "sampler");
    __publicField(this, "bufferSize", 0);
    __publicField(this, "canvas");
    __publicField(this, "blasOffset", 0);
    __publicField(this, "vertexCount", 0);
    this.canvas = e;
  }
  async init() {
    if (!navigator.gpu) throw new Error("WebGPU not supported.");
    const e = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!e) throw new Error("No adapter");
    console.log("Max Storage Buffers Per Shader Stage:", e.limits.maxStorageBuffersPerShaderStage), this.device = await e.requestDevice({ requiredLimits: { maxStorageBuffersPerShaderStage: 10 } }), this.context = this.canvas.getContext("webgpu"), this.context.configure({ device: this.device, format: "rgba8unorm", usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT }), this.sceneUniformBuffer = this.device.createBuffer({ size: 128, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }), this.sampler = this.device.createSampler({ magFilter: "linear", minFilter: "linear", mipmapFilter: "linear", addressModeU: "repeat", addressModeV: "repeat" }), this.createDefaultTexture(), this.texture = this.defaultTexture;
  }
  createDefaultTexture() {
    const e = new Uint8Array([255, 255, 255, 255]);
    this.defaultTexture = this.device.createTexture({ size: [1, 1, 1], format: "rgba8unorm", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST }), this.device.queue.writeTexture({ texture: this.defaultTexture, origin: [0, 0, 0] }, e, { bytesPerRow: 256, rowsPerImage: 1 }, [1, 1]);
  }
  buildPipeline(e, i) {
    let r = Ri;
    r = r.replace(/const\s+MAX_DEPTH\s*=\s*\d+u;/, `const MAX_DEPTH = ${e}u;`), r = r.replace(/const\s+SPP\s*=\s*\d+u;/, `const SPP = ${i}u;`);
    const s = this.device.createShaderModule({ label: "RayTracing", code: r });
    this.pipeline = this.device.createComputePipeline({ label: "Main Pipeline", layout: "auto", compute: { module: s, entryPoint: "main" } }), this.bindGroupLayout = this.pipeline.getBindGroupLayout(0);
  }
  updateScreenSize(e, i) {
    this.canvas.width = e, this.canvas.height = i, this.renderTarget && this.renderTarget.destroy(), this.renderTarget = this.device.createTexture({ size: [e, i], format: "rgba8unorm", usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC }), this.renderTargetView = this.renderTarget.createView(), this.bufferSize = e * i * 16, this.accumulateBuffer && this.accumulateBuffer.destroy(), this.accumulateBuffer = this.device.createBuffer({ size: this.bufferSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  }
  resetAccumulation() {
    this.accumulateBuffer && this.device.queue.writeBuffer(this.accumulateBuffer, 0, new Float32Array(this.bufferSize / 4));
  }
  async loadTexturesFromWorld(e) {
    const i = e.textureCount;
    if (i === 0) {
      this.createDefaultTexture();
      return;
    }
    console.log(`Loading ${i} textures...`);
    const r = [];
    for (let s = 0; s < i; s++) {
      const a = e.getTexture(s);
      if (a) try {
        const o = new Blob([a]), l = await createImageBitmap(o, { resizeWidth: 1024, resizeHeight: 1024 });
        r.push(l);
      } catch (o) {
        console.warn(`Failed tex ${s}`, o), r.push(await this.createFallbackBitmap());
      }
      else r.push(await this.createFallbackBitmap());
    }
    this.texture && this.texture.destroy(), this.texture = this.device.createTexture({ size: [1024, 1024, r.length], format: "rgba8unorm", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT });
    for (let s = 0; s < r.length; s++) this.device.queue.copyExternalImageToTexture({ source: r[s] }, { texture: this.texture, origin: [0, 0, s] }, [1024, 1024]);
  }
  async createFallbackBitmap() {
    const e = document.createElement("canvas");
    e.width = 1024, e.height = 1024;
    const i = e.getContext("2d");
    return i.fillStyle = "white", i.fillRect(0, 0, 1024, 1024), await createImageBitmap(e);
  }
  ensureBuffer(e, i, r) {
    if (e && e.size >= i) return e;
    e && e.destroy();
    let s = Math.ceil(i * 1.5);
    return s = s + 3 & -4, s = Math.max(s, 4), this.device.createBuffer({ label: r, size: s, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  }
  updateBuffer(e, i) {
    const r = i.byteLength;
    let s = false, a;
    return e === "index" ? ((!this.indexBuffer || this.indexBuffer.size < r) && (s = true), this.indexBuffer = this.ensureBuffer(this.indexBuffer, r, "IndexBuffer"), a = this.indexBuffer) : e === "attr" ? ((!this.attrBuffer || this.attrBuffer.size < r) && (s = true), this.attrBuffer = this.ensureBuffer(this.attrBuffer, r, "AttrBuffer"), a = this.attrBuffer) : ((!this.instanceBuffer || this.instanceBuffer.size < r) && (s = true), this.instanceBuffer = this.ensureBuffer(this.instanceBuffer, r, "InstanceBuffer"), a = this.instanceBuffer), this.device.queue.writeBuffer(a, 0, i, 0, i.length), s;
  }
  updateCombinedGeometry(e, i, r) {
    const s = e.byteLength + i.byteLength + r.byteLength;
    let a = false;
    (!this.geometryBuffer || this.geometryBuffer.size < s) && (a = true);
    const o = e.length / 4;
    this.vertexCount = o, this.geometryBuffer = this.ensureBuffer(this.geometryBuffer, s, "GeometryBuffer"), !(r.length >= o * 2) && o > 0 && console.warn(`UV buffer mismatch: V=${o}, UV=${r.length / 2}. Filling 0.`);
    const m = e.length, C = i.length, f = r.length, g = m + C + f, v = new Float32Array(g);
    return v.set(e, 0), v.set(i, m), v.set(r, m + C), this.device.queue.writeBuffer(this.geometryBuffer, 0, v), a;
  }
  updateCombinedBVH(e, i) {
    const r = e.byteLength, s = i.byteLength, a = r + s;
    let o = false;
    return (!this.nodesBuffer || this.nodesBuffer.size < a) && (o = true), this.nodesBuffer = this.ensureBuffer(this.nodesBuffer, a, "NodesBuffer"), this.device.queue.writeBuffer(this.nodesBuffer, 0, e), this.device.queue.writeBuffer(this.nodesBuffer, r, i), this.blasOffset = e.length / 8, o;
  }
  updateSceneUniforms(e, i) {
    if (!this.sceneUniformBuffer) return;
    this.device.queue.writeBuffer(this.sceneUniformBuffer, 0, e);
    const r = new Uint32Array([i, this.blasOffset, this.vertexCount, 0]);
    this.device.queue.writeBuffer(this.sceneUniformBuffer, 96, r);
  }
  recreateBindGroup() {
    !this.renderTargetView || !this.accumulateBuffer || !this.geometryBuffer || !this.nodesBuffer || !this.sceneUniformBuffer || (this.bindGroup = this.device.createBindGroup({ layout: this.bindGroupLayout, entries: [{ binding: 0, resource: this.renderTargetView }, { binding: 1, resource: { buffer: this.accumulateBuffer } }, { binding: 2, resource: { buffer: this.sceneUniformBuffer } }, { binding: 3, resource: { buffer: this.geometryBuffer } }, { binding: 4, resource: { buffer: this.indexBuffer } }, { binding: 5, resource: { buffer: this.attrBuffer } }, { binding: 6, resource: { buffer: this.nodesBuffer } }, { binding: 7, resource: { buffer: this.instanceBuffer } }, { binding: 8, resource: this.texture.createView({ dimension: "2d-array" }) }, { binding: 9, resource: this.sampler }] }));
  }
  render(e) {
    if (!this.bindGroup) return;
    const i = new Uint32Array([e]);
    this.device.queue.writeBuffer(this.sceneUniformBuffer, 96, i);
    const r = Math.ceil(this.canvas.width / 8), s = Math.ceil(this.canvas.height / 8), a = this.device.createCommandEncoder(), o = a.beginComputePass();
    o.setPipeline(this.pipeline), o.setBindGroup(0, this.bindGroup), o.dispatchWorkgroups(r, s), o.end(), a.copyTextureToTexture({ texture: this.renderTarget }, { texture: this.context.getCurrentTexture() }, { width: this.canvas.width, height: this.canvas.height, depthOrArrayLayers: 1 }), this.device.queue.submit([a.finish()]);
  }
}
let p;
function Ti(t) {
  const e = p.__externref_table_alloc();
  return p.__wbindgen_externrefs.set(e, t), e;
}
function Ei(t, e) {
  return t = t >>> 0, se().subarray(t / 1, t / 1 + e);
}
let $ = null;
function Ot() {
  return ($ === null || $.buffer.detached === true || $.buffer.detached === void 0 && $.buffer !== p.memory.buffer) && ($ = new DataView(p.memory.buffer)), $;
}
function Ve(t, e) {
  return t = t >>> 0, Ii(t, e);
}
let be = null;
function se() {
  return (be === null || be.byteLength === 0) && (be = new Uint8Array(p.memory.buffer)), be;
}
function Ai(t, e) {
  try {
    return t.apply(this, e);
  } catch (i) {
    const r = Ti(i);
    p.__wbindgen_exn_store(r);
  }
}
function Gt(t) {
  return t == null;
}
function qt(t, e) {
  const i = e(t.length * 1, 1) >>> 0;
  return se().set(t, i / 1), z = t.length, i;
}
function ct(t, e, i) {
  if (i === void 0) {
    const l = Ce.encode(t), m = e(l.length, 1) >>> 0;
    return se().subarray(m, m + l.length).set(l), z = l.length, m;
  }
  let r = t.length, s = e(r, 1) >>> 0;
  const a = se();
  let o = 0;
  for (; o < r; o++) {
    const l = t.charCodeAt(o);
    if (l > 127) break;
    a[s + o] = l;
  }
  if (o !== r) {
    o !== 0 && (t = t.slice(o)), s = i(s, r, r = o + t.length * 3, 1) >>> 0;
    const l = se().subarray(s + o, s + r), m = Ce.encodeInto(t, l);
    o += m.written, s = i(s, r, o, 1) >>> 0;
  }
  return z = o, s;
}
let $e = new TextDecoder("utf-8", { ignoreBOM: true, fatal: true });
$e.decode();
const Mi = 2146435072;
let lt = 0;
function Ii(t, e) {
  return lt += e, lt >= Mi && ($e = new TextDecoder("utf-8", { ignoreBOM: true, fatal: true }), $e.decode(), lt = e), $e.decode(se().subarray(t, t + e));
}
const Ce = new TextEncoder();
"encodeInto" in Ce || (Ce.encodeInto = function(t, e) {
  const i = Ce.encode(t);
  return e.set(i), { read: t.length, written: i.length };
});
let z = 0;
typeof FinalizationRegistry > "u" || new FinalizationRegistry((t) => p.__wbg_renderbuffers_free(t >>> 0, 1));
const jt = typeof FinalizationRegistry > "u" ? { register: () => {
}, unregister: () => {
} } : new FinalizationRegistry((t) => p.__wbg_world_free(t >>> 0, 1));
class ht {
  __destroy_into_raw() {
    const e = this.__wbg_ptr;
    return this.__wbg_ptr = 0, jt.unregister(this), e;
  }
  free() {
    const e = this.__destroy_into_raw();
    p.__wbg_world_free(e, 0);
  }
  camera_ptr() {
    return p.world_camera_ptr(this.__wbg_ptr) >>> 0;
  }
  indices_len() {
    return p.world_indices_len(this.__wbg_ptr) >>> 0;
  }
  indices_ptr() {
    return p.world_indices_ptr(this.__wbg_ptr) >>> 0;
  }
  normals_len() {
    return p.world_normals_len(this.__wbg_ptr) >>> 0;
  }
  normals_ptr() {
    return p.world_normals_ptr(this.__wbg_ptr) >>> 0;
  }
  vertices_len() {
    return p.world_vertices_len(this.__wbg_ptr) >>> 0;
  }
  vertices_ptr() {
    return p.world_vertices_ptr(this.__wbg_ptr) >>> 0;
  }
  instances_len() {
    return p.world_instances_len(this.__wbg_ptr) >>> 0;
  }
  instances_ptr() {
    return p.world_instances_ptr(this.__wbg_ptr) >>> 0;
  }
  set_animation(e) {
    p.world_set_animation(this.__wbg_ptr, e);
  }
  update_camera(e, i) {
    p.world_update_camera(this.__wbg_ptr, e, i);
  }
  attributes_len() {
    return p.world_attributes_len(this.__wbg_ptr) >>> 0;
  }
  attributes_ptr() {
    return p.world_attributes_ptr(this.__wbg_ptr) >>> 0;
  }
  get_texture_ptr(e) {
    return p.world_get_texture_ptr(this.__wbg_ptr, e) >>> 0;
  }
  get_texture_size(e) {
    return p.world_get_texture_size(this.__wbg_ptr, e) >>> 0;
  }
  get_texture_count() {
    return p.world_get_texture_count(this.__wbg_ptr) >>> 0;
  }
  get_animation_name(e) {
    let i, r;
    try {
      const s = p.world_get_animation_name(this.__wbg_ptr, e);
      return i = s[0], r = s[1], Ve(s[0], s[1]);
    } finally {
      p.__wbindgen_free(i, r, 1);
    }
  }
  load_animation_glb(e) {
    const i = qt(e, p.__wbindgen_malloc), r = z;
    p.world_load_animation_glb(this.__wbg_ptr, i, r);
  }
  get_animation_count() {
    return p.world_get_animation_count(this.__wbg_ptr) >>> 0;
  }
  constructor(e, i, r) {
    const s = ct(e, p.__wbindgen_malloc, p.__wbindgen_realloc), a = z;
    var o = Gt(i) ? 0 : ct(i, p.__wbindgen_malloc, p.__wbindgen_realloc), l = z, m = Gt(r) ? 0 : qt(r, p.__wbindgen_malloc), C = z;
    const f = p.world_new(s, a, o, l, m, C);
    return this.__wbg_ptr = f >>> 0, jt.register(this, this.__wbg_ptr, this), this;
  }
  update(e) {
    p.world_update(this.__wbg_ptr, e);
  }
  uvs_len() {
    return p.world_uvs_len(this.__wbg_ptr) >>> 0;
  }
  uvs_ptr() {
    return p.world_uvs_ptr(this.__wbg_ptr) >>> 0;
  }
  blas_len() {
    return p.world_blas_len(this.__wbg_ptr) >>> 0;
  }
  blas_ptr() {
    return p.world_blas_ptr(this.__wbg_ptr) >>> 0;
  }
  tlas_len() {
    return p.world_tlas_len(this.__wbg_ptr) >>> 0;
  }
  tlas_ptr() {
    return p.world_tlas_ptr(this.__wbg_ptr) >>> 0;
  }
}
Symbol.dispose && (ht.prototype[Symbol.dispose] = ht.prototype.free);
const Ui = /* @__PURE__ */ new Set(["basic", "cors", "default"]);
async function Wi(t, e) {
  if (typeof Response == "function" && t instanceof Response) {
    if (typeof WebAssembly.instantiateStreaming == "function") try {
      return await WebAssembly.instantiateStreaming(t, e);
    } catch (r) {
      if (t.ok && Ui.has(t.type) && t.headers.get("Content-Type") !== "application/wasm") console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", r);
      else throw r;
    }
    const i = await t.arrayBuffer();
    return await WebAssembly.instantiate(i, e);
  } else {
    const i = await WebAssembly.instantiate(t, e);
    return i instanceof WebAssembly.Instance ? { instance: i, module: t } : i;
  }
}
function Li() {
  const t = {};
  return t.wbg = {}, t.wbg.__wbg___wbindgen_throw_dd24417ed36fc46e = function(e, i) {
    throw new Error(Ve(e, i));
  }, t.wbg.__wbg_error_7534b8e9a36f1ab4 = function(e, i) {
    let r, s;
    try {
      r = e, s = i, console.error(Ve(e, i));
    } finally {
      p.__wbindgen_free(r, s, 1);
    }
  }, t.wbg.__wbg_getRandomValues_1c61fac11405ffdc = function() {
    return Ai(function(e, i) {
      globalThis.crypto.getRandomValues(Ei(e, i));
    }, arguments);
  }, t.wbg.__wbg_log_1d990106d99dacb7 = function(e) {
    console.log(e);
  }, t.wbg.__wbg_new_8a6f238a6ece86ea = function() {
    return new Error();
  }, t.wbg.__wbg_stack_0ed75d68575b0f3c = function(e, i) {
    const r = i.stack, s = ct(r, p.__wbindgen_malloc, p.__wbindgen_realloc), a = z;
    Ot().setInt32(e + 4, a, true), Ot().setInt32(e + 0, s, true);
  }, t.wbg.__wbindgen_cast_2241b6af4c4b2941 = function(e, i) {
    return Ve(e, i);
  }, t.wbg.__wbindgen_init_externref_table = function() {
    const e = p.__wbindgen_externrefs, i = e.grow(4);
    e.set(0, void 0), e.set(i + 0, void 0), e.set(i + 1, null), e.set(i + 2, true), e.set(i + 3, false);
  }, t;
}
function Di(t, e) {
  return p = t.exports, Jt.__wbindgen_wasm_module = e, $ = null, be = null, p.__wbindgen_start(), p;
}
async function Jt(t) {
  if (p !== void 0) return p;
  typeof t < "u" && (Object.getPrototypeOf(t) === Object.prototype ? { module_or_path: t } = t : console.warn("using deprecated parameters for the initialization function; pass a single object instead")), typeof t > "u" && (t = new URL("/webgpu-raytracer/assets/rust_shader_tools_bg-CC5HVMsp.wasm", import.meta.url));
  const e = Li();
  (typeof t == "string" || typeof Request == "function" && t instanceof Request || typeof URL == "function" && t instanceof URL) && (t = fetch(t));
  const { instance: i, module: r } = await Wi(await t, e);
  return Di(i, r);
}
class Pi {
  constructor() {
    __publicField(this, "world", null);
    __publicField(this, "wasmMemory", null);
  }
  async initWasm() {
    const e = await Jt();
    this.wasmMemory = e.memory, console.log("Wasm initialized");
  }
  loadScene(e, i, r) {
    this.world && this.world.free(), this.world = new ht(e, i, r);
  }
  update(e) {
    var _a;
    (_a = this.world) == null ? void 0 : _a.update(e);
  }
  updateCamera(e, i) {
    var _a;
    (_a = this.world) == null ? void 0 : _a.update_camera(e, i);
  }
  loadAnimation(e) {
    var _a;
    (_a = this.world) == null ? void 0 : _a.load_animation_glb(e);
  }
  getAnimationList() {
    if (!this.world) return [];
    const e = this.world.get_animation_count(), i = [];
    for (let r = 0; r < e; r++) i.push(this.world.get_animation_name(r));
    return i;
  }
  setAnimation(e) {
    var _a;
    (_a = this.world) == null ? void 0 : _a.set_animation(e);
  }
  getF32(e, i) {
    return new Float32Array(this.wasmMemory.buffer, e, i);
  }
  getU32(e, i) {
    return new Uint32Array(this.wasmMemory.buffer, e, i);
  }
  get vertices() {
    return this.getF32(this.world.vertices_ptr(), this.world.vertices_len());
  }
  get normals() {
    return this.getF32(this.world.normals_ptr(), this.world.normals_len());
  }
  get uvs() {
    return this.getF32(this.world.uvs_ptr(), this.world.uvs_len());
  }
  get indices() {
    return this.getU32(this.world.indices_ptr(), this.world.indices_len());
  }
  get attributes() {
    return this.getF32(this.world.attributes_ptr(), this.world.attributes_len());
  }
  get tlas() {
    return this.getF32(this.world.tlas_ptr(), this.world.tlas_len());
  }
  get blas() {
    return this.getF32(this.world.blas_ptr(), this.world.blas_len());
  }
  get instances() {
    return this.getF32(this.world.instances_ptr(), this.world.instances_len());
  }
  get cameraData() {
    return this.getF32(this.world.camera_ptr(), 24);
  }
  get textureCount() {
    var _a;
    return ((_a = this.world) == null ? void 0 : _a.get_texture_count()) || 0;
  }
  getTexture(e) {
    if (!this.world) return null;
    const i = this.world.get_texture_ptr(e), r = this.world.get_texture_size(e);
    return !i || r === 0 ? null : new Uint8Array(this.wasmMemory.buffer, i, r).slice();
  }
  get hasWorld() {
    return !!this.world;
  }
  printStats() {
    this.world && console.log(`Scene Stats: V=${this.vertices.length / 4}, Tri=${this.indices.length / 3}, BLAS=${this.blas.length / 8}, TLAS=${this.tlas.length / 8}`);
  }
}
const S = { defaultWidth: 720, defaultHeight: 480, defaultDepth: 10, defaultSPP: 1, signalingServerUrl: "ws://localhost:8080", ids: { canvas: "gpu-canvas", renderBtn: "render-btn", sceneSelect: "scene-select", resWidth: "res-width", resHeight: "res-height", objFile: "obj-file", maxDepth: "max-depth", sppFrame: "spp-frame", recompileBtn: "recompile-btn", updateInterval: "update-interval", animSelect: "anim-select", recordBtn: "record-btn", recFps: "rec-fps", recDuration: "rec-duration", recSpp: "rec-spp", recBatch: "rec-batch", btnHost: "btn-host", btnWorker: "btn-worker", btnSendScene: "btn-send-scene", statusDiv: "status" } };
class zi {
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
    __publicField(this, "btnHost");
    __publicField(this, "btnWorker");
    __publicField(this, "btnSendScene");
    __publicField(this, "statusDiv");
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
    __publicField(this, "onSendScene", null);
    this.canvas = this.el(S.ids.canvas), this.btnRender = this.el(S.ids.renderBtn), this.sceneSelect = this.el(S.ids.sceneSelect), this.inputWidth = this.el(S.ids.resWidth), this.inputHeight = this.el(S.ids.resHeight), this.inputFile = this.setupFileInput(), this.inputDepth = this.el(S.ids.maxDepth), this.inputSPP = this.el(S.ids.sppFrame), this.btnRecompile = this.el(S.ids.recompileBtn), this.inputUpdateInterval = this.el(S.ids.updateInterval), this.animSelect = this.el(S.ids.animSelect), this.btnRecord = this.el(S.ids.recordBtn), this.inputRecFps = this.el(S.ids.recFps), this.inputRecDur = this.el(S.ids.recDuration), this.inputRecSpp = this.el(S.ids.recSpp), this.inputRecBatch = this.el(S.ids.recBatch), this.btnHost = this.el(S.ids.btnHost), this.btnWorker = this.el(S.ids.btnWorker), this.btnSendScene = this.el(S.ids.btnSendScene), this.statusDiv = this.el(S.ids.statusDiv), this.statsDiv = this.createStatsDiv(), this.bindEvents();
  }
  el(e) {
    const i = document.getElementById(e);
    if (!i) throw new Error(`Element not found: ${e}`);
    return i;
  }
  setupFileInput() {
    const e = this.el(S.ids.objFile);
    return e && (e.accept = ".obj,.glb,.vrm"), e;
  }
  createStatsDiv() {
    const e = document.createElement("div");
    return Object.assign(e.style, { position: "fixed", bottom: "10px", left: "10px", color: "#0f0", background: "rgba(0,0,0,0.7)", padding: "8px", fontFamily: "monospace", fontSize: "14px", pointerEvents: "none", zIndex: "9999", borderRadius: "4px" }), document.body.appendChild(e), e;
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
      return (_a = this.onResolutionChange) == null ? void 0 : _a.call(this, parseInt(this.inputWidth.value) || S.defaultWidth, parseInt(this.inputHeight.value) || S.defaultHeight);
    };
    this.inputWidth.addEventListener("change", e), this.inputHeight.addEventListener("change", e), this.btnRecompile.addEventListener("click", () => {
      var _a;
      return (_a = this.onRecompile) == null ? void 0 : _a.call(this, parseInt(this.inputDepth.value) || 10, parseInt(this.inputSPP.value) || 1);
    }), this.inputFile.addEventListener("change", (i) => {
      var _a, _b;
      const r = (_a = i.target.files) == null ? void 0 : _a[0];
      r && ((_b = this.onFileSelect) == null ? void 0 : _b.call(this, r));
    }), this.animSelect.addEventListener("change", () => {
      var _a;
      const i = parseInt(this.animSelect.value, 10);
      (_a = this.onAnimSelect) == null ? void 0 : _a.call(this, i);
    }), this.btnRecord.addEventListener("click", () => {
      var _a;
      return (_a = this.onRecordStart) == null ? void 0 : _a.call(this);
    }), this.btnHost.addEventListener("click", () => {
      var _a;
      return (_a = this.onConnectHost) == null ? void 0 : _a.call(this);
    }), this.btnWorker.addEventListener("click", () => {
      var _a;
      return (_a = this.onConnectWorker) == null ? void 0 : _a.call(this);
    }), this.btnSendScene.addEventListener("click", () => {
      var _a;
      return (_a = this.onSendScene) == null ? void 0 : _a.call(this);
    });
  }
  updateRenderButton(e) {
    this.btnRender.textContent = e ? "Stop Rendering" : "Resume Rendering";
  }
  updateStats(e, i, r) {
    this.statsDiv.textContent = `FPS: ${e} | ${i.toFixed(2)}ms | Frame: ${r}`;
  }
  setStatus(e) {
    this.statusDiv.textContent = e;
  }
  setConnectionState(e) {
    e === "host" ? (this.btnHost.textContent = "Disconnect", this.btnHost.disabled = false, this.btnWorker.textContent = "Worker", this.btnWorker.disabled = true, this.btnSendScene.style.display = "inline-block", this.btnSendScene.disabled = true) : e === "worker" ? (this.btnHost.textContent = "Host", this.btnHost.disabled = true, this.btnWorker.textContent = "Disconnect", this.btnWorker.disabled = false, this.btnSendScene.style.display = "none") : (this.btnHost.textContent = "Host", this.btnHost.disabled = false, this.btnWorker.textContent = "Worker", this.btnWorker.disabled = false, this.btnSendScene.style.display = "none", this.statusDiv.textContent = "Offline");
  }
  setSendSceneEnabled(e) {
    this.btnSendScene.disabled = !e;
  }
  setSendSceneText(e) {
    this.btnSendScene.textContent = e;
  }
  setRecordingState(e, i) {
    e ? (this.btnRecord.disabled = true, this.btnRecord.textContent = i || "Recording...", this.btnRender.textContent = "Resume Rendering") : (this.btnRecord.disabled = false, this.btnRecord.textContent = "\u25CF Rec");
  }
  updateAnimList(e) {
    if (this.animSelect.innerHTML = "", e.length === 0) {
      const i = document.createElement("option");
      i.text = "No Anim", this.animSelect.add(i), this.animSelect.disabled = true;
      return;
    }
    this.animSelect.disabled = false, e.forEach((i, r) => {
      const s = document.createElement("option");
      s.text = `[${r}] ${i}`, s.value = r.toString(), this.animSelect.add(s);
    }), this.animSelect.value = "0";
  }
  getRenderConfig() {
    return { width: parseInt(this.inputWidth.value, 10) || S.defaultWidth, height: parseInt(this.inputHeight.value, 10) || S.defaultHeight, fps: parseInt(this.inputRecFps.value, 10) || 30, duration: parseFloat(this.inputRecDur.value) || 3, spp: parseInt(this.inputRecSpp.value, 10) || 64, batch: parseInt(this.inputRecBatch.value, 10) || 4, anim: parseInt(this.animSelect.value, 10) || 0 };
  }
  setRenderConfig(e) {
    this.inputWidth.value = e.width.toString(), this.inputHeight.value = e.height.toString(), this.inputRecFps.value = e.fps.toString(), this.inputRecDur.value = e.duration.toString(), this.inputRecSpp.value = e.spp.toString(), this.inputRecBatch.value = e.batch.toString();
  }
}
var Ct = (t, e, i) => {
  if (!e.has(t)) throw TypeError("Cannot " + i);
}, n = (t, e, i) => (Ct(t, e, "read from private field"), i ? i.call(t) : e.get(t)), d = (t, e, i) => {
  if (e.has(t)) throw TypeError("Cannot add the same private member more than once");
  e instanceof WeakSet ? e.add(t) : e.set(t, i);
}, _ = (t, e, i, r) => (Ct(t, e, "write to private field"), e.set(t, i), i), c = (t, e, i) => (Ct(t, e, "access private method"), i), Zt = class {
  constructor(t) {
    this.value = t;
  }
}, xt = class {
  constructor(t) {
    this.value = t;
  }
}, Qt = (t) => t < 256 ? 1 : t < 65536 ? 2 : t < 1 << 24 ? 3 : t < 2 ** 32 ? 4 : t < 2 ** 40 ? 5 : 6, Fi = (t) => {
  if (t < 127) return 1;
  if (t < 16383) return 2;
  if (t < (1 << 21) - 1) return 3;
  if (t < (1 << 28) - 1) return 4;
  if (t < 2 ** 35 - 1) return 5;
  if (t < 2 ** 42 - 1) return 6;
  throw new Error("EBML VINT size not supported " + t);
}, ee = (t, e, i) => {
  let r = 0;
  for (let s = e; s < i; s++) {
    let a = Math.floor(s / 8), o = t[a], l = 7 - (s & 7), m = (o & 1 << l) >> l;
    r <<= 1, r |= m;
  }
  return r;
}, Hi = (t, e, i, r) => {
  for (let s = e; s < i; s++) {
    let a = Math.floor(s / 8), o = t[a], l = 7 - (s & 7);
    o &= ~(1 << l), o |= (r & 1 << i - s - 1) >> i - s - 1 << l, t[a] = o;
  }
}, st = class {
}, Rt = class extends st {
  constructor() {
    super(...arguments), this.buffer = null;
  }
}, Bt = class extends st {
  constructor(t) {
    if (super(), this.options = t, typeof t != "object") throw new TypeError("StreamTarget requires an options object to be passed to its constructor.");
    if (t.onData) {
      if (typeof t.onData != "function") throw new TypeError("options.onData, when provided, must be a function.");
      if (t.onData.length < 2) throw new TypeError("options.onData, when provided, must be a function that takes in at least two arguments (data and position). Ignoring the position argument, which specifies the byte offset at which the data is to be written, can lead to broken outputs.");
    }
    if (t.onHeader && typeof t.onHeader != "function") throw new TypeError("options.onHeader, when provided, must be a function.");
    if (t.onCluster && typeof t.onCluster != "function") throw new TypeError("options.onCluster, when provided, must be a function.");
    if (t.chunked !== void 0 && typeof t.chunked != "boolean") throw new TypeError("options.chunked, when provided, must be a boolean.");
    if (t.chunkSize !== void 0 && (!Number.isInteger(t.chunkSize) || t.chunkSize < 1024)) throw new TypeError("options.chunkSize, when provided, must be an integer and not smaller than 1024.");
  }
}, ei = class extends st {
  constructor(t, e) {
    if (super(), this.stream = t, this.options = e, !(t instanceof FileSystemWritableFileStream)) throw new TypeError("FileSystemWritableFileStreamTarget requires a FileSystemWritableFileStream instance.");
    if (e !== void 0 && typeof e != "object") throw new TypeError("FileSystemWritableFileStreamTarget's options, when provided, must be an object.");
    if (e && e.chunkSize !== void 0 && (!Number.isInteger(e.chunkSize) || e.chunkSize <= 0)) throw new TypeError("options.chunkSize, when provided, must be a positive integer");
  }
}, F, b, ut, ti, ft, ii, pt, ni, Oe, wt, gt, ri, si = class {
  constructor() {
    d(this, ut), d(this, ft), d(this, pt), d(this, Oe), d(this, gt), this.pos = 0, d(this, F, new Uint8Array(8)), d(this, b, new DataView(n(this, F).buffer)), this.offsets = /* @__PURE__ */ new WeakMap(), this.dataOffsets = /* @__PURE__ */ new WeakMap();
  }
  seek(t) {
    this.pos = t;
  }
  writeEBMLVarInt(t, e = Fi(t)) {
    let i = 0;
    switch (e) {
      case 1:
        n(this, b).setUint8(i++, 128 | t);
        break;
      case 2:
        n(this, b).setUint8(i++, 64 | t >> 8), n(this, b).setUint8(i++, t);
        break;
      case 3:
        n(this, b).setUint8(i++, 32 | t >> 16), n(this, b).setUint8(i++, t >> 8), n(this, b).setUint8(i++, t);
        break;
      case 4:
        n(this, b).setUint8(i++, 16 | t >> 24), n(this, b).setUint8(i++, t >> 16), n(this, b).setUint8(i++, t >> 8), n(this, b).setUint8(i++, t);
        break;
      case 5:
        n(this, b).setUint8(i++, 8 | t / 2 ** 32 & 7), n(this, b).setUint8(i++, t >> 24), n(this, b).setUint8(i++, t >> 16), n(this, b).setUint8(i++, t >> 8), n(this, b).setUint8(i++, t);
        break;
      case 6:
        n(this, b).setUint8(i++, 4 | t / 2 ** 40 & 3), n(this, b).setUint8(i++, t / 2 ** 32 | 0), n(this, b).setUint8(i++, t >> 24), n(this, b).setUint8(i++, t >> 16), n(this, b).setUint8(i++, t >> 8), n(this, b).setUint8(i++, t);
        break;
      default:
        throw new Error("Bad EBML VINT size " + e);
    }
    this.write(n(this, F).subarray(0, i));
  }
  writeEBML(t) {
    if (t !== null) if (t instanceof Uint8Array) this.write(t);
    else if (Array.isArray(t)) for (let e of t) this.writeEBML(e);
    else if (this.offsets.set(t, this.pos), c(this, Oe, wt).call(this, t.id), Array.isArray(t.data)) {
      let e = this.pos, i = t.size === -1 ? 1 : t.size ?? 4;
      t.size === -1 ? c(this, ut, ti).call(this, 255) : this.seek(this.pos + i);
      let r = this.pos;
      if (this.dataOffsets.set(t, r), this.writeEBML(t.data), t.size !== -1) {
        let s = this.pos - r, a = this.pos;
        this.seek(e), this.writeEBMLVarInt(s, i), this.seek(a);
      }
    } else if (typeof t.data == "number") {
      let e = t.size ?? Qt(t.data);
      this.writeEBMLVarInt(e), c(this, Oe, wt).call(this, t.data, e);
    } else typeof t.data == "string" ? (this.writeEBMLVarInt(t.data.length), c(this, gt, ri).call(this, t.data)) : t.data instanceof Uint8Array ? (this.writeEBMLVarInt(t.data.byteLength, t.size), this.write(t.data)) : t.data instanceof Zt ? (this.writeEBMLVarInt(4), c(this, ft, ii).call(this, t.data.value)) : t.data instanceof xt && (this.writeEBMLVarInt(8), c(this, pt, ni).call(this, t.data.value));
  }
};
F = /* @__PURE__ */ new WeakMap();
b = /* @__PURE__ */ new WeakMap();
ut = /* @__PURE__ */ new WeakSet();
ti = function(t) {
  n(this, b).setUint8(0, t), this.write(n(this, F).subarray(0, 1));
};
ft = /* @__PURE__ */ new WeakSet();
ii = function(t) {
  n(this, b).setFloat32(0, t, false), this.write(n(this, F).subarray(0, 4));
};
pt = /* @__PURE__ */ new WeakSet();
ni = function(t) {
  n(this, b).setFloat64(0, t, false), this.write(n(this, F));
};
Oe = /* @__PURE__ */ new WeakSet();
wt = function(t, e = Qt(t)) {
  let i = 0;
  switch (e) {
    case 6:
      n(this, b).setUint8(i++, t / 2 ** 40 | 0);
    case 5:
      n(this, b).setUint8(i++, t / 2 ** 32 | 0);
    case 4:
      n(this, b).setUint8(i++, t >> 24);
    case 3:
      n(this, b).setUint8(i++, t >> 16);
    case 2:
      n(this, b).setUint8(i++, t >> 8);
    case 1:
      n(this, b).setUint8(i++, t);
      break;
    default:
      throw new Error("Bad UINT size " + e);
  }
  this.write(n(this, F).subarray(0, i));
};
gt = /* @__PURE__ */ new WeakSet();
ri = function(t) {
  this.write(new Uint8Array(t.split("").map((e) => e.charCodeAt(0))));
};
var Ge, J, Ie, qe, _t, Ni = class extends si {
  constructor(t) {
    super(), d(this, qe), d(this, Ge, void 0), d(this, J, new ArrayBuffer(2 ** 16)), d(this, Ie, new Uint8Array(n(this, J))), _(this, Ge, t);
  }
  write(t) {
    c(this, qe, _t).call(this, this.pos + t.byteLength), n(this, Ie).set(t, this.pos), this.pos += t.byteLength;
  }
  finalize() {
    c(this, qe, _t).call(this, this.pos), n(this, Ge).buffer = n(this, J).slice(0, this.pos);
  }
};
Ge = /* @__PURE__ */ new WeakMap();
J = /* @__PURE__ */ new WeakMap();
Ie = /* @__PURE__ */ new WeakMap();
qe = /* @__PURE__ */ new WeakSet();
_t = function(t) {
  let e = n(this, J).byteLength;
  for (; e < t; ) e *= 2;
  if (e === n(this, J).byteLength) return;
  let i = new ArrayBuffer(e), r = new Uint8Array(i);
  r.set(n(this, Ie), 0), _(this, J, i), _(this, Ie, r);
};
var te, T, E, O, ze = class extends si {
  constructor(t) {
    super(), this.target = t, d(this, te, false), d(this, T, void 0), d(this, E, void 0), d(this, O, void 0);
  }
  write(t) {
    if (!n(this, te)) return;
    let e = this.pos;
    if (e < n(this, E)) {
      if (e + t.byteLength <= n(this, E)) return;
      t = t.subarray(n(this, E) - e), e = 0;
    }
    let i = e + t.byteLength - n(this, E), r = n(this, T).byteLength;
    for (; r < i; ) r *= 2;
    if (r !== n(this, T).byteLength) {
      let s = new Uint8Array(r);
      s.set(n(this, T), 0), _(this, T, s);
    }
    n(this, T).set(t, e - n(this, E)), _(this, O, Math.max(n(this, O), e + t.byteLength));
  }
  startTrackingWrites() {
    _(this, te, true), _(this, T, new Uint8Array(2 ** 10)), _(this, E, this.pos), _(this, O, this.pos);
  }
  getTrackedWrites() {
    if (!n(this, te)) throw new Error("Can't get tracked writes since nothing was tracked.");
    let e = { data: n(this, T).subarray(0, n(this, O) - n(this, E)), start: n(this, E), end: n(this, O) };
    return _(this, T, void 0), _(this, te, false), e;
  }
};
te = /* @__PURE__ */ new WeakMap();
T = /* @__PURE__ */ new WeakMap();
E = /* @__PURE__ */ new WeakMap();
O = /* @__PURE__ */ new WeakMap();
var Vi = 2 ** 24, $i = 2, G, ae, xe, ye, L, R, Ze, mt, Tt, ai, Et, oi, Re, Qe, At = class extends ze {
  constructor(t, e) {
    var _a, _b;
    super(t), d(this, Ze), d(this, Tt), d(this, Et), d(this, Re), d(this, G, []), d(this, ae, 0), d(this, xe, void 0), d(this, ye, void 0), d(this, L, void 0), d(this, R, []), _(this, xe, e), _(this, ye, ((_a = t.options) == null ? void 0 : _a.chunked) ?? false), _(this, L, ((_b = t.options) == null ? void 0 : _b.chunkSize) ?? Vi);
  }
  write(t) {
    super.write(t), n(this, G).push({ data: t.slice(), start: this.pos }), this.pos += t.byteLength;
  }
  flush() {
    var _a, _b;
    if (n(this, G).length === 0) return;
    let t = [], e = [...n(this, G)].sort((i, r) => i.start - r.start);
    t.push({ start: e[0].start, size: e[0].data.byteLength });
    for (let i = 1; i < e.length; i++) {
      let r = t[t.length - 1], s = e[i];
      s.start <= r.start + r.size ? r.size = Math.max(r.size, s.start + s.data.byteLength - r.start) : t.push({ start: s.start, size: s.data.byteLength });
    }
    for (let i of t) {
      i.data = new Uint8Array(i.size);
      for (let r of n(this, G)) i.start <= r.start && r.start < i.start + i.size && i.data.set(r.data, r.start - i.start);
      if (n(this, ye)) c(this, Ze, mt).call(this, i.data, i.start), c(this, Re, Qe).call(this);
      else {
        if (n(this, xe) && i.start < n(this, ae)) throw new Error("Internal error: Monotonicity violation.");
        (_b = (_a = this.target.options).onData) == null ? void 0 : _b.call(_a, i.data, i.start), _(this, ae, i.start + i.data.byteLength);
      }
    }
    n(this, G).length = 0;
  }
  finalize() {
    n(this, ye) && c(this, Re, Qe).call(this, true);
  }
};
G = /* @__PURE__ */ new WeakMap();
ae = /* @__PURE__ */ new WeakMap();
xe = /* @__PURE__ */ new WeakMap();
ye = /* @__PURE__ */ new WeakMap();
L = /* @__PURE__ */ new WeakMap();
R = /* @__PURE__ */ new WeakMap();
Ze = /* @__PURE__ */ new WeakSet();
mt = function(t, e) {
  let i = n(this, R).findIndex((l) => l.start <= e && e < l.start + n(this, L));
  i === -1 && (i = c(this, Et, oi).call(this, e));
  let r = n(this, R)[i], s = e - r.start, a = t.subarray(0, Math.min(n(this, L) - s, t.byteLength));
  r.data.set(a, s);
  let o = { start: s, end: s + a.byteLength };
  if (c(this, Tt, ai).call(this, r, o), r.written[0].start === 0 && r.written[0].end === n(this, L) && (r.shouldFlush = true), n(this, R).length > $i) {
    for (let l = 0; l < n(this, R).length - 1; l++) n(this, R)[l].shouldFlush = true;
    c(this, Re, Qe).call(this);
  }
  a.byteLength < t.byteLength && c(this, Ze, mt).call(this, t.subarray(a.byteLength), e + a.byteLength);
};
Tt = /* @__PURE__ */ new WeakSet();
ai = function(t, e) {
  let i = 0, r = t.written.length - 1, s = -1;
  for (; i <= r; ) {
    let a = Math.floor(i + (r - i + 1) / 2);
    t.written[a].start <= e.start ? (i = a + 1, s = a) : r = a - 1;
  }
  for (t.written.splice(s + 1, 0, e), (s === -1 || t.written[s].end < e.start) && s++; s < t.written.length - 1 && t.written[s].end >= t.written[s + 1].start; ) t.written[s].end = Math.max(t.written[s].end, t.written[s + 1].end), t.written.splice(s + 1, 1);
};
Et = /* @__PURE__ */ new WeakSet();
oi = function(t) {
  let i = { start: Math.floor(t / n(this, L)) * n(this, L), data: new Uint8Array(n(this, L)), written: [], shouldFlush: false };
  return n(this, R).push(i), n(this, R).sort((r, s) => r.start - s.start), n(this, R).indexOf(i);
};
Re = /* @__PURE__ */ new WeakSet();
Qe = function(t = false) {
  var _a, _b;
  for (let e = 0; e < n(this, R).length; e++) {
    let i = n(this, R)[e];
    if (!(!i.shouldFlush && !t)) {
      for (let r of i.written) {
        if (n(this, xe) && i.start + r.start < n(this, ae)) throw new Error("Internal error: Monotonicity violation.");
        (_b = (_a = this.target.options).onData) == null ? void 0 : _b.call(_a, i.data.subarray(r.start, r.end), i.start + r.start), _(this, ae, i.start + r.end);
      }
      n(this, R).splice(e--, 1);
    }
  }
};
var Oi = class extends At {
  constructor(t, e) {
    var _a;
    super(new Bt({ onData: (i, r) => t.stream.write({ type: "write", data: i, position: r }), chunked: true, chunkSize: (_a = t.options) == null ? void 0 : _a.chunkSize }), e);
  }
}, fe = 1, Ue = 2, et = 3, Gi = 1, qi = 2, ji = 17, Ki = 2 ** 15, Be = 2 ** 13, Kt = "https://github.com/Vanilagy/webm-muxer", di = 6, li = 5, Yi = ["strict", "offset", "permissive"], w, h, We, Le, U, pe, ie, Z, we, H, oe, de, M, Fe, le, D, P, q, Te, Ee, ce, he, tt, De, Ae, bt, ci, yt, hi, Mt, ui, It, fi, Ut, pi, Wt, wi, Lt, gi, at, Dt, ot, Pt, zt, _i, j, ne, K, re, vt, mi, St, bi, ve, je, Se, Ke, Ft, yi, A, I, ue, Pe, Me, it, Ht, vi, nt, Nt, ke, Ye, Si = class {
  constructor(t) {
    d(this, bt), d(this, yt), d(this, Mt), d(this, It), d(this, Ut), d(this, Wt), d(this, Lt), d(this, at), d(this, ot), d(this, zt), d(this, j), d(this, K), d(this, vt), d(this, St), d(this, ve), d(this, Se), d(this, Ft), d(this, A), d(this, ue), d(this, Me), d(this, Ht), d(this, nt), d(this, ke), d(this, w, void 0), d(this, h, void 0), d(this, We, void 0), d(this, Le, void 0), d(this, U, void 0), d(this, pe, void 0), d(this, ie, void 0), d(this, Z, void 0), d(this, we, void 0), d(this, H, void 0), d(this, oe, void 0), d(this, de, void 0), d(this, M, void 0), d(this, Fe, void 0), d(this, le, 0), d(this, D, []), d(this, P, []), d(this, q, []), d(this, Te, void 0), d(this, Ee, void 0), d(this, ce, -1), d(this, he, -1), d(this, tt, -1), d(this, De, void 0), d(this, Ae, false), c(this, bt, ci).call(this, t), _(this, w, { type: "webm", firstTimestampBehavior: "strict", ...t }), this.target = t.target;
    let e = !!n(this, w).streaming;
    if (t.target instanceof Rt) _(this, h, new Ni(t.target));
    else if (t.target instanceof Bt) _(this, h, new At(t.target, e));
    else if (t.target instanceof ei) _(this, h, new Oi(t.target, e));
    else throw new Error(`Invalid target: ${t.target}`);
    c(this, yt, hi).call(this);
  }
  addVideoChunk(t, e, i) {
    if (!(t instanceof EncodedVideoChunk)) throw new TypeError("addVideoChunk's first argument (chunk) must be of type EncodedVideoChunk.");
    if (e && typeof e != "object") throw new TypeError("addVideoChunk's second argument (meta), when provided, must be an object.");
    if (i !== void 0 && (!Number.isFinite(i) || i < 0)) throw new TypeError("addVideoChunk's third argument (timestamp), when provided, must be a non-negative real number.");
    let r = new Uint8Array(t.byteLength);
    t.copyTo(r), this.addVideoChunkRaw(r, t.type, i ?? t.timestamp, e);
  }
  addVideoChunkRaw(t, e, i, r) {
    if (!(t instanceof Uint8Array)) throw new TypeError("addVideoChunkRaw's first argument (data) must be an instance of Uint8Array.");
    if (e !== "key" && e !== "delta") throw new TypeError("addVideoChunkRaw's second argument (type) must be either 'key' or 'delta'.");
    if (!Number.isFinite(i) || i < 0) throw new TypeError("addVideoChunkRaw's third argument (timestamp) must be a non-negative real number.");
    if (r && typeof r != "object") throw new TypeError("addVideoChunkRaw's fourth argument (meta), when provided, must be an object.");
    if (c(this, ke, Ye).call(this), !n(this, w).video) throw new Error("No video track declared.");
    n(this, Te) === void 0 && _(this, Te, i), r && c(this, vt, mi).call(this, r);
    let s = c(this, Se, Ke).call(this, t, e, i, fe);
    for (n(this, w).video.codec === "V_VP9" && c(this, St, bi).call(this, s), _(this, ce, s.timestamp); n(this, P).length > 0 && n(this, P)[0].timestamp <= s.timestamp; ) {
      let a = n(this, P).shift();
      c(this, A, I).call(this, a, false);
    }
    !n(this, w).audio || s.timestamp <= n(this, he) ? c(this, A, I).call(this, s, true) : n(this, D).push(s), c(this, ve, je).call(this), c(this, j, ne).call(this);
  }
  addAudioChunk(t, e, i) {
    if (!(t instanceof EncodedAudioChunk)) throw new TypeError("addAudioChunk's first argument (chunk) must be of type EncodedAudioChunk.");
    if (e && typeof e != "object") throw new TypeError("addAudioChunk's second argument (meta), when provided, must be an object.");
    if (i !== void 0 && (!Number.isFinite(i) || i < 0)) throw new TypeError("addAudioChunk's third argument (timestamp), when provided, must be a non-negative real number.");
    let r = new Uint8Array(t.byteLength);
    t.copyTo(r), this.addAudioChunkRaw(r, t.type, i ?? t.timestamp, e);
  }
  addAudioChunkRaw(t, e, i, r) {
    if (!(t instanceof Uint8Array)) throw new TypeError("addAudioChunkRaw's first argument (data) must be an instance of Uint8Array.");
    if (e !== "key" && e !== "delta") throw new TypeError("addAudioChunkRaw's second argument (type) must be either 'key' or 'delta'.");
    if (!Number.isFinite(i) || i < 0) throw new TypeError("addAudioChunkRaw's third argument (timestamp) must be a non-negative real number.");
    if (r && typeof r != "object") throw new TypeError("addAudioChunkRaw's fourth argument (meta), when provided, must be an object.");
    if (c(this, ke, Ye).call(this), !n(this, w).audio) throw new Error("No audio track declared.");
    n(this, Ee) === void 0 && _(this, Ee, i), (r == null ? void 0 : r.decoderConfig) && (n(this, w).streaming ? _(this, H, c(this, ue, Pe).call(this, r.decoderConfig.description)) : c(this, Me, it).call(this, n(this, H), r.decoderConfig.description));
    let s = c(this, Se, Ke).call(this, t, e, i, Ue);
    for (_(this, he, s.timestamp); n(this, D).length > 0 && n(this, D)[0].timestamp <= s.timestamp; ) {
      let a = n(this, D).shift();
      c(this, A, I).call(this, a, true);
    }
    !n(this, w).video || s.timestamp <= n(this, ce) ? c(this, A, I).call(this, s, !n(this, w).video) : n(this, P).push(s), c(this, ve, je).call(this), c(this, j, ne).call(this);
  }
  addSubtitleChunk(t, e, i) {
    if (typeof t != "object" || !t) throw new TypeError("addSubtitleChunk's first argument (chunk) must be an object.");
    if (!(t.body instanceof Uint8Array)) throw new TypeError("body must be an instance of Uint8Array.");
    if (!Number.isFinite(t.timestamp) || t.timestamp < 0) throw new TypeError("timestamp must be a non-negative real number.");
    if (!Number.isFinite(t.duration) || t.duration < 0) throw new TypeError("duration must be a non-negative real number.");
    if (t.additions && !(t.additions instanceof Uint8Array)) throw new TypeError("additions, when present, must be an instance of Uint8Array.");
    if (typeof e != "object") throw new TypeError("addSubtitleChunk's second argument (meta) must be an object.");
    if (c(this, ke, Ye).call(this), !n(this, w).subtitles) throw new Error("No subtitle track declared.");
    (e == null ? void 0 : e.decoderConfig) && (n(this, w).streaming ? _(this, oe, c(this, ue, Pe).call(this, e.decoderConfig.description)) : c(this, Me, it).call(this, n(this, oe), e.decoderConfig.description));
    let r = c(this, Se, Ke).call(this, t.body, "key", i ?? t.timestamp, et, t.duration, t.additions);
    _(this, tt, r.timestamp), n(this, q).push(r), c(this, ve, je).call(this), c(this, j, ne).call(this);
  }
  finalize() {
    if (n(this, Ae)) throw new Error("Cannot finalize a muxer more than once.");
    for (; n(this, D).length > 0; ) c(this, A, I).call(this, n(this, D).shift(), true);
    for (; n(this, P).length > 0; ) c(this, A, I).call(this, n(this, P).shift(), true);
    for (; n(this, q).length > 0 && n(this, q)[0].timestamp <= n(this, le); ) c(this, A, I).call(this, n(this, q).shift(), false);
    if (n(this, M) && c(this, nt, Nt).call(this), n(this, h).writeEBML(n(this, de)), !n(this, w).streaming) {
      let t = n(this, h).pos, e = n(this, h).pos - n(this, K, re);
      n(this, h).seek(n(this, h).offsets.get(n(this, We)) + 4), n(this, h).writeEBMLVarInt(e, di), n(this, ie).data = new xt(n(this, le)), n(this, h).seek(n(this, h).offsets.get(n(this, ie))), n(this, h).writeEBML(n(this, ie)), n(this, U).data[0].data[1].data = n(this, h).offsets.get(n(this, de)) - n(this, K, re), n(this, U).data[1].data[1].data = n(this, h).offsets.get(n(this, Le)) - n(this, K, re), n(this, U).data[2].data[1].data = n(this, h).offsets.get(n(this, pe)) - n(this, K, re), n(this, h).seek(n(this, h).offsets.get(n(this, U))), n(this, h).writeEBML(n(this, U)), n(this, h).seek(t);
    }
    c(this, j, ne).call(this), n(this, h).finalize(), _(this, Ae, true);
  }
};
w = /* @__PURE__ */ new WeakMap();
h = /* @__PURE__ */ new WeakMap();
We = /* @__PURE__ */ new WeakMap();
Le = /* @__PURE__ */ new WeakMap();
U = /* @__PURE__ */ new WeakMap();
pe = /* @__PURE__ */ new WeakMap();
ie = /* @__PURE__ */ new WeakMap();
Z = /* @__PURE__ */ new WeakMap();
we = /* @__PURE__ */ new WeakMap();
H = /* @__PURE__ */ new WeakMap();
oe = /* @__PURE__ */ new WeakMap();
de = /* @__PURE__ */ new WeakMap();
M = /* @__PURE__ */ new WeakMap();
Fe = /* @__PURE__ */ new WeakMap();
le = /* @__PURE__ */ new WeakMap();
D = /* @__PURE__ */ new WeakMap();
P = /* @__PURE__ */ new WeakMap();
q = /* @__PURE__ */ new WeakMap();
Te = /* @__PURE__ */ new WeakMap();
Ee = /* @__PURE__ */ new WeakMap();
ce = /* @__PURE__ */ new WeakMap();
he = /* @__PURE__ */ new WeakMap();
tt = /* @__PURE__ */ new WeakMap();
De = /* @__PURE__ */ new WeakMap();
Ae = /* @__PURE__ */ new WeakMap();
bt = /* @__PURE__ */ new WeakSet();
ci = function(t) {
  if (typeof t != "object") throw new TypeError("The muxer requires an options object to be passed to its constructor.");
  if (!(t.target instanceof st)) throw new TypeError("The target must be provided and an instance of Target.");
  if (t.video) {
    if (typeof t.video.codec != "string") throw new TypeError(`Invalid video codec: ${t.video.codec}. Must be a string.`);
    if (!Number.isInteger(t.video.width) || t.video.width <= 0) throw new TypeError(`Invalid video width: ${t.video.width}. Must be a positive integer.`);
    if (!Number.isInteger(t.video.height) || t.video.height <= 0) throw new TypeError(`Invalid video height: ${t.video.height}. Must be a positive integer.`);
    if (t.video.frameRate !== void 0 && (!Number.isFinite(t.video.frameRate) || t.video.frameRate <= 0)) throw new TypeError(`Invalid video frame rate: ${t.video.frameRate}. Must be a positive number.`);
    if (t.video.alpha !== void 0 && typeof t.video.alpha != "boolean") throw new TypeError(`Invalid video alpha: ${t.video.alpha}. Must be a boolean.`);
  }
  if (t.audio) {
    if (typeof t.audio.codec != "string") throw new TypeError(`Invalid audio codec: ${t.audio.codec}. Must be a string.`);
    if (!Number.isInteger(t.audio.numberOfChannels) || t.audio.numberOfChannels <= 0) throw new TypeError(`Invalid number of audio channels: ${t.audio.numberOfChannels}. Must be a positive integer.`);
    if (!Number.isInteger(t.audio.sampleRate) || t.audio.sampleRate <= 0) throw new TypeError(`Invalid audio sample rate: ${t.audio.sampleRate}. Must be a positive integer.`);
    if (t.audio.bitDepth !== void 0 && (!Number.isInteger(t.audio.bitDepth) || t.audio.bitDepth <= 0)) throw new TypeError(`Invalid audio bit depth: ${t.audio.bitDepth}. Must be a positive integer.`);
  }
  if (t.subtitles && typeof t.subtitles.codec != "string") throw new TypeError(`Invalid subtitles codec: ${t.subtitles.codec}. Must be a string.`);
  if (t.type !== void 0 && !["webm", "matroska"].includes(t.type)) throw new TypeError(`Invalid type: ${t.type}. Must be 'webm' or 'matroska'.`);
  if (t.firstTimestampBehavior && !Yi.includes(t.firstTimestampBehavior)) throw new TypeError(`Invalid first timestamp behavior: ${t.firstTimestampBehavior}`);
  if (t.streaming !== void 0 && typeof t.streaming != "boolean") throw new TypeError(`Invalid streaming option: ${t.streaming}. Must be a boolean.`);
};
yt = /* @__PURE__ */ new WeakSet();
hi = function() {
  n(this, h) instanceof ze && n(this, h).target.options.onHeader && n(this, h).startTrackingWrites(), c(this, Mt, ui).call(this), n(this, w).streaming || c(this, Wt, wi).call(this), c(this, Lt, gi).call(this), c(this, It, fi).call(this), c(this, Ut, pi).call(this), n(this, w).streaming || (c(this, at, Dt).call(this), c(this, ot, Pt).call(this)), c(this, zt, _i).call(this), c(this, j, ne).call(this);
};
Mt = /* @__PURE__ */ new WeakSet();
ui = function() {
  let t = { id: 440786851, data: [{ id: 17030, data: 1 }, { id: 17143, data: 1 }, { id: 17138, data: 4 }, { id: 17139, data: 8 }, { id: 17026, data: n(this, w).type ?? "webm" }, { id: 17031, data: 2 }, { id: 17029, data: 2 }] };
  n(this, h).writeEBML(t);
};
It = /* @__PURE__ */ new WeakSet();
fi = function() {
  _(this, we, { id: 236, size: 4, data: new Uint8Array(Be) }), _(this, H, { id: 236, size: 4, data: new Uint8Array(Be) }), _(this, oe, { id: 236, size: 4, data: new Uint8Array(Be) });
};
Ut = /* @__PURE__ */ new WeakSet();
pi = function() {
  _(this, Z, { id: 21936, data: [{ id: 21937, data: 2 }, { id: 21946, data: 2 }, { id: 21947, data: 2 }, { id: 21945, data: 0 }] });
};
Wt = /* @__PURE__ */ new WeakSet();
wi = function() {
  const t = new Uint8Array([28, 83, 187, 107]), e = new Uint8Array([21, 73, 169, 102]), i = new Uint8Array([22, 84, 174, 107]);
  _(this, U, { id: 290298740, data: [{ id: 19899, data: [{ id: 21419, data: t }, { id: 21420, size: 5, data: 0 }] }, { id: 19899, data: [{ id: 21419, data: e }, { id: 21420, size: 5, data: 0 }] }, { id: 19899, data: [{ id: 21419, data: i }, { id: 21420, size: 5, data: 0 }] }] });
};
Lt = /* @__PURE__ */ new WeakSet();
gi = function() {
  let t = { id: 17545, data: new xt(0) };
  _(this, ie, t);
  let e = { id: 357149030, data: [{ id: 2807729, data: 1e6 }, { id: 19840, data: Kt }, { id: 22337, data: Kt }, n(this, w).streaming ? null : t] };
  _(this, Le, e);
};
at = /* @__PURE__ */ new WeakSet();
Dt = function() {
  let t = { id: 374648427, data: [] };
  _(this, pe, t), n(this, w).video && t.data.push({ id: 174, data: [{ id: 215, data: fe }, { id: 29637, data: fe }, { id: 131, data: Gi }, { id: 134, data: n(this, w).video.codec }, n(this, we), n(this, w).video.frameRate ? { id: 2352003, data: 1e9 / n(this, w).video.frameRate } : null, { id: 224, data: [{ id: 176, data: n(this, w).video.width }, { id: 186, data: n(this, w).video.height }, n(this, w).video.alpha ? { id: 21440, data: 1 } : null, n(this, Z)] }] }), n(this, w).audio && (_(this, H, n(this, w).streaming ? n(this, H) || null : { id: 236, size: 4, data: new Uint8Array(Be) }), t.data.push({ id: 174, data: [{ id: 215, data: Ue }, { id: 29637, data: Ue }, { id: 131, data: qi }, { id: 134, data: n(this, w).audio.codec }, n(this, H), { id: 225, data: [{ id: 181, data: new Zt(n(this, w).audio.sampleRate) }, { id: 159, data: n(this, w).audio.numberOfChannels }, n(this, w).audio.bitDepth ? { id: 25188, data: n(this, w).audio.bitDepth } : null] }] })), n(this, w).subtitles && t.data.push({ id: 174, data: [{ id: 215, data: et }, { id: 29637, data: et }, { id: 131, data: ji }, { id: 134, data: n(this, w).subtitles.codec }, n(this, oe)] });
};
ot = /* @__PURE__ */ new WeakSet();
Pt = function() {
  let t = { id: 408125543, size: n(this, w).streaming ? -1 : di, data: [n(this, w).streaming ? null : n(this, U), n(this, Le), n(this, pe)] };
  if (_(this, We, t), n(this, h).writeEBML(t), n(this, h) instanceof ze && n(this, h).target.options.onHeader) {
    let { data: e, start: i } = n(this, h).getTrackedWrites();
    n(this, h).target.options.onHeader(e, i);
  }
};
zt = /* @__PURE__ */ new WeakSet();
_i = function() {
  _(this, de, { id: 475249515, data: [] });
};
j = /* @__PURE__ */ new WeakSet();
ne = function() {
  n(this, h) instanceof At && n(this, h).flush();
};
K = /* @__PURE__ */ new WeakSet();
re = function() {
  return n(this, h).dataOffsets.get(n(this, We));
};
vt = /* @__PURE__ */ new WeakSet();
mi = function(t) {
  if (t.decoderConfig) {
    if (t.decoderConfig.colorSpace) {
      let e = t.decoderConfig.colorSpace;
      if (_(this, De, e), n(this, Z).data = [{ id: 21937, data: { rgb: 1, bt709: 1, bt470bg: 5, smpte170m: 6 }[e.matrix] }, { id: 21946, data: { bt709: 1, smpte170m: 6, "iec61966-2-1": 13 }[e.transfer] }, { id: 21947, data: { bt709: 1, bt470bg: 5, smpte170m: 6 }[e.primaries] }, { id: 21945, data: [1, 2][Number(e.fullRange)] }], !n(this, w).streaming) {
        let i = n(this, h).pos;
        n(this, h).seek(n(this, h).offsets.get(n(this, Z))), n(this, h).writeEBML(n(this, Z)), n(this, h).seek(i);
      }
    }
    t.decoderConfig.description && (n(this, w).streaming ? _(this, we, c(this, ue, Pe).call(this, t.decoderConfig.description)) : c(this, Me, it).call(this, n(this, we), t.decoderConfig.description));
  }
};
St = /* @__PURE__ */ new WeakSet();
bi = function(t) {
  if (t.type !== "key" || !n(this, De)) return;
  let e = 0;
  if (ee(t.data, 0, 2) !== 2) return;
  e += 2;
  let i = (ee(t.data, e + 1, e + 2) << 1) + ee(t.data, e + 0, e + 1);
  e += 2, i === 3 && e++;
  let r = ee(t.data, e + 0, e + 1);
  if (e++, r) return;
  let s = ee(t.data, e + 0, e + 1);
  if (e++, s !== 0) return;
  e += 2;
  let a = ee(t.data, e + 0, e + 24);
  if (e += 24, a !== 4817730) return;
  i >= 2 && e++;
  let o = { rgb: 7, bt709: 2, bt470bg: 1, smpte170m: 3 }[n(this, De).matrix];
  Hi(t.data, e + 0, e + 3, o);
};
ve = /* @__PURE__ */ new WeakSet();
je = function() {
  let t = Math.min(n(this, w).video ? n(this, ce) : 1 / 0, n(this, w).audio ? n(this, he) : 1 / 0), e = n(this, q);
  for (; e.length > 0 && e[0].timestamp <= t; ) c(this, A, I).call(this, e.shift(), !n(this, w).video && !n(this, w).audio);
};
Se = /* @__PURE__ */ new WeakSet();
Ke = function(t, e, i, r, s, a) {
  let o = c(this, Ft, yi).call(this, i, r);
  return { data: t, additions: a, type: e, timestamp: o, duration: s, trackNumber: r };
};
Ft = /* @__PURE__ */ new WeakSet();
yi = function(t, e) {
  let i = e === fe ? n(this, ce) : e === Ue ? n(this, he) : n(this, tt);
  if (e !== et) {
    let r = e === fe ? n(this, Te) : n(this, Ee);
    if (n(this, w).firstTimestampBehavior === "strict" && i === -1 && t !== 0) throw new Error(`The first chunk for your media track must have a timestamp of 0 (received ${t}). Non-zero first timestamps are often caused by directly piping frames or audio data from a MediaStreamTrack into the encoder. Their timestamps are typically relative to the age of the document, which is probably what you want.

If you want to offset all timestamps of a track such that the first one is zero, set firstTimestampBehavior: 'offset' in the options.
If you want to allow non-zero first timestamps, set firstTimestampBehavior: 'permissive'.
`);
    n(this, w).firstTimestampBehavior === "offset" && (t -= r);
  }
  if (t < i) throw new Error(`Timestamps must be monotonically increasing (went from ${i} to ${t}).`);
  if (t < 0) throw new Error(`Timestamps must be non-negative (received ${t}).`);
  return t;
};
A = /* @__PURE__ */ new WeakSet();
I = function(t, e) {
  n(this, w).streaming && !n(this, pe) && (c(this, at, Dt).call(this), c(this, ot, Pt).call(this));
  let i = Math.floor(t.timestamp / 1e3), r = i - n(this, Fe), s = e && t.type === "key" && r >= 1e3, a = r >= Ki;
  if ((!n(this, M) || s || a) && (c(this, Ht, vi).call(this, i), r = 0), r < 0) return;
  let o = new Uint8Array(4), l = new DataView(o.buffer);
  if (l.setUint8(0, 128 | t.trackNumber), l.setInt16(1, r, false), t.duration === void 0 && !t.additions) {
    l.setUint8(3, +(t.type === "key") << 7);
    let m = { id: 163, data: [o, t.data] };
    n(this, h).writeEBML(m);
  } else {
    let m = Math.floor(t.duration / 1e3), C = { id: 160, data: [{ id: 161, data: [o, t.data] }, t.duration !== void 0 ? { id: 155, data: m } : null, t.additions ? { id: 30113, data: t.additions } : null] };
    n(this, h).writeEBML(C);
  }
  _(this, le, Math.max(n(this, le), i));
};
ue = /* @__PURE__ */ new WeakSet();
Pe = function(t) {
  return { id: 25506, size: 4, data: new Uint8Array(t) };
};
Me = /* @__PURE__ */ new WeakSet();
it = function(t, e) {
  let i = n(this, h).pos;
  n(this, h).seek(n(this, h).offsets.get(t));
  let r = 6 + e.byteLength, s = Be - r;
  if (s < 0) {
    let a = e.byteLength + s;
    e instanceof ArrayBuffer ? e = e.slice(0, a) : e = e.buffer.slice(0, a), s = 0;
  }
  t = [c(this, ue, Pe).call(this, e), { id: 236, size: 4, data: new Uint8Array(s) }], n(this, h).writeEBML(t), n(this, h).seek(i);
};
Ht = /* @__PURE__ */ new WeakSet();
vi = function(t) {
  n(this, M) && c(this, nt, Nt).call(this), n(this, h) instanceof ze && n(this, h).target.options.onCluster && n(this, h).startTrackingWrites(), _(this, M, { id: 524531317, size: n(this, w).streaming ? -1 : li, data: [{ id: 231, data: t }] }), n(this, h).writeEBML(n(this, M)), _(this, Fe, t);
  let e = n(this, h).offsets.get(n(this, M)) - n(this, K, re);
  n(this, de).data.push({ id: 187, data: [{ id: 179, data: t }, n(this, w).video ? { id: 183, data: [{ id: 247, data: fe }, { id: 241, data: e }] } : null, n(this, w).audio ? { id: 183, data: [{ id: 247, data: Ue }, { id: 241, data: e }] } : null] });
};
nt = /* @__PURE__ */ new WeakSet();
Nt = function() {
  if (!n(this, w).streaming) {
    let t = n(this, h).pos - n(this, h).dataOffsets.get(n(this, M)), e = n(this, h).pos;
    n(this, h).seek(n(this, h).offsets.get(n(this, M)) + 4), n(this, h).writeEBMLVarInt(t, li), n(this, h).seek(e);
  }
  if (n(this, h) instanceof ze && n(this, h).target.options.onCluster) {
    let { data: t, start: e } = n(this, h).getTrackedWrites();
    n(this, h).target.options.onCluster(t, e, n(this, Fe));
  }
};
ke = /* @__PURE__ */ new WeakSet();
Ye = function() {
  if (n(this, Ae)) throw new Error("Cannot add new video or audio chunks after the file has been finalized.");
};
new TextEncoder();
const Xi = Object.freeze(Object.defineProperty({ __proto__: null, ArrayBufferTarget: Rt, FileSystemWritableFileStreamTarget: ei, Muxer: Si, StreamTarget: Bt }, Symbol.toStringTag, { value: "Module" }));
class Ji {
  constructor(e, i, r) {
    __publicField(this, "isRecording", false);
    __publicField(this, "renderer");
    __publicField(this, "worldBridge");
    __publicField(this, "canvas");
    this.renderer = e, this.worldBridge = i, this.canvas = r;
  }
  get recording() {
    return this.isRecording;
  }
  async record(e, i, r) {
    if (this.isRecording) return;
    this.isRecording = true;
    const s = Math.ceil(e.fps * e.duration);
    console.log(`Starting recording: ${s} frames @ ${e.fps}fps (VP9)`);
    const a = new Si({ target: new Rt(), video: { codec: "V_VP9", width: this.canvas.width, height: this.canvas.height, frameRate: e.fps } }), o = new VideoEncoder({ output: (l, m) => a.addVideoChunk(l, m), error: (l) => console.error("VideoEncoder Error:", l) });
    o.configure({ codec: "vp09.00.10.08", width: this.canvas.width, height: this.canvas.height, bitrate: 12e6 });
    try {
      await this.renderAndEncode(s, e, o, i, e.startFrame || 0), await o.flush(), a.finalize();
      const { buffer: l } = a.target, m = new Blob([l], { type: "video/webm" }), C = URL.createObjectURL(m);
      r(C, m);
    } catch (l) {
      throw console.error("Recording failed:", l), l;
    } finally {
      this.isRecording = false;
    }
  }
  async recordChunks(e, i) {
    if (this.isRecording) throw new Error("Already recording");
    this.isRecording = true;
    const r = [], s = Math.ceil(e.fps * e.duration), a = new VideoEncoder({ output: (o, l) => {
      const m = new Uint8Array(o.byteLength);
      o.copyTo(m), r.push({ type: o.type, timestamp: o.timestamp, duration: o.duration, data: m.buffer, decoderConfig: l == null ? void 0 : l.decoderConfig });
    }, error: (o) => console.error("VideoEncoder Error:", o) });
    a.configure({ codec: "vp09.00.10.08", width: this.canvas.width, height: this.canvas.height, bitrate: 12e6 });
    try {
      return await this.renderAndEncode(s, e, a, i, e.startFrame || 0), await a.flush(), r;
    } finally {
      this.isRecording = false;
    }
  }
  async renderAndEncode(e, i, r, s, a = 0) {
    for (let o = 0; o < e; o++) {
      s(o, e), await new Promise((f) => setTimeout(f, 0));
      const l = a + o, m = l / i.fps;
      this.worldBridge.update(m), await this.updateSceneBuffers(), await this.renderFrame(i.spp, i.batch), r.encodeQueueSize > 5 && await r.flush();
      const C = new VideoFrame(this.canvas, { timestamp: l * 1e6 / i.fps, duration: 1e6 / i.fps });
      r.encode(C, { keyFrame: o % i.fps === 0 }), C.close();
    }
  }
  async updateSceneBuffers() {
    let e = false;
    e || (e = this.renderer.updateCombinedBVH(this.worldBridge.tlas, this.worldBridge.blas)), e || (e = this.renderer.updateBuffer("instance", this.worldBridge.instances)), e || (e = this.renderer.updateCombinedGeometry(this.worldBridge.vertices, this.worldBridge.normals, this.worldBridge.uvs)), e || (e = this.renderer.updateBuffer("index", this.worldBridge.indices)), e || (e = this.renderer.updateBuffer("attr", this.worldBridge.attributes)), this.worldBridge.updateCamera(this.canvas.width, this.canvas.height), this.renderer.updateSceneUniforms(this.worldBridge.cameraData, 0), e && this.renderer.recreateBindGroup(), this.renderer.resetAccumulation();
  }
  async renderFrame(e, i) {
    let r = 0;
    for (; r < e; ) {
      const s = Math.min(i, e - r);
      for (let a = 0; a < s; a++) this.renderer.render(r + a);
      r += s, await this.renderer.device.queue.onSubmittedWorkDone(), r < e && await new Promise((a) => setTimeout(a, 0));
    }
  }
}
const Zi = { iceServers: [{ urls: "stun:stun.l.google.com:19302" }] };
class Yt {
  constructor(e, i) {
    __publicField(this, "pc");
    __publicField(this, "dc", null);
    __publicField(this, "remoteId");
    __publicField(this, "sendSignal");
    __publicField(this, "receiveBuffer", new Uint8Array(0));
    __publicField(this, "receivedBytes", 0);
    __publicField(this, "sceneMeta", null);
    __publicField(this, "resultMeta", null);
    __publicField(this, "onSceneReceived", null);
    __publicField(this, "onRenderRequest", null);
    __publicField(this, "onRenderResult", null);
    __publicField(this, "onDataChannelOpen", null);
    __publicField(this, "onAckReceived", null);
    this.remoteId = e, this.sendSignal = i, this.pc = new RTCPeerConnection(Zi), this.pc.onicecandidate = (r) => {
      r.candidate && this.sendSignal({ type: "candidate", candidate: r.candidate.toJSON(), targetId: this.remoteId });
    };
  }
  async startAsHost() {
    this.dc = this.pc.createDataChannel("render-channel"), this.setupDataChannel();
    const e = await this.pc.createOffer();
    await this.pc.setLocalDescription(e), this.sendSignal({ type: "offer", sdp: e, targetId: this.remoteId });
  }
  async handleOffer(e) {
    this.pc.ondatachannel = (r) => {
      this.dc = r.channel, this.setupDataChannel();
    }, await this.pc.setRemoteDescription(new RTCSessionDescription(e));
    const i = await this.pc.createAnswer();
    await this.pc.setLocalDescription(i), this.sendSignal({ type: "answer", sdp: i, targetId: this.remoteId });
  }
  async handleAnswer(e) {
    await this.pc.setRemoteDescription(new RTCSessionDescription(e));
  }
  async handleCandidate(e) {
    await this.pc.addIceCandidate(new RTCIceCandidate(e));
  }
  async sendScene(e, i, r) {
    if (!this.dc || this.dc.readyState !== "open") return;
    let s;
    typeof e == "string" ? s = new TextEncoder().encode(e) : s = new Uint8Array(e);
    const a = { type: "SCENE_INIT", totalBytes: s.byteLength, config: { ...r, fileType: i } };
    this.sendData(a), await this.sendBinaryChunks(s);
  }
  async sendRenderResult(e, i) {
    if (!this.dc || this.dc.readyState !== "open") return;
    let r = 0;
    const s = e.map((l) => {
      const m = l.data.byteLength;
      return r += m, { type: l.type, timestamp: l.timestamp, duration: l.duration, size: m, decoderConfig: l.decoderConfig };
    });
    console.log(`[RTC] Sending Render Result: ${r} bytes, ${e.length} chunks`), this.sendData({ type: "RENDER_RESULT", startFrame: i, totalBytes: r, chunksMeta: s });
    const a = new Uint8Array(r);
    let o = 0;
    for (const l of e) a.set(new Uint8Array(l.data), o), o += l.data.byteLength;
    await this.sendBinaryChunks(a);
  }
  async sendBinaryChunks(e) {
    let r = 0;
    const s = () => new Promise((a) => {
      const o = setInterval(() => {
        (!this.dc || this.dc.bufferedAmount < 65536) && (clearInterval(o), a());
      }, 5);
    });
    for (; r < e.byteLength; ) {
      this.dc && this.dc.bufferedAmount > 256 * 1024 && await s();
      const a = Math.min(r + 16384, e.byteLength);
      if (this.dc) try {
        this.dc.send(e.subarray(r, a));
      } catch {
      }
      r = a, r % (16384 * 5) === 0 && await new Promise((o) => setTimeout(o, 0));
    }
    console.log("[RTC] Transfer Complete");
  }
  setupDataChannel() {
    this.dc && (this.dc.binaryType = "arraybuffer", this.dc.onopen = () => {
      console.log("[RTC] DataChannel Open"), this.onDataChannelOpen && this.onDataChannelOpen();
    }, this.dc.onmessage = (e) => {
      const i = e.data;
      if (typeof i == "string") try {
        const r = JSON.parse(i);
        this.handleControlMessage(r);
      } catch {
      }
      else i instanceof ArrayBuffer && this.handleBinaryChunk(i);
    });
  }
  handleControlMessage(e) {
    var _a;
    e.type === "SCENE_INIT" ? (console.log(`[RTC] Receiving Scene: ${e.config.fileType}, ${e.totalBytes} bytes`), this.sceneMeta = { config: e.config, totalBytes: e.totalBytes }, this.receiveBuffer = new Uint8Array(e.totalBytes), this.receivedBytes = 0) : e.type === "SCENE_ACK" ? (console.log(`[RTC] Scene ACK: ${e.receivedBytes} bytes`), this.onAckReceived && this.onAckReceived(e.receivedBytes)) : e.type === "RENDER_REQUEST" ? (console.log(`[RTC] Render Request: Frame ${e.startFrame}, Count ${e.frameCount}`), (_a = this.onRenderRequest) == null ? void 0 : _a.call(this, e.startFrame, e.frameCount, e.config)) : e.type === "RENDER_RESULT" && (console.log(`[RTC] Receiving Render Result: ${e.totalBytes} bytes`), this.resultMeta = { startFrame: e.startFrame, totalBytes: e.totalBytes, chunksMeta: e.chunksMeta }, this.receiveBuffer = new Uint8Array(e.totalBytes), this.receivedBytes = 0);
  }
  handleBinaryChunk(e) {
    var _a, _b;
    const i = new Uint8Array(e);
    if (this.receiveBuffer.set(i, this.receivedBytes), this.receivedBytes += i.byteLength, this.sceneMeta) {
      if (this.receivedBytes >= this.sceneMeta.totalBytes) {
        console.log("[RTC] Scene Download Complete!");
        let r;
        this.sceneMeta.config.fileType === "obj" ? r = new TextDecoder().decode(this.receiveBuffer) : r = this.receiveBuffer.buffer, (_a = this.onSceneReceived) == null ? void 0 : _a.call(this, r, this.sceneMeta.config), this.sceneMeta = null;
      }
    } else if (this.resultMeta && this.receivedBytes >= this.resultMeta.totalBytes) {
      console.log("[RTC] Render Result Complete!");
      const r = [];
      let s = 0;
      for (const a of this.resultMeta.chunksMeta) {
        const o = this.receiveBuffer.slice(s, s + a.size);
        r.push({ type: a.type, timestamp: a.timestamp, duration: a.duration, data: o.buffer, decoderConfig: a.decoderConfig }), s += a.size;
      }
      (_b = this.onRenderResult) == null ? void 0 : _b.call(this, r, this.resultMeta.startFrame), this.resultMeta = null;
    }
  }
  sendData(e) {
    var _a;
    ((_a = this.dc) == null ? void 0 : _a.readyState) === "open" && this.dc.send(JSON.stringify(e));
  }
  sendAck(e) {
    this.sendData({ type: "SCENE_ACK", receivedBytes: e });
  }
  sendRenderRequest(e, i, r) {
    const s = { type: "RENDER_REQUEST", startFrame: e, frameCount: i, config: r };
    this.sendData(s);
  }
  close() {
    this.dc && (this.dc.close(), this.dc = null), this.pc && this.pc.close(), console.log(`[RTC] Connection closed: ${this.remoteId}`);
  }
}
class Qi {
  constructor() {
    __publicField(this, "ws", null);
    __publicField(this, "myRole", null);
    __publicField(this, "workers", /* @__PURE__ */ new Map());
    __publicField(this, "hostClient", null);
    __publicField(this, "onStatusChange", null);
    __publicField(this, "onWorkerJoined", null);
    __publicField(this, "onHostConnected", null);
    __publicField(this, "onSceneReceived", null);
    __publicField(this, "onHostHello", null);
    __publicField(this, "onRenderResult", null);
    __publicField(this, "onRenderRequest", null);
  }
  connect(e) {
    var _a;
    this.ws || (this.myRole = e, (_a = this.onStatusChange) == null ? void 0 : _a.call(this, `Connecting as ${e.toUpperCase()}...`), this.ws = new WebSocket(S.signalingServerUrl), this.ws.onopen = () => {
      var _a2;
      console.log("WS Connected"), (_a2 = this.onStatusChange) == null ? void 0 : _a2.call(this, `Waiting for Peer (${e.toUpperCase()})`), this.sendSignal({ type: e === "host" ? "register_host" : "register_worker" });
    }, this.ws.onmessage = (i) => {
      const r = JSON.parse(i.data);
      this.handleMessage(r);
    }, this.ws.onclose = () => {
      var _a2;
      (_a2 = this.onStatusChange) == null ? void 0 : _a2.call(this, "Disconnected"), this.ws = null;
    });
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
  async sendRenderResult(e, i) {
    this.hostClient && await this.hostClient.sendRenderResult(e, i);
  }
  sendSignal(e) {
    var _a;
    ((_a = this.ws) == null ? void 0 : _a.readyState) === WebSocket.OPEN && this.ws.send(JSON.stringify(e));
  }
  async handleMessage(e) {
    this.myRole === "host" ? await this.handleHostMessage(e) : await this.handleWorkerMessage(e);
  }
  async handleHostMessage(e) {
    var _a, _b;
    switch (e.type) {
      case "worker_joined":
        console.log(`Worker joined: ${e.workerId}`);
        const i = new Yt(e.workerId, (r) => this.sendSignal(r));
        this.workers.set(e.workerId, i), i.onDataChannelOpen = () => {
          var _a2;
          console.log(`[Host] Open for ${e.workerId}`), i.sendData({ type: "HELLO", msg: "Hello from Host!" }), (_a2 = this.onWorkerJoined) == null ? void 0 : _a2.call(this, e.workerId);
        }, i.onAckReceived = (r) => {
          console.log(`Worker ${e.workerId} ACK: ${r}`);
        }, i.onRenderResult = (r, s) => {
          var _a2;
          console.log(`Received Render Result from ${e.workerId}: ${r.length} chunks`), (_a2 = this.onRenderResult) == null ? void 0 : _a2.call(this, r, s, e.workerId);
        }, await i.startAsHost();
        break;
      case "answer":
        e.fromId && await ((_a = this.workers.get(e.fromId)) == null ? void 0 : _a.handleAnswer(e.sdp));
        break;
      case "candidate":
        e.fromId && await ((_b = this.workers.get(e.fromId)) == null ? void 0 : _b.handleCandidate(e.candidate));
        break;
      case "host_exists":
        alert("Host already exists!");
        break;
    }
  }
  async handleWorkerMessage(e) {
    var _a, _b, _c;
    switch (e.type) {
      case "offer":
        e.fromId && (this.hostClient = new Yt(e.fromId, (i) => this.sendSignal(i)), await this.hostClient.handleOffer(e.sdp), (_a = this.onStatusChange) == null ? void 0 : _a.call(this, "Connected to Host!"), (_b = this.onHostConnected) == null ? void 0 : _b.call(this), this.hostClient.onDataChannelOpen = () => {
          var _a2, _b2;
          (_a2 = this.hostClient) == null ? void 0 : _a2.sendData({ type: "HELLO", msg: "Hello from Worker!" }), (_b2 = this.onHostHello) == null ? void 0 : _b2.call(this);
        }, this.hostClient.onSceneReceived = (i, r) => {
          var _a2, _b2;
          (_a2 = this.onSceneReceived) == null ? void 0 : _a2.call(this, i, r);
          const s = typeof i == "string" ? i.length : i.byteLength;
          (_b2 = this.hostClient) == null ? void 0 : _b2.sendAck(s);
        }, this.hostClient.onRenderRequest = (i, r, s) => {
          var _a2;
          (_a2 = this.onRenderRequest) == null ? void 0 : _a2.call(this, i, r, s);
        });
        break;
      case "candidate":
        await ((_c = this.hostClient) == null ? void 0 : _c.handleCandidate(e.candidate));
        break;
    }
  }
  async broadcastScene(e, i, r) {
    const s = Array.from(this.workers.values()).map((a) => a.sendScene(e, i, r));
    await Promise.all(s);
  }
  async sendRenderRequest(e, i, r, s) {
    const a = this.workers.get(e);
    a && await a.sendRenderRequest(i, r, s);
  }
}
let B = false, W = null, Y = null;
const u = new zi(), k = new Bi(u.canvas), y = new Pi(), Xe = new Ji(k, y, u.canvas), x = new Qi();
let X = 0, kt = 0, Ne = 0, Xt = performance.now();
const en = () => {
  const t = parseInt(u.inputDepth.value, 10) || S.defaultDepth, e = parseInt(u.inputSPP.value, 10) || S.defaultSPP;
  k.buildPipeline(t, e);
}, Vt = () => {
  const { width: t, height: e } = u.getRenderConfig();
  k.updateScreenSize(t, e), y.hasWorld && (y.updateCamera(t, e), k.updateSceneUniforms(y.cameraData, 0)), k.recreateBindGroup(), k.resetAccumulation(), X = 0, kt = 0;
}, Je = async (t, e = true) => {
  B = false, console.log(`Loading Scene: ${t}...`);
  let i, r;
  t === "viewer" && W && (Y === "obj" ? i = W : Y === "glb" && (r = new Uint8Array(W))), y.loadScene(t, i, r), y.printStats(), await k.loadTexturesFromWorld(y), await tn(), Vt(), u.updateAnimList(y.getAnimationList()), e && (B = true, u.updateRenderButton(true));
}, tn = async () => {
  k.updateCombinedGeometry(y.vertices, y.normals, y.uvs), k.updateCombinedBVH(y.tlas, y.blas), k.updateBuffer("index", y.indices), k.updateBuffer("attr", y.attributes), k.updateBuffer("instance", y.instances);
}, rt = () => {
  if (Xe.recording || (requestAnimationFrame(rt), !B || !y.hasWorld)) return;
  let t = parseInt(u.inputUpdateInterval.value, 10) || 0;
  if (t < 0 && (t = 0), t > 0 && X >= t) {
    y.update(kt / t / 60);
    let i = false;
    i || (i = k.updateCombinedBVH(y.tlas, y.blas)), i || (i = k.updateBuffer("instance", y.instances)), i || (i = k.updateCombinedGeometry(y.vertices, y.normals, y.uvs)), i || (i = k.updateBuffer("index", y.indices)), i || (i = k.updateBuffer("attr", y.attributes)), y.updateCamera(u.canvas.width, u.canvas.height), k.updateSceneUniforms(y.cameraData, 0), i && k.recreateBindGroup(), k.resetAccumulation(), X = 0;
  }
  X++, Ne++, kt++, k.render(X);
  const e = performance.now();
  e - Xt >= 1e3 && (u.updateStats(Ne, 1e3 / Ne, X), Ne = 0, Xt = e);
}, nn = () => {
  u.onRenderStart = () => {
    B = true;
  }, u.onRenderStop = () => {
    B = false;
  }, u.onSceneSelect = (f) => Je(f, false), u.onResolutionChange = Vt, u.onRecompile = (f, g) => {
    B = false, k.buildPipeline(f, g), k.recreateBindGroup(), k.resetAccumulation(), X = 0, B = true;
  }, u.onFileSelect = async (f) => {
    var _a;
    ((_a = f.name.split(".").pop()) == null ? void 0 : _a.toLowerCase()) === "obj" ? (W = await f.text(), Y = "obj") : (W = await f.arrayBuffer(), Y = "glb"), u.sceneSelect.value = "viewer", Je("viewer", false);
  }, u.onAnimSelect = (f) => y.setAnimation(f);
  let t = [], e = /* @__PURE__ */ new Map(), i = 0, r = 0, s = 0, a = null;
  const o = 20, l = async (f) => {
    if (t.length === 0) return;
    const g = t.shift();
    g && (console.log(`Assigning Job ${g.start} - ${g.start + g.count} to ${f}`), await x.sendRenderRequest(f, g.start, g.count, { ...a, fileType: "obj" }));
  };
  u.onRecordStart = async () => {
    if (!Xe.recording) if (m === "host" && x.getWorkerCount() > 0) {
      a = u.getRenderConfig(), s = Math.ceil(a.fps * a.duration);
      const f = x.getWorkerIds();
      if (f.length === 0 || !confirm(`Distribute recording to ${f.length} workers (Dynamic Load Balancing)?`)) return;
      t = [], e.clear(), i = 0;
      for (let g = 0; g < s; g += o) {
        const v = Math.min(o, s - g);
        t.push({ start: g, count: v });
      }
      r = t.length, f.forEach((g) => l(g)), u.setStatus(`Distributed Progress: 0 / ${r} jobs`);
    } else {
      B = false, u.setRecordingState(true);
      const f = u.getRenderConfig();
      try {
        await Xe.record(f, (g, v) => u.setRecordingState(true, `Rec: ${g}/${v} (${Math.round(g / v * 100)}%)`), (g) => {
          const v = document.createElement("a");
          v.href = g, v.download = `raytrace_${Date.now()}.webm`, v.click(), URL.revokeObjectURL(g);
        });
      } catch {
        alert("Recording failed.");
      } finally {
        u.setRecordingState(false), B = true, u.updateRenderButton(true), requestAnimationFrame(rt);
      }
    }
  };
  let m = null;
  u.onConnectHost = () => {
    m === "host" ? (x.disconnect(), m = null, u.setConnectionState(null)) : (x.connect("host"), m = "host", u.setConnectionState("host"));
  }, u.onConnectWorker = () => {
    m === "worker" ? (x.disconnect(), m = null, u.setConnectionState(null)) : (x.connect("worker"), m = "worker", u.setConnectionState("worker"));
  }, u.onSendScene = async () => {
    if (!W || !Y) {
      alert("No scene loaded!");
      return;
    }
    u.setSendSceneText("Sending..."), u.setSendSceneEnabled(false);
    const f = u.getRenderConfig();
    await x.broadcastScene(W, Y, f), u.setSendSceneText("Send Scene"), u.setSendSceneEnabled(true);
  }, x.onStatusChange = (f) => u.setStatus(`Status: ${f}`), x.onWorkerJoined = (f) => {
    u.setStatus(`Worker Joined: ${f}`), u.setSendSceneEnabled(true), m === "host" && t.length > 0 && l(f);
  }, x.onRenderRequest = async (f, g, v) => {
    console.log(`[Worker] Received Render Request: Frames ${f} - ${f + g}`), u.setStatus(`Remote Rendering: ${f}-${f + g}`), B = false;
    const N = { ...v, startFrame: f, duration: g / v.fps };
    try {
      u.setRecordingState(true, `Remote: ${g} f`);
      const V = await Xe.recordChunks(N, (dt, He) => u.setRecordingState(true, `Remote: ${dt}/${He}`));
      console.log("Sending Chunks back to Host..."), u.setRecordingState(true, "Uploading..."), await x.sendRenderResult(V, f), u.setRecordingState(false), u.setStatus("Idle");
    } catch (V) {
      console.error("Remote Recording Failed", V), u.setStatus("Recording Failed");
    } finally {
      B = true, requestAnimationFrame(rt);
    }
  }, x.onRenderResult = async (f, g, v) => {
    console.log(`[Host] Received ${f.length} chunks for ${g} from ${v}`), e.set(g, f), i++, u.setStatus(`Distributed Progress: ${i} / ${r} jobs`), await l(v), i >= r && (console.log("All jobs complete. Muxing..."), u.setStatus("Muxing..."), await C());
  };
  const C = async () => {
    const f = Array.from(e.keys()).sort((_e, Q) => _e - Q), { Muxer: g, ArrayBufferTarget: v } = await xi(async () => {
      const { Muxer: _e, ArrayBufferTarget: Q } = await Promise.resolve().then(() => Xi);
      return { Muxer: _e, ArrayBufferTarget: Q };
    }, void 0), N = new g({ target: new v(), video: { codec: "V_VP9", width: a.width, height: a.height, frameRate: a.fps } });
    for (const _e of f) {
      const Q = e.get(_e);
      if (Q) for (const me of Q) N.addVideoChunk(new EncodedVideoChunk({ type: me.type, timestamp: me.timestamp, duration: me.duration, data: me.data }), { decoderConfig: me.decoderConfig });
    }
    N.finalize();
    const { buffer: V } = N.target, dt = new Blob([V], { type: "video/webm" }), He = URL.createObjectURL(dt), ge = document.createElement("a");
    ge.href = He, ge.download = `distributed_trace_${Date.now()}.webm`, document.body.appendChild(ge), ge.click(), document.body.removeChild(ge), URL.revokeObjectURL(He), u.setStatus("Finished!");
  };
  x.onSceneReceived = async (f, g) => {
    console.log("Scene received successfully."), u.setRenderConfig(g), Y = g.fileType, g.fileType, W = f, u.sceneSelect.value = "viewer", await Je("viewer", false), g.anim !== void 0 && (u.animSelect.value = g.anim.toString(), y.setAnimation(g.anim));
  }, u.setConnectionState(null);
};
async function rn() {
  try {
    await k.init(), await y.initWasm();
  } catch (t) {
    alert("Init failed: " + t);
    return;
  }
  nn(), en(), Vt(), Je("cornell", false), requestAnimationFrame(rt);
}
rn().catch(console.error);
