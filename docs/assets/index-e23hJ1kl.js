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
  function n(s) {
    const a = {};
    return s.integrity && (a.integrity = s.integrity), s.referrerPolicy && (a.referrerPolicy = s.referrerPolicy), s.crossOrigin === "use-credentials" ? a.credentials = "include" : s.crossOrigin === "anonymous" ? a.credentials = "omit" : a.credentials = "same-origin", a;
  }
  function r(s) {
    if (s.ep) return;
    s.ep = true;
    const a = n(s);
    fetch(s.href, a);
  }
})();
const In = "modulepreload", Un = function(t) {
  return "/webgpu-raytracer/" + t;
}, Gt = {}, Ln = function(e, n, r) {
  let s = Promise.resolve();
  if (n && n.length > 0) {
    let w = function(b) {
      return Promise.all(b.map((k) => Promise.resolve(k).then((V) => ({ status: "fulfilled", value: V }), (V) => ({ status: "rejected", reason: V }))));
    };
    document.getElementsByTagName("link");
    const o = document.querySelector("meta[property=csp-nonce]"), l = (o == null ? void 0 : o.nonce) || (o == null ? void 0 : o.getAttribute("nonce"));
    s = w(n.map((b) => {
      if (b = Un(b), b in Gt) return;
      Gt[b] = true;
      const k = b.endsWith(".css"), V = k ? '[rel="stylesheet"]' : "";
      if (document.querySelector(`link[href="${b}"]${V}`)) return;
      const R = document.createElement("link");
      if (R.rel = k ? "stylesheet" : In, k || (R.as = "script"), R.crossOrigin = "", R.href = b, l && R.setAttribute("nonce", l), document.head.appendChild(R), k) return new Promise((An, Mn) => {
        R.addEventListener("load", An), R.addEventListener("error", () => Mn(new Error(`Unable to preload CSS for ${b}`)));
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
}, Dn = `// =========================================================
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
class Pn {
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
  buildPipeline(e, n) {
    let r = Dn;
    r = r.replace(/const\s+MAX_DEPTH\s*=\s*\d+u;/, `const MAX_DEPTH = ${e}u;`), r = r.replace(/const\s+SPP\s*=\s*\d+u;/, `const SPP = ${n}u;`);
    const s = this.device.createShaderModule({ label: "RayTracing", code: r });
    this.pipeline = this.device.createComputePipeline({ label: "Main Pipeline", layout: "auto", compute: { module: s, entryPoint: "main" } }), this.bindGroupLayout = this.pipeline.getBindGroupLayout(0);
  }
  updateScreenSize(e, n) {
    this.canvas.width = e, this.canvas.height = n, this.renderTarget && this.renderTarget.destroy(), this.renderTarget = this.device.createTexture({ size: [e, n], format: "rgba8unorm", usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC }), this.renderTargetView = this.renderTarget.createView(), this.bufferSize = e * n * 16, this.accumulateBuffer && this.accumulateBuffer.destroy(), this.accumulateBuffer = this.device.createBuffer({ size: this.bufferSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  }
  resetAccumulation() {
    this.accumulateBuffer && this.device.queue.writeBuffer(this.accumulateBuffer, 0, new Float32Array(this.bufferSize / 4));
  }
  async loadTexturesFromWorld(e) {
    const n = e.textureCount;
    if (n === 0) {
      this.createDefaultTexture();
      return;
    }
    console.log(`Loading ${n} textures...`);
    const r = [];
    for (let s = 0; s < n; s++) {
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
    const n = e.getContext("2d");
    return n.fillStyle = "white", n.fillRect(0, 0, 1024, 1024), await createImageBitmap(e);
  }
  ensureBuffer(e, n, r) {
    if (e && e.size >= n) return e;
    e && e.destroy();
    let s = Math.ceil(n * 1.5);
    return s = s + 3 & -4, s = Math.max(s, 4), this.device.createBuffer({ label: r, size: s, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  }
  updateBuffer(e, n) {
    const r = n.byteLength;
    let s = false, a;
    return e === "index" ? ((!this.indexBuffer || this.indexBuffer.size < r) && (s = true), this.indexBuffer = this.ensureBuffer(this.indexBuffer, r, "IndexBuffer"), a = this.indexBuffer) : e === "attr" ? ((!this.attrBuffer || this.attrBuffer.size < r) && (s = true), this.attrBuffer = this.ensureBuffer(this.attrBuffer, r, "AttrBuffer"), a = this.attrBuffer) : ((!this.instanceBuffer || this.instanceBuffer.size < r) && (s = true), this.instanceBuffer = this.ensureBuffer(this.instanceBuffer, r, "InstanceBuffer"), a = this.instanceBuffer), this.device.queue.writeBuffer(a, 0, n, 0, n.length), s;
  }
  updateCombinedGeometry(e, n, r) {
    const s = e.byteLength + n.byteLength + r.byteLength;
    let a = false;
    (!this.geometryBuffer || this.geometryBuffer.size < s) && (a = true);
    const o = e.length / 4;
    this.vertexCount = o, this.geometryBuffer = this.ensureBuffer(this.geometryBuffer, s, "GeometryBuffer"), !(r.length >= o * 2) && o > 0 && console.warn(`UV buffer mismatch: V=${o}, UV=${r.length / 2}. Filling 0.`);
    const w = e.length, b = n.length, k = r.length, V = w + b + k, R = new Float32Array(V);
    return R.set(e, 0), R.set(n, w), R.set(r, w + b), this.device.queue.writeBuffer(this.geometryBuffer, 0, R), a;
  }
  updateCombinedBVH(e, n) {
    const r = e.byteLength, s = n.byteLength, a = r + s;
    let o = false;
    return (!this.nodesBuffer || this.nodesBuffer.size < a) && (o = true), this.nodesBuffer = this.ensureBuffer(this.nodesBuffer, a, "NodesBuffer"), this.device.queue.writeBuffer(this.nodesBuffer, 0, e), this.device.queue.writeBuffer(this.nodesBuffer, r, n), this.blasOffset = e.length / 8, o;
  }
  updateSceneUniforms(e, n) {
    if (!this.sceneUniformBuffer) return;
    this.device.queue.writeBuffer(this.sceneUniformBuffer, 0, e);
    const r = new Uint32Array([n, this.blasOffset, this.vertexCount, 0]);
    this.device.queue.writeBuffer(this.sceneUniformBuffer, 96, r);
  }
  recreateBindGroup() {
    !this.renderTargetView || !this.accumulateBuffer || !this.geometryBuffer || !this.nodesBuffer || !this.sceneUniformBuffer || (this.bindGroup = this.device.createBindGroup({ layout: this.bindGroupLayout, entries: [{ binding: 0, resource: this.renderTargetView }, { binding: 1, resource: { buffer: this.accumulateBuffer } }, { binding: 2, resource: { buffer: this.sceneUniformBuffer } }, { binding: 3, resource: { buffer: this.geometryBuffer } }, { binding: 4, resource: { buffer: this.indexBuffer } }, { binding: 5, resource: { buffer: this.attrBuffer } }, { binding: 6, resource: { buffer: this.nodesBuffer } }, { binding: 7, resource: { buffer: this.instanceBuffer } }, { binding: 8, resource: this.texture.createView({ dimension: "2d-array" }) }, { binding: 9, resource: this.sampler }] }));
  }
  render(e) {
    if (!this.bindGroup) return;
    const n = new Uint32Array([e]);
    this.device.queue.writeBuffer(this.sceneUniformBuffer, 96, n);
    const r = Math.ceil(this.canvas.width / 8), s = Math.ceil(this.canvas.height / 8), a = this.device.createCommandEncoder(), o = a.beginComputePass();
    o.setPipeline(this.pipeline), o.setBindGroup(0, this.bindGroup), o.dispatchWorkgroups(r, s), o.end(), a.copyTextureToTexture({ texture: this.renderTarget }, { texture: this.context.getCurrentTexture() }, { width: this.canvas.width, height: this.canvas.height, depthOrArrayLayers: 1 }), this.device.queue.submit([a.finish()]);
  }
}
let f;
function zn(t) {
  const e = f.__externref_table_alloc();
  return f.__wbindgen_externrefs.set(e, t), e;
}
function Fn(t, e) {
  return t = t >>> 0, ae().subarray(t / 1, t / 1 + e);
}
let q = null;
function jt() {
  return (q === null || q.buffer.detached === true || q.buffer.detached === void 0 && q.buffer !== f.memory.buffer) && (q = new DataView(f.memory.buffer)), q;
}
function $e(t, e) {
  return t = t >>> 0, Nn(t, e);
}
let _e = null;
function ae() {
  return (_e === null || _e.byteLength === 0) && (_e = new Uint8Array(f.memory.buffer)), _e;
}
function $n(t, e) {
  try {
    return t.apply(this, e);
  } catch (n) {
    const r = zn(n);
    f.__wbindgen_exn_store(r);
  }
}
function Kt(t) {
  return t == null;
}
function Yt(t, e) {
  const n = e(t.length * 1, 1) >>> 0;
  return ae().set(t, n / 1), $ = t.length, n;
}
function ut(t, e, n) {
  if (n === void 0) {
    const l = Se.encode(t), w = e(l.length, 1) >>> 0;
    return ae().subarray(w, w + l.length).set(l), $ = l.length, w;
  }
  let r = t.length, s = e(r, 1) >>> 0;
  const a = ae();
  let o = 0;
  for (; o < r; o++) {
    const l = t.charCodeAt(o);
    if (l > 127) break;
    a[s + o] = l;
  }
  if (o !== r) {
    o !== 0 && (t = t.slice(o)), s = n(s, r, r = o + t.length * 3, 1) >>> 0;
    const l = ae().subarray(s + o, s + r), w = Se.encodeInto(t, l);
    o += w.written, s = n(s, r, o, 1) >>> 0;
  }
  return $ = o, s;
}
let He = new TextDecoder("utf-8", { ignoreBOM: true, fatal: true });
He.decode();
const Hn = 2146435072;
let ct = 0;
function Nn(t, e) {
  return ct += e, ct >= Hn && (He = new TextDecoder("utf-8", { ignoreBOM: true, fatal: true }), He.decode(), ct = e), He.decode(ae().subarray(t, t + e));
}
const Se = new TextEncoder();
"encodeInto" in Se || (Se.encodeInto = function(t, e) {
  const n = Se.encode(t);
  return e.set(n), { read: t.length, written: n.length };
});
let $ = 0;
typeof FinalizationRegistry > "u" || new FinalizationRegistry((t) => f.__wbg_renderbuffers_free(t >>> 0, 1));
const Jt = typeof FinalizationRegistry > "u" ? { register: () => {
}, unregister: () => {
} } : new FinalizationRegistry((t) => f.__wbg_world_free(t >>> 0, 1));
class ft {
  __destroy_into_raw() {
    const e = this.__wbg_ptr;
    return this.__wbg_ptr = 0, Jt.unregister(this), e;
  }
  free() {
    const e = this.__destroy_into_raw();
    f.__wbg_world_free(e, 0);
  }
  camera_ptr() {
    return f.world_camera_ptr(this.__wbg_ptr) >>> 0;
  }
  indices_len() {
    return f.world_indices_len(this.__wbg_ptr) >>> 0;
  }
  indices_ptr() {
    return f.world_indices_ptr(this.__wbg_ptr) >>> 0;
  }
  normals_len() {
    return f.world_normals_len(this.__wbg_ptr) >>> 0;
  }
  normals_ptr() {
    return f.world_normals_ptr(this.__wbg_ptr) >>> 0;
  }
  vertices_len() {
    return f.world_vertices_len(this.__wbg_ptr) >>> 0;
  }
  vertices_ptr() {
    return f.world_vertices_ptr(this.__wbg_ptr) >>> 0;
  }
  instances_len() {
    return f.world_instances_len(this.__wbg_ptr) >>> 0;
  }
  instances_ptr() {
    return f.world_instances_ptr(this.__wbg_ptr) >>> 0;
  }
  set_animation(e) {
    f.world_set_animation(this.__wbg_ptr, e);
  }
  update_camera(e, n) {
    f.world_update_camera(this.__wbg_ptr, e, n);
  }
  attributes_len() {
    return f.world_attributes_len(this.__wbg_ptr) >>> 0;
  }
  attributes_ptr() {
    return f.world_attributes_ptr(this.__wbg_ptr) >>> 0;
  }
  get_texture_ptr(e) {
    return f.world_get_texture_ptr(this.__wbg_ptr, e) >>> 0;
  }
  get_texture_size(e) {
    return f.world_get_texture_size(this.__wbg_ptr, e) >>> 0;
  }
  get_texture_count() {
    return f.world_get_texture_count(this.__wbg_ptr) >>> 0;
  }
  get_animation_name(e) {
    let n, r;
    try {
      const s = f.world_get_animation_name(this.__wbg_ptr, e);
      return n = s[0], r = s[1], $e(s[0], s[1]);
    } finally {
      f.__wbindgen_free(n, r, 1);
    }
  }
  load_animation_glb(e) {
    const n = Yt(e, f.__wbindgen_malloc), r = $;
    f.world_load_animation_glb(this.__wbg_ptr, n, r);
  }
  get_animation_count() {
    return f.world_get_animation_count(this.__wbg_ptr) >>> 0;
  }
  constructor(e, n, r) {
    const s = ut(e, f.__wbindgen_malloc, f.__wbindgen_realloc), a = $;
    var o = Kt(n) ? 0 : ut(n, f.__wbindgen_malloc, f.__wbindgen_realloc), l = $, w = Kt(r) ? 0 : Yt(r, f.__wbindgen_malloc), b = $;
    const k = f.world_new(s, a, o, l, w, b);
    return this.__wbg_ptr = k >>> 0, Jt.register(this, this.__wbg_ptr, this), this;
  }
  update(e) {
    f.world_update(this.__wbg_ptr, e);
  }
  uvs_len() {
    return f.world_uvs_len(this.__wbg_ptr) >>> 0;
  }
  uvs_ptr() {
    return f.world_uvs_ptr(this.__wbg_ptr) >>> 0;
  }
  blas_len() {
    return f.world_blas_len(this.__wbg_ptr) >>> 0;
  }
  blas_ptr() {
    return f.world_blas_ptr(this.__wbg_ptr) >>> 0;
  }
  tlas_len() {
    return f.world_tlas_len(this.__wbg_ptr) >>> 0;
  }
  tlas_ptr() {
    return f.world_tlas_ptr(this.__wbg_ptr) >>> 0;
  }
}
Symbol.dispose && (ft.prototype[Symbol.dispose] = ft.prototype.free);
const On = /* @__PURE__ */ new Set(["basic", "cors", "default"]);
async function Vn(t, e) {
  if (typeof Response == "function" && t instanceof Response) {
    if (typeof WebAssembly.instantiateStreaming == "function") try {
      return await WebAssembly.instantiateStreaming(t, e);
    } catch (r) {
      if (t.ok && On.has(t.type) && t.headers.get("Content-Type") !== "application/wasm") console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", r);
      else throw r;
    }
    const n = await t.arrayBuffer();
    return await WebAssembly.instantiate(n, e);
  } else {
    const n = await WebAssembly.instantiate(t, e);
    return n instanceof WebAssembly.Instance ? { instance: n, module: t } : n;
  }
}
function qn() {
  const t = {};
  return t.wbg = {}, t.wbg.__wbg___wbindgen_throw_dd24417ed36fc46e = function(e, n) {
    throw new Error($e(e, n));
  }, t.wbg.__wbg_error_7534b8e9a36f1ab4 = function(e, n) {
    let r, s;
    try {
      r = e, s = n, console.error($e(e, n));
    } finally {
      f.__wbindgen_free(r, s, 1);
    }
  }, t.wbg.__wbg_getRandomValues_1c61fac11405ffdc = function() {
    return $n(function(e, n) {
      globalThis.crypto.getRandomValues(Fn(e, n));
    }, arguments);
  }, t.wbg.__wbg_log_1d990106d99dacb7 = function(e) {
    console.log(e);
  }, t.wbg.__wbg_new_8a6f238a6ece86ea = function() {
    return new Error();
  }, t.wbg.__wbg_stack_0ed75d68575b0f3c = function(e, n) {
    const r = n.stack, s = ut(r, f.__wbindgen_malloc, f.__wbindgen_realloc), a = $;
    jt().setInt32(e + 4, a, true), jt().setInt32(e + 0, s, true);
  }, t.wbg.__wbindgen_cast_2241b6af4c4b2941 = function(e, n) {
    return $e(e, n);
  }, t.wbg.__wbindgen_init_externref_table = function() {
    const e = f.__wbindgen_externrefs, n = e.grow(4);
    e.set(0, void 0), e.set(n + 0, void 0), e.set(n + 1, null), e.set(n + 2, true), e.set(n + 3, false);
  }, t;
}
function Gn(t, e) {
  return f = t.exports, tn.__wbindgen_wasm_module = e, q = null, _e = null, f.__wbindgen_start(), f;
}
async function tn(t) {
  if (f !== void 0) return f;
  typeof t < "u" && (Object.getPrototypeOf(t) === Object.prototype ? { module_or_path: t } = t : console.warn("using deprecated parameters for the initialization function; pass a single object instead")), typeof t > "u" && (t = new URL("/webgpu-raytracer/assets/rust_shader_tools_bg-CC5HVMsp.wasm", import.meta.url));
  const e = qn();
  (typeof t == "string" || typeof Request == "function" && t instanceof Request || typeof URL == "function" && t instanceof URL) && (t = fetch(t));
  const { instance: n, module: r } = await Vn(await t, e);
  return Gn(n, r);
}
class jn {
  constructor() {
    __publicField(this, "world", null);
    __publicField(this, "wasmMemory", null);
  }
  async initWasm() {
    const e = await tn();
    this.wasmMemory = e.memory, console.log("Wasm initialized");
  }
  loadScene(e, n, r) {
    this.world && this.world.free(), this.world = new ft(e, n, r);
  }
  update(e) {
    var _a;
    (_a = this.world) == null ? void 0 : _a.update(e);
  }
  updateCamera(e, n) {
    var _a;
    (_a = this.world) == null ? void 0 : _a.update_camera(e, n);
  }
  loadAnimation(e) {
    var _a;
    (_a = this.world) == null ? void 0 : _a.load_animation_glb(e);
  }
  getAnimationList() {
    if (!this.world) return [];
    const e = this.world.get_animation_count(), n = [];
    for (let r = 0; r < e; r++) n.push(this.world.get_animation_name(r));
    return n;
  }
  setAnimation(e) {
    var _a;
    (_a = this.world) == null ? void 0 : _a.set_animation(e);
  }
  getF32(e, n) {
    return new Float32Array(this.wasmMemory.buffer, e, n);
  }
  getU32(e, n) {
    return new Uint32Array(this.wasmMemory.buffer, e, n);
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
    const n = this.world.get_texture_ptr(e), r = this.world.get_texture_size(e);
    return !n || r === 0 ? null : new Uint8Array(this.wasmMemory.buffer, n, r).slice();
  }
  get hasWorld() {
    return !!this.world;
  }
  printStats() {
    this.world && console.log(`Scene Stats: V=${this.vertices.length / 4}, Tri=${this.indices.length / 3}, BLAS=${this.blas.length / 8}, TLAS=${this.tlas.length / 8}`);
  }
}
const y = { defaultWidth: 720, defaultHeight: 480, defaultDepth: 10, defaultSPP: 1, signalingServerUrl: "ws://localhost:8080", rtcConfig: { iceServers: JSON.parse('[{"urls": "stun:stun.l.google.com:19302"}]') }, ids: { canvas: "gpu-canvas", renderBtn: "render-btn", sceneSelect: "scene-select", resWidth: "res-width", resHeight: "res-height", objFile: "obj-file", maxDepth: "max-depth", sppFrame: "spp-frame", recompileBtn: "recompile-btn", updateInterval: "update-interval", animSelect: "anim-select", recordBtn: "record-btn", recFps: "rec-fps", recDuration: "rec-duration", recSpp: "rec-spp", recBatch: "rec-batch", btnHost: "btn-host", btnWorker: "btn-worker", btnSendScene: "btn-send-scene", statusDiv: "status" } };
class Kn {
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
    this.canvas = this.el(y.ids.canvas), this.btnRender = this.el(y.ids.renderBtn), this.sceneSelect = this.el(y.ids.sceneSelect), this.inputWidth = this.el(y.ids.resWidth), this.inputHeight = this.el(y.ids.resHeight), this.inputFile = this.setupFileInput(), this.inputDepth = this.el(y.ids.maxDepth), this.inputSPP = this.el(y.ids.sppFrame), this.btnRecompile = this.el(y.ids.recompileBtn), this.inputUpdateInterval = this.el(y.ids.updateInterval), this.animSelect = this.el(y.ids.animSelect), this.btnRecord = this.el(y.ids.recordBtn), this.inputRecFps = this.el(y.ids.recFps), this.inputRecDur = this.el(y.ids.recDuration), this.inputRecSpp = this.el(y.ids.recSpp), this.inputRecBatch = this.el(y.ids.recBatch), this.btnHost = this.el(y.ids.btnHost), this.btnWorker = this.el(y.ids.btnWorker), this.btnSendScene = this.el(y.ids.btnSendScene), this.statusDiv = this.el(y.ids.statusDiv), this.statsDiv = this.createStatsDiv(), this.bindEvents();
  }
  el(e) {
    const n = document.getElementById(e);
    if (!n) throw new Error(`Element not found: ${e}`);
    return n;
  }
  setupFileInput() {
    const e = this.el(y.ids.objFile);
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
      return (_a = this.onResolutionChange) == null ? void 0 : _a.call(this, parseInt(this.inputWidth.value) || y.defaultWidth, parseInt(this.inputHeight.value) || y.defaultHeight);
    };
    this.inputWidth.addEventListener("change", e), this.inputHeight.addEventListener("change", e), this.btnRecompile.addEventListener("click", () => {
      var _a;
      return (_a = this.onRecompile) == null ? void 0 : _a.call(this, parseInt(this.inputDepth.value) || 10, parseInt(this.inputSPP.value) || 1);
    }), this.inputFile.addEventListener("change", (n) => {
      var _a, _b;
      const r = (_a = n.target.files) == null ? void 0 : _a[0];
      r && ((_b = this.onFileSelect) == null ? void 0 : _b.call(this, r));
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
    }), this.btnSendScene.addEventListener("click", () => {
      var _a;
      return (_a = this.onSendScene) == null ? void 0 : _a.call(this);
    });
  }
  updateRenderButton(e) {
    this.btnRender.textContent = e ? "Stop Rendering" : "Resume Rendering";
  }
  updateStats(e, n, r) {
    this.statsDiv.textContent = `FPS: ${e} | ${n.toFixed(2)}ms | Frame: ${r}`;
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
  setRecordingState(e, n) {
    e ? (this.btnRecord.disabled = true, this.btnRecord.textContent = n || "Recording...", this.btnRender.textContent = "Resume Rendering") : (this.btnRecord.disabled = false, this.btnRecord.textContent = "\u25CF Rec");
  }
  updateAnimList(e) {
    if (this.animSelect.innerHTML = "", e.length === 0) {
      const n = document.createElement("option");
      n.text = "No Anim", this.animSelect.add(n), this.animSelect.disabled = true;
      return;
    }
    this.animSelect.disabled = false, e.forEach((n, r) => {
      const s = document.createElement("option");
      s.text = `[${r}] ${n}`, s.value = r.toString(), this.animSelect.add(s);
    }), this.animSelect.value = "0";
  }
  getRenderConfig() {
    return { width: parseInt(this.inputWidth.value, 10) || y.defaultWidth, height: parseInt(this.inputHeight.value, 10) || y.defaultHeight, fps: parseInt(this.inputRecFps.value, 10) || 30, duration: parseFloat(this.inputRecDur.value) || 3, spp: parseInt(this.inputRecSpp.value, 10) || 64, batch: parseInt(this.inputRecBatch.value, 10) || 4, anim: parseInt(this.animSelect.value, 10) || 0 };
  }
  setRenderConfig(e) {
    this.inputWidth.value = e.width.toString(), this.inputHeight.value = e.height.toString(), this.inputRecFps.value = e.fps.toString(), this.inputRecDur.value = e.duration.toString(), this.inputRecSpp.value = e.spp.toString(), this.inputRecBatch.value = e.batch.toString();
  }
}
var Tt = (t, e, n) => {
  if (!e.has(t)) throw TypeError("Cannot " + n);
}, i = (t, e, n) => (Tt(t, e, "read from private field"), n ? n.call(t) : e.get(t)), d = (t, e, n) => {
  if (e.has(t)) throw TypeError("Cannot add the same private member more than once");
  e instanceof WeakSet ? e.add(t) : e.set(t, n);
}, g = (t, e, n, r) => (Tt(t, e, "write to private field"), e.set(t, n), n), c = (t, e, n) => (Tt(t, e, "access private method"), n), nn = class {
  constructor(t) {
    this.value = t;
  }
}, Bt = class {
  constructor(t) {
    this.value = t;
  }
}, rn = (t) => t < 256 ? 1 : t < 65536 ? 2 : t < 1 << 24 ? 3 : t < 2 ** 32 ? 4 : t < 2 ** 40 ? 5 : 6, Yn = (t) => {
  if (t < 127) return 1;
  if (t < 16383) return 2;
  if (t < (1 << 21) - 1) return 3;
  if (t < (1 << 28) - 1) return 4;
  if (t < 2 ** 35 - 1) return 5;
  if (t < 2 ** 42 - 1) return 6;
  throw new Error("EBML VINT size not supported " + t);
}, te = (t, e, n) => {
  let r = 0;
  for (let s = e; s < n; s++) {
    let a = Math.floor(s / 8), o = t[a], l = 7 - (s & 7), w = (o & 1 << l) >> l;
    r <<= 1, r |= w;
  }
  return r;
}, Jn = (t, e, n, r) => {
  for (let s = e; s < n; s++) {
    let a = Math.floor(s / 8), o = t[a], l = 7 - (s & 7);
    o &= ~(1 << l), o |= (r & 1 << n - s - 1) >> n - s - 1 << l, t[a] = o;
  }
}, at = class {
}, Et = class extends at {
  constructor() {
    super(...arguments), this.buffer = null;
  }
}, Wt = class extends at {
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
}, sn = class extends at {
  constructor(t, e) {
    if (super(), this.stream = t, this.options = e, !(t instanceof FileSystemWritableFileStream)) throw new TypeError("FileSystemWritableFileStreamTarget requires a FileSystemWritableFileStream instance.");
    if (e !== void 0 && typeof e != "object") throw new TypeError("FileSystemWritableFileStreamTarget's options, when provided, must be an object.");
    if (e && e.chunkSize !== void 0 && (!Number.isInteger(e.chunkSize) || e.chunkSize <= 0)) throw new TypeError("options.chunkSize, when provided, must be a positive integer");
  }
}, H, _, pt, an, gt, on, wt, dn, Ne, _t, mt, ln, cn = class {
  constructor() {
    d(this, pt), d(this, gt), d(this, wt), d(this, Ne), d(this, mt), this.pos = 0, d(this, H, new Uint8Array(8)), d(this, _, new DataView(i(this, H).buffer)), this.offsets = /* @__PURE__ */ new WeakMap(), this.dataOffsets = /* @__PURE__ */ new WeakMap();
  }
  seek(t) {
    this.pos = t;
  }
  writeEBMLVarInt(t, e = Yn(t)) {
    let n = 0;
    switch (e) {
      case 1:
        i(this, _).setUint8(n++, 128 | t);
        break;
      case 2:
        i(this, _).setUint8(n++, 64 | t >> 8), i(this, _).setUint8(n++, t);
        break;
      case 3:
        i(this, _).setUint8(n++, 32 | t >> 16), i(this, _).setUint8(n++, t >> 8), i(this, _).setUint8(n++, t);
        break;
      case 4:
        i(this, _).setUint8(n++, 16 | t >> 24), i(this, _).setUint8(n++, t >> 16), i(this, _).setUint8(n++, t >> 8), i(this, _).setUint8(n++, t);
        break;
      case 5:
        i(this, _).setUint8(n++, 8 | t / 2 ** 32 & 7), i(this, _).setUint8(n++, t >> 24), i(this, _).setUint8(n++, t >> 16), i(this, _).setUint8(n++, t >> 8), i(this, _).setUint8(n++, t);
        break;
      case 6:
        i(this, _).setUint8(n++, 4 | t / 2 ** 40 & 3), i(this, _).setUint8(n++, t / 2 ** 32 | 0), i(this, _).setUint8(n++, t >> 24), i(this, _).setUint8(n++, t >> 16), i(this, _).setUint8(n++, t >> 8), i(this, _).setUint8(n++, t);
        break;
      default:
        throw new Error("Bad EBML VINT size " + e);
    }
    this.write(i(this, H).subarray(0, n));
  }
  writeEBML(t) {
    if (t !== null) if (t instanceof Uint8Array) this.write(t);
    else if (Array.isArray(t)) for (let e of t) this.writeEBML(e);
    else if (this.offsets.set(t, this.pos), c(this, Ne, _t).call(this, t.id), Array.isArray(t.data)) {
      let e = this.pos, n = t.size === -1 ? 1 : t.size ?? 4;
      t.size === -1 ? c(this, pt, an).call(this, 255) : this.seek(this.pos + n);
      let r = this.pos;
      if (this.dataOffsets.set(t, r), this.writeEBML(t.data), t.size !== -1) {
        let s = this.pos - r, a = this.pos;
        this.seek(e), this.writeEBMLVarInt(s, n), this.seek(a);
      }
    } else if (typeof t.data == "number") {
      let e = t.size ?? rn(t.data);
      this.writeEBMLVarInt(e), c(this, Ne, _t).call(this, t.data, e);
    } else typeof t.data == "string" ? (this.writeEBMLVarInt(t.data.length), c(this, mt, ln).call(this, t.data)) : t.data instanceof Uint8Array ? (this.writeEBMLVarInt(t.data.byteLength, t.size), this.write(t.data)) : t.data instanceof nn ? (this.writeEBMLVarInt(4), c(this, gt, on).call(this, t.data.value)) : t.data instanceof Bt && (this.writeEBMLVarInt(8), c(this, wt, dn).call(this, t.data.value));
  }
};
H = /* @__PURE__ */ new WeakMap();
_ = /* @__PURE__ */ new WeakMap();
pt = /* @__PURE__ */ new WeakSet();
an = function(t) {
  i(this, _).setUint8(0, t), this.write(i(this, H).subarray(0, 1));
};
gt = /* @__PURE__ */ new WeakSet();
on = function(t) {
  i(this, _).setFloat32(0, t, false), this.write(i(this, H).subarray(0, 4));
};
wt = /* @__PURE__ */ new WeakSet();
dn = function(t) {
  i(this, _).setFloat64(0, t, false), this.write(i(this, H));
};
Ne = /* @__PURE__ */ new WeakSet();
_t = function(t, e = rn(t)) {
  let n = 0;
  switch (e) {
    case 6:
      i(this, _).setUint8(n++, t / 2 ** 40 | 0);
    case 5:
      i(this, _).setUint8(n++, t / 2 ** 32 | 0);
    case 4:
      i(this, _).setUint8(n++, t >> 24);
    case 3:
      i(this, _).setUint8(n++, t >> 16);
    case 2:
      i(this, _).setUint8(n++, t >> 8);
    case 1:
      i(this, _).setUint8(n++, t);
      break;
    default:
      throw new Error("Bad UINT size " + e);
  }
  this.write(i(this, H).subarray(0, n));
};
mt = /* @__PURE__ */ new WeakSet();
ln = function(t) {
  this.write(new Uint8Array(t.split("").map((e) => e.charCodeAt(0))));
};
var Oe, Q, We, Ve, bt, Xn = class extends cn {
  constructor(t) {
    super(), d(this, Ve), d(this, Oe, void 0), d(this, Q, new ArrayBuffer(2 ** 16)), d(this, We, new Uint8Array(i(this, Q))), g(this, Oe, t);
  }
  write(t) {
    c(this, Ve, bt).call(this, this.pos + t.byteLength), i(this, We).set(t, this.pos), this.pos += t.byteLength;
  }
  finalize() {
    c(this, Ve, bt).call(this, this.pos), i(this, Oe).buffer = i(this, Q).slice(0, this.pos);
  }
};
Oe = /* @__PURE__ */ new WeakMap();
Q = /* @__PURE__ */ new WeakMap();
We = /* @__PURE__ */ new WeakMap();
Ve = /* @__PURE__ */ new WeakSet();
bt = function(t) {
  let e = i(this, Q).byteLength;
  for (; e < t; ) e *= 2;
  if (e === i(this, Q).byteLength) return;
  let n = new ArrayBuffer(e), r = new Uint8Array(n);
  r.set(i(this, We), 0), g(this, Q, n), g(this, We, r);
};
var ne, B, E, G, Pe = class extends cn {
  constructor(t) {
    super(), this.target = t, d(this, ne, false), d(this, B, void 0), d(this, E, void 0), d(this, G, void 0);
  }
  write(t) {
    if (!i(this, ne)) return;
    let e = this.pos;
    if (e < i(this, E)) {
      if (e + t.byteLength <= i(this, E)) return;
      t = t.subarray(i(this, E) - e), e = 0;
    }
    let n = e + t.byteLength - i(this, E), r = i(this, B).byteLength;
    for (; r < n; ) r *= 2;
    if (r !== i(this, B).byteLength) {
      let s = new Uint8Array(r);
      s.set(i(this, B), 0), g(this, B, s);
    }
    i(this, B).set(t, e - i(this, E)), g(this, G, Math.max(i(this, G), e + t.byteLength));
  }
  startTrackingWrites() {
    g(this, ne, true), g(this, B, new Uint8Array(2 ** 10)), g(this, E, this.pos), g(this, G, this.pos);
  }
  getTrackedWrites() {
    if (!i(this, ne)) throw new Error("Can't get tracked writes since nothing was tracked.");
    let e = { data: i(this, B).subarray(0, i(this, G) - i(this, E)), start: i(this, E), end: i(this, G) };
    return g(this, B, void 0), g(this, ne, false), e;
  }
};
ne = /* @__PURE__ */ new WeakMap();
B = /* @__PURE__ */ new WeakMap();
E = /* @__PURE__ */ new WeakMap();
G = /* @__PURE__ */ new WeakMap();
var Qn = 2 ** 24, Zn = 2, j, oe, ke, me, P, C, Je, yt, At, hn, Mt, un, Re, Xe, It = class extends Pe {
  constructor(t, e) {
    var _a, _b;
    super(t), d(this, Je), d(this, At), d(this, Mt), d(this, Re), d(this, j, []), d(this, oe, 0), d(this, ke, void 0), d(this, me, void 0), d(this, P, void 0), d(this, C, []), g(this, ke, e), g(this, me, ((_a = t.options) == null ? void 0 : _a.chunked) ?? false), g(this, P, ((_b = t.options) == null ? void 0 : _b.chunkSize) ?? Qn);
  }
  write(t) {
    super.write(t), i(this, j).push({ data: t.slice(), start: this.pos }), this.pos += t.byteLength;
  }
  flush() {
    var _a, _b;
    if (i(this, j).length === 0) return;
    let t = [], e = [...i(this, j)].sort((n, r) => n.start - r.start);
    t.push({ start: e[0].start, size: e[0].data.byteLength });
    for (let n = 1; n < e.length; n++) {
      let r = t[t.length - 1], s = e[n];
      s.start <= r.start + r.size ? r.size = Math.max(r.size, s.start + s.data.byteLength - r.start) : t.push({ start: s.start, size: s.data.byteLength });
    }
    for (let n of t) {
      n.data = new Uint8Array(n.size);
      for (let r of i(this, j)) n.start <= r.start && r.start < n.start + n.size && n.data.set(r.data, r.start - n.start);
      if (i(this, me)) c(this, Je, yt).call(this, n.data, n.start), c(this, Re, Xe).call(this);
      else {
        if (i(this, ke) && n.start < i(this, oe)) throw new Error("Internal error: Monotonicity violation.");
        (_b = (_a = this.target.options).onData) == null ? void 0 : _b.call(_a, n.data, n.start), g(this, oe, n.start + n.data.byteLength);
      }
    }
    i(this, j).length = 0;
  }
  finalize() {
    i(this, me) && c(this, Re, Xe).call(this, true);
  }
};
j = /* @__PURE__ */ new WeakMap();
oe = /* @__PURE__ */ new WeakMap();
ke = /* @__PURE__ */ new WeakMap();
me = /* @__PURE__ */ new WeakMap();
P = /* @__PURE__ */ new WeakMap();
C = /* @__PURE__ */ new WeakMap();
Je = /* @__PURE__ */ new WeakSet();
yt = function(t, e) {
  let n = i(this, C).findIndex((l) => l.start <= e && e < l.start + i(this, P));
  n === -1 && (n = c(this, Mt, un).call(this, e));
  let r = i(this, C)[n], s = e - r.start, a = t.subarray(0, Math.min(i(this, P) - s, t.byteLength));
  r.data.set(a, s);
  let o = { start: s, end: s + a.byteLength };
  if (c(this, At, hn).call(this, r, o), r.written[0].start === 0 && r.written[0].end === i(this, P) && (r.shouldFlush = true), i(this, C).length > Zn) {
    for (let l = 0; l < i(this, C).length - 1; l++) i(this, C)[l].shouldFlush = true;
    c(this, Re, Xe).call(this);
  }
  a.byteLength < t.byteLength && c(this, Je, yt).call(this, t.subarray(a.byteLength), e + a.byteLength);
};
At = /* @__PURE__ */ new WeakSet();
hn = function(t, e) {
  let n = 0, r = t.written.length - 1, s = -1;
  for (; n <= r; ) {
    let a = Math.floor(n + (r - n + 1) / 2);
    t.written[a].start <= e.start ? (n = a + 1, s = a) : r = a - 1;
  }
  for (t.written.splice(s + 1, 0, e), (s === -1 || t.written[s].end < e.start) && s++; s < t.written.length - 1 && t.written[s].end >= t.written[s + 1].start; ) t.written[s].end = Math.max(t.written[s].end, t.written[s + 1].end), t.written.splice(s + 1, 1);
};
Mt = /* @__PURE__ */ new WeakSet();
un = function(t) {
  let n = { start: Math.floor(t / i(this, P)) * i(this, P), data: new Uint8Array(i(this, P)), written: [], shouldFlush: false };
  return i(this, C).push(n), i(this, C).sort((r, s) => r.start - s.start), i(this, C).indexOf(n);
};
Re = /* @__PURE__ */ new WeakSet();
Xe = function(t = false) {
  var _a, _b;
  for (let e = 0; e < i(this, C).length; e++) {
    let n = i(this, C)[e];
    if (!(!n.shouldFlush && !t)) {
      for (let r of n.written) {
        if (i(this, ke) && n.start + r.start < i(this, oe)) throw new Error("Internal error: Monotonicity violation.");
        (_b = (_a = this.target.options).onData) == null ? void 0 : _b.call(_a, n.data.subarray(r.start, r.end), n.start + r.start), g(this, oe, n.start + r.end);
      }
      i(this, C).splice(e--, 1);
    }
  }
};
var ei = class extends It {
  constructor(t, e) {
    var _a;
    super(new Wt({ onData: (n, r) => t.stream.write({ type: "write", data: n, position: r }), chunked: true, chunkSize: (_a = t.options) == null ? void 0 : _a.chunkSize }), e);
  }
}, pe = 1, Ae = 2, Qe = 3, ti = 1, ni = 2, ii = 17, ri = 2 ** 15, Ce = 2 ** 13, Xt = "https://github.com/Vanilagy/webm-muxer", fn = 6, pn = 5, si = ["strict", "offset", "permissive"], p, h, Me, Ie, D, ge, ie, Z, we, N, de, le, A, ze, ce, z, F, K, xe, Te, he, ue, Ze, Ue, Be, vt, gn, St, wn, Ut, _n, Lt, mn, Dt, bn, Pt, yn, zt, vn, ot, Ft, dt, $t, Ht, Sn, Y, re, J, se, kt, kn, Rt, Rn, be, qe, ye, Ge, Nt, Cn, W, U, fe, Le, Ee, et, Ot, xn, tt, Vt, ve, je, Tn = class {
  constructor(t) {
    d(this, vt), d(this, St), d(this, Ut), d(this, Lt), d(this, Dt), d(this, Pt), d(this, zt), d(this, ot), d(this, dt), d(this, Ht), d(this, Y), d(this, J), d(this, kt), d(this, Rt), d(this, be), d(this, ye), d(this, Nt), d(this, W), d(this, fe), d(this, Ee), d(this, Ot), d(this, tt), d(this, ve), d(this, p, void 0), d(this, h, void 0), d(this, Me, void 0), d(this, Ie, void 0), d(this, D, void 0), d(this, ge, void 0), d(this, ie, void 0), d(this, Z, void 0), d(this, we, void 0), d(this, N, void 0), d(this, de, void 0), d(this, le, void 0), d(this, A, void 0), d(this, ze, void 0), d(this, ce, 0), d(this, z, []), d(this, F, []), d(this, K, []), d(this, xe, void 0), d(this, Te, void 0), d(this, he, -1), d(this, ue, -1), d(this, Ze, -1), d(this, Ue, void 0), d(this, Be, false), c(this, vt, gn).call(this, t), g(this, p, { type: "webm", firstTimestampBehavior: "strict", ...t }), this.target = t.target;
    let e = !!i(this, p).streaming;
    if (t.target instanceof Et) g(this, h, new Xn(t.target));
    else if (t.target instanceof Wt) g(this, h, new It(t.target, e));
    else if (t.target instanceof sn) g(this, h, new ei(t.target, e));
    else throw new Error(`Invalid target: ${t.target}`);
    c(this, St, wn).call(this);
  }
  addVideoChunk(t, e, n) {
    if (!(t instanceof EncodedVideoChunk)) throw new TypeError("addVideoChunk's first argument (chunk) must be of type EncodedVideoChunk.");
    if (e && typeof e != "object") throw new TypeError("addVideoChunk's second argument (meta), when provided, must be an object.");
    if (n !== void 0 && (!Number.isFinite(n) || n < 0)) throw new TypeError("addVideoChunk's third argument (timestamp), when provided, must be a non-negative real number.");
    let r = new Uint8Array(t.byteLength);
    t.copyTo(r), this.addVideoChunkRaw(r, t.type, n ?? t.timestamp, e);
  }
  addVideoChunkRaw(t, e, n, r) {
    if (!(t instanceof Uint8Array)) throw new TypeError("addVideoChunkRaw's first argument (data) must be an instance of Uint8Array.");
    if (e !== "key" && e !== "delta") throw new TypeError("addVideoChunkRaw's second argument (type) must be either 'key' or 'delta'.");
    if (!Number.isFinite(n) || n < 0) throw new TypeError("addVideoChunkRaw's third argument (timestamp) must be a non-negative real number.");
    if (r && typeof r != "object") throw new TypeError("addVideoChunkRaw's fourth argument (meta), when provided, must be an object.");
    if (c(this, ve, je).call(this), !i(this, p).video) throw new Error("No video track declared.");
    i(this, xe) === void 0 && g(this, xe, n), r && c(this, kt, kn).call(this, r);
    let s = c(this, ye, Ge).call(this, t, e, n, pe);
    for (i(this, p).video.codec === "V_VP9" && c(this, Rt, Rn).call(this, s), g(this, he, s.timestamp); i(this, F).length > 0 && i(this, F)[0].timestamp <= s.timestamp; ) {
      let a = i(this, F).shift();
      c(this, W, U).call(this, a, false);
    }
    !i(this, p).audio || s.timestamp <= i(this, ue) ? c(this, W, U).call(this, s, true) : i(this, z).push(s), c(this, be, qe).call(this), c(this, Y, re).call(this);
  }
  addAudioChunk(t, e, n) {
    if (!(t instanceof EncodedAudioChunk)) throw new TypeError("addAudioChunk's first argument (chunk) must be of type EncodedAudioChunk.");
    if (e && typeof e != "object") throw new TypeError("addAudioChunk's second argument (meta), when provided, must be an object.");
    if (n !== void 0 && (!Number.isFinite(n) || n < 0)) throw new TypeError("addAudioChunk's third argument (timestamp), when provided, must be a non-negative real number.");
    let r = new Uint8Array(t.byteLength);
    t.copyTo(r), this.addAudioChunkRaw(r, t.type, n ?? t.timestamp, e);
  }
  addAudioChunkRaw(t, e, n, r) {
    if (!(t instanceof Uint8Array)) throw new TypeError("addAudioChunkRaw's first argument (data) must be an instance of Uint8Array.");
    if (e !== "key" && e !== "delta") throw new TypeError("addAudioChunkRaw's second argument (type) must be either 'key' or 'delta'.");
    if (!Number.isFinite(n) || n < 0) throw new TypeError("addAudioChunkRaw's third argument (timestamp) must be a non-negative real number.");
    if (r && typeof r != "object") throw new TypeError("addAudioChunkRaw's fourth argument (meta), when provided, must be an object.");
    if (c(this, ve, je).call(this), !i(this, p).audio) throw new Error("No audio track declared.");
    i(this, Te) === void 0 && g(this, Te, n), (r == null ? void 0 : r.decoderConfig) && (i(this, p).streaming ? g(this, N, c(this, fe, Le).call(this, r.decoderConfig.description)) : c(this, Ee, et).call(this, i(this, N), r.decoderConfig.description));
    let s = c(this, ye, Ge).call(this, t, e, n, Ae);
    for (g(this, ue, s.timestamp); i(this, z).length > 0 && i(this, z)[0].timestamp <= s.timestamp; ) {
      let a = i(this, z).shift();
      c(this, W, U).call(this, a, true);
    }
    !i(this, p).video || s.timestamp <= i(this, he) ? c(this, W, U).call(this, s, !i(this, p).video) : i(this, F).push(s), c(this, be, qe).call(this), c(this, Y, re).call(this);
  }
  addSubtitleChunk(t, e, n) {
    if (typeof t != "object" || !t) throw new TypeError("addSubtitleChunk's first argument (chunk) must be an object.");
    if (!(t.body instanceof Uint8Array)) throw new TypeError("body must be an instance of Uint8Array.");
    if (!Number.isFinite(t.timestamp) || t.timestamp < 0) throw new TypeError("timestamp must be a non-negative real number.");
    if (!Number.isFinite(t.duration) || t.duration < 0) throw new TypeError("duration must be a non-negative real number.");
    if (t.additions && !(t.additions instanceof Uint8Array)) throw new TypeError("additions, when present, must be an instance of Uint8Array.");
    if (typeof e != "object") throw new TypeError("addSubtitleChunk's second argument (meta) must be an object.");
    if (c(this, ve, je).call(this), !i(this, p).subtitles) throw new Error("No subtitle track declared.");
    (e == null ? void 0 : e.decoderConfig) && (i(this, p).streaming ? g(this, de, c(this, fe, Le).call(this, e.decoderConfig.description)) : c(this, Ee, et).call(this, i(this, de), e.decoderConfig.description));
    let r = c(this, ye, Ge).call(this, t.body, "key", n ?? t.timestamp, Qe, t.duration, t.additions);
    g(this, Ze, r.timestamp), i(this, K).push(r), c(this, be, qe).call(this), c(this, Y, re).call(this);
  }
  finalize() {
    if (i(this, Be)) throw new Error("Cannot finalize a muxer more than once.");
    for (; i(this, z).length > 0; ) c(this, W, U).call(this, i(this, z).shift(), true);
    for (; i(this, F).length > 0; ) c(this, W, U).call(this, i(this, F).shift(), true);
    for (; i(this, K).length > 0 && i(this, K)[0].timestamp <= i(this, ce); ) c(this, W, U).call(this, i(this, K).shift(), false);
    if (i(this, A) && c(this, tt, Vt).call(this), i(this, h).writeEBML(i(this, le)), !i(this, p).streaming) {
      let t = i(this, h).pos, e = i(this, h).pos - i(this, J, se);
      i(this, h).seek(i(this, h).offsets.get(i(this, Me)) + 4), i(this, h).writeEBMLVarInt(e, fn), i(this, ie).data = new Bt(i(this, ce)), i(this, h).seek(i(this, h).offsets.get(i(this, ie))), i(this, h).writeEBML(i(this, ie)), i(this, D).data[0].data[1].data = i(this, h).offsets.get(i(this, le)) - i(this, J, se), i(this, D).data[1].data[1].data = i(this, h).offsets.get(i(this, Ie)) - i(this, J, se), i(this, D).data[2].data[1].data = i(this, h).offsets.get(i(this, ge)) - i(this, J, se), i(this, h).seek(i(this, h).offsets.get(i(this, D))), i(this, h).writeEBML(i(this, D)), i(this, h).seek(t);
    }
    c(this, Y, re).call(this), i(this, h).finalize(), g(this, Be, true);
  }
};
p = /* @__PURE__ */ new WeakMap();
h = /* @__PURE__ */ new WeakMap();
Me = /* @__PURE__ */ new WeakMap();
Ie = /* @__PURE__ */ new WeakMap();
D = /* @__PURE__ */ new WeakMap();
ge = /* @__PURE__ */ new WeakMap();
ie = /* @__PURE__ */ new WeakMap();
Z = /* @__PURE__ */ new WeakMap();
we = /* @__PURE__ */ new WeakMap();
N = /* @__PURE__ */ new WeakMap();
de = /* @__PURE__ */ new WeakMap();
le = /* @__PURE__ */ new WeakMap();
A = /* @__PURE__ */ new WeakMap();
ze = /* @__PURE__ */ new WeakMap();
ce = /* @__PURE__ */ new WeakMap();
z = /* @__PURE__ */ new WeakMap();
F = /* @__PURE__ */ new WeakMap();
K = /* @__PURE__ */ new WeakMap();
xe = /* @__PURE__ */ new WeakMap();
Te = /* @__PURE__ */ new WeakMap();
he = /* @__PURE__ */ new WeakMap();
ue = /* @__PURE__ */ new WeakMap();
Ze = /* @__PURE__ */ new WeakMap();
Ue = /* @__PURE__ */ new WeakMap();
Be = /* @__PURE__ */ new WeakMap();
vt = /* @__PURE__ */ new WeakSet();
gn = function(t) {
  if (typeof t != "object") throw new TypeError("The muxer requires an options object to be passed to its constructor.");
  if (!(t.target instanceof at)) throw new TypeError("The target must be provided and an instance of Target.");
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
  if (t.firstTimestampBehavior && !si.includes(t.firstTimestampBehavior)) throw new TypeError(`Invalid first timestamp behavior: ${t.firstTimestampBehavior}`);
  if (t.streaming !== void 0 && typeof t.streaming != "boolean") throw new TypeError(`Invalid streaming option: ${t.streaming}. Must be a boolean.`);
};
St = /* @__PURE__ */ new WeakSet();
wn = function() {
  i(this, h) instanceof Pe && i(this, h).target.options.onHeader && i(this, h).startTrackingWrites(), c(this, Ut, _n).call(this), i(this, p).streaming || c(this, Pt, yn).call(this), c(this, zt, vn).call(this), c(this, Lt, mn).call(this), c(this, Dt, bn).call(this), i(this, p).streaming || (c(this, ot, Ft).call(this), c(this, dt, $t).call(this)), c(this, Ht, Sn).call(this), c(this, Y, re).call(this);
};
Ut = /* @__PURE__ */ new WeakSet();
_n = function() {
  let t = { id: 440786851, data: [{ id: 17030, data: 1 }, { id: 17143, data: 1 }, { id: 17138, data: 4 }, { id: 17139, data: 8 }, { id: 17026, data: i(this, p).type ?? "webm" }, { id: 17031, data: 2 }, { id: 17029, data: 2 }] };
  i(this, h).writeEBML(t);
};
Lt = /* @__PURE__ */ new WeakSet();
mn = function() {
  g(this, we, { id: 236, size: 4, data: new Uint8Array(Ce) }), g(this, N, { id: 236, size: 4, data: new Uint8Array(Ce) }), g(this, de, { id: 236, size: 4, data: new Uint8Array(Ce) });
};
Dt = /* @__PURE__ */ new WeakSet();
bn = function() {
  g(this, Z, { id: 21936, data: [{ id: 21937, data: 2 }, { id: 21946, data: 2 }, { id: 21947, data: 2 }, { id: 21945, data: 0 }] });
};
Pt = /* @__PURE__ */ new WeakSet();
yn = function() {
  const t = new Uint8Array([28, 83, 187, 107]), e = new Uint8Array([21, 73, 169, 102]), n = new Uint8Array([22, 84, 174, 107]);
  g(this, D, { id: 290298740, data: [{ id: 19899, data: [{ id: 21419, data: t }, { id: 21420, size: 5, data: 0 }] }, { id: 19899, data: [{ id: 21419, data: e }, { id: 21420, size: 5, data: 0 }] }, { id: 19899, data: [{ id: 21419, data: n }, { id: 21420, size: 5, data: 0 }] }] });
};
zt = /* @__PURE__ */ new WeakSet();
vn = function() {
  let t = { id: 17545, data: new Bt(0) };
  g(this, ie, t);
  let e = { id: 357149030, data: [{ id: 2807729, data: 1e6 }, { id: 19840, data: Xt }, { id: 22337, data: Xt }, i(this, p).streaming ? null : t] };
  g(this, Ie, e);
};
ot = /* @__PURE__ */ new WeakSet();
Ft = function() {
  let t = { id: 374648427, data: [] };
  g(this, ge, t), i(this, p).video && t.data.push({ id: 174, data: [{ id: 215, data: pe }, { id: 29637, data: pe }, { id: 131, data: ti }, { id: 134, data: i(this, p).video.codec }, i(this, we), i(this, p).video.frameRate ? { id: 2352003, data: 1e9 / i(this, p).video.frameRate } : null, { id: 224, data: [{ id: 176, data: i(this, p).video.width }, { id: 186, data: i(this, p).video.height }, i(this, p).video.alpha ? { id: 21440, data: 1 } : null, i(this, Z)] }] }), i(this, p).audio && (g(this, N, i(this, p).streaming ? i(this, N) || null : { id: 236, size: 4, data: new Uint8Array(Ce) }), t.data.push({ id: 174, data: [{ id: 215, data: Ae }, { id: 29637, data: Ae }, { id: 131, data: ni }, { id: 134, data: i(this, p).audio.codec }, i(this, N), { id: 225, data: [{ id: 181, data: new nn(i(this, p).audio.sampleRate) }, { id: 159, data: i(this, p).audio.numberOfChannels }, i(this, p).audio.bitDepth ? { id: 25188, data: i(this, p).audio.bitDepth } : null] }] })), i(this, p).subtitles && t.data.push({ id: 174, data: [{ id: 215, data: Qe }, { id: 29637, data: Qe }, { id: 131, data: ii }, { id: 134, data: i(this, p).subtitles.codec }, i(this, de)] });
};
dt = /* @__PURE__ */ new WeakSet();
$t = function() {
  let t = { id: 408125543, size: i(this, p).streaming ? -1 : fn, data: [i(this, p).streaming ? null : i(this, D), i(this, Ie), i(this, ge)] };
  if (g(this, Me, t), i(this, h).writeEBML(t), i(this, h) instanceof Pe && i(this, h).target.options.onHeader) {
    let { data: e, start: n } = i(this, h).getTrackedWrites();
    i(this, h).target.options.onHeader(e, n);
  }
};
Ht = /* @__PURE__ */ new WeakSet();
Sn = function() {
  g(this, le, { id: 475249515, data: [] });
};
Y = /* @__PURE__ */ new WeakSet();
re = function() {
  i(this, h) instanceof It && i(this, h).flush();
};
J = /* @__PURE__ */ new WeakSet();
se = function() {
  return i(this, h).dataOffsets.get(i(this, Me));
};
kt = /* @__PURE__ */ new WeakSet();
kn = function(t) {
  if (t.decoderConfig) {
    if (t.decoderConfig.colorSpace) {
      let e = t.decoderConfig.colorSpace;
      if (g(this, Ue, e), i(this, Z).data = [{ id: 21937, data: { rgb: 1, bt709: 1, bt470bg: 5, smpte170m: 6 }[e.matrix] }, { id: 21946, data: { bt709: 1, smpte170m: 6, "iec61966-2-1": 13 }[e.transfer] }, { id: 21947, data: { bt709: 1, bt470bg: 5, smpte170m: 6 }[e.primaries] }, { id: 21945, data: [1, 2][Number(e.fullRange)] }], !i(this, p).streaming) {
        let n = i(this, h).pos;
        i(this, h).seek(i(this, h).offsets.get(i(this, Z))), i(this, h).writeEBML(i(this, Z)), i(this, h).seek(n);
      }
    }
    t.decoderConfig.description && (i(this, p).streaming ? g(this, we, c(this, fe, Le).call(this, t.decoderConfig.description)) : c(this, Ee, et).call(this, i(this, we), t.decoderConfig.description));
  }
};
Rt = /* @__PURE__ */ new WeakSet();
Rn = function(t) {
  if (t.type !== "key" || !i(this, Ue)) return;
  let e = 0;
  if (te(t.data, 0, 2) !== 2) return;
  e += 2;
  let n = (te(t.data, e + 1, e + 2) << 1) + te(t.data, e + 0, e + 1);
  e += 2, n === 3 && e++;
  let r = te(t.data, e + 0, e + 1);
  if (e++, r) return;
  let s = te(t.data, e + 0, e + 1);
  if (e++, s !== 0) return;
  e += 2;
  let a = te(t.data, e + 0, e + 24);
  if (e += 24, a !== 4817730) return;
  n >= 2 && e++;
  let o = { rgb: 7, bt709: 2, bt470bg: 1, smpte170m: 3 }[i(this, Ue).matrix];
  Jn(t.data, e + 0, e + 3, o);
};
be = /* @__PURE__ */ new WeakSet();
qe = function() {
  let t = Math.min(i(this, p).video ? i(this, he) : 1 / 0, i(this, p).audio ? i(this, ue) : 1 / 0), e = i(this, K);
  for (; e.length > 0 && e[0].timestamp <= t; ) c(this, W, U).call(this, e.shift(), !i(this, p).video && !i(this, p).audio);
};
ye = /* @__PURE__ */ new WeakSet();
Ge = function(t, e, n, r, s, a) {
  let o = c(this, Nt, Cn).call(this, n, r);
  return { data: t, additions: a, type: e, timestamp: o, duration: s, trackNumber: r };
};
Nt = /* @__PURE__ */ new WeakSet();
Cn = function(t, e) {
  let n = e === pe ? i(this, he) : e === Ae ? i(this, ue) : i(this, Ze);
  if (e !== Qe) {
    let r = e === pe ? i(this, xe) : i(this, Te);
    if (i(this, p).firstTimestampBehavior === "strict" && n === -1 && t !== 0) throw new Error(`The first chunk for your media track must have a timestamp of 0 (received ${t}). Non-zero first timestamps are often caused by directly piping frames or audio data from a MediaStreamTrack into the encoder. Their timestamps are typically relative to the age of the document, which is probably what you want.

If you want to offset all timestamps of a track such that the first one is zero, set firstTimestampBehavior: 'offset' in the options.
If you want to allow non-zero first timestamps, set firstTimestampBehavior: 'permissive'.
`);
    i(this, p).firstTimestampBehavior === "offset" && (t -= r);
  }
  if (t < n) throw new Error(`Timestamps must be monotonically increasing (went from ${n} to ${t}).`);
  if (t < 0) throw new Error(`Timestamps must be non-negative (received ${t}).`);
  return t;
};
W = /* @__PURE__ */ new WeakSet();
U = function(t, e) {
  i(this, p).streaming && !i(this, ge) && (c(this, ot, Ft).call(this), c(this, dt, $t).call(this));
  let n = Math.floor(t.timestamp / 1e3), r = n - i(this, ze), s = e && t.type === "key" && r >= 1e3, a = r >= ri;
  if ((!i(this, A) || s || a) && (c(this, Ot, xn).call(this, n), r = 0), r < 0) return;
  let o = new Uint8Array(4), l = new DataView(o.buffer);
  if (l.setUint8(0, 128 | t.trackNumber), l.setInt16(1, r, false), t.duration === void 0 && !t.additions) {
    l.setUint8(3, +(t.type === "key") << 7);
    let w = { id: 163, data: [o, t.data] };
    i(this, h).writeEBML(w);
  } else {
    let w = Math.floor(t.duration / 1e3), b = { id: 160, data: [{ id: 161, data: [o, t.data] }, t.duration !== void 0 ? { id: 155, data: w } : null, t.additions ? { id: 30113, data: t.additions } : null] };
    i(this, h).writeEBML(b);
  }
  g(this, ce, Math.max(i(this, ce), n));
};
fe = /* @__PURE__ */ new WeakSet();
Le = function(t) {
  return { id: 25506, size: 4, data: new Uint8Array(t) };
};
Ee = /* @__PURE__ */ new WeakSet();
et = function(t, e) {
  let n = i(this, h).pos;
  i(this, h).seek(i(this, h).offsets.get(t));
  let r = 6 + e.byteLength, s = Ce - r;
  if (s < 0) {
    let a = e.byteLength + s;
    e instanceof ArrayBuffer ? e = e.slice(0, a) : e = e.buffer.slice(0, a), s = 0;
  }
  t = [c(this, fe, Le).call(this, e), { id: 236, size: 4, data: new Uint8Array(s) }], i(this, h).writeEBML(t), i(this, h).seek(n);
};
Ot = /* @__PURE__ */ new WeakSet();
xn = function(t) {
  i(this, A) && c(this, tt, Vt).call(this), i(this, h) instanceof Pe && i(this, h).target.options.onCluster && i(this, h).startTrackingWrites(), g(this, A, { id: 524531317, size: i(this, p).streaming ? -1 : pn, data: [{ id: 231, data: t }] }), i(this, h).writeEBML(i(this, A)), g(this, ze, t);
  let e = i(this, h).offsets.get(i(this, A)) - i(this, J, se);
  i(this, le).data.push({ id: 187, data: [{ id: 179, data: t }, i(this, p).video ? { id: 183, data: [{ id: 247, data: pe }, { id: 241, data: e }] } : null, i(this, p).audio ? { id: 183, data: [{ id: 247, data: Ae }, { id: 241, data: e }] } : null] });
};
tt = /* @__PURE__ */ new WeakSet();
Vt = function() {
  if (!i(this, p).streaming) {
    let t = i(this, h).pos - i(this, h).dataOffsets.get(i(this, A)), e = i(this, h).pos;
    i(this, h).seek(i(this, h).offsets.get(i(this, A)) + 4), i(this, h).writeEBMLVarInt(t, pn), i(this, h).seek(e);
  }
  if (i(this, h) instanceof Pe && i(this, h).target.options.onCluster) {
    let { data: t, start: e } = i(this, h).getTrackedWrites();
    i(this, h).target.options.onCluster(t, e, i(this, ze));
  }
};
ve = /* @__PURE__ */ new WeakSet();
je = function() {
  if (i(this, Be)) throw new Error("Cannot add new video or audio chunks after the file has been finalized.");
};
new TextEncoder();
const ai = Object.freeze(Object.defineProperty({ __proto__: null, ArrayBufferTarget: Et, FileSystemWritableFileStreamTarget: sn, Muxer: Tn, StreamTarget: Wt }, Symbol.toStringTag, { value: "Module" }));
class oi {
  constructor(e, n, r) {
    __publicField(this, "isRecording", false);
    __publicField(this, "renderer");
    __publicField(this, "worldBridge");
    __publicField(this, "canvas");
    this.renderer = e, this.worldBridge = n, this.canvas = r;
  }
  get recording() {
    return this.isRecording;
  }
  async record(e, n, r) {
    if (this.isRecording) return;
    this.isRecording = true;
    const s = Math.ceil(e.fps * e.duration);
    console.log(`Starting recording: ${s} frames @ ${e.fps}fps (VP9)`);
    const a = new Tn({ target: new Et(), video: { codec: "V_VP9", width: this.canvas.width, height: this.canvas.height, frameRate: e.fps } }), o = new VideoEncoder({ output: (l, w) => a.addVideoChunk(l, w), error: (l) => console.error("VideoEncoder Error:", l) });
    o.configure({ codec: "vp09.00.10.08", width: this.canvas.width, height: this.canvas.height, bitrate: 12e6 });
    try {
      await this.renderAndEncode(s, e, o, n, e.startFrame || 0), await o.flush(), a.finalize();
      const { buffer: l } = a.target, w = new Blob([l], { type: "video/webm" }), b = URL.createObjectURL(w);
      r(b, w);
    } catch (l) {
      throw console.error("Recording failed:", l), l;
    } finally {
      this.isRecording = false;
    }
  }
  async recordChunks(e, n) {
    if (this.isRecording) throw new Error("Already recording");
    this.isRecording = true;
    const r = [], s = Math.ceil(e.fps * e.duration), a = new VideoEncoder({ output: (o, l) => {
      const w = new Uint8Array(o.byteLength);
      o.copyTo(w), r.push({ type: o.type, timestamp: o.timestamp, duration: o.duration, data: w.buffer, decoderConfig: l == null ? void 0 : l.decoderConfig });
    }, error: (o) => console.error("VideoEncoder Error:", o) });
    a.configure({ codec: "vp09.00.10.08", width: this.canvas.width, height: this.canvas.height, bitrate: 12e6 });
    try {
      return await this.renderAndEncode(s, e, a, n, e.startFrame || 0), await a.flush(), r;
    } finally {
      this.isRecording = false;
    }
  }
  async renderAndEncode(e, n, r, s, a = 0) {
    for (let o = 0; o < e; o++) {
      s(o, e), await new Promise((k) => setTimeout(k, 0));
      const l = a + o, w = l / n.fps;
      this.worldBridge.update(w), await this.updateSceneBuffers(), await this.renderFrame(n.spp, n.batch), r.encodeQueueSize > 5 && await r.flush();
      const b = new VideoFrame(this.canvas, { timestamp: l * 1e6 / n.fps, duration: 1e6 / n.fps });
      r.encode(b, { keyFrame: o % n.fps === 0 }), b.close();
    }
  }
  async updateSceneBuffers() {
    let e = false;
    e || (e = this.renderer.updateCombinedBVH(this.worldBridge.tlas, this.worldBridge.blas)), e || (e = this.renderer.updateBuffer("instance", this.worldBridge.instances)), e || (e = this.renderer.updateCombinedGeometry(this.worldBridge.vertices, this.worldBridge.normals, this.worldBridge.uvs)), e || (e = this.renderer.updateBuffer("index", this.worldBridge.indices)), e || (e = this.renderer.updateBuffer("attr", this.worldBridge.attributes)), this.worldBridge.updateCamera(this.canvas.width, this.canvas.height), this.renderer.updateSceneUniforms(this.worldBridge.cameraData, 0), e && this.renderer.recreateBindGroup(), this.renderer.resetAccumulation();
  }
  async renderFrame(e, n) {
    let r = 0;
    for (; r < e; ) {
      const s = Math.min(n, e - r);
      for (let a = 0; a < s; a++) this.renderer.render(r + a);
      r += s, await this.renderer.device.queue.onSubmittedWorkDone(), r < e && await new Promise((a) => setTimeout(a, 0));
    }
  }
}
const di = { iceServers: [{ urls: "stun:stun.l.google.com:19302" }] };
class Qt {
  constructor(e, n) {
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
    __publicField(this, "onWorkerReady", null);
    this.remoteId = e, this.sendSignal = n, this.pc = new RTCPeerConnection(di), this.pc.onicecandidate = (r) => {
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
    const n = await this.pc.createAnswer();
    await this.pc.setLocalDescription(n), this.sendSignal({ type: "answer", sdp: n, targetId: this.remoteId });
  }
  async handleAnswer(e) {
    await this.pc.setRemoteDescription(new RTCSessionDescription(e));
  }
  async handleCandidate(e) {
    await this.pc.addIceCandidate(new RTCIceCandidate(e));
  }
  async sendScene(e, n, r) {
    if (!this.dc || this.dc.readyState !== "open") return;
    let s;
    typeof e == "string" ? s = new TextEncoder().encode(e) : s = new Uint8Array(e);
    const a = { type: "SCENE_INIT", totalBytes: s.byteLength, config: { ...r, fileType: n } };
    this.sendData(a), await this.sendBinaryChunks(s);
  }
  async sendRenderResult(e, n) {
    if (!this.dc || this.dc.readyState !== "open") return;
    let r = 0;
    const s = e.map((l) => {
      const w = l.data.byteLength;
      return r += w, { type: l.type, timestamp: l.timestamp, duration: l.duration, size: w, decoderConfig: l.decoderConfig };
    });
    console.log(`[RTC] Sending Render Result: ${r} bytes, ${e.length} chunks`), this.sendData({ type: "RENDER_RESULT", startFrame: n, totalBytes: r, chunksMeta: s });
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
      const n = e.data;
      if (typeof n == "string") try {
        const r = JSON.parse(n);
        this.handleControlMessage(r);
      } catch {
      }
      else n instanceof ArrayBuffer && this.handleBinaryChunk(n);
    });
  }
  handleControlMessage(e) {
    var _a, _b;
    e.type === "SCENE_INIT" ? (console.log(`[RTC] Receiving Scene: ${e.config.fileType}, ${e.totalBytes} bytes`), this.sceneMeta = { config: e.config, totalBytes: e.totalBytes }, this.receiveBuffer = new Uint8Array(e.totalBytes), this.receivedBytes = 0) : e.type === "SCENE_ACK" ? (console.log(`[RTC] Scene ACK: ${e.receivedBytes} bytes`), this.onAckReceived && this.onAckReceived(e.receivedBytes)) : e.type === "RENDER_REQUEST" ? (console.log(`[RTC] Render Request: Frame ${e.startFrame}, Count ${e.frameCount}`), (_a = this.onRenderRequest) == null ? void 0 : _a.call(this, e.startFrame, e.frameCount, e.config)) : e.type === "RENDER_RESULT" ? (console.log(`[RTC] Receiving Render Result: ${e.totalBytes} bytes`), this.resultMeta = { startFrame: e.startFrame, totalBytes: e.totalBytes, chunksMeta: e.chunksMeta }, this.receiveBuffer = new Uint8Array(e.totalBytes), this.receivedBytes = 0) : e.type === "WORKER_READY" && (console.log("[RTC] Worker Ready Signal Received"), (_b = this.onWorkerReady) == null ? void 0 : _b.call(this));
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
      let r = 0;
      for (const s of this.resultMeta.chunksMeta) {
        const a = this.receiveBuffer.slice(r, r + s.size);
        n.push({ type: s.type, timestamp: s.timestamp, duration: s.duration, data: a.buffer, decoderConfig: s.decoderConfig }), r += s.size;
      }
      (_b = this.onRenderResult) == null ? void 0 : _b.call(this, n, this.resultMeta.startFrame), this.resultMeta = null;
    }
  }
  sendData(e) {
    var _a;
    ((_a = this.dc) == null ? void 0 : _a.readyState) === "open" && this.dc.send(JSON.stringify(e));
  }
  sendAck(e) {
    this.sendData({ type: "SCENE_ACK", receivedBytes: e });
  }
  sendRenderRequest(e, n, r) {
    const s = { type: "RENDER_REQUEST", startFrame: e, frameCount: n, config: r };
    this.sendData(s);
  }
  sendWorkerReady() {
    this.sendData({ type: "WORKER_READY" });
  }
  close() {
    this.dc && (this.dc.close(), this.dc = null), this.pc && this.pc.close(), console.log(`[RTC] Connection closed: ${this.remoteId}`);
  }
}
class li {
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
  }
  connect(e) {
    var _a;
    if (this.ws) return;
    this.myRole = e, (_a = this.onStatusChange) == null ? void 0 : _a.call(this, `Connecting as ${e.toUpperCase()}...`);
    const n = "xWUaLfXQQkHZ9VmF";
    this.ws = new WebSocket(`${y.signalingServerUrl}?token=${n}`), this.ws.onopen = () => {
      var _a2;
      console.log("WS Connected"), (_a2 = this.onStatusChange) == null ? void 0 : _a2.call(this, `Waiting for Peer (${e.toUpperCase()})`), this.sendSignal({ type: e === "host" ? "register_host" : "register_worker" });
    }, this.ws.onmessage = (r) => {
      const s = JSON.parse(r.data);
      this.handleMessage(s);
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
    this.hostClient && await this.hostClient.sendRenderResult(e, n);
  }
  sendSignal(e) {
    var _a;
    ((_a = this.ws) == null ? void 0 : _a.readyState) === WebSocket.OPEN && this.ws.send(JSON.stringify(e));
  }
  async handleMessage(e) {
    this.myRole === "host" ? await this.handleHostMessage(e) : await this.handleWorkerMessage(e);
  }
  async handleHostMessage(e) {
    var _a, _b, _c, _d, _e2;
    switch (e.type) {
      case "worker_joined":
        console.log(`Worker joined: ${e.workerId}`);
        const n = new Qt(e.workerId, (r) => this.sendSignal(r));
        this.workers.set(e.workerId, n), n.onDataChannelOpen = () => {
          var _a2;
          console.log(`[Host] Open for ${e.workerId}`), n.sendData({ type: "HELLO", msg: "Hello from Host!" }), (_a2 = this.onWorkerJoined) == null ? void 0 : _a2.call(this, e.workerId);
        }, n.onAckReceived = (r) => {
          console.log(`Worker ${e.workerId} ACK: ${r}`);
        }, n.onRenderResult = (r, s) => {
          var _a2;
          console.log(`Received Render Result from ${e.workerId}: ${r.length} chunks`), (_a2 = this.onRenderResult) == null ? void 0 : _a2.call(this, r, s, e.workerId);
        }, n.onWorkerReady = () => {
          var _a2;
          (_a2 = this.onWorkerReady) == null ? void 0 : _a2.call(this, e.workerId);
        }, await n.startAsHost();
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
        e.workerId && ((_e2 = this.onWorkerReady) == null ? void 0 : _e2.call(this, e.workerId));
        break;
    }
  }
  async sendWorkerReady() {
    this.hostClient && this.hostClient.sendWorkerReady();
  }
  async handleWorkerMessage(e) {
    var _a, _b, _c;
    switch (e.type) {
      case "offer":
        e.fromId && (this.hostClient = new Qt(e.fromId, (n) => this.sendSignal(n)), await this.hostClient.handleOffer(e.sdp), (_a = this.onStatusChange) == null ? void 0 : _a.call(this, "Connected to Host!"), (_b = this.onHostConnected) == null ? void 0 : _b.call(this), this.hostClient.onDataChannelOpen = () => {
          var _a2, _b2;
          (_a2 = this.hostClient) == null ? void 0 : _a2.sendData({ type: "HELLO", msg: "Hello from Worker!" }), (_b2 = this.onHostHello) == null ? void 0 : _b2.call(this);
        }, this.hostClient.onSceneReceived = (n, r) => {
          var _a2, _b2;
          (_a2 = this.onSceneReceived) == null ? void 0 : _a2.call(this, n, r);
          const s = typeof n == "string" ? n.length : n.byteLength;
          (_b2 = this.hostClient) == null ? void 0 : _b2.sendAck(s);
        }, this.hostClient.onRenderRequest = (n, r, s) => {
          var _a2;
          (_a2 = this.onRenderRequest) == null ? void 0 : _a2.call(this, n, r, s);
        });
        break;
      case "candidate":
        await ((_c = this.hostClient) == null ? void 0 : _c.handleCandidate(e.candidate));
        break;
    }
  }
  async broadcastScene(e, n, r) {
    const s = Array.from(this.workers.values()).map((a) => a.sendScene(e, n, r));
    await Promise.all(s);
  }
  async sendSceneToWorker(e, n, r, s) {
    const a = this.workers.get(e);
    a && await a.sendScene(n, r, s);
  }
  async sendRenderRequest(e, n, r, s) {
    const a = this.workers.get(e);
    a && await a.sendRenderRequest(n, r, s);
  }
}
let T = false, x = null, M = null, L = null, O = [], nt = /* @__PURE__ */ new Map(), Ke = 0, it = 0, ht = 0, ee = null, I = /* @__PURE__ */ new Map(), De = /* @__PURE__ */ new Map(), Ct = false, Ye = null;
const Zt = 20, u = new Kn(), v = new Pn(u.canvas), m = new jn(), rt = new oi(v, m, u.canvas), S = new li();
let X = 0, xt = 0, Fe = 0, en = performance.now();
const ci = () => {
  const t = parseInt(u.inputDepth.value, 10) || y.defaultDepth, e = parseInt(u.inputSPP.value, 10) || y.defaultSPP;
  v.buildPipeline(t, e);
}, qt = () => {
  const { width: t, height: e } = u.getRenderConfig();
  v.updateScreenSize(t, e), m.hasWorld && (m.updateCamera(t, e), v.updateSceneUniforms(m.cameraData, 0)), v.recreateBindGroup(), v.resetAccumulation(), X = 0, xt = 0;
}, st = async (t, e = true) => {
  T = false, console.log(`Loading Scene: ${t}...`);
  let n, r;
  t === "viewer" && x && (M === "obj" ? n = x : M === "glb" && (r = new Uint8Array(x))), m.loadScene(t, n, r), m.printStats(), await v.loadTexturesFromWorld(m), await hi(), qt(), u.updateAnimList(m.getAnimationList()), e && (T = true, u.updateRenderButton(true));
}, hi = async () => {
  v.updateCombinedGeometry(m.vertices, m.normals, m.uvs), v.updateCombinedBVH(m.tlas, m.blas), v.updateBuffer("index", m.indices), v.updateBuffer("attr", m.attributes), v.updateBuffer("instance", m.instances);
}, lt = () => {
  if (rt.recording || (requestAnimationFrame(lt), !T || !m.hasWorld)) return;
  let t = parseInt(u.inputUpdateInterval.value, 10) || 0;
  if (t < 0 && (t = 0), t > 0 && X >= t) {
    m.update(xt / t / 60);
    let n = false;
    n || (n = v.updateCombinedBVH(m.tlas, m.blas)), n || (n = v.updateBuffer("instance", m.instances)), n || (n = v.updateCombinedGeometry(m.vertices, m.normals, m.uvs)), n || (n = v.updateBuffer("index", m.indices)), n || (n = v.updateBuffer("attr", m.attributes)), m.updateCamera(u.canvas.width, u.canvas.height), v.updateSceneUniforms(m.cameraData, 0), n && v.recreateBindGroup(), v.resetAccumulation(), X = 0;
  }
  X++, Fe++, xt++, v.render(X);
  const e = performance.now();
  e - en >= 1e3 && (u.updateStats(Fe, 1e3 / Fe, X), Fe = 0, en = e);
}, Bn = async (t) => {
  if (!x || !M) return;
  const e = u.getRenderConfig();
  t ? (console.log(`Sending scene to specific worker: ${t}`), I.set(t, "loading"), await S.sendSceneToWorker(t, x, M, e)) : (console.log("Broadcasting scene to all workers..."), S.getWorkerIds().forEach((n) => I.set(n, "loading")), await S.broadcastScene(x, M, e));
}, En = async (t) => {
  if (I.get(t) !== "idle") {
    console.log(`Worker ${t} is ${I.get(t)}, skipping assignment.`);
    return;
  }
  if (O.length === 0) return;
  const e = O.shift();
  e && (I.set(t, "busy"), De.set(t, e), console.log(`Assigning Job ${e.start} - ${e.start + e.count} to ${t}`), await S.sendRenderRequest(t, e.start, e.count, { ...ee, fileType: "obj" }));
}, ui = async () => {
  const t = Array.from(nt.keys()).sort((w, b) => w - b), { Muxer: e, ArrayBufferTarget: n } = await Ln(async () => {
    const { Muxer: w, ArrayBufferTarget: b } = await Promise.resolve().then(() => ai);
    return { Muxer: w, ArrayBufferTarget: b };
  }, void 0), r = new e({ target: new n(), video: { codec: "V_VP9", width: ee.width, height: ee.height, frameRate: ee.fps } });
  for (const w of t) {
    const b = nt.get(w);
    if (b) for (const k of b) r.addVideoChunk(new EncodedVideoChunk({ type: k.type, timestamp: k.timestamp, duration: k.duration, data: k.data }), { decoderConfig: k.decoderConfig });
  }
  r.finalize();
  const { buffer: s } = r.target, a = new Blob([s], { type: "video/webm" }), o = URL.createObjectURL(a), l = document.createElement("a");
  l.href = o, l.download = `distributed_trace_${Date.now()}.webm`, document.body.appendChild(l), l.click(), document.body.removeChild(l), URL.revokeObjectURL(o), u.setStatus("Finished!");
}, Wn = async (t, e, n) => {
  console.log(`[Worker] Starting Render: Frames ${t} - ${t + e}`), u.setStatus(`Remote Rendering: ${t}-${t + e}`), T = false;
  const r = { ...n, startFrame: t, duration: e / n.fps };
  try {
    u.setRecordingState(true, `Remote: ${e} f`);
    const s = await rt.recordChunks(r, (a, o) => u.setRecordingState(true, `Remote: ${a}/${o}`));
    console.log("Sending Chunks back to Host..."), u.setRecordingState(true, "Uploading..."), await S.sendRenderResult(s, t), u.setRecordingState(false), u.setStatus("Idle");
  } catch (s) {
    console.error("Remote Recording Failed", s), u.setStatus("Recording Failed");
  } finally {
    T = true, requestAnimationFrame(lt);
  }
}, fi = async () => {
  if (!Ye) return;
  const { start: t, count: e, config: n } = Ye;
  Ye = null, await Wn(t, e, n);
};
S.onStatusChange = (t) => u.setStatus(`Status: ${t}`);
S.onWorkerLeft = (t) => {
  console.log(`Worker Left: ${t}`), u.setStatus(`Worker Left: ${t}`), I.delete(t);
  const e = De.get(t);
  e && (console.warn(`Worker ${t} failed job ${e.start}. Re-queueing.`), O.unshift(e), De.delete(t), u.setStatus(`Re-queued Job ${e.start}`));
};
S.onWorkerReady = (t) => {
  console.log(`Worker ${t} is READY`), u.setStatus(`Worker ${t} Ready!`), I.set(t, "idle"), L === "host" && O.length > 0 && En(t);
};
S.onWorkerJoined = (t) => {
  u.setStatus(`Worker Joined: ${t}`), u.setSendSceneEnabled(true), I.set(t, "idle"), L === "host" && O.length > 0 && Bn(t);
};
S.onRenderRequest = async (t, e, n) => {
  if (console.log(`[Worker] Received Render Request: Frames ${t} - ${t + e}`), Ct) {
    console.log(`[Worker] Scene loading in progress. Queueing Render Request for ${t}`), Ye = { start: t, count: e, config: n };
    return;
  }
  await Wn(t, e, n);
};
S.onRenderResult = async (t, e, n) => {
  console.log(`[Host] Received ${t.length} chunks for ${e} from ${n}`), nt.set(e, t), Ke++, u.setStatus(`Distributed Progress: ${Ke} / ${it} jobs`), I.set(n, "idle"), De.delete(n), await En(n), Ke >= it && (console.log("All jobs complete. Muxing..."), u.setStatus("Muxing..."), await ui());
};
S.onSceneReceived = async (t, e) => {
  console.log("Scene received successfully."), Ct = true, u.setRenderConfig(e), M = e.fileType, e.fileType, x = t, u.sceneSelect.value = "viewer", await st("viewer", false), e.anim !== void 0 && (u.animSelect.value = e.anim.toString(), m.setAnimation(e.anim)), Ct = false, console.log("Scene Loaded. Sending WORKER_READY."), await S.sendWorkerReady(), fi();
};
const pi = () => {
  u.onRenderStart = () => {
    T = true;
  }, u.onRenderStop = () => {
    T = false;
  }, u.onSceneSelect = (t) => st(t, false), u.onResolutionChange = qt, u.onRecompile = (t, e) => {
    T = false, v.buildPipeline(t, e), v.recreateBindGroup(), v.resetAccumulation(), X = 0, T = true;
  }, u.onFileSelect = async (t) => {
    var _a;
    ((_a = t.name.split(".").pop()) == null ? void 0 : _a.toLowerCase()) === "obj" ? (x = await t.text(), M = "obj") : (x = await t.arrayBuffer(), M = "glb"), u.sceneSelect.value = "viewer", st("viewer", false);
  }, u.onAnimSelect = (t) => m.setAnimation(t), u.onRecordStart = async () => {
    if (!rt.recording) if (L === "host") {
      const t = S.getWorkerIds();
      if (ee = u.getRenderConfig(), ht = Math.ceil(ee.fps * ee.duration), !confirm(`Distribute recording? (Workers: ${t.length})
Auto Scene Sync enabled.`)) return;
      O = [], nt.clear(), Ke = 0, De.clear();
      for (let e = 0; e < ht; e += Zt) {
        const n = Math.min(Zt, ht - e);
        O.push({ start: e, count: n });
      }
      it = O.length, t.forEach((e) => I.set(e, "idle")), u.setStatus(`Distributed Progress: 0 / ${it} jobs (Waiting for workers...)`), t.length > 0 ? (u.setStatus("Syncing Scene to Workers..."), await Bn()) : console.log("No workers yet. Waiting...");
    } else {
      T = false, u.setRecordingState(true);
      const t = u.getRenderConfig();
      try {
        await rt.record(t, (e, n) => u.setRecordingState(true, `Rec: ${e}/${n} (${Math.round(e / n * 100)}%)`), (e) => {
          const n = document.createElement("a");
          n.href = e, n.download = `raytrace_${Date.now()}.webm`, n.click(), URL.revokeObjectURL(e);
        });
      } catch {
        alert("Recording failed.");
      } finally {
        u.setRecordingState(false), T = true, u.updateRenderButton(true), requestAnimationFrame(lt);
      }
    }
  }, u.onConnectHost = () => {
    L === "host" ? (S.disconnect(), L = null, u.setConnectionState(null)) : (S.connect("host"), L = "host", u.setConnectionState("host"));
  }, u.onConnectWorker = () => {
    L === "worker" ? (S.disconnect(), L = null, u.setConnectionState(null)) : (S.connect("worker"), L = "worker", u.setConnectionState("worker"));
  }, u.onSendScene = async () => {
    if (!x || !M) {
      alert("No scene loaded!");
      return;
    }
    u.setSendSceneText("Sending..."), u.setSendSceneEnabled(false);
    const t = u.getRenderConfig();
    await S.broadcastScene(x, M, t), u.setSendSceneText("Send Scene"), u.setSendSceneEnabled(true);
  }, u.setConnectionState(null);
};
async function gi() {
  try {
    await v.init(), await m.initWasm();
  } catch (t) {
    alert("Init failed: " + t);
    return;
  }
  pi(), ci(), qt(), st("cornell", false), requestAnimationFrame(lt);
}
gi().catch(console.error);
