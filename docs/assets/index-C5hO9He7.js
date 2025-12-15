var __defProp = Object.defineProperty;
var __defNormalProp = (obj, key, value) => key in obj ? __defProp(obj, key, { enumerable: true, configurable: true, writable: true, value }) : obj[key] = value;
var __publicField = (obj, key, value) => __defNormalProp(obj, typeof key !== "symbol" ? key + "" : key, value);
(function() {
  const t = document.createElement("link").relList;
  if (t && t.supports && t.supports("modulepreload")) return;
  for (const s of document.querySelectorAll('link[rel="modulepreload"]')) n(s);
  new MutationObserver((s) => {
    for (const o of s) if (o.type === "childList") for (const h of o.addedNodes) h.tagName === "LINK" && h.rel === "modulepreload" && n(h);
  }).observe(document, { childList: true, subtree: true });
  function r(s) {
    const o = {};
    return s.integrity && (o.integrity = s.integrity), s.referrerPolicy && (o.referrerPolicy = s.referrerPolicy), s.crossOrigin === "use-credentials" ? o.credentials = "include" : s.crossOrigin === "anonymous" ? o.credentials = "omit" : o.credentials = "same-origin", o;
  }
  function n(s) {
    if (s.ep) return;
    s.ep = true;
    const o = r(s);
    fetch(s.href, o);
  }
})();
const Bi = `// =========================================================
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
    pad1: u32,
    pad2: u32
}

struct Vertex {
    pos: vec4<f32>,
    normal: vec4<f32>,
    uv: vec2<f32>,
    pad: vec2<f32> // Pad to 48 bytes (vec4 + vec4 + vec4)
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

@group(0) @binding(3) var<storage, read> geometry: array<Vertex>; // Merged Verts/Normals/UVs
@group(0) @binding(4) var<storage, read> indices: array<u32>;
@group(0) @binding(5) var<storage, read> attributes: array<TriangleAttributes>;
@group(0) @binding(6) var<storage, read> nodes: array<BVHNode>; // Merged TLAS/BLAS
@group(0) @binding(7) var<storage, read> instances: array<Instance>;

@group(0) @binding(8) var tex: texture_2d_array<f32>;
@group(0) @binding(9) var smp: sampler;

// --- Helpers ---

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
    var stack: array<u32, 64>;
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
                let v0 = geometry[indices[b]].pos.xyz;
                let v1 = geometry[indices[b+1u]].pos.xyz;
                let v2 = geometry[indices[b+2u]].pos.xyz;
                
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
    var stack: array<u32, 64>;
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

        // Retrieve properties from merged geometry buffer
        let v0 = geometry[i0];
        let v1 = geometry[i1];
        let v2 = geometry[i2];

        // Normal Interpolation
        let inv = get_inv_transform(inst);
        let r_local = Ray((inv * vec4(ray.origin, 1.)).xyz, (inv * vec4(ray.direction, 0.)).xyz);

        let e1 = v1.pos.xyz - v0.pos.xyz;
        let e2 = v2.pos.xyz - v0.pos.xyz;
        let h = cross(r_local.direction, e2);
        let a = dot(e1, h);
        let f = 1.0 / a;
        let s = r_local.origin - v0.pos.xyz;
        let u = f * dot(s, h);
        let q = cross(s, e1);
        let v = f * dot(r_local.direction, q);
        let w = 1.0 - u - v;

        let ln = normalize(v0.normal.xyz * w + v1.normal.xyz * u + v2.normal.xyz * v);
        let wn = normalize((vec4(ln, 0.0) * inv).xyz);

        var n = wn;
        let front = dot(ray.direction, n) < 0.0;
        n = select(-n, n, front);

        // Interpolate UV
        let tex_uv = v0.uv * w + v1.uv * u + v2.uv * v;

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
        if (tex_idx > -0.5) {
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
class Si {
  constructor(t) {
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
    this.canvas = t;
  }
  async init() {
    if (!navigator.gpu) throw new Error("WebGPU not supported.");
    const t = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!t) throw new Error("No adapter");
    console.log("Max Storage Buffers Per Shader Stage:", t.limits.maxStorageBuffersPerShaderStage), this.device = await t.requestDevice({ requiredLimits: { maxStorageBuffersPerShaderStage: 10 } }), this.context = this.canvas.getContext("webgpu"), this.context.configure({ device: this.device, format: "rgba8unorm", usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT }), this.sceneUniformBuffer = this.device.createBuffer({ size: 128, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }), this.sampler = this.device.createSampler({ magFilter: "linear", minFilter: "linear", mipmapFilter: "linear", addressModeU: "repeat", addressModeV: "repeat" }), this.createDefaultTexture(), this.texture = this.defaultTexture;
  }
  createDefaultTexture() {
    const t = new Uint8Array([255, 255, 255, 255]);
    this.defaultTexture = this.device.createTexture({ size: [1, 1, 1], format: "rgba8unorm", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST }), this.device.queue.writeTexture({ texture: this.defaultTexture, origin: [0, 0, 0] }, t, { bytesPerRow: 256, rowsPerImage: 1 }, [1, 1]);
  }
  buildPipeline(t, r) {
    let n = Bi;
    n = n.replace(/const\s+MAX_DEPTH\s*=\s*\d+u;/, `const MAX_DEPTH = ${t}u;`), n = n.replace(/const\s+SPP\s*=\s*\d+u;/, `const SPP = ${r}u;`);
    const s = this.device.createShaderModule({ label: "RayTracing", code: n });
    this.pipeline = this.device.createComputePipeline({ label: "Main Pipeline", layout: "auto", compute: { module: s, entryPoint: "main" } }), this.bindGroupLayout = this.pipeline.getBindGroupLayout(0);
  }
  updateScreenSize(t, r) {
    this.canvas.width = t, this.canvas.height = r, this.renderTarget && this.renderTarget.destroy(), this.renderTarget = this.device.createTexture({ size: [t, r], format: "rgba8unorm", usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC }), this.renderTargetView = this.renderTarget.createView(), this.bufferSize = t * r * 16, this.accumulateBuffer && this.accumulateBuffer.destroy(), this.accumulateBuffer = this.device.createBuffer({ size: this.bufferSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  }
  resetAccumulation() {
    this.accumulateBuffer && this.device.queue.writeBuffer(this.accumulateBuffer, 0, new Float32Array(this.bufferSize / 4));
  }
  async loadTexturesFromWorld(t) {
    const r = t.textureCount;
    if (r === 0) {
      this.createDefaultTexture();
      return;
    }
    console.log(`Loading ${r} textures...`);
    const n = [];
    for (let s = 0; s < r; s++) {
      const o = t.getTexture(s);
      if (o) try {
        const h = new Blob([o]), m = await createImageBitmap(h, { resizeWidth: 1024, resizeHeight: 1024 });
        n.push(m);
      } catch (h) {
        console.warn(`Failed tex ${s}`, h), n.push(await this.createFallbackBitmap());
      }
      else n.push(await this.createFallbackBitmap());
    }
    this.texture && this.texture.destroy(), this.texture = this.device.createTexture({ size: [1024, 1024, n.length], format: "rgba8unorm", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT });
    for (let s = 0; s < n.length; s++) this.device.queue.copyExternalImageToTexture({ source: n[s] }, { texture: this.texture, origin: [0, 0, s] }, [1024, 1024]);
  }
  async createFallbackBitmap() {
    const t = document.createElement("canvas");
    t.width = 1024, t.height = 1024;
    const r = t.getContext("2d");
    return r.fillStyle = "white", r.fillRect(0, 0, 1024, 1024), await createImageBitmap(t);
  }
  ensureBuffer(t, r, n) {
    if (t && t.size >= r) return t;
    t && t.destroy();
    let s = Math.ceil(r * 1.5);
    return s = s + 3 & -4, s = Math.max(s, 4), this.device.createBuffer({ label: n, size: s, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  }
  updateBuffer(t, r) {
    const n = r.byteLength;
    let s = false, o;
    return t === "index" ? ((!this.indexBuffer || this.indexBuffer.size < n) && (s = true), this.indexBuffer = this.ensureBuffer(this.indexBuffer, n, "IndexBuffer"), o = this.indexBuffer) : t === "attr" ? ((!this.attrBuffer || this.attrBuffer.size < n) && (s = true), this.attrBuffer = this.ensureBuffer(this.attrBuffer, n, "AttrBuffer"), o = this.attrBuffer) : ((!this.instanceBuffer || this.instanceBuffer.size < n) && (s = true), this.instanceBuffer = this.ensureBuffer(this.instanceBuffer, n, "InstanceBuffer"), o = this.instanceBuffer), this.device.queue.writeBuffer(o, 0, r, 0, r.length), s;
  }
  updateCombinedGeometry(t, r, n) {
    const s = t.length / 4, o = 12, h = s * o * 4;
    let m = false;
    (!this.geometryBuffer || this.geometryBuffer.size < h) && (m = true), this.geometryBuffer = this.ensureBuffer(this.geometryBuffer, h, "GeometryBuffer");
    const g = n.length >= s * 2;
    !g && s > 0 && console.warn(`UV buffer mismatch: V=${s}, UV=${n.length / 2}. Filling 0.`);
    const p = new Float32Array(s * o);
    for (let y = 0; y < s; y++) {
      const u = y * o;
      p[u] = t[y * 4], p[u + 1] = t[y * 4 + 1], p[u + 2] = t[y * 4 + 2], p[u + 3] = 1, p[u + 4] = r[y * 4], p[u + 5] = r[y * 4 + 1], p[u + 6] = r[y * 4 + 2], p[u + 7] = 0, g ? (p[u + 8] = n[y * 2], p[u + 9] = n[y * 2 + 1]) : (p[u + 8] = 0, p[u + 9] = 0), p[u + 10] = 0, p[u + 11] = 0;
    }
    return this.device.queue.writeBuffer(this.geometryBuffer, 0, p), m;
  }
  updateCombinedBVH(t, r) {
    const n = t.byteLength, s = r.byteLength, o = n + s;
    let h = false;
    return (!this.nodesBuffer || this.nodesBuffer.size < o) && (h = true), this.nodesBuffer = this.ensureBuffer(this.nodesBuffer, o, "NodesBuffer"), this.device.queue.writeBuffer(this.nodesBuffer, 0, t), this.device.queue.writeBuffer(this.nodesBuffer, n, r), this.blasOffset = t.length / 8, h;
  }
  updateSceneUniforms(t, r) {
    if (!this.sceneUniformBuffer) return;
    this.device.queue.writeBuffer(this.sceneUniformBuffer, 0, t);
    const n = new Uint32Array([r, this.blasOffset]);
    this.device.queue.writeBuffer(this.sceneUniformBuffer, 96, n);
  }
  recreateBindGroup() {
    !this.renderTargetView || !this.accumulateBuffer || !this.geometryBuffer || !this.nodesBuffer || !this.sceneUniformBuffer || (this.bindGroup = this.device.createBindGroup({ layout: this.bindGroupLayout, entries: [{ binding: 0, resource: this.renderTargetView }, { binding: 1, resource: { buffer: this.accumulateBuffer } }, { binding: 2, resource: { buffer: this.sceneUniformBuffer } }, { binding: 3, resource: { buffer: this.geometryBuffer } }, { binding: 4, resource: { buffer: this.indexBuffer } }, { binding: 5, resource: { buffer: this.attrBuffer } }, { binding: 6, resource: { buffer: this.nodesBuffer } }, { binding: 7, resource: { buffer: this.instanceBuffer } }, { binding: 8, resource: this.texture.createView({ dimension: "2d-array" }) }, { binding: 9, resource: this.sampler }] }));
  }
  render(t) {
    if (!this.bindGroup) return;
    const r = new Uint32Array([t]);
    this.device.queue.writeBuffer(this.sceneUniformBuffer, 96, r);
    const n = Math.ceil(this.canvas.width / 8), s = Math.ceil(this.canvas.height / 8), o = this.device.createCommandEncoder(), h = o.beginComputePass();
    h.setPipeline(this.pipeline), h.setBindGroup(0, this.bindGroup), h.dispatchWorkgroups(n, s), h.end(), o.copyTextureToTexture({ texture: this.renderTarget }, { texture: this.context.getCurrentTexture() }, { width: this.canvas.width, height: this.canvas.height, depthOrArrayLayers: 1 }), this.device.queue.submit([o.finish()]);
  }
}
let c;
function Ei(e) {
  const t = c.__externref_table_alloc();
  return c.__wbindgen_externrefs.set(t, e), t;
}
function Ci(e, t) {
  return e = e >>> 0, ne().subarray(e / 1, e / 1 + t);
}
let H = null;
function Gt() {
  return (H === null || H.buffer.detached === true || H.buffer.detached === void 0 && H.buffer !== c.memory.buffer) && (H = new DataView(c.memory.buffer)), H;
}
function Ge(e, t) {
  return e = e >>> 0, Ui(e, t);
}
let be = null;
function ne() {
  return (be === null || be.byteLength === 0) && (be = new Uint8Array(c.memory.buffer)), be;
}
function Mi(e, t) {
  try {
    return e.apply(this, t);
  } catch (r) {
    const n = Ei(r);
    c.__wbindgen_exn_store(n);
  }
}
function Ht(e) {
  return e == null;
}
function $t(e, t) {
  const r = t(e.length * 1, 1) >>> 0;
  return ne().set(e, r / 1), F = e.length, r;
}
function ct(e, t, r) {
  if (r === void 0) {
    const m = Te.encode(e), g = t(m.length, 1) >>> 0;
    return ne().subarray(g, g + m.length).set(m), F = m.length, g;
  }
  let n = e.length, s = t(n, 1) >>> 0;
  const o = ne();
  let h = 0;
  for (; h < n; h++) {
    const m = e.charCodeAt(h);
    if (m > 127) break;
    o[s + h] = m;
  }
  if (h !== n) {
    h !== 0 && (e = e.slice(h)), s = r(s, n, n = h + e.length * 3, 1) >>> 0;
    const m = ne().subarray(s + h, s + n), g = Te.encodeInto(e, m);
    h += g.written, s = r(s, n, h, 1) >>> 0;
  }
  return F = h, s;
}
let He = new TextDecoder("utf-8", { ignoreBOM: true, fatal: true });
He.decode();
const Ii = 2146435072;
let lt = 0;
function Ui(e, t) {
  return lt += t, lt >= Ii && (He = new TextDecoder("utf-8", { ignoreBOM: true, fatal: true }), He.decode(), lt = t), He.decode(ne().subarray(e, e + t));
}
const Te = new TextEncoder();
"encodeInto" in Te || (Te.encodeInto = function(e, t) {
  const r = Te.encode(e);
  return t.set(r), { read: e.length, written: r.length };
});
let F = 0;
typeof FinalizationRegistry > "u" || new FinalizationRegistry((e) => c.__wbg_renderbuffers_free(e >>> 0, 1));
const qt = typeof FinalizationRegistry > "u" ? { register: () => {
}, unregister: () => {
} } : new FinalizationRegistry((e) => c.__wbg_world_free(e >>> 0, 1));
class ht {
  __destroy_into_raw() {
    const t = this.__wbg_ptr;
    return this.__wbg_ptr = 0, qt.unregister(this), t;
  }
  free() {
    const t = this.__destroy_into_raw();
    c.__wbg_world_free(t, 0);
  }
  camera_ptr() {
    return c.world_camera_ptr(this.__wbg_ptr) >>> 0;
  }
  indices_len() {
    return c.world_indices_len(this.__wbg_ptr) >>> 0;
  }
  indices_ptr() {
    return c.world_indices_ptr(this.__wbg_ptr) >>> 0;
  }
  normals_len() {
    return c.world_normals_len(this.__wbg_ptr) >>> 0;
  }
  normals_ptr() {
    return c.world_normals_ptr(this.__wbg_ptr) >>> 0;
  }
  vertices_len() {
    return c.world_vertices_len(this.__wbg_ptr) >>> 0;
  }
  vertices_ptr() {
    return c.world_vertices_ptr(this.__wbg_ptr) >>> 0;
  }
  instances_len() {
    return c.world_instances_len(this.__wbg_ptr) >>> 0;
  }
  instances_ptr() {
    return c.world_instances_ptr(this.__wbg_ptr) >>> 0;
  }
  set_animation(t) {
    c.world_set_animation(this.__wbg_ptr, t);
  }
  update_camera(t, r) {
    c.world_update_camera(this.__wbg_ptr, t, r);
  }
  attributes_len() {
    return c.world_attributes_len(this.__wbg_ptr) >>> 0;
  }
  attributes_ptr() {
    return c.world_attributes_ptr(this.__wbg_ptr) >>> 0;
  }
  get_texture_ptr(t) {
    return c.world_get_texture_ptr(this.__wbg_ptr, t) >>> 0;
  }
  get_texture_size(t) {
    return c.world_get_texture_size(this.__wbg_ptr, t) >>> 0;
  }
  get_texture_count() {
    return c.world_get_texture_count(this.__wbg_ptr) >>> 0;
  }
  get_animation_name(t) {
    let r, n;
    try {
      const s = c.world_get_animation_name(this.__wbg_ptr, t);
      return r = s[0], n = s[1], Ge(s[0], s[1]);
    } finally {
      c.__wbindgen_free(r, n, 1);
    }
  }
  load_animation_glb(t) {
    const r = $t(t, c.__wbindgen_malloc), n = F;
    c.world_load_animation_glb(this.__wbg_ptr, r, n);
  }
  get_animation_count() {
    return c.world_get_animation_count(this.__wbg_ptr) >>> 0;
  }
  constructor(t, r, n) {
    const s = ct(t, c.__wbindgen_malloc, c.__wbindgen_realloc), o = F;
    var h = Ht(r) ? 0 : ct(r, c.__wbindgen_malloc, c.__wbindgen_realloc), m = F, g = Ht(n) ? 0 : $t(n, c.__wbindgen_malloc), p = F;
    const y = c.world_new(s, o, h, m, g, p);
    return this.__wbg_ptr = y >>> 0, qt.register(this, this.__wbg_ptr, this), this;
  }
  update(t) {
    c.world_update(this.__wbg_ptr, t);
  }
  uvs_len() {
    return c.world_uvs_len(this.__wbg_ptr) >>> 0;
  }
  uvs_ptr() {
    return c.world_uvs_ptr(this.__wbg_ptr) >>> 0;
  }
  blas_len() {
    return c.world_blas_len(this.__wbg_ptr) >>> 0;
  }
  blas_ptr() {
    return c.world_blas_ptr(this.__wbg_ptr) >>> 0;
  }
  tlas_len() {
    return c.world_tlas_len(this.__wbg_ptr) >>> 0;
  }
  tlas_ptr() {
    return c.world_tlas_ptr(this.__wbg_ptr) >>> 0;
  }
}
Symbol.dispose && (ht.prototype[Symbol.dispose] = ht.prototype.free);
const Ai = /* @__PURE__ */ new Set(["basic", "cors", "default"]);
async function zi(e, t) {
  if (typeof Response == "function" && e instanceof Response) {
    if (typeof WebAssembly.instantiateStreaming == "function") try {
      return await WebAssembly.instantiateStreaming(e, t);
    } catch (n) {
      if (e.ok && Ai.has(e.type) && e.headers.get("Content-Type") !== "application/wasm") console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", n);
      else throw n;
    }
    const r = await e.arrayBuffer();
    return await WebAssembly.instantiate(r, t);
  } else {
    const r = await WebAssembly.instantiate(e, t);
    return r instanceof WebAssembly.Instance ? { instance: r, module: e } : r;
  }
}
function Wi() {
  const e = {};
  return e.wbg = {}, e.wbg.__wbg___wbindgen_throw_dd24417ed36fc46e = function(t, r) {
    throw new Error(Ge(t, r));
  }, e.wbg.__wbg_error_7534b8e9a36f1ab4 = function(t, r) {
    let n, s;
    try {
      n = t, s = r, console.error(Ge(t, r));
    } finally {
      c.__wbindgen_free(n, s, 1);
    }
  }, e.wbg.__wbg_getRandomValues_1c61fac11405ffdc = function() {
    return Mi(function(t, r) {
      globalThis.crypto.getRandomValues(Ci(t, r));
    }, arguments);
  }, e.wbg.__wbg_log_1d990106d99dacb7 = function(t) {
    console.log(t);
  }, e.wbg.__wbg_new_8a6f238a6ece86ea = function() {
    return new Error();
  }, e.wbg.__wbg_stack_0ed75d68575b0f3c = function(t, r) {
    const n = r.stack, s = ct(n, c.__wbindgen_malloc, c.__wbindgen_realloc), o = F;
    Gt().setInt32(t + 4, o, true), Gt().setInt32(t + 0, s, true);
  }, e.wbg.__wbindgen_cast_2241b6af4c4b2941 = function(t, r) {
    return Ge(t, r);
  }, e.wbg.__wbindgen_init_externref_table = function() {
    const t = c.__wbindgen_externrefs, r = t.grow(4);
    t.set(0, void 0), t.set(r + 0, void 0), t.set(r + 1, null), t.set(r + 2, true), t.set(r + 3, false);
  }, e;
}
function Li(e, t) {
  return c = e.exports, Qt.__wbindgen_wasm_module = t, H = null, be = null, c.__wbindgen_start(), c;
}
async function Qt(e) {
  if (c !== void 0) return c;
  typeof e < "u" && (Object.getPrototypeOf(e) === Object.prototype ? { module_or_path: e } = e : console.warn("using deprecated parameters for the initialization function; pass a single object instead")), typeof e > "u" && (e = new URL("/webgpu-raytracer/assets/rust_shader_tools_bg-CC5HVMsp.wasm", import.meta.url));
  const t = Wi();
  (typeof e == "string" || typeof Request == "function" && e instanceof Request || typeof URL == "function" && e instanceof URL) && (e = fetch(e));
  const { instance: r, module: n } = await zi(await e, t);
  return Li(r, n);
}
class Ri {
  constructor() {
    __publicField(this, "world", null);
    __publicField(this, "wasmMemory", null);
  }
  async initWasm() {
    const t = await Qt();
    this.wasmMemory = t.memory, console.log("Wasm initialized");
  }
  loadScene(t, r, n) {
    this.world && this.world.free(), this.world = new ht(t, r, n);
  }
  update(t) {
    var _a;
    (_a = this.world) == null ? void 0 : _a.update(t);
  }
  updateCamera(t, r) {
    var _a;
    (_a = this.world) == null ? void 0 : _a.update_camera(t, r);
  }
  loadAnimation(t) {
    var _a;
    (_a = this.world) == null ? void 0 : _a.load_animation_glb(t);
  }
  getAnimationList() {
    if (!this.world) return [];
    const t = this.world.get_animation_count(), r = [];
    for (let n = 0; n < t; n++) r.push(this.world.get_animation_name(n));
    return r;
  }
  setAnimation(t) {
    var _a;
    (_a = this.world) == null ? void 0 : _a.set_animation(t);
  }
  getF32(t, r) {
    return new Float32Array(this.wasmMemory.buffer, t, r);
  }
  getU32(t, r) {
    return new Uint32Array(this.wasmMemory.buffer, t, r);
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
  getTexture(t) {
    if (!this.world) return null;
    const r = this.world.get_texture_ptr(t), n = this.world.get_texture_size(t);
    return !r || n === 0 ? null : new Uint8Array(this.wasmMemory.buffer, r, n).slice();
  }
  get hasWorld() {
    return !!this.world;
  }
  printStats() {
    this.world && console.log(`Scene Stats: V=${this.vertices.length / 4}, Tri=${this.indices.length / 3}, BLAS=${this.blas.length / 8}, TLAS=${this.tlas.length / 8}`);
  }
}
var Tt = (e, t, r) => {
  if (!t.has(e)) throw TypeError("Cannot " + r);
}, i = (e, t, r) => (Tt(e, t, "read from private field"), r ? r.call(e) : t.get(e)), a = (e, t, r) => {
  if (t.has(e)) throw TypeError("Cannot add the same private member more than once");
  t instanceof WeakSet ? t.add(e) : t.set(e, r);
}, _ = (e, t, r, n) => (Tt(e, t, "write to private field"), t.set(e, r), r), d = (e, t, r) => (Tt(e, t, "access private method"), r), Zt = class {
  constructor(e) {
    this.value = e;
  }
}, Bt = class {
  constructor(e) {
    this.value = e;
  }
}, Jt = (e) => e < 256 ? 1 : e < 65536 ? 2 : e < 1 << 24 ? 3 : e < 2 ** 32 ? 4 : e < 2 ** 40 ? 5 : 6, Pi = (e) => {
  if (e < 127) return 1;
  if (e < 16383) return 2;
  if (e < (1 << 21) - 1) return 3;
  if (e < (1 << 28) - 1) return 4;
  if (e < 2 ** 35 - 1) return 5;
  if (e < 2 ** 42 - 1) return 6;
  throw new Error("EBML VINT size not supported " + e);
}, J = (e, t, r) => {
  let n = 0;
  for (let s = t; s < r; s++) {
    let o = Math.floor(s / 8), h = e[o], m = 7 - (s & 7), g = (h & 1 << m) >> m;
    n <<= 1, n |= g;
  }
  return n;
}, Fi = (e, t, r, n) => {
  for (let s = t; s < r; s++) {
    let o = Math.floor(s / 8), h = e[o], m = 7 - (s & 7);
    h &= ~(1 << m), h |= (n & 1 << r - s - 1) >> r - s - 1 << m, e[o] = h;
  }
}, rt = class {
}, ei = class extends rt {
  constructor() {
    super(...arguments), this.buffer = null;
  }
}, ti = class extends rt {
  constructor(e) {
    if (super(), this.options = e, typeof e != "object") throw new TypeError("StreamTarget requires an options object to be passed to its constructor.");
    if (e.onData) {
      if (typeof e.onData != "function") throw new TypeError("options.onData, when provided, must be a function.");
      if (e.onData.length < 2) throw new TypeError("options.onData, when provided, must be a function that takes in at least two arguments (data and position). Ignoring the position argument, which specifies the byte offset at which the data is to be written, can lead to broken outputs.");
    }
    if (e.onHeader && typeof e.onHeader != "function") throw new TypeError("options.onHeader, when provided, must be a function.");
    if (e.onCluster && typeof e.onCluster != "function") throw new TypeError("options.onCluster, when provided, must be a function.");
    if (e.chunked !== void 0 && typeof e.chunked != "boolean") throw new TypeError("options.chunked, when provided, must be a boolean.");
    if (e.chunkSize !== void 0 && (!Number.isInteger(e.chunkSize) || e.chunkSize < 1024)) throw new TypeError("options.chunkSize, when provided, must be an integer and not smaller than 1024.");
  }
}, Vi = class extends rt {
  constructor(e, t) {
    if (super(), this.stream = e, this.options = t, !(e instanceof FileSystemWritableFileStream)) throw new TypeError("FileSystemWritableFileStreamTarget requires a FileSystemWritableFileStream instance.");
    if (t !== void 0 && typeof t != "object") throw new TypeError("FileSystemWritableFileStreamTarget's options, when provided, must be an object.");
    if (t && t.chunkSize !== void 0 && (!Number.isInteger(t.chunkSize) || t.chunkSize <= 0)) throw new TypeError("options.chunkSize, when provided, must be a positive integer");
  }
}, V, w, ut, ii, ft, ri, _t, ni, $e, mt, wt, si, ai = class {
  constructor() {
    a(this, ut), a(this, ft), a(this, _t), a(this, $e), a(this, wt), this.pos = 0, a(this, V, new Uint8Array(8)), a(this, w, new DataView(i(this, V).buffer)), this.offsets = /* @__PURE__ */ new WeakMap(), this.dataOffsets = /* @__PURE__ */ new WeakMap();
  }
  seek(e) {
    this.pos = e;
  }
  writeEBMLVarInt(e, t = Pi(e)) {
    let r = 0;
    switch (t) {
      case 1:
        i(this, w).setUint8(r++, 128 | e);
        break;
      case 2:
        i(this, w).setUint8(r++, 64 | e >> 8), i(this, w).setUint8(r++, e);
        break;
      case 3:
        i(this, w).setUint8(r++, 32 | e >> 16), i(this, w).setUint8(r++, e >> 8), i(this, w).setUint8(r++, e);
        break;
      case 4:
        i(this, w).setUint8(r++, 16 | e >> 24), i(this, w).setUint8(r++, e >> 16), i(this, w).setUint8(r++, e >> 8), i(this, w).setUint8(r++, e);
        break;
      case 5:
        i(this, w).setUint8(r++, 8 | e / 2 ** 32 & 7), i(this, w).setUint8(r++, e >> 24), i(this, w).setUint8(r++, e >> 16), i(this, w).setUint8(r++, e >> 8), i(this, w).setUint8(r++, e);
        break;
      case 6:
        i(this, w).setUint8(r++, 4 | e / 2 ** 40 & 3), i(this, w).setUint8(r++, e / 2 ** 32 | 0), i(this, w).setUint8(r++, e >> 24), i(this, w).setUint8(r++, e >> 16), i(this, w).setUint8(r++, e >> 8), i(this, w).setUint8(r++, e);
        break;
      default:
        throw new Error("Bad EBML VINT size " + t);
    }
    this.write(i(this, V).subarray(0, r));
  }
  writeEBML(e) {
    if (e !== null) if (e instanceof Uint8Array) this.write(e);
    else if (Array.isArray(e)) for (let t of e) this.writeEBML(t);
    else if (this.offsets.set(e, this.pos), d(this, $e, mt).call(this, e.id), Array.isArray(e.data)) {
      let t = this.pos, r = e.size === -1 ? 1 : e.size ?? 4;
      e.size === -1 ? d(this, ut, ii).call(this, 255) : this.seek(this.pos + r);
      let n = this.pos;
      if (this.dataOffsets.set(e, n), this.writeEBML(e.data), e.size !== -1) {
        let s = this.pos - n, o = this.pos;
        this.seek(t), this.writeEBMLVarInt(s, r), this.seek(o);
      }
    } else if (typeof e.data == "number") {
      let t = e.size ?? Jt(e.data);
      this.writeEBMLVarInt(t), d(this, $e, mt).call(this, e.data, t);
    } else typeof e.data == "string" ? (this.writeEBMLVarInt(e.data.length), d(this, wt, si).call(this, e.data)) : e.data instanceof Uint8Array ? (this.writeEBMLVarInt(e.data.byteLength, e.size), this.write(e.data)) : e.data instanceof Zt ? (this.writeEBMLVarInt(4), d(this, ft, ri).call(this, e.data.value)) : e.data instanceof Bt && (this.writeEBMLVarInt(8), d(this, _t, ni).call(this, e.data.value));
  }
};
V = /* @__PURE__ */ new WeakMap();
w = /* @__PURE__ */ new WeakMap();
ut = /* @__PURE__ */ new WeakSet();
ii = function(e) {
  i(this, w).setUint8(0, e), this.write(i(this, V).subarray(0, 1));
};
ft = /* @__PURE__ */ new WeakSet();
ri = function(e) {
  i(this, w).setFloat32(0, e, false), this.write(i(this, V).subarray(0, 4));
};
_t = /* @__PURE__ */ new WeakSet();
ni = function(e) {
  i(this, w).setFloat64(0, e, false), this.write(i(this, V));
};
$e = /* @__PURE__ */ new WeakSet();
mt = function(e, t = Jt(e)) {
  let r = 0;
  switch (t) {
    case 6:
      i(this, w).setUint8(r++, e / 2 ** 40 | 0);
    case 5:
      i(this, w).setUint8(r++, e / 2 ** 32 | 0);
    case 4:
      i(this, w).setUint8(r++, e >> 24);
    case 3:
      i(this, w).setUint8(r++, e >> 16);
    case 2:
      i(this, w).setUint8(r++, e >> 8);
    case 1:
      i(this, w).setUint8(r++, e);
      break;
    default:
      throw new Error("Bad UINT size " + t);
  }
  this.write(i(this, V).subarray(0, r));
};
wt = /* @__PURE__ */ new WeakSet();
si = function(e) {
  this.write(new Uint8Array(e.split("").map((t) => t.charCodeAt(0))));
};
var qe, K, Ae, je, pt, Di = class extends ai {
  constructor(e) {
    super(), a(this, je), a(this, qe, void 0), a(this, K, new ArrayBuffer(2 ** 16)), a(this, Ae, new Uint8Array(i(this, K))), _(this, qe, e);
  }
  write(e) {
    d(this, je, pt).call(this, this.pos + e.byteLength), i(this, Ae).set(e, this.pos), this.pos += e.byteLength;
  }
  finalize() {
    d(this, je, pt).call(this, this.pos), i(this, qe).buffer = i(this, K).slice(0, this.pos);
  }
};
qe = /* @__PURE__ */ new WeakMap();
K = /* @__PURE__ */ new WeakMap();
Ae = /* @__PURE__ */ new WeakMap();
je = /* @__PURE__ */ new WeakSet();
pt = function(e) {
  let t = i(this, K).byteLength;
  for (; t < e; ) t *= 2;
  if (t === i(this, K).byteLength) return;
  let r = new ArrayBuffer(t), n = new Uint8Array(r);
  n.set(i(this, Ae), 0), _(this, K, r), _(this, Ae, n);
};
var ee, S, E, $, Fe = class extends ai {
  constructor(e) {
    super(), this.target = e, a(this, ee, false), a(this, S, void 0), a(this, E, void 0), a(this, $, void 0);
  }
  write(e) {
    if (!i(this, ee)) return;
    let t = this.pos;
    if (t < i(this, E)) {
      if (t + e.byteLength <= i(this, E)) return;
      e = e.subarray(i(this, E) - t), t = 0;
    }
    let r = t + e.byteLength - i(this, E), n = i(this, S).byteLength;
    for (; n < r; ) n *= 2;
    if (n !== i(this, S).byteLength) {
      let s = new Uint8Array(n);
      s.set(i(this, S), 0), _(this, S, s);
    }
    i(this, S).set(e, t - i(this, E)), _(this, $, Math.max(i(this, $), t + e.byteLength));
  }
  startTrackingWrites() {
    _(this, ee, true), _(this, S, new Uint8Array(2 ** 10)), _(this, E, this.pos), _(this, $, this.pos);
  }
  getTrackedWrites() {
    if (!i(this, ee)) throw new Error("Can't get tracked writes since nothing was tracked.");
    let t = { data: i(this, S).subarray(0, i(this, $) - i(this, E)), start: i(this, E), end: i(this, $) };
    return _(this, S, void 0), _(this, ee, false), t;
  }
};
ee = /* @__PURE__ */ new WeakMap();
S = /* @__PURE__ */ new WeakMap();
E = /* @__PURE__ */ new WeakMap();
$ = /* @__PURE__ */ new WeakMap();
var Ni = 2 ** 24, Oi = 2, q, se, Be, ve, z, x, Qe, gt, St, oi, Et, di, Se, Ze, Ct = class extends Fe {
  constructor(e, t) {
    var _a, _b;
    super(e), a(this, Qe), a(this, St), a(this, Et), a(this, Se), a(this, q, []), a(this, se, 0), a(this, Be, void 0), a(this, ve, void 0), a(this, z, void 0), a(this, x, []), _(this, Be, t), _(this, ve, ((_a = e.options) == null ? void 0 : _a.chunked) ?? false), _(this, z, ((_b = e.options) == null ? void 0 : _b.chunkSize) ?? Ni);
  }
  write(e) {
    super.write(e), i(this, q).push({ data: e.slice(), start: this.pos }), this.pos += e.byteLength;
  }
  flush() {
    var _a, _b;
    if (i(this, q).length === 0) return;
    let e = [], t = [...i(this, q)].sort((r, n) => r.start - n.start);
    e.push({ start: t[0].start, size: t[0].data.byteLength });
    for (let r = 1; r < t.length; r++) {
      let n = e[e.length - 1], s = t[r];
      s.start <= n.start + n.size ? n.size = Math.max(n.size, s.start + s.data.byteLength - n.start) : e.push({ start: s.start, size: s.data.byteLength });
    }
    for (let r of e) {
      r.data = new Uint8Array(r.size);
      for (let n of i(this, q)) r.start <= n.start && n.start < r.start + r.size && r.data.set(n.data, n.start - r.start);
      if (i(this, ve)) d(this, Qe, gt).call(this, r.data, r.start), d(this, Se, Ze).call(this);
      else {
        if (i(this, Be) && r.start < i(this, se)) throw new Error("Internal error: Monotonicity violation.");
        (_b = (_a = this.target.options).onData) == null ? void 0 : _b.call(_a, r.data, r.start), _(this, se, r.start + r.data.byteLength);
      }
    }
    i(this, q).length = 0;
  }
  finalize() {
    i(this, ve) && d(this, Se, Ze).call(this, true);
  }
};
q = /* @__PURE__ */ new WeakMap();
se = /* @__PURE__ */ new WeakMap();
Be = /* @__PURE__ */ new WeakMap();
ve = /* @__PURE__ */ new WeakMap();
z = /* @__PURE__ */ new WeakMap();
x = /* @__PURE__ */ new WeakMap();
Qe = /* @__PURE__ */ new WeakSet();
gt = function(e, t) {
  let r = i(this, x).findIndex((m) => m.start <= t && t < m.start + i(this, z));
  r === -1 && (r = d(this, Et, di).call(this, t));
  let n = i(this, x)[r], s = t - n.start, o = e.subarray(0, Math.min(i(this, z) - s, e.byteLength));
  n.data.set(o, s);
  let h = { start: s, end: s + o.byteLength };
  if (d(this, St, oi).call(this, n, h), n.written[0].start === 0 && n.written[0].end === i(this, z) && (n.shouldFlush = true), i(this, x).length > Oi) {
    for (let m = 0; m < i(this, x).length - 1; m++) i(this, x)[m].shouldFlush = true;
    d(this, Se, Ze).call(this);
  }
  o.byteLength < e.byteLength && d(this, Qe, gt).call(this, e.subarray(o.byteLength), t + o.byteLength);
};
St = /* @__PURE__ */ new WeakSet();
oi = function(e, t) {
  let r = 0, n = e.written.length - 1, s = -1;
  for (; r <= n; ) {
    let o = Math.floor(r + (n - r + 1) / 2);
    e.written[o].start <= t.start ? (r = o + 1, s = o) : n = o - 1;
  }
  for (e.written.splice(s + 1, 0, t), (s === -1 || e.written[s].end < t.start) && s++; s < e.written.length - 1 && e.written[s].end >= e.written[s + 1].start; ) e.written[s].end = Math.max(e.written[s].end, e.written[s + 1].end), e.written.splice(s + 1, 1);
};
Et = /* @__PURE__ */ new WeakSet();
di = function(e) {
  let r = { start: Math.floor(e / i(this, z)) * i(this, z), data: new Uint8Array(i(this, z)), written: [], shouldFlush: false };
  return i(this, x).push(r), i(this, x).sort((n, s) => n.start - s.start), i(this, x).indexOf(r);
};
Se = /* @__PURE__ */ new WeakSet();
Ze = function(e = false) {
  var _a, _b;
  for (let t = 0; t < i(this, x).length; t++) {
    let r = i(this, x)[t];
    if (!(!r.shouldFlush && !e)) {
      for (let n of r.written) {
        if (i(this, Be) && r.start + n.start < i(this, se)) throw new Error("Internal error: Monotonicity violation.");
        (_b = (_a = this.target.options).onData) == null ? void 0 : _b.call(_a, r.data.subarray(n.start, n.end), r.start + n.start), _(this, se, r.start + n.end);
      }
      i(this, x).splice(t--, 1);
    }
  }
};
var Gi = class extends Ct {
  constructor(e, t) {
    var _a;
    super(new ti({ onData: (r, n) => e.stream.write({ type: "write", data: r, position: n }), chunked: true, chunkSize: (_a = e.options) == null ? void 0 : _a.chunkSize }), t);
  }
}, ue = 1, ze = 2, Je = 3, Hi = 1, $i = 2, qi = 17, ji = 2 ** 15, Ee = 2 ** 13, jt = "https://github.com/Vanilagy/webm-muxer", li = 6, ci = 5, Yi = ["strict", "offset", "permissive"], f, l, We, Le, A, fe, te, Q, _e, D, ae, oe, M, Ve, de, R, P, j, Ce, Me, le, ce, et, Re, Ie, bt, hi, vt, ui, Mt, fi, It, _i, Ut, mi, At, wi, zt, pi, nt, Wt, st, Lt, Rt, gi, Y, ie, X, re, yt, bi, xt, vi, ye, Ye, xe, Xe, Pt, yi, C, U, he, Pe, Ue, tt, Ft, xi, it, Vt, ke, Ke, Xi = class {
  constructor(e) {
    a(this, bt), a(this, vt), a(this, Mt), a(this, It), a(this, Ut), a(this, At), a(this, zt), a(this, nt), a(this, st), a(this, Rt), a(this, Y), a(this, X), a(this, yt), a(this, xt), a(this, ye), a(this, xe), a(this, Pt), a(this, C), a(this, he), a(this, Ue), a(this, Ft), a(this, it), a(this, ke), a(this, f, void 0), a(this, l, void 0), a(this, We, void 0), a(this, Le, void 0), a(this, A, void 0), a(this, fe, void 0), a(this, te, void 0), a(this, Q, void 0), a(this, _e, void 0), a(this, D, void 0), a(this, ae, void 0), a(this, oe, void 0), a(this, M, void 0), a(this, Ve, void 0), a(this, de, 0), a(this, R, []), a(this, P, []), a(this, j, []), a(this, Ce, void 0), a(this, Me, void 0), a(this, le, -1), a(this, ce, -1), a(this, et, -1), a(this, Re, void 0), a(this, Ie, false), d(this, bt, hi).call(this, e), _(this, f, { type: "webm", firstTimestampBehavior: "strict", ...e }), this.target = e.target;
    let t = !!i(this, f).streaming;
    if (e.target instanceof ei) _(this, l, new Di(e.target));
    else if (e.target instanceof ti) _(this, l, new Ct(e.target, t));
    else if (e.target instanceof Vi) _(this, l, new Gi(e.target, t));
    else throw new Error(`Invalid target: ${e.target}`);
    d(this, vt, ui).call(this);
  }
  addVideoChunk(e, t, r) {
    if (!(e instanceof EncodedVideoChunk)) throw new TypeError("addVideoChunk's first argument (chunk) must be of type EncodedVideoChunk.");
    if (t && typeof t != "object") throw new TypeError("addVideoChunk's second argument (meta), when provided, must be an object.");
    if (r !== void 0 && (!Number.isFinite(r) || r < 0)) throw new TypeError("addVideoChunk's third argument (timestamp), when provided, must be a non-negative real number.");
    let n = new Uint8Array(e.byteLength);
    e.copyTo(n), this.addVideoChunkRaw(n, e.type, r ?? e.timestamp, t);
  }
  addVideoChunkRaw(e, t, r, n) {
    if (!(e instanceof Uint8Array)) throw new TypeError("addVideoChunkRaw's first argument (data) must be an instance of Uint8Array.");
    if (t !== "key" && t !== "delta") throw new TypeError("addVideoChunkRaw's second argument (type) must be either 'key' or 'delta'.");
    if (!Number.isFinite(r) || r < 0) throw new TypeError("addVideoChunkRaw's third argument (timestamp) must be a non-negative real number.");
    if (n && typeof n != "object") throw new TypeError("addVideoChunkRaw's fourth argument (meta), when provided, must be an object.");
    if (d(this, ke, Ke).call(this), !i(this, f).video) throw new Error("No video track declared.");
    i(this, Ce) === void 0 && _(this, Ce, r), n && d(this, yt, bi).call(this, n);
    let s = d(this, xe, Xe).call(this, e, t, r, ue);
    for (i(this, f).video.codec === "V_VP9" && d(this, xt, vi).call(this, s), _(this, le, s.timestamp); i(this, P).length > 0 && i(this, P)[0].timestamp <= s.timestamp; ) {
      let o = i(this, P).shift();
      d(this, C, U).call(this, o, false);
    }
    !i(this, f).audio || s.timestamp <= i(this, ce) ? d(this, C, U).call(this, s, true) : i(this, R).push(s), d(this, ye, Ye).call(this), d(this, Y, ie).call(this);
  }
  addAudioChunk(e, t, r) {
    if (!(e instanceof EncodedAudioChunk)) throw new TypeError("addAudioChunk's first argument (chunk) must be of type EncodedAudioChunk.");
    if (t && typeof t != "object") throw new TypeError("addAudioChunk's second argument (meta), when provided, must be an object.");
    if (r !== void 0 && (!Number.isFinite(r) || r < 0)) throw new TypeError("addAudioChunk's third argument (timestamp), when provided, must be a non-negative real number.");
    let n = new Uint8Array(e.byteLength);
    e.copyTo(n), this.addAudioChunkRaw(n, e.type, r ?? e.timestamp, t);
  }
  addAudioChunkRaw(e, t, r, n) {
    if (!(e instanceof Uint8Array)) throw new TypeError("addAudioChunkRaw's first argument (data) must be an instance of Uint8Array.");
    if (t !== "key" && t !== "delta") throw new TypeError("addAudioChunkRaw's second argument (type) must be either 'key' or 'delta'.");
    if (!Number.isFinite(r) || r < 0) throw new TypeError("addAudioChunkRaw's third argument (timestamp) must be a non-negative real number.");
    if (n && typeof n != "object") throw new TypeError("addAudioChunkRaw's fourth argument (meta), when provided, must be an object.");
    if (d(this, ke, Ke).call(this), !i(this, f).audio) throw new Error("No audio track declared.");
    i(this, Me) === void 0 && _(this, Me, r), (n == null ? void 0 : n.decoderConfig) && (i(this, f).streaming ? _(this, D, d(this, he, Pe).call(this, n.decoderConfig.description)) : d(this, Ue, tt).call(this, i(this, D), n.decoderConfig.description));
    let s = d(this, xe, Xe).call(this, e, t, r, ze);
    for (_(this, ce, s.timestamp); i(this, R).length > 0 && i(this, R)[0].timestamp <= s.timestamp; ) {
      let o = i(this, R).shift();
      d(this, C, U).call(this, o, true);
    }
    !i(this, f).video || s.timestamp <= i(this, le) ? d(this, C, U).call(this, s, !i(this, f).video) : i(this, P).push(s), d(this, ye, Ye).call(this), d(this, Y, ie).call(this);
  }
  addSubtitleChunk(e, t, r) {
    if (typeof e != "object" || !e) throw new TypeError("addSubtitleChunk's first argument (chunk) must be an object.");
    if (!(e.body instanceof Uint8Array)) throw new TypeError("body must be an instance of Uint8Array.");
    if (!Number.isFinite(e.timestamp) || e.timestamp < 0) throw new TypeError("timestamp must be a non-negative real number.");
    if (!Number.isFinite(e.duration) || e.duration < 0) throw new TypeError("duration must be a non-negative real number.");
    if (e.additions && !(e.additions instanceof Uint8Array)) throw new TypeError("additions, when present, must be an instance of Uint8Array.");
    if (typeof t != "object") throw new TypeError("addSubtitleChunk's second argument (meta) must be an object.");
    if (d(this, ke, Ke).call(this), !i(this, f).subtitles) throw new Error("No subtitle track declared.");
    (t == null ? void 0 : t.decoderConfig) && (i(this, f).streaming ? _(this, ae, d(this, he, Pe).call(this, t.decoderConfig.description)) : d(this, Ue, tt).call(this, i(this, ae), t.decoderConfig.description));
    let n = d(this, xe, Xe).call(this, e.body, "key", r ?? e.timestamp, Je, e.duration, e.additions);
    _(this, et, n.timestamp), i(this, j).push(n), d(this, ye, Ye).call(this), d(this, Y, ie).call(this);
  }
  finalize() {
    if (i(this, Ie)) throw new Error("Cannot finalize a muxer more than once.");
    for (; i(this, R).length > 0; ) d(this, C, U).call(this, i(this, R).shift(), true);
    for (; i(this, P).length > 0; ) d(this, C, U).call(this, i(this, P).shift(), true);
    for (; i(this, j).length > 0 && i(this, j)[0].timestamp <= i(this, de); ) d(this, C, U).call(this, i(this, j).shift(), false);
    if (i(this, M) && d(this, it, Vt).call(this), i(this, l).writeEBML(i(this, oe)), !i(this, f).streaming) {
      let e = i(this, l).pos, t = i(this, l).pos - i(this, X, re);
      i(this, l).seek(i(this, l).offsets.get(i(this, We)) + 4), i(this, l).writeEBMLVarInt(t, li), i(this, te).data = new Bt(i(this, de)), i(this, l).seek(i(this, l).offsets.get(i(this, te))), i(this, l).writeEBML(i(this, te)), i(this, A).data[0].data[1].data = i(this, l).offsets.get(i(this, oe)) - i(this, X, re), i(this, A).data[1].data[1].data = i(this, l).offsets.get(i(this, Le)) - i(this, X, re), i(this, A).data[2].data[1].data = i(this, l).offsets.get(i(this, fe)) - i(this, X, re), i(this, l).seek(i(this, l).offsets.get(i(this, A))), i(this, l).writeEBML(i(this, A)), i(this, l).seek(e);
    }
    d(this, Y, ie).call(this), i(this, l).finalize(), _(this, Ie, true);
  }
};
f = /* @__PURE__ */ new WeakMap();
l = /* @__PURE__ */ new WeakMap();
We = /* @__PURE__ */ new WeakMap();
Le = /* @__PURE__ */ new WeakMap();
A = /* @__PURE__ */ new WeakMap();
fe = /* @__PURE__ */ new WeakMap();
te = /* @__PURE__ */ new WeakMap();
Q = /* @__PURE__ */ new WeakMap();
_e = /* @__PURE__ */ new WeakMap();
D = /* @__PURE__ */ new WeakMap();
ae = /* @__PURE__ */ new WeakMap();
oe = /* @__PURE__ */ new WeakMap();
M = /* @__PURE__ */ new WeakMap();
Ve = /* @__PURE__ */ new WeakMap();
de = /* @__PURE__ */ new WeakMap();
R = /* @__PURE__ */ new WeakMap();
P = /* @__PURE__ */ new WeakMap();
j = /* @__PURE__ */ new WeakMap();
Ce = /* @__PURE__ */ new WeakMap();
Me = /* @__PURE__ */ new WeakMap();
le = /* @__PURE__ */ new WeakMap();
ce = /* @__PURE__ */ new WeakMap();
et = /* @__PURE__ */ new WeakMap();
Re = /* @__PURE__ */ new WeakMap();
Ie = /* @__PURE__ */ new WeakMap();
bt = /* @__PURE__ */ new WeakSet();
hi = function(e) {
  if (typeof e != "object") throw new TypeError("The muxer requires an options object to be passed to its constructor.");
  if (!(e.target instanceof rt)) throw new TypeError("The target must be provided and an instance of Target.");
  if (e.video) {
    if (typeof e.video.codec != "string") throw new TypeError(`Invalid video codec: ${e.video.codec}. Must be a string.`);
    if (!Number.isInteger(e.video.width) || e.video.width <= 0) throw new TypeError(`Invalid video width: ${e.video.width}. Must be a positive integer.`);
    if (!Number.isInteger(e.video.height) || e.video.height <= 0) throw new TypeError(`Invalid video height: ${e.video.height}. Must be a positive integer.`);
    if (e.video.frameRate !== void 0 && (!Number.isFinite(e.video.frameRate) || e.video.frameRate <= 0)) throw new TypeError(`Invalid video frame rate: ${e.video.frameRate}. Must be a positive number.`);
    if (e.video.alpha !== void 0 && typeof e.video.alpha != "boolean") throw new TypeError(`Invalid video alpha: ${e.video.alpha}. Must be a boolean.`);
  }
  if (e.audio) {
    if (typeof e.audio.codec != "string") throw new TypeError(`Invalid audio codec: ${e.audio.codec}. Must be a string.`);
    if (!Number.isInteger(e.audio.numberOfChannels) || e.audio.numberOfChannels <= 0) throw new TypeError(`Invalid number of audio channels: ${e.audio.numberOfChannels}. Must be a positive integer.`);
    if (!Number.isInteger(e.audio.sampleRate) || e.audio.sampleRate <= 0) throw new TypeError(`Invalid audio sample rate: ${e.audio.sampleRate}. Must be a positive integer.`);
    if (e.audio.bitDepth !== void 0 && (!Number.isInteger(e.audio.bitDepth) || e.audio.bitDepth <= 0)) throw new TypeError(`Invalid audio bit depth: ${e.audio.bitDepth}. Must be a positive integer.`);
  }
  if (e.subtitles && typeof e.subtitles.codec != "string") throw new TypeError(`Invalid subtitles codec: ${e.subtitles.codec}. Must be a string.`);
  if (e.type !== void 0 && !["webm", "matroska"].includes(e.type)) throw new TypeError(`Invalid type: ${e.type}. Must be 'webm' or 'matroska'.`);
  if (e.firstTimestampBehavior && !Yi.includes(e.firstTimestampBehavior)) throw new TypeError(`Invalid first timestamp behavior: ${e.firstTimestampBehavior}`);
  if (e.streaming !== void 0 && typeof e.streaming != "boolean") throw new TypeError(`Invalid streaming option: ${e.streaming}. Must be a boolean.`);
};
vt = /* @__PURE__ */ new WeakSet();
ui = function() {
  i(this, l) instanceof Fe && i(this, l).target.options.onHeader && i(this, l).startTrackingWrites(), d(this, Mt, fi).call(this), i(this, f).streaming || d(this, At, wi).call(this), d(this, zt, pi).call(this), d(this, It, _i).call(this), d(this, Ut, mi).call(this), i(this, f).streaming || (d(this, nt, Wt).call(this), d(this, st, Lt).call(this)), d(this, Rt, gi).call(this), d(this, Y, ie).call(this);
};
Mt = /* @__PURE__ */ new WeakSet();
fi = function() {
  let e = { id: 440786851, data: [{ id: 17030, data: 1 }, { id: 17143, data: 1 }, { id: 17138, data: 4 }, { id: 17139, data: 8 }, { id: 17026, data: i(this, f).type ?? "webm" }, { id: 17031, data: 2 }, { id: 17029, data: 2 }] };
  i(this, l).writeEBML(e);
};
It = /* @__PURE__ */ new WeakSet();
_i = function() {
  _(this, _e, { id: 236, size: 4, data: new Uint8Array(Ee) }), _(this, D, { id: 236, size: 4, data: new Uint8Array(Ee) }), _(this, ae, { id: 236, size: 4, data: new Uint8Array(Ee) });
};
Ut = /* @__PURE__ */ new WeakSet();
mi = function() {
  _(this, Q, { id: 21936, data: [{ id: 21937, data: 2 }, { id: 21946, data: 2 }, { id: 21947, data: 2 }, { id: 21945, data: 0 }] });
};
At = /* @__PURE__ */ new WeakSet();
wi = function() {
  const e = new Uint8Array([28, 83, 187, 107]), t = new Uint8Array([21, 73, 169, 102]), r = new Uint8Array([22, 84, 174, 107]);
  _(this, A, { id: 290298740, data: [{ id: 19899, data: [{ id: 21419, data: e }, { id: 21420, size: 5, data: 0 }] }, { id: 19899, data: [{ id: 21419, data: t }, { id: 21420, size: 5, data: 0 }] }, { id: 19899, data: [{ id: 21419, data: r }, { id: 21420, size: 5, data: 0 }] }] });
};
zt = /* @__PURE__ */ new WeakSet();
pi = function() {
  let e = { id: 17545, data: new Bt(0) };
  _(this, te, e);
  let t = { id: 357149030, data: [{ id: 2807729, data: 1e6 }, { id: 19840, data: jt }, { id: 22337, data: jt }, i(this, f).streaming ? null : e] };
  _(this, Le, t);
};
nt = /* @__PURE__ */ new WeakSet();
Wt = function() {
  let e = { id: 374648427, data: [] };
  _(this, fe, e), i(this, f).video && e.data.push({ id: 174, data: [{ id: 215, data: ue }, { id: 29637, data: ue }, { id: 131, data: Hi }, { id: 134, data: i(this, f).video.codec }, i(this, _e), i(this, f).video.frameRate ? { id: 2352003, data: 1e9 / i(this, f).video.frameRate } : null, { id: 224, data: [{ id: 176, data: i(this, f).video.width }, { id: 186, data: i(this, f).video.height }, i(this, f).video.alpha ? { id: 21440, data: 1 } : null, i(this, Q)] }] }), i(this, f).audio && (_(this, D, i(this, f).streaming ? i(this, D) || null : { id: 236, size: 4, data: new Uint8Array(Ee) }), e.data.push({ id: 174, data: [{ id: 215, data: ze }, { id: 29637, data: ze }, { id: 131, data: $i }, { id: 134, data: i(this, f).audio.codec }, i(this, D), { id: 225, data: [{ id: 181, data: new Zt(i(this, f).audio.sampleRate) }, { id: 159, data: i(this, f).audio.numberOfChannels }, i(this, f).audio.bitDepth ? { id: 25188, data: i(this, f).audio.bitDepth } : null] }] })), i(this, f).subtitles && e.data.push({ id: 174, data: [{ id: 215, data: Je }, { id: 29637, data: Je }, { id: 131, data: qi }, { id: 134, data: i(this, f).subtitles.codec }, i(this, ae)] });
};
st = /* @__PURE__ */ new WeakSet();
Lt = function() {
  let e = { id: 408125543, size: i(this, f).streaming ? -1 : li, data: [i(this, f).streaming ? null : i(this, A), i(this, Le), i(this, fe)] };
  if (_(this, We, e), i(this, l).writeEBML(e), i(this, l) instanceof Fe && i(this, l).target.options.onHeader) {
    let { data: t, start: r } = i(this, l).getTrackedWrites();
    i(this, l).target.options.onHeader(t, r);
  }
};
Rt = /* @__PURE__ */ new WeakSet();
gi = function() {
  _(this, oe, { id: 475249515, data: [] });
};
Y = /* @__PURE__ */ new WeakSet();
ie = function() {
  i(this, l) instanceof Ct && i(this, l).flush();
};
X = /* @__PURE__ */ new WeakSet();
re = function() {
  return i(this, l).dataOffsets.get(i(this, We));
};
yt = /* @__PURE__ */ new WeakSet();
bi = function(e) {
  if (e.decoderConfig) {
    if (e.decoderConfig.colorSpace) {
      let t = e.decoderConfig.colorSpace;
      if (_(this, Re, t), i(this, Q).data = [{ id: 21937, data: { rgb: 1, bt709: 1, bt470bg: 5, smpte170m: 6 }[t.matrix] }, { id: 21946, data: { bt709: 1, smpte170m: 6, "iec61966-2-1": 13 }[t.transfer] }, { id: 21947, data: { bt709: 1, bt470bg: 5, smpte170m: 6 }[t.primaries] }, { id: 21945, data: [1, 2][Number(t.fullRange)] }], !i(this, f).streaming) {
        let r = i(this, l).pos;
        i(this, l).seek(i(this, l).offsets.get(i(this, Q))), i(this, l).writeEBML(i(this, Q)), i(this, l).seek(r);
      }
    }
    e.decoderConfig.description && (i(this, f).streaming ? _(this, _e, d(this, he, Pe).call(this, e.decoderConfig.description)) : d(this, Ue, tt).call(this, i(this, _e), e.decoderConfig.description));
  }
};
xt = /* @__PURE__ */ new WeakSet();
vi = function(e) {
  if (e.type !== "key" || !i(this, Re)) return;
  let t = 0;
  if (J(e.data, 0, 2) !== 2) return;
  t += 2;
  let r = (J(e.data, t + 1, t + 2) << 1) + J(e.data, t + 0, t + 1);
  t += 2, r === 3 && t++;
  let n = J(e.data, t + 0, t + 1);
  if (t++, n) return;
  let s = J(e.data, t + 0, t + 1);
  if (t++, s !== 0) return;
  t += 2;
  let o = J(e.data, t + 0, t + 24);
  if (t += 24, o !== 4817730) return;
  r >= 2 && t++;
  let h = { rgb: 7, bt709: 2, bt470bg: 1, smpte170m: 3 }[i(this, Re).matrix];
  Fi(e.data, t + 0, t + 3, h);
};
ye = /* @__PURE__ */ new WeakSet();
Ye = function() {
  let e = Math.min(i(this, f).video ? i(this, le) : 1 / 0, i(this, f).audio ? i(this, ce) : 1 / 0), t = i(this, j);
  for (; t.length > 0 && t[0].timestamp <= e; ) d(this, C, U).call(this, t.shift(), !i(this, f).video && !i(this, f).audio);
};
xe = /* @__PURE__ */ new WeakSet();
Xe = function(e, t, r, n, s, o) {
  let h = d(this, Pt, yi).call(this, r, n);
  return { data: e, additions: o, type: t, timestamp: h, duration: s, trackNumber: n };
};
Pt = /* @__PURE__ */ new WeakSet();
yi = function(e, t) {
  let r = t === ue ? i(this, le) : t === ze ? i(this, ce) : i(this, et);
  if (t !== Je) {
    let n = t === ue ? i(this, Ce) : i(this, Me);
    if (i(this, f).firstTimestampBehavior === "strict" && r === -1 && e !== 0) throw new Error(`The first chunk for your media track must have a timestamp of 0 (received ${e}). Non-zero first timestamps are often caused by directly piping frames or audio data from a MediaStreamTrack into the encoder. Their timestamps are typically relative to the age of the document, which is probably what you want.

If you want to offset all timestamps of a track such that the first one is zero, set firstTimestampBehavior: 'offset' in the options.
If you want to allow non-zero first timestamps, set firstTimestampBehavior: 'permissive'.
`);
    i(this, f).firstTimestampBehavior === "offset" && (e -= n);
  }
  if (e < r) throw new Error(`Timestamps must be monotonically increasing (went from ${r} to ${e}).`);
  if (e < 0) throw new Error(`Timestamps must be non-negative (received ${e}).`);
  return e;
};
C = /* @__PURE__ */ new WeakSet();
U = function(e, t) {
  i(this, f).streaming && !i(this, fe) && (d(this, nt, Wt).call(this), d(this, st, Lt).call(this));
  let r = Math.floor(e.timestamp / 1e3), n = r - i(this, Ve), s = t && e.type === "key" && n >= 1e3, o = n >= ji;
  if ((!i(this, M) || s || o) && (d(this, Ft, xi).call(this, r), n = 0), n < 0) return;
  let h = new Uint8Array(4), m = new DataView(h.buffer);
  if (m.setUint8(0, 128 | e.trackNumber), m.setInt16(1, n, false), e.duration === void 0 && !e.additions) {
    m.setUint8(3, +(e.type === "key") << 7);
    let g = { id: 163, data: [h, e.data] };
    i(this, l).writeEBML(g);
  } else {
    let g = Math.floor(e.duration / 1e3), p = { id: 160, data: [{ id: 161, data: [h, e.data] }, e.duration !== void 0 ? { id: 155, data: g } : null, e.additions ? { id: 30113, data: e.additions } : null] };
    i(this, l).writeEBML(p);
  }
  _(this, de, Math.max(i(this, de), r));
};
he = /* @__PURE__ */ new WeakSet();
Pe = function(e) {
  return { id: 25506, size: 4, data: new Uint8Array(e) };
};
Ue = /* @__PURE__ */ new WeakSet();
tt = function(e, t) {
  let r = i(this, l).pos;
  i(this, l).seek(i(this, l).offsets.get(e));
  let n = 6 + t.byteLength, s = Ee - n;
  if (s < 0) {
    let o = t.byteLength + s;
    t instanceof ArrayBuffer ? t = t.slice(0, o) : t = t.buffer.slice(0, o), s = 0;
  }
  e = [d(this, he, Pe).call(this, t), { id: 236, size: 4, data: new Uint8Array(s) }], i(this, l).writeEBML(e), i(this, l).seek(r);
};
Ft = /* @__PURE__ */ new WeakSet();
xi = function(e) {
  i(this, M) && d(this, it, Vt).call(this), i(this, l) instanceof Fe && i(this, l).target.options.onCluster && i(this, l).startTrackingWrites(), _(this, M, { id: 524531317, size: i(this, f).streaming ? -1 : ci, data: [{ id: 231, data: e }] }), i(this, l).writeEBML(i(this, M)), _(this, Ve, e);
  let t = i(this, l).offsets.get(i(this, M)) - i(this, X, re);
  i(this, oe).data.push({ id: 187, data: [{ id: 179, data: e }, i(this, f).video ? { id: 183, data: [{ id: 247, data: ue }, { id: 241, data: t }] } : null, i(this, f).audio ? { id: 183, data: [{ id: 247, data: ze }, { id: 241, data: t }] } : null] });
};
it = /* @__PURE__ */ new WeakSet();
Vt = function() {
  if (!i(this, f).streaming) {
    let e = i(this, l).pos - i(this, l).dataOffsets.get(i(this, M)), t = i(this, l).pos;
    i(this, l).seek(i(this, l).offsets.get(i(this, M)) + 4), i(this, l).writeEBMLVarInt(e, ci), i(this, l).seek(t);
  }
  if (i(this, l) instanceof Fe && i(this, l).target.options.onCluster) {
    let { data: e, start: t } = i(this, l).getTrackedWrites();
    i(this, l).target.options.onCluster(e, t, i(this, Ve));
  }
};
ke = /* @__PURE__ */ new WeakSet();
Ke = function() {
  if (i(this, Ie)) throw new Error("Cannot add new video or audio chunks after the file has been finalized.");
};
new TextEncoder();
const T = document.getElementById("gpu-canvas"), I = document.getElementById("render-btn"), Yt = document.getElementById("scene-select"), Xt = document.getElementById("res-width"), Kt = document.getElementById("res-height"), kt = document.getElementById("obj-file");
kt && (kt.accept = ".obj,.glb,.vrm");
const Ki = document.getElementById("max-depth"), Qi = document.getElementById("spp-frame"), Zi = document.getElementById("recompile-btn"), Ji = document.getElementById("update-interval"), W = document.getElementById("anim-select"), L = document.getElementById("record-btn"), er = document.getElementById("rec-fps"), tr = document.getElementById("rec-duration"), ir = document.getElementById("rec-spp"), rr = document.getElementById("rec-batch"), Dt = document.createElement("div");
Object.assign(Dt.style, { position: "fixed", bottom: "10px", left: "10px", color: "#0f0", background: "rgba(0,0,0,0.7)", padding: "8px", fontFamily: "monospace", fontSize: "14px", pointerEvents: "none", zIndex: "9999", borderRadius: "4px" });
document.body.appendChild(Dt);
let G = 0, B = false, Ne = false, ge = null, Oe = null;
async function nr() {
  const e = new Si(T), t = new Ri();
  let r = 0;
  try {
    await e.init(), await t.initWasm();
  } catch (u) {
    alert("Initialization failed: " + u), console.error(u);
    return;
  }
  const n = () => {
    const u = parseInt(Ki.value, 10) || 10, b = parseInt(Qi.value, 10) || 1;
    e.buildPipeline(u, b);
  };
  n();
  const s = () => {
    const u = parseInt(Xt.value, 10) || 720, b = parseInt(Kt.value, 10) || 480;
    e.updateScreenSize(u, b), t.hasWorld && (t.updateCamera(u, b), e.updateSceneUniforms(t.cameraData, 0)), e.recreateBindGroup(), e.resetAccumulation(), G = 0, r = 0;
  }, o = async (u, b = true) => {
    B = false, console.log(`Loading Scene: ${u}...`);
    let v, k;
    u === "viewer" && ge && (Oe === "obj" ? v = ge : Oe === "glb" && (k = new Uint8Array(ge))), t.loadScene(u, v, k), t.printStats(), await e.loadTexturesFromWorld(t), e.updateCombinedGeometry(t.vertices, t.normals, t.uvs), e.updateCombinedBVH(t.tlas, t.blas), e.updateBuffer("index", t.indices), e.updateBuffer("attr", t.attributes), e.updateBuffer("instance", t.instances), s(), y(), b && (B = true, I && (I.textContent = "Stop Rendering"));
  }, h = async () => {
    if (Ne) return;
    B = false, Ne = true, L.textContent = "Initializing...", L.disabled = true, I && (I.textContent = "Resume Rendering");
    const u = parseInt(er.value, 10) || 30, b = parseInt(tr.value, 10) || 3, v = u * b, k = parseInt(ir.value, 10) || 64, ki = parseInt(rr.value, 10) || 4;
    console.log(`Starting recording: ${v} frames @ ${u}fps (VP9)`);
    const at = new Xi({ target: new ei(), video: { codec: "V_VP9", width: T.width, height: T.height, frameRate: u } }), me = new VideoEncoder({ output: (N, ot) => at.addVideoChunk(N, ot), error: (N) => console.error("VideoEncoder Error:", N) });
    me.configure({ codec: "vp09.00.10.08", width: T.width, height: T.height, bitrate: 12e6 });
    try {
      for (let O = 0; O < v; O++) {
        L.textContent = `Rec: ${O}/${v} (${Math.round(O / v * 100)}%)`, await new Promise((De) => setTimeout(De, 0));
        const Ti = O / u;
        t.update(Ti);
        let Z = false;
        Z || (Z = e.updateCombinedBVH(t.tlas, t.blas)), Z || (Z = e.updateBuffer("instance", t.instances)), Z || (Z = e.updateCombinedGeometry(t.vertices, t.normals, t.uvs)), Z || (Z = e.updateBuffer("index", t.indices)), Z || (Z = e.updateBuffer("attr", t.attributes)), t.updateCamera(T.width, T.height), e.updateSceneUniforms(t.cameraData, 0), Z && e.recreateBindGroup(), e.resetAccumulation();
        let we = 0;
        for (; we < k; ) {
          const De = Math.min(ki, k - we);
          for (let pe = 0; pe < De; pe++) e.render(we + pe);
          we += De, await e.device.queue.onSubmittedWorkDone(), we < k && await new Promise((pe) => setTimeout(pe, 0));
        }
        me.encodeQueueSize > 5 && await me.flush();
        const Ot = new VideoFrame(T, { timestamp: O * 1e6 / u, duration: 1e6 / u });
        me.encode(Ot, { keyFrame: O % u === 0 }), Ot.close();
      }
      L.textContent = "Finalizing...", await me.flush(), at.finalize();
      const { buffer: N } = at.target, ot = new Blob([N], { type: "video/webm" }), Nt = URL.createObjectURL(ot), dt = document.createElement("a");
      dt.href = Nt, dt.download = `raytrace_${Date.now()}.webm`, dt.click(), URL.revokeObjectURL(Nt);
    } catch (N) {
      console.error("Recording failed:", N), alert("Recording failed. See console.");
    } finally {
      Ne = false, B = true, L.textContent = "\u25CF Rec", L.disabled = false, I && (I.textContent = "Stop Rendering"), requestAnimationFrame(p);
    }
  };
  let m = performance.now(), g = 0;
  const p = () => {
    if (Ne || (requestAnimationFrame(p), !B || !t.hasWorld)) return;
    let u = parseInt(Ji.value, 10);
    if ((isNaN(u) || u < 0) && (u = 0), u > 0 && G >= u) {
      t.update(r / u / 60);
      let v = false;
      v || (v = e.updateCombinedBVH(t.tlas, t.blas)), v || (v = e.updateBuffer("instance", t.instances)), v || (v = e.updateCombinedGeometry(t.vertices, t.normals, t.uvs)), v || (v = e.updateBuffer("index", t.indices)), v || (v = e.updateBuffer("attr", t.attributes)), t.updateCamera(T.width, T.height), e.updateSceneUniforms(t.cameraData, 0), v && e.recreateBindGroup(), e.resetAccumulation(), G = 0;
    }
    G++, g++, r++, e.render(G);
    const b = performance.now();
    b - m >= 1e3 && (Dt.textContent = `FPS: ${g} | ${(1e3 / g).toFixed(2)}ms | Frame: ${G}`, g = 0, m = b);
  };
  I && I.addEventListener("click", () => {
    B = !B, I.textContent = B ? "Stop Rendering" : "Resume Rendering";
  }), L && L.addEventListener("click", h), Yt.addEventListener("change", (u) => o(u.target.value, false)), Xt.addEventListener("change", s), Kt.addEventListener("change", s), Zi.addEventListener("click", () => {
    B = false, n(), e.recreateBindGroup(), e.resetAccumulation(), G = 0, B = true;
  }), kt.addEventListener("change", async (u) => {
    var _a, _b;
    const b = (_a = u.target.files) == null ? void 0 : _a[0];
    if (!b) return;
    ((_b = b.name.split(".").pop()) == null ? void 0 : _b.toLowerCase()) === "obj" ? (ge = await b.text(), Oe = "obj") : (ge = await b.arrayBuffer(), Oe = "glb"), Yt.value = "viewer", o("viewer", false);
  });
  const y = () => {
    const u = t.getAnimationList();
    if (W.innerHTML = "", u.length === 0) {
      const b = document.createElement("option");
      b.text = "No Anim", W.add(b), W.disabled = true;
      return;
    }
    W.disabled = false, u.forEach((b, v) => {
      const k = document.createElement("option");
      k.text = `[${v}] ${b}`, k.value = v.toString(), W.add(k);
    }), W.value = "0";
  };
  W.addEventListener("change", () => {
    const u = parseInt(W.value, 10);
    t.setAnimation(u);
  }), s(), o("cornell", false), requestAnimationFrame(p);
}
nr().catch(console.error);
