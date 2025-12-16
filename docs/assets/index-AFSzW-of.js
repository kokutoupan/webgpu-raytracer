var __defProp = Object.defineProperty;
var __defNormalProp = (obj, key, value) => key in obj ? __defProp(obj, key, { enumerable: true, configurable: true, writable: true, value }) : obj[key] = value;
var __publicField = (obj, key, value) => __defNormalProp(obj, typeof key !== "symbol" ? key + "" : key, value);
(function() {
  const e = document.createElement("link").relList;
  if (e && e.supports && e.supports("modulepreload")) return;
  for (const s of document.querySelectorAll('link[rel="modulepreload"]')) r(s);
  new MutationObserver((s) => {
    for (const a of s) if (a.type === "childList") for (const c of a.addedNodes) c.tagName === "LINK" && c.rel === "modulepreload" && r(c);
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
const ui = `// =========================================================
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
class fi {
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
    let r = ui;
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
        const c = new Blob([a]), u = await createImageBitmap(c, { resizeWidth: 1024, resizeHeight: 1024 });
        r.push(u);
      } catch (c) {
        console.warn(`Failed tex ${s}`, c), r.push(await this.createFallbackBitmap());
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
    const c = e.length / 4;
    this.vertexCount = c, this.geometryBuffer = this.ensureBuffer(this.geometryBuffer, s, "GeometryBuffer"), !(r.length >= c * 2) && c > 0 && console.warn(`UV buffer mismatch: V=${c}, UV=${r.length / 2}. Filling 0.`);
    const b = e.length, y = i.length, z = r.length, hi = b + y + z, Me = new Float32Array(hi);
    return Me.set(e, 0), Me.set(i, b), Me.set(r, b + y), this.device.queue.writeBuffer(this.geometryBuffer, 0, Me), a;
  }
  updateCombinedBVH(e, i) {
    const r = e.byteLength, s = i.byteLength, a = r + s;
    let c = false;
    return (!this.nodesBuffer || this.nodesBuffer.size < a) && (c = true), this.nodesBuffer = this.ensureBuffer(this.nodesBuffer, a, "NodesBuffer"), this.device.queue.writeBuffer(this.nodesBuffer, 0, e), this.device.queue.writeBuffer(this.nodesBuffer, r, i), this.blasOffset = e.length / 8, c;
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
    const r = Math.ceil(this.canvas.width / 8), s = Math.ceil(this.canvas.height / 8), a = this.device.createCommandEncoder(), c = a.beginComputePass();
    c.setPipeline(this.pipeline), c.setBindGroup(0, this.bindGroup), c.dispatchWorkgroups(r, s), c.end(), a.copyTextureToTexture({ texture: this.renderTarget }, { texture: this.context.getCurrentTexture() }, { width: this.canvas.width, height: this.canvas.height, depthOrArrayLayers: 1 }), this.device.queue.submit([a.finish()]);
  }
}
let h;
function pi(t) {
  const e = h.__externref_table_alloc();
  return h.__wbindgen_externrefs.set(e, t), e;
}
function _i(t, e) {
  return t = t >>> 0, ee().subarray(t / 1, t / 1 + e);
}
let P = null;
function Mt() {
  return (P === null || P.buffer.detached === true || P.buffer.detached === void 0 && P.buffer !== h.memory.buffer) && (P = new DataView(h.memory.buffer)), P;
}
function We(t, e) {
  return t = t >>> 0, mi(t, e);
}
let he = null;
function ee() {
  return (he === null || he.byteLength === 0) && (he = new Uint8Array(h.memory.buffer)), he;
}
function wi(t, e) {
  try {
    return t.apply(this, e);
  } catch (i) {
    const r = pi(i);
    h.__wbindgen_exn_store(r);
  }
}
function Ut(t) {
  return t == null;
}
function Wt(t, e) {
  const i = e(t.length * 1, 1) >>> 0;
  return ee().set(t, i / 1), W = t.length, i;
}
function Qe(t, e, i) {
  if (i === void 0) {
    const u = we.encode(t), b = e(u.length, 1) >>> 0;
    return ee().subarray(b, b + u.length).set(u), W = u.length, b;
  }
  let r = t.length, s = e(r, 1) >>> 0;
  const a = ee();
  let c = 0;
  for (; c < r; c++) {
    const u = t.charCodeAt(c);
    if (u > 127) break;
    a[s + c] = u;
  }
  if (c !== r) {
    c !== 0 && (t = t.slice(c)), s = i(s, r, r = c + t.length * 3, 1) >>> 0;
    const u = ee().subarray(s + c, s + r), b = we.encodeInto(t, u);
    c += b.written, s = i(s, r, c, 1) >>> 0;
  }
  return W = c, s;
}
let Le = new TextDecoder("utf-8", { ignoreBOM: true, fatal: true });
Le.decode();
const gi = 2146435072;
let Ze = 0;
function mi(t, e) {
  return Ze += e, Ze >= gi && (Le = new TextDecoder("utf-8", { ignoreBOM: true, fatal: true }), Le.decode(), Ze = e), Le.decode(ee().subarray(t, t + e));
}
const we = new TextEncoder();
"encodeInto" in we || (we.encodeInto = function(t, e) {
  const i = we.encode(t);
  return e.set(i), { read: t.length, written: i.length };
});
let W = 0;
typeof FinalizationRegistry > "u" || new FinalizationRegistry((t) => h.__wbg_renderbuffers_free(t >>> 0, 1));
const Lt = typeof FinalizationRegistry > "u" ? { register: () => {
}, unregister: () => {
} } : new FinalizationRegistry((t) => h.__wbg_world_free(t >>> 0, 1));
class et {
  __destroy_into_raw() {
    const e = this.__wbg_ptr;
    return this.__wbg_ptr = 0, Lt.unregister(this), e;
  }
  free() {
    const e = this.__destroy_into_raw();
    h.__wbg_world_free(e, 0);
  }
  camera_ptr() {
    return h.world_camera_ptr(this.__wbg_ptr) >>> 0;
  }
  indices_len() {
    return h.world_indices_len(this.__wbg_ptr) >>> 0;
  }
  indices_ptr() {
    return h.world_indices_ptr(this.__wbg_ptr) >>> 0;
  }
  normals_len() {
    return h.world_normals_len(this.__wbg_ptr) >>> 0;
  }
  normals_ptr() {
    return h.world_normals_ptr(this.__wbg_ptr) >>> 0;
  }
  vertices_len() {
    return h.world_vertices_len(this.__wbg_ptr) >>> 0;
  }
  vertices_ptr() {
    return h.world_vertices_ptr(this.__wbg_ptr) >>> 0;
  }
  instances_len() {
    return h.world_instances_len(this.__wbg_ptr) >>> 0;
  }
  instances_ptr() {
    return h.world_instances_ptr(this.__wbg_ptr) >>> 0;
  }
  set_animation(e) {
    h.world_set_animation(this.__wbg_ptr, e);
  }
  update_camera(e, i) {
    h.world_update_camera(this.__wbg_ptr, e, i);
  }
  attributes_len() {
    return h.world_attributes_len(this.__wbg_ptr) >>> 0;
  }
  attributes_ptr() {
    return h.world_attributes_ptr(this.__wbg_ptr) >>> 0;
  }
  get_texture_ptr(e) {
    return h.world_get_texture_ptr(this.__wbg_ptr, e) >>> 0;
  }
  get_texture_size(e) {
    return h.world_get_texture_size(this.__wbg_ptr, e) >>> 0;
  }
  get_texture_count() {
    return h.world_get_texture_count(this.__wbg_ptr) >>> 0;
  }
  get_animation_name(e) {
    let i, r;
    try {
      const s = h.world_get_animation_name(this.__wbg_ptr, e);
      return i = s[0], r = s[1], We(s[0], s[1]);
    } finally {
      h.__wbindgen_free(i, r, 1);
    }
  }
  load_animation_glb(e) {
    const i = Wt(e, h.__wbindgen_malloc), r = W;
    h.world_load_animation_glb(this.__wbg_ptr, i, r);
  }
  get_animation_count() {
    return h.world_get_animation_count(this.__wbg_ptr) >>> 0;
  }
  constructor(e, i, r) {
    const s = Qe(e, h.__wbindgen_malloc, h.__wbindgen_realloc), a = W;
    var c = Ut(i) ? 0 : Qe(i, h.__wbindgen_malloc, h.__wbindgen_realloc), u = W, b = Ut(r) ? 0 : Wt(r, h.__wbindgen_malloc), y = W;
    const z = h.world_new(s, a, c, u, b, y);
    return this.__wbg_ptr = z >>> 0, Lt.register(this, this.__wbg_ptr, this), this;
  }
  update(e) {
    h.world_update(this.__wbg_ptr, e);
  }
  uvs_len() {
    return h.world_uvs_len(this.__wbg_ptr) >>> 0;
  }
  uvs_ptr() {
    return h.world_uvs_ptr(this.__wbg_ptr) >>> 0;
  }
  blas_len() {
    return h.world_blas_len(this.__wbg_ptr) >>> 0;
  }
  blas_ptr() {
    return h.world_blas_ptr(this.__wbg_ptr) >>> 0;
  }
  tlas_len() {
    return h.world_tlas_len(this.__wbg_ptr) >>> 0;
  }
  tlas_ptr() {
    return h.world_tlas_ptr(this.__wbg_ptr) >>> 0;
  }
}
Symbol.dispose && (et.prototype[Symbol.dispose] = et.prototype.free);
const bi = /* @__PURE__ */ new Set(["basic", "cors", "default"]);
async function vi(t, e) {
  if (typeof Response == "function" && t instanceof Response) {
    if (typeof WebAssembly.instantiateStreaming == "function") try {
      return await WebAssembly.instantiateStreaming(t, e);
    } catch (r) {
      if (t.ok && bi.has(t.type) && t.headers.get("Content-Type") !== "application/wasm") console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", r);
      else throw r;
    }
    const i = await t.arrayBuffer();
    return await WebAssembly.instantiate(i, e);
  } else {
    const i = await WebAssembly.instantiate(t, e);
    return i instanceof WebAssembly.Instance ? { instance: i, module: t } : i;
  }
}
function yi() {
  const t = {};
  return t.wbg = {}, t.wbg.__wbg___wbindgen_throw_dd24417ed36fc46e = function(e, i) {
    throw new Error(We(e, i));
  }, t.wbg.__wbg_error_7534b8e9a36f1ab4 = function(e, i) {
    let r, s;
    try {
      r = e, s = i, console.error(We(e, i));
    } finally {
      h.__wbindgen_free(r, s, 1);
    }
  }, t.wbg.__wbg_getRandomValues_1c61fac11405ffdc = function() {
    return wi(function(e, i) {
      globalThis.crypto.getRandomValues(_i(e, i));
    }, arguments);
  }, t.wbg.__wbg_log_1d990106d99dacb7 = function(e) {
    console.log(e);
  }, t.wbg.__wbg_new_8a6f238a6ece86ea = function() {
    return new Error();
  }, t.wbg.__wbg_stack_0ed75d68575b0f3c = function(e, i) {
    const r = i.stack, s = Qe(r, h.__wbindgen_malloc, h.__wbindgen_realloc), a = W;
    Mt().setInt32(e + 4, a, true), Mt().setInt32(e + 0, s, true);
  }, t.wbg.__wbindgen_cast_2241b6af4c4b2941 = function(e, i) {
    return We(e, i);
  }, t.wbg.__wbindgen_init_externref_table = function() {
    const e = h.__wbindgen_externrefs, i = e.grow(4);
    e.set(0, void 0), e.set(i + 0, void 0), e.set(i + 1, null), e.set(i + 2, true), e.set(i + 3, false);
  }, t;
}
function Si(t, e) {
  return h = t.exports, Ft.__wbindgen_wasm_module = e, P = null, he = null, h.__wbindgen_start(), h;
}
async function Ft(t) {
  if (h !== void 0) return h;
  typeof t < "u" && (Object.getPrototypeOf(t) === Object.prototype ? { module_or_path: t } = t : console.warn("using deprecated parameters for the initialization function; pass a single object instead")), typeof t > "u" && (t = new URL("/webgpu-raytracer/assets/rust_shader_tools_bg-CC5HVMsp.wasm", import.meta.url));
  const e = yi();
  (typeof t == "string" || typeof Request == "function" && t instanceof Request || typeof URL == "function" && t instanceof URL) && (t = fetch(t));
  const { instance: i, module: r } = await vi(await t, e);
  return Si(i, r);
}
class ki {
  constructor() {
    __publicField(this, "world", null);
    __publicField(this, "wasmMemory", null);
  }
  async initWasm() {
    const e = await Ft();
    this.wasmMemory = e.memory, console.log("Wasm initialized");
  }
  loadScene(e, i, r) {
    this.world && this.world.free(), this.world = new et(e, i, r);
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
const m = { defaultWidth: 720, defaultHeight: 480, defaultDepth: 10, defaultSPP: 1, signalingServerUrl: "ws://localhost:8080", ids: { canvas: "gpu-canvas", renderBtn: "render-btn", sceneSelect: "scene-select", resWidth: "res-width", resHeight: "res-height", objFile: "obj-file", maxDepth: "max-depth", sppFrame: "spp-frame", recompileBtn: "recompile-btn", updateInterval: "update-interval", animSelect: "anim-select", recordBtn: "record-btn", recFps: "rec-fps", recDuration: "rec-duration", recSpp: "rec-spp", recBatch: "rec-batch", btnHost: "btn-host", btnWorker: "btn-worker", btnSendScene: "btn-send-scene", statusDiv: "status" } };
class xi {
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
    this.canvas = this.el(m.ids.canvas), this.btnRender = this.el(m.ids.renderBtn), this.sceneSelect = this.el(m.ids.sceneSelect), this.inputWidth = this.el(m.ids.resWidth), this.inputHeight = this.el(m.ids.resHeight), this.inputFile = this.setupFileInput(), this.inputDepth = this.el(m.ids.maxDepth), this.inputSPP = this.el(m.ids.sppFrame), this.btnRecompile = this.el(m.ids.recompileBtn), this.inputUpdateInterval = this.el(m.ids.updateInterval), this.animSelect = this.el(m.ids.animSelect), this.btnRecord = this.el(m.ids.recordBtn), this.inputRecFps = this.el(m.ids.recFps), this.inputRecDur = this.el(m.ids.recDuration), this.inputRecSpp = this.el(m.ids.recSpp), this.inputRecBatch = this.el(m.ids.recBatch), this.btnHost = this.el(m.ids.btnHost), this.btnWorker = this.el(m.ids.btnWorker), this.btnSendScene = this.el(m.ids.btnSendScene), this.statusDiv = this.el(m.ids.statusDiv), this.statsDiv = this.createStatsDiv(), this.bindEvents();
  }
  el(e) {
    const i = document.getElementById(e);
    if (!i) throw new Error(`Element not found: ${e}`);
    return i;
  }
  setupFileInput() {
    const e = this.el(m.ids.objFile);
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
      return (_a = this.onResolutionChange) == null ? void 0 : _a.call(this, parseInt(this.inputWidth.value) || m.defaultWidth, parseInt(this.inputHeight.value) || m.defaultHeight);
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
  setHostWorkerButtonsEnabled(e) {
    this.btnHost.disabled = !e, this.btnWorker.disabled = !e;
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
    return { width: parseInt(this.inputWidth.value, 10) || m.defaultWidth, height: parseInt(this.inputHeight.value, 10) || m.defaultHeight, fps: parseInt(this.inputRecFps.value, 10) || 30, duration: parseInt(this.inputRecDur.value, 10) || 3, spp: parseInt(this.inputRecSpp.value, 10) || 64, batch: parseInt(this.inputRecBatch.value, 10) || 4, anim: parseInt(this.animSelect.value, 10) || 0 };
  }
  setRenderConfig(e) {
    this.inputWidth.value = e.width.toString(), this.inputHeight.value = e.height.toString(), this.inputRecFps.value = e.fps.toString(), this.inputRecDur.value = e.duration.toString(), this.inputRecSpp.value = e.spp.toString(), this.inputRecBatch.value = e.batch.toString();
  }
}
var pt = (t, e, i) => {
  if (!e.has(t)) throw TypeError("Cannot " + i);
}, n = (t, e, i) => (pt(t, e, "read from private field"), i ? i.call(t) : e.get(t)), o = (t, e, i) => {
  if (e.has(t)) throw TypeError("Cannot add the same private member more than once");
  e instanceof WeakSet ? e.add(t) : e.set(t, i);
}, p = (t, e, i, r) => (pt(t, e, "write to private field"), e.set(t, i), i), d = (t, e, i) => (pt(t, e, "access private method"), i), Nt = class {
  constructor(t) {
    this.value = t;
  }
}, _t = class {
  constructor(t) {
    this.value = t;
  }
}, Ht = (t) => t < 256 ? 1 : t < 65536 ? 2 : t < 1 << 24 ? 3 : t < 2 ** 32 ? 4 : t < 2 ** 40 ? 5 : 6, Ci = (t) => {
  if (t < 127) return 1;
  if (t < 16383) return 2;
  if (t < (1 << 21) - 1) return 3;
  if (t < (1 << 28) - 1) return 4;
  if (t < 2 ** 35 - 1) return 5;
  if (t < 2 ** 42 - 1) return 6;
  throw new Error("EBML VINT size not supported " + t);
}, K = (t, e, i) => {
  let r = 0;
  for (let s = e; s < i; s++) {
    let a = Math.floor(s / 8), c = t[a], u = 7 - (s & 7), b = (c & 1 << u) >> u;
    r <<= 1, r |= b;
  }
  return r;
}, Ti = (t, e, i, r) => {
  for (let s = e; s < i; s++) {
    let a = Math.floor(s / 8), c = t[a], u = 7 - (s & 7);
    c &= ~(1 << u), c |= (r & 1 << i - s - 1) >> i - s - 1 << u, t[a] = c;
  }
}, Ye = class {
}, Vt = class extends Ye {
  constructor() {
    super(...arguments), this.buffer = null;
  }
}, Ot = class extends Ye {
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
}, Bi = class extends Ye {
  constructor(t, e) {
    if (super(), this.stream = t, this.options = e, !(t instanceof FileSystemWritableFileStream)) throw new TypeError("FileSystemWritableFileStreamTarget requires a FileSystemWritableFileStream instance.");
    if (e !== void 0 && typeof e != "object") throw new TypeError("FileSystemWritableFileStreamTarget's options, when provided, must be an object.");
    if (e && e.chunkSize !== void 0 && (!Number.isInteger(e.chunkSize) || e.chunkSize <= 0)) throw new TypeError("options.chunkSize, when provided, must be a positive integer");
  }
}, L, w, tt, $t, it, Gt, nt, qt, De, rt, st, jt, Kt = class {
  constructor() {
    o(this, tt), o(this, it), o(this, nt), o(this, De), o(this, st), this.pos = 0, o(this, L, new Uint8Array(8)), o(this, w, new DataView(n(this, L).buffer)), this.offsets = /* @__PURE__ */ new WeakMap(), this.dataOffsets = /* @__PURE__ */ new WeakMap();
  }
  seek(t) {
    this.pos = t;
  }
  writeEBMLVarInt(t, e = Ci(t)) {
    let i = 0;
    switch (e) {
      case 1:
        n(this, w).setUint8(i++, 128 | t);
        break;
      case 2:
        n(this, w).setUint8(i++, 64 | t >> 8), n(this, w).setUint8(i++, t);
        break;
      case 3:
        n(this, w).setUint8(i++, 32 | t >> 16), n(this, w).setUint8(i++, t >> 8), n(this, w).setUint8(i++, t);
        break;
      case 4:
        n(this, w).setUint8(i++, 16 | t >> 24), n(this, w).setUint8(i++, t >> 16), n(this, w).setUint8(i++, t >> 8), n(this, w).setUint8(i++, t);
        break;
      case 5:
        n(this, w).setUint8(i++, 8 | t / 2 ** 32 & 7), n(this, w).setUint8(i++, t >> 24), n(this, w).setUint8(i++, t >> 16), n(this, w).setUint8(i++, t >> 8), n(this, w).setUint8(i++, t);
        break;
      case 6:
        n(this, w).setUint8(i++, 4 | t / 2 ** 40 & 3), n(this, w).setUint8(i++, t / 2 ** 32 | 0), n(this, w).setUint8(i++, t >> 24), n(this, w).setUint8(i++, t >> 16), n(this, w).setUint8(i++, t >> 8), n(this, w).setUint8(i++, t);
        break;
      default:
        throw new Error("Bad EBML VINT size " + e);
    }
    this.write(n(this, L).subarray(0, i));
  }
  writeEBML(t) {
    if (t !== null) if (t instanceof Uint8Array) this.write(t);
    else if (Array.isArray(t)) for (let e of t) this.writeEBML(e);
    else if (this.offsets.set(t, this.pos), d(this, De, rt).call(this, t.id), Array.isArray(t.data)) {
      let e = this.pos, i = t.size === -1 ? 1 : t.size ?? 4;
      t.size === -1 ? d(this, tt, $t).call(this, 255) : this.seek(this.pos + i);
      let r = this.pos;
      if (this.dataOffsets.set(t, r), this.writeEBML(t.data), t.size !== -1) {
        let s = this.pos - r, a = this.pos;
        this.seek(e), this.writeEBMLVarInt(s, i), this.seek(a);
      }
    } else if (typeof t.data == "number") {
      let e = t.size ?? Ht(t.data);
      this.writeEBMLVarInt(e), d(this, De, rt).call(this, t.data, e);
    } else typeof t.data == "string" ? (this.writeEBMLVarInt(t.data.length), d(this, st, jt).call(this, t.data)) : t.data instanceof Uint8Array ? (this.writeEBMLVarInt(t.data.byteLength, t.size), this.write(t.data)) : t.data instanceof Nt ? (this.writeEBMLVarInt(4), d(this, it, Gt).call(this, t.data.value)) : t.data instanceof _t && (this.writeEBMLVarInt(8), d(this, nt, qt).call(this, t.data.value));
  }
};
L = /* @__PURE__ */ new WeakMap();
w = /* @__PURE__ */ new WeakMap();
tt = /* @__PURE__ */ new WeakSet();
$t = function(t) {
  n(this, w).setUint8(0, t), this.write(n(this, L).subarray(0, 1));
};
it = /* @__PURE__ */ new WeakSet();
Gt = function(t) {
  n(this, w).setFloat32(0, t, false), this.write(n(this, L).subarray(0, 4));
};
nt = /* @__PURE__ */ new WeakSet();
qt = function(t) {
  n(this, w).setFloat64(0, t, false), this.write(n(this, L));
};
De = /* @__PURE__ */ new WeakSet();
rt = function(t, e = Ht(t)) {
  let i = 0;
  switch (e) {
    case 6:
      n(this, w).setUint8(i++, t / 2 ** 40 | 0);
    case 5:
      n(this, w).setUint8(i++, t / 2 ** 32 | 0);
    case 4:
      n(this, w).setUint8(i++, t >> 24);
    case 3:
      n(this, w).setUint8(i++, t >> 16);
    case 2:
      n(this, w).setUint8(i++, t >> 8);
    case 1:
      n(this, w).setUint8(i++, t);
      break;
    default:
      throw new Error("Bad UINT size " + e);
  }
  this.write(n(this, L).subarray(0, i));
};
st = /* @__PURE__ */ new WeakSet();
jt = function(t) {
  this.write(new Uint8Array(t.split("").map((e) => e.charCodeAt(0))));
};
var ze, q, xe, Pe, at, Ei = class extends Kt {
  constructor(t) {
    super(), o(this, Pe), o(this, ze, void 0), o(this, q, new ArrayBuffer(2 ** 16)), o(this, xe, new Uint8Array(n(this, q))), p(this, ze, t);
  }
  write(t) {
    d(this, Pe, at).call(this, this.pos + t.byteLength), n(this, xe).set(t, this.pos), this.pos += t.byteLength;
  }
  finalize() {
    d(this, Pe, at).call(this, this.pos), n(this, ze).buffer = n(this, q).slice(0, this.pos);
  }
};
ze = /* @__PURE__ */ new WeakMap();
q = /* @__PURE__ */ new WeakMap();
xe = /* @__PURE__ */ new WeakMap();
Pe = /* @__PURE__ */ new WeakSet();
at = function(t) {
  let e = n(this, q).byteLength;
  for (; e < t; ) e *= 2;
  if (e === n(this, q).byteLength) return;
  let i = new ArrayBuffer(e), r = new Uint8Array(i);
  r.set(n(this, xe), 0), p(this, q, i), p(this, xe, r);
};
var X, k, x, F, Ae = class extends Kt {
  constructor(t) {
    super(), this.target = t, o(this, X, false), o(this, k, void 0), o(this, x, void 0), o(this, F, void 0);
  }
  write(t) {
    if (!n(this, X)) return;
    let e = this.pos;
    if (e < n(this, x)) {
      if (e + t.byteLength <= n(this, x)) return;
      t = t.subarray(n(this, x) - e), e = 0;
    }
    let i = e + t.byteLength - n(this, x), r = n(this, k).byteLength;
    for (; r < i; ) r *= 2;
    if (r !== n(this, k).byteLength) {
      let s = new Uint8Array(r);
      s.set(n(this, k), 0), p(this, k, s);
    }
    n(this, k).set(t, e - n(this, x)), p(this, F, Math.max(n(this, F), e + t.byteLength));
  }
  startTrackingWrites() {
    p(this, X, true), p(this, k, new Uint8Array(2 ** 10)), p(this, x, this.pos), p(this, F, this.pos);
  }
  getTrackedWrites() {
    if (!n(this, X)) throw new Error("Can't get tracked writes since nothing was tracked.");
    let e = { data: n(this, k).subarray(0, n(this, F) - n(this, x)), start: n(this, x), end: n(this, F) };
    return p(this, k, void 0), p(this, X, false), e;
  }
};
X = /* @__PURE__ */ new WeakMap();
k = /* @__PURE__ */ new WeakMap();
x = /* @__PURE__ */ new WeakMap();
F = /* @__PURE__ */ new WeakMap();
var Ri = 2 ** 24, Ai = 2, N, te, ge, ue, I, S, Oe, ot, wt, Yt, gt, Xt, me, $e, mt = class extends Ae {
  constructor(t, e) {
    var _a, _b;
    super(t), o(this, Oe), o(this, wt), o(this, gt), o(this, me), o(this, N, []), o(this, te, 0), o(this, ge, void 0), o(this, ue, void 0), o(this, I, void 0), o(this, S, []), p(this, ge, e), p(this, ue, ((_a = t.options) == null ? void 0 : _a.chunked) ?? false), p(this, I, ((_b = t.options) == null ? void 0 : _b.chunkSize) ?? Ri);
  }
  write(t) {
    super.write(t), n(this, N).push({ data: t.slice(), start: this.pos }), this.pos += t.byteLength;
  }
  flush() {
    var _a, _b;
    if (n(this, N).length === 0) return;
    let t = [], e = [...n(this, N)].sort((i, r) => i.start - r.start);
    t.push({ start: e[0].start, size: e[0].data.byteLength });
    for (let i = 1; i < e.length; i++) {
      let r = t[t.length - 1], s = e[i];
      s.start <= r.start + r.size ? r.size = Math.max(r.size, s.start + s.data.byteLength - r.start) : t.push({ start: s.start, size: s.data.byteLength });
    }
    for (let i of t) {
      i.data = new Uint8Array(i.size);
      for (let r of n(this, N)) i.start <= r.start && r.start < i.start + i.size && i.data.set(r.data, r.start - i.start);
      if (n(this, ue)) d(this, Oe, ot).call(this, i.data, i.start), d(this, me, $e).call(this);
      else {
        if (n(this, ge) && i.start < n(this, te)) throw new Error("Internal error: Monotonicity violation.");
        (_b = (_a = this.target.options).onData) == null ? void 0 : _b.call(_a, i.data, i.start), p(this, te, i.start + i.data.byteLength);
      }
    }
    n(this, N).length = 0;
  }
  finalize() {
    n(this, ue) && d(this, me, $e).call(this, true);
  }
};
N = /* @__PURE__ */ new WeakMap();
te = /* @__PURE__ */ new WeakMap();
ge = /* @__PURE__ */ new WeakMap();
ue = /* @__PURE__ */ new WeakMap();
I = /* @__PURE__ */ new WeakMap();
S = /* @__PURE__ */ new WeakMap();
Oe = /* @__PURE__ */ new WeakSet();
ot = function(t, e) {
  let i = n(this, S).findIndex((u) => u.start <= e && e < u.start + n(this, I));
  i === -1 && (i = d(this, gt, Xt).call(this, e));
  let r = n(this, S)[i], s = e - r.start, a = t.subarray(0, Math.min(n(this, I) - s, t.byteLength));
  r.data.set(a, s);
  let c = { start: s, end: s + a.byteLength };
  if (d(this, wt, Yt).call(this, r, c), r.written[0].start === 0 && r.written[0].end === n(this, I) && (r.shouldFlush = true), n(this, S).length > Ai) {
    for (let u = 0; u < n(this, S).length - 1; u++) n(this, S)[u].shouldFlush = true;
    d(this, me, $e).call(this);
  }
  a.byteLength < t.byteLength && d(this, Oe, ot).call(this, t.subarray(a.byteLength), e + a.byteLength);
};
wt = /* @__PURE__ */ new WeakSet();
Yt = function(t, e) {
  let i = 0, r = t.written.length - 1, s = -1;
  for (; i <= r; ) {
    let a = Math.floor(i + (r - i + 1) / 2);
    t.written[a].start <= e.start ? (i = a + 1, s = a) : r = a - 1;
  }
  for (t.written.splice(s + 1, 0, e), (s === -1 || t.written[s].end < e.start) && s++; s < t.written.length - 1 && t.written[s].end >= t.written[s + 1].start; ) t.written[s].end = Math.max(t.written[s].end, t.written[s + 1].end), t.written.splice(s + 1, 1);
};
gt = /* @__PURE__ */ new WeakSet();
Xt = function(t) {
  let i = { start: Math.floor(t / n(this, I)) * n(this, I), data: new Uint8Array(n(this, I)), written: [], shouldFlush: false };
  return n(this, S).push(i), n(this, S).sort((r, s) => r.start - s.start), n(this, S).indexOf(i);
};
me = /* @__PURE__ */ new WeakSet();
$e = function(t = false) {
  var _a, _b;
  for (let e = 0; e < n(this, S).length; e++) {
    let i = n(this, S)[e];
    if (!(!i.shouldFlush && !t)) {
      for (let r of i.written) {
        if (n(this, ge) && i.start + r.start < n(this, te)) throw new Error("Internal error: Monotonicity violation.");
        (_b = (_a = this.target.options).onData) == null ? void 0 : _b.call(_a, i.data.subarray(r.start, r.end), i.start + r.start), p(this, te, i.start + r.end);
      }
      n(this, S).splice(e--, 1);
    }
  }
};
var Ii = class extends mt {
  constructor(t, e) {
    var _a;
    super(new Ot({ onData: (i, r) => t.stream.write({ type: "write", data: i, position: r }), chunked: true, chunkSize: (_a = t.options) == null ? void 0 : _a.chunkSize }), e);
  }
}, de = 1, Ce = 2, Ge = 3, Mi = 1, Ui = 2, Wi = 17, Li = 2 ** 15, be = 2 ** 13, Dt = "https://github.com/Vanilagy/webm-muxer", Jt = 6, Zt = 5, Di = ["strict", "offset", "permissive"], f, l, Te, Be, E, le, J, j, ce, D, ie, ne, T, Ie, re, M, U, H, ve, ye, se, ae, qe, Ee, Se, dt, Qt, lt, ei, bt, ti, vt, ii, yt, ni, St, ri, kt, si, Xe, xt, Je, Ct, Tt, ai, V, Z, O, Q, ct, oi, ht, di, fe, Fe, pe, Ne, Bt, li, C, B, oe, Re, ke, je, Et, ci, Ke, Rt, _e, He, zi = class {
  constructor(t) {
    o(this, dt), o(this, lt), o(this, bt), o(this, vt), o(this, yt), o(this, St), o(this, kt), o(this, Xe), o(this, Je), o(this, Tt), o(this, V), o(this, O), o(this, ct), o(this, ht), o(this, fe), o(this, pe), o(this, Bt), o(this, C), o(this, oe), o(this, ke), o(this, Et), o(this, Ke), o(this, _e), o(this, f, void 0), o(this, l, void 0), o(this, Te, void 0), o(this, Be, void 0), o(this, E, void 0), o(this, le, void 0), o(this, J, void 0), o(this, j, void 0), o(this, ce, void 0), o(this, D, void 0), o(this, ie, void 0), o(this, ne, void 0), o(this, T, void 0), o(this, Ie, void 0), o(this, re, 0), o(this, M, []), o(this, U, []), o(this, H, []), o(this, ve, void 0), o(this, ye, void 0), o(this, se, -1), o(this, ae, -1), o(this, qe, -1), o(this, Ee, void 0), o(this, Se, false), d(this, dt, Qt).call(this, t), p(this, f, { type: "webm", firstTimestampBehavior: "strict", ...t }), this.target = t.target;
    let e = !!n(this, f).streaming;
    if (t.target instanceof Vt) p(this, l, new Ei(t.target));
    else if (t.target instanceof Ot) p(this, l, new mt(t.target, e));
    else if (t.target instanceof Bi) p(this, l, new Ii(t.target, e));
    else throw new Error(`Invalid target: ${t.target}`);
    d(this, lt, ei).call(this);
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
    if (d(this, _e, He).call(this), !n(this, f).video) throw new Error("No video track declared.");
    n(this, ve) === void 0 && p(this, ve, i), r && d(this, ct, oi).call(this, r);
    let s = d(this, pe, Ne).call(this, t, e, i, de);
    for (n(this, f).video.codec === "V_VP9" && d(this, ht, di).call(this, s), p(this, se, s.timestamp); n(this, U).length > 0 && n(this, U)[0].timestamp <= s.timestamp; ) {
      let a = n(this, U).shift();
      d(this, C, B).call(this, a, false);
    }
    !n(this, f).audio || s.timestamp <= n(this, ae) ? d(this, C, B).call(this, s, true) : n(this, M).push(s), d(this, fe, Fe).call(this), d(this, V, Z).call(this);
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
    if (d(this, _e, He).call(this), !n(this, f).audio) throw new Error("No audio track declared.");
    n(this, ye) === void 0 && p(this, ye, i), (r == null ? void 0 : r.decoderConfig) && (n(this, f).streaming ? p(this, D, d(this, oe, Re).call(this, r.decoderConfig.description)) : d(this, ke, je).call(this, n(this, D), r.decoderConfig.description));
    let s = d(this, pe, Ne).call(this, t, e, i, Ce);
    for (p(this, ae, s.timestamp); n(this, M).length > 0 && n(this, M)[0].timestamp <= s.timestamp; ) {
      let a = n(this, M).shift();
      d(this, C, B).call(this, a, true);
    }
    !n(this, f).video || s.timestamp <= n(this, se) ? d(this, C, B).call(this, s, !n(this, f).video) : n(this, U).push(s), d(this, fe, Fe).call(this), d(this, V, Z).call(this);
  }
  addSubtitleChunk(t, e, i) {
    if (typeof t != "object" || !t) throw new TypeError("addSubtitleChunk's first argument (chunk) must be an object.");
    if (!(t.body instanceof Uint8Array)) throw new TypeError("body must be an instance of Uint8Array.");
    if (!Number.isFinite(t.timestamp) || t.timestamp < 0) throw new TypeError("timestamp must be a non-negative real number.");
    if (!Number.isFinite(t.duration) || t.duration < 0) throw new TypeError("duration must be a non-negative real number.");
    if (t.additions && !(t.additions instanceof Uint8Array)) throw new TypeError("additions, when present, must be an instance of Uint8Array.");
    if (typeof e != "object") throw new TypeError("addSubtitleChunk's second argument (meta) must be an object.");
    if (d(this, _e, He).call(this), !n(this, f).subtitles) throw new Error("No subtitle track declared.");
    (e == null ? void 0 : e.decoderConfig) && (n(this, f).streaming ? p(this, ie, d(this, oe, Re).call(this, e.decoderConfig.description)) : d(this, ke, je).call(this, n(this, ie), e.decoderConfig.description));
    let r = d(this, pe, Ne).call(this, t.body, "key", i ?? t.timestamp, Ge, t.duration, t.additions);
    p(this, qe, r.timestamp), n(this, H).push(r), d(this, fe, Fe).call(this), d(this, V, Z).call(this);
  }
  finalize() {
    if (n(this, Se)) throw new Error("Cannot finalize a muxer more than once.");
    for (; n(this, M).length > 0; ) d(this, C, B).call(this, n(this, M).shift(), true);
    for (; n(this, U).length > 0; ) d(this, C, B).call(this, n(this, U).shift(), true);
    for (; n(this, H).length > 0 && n(this, H)[0].timestamp <= n(this, re); ) d(this, C, B).call(this, n(this, H).shift(), false);
    if (n(this, T) && d(this, Ke, Rt).call(this), n(this, l).writeEBML(n(this, ne)), !n(this, f).streaming) {
      let t = n(this, l).pos, e = n(this, l).pos - n(this, O, Q);
      n(this, l).seek(n(this, l).offsets.get(n(this, Te)) + 4), n(this, l).writeEBMLVarInt(e, Jt), n(this, J).data = new _t(n(this, re)), n(this, l).seek(n(this, l).offsets.get(n(this, J))), n(this, l).writeEBML(n(this, J)), n(this, E).data[0].data[1].data = n(this, l).offsets.get(n(this, ne)) - n(this, O, Q), n(this, E).data[1].data[1].data = n(this, l).offsets.get(n(this, Be)) - n(this, O, Q), n(this, E).data[2].data[1].data = n(this, l).offsets.get(n(this, le)) - n(this, O, Q), n(this, l).seek(n(this, l).offsets.get(n(this, E))), n(this, l).writeEBML(n(this, E)), n(this, l).seek(t);
    }
    d(this, V, Z).call(this), n(this, l).finalize(), p(this, Se, true);
  }
};
f = /* @__PURE__ */ new WeakMap();
l = /* @__PURE__ */ new WeakMap();
Te = /* @__PURE__ */ new WeakMap();
Be = /* @__PURE__ */ new WeakMap();
E = /* @__PURE__ */ new WeakMap();
le = /* @__PURE__ */ new WeakMap();
J = /* @__PURE__ */ new WeakMap();
j = /* @__PURE__ */ new WeakMap();
ce = /* @__PURE__ */ new WeakMap();
D = /* @__PURE__ */ new WeakMap();
ie = /* @__PURE__ */ new WeakMap();
ne = /* @__PURE__ */ new WeakMap();
T = /* @__PURE__ */ new WeakMap();
Ie = /* @__PURE__ */ new WeakMap();
re = /* @__PURE__ */ new WeakMap();
M = /* @__PURE__ */ new WeakMap();
U = /* @__PURE__ */ new WeakMap();
H = /* @__PURE__ */ new WeakMap();
ve = /* @__PURE__ */ new WeakMap();
ye = /* @__PURE__ */ new WeakMap();
se = /* @__PURE__ */ new WeakMap();
ae = /* @__PURE__ */ new WeakMap();
qe = /* @__PURE__ */ new WeakMap();
Ee = /* @__PURE__ */ new WeakMap();
Se = /* @__PURE__ */ new WeakMap();
dt = /* @__PURE__ */ new WeakSet();
Qt = function(t) {
  if (typeof t != "object") throw new TypeError("The muxer requires an options object to be passed to its constructor.");
  if (!(t.target instanceof Ye)) throw new TypeError("The target must be provided and an instance of Target.");
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
  if (t.firstTimestampBehavior && !Di.includes(t.firstTimestampBehavior)) throw new TypeError(`Invalid first timestamp behavior: ${t.firstTimestampBehavior}`);
  if (t.streaming !== void 0 && typeof t.streaming != "boolean") throw new TypeError(`Invalid streaming option: ${t.streaming}. Must be a boolean.`);
};
lt = /* @__PURE__ */ new WeakSet();
ei = function() {
  n(this, l) instanceof Ae && n(this, l).target.options.onHeader && n(this, l).startTrackingWrites(), d(this, bt, ti).call(this), n(this, f).streaming || d(this, St, ri).call(this), d(this, kt, si).call(this), d(this, vt, ii).call(this), d(this, yt, ni).call(this), n(this, f).streaming || (d(this, Xe, xt).call(this), d(this, Je, Ct).call(this)), d(this, Tt, ai).call(this), d(this, V, Z).call(this);
};
bt = /* @__PURE__ */ new WeakSet();
ti = function() {
  let t = { id: 440786851, data: [{ id: 17030, data: 1 }, { id: 17143, data: 1 }, { id: 17138, data: 4 }, { id: 17139, data: 8 }, { id: 17026, data: n(this, f).type ?? "webm" }, { id: 17031, data: 2 }, { id: 17029, data: 2 }] };
  n(this, l).writeEBML(t);
};
vt = /* @__PURE__ */ new WeakSet();
ii = function() {
  p(this, ce, { id: 236, size: 4, data: new Uint8Array(be) }), p(this, D, { id: 236, size: 4, data: new Uint8Array(be) }), p(this, ie, { id: 236, size: 4, data: new Uint8Array(be) });
};
yt = /* @__PURE__ */ new WeakSet();
ni = function() {
  p(this, j, { id: 21936, data: [{ id: 21937, data: 2 }, { id: 21946, data: 2 }, { id: 21947, data: 2 }, { id: 21945, data: 0 }] });
};
St = /* @__PURE__ */ new WeakSet();
ri = function() {
  const t = new Uint8Array([28, 83, 187, 107]), e = new Uint8Array([21, 73, 169, 102]), i = new Uint8Array([22, 84, 174, 107]);
  p(this, E, { id: 290298740, data: [{ id: 19899, data: [{ id: 21419, data: t }, { id: 21420, size: 5, data: 0 }] }, { id: 19899, data: [{ id: 21419, data: e }, { id: 21420, size: 5, data: 0 }] }, { id: 19899, data: [{ id: 21419, data: i }, { id: 21420, size: 5, data: 0 }] }] });
};
kt = /* @__PURE__ */ new WeakSet();
si = function() {
  let t = { id: 17545, data: new _t(0) };
  p(this, J, t);
  let e = { id: 357149030, data: [{ id: 2807729, data: 1e6 }, { id: 19840, data: Dt }, { id: 22337, data: Dt }, n(this, f).streaming ? null : t] };
  p(this, Be, e);
};
Xe = /* @__PURE__ */ new WeakSet();
xt = function() {
  let t = { id: 374648427, data: [] };
  p(this, le, t), n(this, f).video && t.data.push({ id: 174, data: [{ id: 215, data: de }, { id: 29637, data: de }, { id: 131, data: Mi }, { id: 134, data: n(this, f).video.codec }, n(this, ce), n(this, f).video.frameRate ? { id: 2352003, data: 1e9 / n(this, f).video.frameRate } : null, { id: 224, data: [{ id: 176, data: n(this, f).video.width }, { id: 186, data: n(this, f).video.height }, n(this, f).video.alpha ? { id: 21440, data: 1 } : null, n(this, j)] }] }), n(this, f).audio && (p(this, D, n(this, f).streaming ? n(this, D) || null : { id: 236, size: 4, data: new Uint8Array(be) }), t.data.push({ id: 174, data: [{ id: 215, data: Ce }, { id: 29637, data: Ce }, { id: 131, data: Ui }, { id: 134, data: n(this, f).audio.codec }, n(this, D), { id: 225, data: [{ id: 181, data: new Nt(n(this, f).audio.sampleRate) }, { id: 159, data: n(this, f).audio.numberOfChannels }, n(this, f).audio.bitDepth ? { id: 25188, data: n(this, f).audio.bitDepth } : null] }] })), n(this, f).subtitles && t.data.push({ id: 174, data: [{ id: 215, data: Ge }, { id: 29637, data: Ge }, { id: 131, data: Wi }, { id: 134, data: n(this, f).subtitles.codec }, n(this, ie)] });
};
Je = /* @__PURE__ */ new WeakSet();
Ct = function() {
  let t = { id: 408125543, size: n(this, f).streaming ? -1 : Jt, data: [n(this, f).streaming ? null : n(this, E), n(this, Be), n(this, le)] };
  if (p(this, Te, t), n(this, l).writeEBML(t), n(this, l) instanceof Ae && n(this, l).target.options.onHeader) {
    let { data: e, start: i } = n(this, l).getTrackedWrites();
    n(this, l).target.options.onHeader(e, i);
  }
};
Tt = /* @__PURE__ */ new WeakSet();
ai = function() {
  p(this, ne, { id: 475249515, data: [] });
};
V = /* @__PURE__ */ new WeakSet();
Z = function() {
  n(this, l) instanceof mt && n(this, l).flush();
};
O = /* @__PURE__ */ new WeakSet();
Q = function() {
  return n(this, l).dataOffsets.get(n(this, Te));
};
ct = /* @__PURE__ */ new WeakSet();
oi = function(t) {
  if (t.decoderConfig) {
    if (t.decoderConfig.colorSpace) {
      let e = t.decoderConfig.colorSpace;
      if (p(this, Ee, e), n(this, j).data = [{ id: 21937, data: { rgb: 1, bt709: 1, bt470bg: 5, smpte170m: 6 }[e.matrix] }, { id: 21946, data: { bt709: 1, smpte170m: 6, "iec61966-2-1": 13 }[e.transfer] }, { id: 21947, data: { bt709: 1, bt470bg: 5, smpte170m: 6 }[e.primaries] }, { id: 21945, data: [1, 2][Number(e.fullRange)] }], !n(this, f).streaming) {
        let i = n(this, l).pos;
        n(this, l).seek(n(this, l).offsets.get(n(this, j))), n(this, l).writeEBML(n(this, j)), n(this, l).seek(i);
      }
    }
    t.decoderConfig.description && (n(this, f).streaming ? p(this, ce, d(this, oe, Re).call(this, t.decoderConfig.description)) : d(this, ke, je).call(this, n(this, ce), t.decoderConfig.description));
  }
};
ht = /* @__PURE__ */ new WeakSet();
di = function(t) {
  if (t.type !== "key" || !n(this, Ee)) return;
  let e = 0;
  if (K(t.data, 0, 2) !== 2) return;
  e += 2;
  let i = (K(t.data, e + 1, e + 2) << 1) + K(t.data, e + 0, e + 1);
  e += 2, i === 3 && e++;
  let r = K(t.data, e + 0, e + 1);
  if (e++, r) return;
  let s = K(t.data, e + 0, e + 1);
  if (e++, s !== 0) return;
  e += 2;
  let a = K(t.data, e + 0, e + 24);
  if (e += 24, a !== 4817730) return;
  i >= 2 && e++;
  let c = { rgb: 7, bt709: 2, bt470bg: 1, smpte170m: 3 }[n(this, Ee).matrix];
  Ti(t.data, e + 0, e + 3, c);
};
fe = /* @__PURE__ */ new WeakSet();
Fe = function() {
  let t = Math.min(n(this, f).video ? n(this, se) : 1 / 0, n(this, f).audio ? n(this, ae) : 1 / 0), e = n(this, H);
  for (; e.length > 0 && e[0].timestamp <= t; ) d(this, C, B).call(this, e.shift(), !n(this, f).video && !n(this, f).audio);
};
pe = /* @__PURE__ */ new WeakSet();
Ne = function(t, e, i, r, s, a) {
  let c = d(this, Bt, li).call(this, i, r);
  return { data: t, additions: a, type: e, timestamp: c, duration: s, trackNumber: r };
};
Bt = /* @__PURE__ */ new WeakSet();
li = function(t, e) {
  let i = e === de ? n(this, se) : e === Ce ? n(this, ae) : n(this, qe);
  if (e !== Ge) {
    let r = e === de ? n(this, ve) : n(this, ye);
    if (n(this, f).firstTimestampBehavior === "strict" && i === -1 && t !== 0) throw new Error(`The first chunk for your media track must have a timestamp of 0 (received ${t}). Non-zero first timestamps are often caused by directly piping frames or audio data from a MediaStreamTrack into the encoder. Their timestamps are typically relative to the age of the document, which is probably what you want.

If you want to offset all timestamps of a track such that the first one is zero, set firstTimestampBehavior: 'offset' in the options.
If you want to allow non-zero first timestamps, set firstTimestampBehavior: 'permissive'.
`);
    n(this, f).firstTimestampBehavior === "offset" && (t -= r);
  }
  if (t < i) throw new Error(`Timestamps must be monotonically increasing (went from ${i} to ${t}).`);
  if (t < 0) throw new Error(`Timestamps must be non-negative (received ${t}).`);
  return t;
};
C = /* @__PURE__ */ new WeakSet();
B = function(t, e) {
  n(this, f).streaming && !n(this, le) && (d(this, Xe, xt).call(this), d(this, Je, Ct).call(this));
  let i = Math.floor(t.timestamp / 1e3), r = i - n(this, Ie), s = e && t.type === "key" && r >= 1e3, a = r >= Li;
  if ((!n(this, T) || s || a) && (d(this, Et, ci).call(this, i), r = 0), r < 0) return;
  let c = new Uint8Array(4), u = new DataView(c.buffer);
  if (u.setUint8(0, 128 | t.trackNumber), u.setInt16(1, r, false), t.duration === void 0 && !t.additions) {
    u.setUint8(3, +(t.type === "key") << 7);
    let b = { id: 163, data: [c, t.data] };
    n(this, l).writeEBML(b);
  } else {
    let b = Math.floor(t.duration / 1e3), y = { id: 160, data: [{ id: 161, data: [c, t.data] }, t.duration !== void 0 ? { id: 155, data: b } : null, t.additions ? { id: 30113, data: t.additions } : null] };
    n(this, l).writeEBML(y);
  }
  p(this, re, Math.max(n(this, re), i));
};
oe = /* @__PURE__ */ new WeakSet();
Re = function(t) {
  return { id: 25506, size: 4, data: new Uint8Array(t) };
};
ke = /* @__PURE__ */ new WeakSet();
je = function(t, e) {
  let i = n(this, l).pos;
  n(this, l).seek(n(this, l).offsets.get(t));
  let r = 6 + e.byteLength, s = be - r;
  if (s < 0) {
    let a = e.byteLength + s;
    e instanceof ArrayBuffer ? e = e.slice(0, a) : e = e.buffer.slice(0, a), s = 0;
  }
  t = [d(this, oe, Re).call(this, e), { id: 236, size: 4, data: new Uint8Array(s) }], n(this, l).writeEBML(t), n(this, l).seek(i);
};
Et = /* @__PURE__ */ new WeakSet();
ci = function(t) {
  n(this, T) && d(this, Ke, Rt).call(this), n(this, l) instanceof Ae && n(this, l).target.options.onCluster && n(this, l).startTrackingWrites(), p(this, T, { id: 524531317, size: n(this, f).streaming ? -1 : Zt, data: [{ id: 231, data: t }] }), n(this, l).writeEBML(n(this, T)), p(this, Ie, t);
  let e = n(this, l).offsets.get(n(this, T)) - n(this, O, Q);
  n(this, ne).data.push({ id: 187, data: [{ id: 179, data: t }, n(this, f).video ? { id: 183, data: [{ id: 247, data: de }, { id: 241, data: e }] } : null, n(this, f).audio ? { id: 183, data: [{ id: 247, data: Ce }, { id: 241, data: e }] } : null] });
};
Ke = /* @__PURE__ */ new WeakSet();
Rt = function() {
  if (!n(this, f).streaming) {
    let t = n(this, l).pos - n(this, l).dataOffsets.get(n(this, T)), e = n(this, l).pos;
    n(this, l).seek(n(this, l).offsets.get(n(this, T)) + 4), n(this, l).writeEBMLVarInt(t, Zt), n(this, l).seek(e);
  }
  if (n(this, l) instanceof Ae && n(this, l).target.options.onCluster) {
    let { data: t, start: e } = n(this, l).getTrackedWrites();
    n(this, l).target.options.onCluster(t, e, n(this, Ie));
  }
};
_e = /* @__PURE__ */ new WeakSet();
He = function() {
  if (n(this, Se)) throw new Error("Cannot add new video or audio chunks after the file has been finalized.");
};
new TextEncoder();
class Pi {
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
    const s = e.fps * e.duration;
    console.log(`Starting recording: ${s} frames @ ${e.fps}fps (VP9)`);
    const a = new zi({ target: new Vt(), video: { codec: "V_VP9", width: this.canvas.width, height: this.canvas.height, frameRate: e.fps } }), c = new VideoEncoder({ output: (u, b) => a.addVideoChunk(u, b), error: (u) => console.error("VideoEncoder Error:", u) });
    c.configure({ codec: "vp09.00.10.08", width: this.canvas.width, height: this.canvas.height, bitrate: 12e6 });
    try {
      await this.renderAndEncode(s, e, c, i), await c.flush(), a.finalize();
      const { buffer: u } = a.target, b = new Blob([u], { type: "video/webm" }), y = URL.createObjectURL(b);
      r(y);
    } catch (u) {
      throw console.error("Recording failed:", u), u;
    } finally {
      this.isRecording = false;
    }
  }
  async renderAndEncode(e, i, r, s) {
    for (let a = 0; a < e; a++) {
      s(a, e), await new Promise((b) => setTimeout(b, 0));
      const c = a / i.fps;
      this.worldBridge.update(c), await this.updateSceneBuffers(), await this.renderFrame(i.spp, i.batch), r.encodeQueueSize > 5 && await r.flush();
      const u = new VideoFrame(this.canvas, { timestamp: a * 1e6 / i.fps, duration: 1e6 / i.fps });
      r.encode(u, { keyFrame: a % i.fps === 0 }), u.close();
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
const Fi = { iceServers: [{ urls: "stun:stun.l.google.com:19302" }] };
class zt {
  constructor(e, i) {
    __publicField(this, "pc");
    __publicField(this, "dc", null);
    __publicField(this, "remoteId");
    __publicField(this, "sendSignal");
    __publicField(this, "receiveBuffer", new Uint8Array(0));
    __publicField(this, "receivedBytes", 0);
    __publicField(this, "sceneMeta", null);
    __publicField(this, "onSceneReceived", null);
    __publicField(this, "onDataChannelOpen", null);
    __publicField(this, "onAckReceived", null);
    this.remoteId = e, this.sendSignal = i, this.pc = new RTCPeerConnection(Fi), this.pc.onicecandidate = (r) => {
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
    typeof e == "string" ? s = new TextEncoder().encode(e) : s = new Uint8Array(e), console.log(`[RTC] Sending Scene: ${i}, ${s.byteLength} bytes`);
    const a = { type: "SCENE_INIT", totalBytes: s.byteLength, config: { ...r, fileType: i } };
    this.sendData(a);
    const c = 16 * 1024;
    let u = 0;
    const b = () => new Promise((y) => {
      const z = setInterval(() => {
        this.dc && this.dc.bufferedAmount < 65536 && (clearInterval(z), y());
      }, 5);
    });
    for (; u < s.byteLength; ) {
      this.dc.bufferedAmount > 256 * 1024 && await b();
      const y = Math.min(u + c, s.byteLength);
      this.dc.send(s.subarray(u, y)), u = y, u % (c * 5) === 0 && await new Promise((z) => setTimeout(z, 0));
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
    e.type === "SCENE_INIT" ? (console.log(`[RTC] Receiving Scene: ${e.config.fileType}, ${e.totalBytes} bytes`), console.log("[RTC] Config:", e.config), this.sceneMeta = { config: e.config, totalBytes: e.totalBytes }, this.receiveBuffer = new Uint8Array(e.totalBytes), this.receivedBytes = 0) : e.type === "SCENE_ACK" && (console.log(`[RTC] Scene ACK: ${e.receivedBytes} bytes`), this.onAckReceived && this.onAckReceived(e.receivedBytes));
  }
  handleBinaryChunk(e) {
    if (!this.sceneMeta) return;
    const i = new Uint8Array(e);
    if (this.receiveBuffer.set(i, this.receivedBytes), this.receivedBytes += i.byteLength, this.receivedBytes >= this.sceneMeta.totalBytes) {
      console.log("[RTC] Scene Download Complete!");
      let r;
      this.sceneMeta.config.fileType === "obj" ? r = new TextDecoder().decode(this.receiveBuffer) : r = this.receiveBuffer.buffer, this.onSceneReceived && this.onSceneReceived(r, this.sceneMeta.config), this.sceneMeta = null;
    }
  }
  sendData(e) {
    var _a;
    ((_a = this.dc) == null ? void 0 : _a.readyState) === "open" && this.dc.send(JSON.stringify(e));
  }
  sendAck(e) {
    this.sendData({ type: "SCENE_ACK", receivedBytes: e });
  }
}
class Ni {
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
  }
  connect(e) {
    var _a;
    this.ws || (this.myRole = e, (_a = this.onStatusChange) == null ? void 0 : _a.call(this, `Connecting as ${e.toUpperCase()}...`), this.ws = new WebSocket(m.signalingServerUrl), this.ws.onopen = () => {
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
        const i = new zt(e.workerId, (r) => this.sendSignal(r));
        this.workers.set(e.workerId, i), i.onDataChannelOpen = () => {
          var _a2;
          console.log(`[Host] Open for ${e.workerId}`), i.sendData({ type: "HELLO", msg: "Hello from Host!" }), (_a2 = this.onWorkerJoined) == null ? void 0 : _a2.call(this, e.workerId);
        }, i.onAckReceived = (r) => {
          console.log(`Worker ${e.workerId} ACK: ${r}`);
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
        e.fromId && (this.hostClient = new zt(e.fromId, (i) => this.sendSignal(i)), await this.hostClient.handleOffer(e.sdp), (_a = this.onStatusChange) == null ? void 0 : _a.call(this, "Connected to Host!"), (_b = this.onHostConnected) == null ? void 0 : _b.call(this), this.hostClient.onDataChannelOpen = () => {
          var _a2, _b2;
          (_a2 = this.hostClient) == null ? void 0 : _a2.sendData({ type: "HELLO", msg: "Hello from Worker!" }), (_b2 = this.onHostHello) == null ? void 0 : _b2.call(this);
        }, this.hostClient.onSceneReceived = (i, r) => {
          var _a2, _b2;
          (_a2 = this.onSceneReceived) == null ? void 0 : _a2.call(this, i, r);
          const s = typeof i == "string" ? i.length : i.byteLength;
          (_b2 = this.hostClient) == null ? void 0 : _b2.sendAck(s);
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
}
let R = false, A = null, $ = null;
const _ = new xi(), v = new fi(_.canvas), g = new ki(), ut = new Pi(v, g, _.canvas), Y = new Ni();
let G = 0, ft = 0, Ue = 0, Pt = performance.now();
const Hi = () => {
  const t = parseInt(_.inputDepth.value, 10) || m.defaultDepth, e = parseInt(_.inputSPP.value, 10) || m.defaultSPP;
  v.buildPipeline(t, e);
}, At = () => {
  const { width: t, height: e } = _.getRenderConfig();
  v.updateScreenSize(t, e), g.hasWorld && (g.updateCamera(t, e), v.updateSceneUniforms(g.cameraData, 0)), v.recreateBindGroup(), v.resetAccumulation(), G = 0, ft = 0;
}, Ve = async (t, e = true) => {
  R = false, console.log(`Loading Scene: ${t}...`);
  let i, r;
  t === "viewer" && A && ($ === "obj" ? i = A : $ === "glb" && (r = new Uint8Array(A))), g.loadScene(t, i, r), g.printStats(), await v.loadTexturesFromWorld(g), await Vi(), At(), _.updateAnimList(g.getAnimationList()), e && (R = true, _.updateRenderButton(true));
}, Vi = async () => {
  v.updateCombinedGeometry(g.vertices, g.normals, g.uvs), v.updateCombinedBVH(g.tlas, g.blas), v.updateBuffer("index", g.indices), v.updateBuffer("attr", g.attributes), v.updateBuffer("instance", g.instances);
}, It = () => {
  if (ut.recording || (requestAnimationFrame(It), !R || !g.hasWorld)) return;
  let t = parseInt(_.inputUpdateInterval.value, 10) || 0;
  if (t < 0 && (t = 0), t > 0 && G >= t) {
    g.update(ft / t / 60);
    let i = false;
    i || (i = v.updateCombinedBVH(g.tlas, g.blas)), i || (i = v.updateBuffer("instance", g.instances)), i || (i = v.updateCombinedGeometry(g.vertices, g.normals, g.uvs)), i || (i = v.updateBuffer("index", g.indices)), i || (i = v.updateBuffer("attr", g.attributes)), g.updateCamera(_.canvas.width, _.canvas.height), v.updateSceneUniforms(g.cameraData, 0), i && v.recreateBindGroup(), v.resetAccumulation(), G = 0;
  }
  G++, Ue++, ft++, v.render(G);
  const e = performance.now();
  e - Pt >= 1e3 && (_.updateStats(Ue, 1e3 / Ue, G), Ue = 0, Pt = e);
}, Oi = () => {
  _.onRenderStart = () => {
    R = true;
  }, _.onRenderStop = () => {
    R = false;
  }, _.onSceneSelect = (t) => Ve(t, false), _.onResolutionChange = At, _.onRecompile = (t, e) => {
    R = false, v.buildPipeline(t, e), v.recreateBindGroup(), v.resetAccumulation(), G = 0, R = true;
  }, _.onFileSelect = async (t) => {
    var _a;
    ((_a = t.name.split(".").pop()) == null ? void 0 : _a.toLowerCase()) === "obj" ? (A = await t.text(), $ = "obj") : (A = await t.arrayBuffer(), $ = "glb"), _.sceneSelect.value = "viewer", Ve("viewer", false);
  }, _.onAnimSelect = (t) => g.setAnimation(t), _.onRecordStart = async () => {
    if (ut.recording) return;
    R = false, _.setRecordingState(true);
    const t = _.getRenderConfig();
    try {
      await ut.record(t, (e, i) => _.setRecordingState(true, `Rec: ${e}/${i} (${Math.round(e / i * 100)}%)`), (e) => {
        const i = document.createElement("a");
        i.href = e, i.download = `raytrace_${Date.now()}.webm`, i.click(), URL.revokeObjectURL(e);
      });
    } catch {
      alert("Recording failed.");
    } finally {
      _.setRecordingState(false), R = true, _.updateRenderButton(true), requestAnimationFrame(It);
    }
  }, _.onConnectHost = () => Y.connect("host"), _.onConnectWorker = () => Y.connect("worker"), _.onSendScene = async () => {
    if (!A || !$) {
      alert("No scene loaded!");
      return;
    }
    _.setSendSceneText("Sending..."), _.setSendSceneEnabled(false);
    const t = _.getRenderConfig();
    await Y.broadcastScene(A, $, t), _.setSendSceneText("Send Scene"), _.setSendSceneEnabled(true);
  }, Y.onStatusChange = (t) => _.setStatus(`Status: ${t}`), Y.onWorkerJoined = (t) => _.setSendSceneEnabled(true), Y.onSceneReceived = async (t, e) => {
    console.log("Scene received successfully."), _.setRenderConfig(e), $ = e.fileType, e.fileType, A = t, _.sceneSelect.value = "viewer", await Ve("viewer", false), e.anim !== void 0 && (_.animSelect.value = e.anim.toString(), g.setAnimation(e.anim));
  };
};
async function $i() {
  try {
    await v.init(), await g.initWasm();
  } catch (t) {
    alert("Init failed: " + t);
    return;
  }
  Oi(), Hi(), At(), Ve("cornell", false), requestAnimationFrame(It);
}
$i().catch(console.error);
