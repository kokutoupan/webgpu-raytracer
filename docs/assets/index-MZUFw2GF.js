var __defProp = Object.defineProperty;
var __defNormalProp = (obj, key, value) => key in obj ? __defProp(obj, key, { enumerable: true, configurable: true, writable: true, value }) : obj[key] = value;
var __publicField = (obj, key, value) => __defNormalProp(obj, typeof key !== "symbol" ? key + "" : key, value);
(function() {
  const t = document.createElement("link").relList;
  if (t && t.supports && t.supports("modulepreload")) return;
  for (const s of document.querySelectorAll('link[rel="modulepreload"]')) r(s);
  new MutationObserver((s) => {
    for (const o of s) if (o.type === "childList") for (const c of o.addedNodes) c.tagName === "LINK" && c.rel === "modulepreload" && r(c);
  }).observe(document, { childList: true, subtree: true });
  function n(s) {
    const o = {};
    return s.integrity && (o.integrity = s.integrity), s.referrerPolicy && (o.referrerPolicy = s.referrerPolicy), s.crossOrigin === "use-credentials" ? o.credentials = "include" : s.crossOrigin === "anonymous" ? o.credentials = "omit" : o.credentials = "same-origin", o;
  }
  function r(s) {
    if (s.ep) return;
    s.ep = true;
    const o = n(s);
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
    __publicField(this, "vertexCount", 0);
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
  buildPipeline(t, n) {
    let r = Bi;
    r = r.replace(/const\s+MAX_DEPTH\s*=\s*\d+u;/, `const MAX_DEPTH = ${t}u;`), r = r.replace(/const\s+SPP\s*=\s*\d+u;/, `const SPP = ${n}u;`);
    const s = this.device.createShaderModule({ label: "RayTracing", code: r });
    this.pipeline = this.device.createComputePipeline({ label: "Main Pipeline", layout: "auto", compute: { module: s, entryPoint: "main" } }), this.bindGroupLayout = this.pipeline.getBindGroupLayout(0);
  }
  updateScreenSize(t, n) {
    this.canvas.width = t, this.canvas.height = n, this.renderTarget && this.renderTarget.destroy(), this.renderTarget = this.device.createTexture({ size: [t, n], format: "rgba8unorm", usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC }), this.renderTargetView = this.renderTarget.createView(), this.bufferSize = t * n * 16, this.accumulateBuffer && this.accumulateBuffer.destroy(), this.accumulateBuffer = this.device.createBuffer({ size: this.bufferSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  }
  resetAccumulation() {
    this.accumulateBuffer && this.device.queue.writeBuffer(this.accumulateBuffer, 0, new Float32Array(this.bufferSize / 4));
  }
  async loadTexturesFromWorld(t) {
    const n = t.textureCount;
    if (n === 0) {
      this.createDefaultTexture();
      return;
    }
    console.log(`Loading ${n} textures...`);
    const r = [];
    for (let s = 0; s < n; s++) {
      const o = t.getTexture(s);
      if (o) try {
        const c = new Blob([o]), m = await createImageBitmap(c, { resizeWidth: 1024, resizeHeight: 1024 });
        r.push(m);
      } catch (c) {
        console.warn(`Failed tex ${s}`, c), r.push(await this.createFallbackBitmap());
      }
      else r.push(await this.createFallbackBitmap());
    }
    this.texture && this.texture.destroy(), this.texture = this.device.createTexture({ size: [1024, 1024, r.length], format: "rgba8unorm", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT });
    for (let s = 0; s < r.length; s++) this.device.queue.copyExternalImageToTexture({ source: r[s] }, { texture: this.texture, origin: [0, 0, s] }, [1024, 1024]);
  }
  async createFallbackBitmap() {
    const t = document.createElement("canvas");
    t.width = 1024, t.height = 1024;
    const n = t.getContext("2d");
    return n.fillStyle = "white", n.fillRect(0, 0, 1024, 1024), await createImageBitmap(t);
  }
  ensureBuffer(t, n, r) {
    if (t && t.size >= n) return t;
    t && t.destroy();
    let s = Math.ceil(n * 1.5);
    return s = s + 3 & -4, s = Math.max(s, 4), this.device.createBuffer({ label: r, size: s, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  }
  updateBuffer(t, n) {
    const r = n.byteLength;
    let s = false, o;
    return t === "index" ? ((!this.indexBuffer || this.indexBuffer.size < r) && (s = true), this.indexBuffer = this.ensureBuffer(this.indexBuffer, r, "IndexBuffer"), o = this.indexBuffer) : t === "attr" ? ((!this.attrBuffer || this.attrBuffer.size < r) && (s = true), this.attrBuffer = this.ensureBuffer(this.attrBuffer, r, "AttrBuffer"), o = this.attrBuffer) : ((!this.instanceBuffer || this.instanceBuffer.size < r) && (s = true), this.instanceBuffer = this.ensureBuffer(this.instanceBuffer, r, "InstanceBuffer"), o = this.instanceBuffer), this.device.queue.writeBuffer(o, 0, n, 0, n.length), s;
  }
  updateCombinedGeometry(t, n, r) {
    const s = t.byteLength + n.byteLength + r.byteLength;
    let o = false;
    (!this.geometryBuffer || this.geometryBuffer.size < s) && (o = true);
    const c = t.length / 4;
    this.vertexCount = c, this.geometryBuffer = this.ensureBuffer(this.geometryBuffer, s, "GeometryBuffer"), !(r.length >= c * 2) && c > 0 && console.warn(`UV buffer mismatch: V=${c}, UV=${r.length / 2}. Filling 0.`);
    const p = t.length, y = n.length, _e = r.length, _ = p + y + _e, w = new Float32Array(_);
    return w.set(t, 0), w.set(n, p), w.set(r, p + y), this.device.queue.writeBuffer(this.geometryBuffer, 0, w), o;
  }
  updateCombinedBVH(t, n) {
    const r = t.byteLength, s = n.byteLength, o = r + s;
    let c = false;
    return (!this.nodesBuffer || this.nodesBuffer.size < o) && (c = true), this.nodesBuffer = this.ensureBuffer(this.nodesBuffer, o, "NodesBuffer"), this.device.queue.writeBuffer(this.nodesBuffer, 0, t), this.device.queue.writeBuffer(this.nodesBuffer, r, n), this.blasOffset = t.length / 8, c;
  }
  updateSceneUniforms(t, n) {
    if (!this.sceneUniformBuffer) return;
    this.device.queue.writeBuffer(this.sceneUniformBuffer, 0, t);
    const r = new Uint32Array([n, this.blasOffset, this.vertexCount, 0]);
    this.device.queue.writeBuffer(this.sceneUniformBuffer, 96, r);
  }
  recreateBindGroup() {
    !this.renderTargetView || !this.accumulateBuffer || !this.geometryBuffer || !this.nodesBuffer || !this.sceneUniformBuffer || (this.bindGroup = this.device.createBindGroup({ layout: this.bindGroupLayout, entries: [{ binding: 0, resource: this.renderTargetView }, { binding: 1, resource: { buffer: this.accumulateBuffer } }, { binding: 2, resource: { buffer: this.sceneUniformBuffer } }, { binding: 3, resource: { buffer: this.geometryBuffer } }, { binding: 4, resource: { buffer: this.indexBuffer } }, { binding: 5, resource: { buffer: this.attrBuffer } }, { binding: 6, resource: { buffer: this.nodesBuffer } }, { binding: 7, resource: { buffer: this.instanceBuffer } }, { binding: 8, resource: this.texture.createView({ dimension: "2d-array" }) }, { binding: 9, resource: this.sampler }] }));
  }
  render(t) {
    if (!this.bindGroup) return;
    const n = new Uint32Array([t]);
    this.device.queue.writeBuffer(this.sceneUniformBuffer, 96, n);
    const r = Math.ceil(this.canvas.width / 8), s = Math.ceil(this.canvas.height / 8), o = this.device.createCommandEncoder(), c = o.beginComputePass();
    c.setPipeline(this.pipeline), c.setBindGroup(0, this.bindGroup), c.dispatchWorkgroups(r, s), c.end(), o.copyTextureToTexture({ texture: this.renderTarget }, { texture: this.context.getCurrentTexture() }, { width: this.canvas.width, height: this.canvas.height, depthOrArrayLayers: 1 }), this.device.queue.submit([o.finish()]);
  }
}
let u;
function Ei(e) {
  const t = u.__externref_table_alloc();
  return u.__wbindgen_externrefs.set(t, e), t;
}
function Ci(e, t) {
  return e = e >>> 0, ne().subarray(e / 1, e / 1 + t);
}
let G = null;
function Gt() {
  return (G === null || G.buffer.detached === true || G.buffer.detached === void 0 && G.buffer !== u.memory.buffer) && (G = new DataView(u.memory.buffer)), G;
}
function Ge(e, t) {
  return e = e >>> 0, Ui(e, t);
}
let be = null;
function ne() {
  return (be === null || be.byteLength === 0) && (be = new Uint8Array(u.memory.buffer)), be;
}
function Mi(e, t) {
  try {
    return e.apply(this, t);
  } catch (n) {
    const r = Ei(n);
    u.__wbindgen_exn_store(r);
  }
}
function Ht(e) {
  return e == null;
}
function $t(e, t) {
  const n = t(e.length * 1, 1) >>> 0;
  return ne().set(e, n / 1), P = e.length, n;
}
function ct(e, t, n) {
  if (n === void 0) {
    const m = Te.encode(e), p = t(m.length, 1) >>> 0;
    return ne().subarray(p, p + m.length).set(m), P = m.length, p;
  }
  let r = e.length, s = t(r, 1) >>> 0;
  const o = ne();
  let c = 0;
  for (; c < r; c++) {
    const m = e.charCodeAt(c);
    if (m > 127) break;
    o[s + c] = m;
  }
  if (c !== r) {
    c !== 0 && (e = e.slice(c)), s = n(s, r, r = c + e.length * 3, 1) >>> 0;
    const m = ne().subarray(s + c, s + r), p = Te.encodeInto(e, m);
    c += p.written, s = n(s, r, c, 1) >>> 0;
  }
  return P = c, s;
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
  const n = Te.encode(e);
  return t.set(n), { read: e.length, written: n.length };
});
let P = 0;
typeof FinalizationRegistry > "u" || new FinalizationRegistry((e) => u.__wbg_renderbuffers_free(e >>> 0, 1));
const qt = typeof FinalizationRegistry > "u" ? { register: () => {
}, unregister: () => {
} } : new FinalizationRegistry((e) => u.__wbg_world_free(e >>> 0, 1));
class ut {
  __destroy_into_raw() {
    const t = this.__wbg_ptr;
    return this.__wbg_ptr = 0, qt.unregister(this), t;
  }
  free() {
    const t = this.__destroy_into_raw();
    u.__wbg_world_free(t, 0);
  }
  camera_ptr() {
    return u.world_camera_ptr(this.__wbg_ptr) >>> 0;
  }
  indices_len() {
    return u.world_indices_len(this.__wbg_ptr) >>> 0;
  }
  indices_ptr() {
    return u.world_indices_ptr(this.__wbg_ptr) >>> 0;
  }
  normals_len() {
    return u.world_normals_len(this.__wbg_ptr) >>> 0;
  }
  normals_ptr() {
    return u.world_normals_ptr(this.__wbg_ptr) >>> 0;
  }
  vertices_len() {
    return u.world_vertices_len(this.__wbg_ptr) >>> 0;
  }
  vertices_ptr() {
    return u.world_vertices_ptr(this.__wbg_ptr) >>> 0;
  }
  instances_len() {
    return u.world_instances_len(this.__wbg_ptr) >>> 0;
  }
  instances_ptr() {
    return u.world_instances_ptr(this.__wbg_ptr) >>> 0;
  }
  set_animation(t) {
    u.world_set_animation(this.__wbg_ptr, t);
  }
  update_camera(t, n) {
    u.world_update_camera(this.__wbg_ptr, t, n);
  }
  attributes_len() {
    return u.world_attributes_len(this.__wbg_ptr) >>> 0;
  }
  attributes_ptr() {
    return u.world_attributes_ptr(this.__wbg_ptr) >>> 0;
  }
  get_texture_ptr(t) {
    return u.world_get_texture_ptr(this.__wbg_ptr, t) >>> 0;
  }
  get_texture_size(t) {
    return u.world_get_texture_size(this.__wbg_ptr, t) >>> 0;
  }
  get_texture_count() {
    return u.world_get_texture_count(this.__wbg_ptr) >>> 0;
  }
  get_animation_name(t) {
    let n, r;
    try {
      const s = u.world_get_animation_name(this.__wbg_ptr, t);
      return n = s[0], r = s[1], Ge(s[0], s[1]);
    } finally {
      u.__wbindgen_free(n, r, 1);
    }
  }
  load_animation_glb(t) {
    const n = $t(t, u.__wbindgen_malloc), r = P;
    u.world_load_animation_glb(this.__wbg_ptr, n, r);
  }
  get_animation_count() {
    return u.world_get_animation_count(this.__wbg_ptr) >>> 0;
  }
  constructor(t, n, r) {
    const s = ct(t, u.__wbindgen_malloc, u.__wbindgen_realloc), o = P;
    var c = Ht(n) ? 0 : ct(n, u.__wbindgen_malloc, u.__wbindgen_realloc), m = P, p = Ht(r) ? 0 : $t(r, u.__wbindgen_malloc), y = P;
    const _e = u.world_new(s, o, c, m, p, y);
    return this.__wbg_ptr = _e >>> 0, qt.register(this, this.__wbg_ptr, this), this;
  }
  update(t) {
    u.world_update(this.__wbg_ptr, t);
  }
  uvs_len() {
    return u.world_uvs_len(this.__wbg_ptr) >>> 0;
  }
  uvs_ptr() {
    return u.world_uvs_ptr(this.__wbg_ptr) >>> 0;
  }
  blas_len() {
    return u.world_blas_len(this.__wbg_ptr) >>> 0;
  }
  blas_ptr() {
    return u.world_blas_ptr(this.__wbg_ptr) >>> 0;
  }
  tlas_len() {
    return u.world_tlas_len(this.__wbg_ptr) >>> 0;
  }
  tlas_ptr() {
    return u.world_tlas_ptr(this.__wbg_ptr) >>> 0;
  }
}
Symbol.dispose && (ut.prototype[Symbol.dispose] = ut.prototype.free);
const Ai = /* @__PURE__ */ new Set(["basic", "cors", "default"]);
async function Wi(e, t) {
  if (typeof Response == "function" && e instanceof Response) {
    if (typeof WebAssembly.instantiateStreaming == "function") try {
      return await WebAssembly.instantiateStreaming(e, t);
    } catch (r) {
      if (e.ok && Ai.has(e.type) && e.headers.get("Content-Type") !== "application/wasm") console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", r);
      else throw r;
    }
    const n = await e.arrayBuffer();
    return await WebAssembly.instantiate(n, t);
  } else {
    const n = await WebAssembly.instantiate(e, t);
    return n instanceof WebAssembly.Instance ? { instance: n, module: e } : n;
  }
}
function Li() {
  const e = {};
  return e.wbg = {}, e.wbg.__wbg___wbindgen_throw_dd24417ed36fc46e = function(t, n) {
    throw new Error(Ge(t, n));
  }, e.wbg.__wbg_error_7534b8e9a36f1ab4 = function(t, n) {
    let r, s;
    try {
      r = t, s = n, console.error(Ge(t, n));
    } finally {
      u.__wbindgen_free(r, s, 1);
    }
  }, e.wbg.__wbg_getRandomValues_1c61fac11405ffdc = function() {
    return Mi(function(t, n) {
      globalThis.crypto.getRandomValues(Ci(t, n));
    }, arguments);
  }, e.wbg.__wbg_log_1d990106d99dacb7 = function(t) {
    console.log(t);
  }, e.wbg.__wbg_new_8a6f238a6ece86ea = function() {
    return new Error();
  }, e.wbg.__wbg_stack_0ed75d68575b0f3c = function(t, n) {
    const r = n.stack, s = ct(r, u.__wbindgen_malloc, u.__wbindgen_realloc), o = P;
    Gt().setInt32(t + 4, o, true), Gt().setInt32(t + 0, s, true);
  }, e.wbg.__wbindgen_cast_2241b6af4c4b2941 = function(t, n) {
    return Ge(t, n);
  }, e.wbg.__wbindgen_init_externref_table = function() {
    const t = u.__wbindgen_externrefs, n = t.grow(4);
    t.set(0, void 0), t.set(n + 0, void 0), t.set(n + 1, null), t.set(n + 2, true), t.set(n + 3, false);
  }, e;
}
function zi(e, t) {
  return u = e.exports, Qt.__wbindgen_wasm_module = t, G = null, be = null, u.__wbindgen_start(), u;
}
async function Qt(e) {
  if (u !== void 0) return u;
  typeof e < "u" && (Object.getPrototypeOf(e) === Object.prototype ? { module_or_path: e } = e : console.warn("using deprecated parameters for the initialization function; pass a single object instead")), typeof e > "u" && (e = new URL("/webgpu-raytracer/assets/rust_shader_tools_bg-CC5HVMsp.wasm", import.meta.url));
  const t = Li();
  (typeof e == "string" || typeof Request == "function" && e instanceof Request || typeof URL == "function" && e instanceof URL) && (e = fetch(e));
  const { instance: n, module: r } = await Wi(await e, t);
  return zi(n, r);
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
  loadScene(t, n, r) {
    this.world && this.world.free(), this.world = new ut(t, n, r);
  }
  update(t) {
    var _a;
    (_a = this.world) == null ? void 0 : _a.update(t);
  }
  updateCamera(t, n) {
    var _a;
    (_a = this.world) == null ? void 0 : _a.update_camera(t, n);
  }
  loadAnimation(t) {
    var _a;
    (_a = this.world) == null ? void 0 : _a.load_animation_glb(t);
  }
  getAnimationList() {
    if (!this.world) return [];
    const t = this.world.get_animation_count(), n = [];
    for (let r = 0; r < t; r++) n.push(this.world.get_animation_name(r));
    return n;
  }
  setAnimation(t) {
    var _a;
    (_a = this.world) == null ? void 0 : _a.set_animation(t);
  }
  getF32(t, n) {
    return new Float32Array(this.wasmMemory.buffer, t, n);
  }
  getU32(t, n) {
    return new Uint32Array(this.wasmMemory.buffer, t, n);
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
    const n = this.world.get_texture_ptr(t), r = this.world.get_texture_size(t);
    return !n || r === 0 ? null : new Uint8Array(this.wasmMemory.buffer, n, r).slice();
  }
  get hasWorld() {
    return !!this.world;
  }
  printStats() {
    this.world && console.log(`Scene Stats: V=${this.vertices.length / 4}, Tri=${this.indices.length / 3}, BLAS=${this.blas.length / 8}, TLAS=${this.tlas.length / 8}`);
  }
}
var Tt = (e, t, n) => {
  if (!t.has(e)) throw TypeError("Cannot " + n);
}, i = (e, t, n) => (Tt(e, t, "read from private field"), n ? n.call(e) : t.get(e)), a = (e, t, n) => {
  if (t.has(e)) throw TypeError("Cannot add the same private member more than once");
  t instanceof WeakSet ? t.add(e) : t.set(e, n);
}, f = (e, t, n, r) => (Tt(e, t, "write to private field"), t.set(e, n), n), d = (e, t, n) => (Tt(e, t, "access private method"), n), Zt = class {
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
}, Z = (e, t, n) => {
  let r = 0;
  for (let s = t; s < n; s++) {
    let o = Math.floor(s / 8), c = e[o], m = 7 - (s & 7), p = (c & 1 << m) >> m;
    r <<= 1, r |= p;
  }
  return r;
}, Fi = (e, t, n, r) => {
  for (let s = t; s < n; s++) {
    let o = Math.floor(s / 8), c = e[o], m = 7 - (s & 7);
    c &= ~(1 << m), c |= (r & 1 << n - s - 1) >> n - s - 1 << m, e[o] = c;
  }
}, nt = class {
}, ei = class extends nt {
  constructor() {
    super(...arguments), this.buffer = null;
  }
}, ti = class extends nt {
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
}, Di = class extends nt {
  constructor(e, t) {
    if (super(), this.stream = e, this.options = t, !(e instanceof FileSystemWritableFileStream)) throw new TypeError("FileSystemWritableFileStreamTarget requires a FileSystemWritableFileStream instance.");
    if (t !== void 0 && typeof t != "object") throw new TypeError("FileSystemWritableFileStreamTarget's options, when provided, must be an object.");
    if (t && t.chunkSize !== void 0 && (!Number.isInteger(t.chunkSize) || t.chunkSize <= 0)) throw new TypeError("options.chunkSize, when provided, must be a positive integer");
  }
}, F, g, ht, ii, ft, ni, _t, ri, $e, mt, gt, si, ai = class {
  constructor() {
    a(this, ht), a(this, ft), a(this, _t), a(this, $e), a(this, gt), this.pos = 0, a(this, F, new Uint8Array(8)), a(this, g, new DataView(i(this, F).buffer)), this.offsets = /* @__PURE__ */ new WeakMap(), this.dataOffsets = /* @__PURE__ */ new WeakMap();
  }
  seek(e) {
    this.pos = e;
  }
  writeEBMLVarInt(e, t = Pi(e)) {
    let n = 0;
    switch (t) {
      case 1:
        i(this, g).setUint8(n++, 128 | e);
        break;
      case 2:
        i(this, g).setUint8(n++, 64 | e >> 8), i(this, g).setUint8(n++, e);
        break;
      case 3:
        i(this, g).setUint8(n++, 32 | e >> 16), i(this, g).setUint8(n++, e >> 8), i(this, g).setUint8(n++, e);
        break;
      case 4:
        i(this, g).setUint8(n++, 16 | e >> 24), i(this, g).setUint8(n++, e >> 16), i(this, g).setUint8(n++, e >> 8), i(this, g).setUint8(n++, e);
        break;
      case 5:
        i(this, g).setUint8(n++, 8 | e / 2 ** 32 & 7), i(this, g).setUint8(n++, e >> 24), i(this, g).setUint8(n++, e >> 16), i(this, g).setUint8(n++, e >> 8), i(this, g).setUint8(n++, e);
        break;
      case 6:
        i(this, g).setUint8(n++, 4 | e / 2 ** 40 & 3), i(this, g).setUint8(n++, e / 2 ** 32 | 0), i(this, g).setUint8(n++, e >> 24), i(this, g).setUint8(n++, e >> 16), i(this, g).setUint8(n++, e >> 8), i(this, g).setUint8(n++, e);
        break;
      default:
        throw new Error("Bad EBML VINT size " + t);
    }
    this.write(i(this, F).subarray(0, n));
  }
  writeEBML(e) {
    if (e !== null) if (e instanceof Uint8Array) this.write(e);
    else if (Array.isArray(e)) for (let t of e) this.writeEBML(t);
    else if (this.offsets.set(e, this.pos), d(this, $e, mt).call(this, e.id), Array.isArray(e.data)) {
      let t = this.pos, n = e.size === -1 ? 1 : e.size ?? 4;
      e.size === -1 ? d(this, ht, ii).call(this, 255) : this.seek(this.pos + n);
      let r = this.pos;
      if (this.dataOffsets.set(e, r), this.writeEBML(e.data), e.size !== -1) {
        let s = this.pos - r, o = this.pos;
        this.seek(t), this.writeEBMLVarInt(s, n), this.seek(o);
      }
    } else if (typeof e.data == "number") {
      let t = e.size ?? Jt(e.data);
      this.writeEBMLVarInt(t), d(this, $e, mt).call(this, e.data, t);
    } else typeof e.data == "string" ? (this.writeEBMLVarInt(e.data.length), d(this, gt, si).call(this, e.data)) : e.data instanceof Uint8Array ? (this.writeEBMLVarInt(e.data.byteLength, e.size), this.write(e.data)) : e.data instanceof Zt ? (this.writeEBMLVarInt(4), d(this, ft, ni).call(this, e.data.value)) : e.data instanceof Bt && (this.writeEBMLVarInt(8), d(this, _t, ri).call(this, e.data.value));
  }
};
F = /* @__PURE__ */ new WeakMap();
g = /* @__PURE__ */ new WeakMap();
ht = /* @__PURE__ */ new WeakSet();
ii = function(e) {
  i(this, g).setUint8(0, e), this.write(i(this, F).subarray(0, 1));
};
ft = /* @__PURE__ */ new WeakSet();
ni = function(e) {
  i(this, g).setFloat32(0, e, false), this.write(i(this, F).subarray(0, 4));
};
_t = /* @__PURE__ */ new WeakSet();
ri = function(e) {
  i(this, g).setFloat64(0, e, false), this.write(i(this, F));
};
$e = /* @__PURE__ */ new WeakSet();
mt = function(e, t = Jt(e)) {
  let n = 0;
  switch (t) {
    case 6:
      i(this, g).setUint8(n++, e / 2 ** 40 | 0);
    case 5:
      i(this, g).setUint8(n++, e / 2 ** 32 | 0);
    case 4:
      i(this, g).setUint8(n++, e >> 24);
    case 3:
      i(this, g).setUint8(n++, e >> 16);
    case 2:
      i(this, g).setUint8(n++, e >> 8);
    case 1:
      i(this, g).setUint8(n++, e);
      break;
    default:
      throw new Error("Bad UINT size " + t);
  }
  this.write(i(this, F).subarray(0, n));
};
gt = /* @__PURE__ */ new WeakSet();
si = function(e) {
  this.write(new Uint8Array(e.split("").map((t) => t.charCodeAt(0))));
};
var qe, X, Ae, je, wt, Vi = class extends ai {
  constructor(e) {
    super(), a(this, je), a(this, qe, void 0), a(this, X, new ArrayBuffer(2 ** 16)), a(this, Ae, new Uint8Array(i(this, X))), f(this, qe, e);
  }
  write(e) {
    d(this, je, wt).call(this, this.pos + e.byteLength), i(this, Ae).set(e, this.pos), this.pos += e.byteLength;
  }
  finalize() {
    d(this, je, wt).call(this, this.pos), i(this, qe).buffer = i(this, X).slice(0, this.pos);
  }
};
qe = /* @__PURE__ */ new WeakMap();
X = /* @__PURE__ */ new WeakMap();
Ae = /* @__PURE__ */ new WeakMap();
je = /* @__PURE__ */ new WeakSet();
wt = function(e) {
  let t = i(this, X).byteLength;
  for (; t < e; ) t *= 2;
  if (t === i(this, X).byteLength) return;
  let n = new ArrayBuffer(t), r = new Uint8Array(n);
  r.set(i(this, Ae), 0), f(this, X, n), f(this, Ae, r);
};
var J, B, S, H, Fe = class extends ai {
  constructor(e) {
    super(), this.target = e, a(this, J, false), a(this, B, void 0), a(this, S, void 0), a(this, H, void 0);
  }
  write(e) {
    if (!i(this, J)) return;
    let t = this.pos;
    if (t < i(this, S)) {
      if (t + e.byteLength <= i(this, S)) return;
      e = e.subarray(i(this, S) - t), t = 0;
    }
    let n = t + e.byteLength - i(this, S), r = i(this, B).byteLength;
    for (; r < n; ) r *= 2;
    if (r !== i(this, B).byteLength) {
      let s = new Uint8Array(r);
      s.set(i(this, B), 0), f(this, B, s);
    }
    i(this, B).set(e, t - i(this, S)), f(this, H, Math.max(i(this, H), t + e.byteLength));
  }
  startTrackingWrites() {
    f(this, J, true), f(this, B, new Uint8Array(2 ** 10)), f(this, S, this.pos), f(this, H, this.pos);
  }
  getTrackedWrites() {
    if (!i(this, J)) throw new Error("Can't get tracked writes since nothing was tracked.");
    let t = { data: i(this, B).subarray(0, i(this, H) - i(this, S)), start: i(this, S), end: i(this, H) };
    return f(this, B, void 0), f(this, J, false), t;
  }
};
J = /* @__PURE__ */ new WeakMap();
B = /* @__PURE__ */ new WeakMap();
S = /* @__PURE__ */ new WeakMap();
H = /* @__PURE__ */ new WeakMap();
var Ni = 2 ** 24, Oi = 2, $, re, Be, ve, A, v, Qe, pt, St, oi, Et, di, Se, Ze, Ct = class extends Fe {
  constructor(e, t) {
    var _a, _b;
    super(e), a(this, Qe), a(this, St), a(this, Et), a(this, Se), a(this, $, []), a(this, re, 0), a(this, Be, void 0), a(this, ve, void 0), a(this, A, void 0), a(this, v, []), f(this, Be, t), f(this, ve, ((_a = e.options) == null ? void 0 : _a.chunked) ?? false), f(this, A, ((_b = e.options) == null ? void 0 : _b.chunkSize) ?? Ni);
  }
  write(e) {
    super.write(e), i(this, $).push({ data: e.slice(), start: this.pos }), this.pos += e.byteLength;
  }
  flush() {
    var _a, _b;
    if (i(this, $).length === 0) return;
    let e = [], t = [...i(this, $)].sort((n, r) => n.start - r.start);
    e.push({ start: t[0].start, size: t[0].data.byteLength });
    for (let n = 1; n < t.length; n++) {
      let r = e[e.length - 1], s = t[n];
      s.start <= r.start + r.size ? r.size = Math.max(r.size, s.start + s.data.byteLength - r.start) : e.push({ start: s.start, size: s.data.byteLength });
    }
    for (let n of e) {
      n.data = new Uint8Array(n.size);
      for (let r of i(this, $)) n.start <= r.start && r.start < n.start + n.size && n.data.set(r.data, r.start - n.start);
      if (i(this, ve)) d(this, Qe, pt).call(this, n.data, n.start), d(this, Se, Ze).call(this);
      else {
        if (i(this, Be) && n.start < i(this, re)) throw new Error("Internal error: Monotonicity violation.");
        (_b = (_a = this.target.options).onData) == null ? void 0 : _b.call(_a, n.data, n.start), f(this, re, n.start + n.data.byteLength);
      }
    }
    i(this, $).length = 0;
  }
  finalize() {
    i(this, ve) && d(this, Se, Ze).call(this, true);
  }
};
$ = /* @__PURE__ */ new WeakMap();
re = /* @__PURE__ */ new WeakMap();
Be = /* @__PURE__ */ new WeakMap();
ve = /* @__PURE__ */ new WeakMap();
A = /* @__PURE__ */ new WeakMap();
v = /* @__PURE__ */ new WeakMap();
Qe = /* @__PURE__ */ new WeakSet();
pt = function(e, t) {
  let n = i(this, v).findIndex((m) => m.start <= t && t < m.start + i(this, A));
  n === -1 && (n = d(this, Et, di).call(this, t));
  let r = i(this, v)[n], s = t - r.start, o = e.subarray(0, Math.min(i(this, A) - s, e.byteLength));
  r.data.set(o, s);
  let c = { start: s, end: s + o.byteLength };
  if (d(this, St, oi).call(this, r, c), r.written[0].start === 0 && r.written[0].end === i(this, A) && (r.shouldFlush = true), i(this, v).length > Oi) {
    for (let m = 0; m < i(this, v).length - 1; m++) i(this, v)[m].shouldFlush = true;
    d(this, Se, Ze).call(this);
  }
  o.byteLength < e.byteLength && d(this, Qe, pt).call(this, e.subarray(o.byteLength), t + o.byteLength);
};
St = /* @__PURE__ */ new WeakSet();
oi = function(e, t) {
  let n = 0, r = e.written.length - 1, s = -1;
  for (; n <= r; ) {
    let o = Math.floor(n + (r - n + 1) / 2);
    e.written[o].start <= t.start ? (n = o + 1, s = o) : r = o - 1;
  }
  for (e.written.splice(s + 1, 0, t), (s === -1 || e.written[s].end < t.start) && s++; s < e.written.length - 1 && e.written[s].end >= e.written[s + 1].start; ) e.written[s].end = Math.max(e.written[s].end, e.written[s + 1].end), e.written.splice(s + 1, 1);
};
Et = /* @__PURE__ */ new WeakSet();
di = function(e) {
  let n = { start: Math.floor(e / i(this, A)) * i(this, A), data: new Uint8Array(i(this, A)), written: [], shouldFlush: false };
  return i(this, v).push(n), i(this, v).sort((r, s) => r.start - s.start), i(this, v).indexOf(n);
};
Se = /* @__PURE__ */ new WeakSet();
Ze = function(e = false) {
  var _a, _b;
  for (let t = 0; t < i(this, v).length; t++) {
    let n = i(this, v)[t];
    if (!(!n.shouldFlush && !e)) {
      for (let r of n.written) {
        if (i(this, Be) && n.start + r.start < i(this, re)) throw new Error("Internal error: Monotonicity violation.");
        (_b = (_a = this.target.options).onData) == null ? void 0 : _b.call(_a, n.data.subarray(r.start, r.end), n.start + r.start), f(this, re, n.start + r.end);
      }
      i(this, v).splice(t--, 1);
    }
  }
};
var Gi = class extends Ct {
  constructor(e, t) {
    var _a;
    super(new ti({ onData: (n, r) => e.stream.write({ type: "write", data: n, position: r }), chunked: true, chunkSize: (_a = e.options) == null ? void 0 : _a.chunkSize }), t);
  }
}, ue = 1, We = 2, Je = 3, Hi = 1, $i = 2, qi = 17, ji = 2 ** 15, Ee = 2 ** 13, jt = "https://github.com/Vanilagy/webm-muxer", li = 6, ci = 5, Yi = ["strict", "offset", "permissive"], h, l, Le, ze, U, he, ee, K, fe, D, se, ae, C, De, oe, z, R, q, Ce, Me, de, le, et, Re, Ie, bt, ui, vt, hi, Mt, fi, It, _i, Ut, mi, At, gi, Wt, wi, rt, Lt, st, zt, Rt, pi, j, te, Y, ie, yt, bi, xt, vi, ye, Ye, xe, Xe, Pt, yi, E, I, ce, Pe, Ue, tt, Ft, xi, it, Dt, ke, Ke, Xi = class {
  constructor(e) {
    a(this, bt), a(this, vt), a(this, Mt), a(this, It), a(this, Ut), a(this, At), a(this, Wt), a(this, rt), a(this, st), a(this, Rt), a(this, j), a(this, Y), a(this, yt), a(this, xt), a(this, ye), a(this, xe), a(this, Pt), a(this, E), a(this, ce), a(this, Ue), a(this, Ft), a(this, it), a(this, ke), a(this, h, void 0), a(this, l, void 0), a(this, Le, void 0), a(this, ze, void 0), a(this, U, void 0), a(this, he, void 0), a(this, ee, void 0), a(this, K, void 0), a(this, fe, void 0), a(this, D, void 0), a(this, se, void 0), a(this, ae, void 0), a(this, C, void 0), a(this, De, void 0), a(this, oe, 0), a(this, z, []), a(this, R, []), a(this, q, []), a(this, Ce, void 0), a(this, Me, void 0), a(this, de, -1), a(this, le, -1), a(this, et, -1), a(this, Re, void 0), a(this, Ie, false), d(this, bt, ui).call(this, e), f(this, h, { type: "webm", firstTimestampBehavior: "strict", ...e }), this.target = e.target;
    let t = !!i(this, h).streaming;
    if (e.target instanceof ei) f(this, l, new Vi(e.target));
    else if (e.target instanceof ti) f(this, l, new Ct(e.target, t));
    else if (e.target instanceof Di) f(this, l, new Gi(e.target, t));
    else throw new Error(`Invalid target: ${e.target}`);
    d(this, vt, hi).call(this);
  }
  addVideoChunk(e, t, n) {
    if (!(e instanceof EncodedVideoChunk)) throw new TypeError("addVideoChunk's first argument (chunk) must be of type EncodedVideoChunk.");
    if (t && typeof t != "object") throw new TypeError("addVideoChunk's second argument (meta), when provided, must be an object.");
    if (n !== void 0 && (!Number.isFinite(n) || n < 0)) throw new TypeError("addVideoChunk's third argument (timestamp), when provided, must be a non-negative real number.");
    let r = new Uint8Array(e.byteLength);
    e.copyTo(r), this.addVideoChunkRaw(r, e.type, n ?? e.timestamp, t);
  }
  addVideoChunkRaw(e, t, n, r) {
    if (!(e instanceof Uint8Array)) throw new TypeError("addVideoChunkRaw's first argument (data) must be an instance of Uint8Array.");
    if (t !== "key" && t !== "delta") throw new TypeError("addVideoChunkRaw's second argument (type) must be either 'key' or 'delta'.");
    if (!Number.isFinite(n) || n < 0) throw new TypeError("addVideoChunkRaw's third argument (timestamp) must be a non-negative real number.");
    if (r && typeof r != "object") throw new TypeError("addVideoChunkRaw's fourth argument (meta), when provided, must be an object.");
    if (d(this, ke, Ke).call(this), !i(this, h).video) throw new Error("No video track declared.");
    i(this, Ce) === void 0 && f(this, Ce, n), r && d(this, yt, bi).call(this, r);
    let s = d(this, xe, Xe).call(this, e, t, n, ue);
    for (i(this, h).video.codec === "V_VP9" && d(this, xt, vi).call(this, s), f(this, de, s.timestamp); i(this, R).length > 0 && i(this, R)[0].timestamp <= s.timestamp; ) {
      let o = i(this, R).shift();
      d(this, E, I).call(this, o, false);
    }
    !i(this, h).audio || s.timestamp <= i(this, le) ? d(this, E, I).call(this, s, true) : i(this, z).push(s), d(this, ye, Ye).call(this), d(this, j, te).call(this);
  }
  addAudioChunk(e, t, n) {
    if (!(e instanceof EncodedAudioChunk)) throw new TypeError("addAudioChunk's first argument (chunk) must be of type EncodedAudioChunk.");
    if (t && typeof t != "object") throw new TypeError("addAudioChunk's second argument (meta), when provided, must be an object.");
    if (n !== void 0 && (!Number.isFinite(n) || n < 0)) throw new TypeError("addAudioChunk's third argument (timestamp), when provided, must be a non-negative real number.");
    let r = new Uint8Array(e.byteLength);
    e.copyTo(r), this.addAudioChunkRaw(r, e.type, n ?? e.timestamp, t);
  }
  addAudioChunkRaw(e, t, n, r) {
    if (!(e instanceof Uint8Array)) throw new TypeError("addAudioChunkRaw's first argument (data) must be an instance of Uint8Array.");
    if (t !== "key" && t !== "delta") throw new TypeError("addAudioChunkRaw's second argument (type) must be either 'key' or 'delta'.");
    if (!Number.isFinite(n) || n < 0) throw new TypeError("addAudioChunkRaw's third argument (timestamp) must be a non-negative real number.");
    if (r && typeof r != "object") throw new TypeError("addAudioChunkRaw's fourth argument (meta), when provided, must be an object.");
    if (d(this, ke, Ke).call(this), !i(this, h).audio) throw new Error("No audio track declared.");
    i(this, Me) === void 0 && f(this, Me, n), (r == null ? void 0 : r.decoderConfig) && (i(this, h).streaming ? f(this, D, d(this, ce, Pe).call(this, r.decoderConfig.description)) : d(this, Ue, tt).call(this, i(this, D), r.decoderConfig.description));
    let s = d(this, xe, Xe).call(this, e, t, n, We);
    for (f(this, le, s.timestamp); i(this, z).length > 0 && i(this, z)[0].timestamp <= s.timestamp; ) {
      let o = i(this, z).shift();
      d(this, E, I).call(this, o, true);
    }
    !i(this, h).video || s.timestamp <= i(this, de) ? d(this, E, I).call(this, s, !i(this, h).video) : i(this, R).push(s), d(this, ye, Ye).call(this), d(this, j, te).call(this);
  }
  addSubtitleChunk(e, t, n) {
    if (typeof e != "object" || !e) throw new TypeError("addSubtitleChunk's first argument (chunk) must be an object.");
    if (!(e.body instanceof Uint8Array)) throw new TypeError("body must be an instance of Uint8Array.");
    if (!Number.isFinite(e.timestamp) || e.timestamp < 0) throw new TypeError("timestamp must be a non-negative real number.");
    if (!Number.isFinite(e.duration) || e.duration < 0) throw new TypeError("duration must be a non-negative real number.");
    if (e.additions && !(e.additions instanceof Uint8Array)) throw new TypeError("additions, when present, must be an instance of Uint8Array.");
    if (typeof t != "object") throw new TypeError("addSubtitleChunk's second argument (meta) must be an object.");
    if (d(this, ke, Ke).call(this), !i(this, h).subtitles) throw new Error("No subtitle track declared.");
    (t == null ? void 0 : t.decoderConfig) && (i(this, h).streaming ? f(this, se, d(this, ce, Pe).call(this, t.decoderConfig.description)) : d(this, Ue, tt).call(this, i(this, se), t.decoderConfig.description));
    let r = d(this, xe, Xe).call(this, e.body, "key", n ?? e.timestamp, Je, e.duration, e.additions);
    f(this, et, r.timestamp), i(this, q).push(r), d(this, ye, Ye).call(this), d(this, j, te).call(this);
  }
  finalize() {
    if (i(this, Ie)) throw new Error("Cannot finalize a muxer more than once.");
    for (; i(this, z).length > 0; ) d(this, E, I).call(this, i(this, z).shift(), true);
    for (; i(this, R).length > 0; ) d(this, E, I).call(this, i(this, R).shift(), true);
    for (; i(this, q).length > 0 && i(this, q)[0].timestamp <= i(this, oe); ) d(this, E, I).call(this, i(this, q).shift(), false);
    if (i(this, C) && d(this, it, Dt).call(this), i(this, l).writeEBML(i(this, ae)), !i(this, h).streaming) {
      let e = i(this, l).pos, t = i(this, l).pos - i(this, Y, ie);
      i(this, l).seek(i(this, l).offsets.get(i(this, Le)) + 4), i(this, l).writeEBMLVarInt(t, li), i(this, ee).data = new Bt(i(this, oe)), i(this, l).seek(i(this, l).offsets.get(i(this, ee))), i(this, l).writeEBML(i(this, ee)), i(this, U).data[0].data[1].data = i(this, l).offsets.get(i(this, ae)) - i(this, Y, ie), i(this, U).data[1].data[1].data = i(this, l).offsets.get(i(this, ze)) - i(this, Y, ie), i(this, U).data[2].data[1].data = i(this, l).offsets.get(i(this, he)) - i(this, Y, ie), i(this, l).seek(i(this, l).offsets.get(i(this, U))), i(this, l).writeEBML(i(this, U)), i(this, l).seek(e);
    }
    d(this, j, te).call(this), i(this, l).finalize(), f(this, Ie, true);
  }
};
h = /* @__PURE__ */ new WeakMap();
l = /* @__PURE__ */ new WeakMap();
Le = /* @__PURE__ */ new WeakMap();
ze = /* @__PURE__ */ new WeakMap();
U = /* @__PURE__ */ new WeakMap();
he = /* @__PURE__ */ new WeakMap();
ee = /* @__PURE__ */ new WeakMap();
K = /* @__PURE__ */ new WeakMap();
fe = /* @__PURE__ */ new WeakMap();
D = /* @__PURE__ */ new WeakMap();
se = /* @__PURE__ */ new WeakMap();
ae = /* @__PURE__ */ new WeakMap();
C = /* @__PURE__ */ new WeakMap();
De = /* @__PURE__ */ new WeakMap();
oe = /* @__PURE__ */ new WeakMap();
z = /* @__PURE__ */ new WeakMap();
R = /* @__PURE__ */ new WeakMap();
q = /* @__PURE__ */ new WeakMap();
Ce = /* @__PURE__ */ new WeakMap();
Me = /* @__PURE__ */ new WeakMap();
de = /* @__PURE__ */ new WeakMap();
le = /* @__PURE__ */ new WeakMap();
et = /* @__PURE__ */ new WeakMap();
Re = /* @__PURE__ */ new WeakMap();
Ie = /* @__PURE__ */ new WeakMap();
bt = /* @__PURE__ */ new WeakSet();
ui = function(e) {
  if (typeof e != "object") throw new TypeError("The muxer requires an options object to be passed to its constructor.");
  if (!(e.target instanceof nt)) throw new TypeError("The target must be provided and an instance of Target.");
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
hi = function() {
  i(this, l) instanceof Fe && i(this, l).target.options.onHeader && i(this, l).startTrackingWrites(), d(this, Mt, fi).call(this), i(this, h).streaming || d(this, At, gi).call(this), d(this, Wt, wi).call(this), d(this, It, _i).call(this), d(this, Ut, mi).call(this), i(this, h).streaming || (d(this, rt, Lt).call(this), d(this, st, zt).call(this)), d(this, Rt, pi).call(this), d(this, j, te).call(this);
};
Mt = /* @__PURE__ */ new WeakSet();
fi = function() {
  let e = { id: 440786851, data: [{ id: 17030, data: 1 }, { id: 17143, data: 1 }, { id: 17138, data: 4 }, { id: 17139, data: 8 }, { id: 17026, data: i(this, h).type ?? "webm" }, { id: 17031, data: 2 }, { id: 17029, data: 2 }] };
  i(this, l).writeEBML(e);
};
It = /* @__PURE__ */ new WeakSet();
_i = function() {
  f(this, fe, { id: 236, size: 4, data: new Uint8Array(Ee) }), f(this, D, { id: 236, size: 4, data: new Uint8Array(Ee) }), f(this, se, { id: 236, size: 4, data: new Uint8Array(Ee) });
};
Ut = /* @__PURE__ */ new WeakSet();
mi = function() {
  f(this, K, { id: 21936, data: [{ id: 21937, data: 2 }, { id: 21946, data: 2 }, { id: 21947, data: 2 }, { id: 21945, data: 0 }] });
};
At = /* @__PURE__ */ new WeakSet();
gi = function() {
  const e = new Uint8Array([28, 83, 187, 107]), t = new Uint8Array([21, 73, 169, 102]), n = new Uint8Array([22, 84, 174, 107]);
  f(this, U, { id: 290298740, data: [{ id: 19899, data: [{ id: 21419, data: e }, { id: 21420, size: 5, data: 0 }] }, { id: 19899, data: [{ id: 21419, data: t }, { id: 21420, size: 5, data: 0 }] }, { id: 19899, data: [{ id: 21419, data: n }, { id: 21420, size: 5, data: 0 }] }] });
};
Wt = /* @__PURE__ */ new WeakSet();
wi = function() {
  let e = { id: 17545, data: new Bt(0) };
  f(this, ee, e);
  let t = { id: 357149030, data: [{ id: 2807729, data: 1e6 }, { id: 19840, data: jt }, { id: 22337, data: jt }, i(this, h).streaming ? null : e] };
  f(this, ze, t);
};
rt = /* @__PURE__ */ new WeakSet();
Lt = function() {
  let e = { id: 374648427, data: [] };
  f(this, he, e), i(this, h).video && e.data.push({ id: 174, data: [{ id: 215, data: ue }, { id: 29637, data: ue }, { id: 131, data: Hi }, { id: 134, data: i(this, h).video.codec }, i(this, fe), i(this, h).video.frameRate ? { id: 2352003, data: 1e9 / i(this, h).video.frameRate } : null, { id: 224, data: [{ id: 176, data: i(this, h).video.width }, { id: 186, data: i(this, h).video.height }, i(this, h).video.alpha ? { id: 21440, data: 1 } : null, i(this, K)] }] }), i(this, h).audio && (f(this, D, i(this, h).streaming ? i(this, D) || null : { id: 236, size: 4, data: new Uint8Array(Ee) }), e.data.push({ id: 174, data: [{ id: 215, data: We }, { id: 29637, data: We }, { id: 131, data: $i }, { id: 134, data: i(this, h).audio.codec }, i(this, D), { id: 225, data: [{ id: 181, data: new Zt(i(this, h).audio.sampleRate) }, { id: 159, data: i(this, h).audio.numberOfChannels }, i(this, h).audio.bitDepth ? { id: 25188, data: i(this, h).audio.bitDepth } : null] }] })), i(this, h).subtitles && e.data.push({ id: 174, data: [{ id: 215, data: Je }, { id: 29637, data: Je }, { id: 131, data: qi }, { id: 134, data: i(this, h).subtitles.codec }, i(this, se)] });
};
st = /* @__PURE__ */ new WeakSet();
zt = function() {
  let e = { id: 408125543, size: i(this, h).streaming ? -1 : li, data: [i(this, h).streaming ? null : i(this, U), i(this, ze), i(this, he)] };
  if (f(this, Le, e), i(this, l).writeEBML(e), i(this, l) instanceof Fe && i(this, l).target.options.onHeader) {
    let { data: t, start: n } = i(this, l).getTrackedWrites();
    i(this, l).target.options.onHeader(t, n);
  }
};
Rt = /* @__PURE__ */ new WeakSet();
pi = function() {
  f(this, ae, { id: 475249515, data: [] });
};
j = /* @__PURE__ */ new WeakSet();
te = function() {
  i(this, l) instanceof Ct && i(this, l).flush();
};
Y = /* @__PURE__ */ new WeakSet();
ie = function() {
  return i(this, l).dataOffsets.get(i(this, Le));
};
yt = /* @__PURE__ */ new WeakSet();
bi = function(e) {
  if (e.decoderConfig) {
    if (e.decoderConfig.colorSpace) {
      let t = e.decoderConfig.colorSpace;
      if (f(this, Re, t), i(this, K).data = [{ id: 21937, data: { rgb: 1, bt709: 1, bt470bg: 5, smpte170m: 6 }[t.matrix] }, { id: 21946, data: { bt709: 1, smpte170m: 6, "iec61966-2-1": 13 }[t.transfer] }, { id: 21947, data: { bt709: 1, bt470bg: 5, smpte170m: 6 }[t.primaries] }, { id: 21945, data: [1, 2][Number(t.fullRange)] }], !i(this, h).streaming) {
        let n = i(this, l).pos;
        i(this, l).seek(i(this, l).offsets.get(i(this, K))), i(this, l).writeEBML(i(this, K)), i(this, l).seek(n);
      }
    }
    e.decoderConfig.description && (i(this, h).streaming ? f(this, fe, d(this, ce, Pe).call(this, e.decoderConfig.description)) : d(this, Ue, tt).call(this, i(this, fe), e.decoderConfig.description));
  }
};
xt = /* @__PURE__ */ new WeakSet();
vi = function(e) {
  if (e.type !== "key" || !i(this, Re)) return;
  let t = 0;
  if (Z(e.data, 0, 2) !== 2) return;
  t += 2;
  let n = (Z(e.data, t + 1, t + 2) << 1) + Z(e.data, t + 0, t + 1);
  t += 2, n === 3 && t++;
  let r = Z(e.data, t + 0, t + 1);
  if (t++, r) return;
  let s = Z(e.data, t + 0, t + 1);
  if (t++, s !== 0) return;
  t += 2;
  let o = Z(e.data, t + 0, t + 24);
  if (t += 24, o !== 4817730) return;
  n >= 2 && t++;
  let c = { rgb: 7, bt709: 2, bt470bg: 1, smpte170m: 3 }[i(this, Re).matrix];
  Fi(e.data, t + 0, t + 3, c);
};
ye = /* @__PURE__ */ new WeakSet();
Ye = function() {
  let e = Math.min(i(this, h).video ? i(this, de) : 1 / 0, i(this, h).audio ? i(this, le) : 1 / 0), t = i(this, q);
  for (; t.length > 0 && t[0].timestamp <= e; ) d(this, E, I).call(this, t.shift(), !i(this, h).video && !i(this, h).audio);
};
xe = /* @__PURE__ */ new WeakSet();
Xe = function(e, t, n, r, s, o) {
  let c = d(this, Pt, yi).call(this, n, r);
  return { data: e, additions: o, type: t, timestamp: c, duration: s, trackNumber: r };
};
Pt = /* @__PURE__ */ new WeakSet();
yi = function(e, t) {
  let n = t === ue ? i(this, de) : t === We ? i(this, le) : i(this, et);
  if (t !== Je) {
    let r = t === ue ? i(this, Ce) : i(this, Me);
    if (i(this, h).firstTimestampBehavior === "strict" && n === -1 && e !== 0) throw new Error(`The first chunk for your media track must have a timestamp of 0 (received ${e}). Non-zero first timestamps are often caused by directly piping frames or audio data from a MediaStreamTrack into the encoder. Their timestamps are typically relative to the age of the document, which is probably what you want.

If you want to offset all timestamps of a track such that the first one is zero, set firstTimestampBehavior: 'offset' in the options.
If you want to allow non-zero first timestamps, set firstTimestampBehavior: 'permissive'.
`);
    i(this, h).firstTimestampBehavior === "offset" && (e -= r);
  }
  if (e < n) throw new Error(`Timestamps must be monotonically increasing (went from ${n} to ${e}).`);
  if (e < 0) throw new Error(`Timestamps must be non-negative (received ${e}).`);
  return e;
};
E = /* @__PURE__ */ new WeakSet();
I = function(e, t) {
  i(this, h).streaming && !i(this, he) && (d(this, rt, Lt).call(this), d(this, st, zt).call(this));
  let n = Math.floor(e.timestamp / 1e3), r = n - i(this, De), s = t && e.type === "key" && r >= 1e3, o = r >= ji;
  if ((!i(this, C) || s || o) && (d(this, Ft, xi).call(this, n), r = 0), r < 0) return;
  let c = new Uint8Array(4), m = new DataView(c.buffer);
  if (m.setUint8(0, 128 | e.trackNumber), m.setInt16(1, r, false), e.duration === void 0 && !e.additions) {
    m.setUint8(3, +(e.type === "key") << 7);
    let p = { id: 163, data: [c, e.data] };
    i(this, l).writeEBML(p);
  } else {
    let p = Math.floor(e.duration / 1e3), y = { id: 160, data: [{ id: 161, data: [c, e.data] }, e.duration !== void 0 ? { id: 155, data: p } : null, e.additions ? { id: 30113, data: e.additions } : null] };
    i(this, l).writeEBML(y);
  }
  f(this, oe, Math.max(i(this, oe), n));
};
ce = /* @__PURE__ */ new WeakSet();
Pe = function(e) {
  return { id: 25506, size: 4, data: new Uint8Array(e) };
};
Ue = /* @__PURE__ */ new WeakSet();
tt = function(e, t) {
  let n = i(this, l).pos;
  i(this, l).seek(i(this, l).offsets.get(e));
  let r = 6 + t.byteLength, s = Ee - r;
  if (s < 0) {
    let o = t.byteLength + s;
    t instanceof ArrayBuffer ? t = t.slice(0, o) : t = t.buffer.slice(0, o), s = 0;
  }
  e = [d(this, ce, Pe).call(this, t), { id: 236, size: 4, data: new Uint8Array(s) }], i(this, l).writeEBML(e), i(this, l).seek(n);
};
Ft = /* @__PURE__ */ new WeakSet();
xi = function(e) {
  i(this, C) && d(this, it, Dt).call(this), i(this, l) instanceof Fe && i(this, l).target.options.onCluster && i(this, l).startTrackingWrites(), f(this, C, { id: 524531317, size: i(this, h).streaming ? -1 : ci, data: [{ id: 231, data: e }] }), i(this, l).writeEBML(i(this, C)), f(this, De, e);
  let t = i(this, l).offsets.get(i(this, C)) - i(this, Y, ie);
  i(this, ae).data.push({ id: 187, data: [{ id: 179, data: e }, i(this, h).video ? { id: 183, data: [{ id: 247, data: ue }, { id: 241, data: t }] } : null, i(this, h).audio ? { id: 183, data: [{ id: 247, data: We }, { id: 241, data: t }] } : null] });
};
it = /* @__PURE__ */ new WeakSet();
Dt = function() {
  if (!i(this, h).streaming) {
    let e = i(this, l).pos - i(this, l).dataOffsets.get(i(this, C)), t = i(this, l).pos;
    i(this, l).seek(i(this, l).offsets.get(i(this, C)) + 4), i(this, l).writeEBMLVarInt(e, ci), i(this, l).seek(t);
  }
  if (i(this, l) instanceof Fe && i(this, l).target.options.onCluster) {
    let { data: e, start: t } = i(this, l).getTrackedWrites();
    i(this, l).target.options.onCluster(e, t, i(this, De));
  }
};
ke = /* @__PURE__ */ new WeakSet();
Ke = function() {
  if (i(this, Ie)) throw new Error("Cannot add new video or audio chunks after the file has been finalized.");
};
new TextEncoder();
const k = document.getElementById("gpu-canvas"), M = document.getElementById("render-btn"), Yt = document.getElementById("scene-select"), Xt = document.getElementById("res-width"), Kt = document.getElementById("res-height"), kt = document.getElementById("obj-file");
kt && (kt.accept = ".obj,.glb,.vrm");
const Ki = document.getElementById("max-depth"), Qi = document.getElementById("spp-frame"), Zi = document.getElementById("recompile-btn"), Ji = document.getElementById("update-interval"), W = document.getElementById("anim-select"), L = document.getElementById("record-btn"), en = document.getElementById("rec-fps"), tn = document.getElementById("rec-duration"), nn = document.getElementById("rec-spp"), rn = document.getElementById("rec-batch"), Vt = document.createElement("div");
Object.assign(Vt.style, { position: "fixed", bottom: "10px", left: "10px", color: "#0f0", background: "rgba(0,0,0,0.7)", padding: "8px", fontFamily: "monospace", fontSize: "14px", pointerEvents: "none", zIndex: "9999", borderRadius: "4px" });
document.body.appendChild(Vt);
let O = 0, T = false, Ne = false, pe = null, Oe = null;
async function sn() {
  const e = new Si(k), t = new Ri();
  let n = 0;
  try {
    await e.init(), await t.initWasm();
  } catch (_) {
    alert("Initialization failed: " + _), console.error(_);
    return;
  }
  const r = () => {
    const _ = parseInt(Ki.value, 10) || 10, w = parseInt(Qi.value, 10) || 1;
    e.buildPipeline(_, w);
  };
  r();
  const s = () => {
    const _ = parseInt(Xt.value, 10) || 720, w = parseInt(Kt.value, 10) || 480;
    e.updateScreenSize(_, w), t.hasWorld && (t.updateCamera(_, w), e.updateSceneUniforms(t.cameraData, 0)), e.recreateBindGroup(), e.resetAccumulation(), O = 0, n = 0;
  }, o = async (_, w = true) => {
    T = false, console.log(`Loading Scene: ${_}...`);
    let b, x;
    _ === "viewer" && pe && (Oe === "obj" ? b = pe : Oe === "glb" && (x = new Uint8Array(pe))), t.loadScene(_, b, x), t.printStats(), await e.loadTexturesFromWorld(t), e.updateCombinedGeometry(t.vertices, t.normals, t.uvs), e.updateCombinedBVH(t.tlas, t.blas), e.updateBuffer("index", t.indices), e.updateBuffer("attr", t.attributes), e.updateBuffer("instance", t.instances), s(), _e(), w && (T = true, M && (M.textContent = "Stop Rendering"));
  }, c = async () => {
    if (Ne) return;
    T = false, Ne = true, L.textContent = "Initializing...", L.disabled = true, M && (M.textContent = "Resume Rendering");
    const _ = parseInt(en.value, 10) || 30, w = parseInt(tn.value, 10) || 3, b = _ * w, x = parseInt(nn.value, 10) || 64, ki = parseInt(rn.value, 10) || 4;
    console.log(`Starting recording: ${b} frames @ ${_}fps (VP9)`);
    const at = new Xi({ target: new ei(), video: { codec: "V_VP9", width: k.width, height: k.height, frameRate: _ } }), me = new VideoEncoder({ output: (V, ot) => at.addVideoChunk(V, ot), error: (V) => console.error("VideoEncoder Error:", V) });
    me.configure({ codec: "vp09.00.10.08", width: k.width, height: k.height, bitrate: 12e6 });
    try {
      for (let N = 0; N < b; N++) {
        L.textContent = `Rec: ${N}/${b} (${Math.round(N / b * 100)}%)`, await new Promise((Ve) => setTimeout(Ve, 0));
        const Ti = N / _;
        t.update(Ti);
        let Q = false;
        Q || (Q = e.updateCombinedBVH(t.tlas, t.blas)), Q || (Q = e.updateBuffer("instance", t.instances)), Q || (Q = e.updateCombinedGeometry(t.vertices, t.normals, t.uvs)), Q || (Q = e.updateBuffer("index", t.indices)), Q || (Q = e.updateBuffer("attr", t.attributes)), t.updateCamera(k.width, k.height), e.updateSceneUniforms(t.cameraData, 0), Q && e.recreateBindGroup(), e.resetAccumulation();
        let ge = 0;
        for (; ge < x; ) {
          const Ve = Math.min(ki, x - ge);
          for (let we = 0; we < Ve; we++) e.render(ge + we);
          ge += Ve, await e.device.queue.onSubmittedWorkDone(), ge < x && await new Promise((we) => setTimeout(we, 0));
        }
        me.encodeQueueSize > 5 && await me.flush();
        const Ot = new VideoFrame(k, { timestamp: N * 1e6 / _, duration: 1e6 / _ });
        me.encode(Ot, { keyFrame: N % _ === 0 }), Ot.close();
      }
      L.textContent = "Finalizing...", await me.flush(), at.finalize();
      const { buffer: V } = at.target, ot = new Blob([V], { type: "video/webm" }), Nt = URL.createObjectURL(ot), dt = document.createElement("a");
      dt.href = Nt, dt.download = `raytrace_${Date.now()}.webm`, dt.click(), URL.revokeObjectURL(Nt);
    } catch (V) {
      console.error("Recording failed:", V), alert("Recording failed. See console.");
    } finally {
      Ne = false, T = true, L.textContent = "\u25CF Rec", L.disabled = false, M && (M.textContent = "Stop Rendering"), requestAnimationFrame(y);
    }
  };
  let m = performance.now(), p = 0;
  const y = () => {
    if (Ne || (requestAnimationFrame(y), !T || !t.hasWorld)) return;
    let _ = parseInt(Ji.value, 10);
    if ((isNaN(_) || _ < 0) && (_ = 0), _ > 0 && O >= _) {
      t.update(n / _ / 60);
      let b = false;
      b || (b = e.updateCombinedBVH(t.tlas, t.blas)), b || (b = e.updateBuffer("instance", t.instances)), b || (b = e.updateCombinedGeometry(t.vertices, t.normals, t.uvs)), b || (b = e.updateBuffer("index", t.indices)), b || (b = e.updateBuffer("attr", t.attributes)), t.updateCamera(k.width, k.height), e.updateSceneUniforms(t.cameraData, 0), b && e.recreateBindGroup(), e.resetAccumulation(), O = 0;
    }
    O++, p++, n++, e.render(O);
    const w = performance.now();
    w - m >= 1e3 && (Vt.textContent = `FPS: ${p} | ${(1e3 / p).toFixed(2)}ms | Frame: ${O}`, p = 0, m = w);
  };
  M && M.addEventListener("click", () => {
    T = !T, M.textContent = T ? "Stop Rendering" : "Resume Rendering";
  }), L && L.addEventListener("click", c), Yt.addEventListener("change", (_) => o(_.target.value, false)), Xt.addEventListener("change", s), Kt.addEventListener("change", s), Zi.addEventListener("click", () => {
    T = false, r(), e.recreateBindGroup(), e.resetAccumulation(), O = 0, T = true;
  }), kt.addEventListener("change", async (_) => {
    var _a, _b;
    const w = (_a = _.target.files) == null ? void 0 : _a[0];
    if (!w) return;
    ((_b = w.name.split(".").pop()) == null ? void 0 : _b.toLowerCase()) === "obj" ? (pe = await w.text(), Oe = "obj") : (pe = await w.arrayBuffer(), Oe = "glb"), Yt.value = "viewer", o("viewer", false);
  });
  const _e = () => {
    const _ = t.getAnimationList();
    if (W.innerHTML = "", _.length === 0) {
      const w = document.createElement("option");
      w.text = "No Anim", W.add(w), W.disabled = true;
      return;
    }
    W.disabled = false, _.forEach((w, b) => {
      const x = document.createElement("option");
      x.text = `[${b}] ${w}`, x.value = b.toString(), W.add(x);
    }), W.value = "0";
  };
  W.addEventListener("change", () => {
    const _ = parseInt(W.value, 10);
    t.setAnimation(_);
  }), s(), o("cornell", false), requestAnimationFrame(y);
}
sn().catch(console.error);
