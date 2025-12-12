var __defProp = Object.defineProperty;
var __defNormalProp = (obj, key, value) => key in obj ? __defProp(obj, key, { enumerable: true, configurable: true, writable: true, value }) : obj[key] = value;
var __publicField = (obj, key, value) => __defNormalProp(obj, typeof key !== "symbol" ? key + "" : key, value);
(function() {
  const e = document.createElement("link").relList;
  if (e && e.supports && e.supports("modulepreload")) return;
  for (const r of document.querySelectorAll('link[rel="modulepreload"]')) s(r);
  new MutationObserver((r) => {
    for (const o of r) if (o.type === "childList") for (const a of o.addedNodes) a.tagName === "LINK" && a.rel === "modulepreload" && s(a);
  }).observe(document, { childList: true, subtree: true });
  function n(r) {
    const o = {};
    return r.integrity && (o.integrity = r.integrity), r.referrerPolicy && (o.referrerPolicy = r.referrerPolicy), r.crossOrigin === "use-credentials" ? o.credentials = "include" : r.crossOrigin === "anonymous" ? o.credentials = "omit" : o.credentials = "same-origin", o;
  }
  function s(r) {
    if (r.ep) return;
    r.ep = true;
    const o = n(r);
    fetch(r.href, o);
  }
})();
const O = `// =========================================================
//   WebGPU Ray Tracer (TLAS & BLAS)
// =========================================================

const PI = 3.141592653589793;
const T_MIN = 0.001;
const T_MAX = 1e30;
// \u3053\u308C\u3089\u306FTypeScript\u5074\u304B\u3089\u7F6E\u63DB\u3055\u308C\u307E\u3059
const MAX_DEPTH = 10u;
const SPP = 1u;

// --- Bindings ---
@group(0) @binding(0) var outputTex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<storage, read_write> accumulateBuffer: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> frame: FrameInfo;
@group(0) @binding(3) var<uniform> camera: Camera;

@group(0) @binding(4) var<storage, read> vertices: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> indices: array<u32>;
@group(0) @binding(6) var<storage, read> attributes: array<TriangleAttributes>;
@group(0) @binding(7) var<storage, read> tlas_nodes: array<BVHNode>;
@group(0) @binding(8) var<storage, read> normals: array<vec4<f32>>;
@group(0) @binding(9) var<storage, read> blas_nodes: array<BVHNode>;
@group(0) @binding(10) var<storage, read> instances: array<Instance>;

struct FrameInfo {
    frame_count: u32
}

struct Camera {
    origin: vec3<f32>,
    lens_radius: f32,
    lower_left_corner: vec3<f32>,
    horizontal: vec3<f32>,
    vertical: vec3<f32>,
    u: vec3<f32>,
    v: vec3<f32>
}
struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>
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

fn get_inv_transform(inst: Instance) -> mat4x4<f32> {
    return mat4x4<f32>(inst.inv_0, inst.inv_1, inst.inv_2, inst.inv_3);
}

struct TriangleAttributes {
    data0: vec4<f32>,
    data1: vec4<f32>
}
struct HitResult {
    t: f32,
    tri_idx: f32,
    inst_idx: i32
}

// --- Random & Physics ---
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

// BLAS Intersection Logic Update
fn intersect_blas(r: Ray, t_min: f32, t_max: f32, node_offset: u32) -> vec2<f32> {
    var closest_t = t_max;
    var hit_idx = -1.0;
    let inv_d = 1.0 / r.direction;
    var stack: array<u32, 32>;
    var stackptr = 0u;

    // Push Root Node (Global Index)
    let root_idx = node_offset;
    let root = blas_nodes[root_idx];

    if intersect_aabb(root.min_b, root.max_b, r, inv_d, t_min, closest_t) < 1e30 {
        stack[stackptr] = root_idx;
        stackptr++;
    }

    while stackptr > 0u {
        stackptr--;
        let idx = stack[stackptr]; // Global Node Index
        let node = blas_nodes[idx];
        let count = u32(node.tri_count);

        if count > 0u {
            // Leaf Node
            let first = u32(node.left_first); // Triangle Index (Sorted)
            for (var i = 0u; i < count; i++) {
                let tri_id = first + i;
                let b = tri_id * 3u;
                // indices & vertices are global buffers, so simple access is correct
                let t = hit_triangle_raw(vertices[indices[b]].xyz, vertices[indices[b + 1u]].xyz, vertices[indices[b + 2u]].xyz, r, t_min, closest_t);
                if t > 0.0 { closest_t = t; hit_idx = f32(tri_id); }
            }
        } else {
            // Internal Node
            // node.left_first is RELATIVE to the start of this BLAS.
            // We must add node_offset to get the Global Index.
            let l = u32(node.left_first) + node_offset; // \u2605Fix: Add offset
            let r_node_idx = l + 1u;

            let nl = blas_nodes[l];
            let nr = blas_nodes[r_node_idx];

            let dl = intersect_aabb(nl.min_b, nl.max_b, r, inv_d, t_min, closest_t);
            let dr = intersect_aabb(nr.min_b, nr.max_b, r, inv_d, t_min, closest_t);

            let hl = dl < 1e30; let hr = dr < 1e30;
            if hl && hr {
                // Push further node first
                if dl < dr { stack[stackptr] = r_node_idx; stackptr++; stack[stackptr] = l; stackptr++; } else { stack[stackptr] = l; stackptr++; stack[stackptr] = r_node_idx; stackptr++; }
            } else if hl { stack[stackptr] = l; stackptr++; } else if hr { stack[stackptr] = r_node_idx; stackptr++; }
        }
    }
    return vec2<f32>(closest_t, hit_idx);
}

fn intersect_tlas(r: Ray, t_min: f32, t_max: f32) -> HitResult {
    var res: HitResult; res.t = t_max; res.tri_idx = -1.0; res.inst_idx = -1;
    if arrayLength(&tlas_nodes) == 0u { return res; }

    let inv_d = 1.0 / r.direction;
    var stack: array<u32, 32>;
    var stackptr = 0u;

    if intersect_aabb(tlas_nodes[0].min_b, tlas_nodes[0].max_b, r, inv_d, t_min, res.t) < 1e30 {
        stack[stackptr] = 0u; stackptr++;
    }

    while stackptr > 0u {
        stackptr--;
        let idx = stack[stackptr];
        let node = tlas_nodes[idx];

        if node.tri_count > 0.5 { // Leaf (Instance)
            let inst_idx = u32(node.left_first);
            let inst = instances[inst_idx];
            let inv = get_inv_transform(inst);
            
            // Transform ray to local space
            let r_local = Ray((inv * vec4(r.origin, 1.0)).xyz, (inv * vec4(r.direction, 0.0)).xyz);
            let blas = intersect_blas(r_local, t_min, res.t, inst.blas_node_offset);

            if blas.y > -0.5 {
                res.t = blas.x;
                res.tri_idx = blas.y;
                res.inst_idx = i32(inst_idx);
            }
        } else {
            // Internal (TLAS)
            let l = u32(node.left_first);
            let r_idx = l + 1u;
            let nl = tlas_nodes[l];
            let nr = tlas_nodes[r_idx];
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
        // Note: tri_idx is global triangle index (offset baked in Rust)

        let i0 = indices[tri_idx * 3u];
        let i1 = indices[tri_idx * 3u + 1u];
        let i2 = indices[tri_idx * 3u + 2u];

        // Normal Interpolation
        let inv = get_inv_transform(inst);
        let r_local = Ray((inv * vec4(ray.origin, 1.)).xyz, (inv * vec4(ray.direction, 0.)).xyz);

        let e1 = vertices[i1].xyz - vertices[i0].xyz;
        let e2 = vertices[i2].xyz - vertices[i0].xyz;
        let h = cross(r_local.direction, e2);
        let a = dot(e1, h);
        let f = 1.0 / a;
        let s = r_local.origin - vertices[i0].xyz;
        let u = f * dot(s, h);
        let q = cross(s, e1);
        let v = f * dot(r_local.direction, q);
        let w = 1.0 - u - v;

        let ln = normalize(normals[i0].xyz * w + normals[i1].xyz * u + normals[i2].xyz * v);
        let wn = normalize((vec4(ln, 0.0) * inv).xyz);

        var n = wn;
        let front = dot(ray.direction, n) < 0.0;
        n = select(-n, n, front);

        // Attributes
        let attr = attributes[tri_idx];
        let albedo = attr.data0.rgb;
        
        // \u2605 \u4FEE\u6B63\u7B87\u6240: \u5909\u6570\u540D\u3092 'type' \u304B\u3089 'mat_type' \u306B\u5909\u66F4
        let mat_type = bitcast<u32>(attr.data0.w); // f32 -> u32

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
        throughput *= albedo;

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
    var rng = init_rng(p_idx, frame.frame_count);

    var col = vec3(0.);
    for (var s = 0u; s < SPP; s++) {
        var off = vec3(0.);
        if camera.lens_radius > 0. {
            let rd = camera.lens_radius * random_in_unit_disk(&rng);
            off = camera.u * rd.x + camera.v * rd.y;
        }
        let u = (f32(id.x) + rand_pcg(&rng)) / f32(dims.x);
        let v = 1. - (f32(id.y) + rand_pcg(&rng)) / f32(dims.y);
        let d = camera.lower_left_corner + u * camera.horizontal + v * camera.vertical - camera.origin - off;
        col += ray_color(Ray(camera.origin + off, d), &rng);
    }
    col /= f32(SPP);

    var acc = vec4(0.);
    if frame.frame_count > 1u { acc = accumulateBuffer[p_idx]; }
    let new_acc = acc + vec4(col, 1.0);
    accumulateBuffer[p_idx] = new_acc;

    let out = sqrt(clamp(new_acc.rgb / new_acc.a, vec3(0.), vec3(1.)));
    textureStore(outputTex, vec2<i32>(id.xy), vec4(out, 1.));
}
`;
class W {
  constructor(e) {
    __publicField(this, "device");
    __publicField(this, "context");
    __publicField(this, "pipeline");
    __publicField(this, "bindGroupLayout");
    __publicField(this, "bindGroup");
    __publicField(this, "renderTarget");
    __publicField(this, "renderTargetView");
    __publicField(this, "accumulateBuffer");
    __publicField(this, "frameUniformBuffer");
    __publicField(this, "cameraUniformBuffer");
    __publicField(this, "vertexBuffer");
    __publicField(this, "normalBuffer");
    __publicField(this, "indexBuffer");
    __publicField(this, "attrBuffer");
    __publicField(this, "tlasBuffer");
    __publicField(this, "blasBuffer");
    __publicField(this, "instanceBuffer");
    __publicField(this, "bufferSize", 0);
    __publicField(this, "canvas");
    this.canvas = e;
  }
  async init() {
    if (!navigator.gpu) throw new Error("WebGPU not supported.");
    const e = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!e) throw new Error("No adapter");
    this.device = await e.requestDevice(), this.context = this.canvas.getContext("webgpu"), this.context.configure({ device: this.device, format: "rgba8unorm", usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT }), this.frameUniformBuffer = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }), this.cameraUniformBuffer = this.device.createBuffer({ size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  }
  buildPipeline(e, n) {
    let s = O;
    s = s.replace(/const\s+MAX_DEPTH\s*=\s*\d+u;/, `const MAX_DEPTH = ${e}u;`), s = s.replace(/const\s+SPP\s*=\s*\d+u;/, `const SPP = ${n}u;`);
    const r = this.device.createShaderModule({ label: "RayTracing", code: s });
    this.pipeline = this.device.createComputePipeline({ label: "Main Pipeline", layout: "auto", compute: { module: r, entryPoint: "main" } }), this.bindGroupLayout = this.pipeline.getBindGroupLayout(0);
  }
  updateScreenSize(e, n) {
    this.canvas.width = e, this.canvas.height = n, this.renderTarget && this.renderTarget.destroy(), this.renderTarget = this.device.createTexture({ size: [e, n], format: "rgba8unorm", usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC }), this.renderTargetView = this.renderTarget.createView(), this.bufferSize = e * n * 16, this.accumulateBuffer && this.accumulateBuffer.destroy(), this.accumulateBuffer = this.device.createBuffer({ size: this.bufferSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  }
  resetAccumulation() {
    this.accumulateBuffer && this.device.queue.writeBuffer(this.accumulateBuffer, 0, new Float32Array(this.bufferSize / 4));
  }
  updateGeometryBuffer(e, n) {
    let r = { tlas: this.tlasBuffer, blas: this.blasBuffer, instance: this.instanceBuffer, vertex: this.vertexBuffer, normal: this.normalBuffer, index: this.indexBuffer, attr: this.attrBuffer }[e];
    if (!r || r.size < n.byteLength) {
      r && r.destroy();
      const o = Math.max(n.byteLength, 4), a = this.device.createBuffer({ size: o, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      switch (this.device.queue.writeBuffer(a, 0, n), e) {
        case "tlas":
          this.tlasBuffer = a;
          break;
        case "blas":
          this.blasBuffer = a;
          break;
        case "instance":
          this.instanceBuffer = a;
          break;
        case "vertex":
          this.vertexBuffer = a;
          break;
        case "normal":
          this.normalBuffer = a;
          break;
        case "index":
          this.indexBuffer = a;
          break;
        case "attr":
          this.attrBuffer = a;
          break;
      }
      return true;
    } else return n.byteLength > 0 && this.device.queue.writeBuffer(r, 0, n), false;
  }
  updateCameraBuffer(e) {
    this.device.queue.writeBuffer(this.cameraUniformBuffer, 0, e);
  }
  updateFrameBuffer(e) {
    this.device.queue.writeBuffer(this.frameUniformBuffer, 0, new Uint32Array([e]));
  }
  recreateBindGroup() {
    !this.renderTargetView || !this.accumulateBuffer || !this.vertexBuffer || !this.tlasBuffer || (this.bindGroup = this.device.createBindGroup({ layout: this.bindGroupLayout, entries: [{ binding: 0, resource: this.renderTargetView }, { binding: 1, resource: { buffer: this.accumulateBuffer } }, { binding: 2, resource: { buffer: this.frameUniformBuffer } }, { binding: 3, resource: { buffer: this.cameraUniformBuffer } }, { binding: 4, resource: { buffer: this.vertexBuffer } }, { binding: 5, resource: { buffer: this.indexBuffer } }, { binding: 6, resource: { buffer: this.attrBuffer } }, { binding: 7, resource: { buffer: this.tlasBuffer } }, { binding: 8, resource: { buffer: this.normalBuffer } }, { binding: 9, resource: { buffer: this.blasBuffer } }, { binding: 10, resource: { buffer: this.instanceBuffer } }] }));
  }
  render(e) {
    if (!this.bindGroup) return;
    this.updateFrameBuffer(e);
    const n = Math.ceil(this.canvas.width / 8), s = Math.ceil(this.canvas.height / 8), r = this.device.createCommandEncoder(), o = r.beginComputePass();
    o.setPipeline(this.pipeline), o.setBindGroup(0, this.bindGroup), o.dispatchWorkgroups(n, s), o.end(), r.copyTextureToTexture({ texture: this.renderTarget }, { texture: this.context.getCurrentTexture() }, { width: this.canvas.width, height: this.canvas.height, depthOrArrayLayers: 1 }), this.device.queue.submit([r.finish()]);
  }
}
let i;
function N(t) {
  const e = i.__externref_table_alloc();
  return i.__wbindgen_externrefs.set(e, t), e;
}
function q(t, e) {
  return t = t >>> 0, b().subarray(t / 1, t / 1 + e);
}
let p = null;
function E() {
  return (p === null || p.buffer.detached === true || p.buffer.detached === void 0 && p.buffer !== i.memory.buffer) && (p = new DataView(i.memory.buffer)), p;
}
function U(t, e) {
  return t = t >>> 0, $(t, e);
}
let v = null;
function b() {
  return (v === null || v.byteLength === 0) && (v = new Uint8Array(i.memory.buffer)), v;
}
function V(t, e) {
  try {
    return t.apply(this, e);
  } catch (n) {
    const s = N(n);
    i.__wbindgen_exn_store(s);
  }
}
function z(t) {
  return t == null;
}
function G(t, e) {
  const n = e(t.length * 1, 1) >>> 0;
  return b().set(t, n / 1), m = t.length, n;
}
function k(t, e, n) {
  if (n === void 0) {
    const u = w.encode(t), d = e(u.length, 1) >>> 0;
    return b().subarray(d, d + u.length).set(u), m = u.length, d;
  }
  let s = t.length, r = e(s, 1) >>> 0;
  const o = b();
  let a = 0;
  for (; a < s; a++) {
    const u = t.charCodeAt(a);
    if (u > 127) break;
    o[r + a] = u;
  }
  if (a !== s) {
    a !== 0 && (t = t.slice(a)), r = n(r, s, s = a + t.length * 3, 1) >>> 0;
    const u = b().subarray(r + a, r + s), d = w.encodeInto(t, u);
    a += d.written, r = n(r, s, a, 1) >>> 0;
  }
  return m = a, r;
}
let x = new TextDecoder("utf-8", { ignoreBOM: true, fatal: true });
x.decode();
const H = 2146435072;
let B = 0;
function $(t, e) {
  return B += e, B >= H && (x = new TextDecoder("utf-8", { ignoreBOM: true, fatal: true }), x.decode(), B = e), x.decode(b().subarray(t, t + e));
}
const w = new TextEncoder();
"encodeInto" in w || (w.encodeInto = function(t, e) {
  const n = w.encode(t);
  return e.set(n), { read: t.length, written: n.length };
});
let m = 0;
typeof FinalizationRegistry > "u" || new FinalizationRegistry((t) => i.__wbg_renderbuffers_free(t >>> 0, 1));
const L = typeof FinalizationRegistry > "u" ? { register: () => {
}, unregister: () => {
} } : new FinalizationRegistry((t) => i.__wbg_world_free(t >>> 0, 1));
class S {
  __destroy_into_raw() {
    const e = this.__wbg_ptr;
    return this.__wbg_ptr = 0, L.unregister(this), e;
  }
  free() {
    const e = this.__destroy_into_raw();
    i.__wbg_world_free(e, 0);
  }
  camera_ptr() {
    return i.world_camera_ptr(this.__wbg_ptr) >>> 0;
  }
  indices_len() {
    return i.world_indices_len(this.__wbg_ptr) >>> 0;
  }
  indices_ptr() {
    return i.world_indices_ptr(this.__wbg_ptr) >>> 0;
  }
  normals_len() {
    return i.world_normals_len(this.__wbg_ptr) >>> 0;
  }
  normals_ptr() {
    return i.world_normals_ptr(this.__wbg_ptr) >>> 0;
  }
  vertices_len() {
    return i.world_vertices_len(this.__wbg_ptr) >>> 0;
  }
  vertices_ptr() {
    return i.world_vertices_ptr(this.__wbg_ptr) >>> 0;
  }
  instances_len() {
    return i.world_instances_len(this.__wbg_ptr) >>> 0;
  }
  instances_ptr() {
    return i.world_instances_ptr(this.__wbg_ptr) >>> 0;
  }
  update_camera(e, n) {
    i.world_update_camera(this.__wbg_ptr, e, n);
  }
  attributes_len() {
    return i.world_attributes_len(this.__wbg_ptr) >>> 0;
  }
  attributes_ptr() {
    return i.world_attributes_ptr(this.__wbg_ptr) >>> 0;
  }
  load_animation_glb(e) {
    const n = G(e, i.__wbindgen_malloc), s = m;
    i.world_load_animation_glb(this.__wbg_ptr, n, s);
  }
  constructor(e, n, s) {
    const r = k(e, i.__wbindgen_malloc, i.__wbindgen_realloc), o = m;
    var a = z(n) ? 0 : k(n, i.__wbindgen_malloc, i.__wbindgen_realloc), u = m, d = z(s) ? 0 : G(s, i.__wbindgen_malloc), c = m;
    const l = i.world_new(r, o, a, u, d, c);
    return this.__wbg_ptr = l >>> 0, L.register(this, this.__wbg_ptr, this), this;
  }
  update(e) {
    i.world_update(this.__wbg_ptr, e);
  }
  blas_len() {
    return i.world_blas_len(this.__wbg_ptr) >>> 0;
  }
  blas_ptr() {
    return i.world_blas_ptr(this.__wbg_ptr) >>> 0;
  }
  tlas_len() {
    return i.world_tlas_len(this.__wbg_ptr) >>> 0;
  }
  tlas_ptr() {
    return i.world_tlas_ptr(this.__wbg_ptr) >>> 0;
  }
}
Symbol.dispose && (S.prototype[Symbol.dispose] = S.prototype.free);
const j = /* @__PURE__ */ new Set(["basic", "cors", "default"]);
async function X(t, e) {
  if (typeof Response == "function" && t instanceof Response) {
    if (typeof WebAssembly.instantiateStreaming == "function") try {
      return await WebAssembly.instantiateStreaming(t, e);
    } catch (s) {
      if (t.ok && j.has(t.type) && t.headers.get("Content-Type") !== "application/wasm") console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", s);
      else throw s;
    }
    const n = await t.arrayBuffer();
    return await WebAssembly.instantiate(n, e);
  } else {
    const n = await WebAssembly.instantiate(t, e);
    return n instanceof WebAssembly.Instance ? { instance: n, module: t } : n;
  }
}
function Y() {
  const t = {};
  return t.wbg = {}, t.wbg.__wbg___wbindgen_throw_dd24417ed36fc46e = function(e, n) {
    throw new Error(U(e, n));
  }, t.wbg.__wbg_error_7534b8e9a36f1ab4 = function(e, n) {
    let s, r;
    try {
      s = e, r = n, console.error(U(e, n));
    } finally {
      i.__wbindgen_free(s, r, 1);
    }
  }, t.wbg.__wbg_getRandomValues_1c61fac11405ffdc = function() {
    return V(function(e, n) {
      globalThis.crypto.getRandomValues(q(e, n));
    }, arguments);
  }, t.wbg.__wbg_new_8a6f238a6ece86ea = function() {
    return new Error();
  }, t.wbg.__wbg_stack_0ed75d68575b0f3c = function(e, n) {
    const s = n.stack, r = k(s, i.__wbindgen_malloc, i.__wbindgen_realloc), o = m;
    E().setInt32(e + 4, o, true), E().setInt32(e + 0, r, true);
  }, t.wbg.__wbindgen_init_externref_table = function() {
    const e = i.__wbindgen_externrefs, n = e.grow(4);
    e.set(0, void 0), e.set(n + 0, void 0), e.set(n + 1, null), e.set(n + 2, true), e.set(n + 3, false);
  }, t;
}
function K(t, e) {
  return i = t.exports, D.__wbindgen_wasm_module = e, p = null, v = null, i.__wbindgen_start(), i;
}
async function D(t) {
  if (i !== void 0) return i;
  typeof t < "u" && (Object.getPrototypeOf(t) === Object.prototype ? { module_or_path: t } = t : console.warn("using deprecated parameters for the initialization function; pass a single object instead")), typeof t > "u" && (t = new URL("/webgpu-raytracer/assets/rust_shader_tools_bg-BK2yN6Qq.wasm", import.meta.url));
  const e = Y();
  (typeof t == "string" || typeof Request == "function" && t instanceof Request || typeof URL == "function" && t instanceof URL) && (t = fetch(t));
  const { instance: n, module: s } = await X(await t, e);
  return K(n, s);
}
class Q {
  constructor() {
    __publicField(this, "world", null);
    __publicField(this, "wasmMemory", null);
  }
  async initWasm() {
    const e = await D();
    this.wasmMemory = e.memory, console.log("Wasm initialized");
  }
  loadScene(e, n, s) {
    this.world && this.world.free(), this.world = new S(e, n, s);
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
  get hasWorld() {
    return !!this.world;
  }
  printStats() {
    this.world && console.log(`Scene Stats: V=${this.vertices.length / 4}, Tri=${this.indices.length / 3}, BLAS=${this.blas.length / 8}, TLAS=${this.tlas.length / 8}`);
  }
}
const J = document.getElementById("gpu-canvas"), T = document.getElementById("render-btn"), F = document.getElementById("scene-select"), C = document.getElementById("res-width"), M = document.getElementById("res-height"), A = document.getElementById("obj-file");
A && (A.accept = ".obj,.glb,.vrm");
const P = document.getElementById("anim-file");
P && (P.accept = ".glb,.gltf");
const Z = document.getElementById("max-depth"), ee = document.getElementById("spp-frame"), te = document.getElementById("recompile-btn"), ne = document.getElementById("update-interval"), I = document.createElement("div");
Object.assign(I.style, { position: "fixed", bottom: "10px", left: "10px", color: "#0f0", background: "rgba(0,0,0,0.7)", padding: "8px", fontFamily: "monospace", fontSize: "14px", pointerEvents: "none", zIndex: "9999", borderRadius: "4px" });
document.body.appendChild(I);
let g = 0, _ = false, h = null, y = null;
async function re() {
  const t = new W(J), e = new Q();
  let n = 0;
  try {
    await t.init(), await e.initWasm();
  } catch (c) {
    alert("Initialization failed: " + c), console.error(c);
    return;
  }
  const s = () => {
    const c = parseInt(Z.value, 10) || 10, l = parseInt(ee.value, 10) || 1;
    t.buildPipeline(c, l);
  };
  s();
  const r = () => {
    const c = parseInt(C.value, 10) || 720, l = parseInt(M.value, 10) || 480;
    t.updateScreenSize(c, l), e.hasWorld && (e.updateCamera(c, l), t.updateCameraBuffer(e.cameraData)), t.recreateBindGroup(), t.resetAccumulation(), g = 0, n = 0;
  }, o = (c, l = true) => {
    _ = false, console.log(`Loading Scene: ${c}...`);
    let f, R;
    c === "viewer" && h && (y === "obj" ? f = h : y === "glb" && (R = new Uint8Array(h))), e.loadScene(c, f, R), e.printStats(), t.updateGeometryBuffer("vertex", e.vertices), t.updateGeometryBuffer("normal", e.normals), t.updateGeometryBuffer("index", e.indices), t.updateGeometryBuffer("attr", e.attributes), t.updateGeometryBuffer("tlas", e.tlas), t.updateGeometryBuffer("blas", e.blas), t.updateGeometryBuffer("instance", e.instances), r(), l && (_ = true, T.textContent = "Stop Rendering");
  };
  let a = performance.now(), u = 0;
  const d = () => {
    if (requestAnimationFrame(d), !_ || !e.hasWorld) return;
    let c = parseInt(ne.value, 10);
    if ((isNaN(c) || c < 0) && (c = 0), c > 0 && g >= c) {
      e.update(n / c / 30);
      let f = false;
      f || (f = t.updateGeometryBuffer("tlas", e.tlas)), f || (f = t.updateGeometryBuffer("blas", e.blas)), f || (f = t.updateGeometryBuffer("instance", e.instances)), f || (f = t.updateGeometryBuffer("vertex", e.vertices)), f || (f = t.updateGeometryBuffer("normal", e.normals)), f || (f = t.updateGeometryBuffer("index", e.indices)), f || (f = t.updateGeometryBuffer("attr", e.attributes)), f && t.recreateBindGroup(), t.resetAccumulation(), g = 0;
    }
    g++, u++, n++, t.render(g);
    const l = performance.now();
    l - a >= 1e3 && (I.textContent = `FPS: ${u} | ${(1e3 / u).toFixed(2)}ms | Frame: ${g}`, u = 0, a = l);
  };
  T.addEventListener("click", () => {
    _ = !_, T.textContent = _ ? "Stop Rendering" : "Resume Rendering";
  }), F.addEventListener("change", (c) => o(c.target.value, false)), C.addEventListener("change", r), M.addEventListener("change", r), te.addEventListener("click", () => {
    _ = false, s(), t.recreateBindGroup(), t.resetAccumulation(), g = 0, _ = true;
  }), A.addEventListener("change", async (c) => {
    var _a, _b;
    const l = (_a = c.target.files) == null ? void 0 : _a[0];
    if (!l) return;
    ((_b = l.name.split(".").pop()) == null ? void 0 : _b.toLowerCase()) === "obj" ? (h = await l.text(), y = "obj") : (h = await l.arrayBuffer(), y = "glb"), F.value = "viewer", o("viewer", false);
  }), P.addEventListener("change", async (c) => {
    var _a;
    const l = (_a = c.target.files) == null ? void 0 : _a[0];
    if (!l) return;
    console.log(`Loading Motion: ${l.name}...`);
    const f = await l.arrayBuffer();
    e.loadAnimation(new Uint8Array(f)), console.log("Motion Loaded!"), c.target.value = "";
  }), r(), o("cornell", false), requestAnimationFrame(d);
}
re().catch(console.error);
