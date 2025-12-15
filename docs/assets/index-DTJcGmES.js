var __defProp = Object.defineProperty;
var __defNormalProp = (obj, key, value) => key in obj ? __defProp(obj, key, { enumerable: true, configurable: true, writable: true, value }) : obj[key] = value;
var __publicField = (obj, key, value) => __defNormalProp(obj, typeof key !== "symbol" ? key + "" : key, value);
(function() {
  const t = document.createElement("link").relList;
  if (t && t.supports && t.supports("modulepreload")) return;
  for (const s of document.querySelectorAll('link[rel="modulepreload"]')) n(s);
  new MutationObserver((s) => {
    for (const o of s) if (o.type === "childList") for (const f of o.addedNodes) f.tagName === "LINK" && f.rel === "modulepreload" && n(f);
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

// Bindings \u8FFD\u52A0
@group(0) @binding(11) var<storage, read> uvs: array<vec2<f32>>;
@group(0) @binding(12) var tex: texture_2d_array<f32>;
@group(0) @binding(13) var smp: sampler;

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
    var stack: array<u32, 64>;
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
    var stack: array<u32, 64>;
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


        // Intersection Interpolation
        let t0_uv = uvs[i0];
        let t1_uv = uvs[i1];
        let t2_uv = uvs[i2];
        let uv = t0_uv * w + t1_uv * u + t2_uv * v;

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
        let tex_idx = attr.data1.y;
        var tex_color = vec3(1.0);
        if (tex_idx > -0.5) {
            tex_color = textureSampleLevel(tex, smp, uv, i32(tex_idx), 0.0).rgb;
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
class Ei {
  constructor(t) {
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
    __publicField(this, "uvBuffer");
    __publicField(this, "texture");
    __publicField(this, "defaultTexture");
    __publicField(this, "sampler");
    __publicField(this, "bufferSize", 0);
    __publicField(this, "canvas");
    this.canvas = t;
  }
  async init() {
    if (!navigator.gpu) throw new Error("WebGPU not supported.");
    const t = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!t) throw new Error("No adapter");
    console.log("Max Storage Buffers Per Shader Stage:", t.limits.maxStorageBuffersPerShaderStage), this.device = await t.requestDevice({ requiredLimits: { maxStorageBuffersPerShaderStage: 16 } }), this.context = this.canvas.getContext("webgpu"), this.context.configure({ device: this.device, format: "rgba8unorm", usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT }), this.frameUniformBuffer = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }), this.cameraUniformBuffer = this.device.createBuffer({ size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }), this.sampler = this.device.createSampler({ magFilter: "linear", minFilter: "linear", mipmapFilter: "linear", addressModeU: "repeat", addressModeV: "repeat" }), this.createDefaultTexture(), this.texture = this.defaultTexture;
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
  async loadTexture(t) {
    const r = await createImageBitmap(t, { resizeWidth: 1024, resizeHeight: 1024 });
    this.texture && this.texture.destroy(), this.texture = this.device.createTexture({ size: [1024, 1024, 1], format: "rgba8unorm", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT }), this.device.queue.copyExternalImageToTexture({ source: r }, { texture: this.texture, origin: [0, 0, 0] }, [1024, 1024]);
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
      if (o) {
        const f = new Blob([o]);
        try {
          const h = await createImageBitmap(f, { resizeWidth: 1024, resizeHeight: 1024 });
          n.push(h);
        } catch (h) {
          console.warn(`Failed to load texture ${s}`, h), n.push(await this.createFallbackBitmap());
        }
      } else n.push(await this.createFallbackBitmap());
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
  updateGeometryBuffer(t, r) {
    let s = { tlas: this.tlasBuffer, blas: this.blasBuffer, instance: this.instanceBuffer, vertex: this.vertexBuffer, normal: this.normalBuffer, index: this.indexBuffer, attr: this.attrBuffer, uv: this.uvBuffer }[t];
    if (!s || s.size < r.byteLength) {
      s && s.destroy();
      const o = r.byteLength;
      let f = Math.ceil(o * 1.5);
      f = f + 3 & -4, f = Math.max(f, 4);
      const h = this.device.createBuffer({ size: f, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      switch (this.device.queue.writeBuffer(h, 0, r, 0, r.length), t) {
        case "tlas":
          this.tlasBuffer = h;
          break;
        case "blas":
          this.blasBuffer = h;
          break;
        case "instance":
          this.instanceBuffer = h;
          break;
        case "vertex":
          this.vertexBuffer = h;
          break;
        case "normal":
          this.normalBuffer = h;
          break;
        case "index":
          this.indexBuffer = h;
          break;
        case "attr":
          this.attrBuffer = h;
          break;
        case "uv":
          this.uvBuffer = h;
          break;
      }
      return true;
    } else return r.byteLength > 0 && this.device.queue.writeBuffer(s, 0, r, 0, r.length), false;
  }
  updateCameraBuffer(t) {
    this.device.queue.writeBuffer(this.cameraUniformBuffer, 0, t);
  }
  updateFrameBuffer(t) {
    this.device.queue.writeBuffer(this.frameUniformBuffer, 0, new Uint32Array([t]));
  }
  recreateBindGroup() {
    !this.renderTargetView || !this.accumulateBuffer || !this.vertexBuffer || !this.tlasBuffer || !this.uvBuffer || (this.bindGroup = this.device.createBindGroup({ layout: this.bindGroupLayout, entries: [{ binding: 0, resource: this.renderTargetView }, { binding: 1, resource: { buffer: this.accumulateBuffer } }, { binding: 2, resource: { buffer: this.frameUniformBuffer } }, { binding: 3, resource: { buffer: this.cameraUniformBuffer } }, { binding: 4, resource: { buffer: this.vertexBuffer } }, { binding: 5, resource: { buffer: this.indexBuffer } }, { binding: 6, resource: { buffer: this.attrBuffer } }, { binding: 7, resource: { buffer: this.tlasBuffer } }, { binding: 8, resource: { buffer: this.normalBuffer } }, { binding: 9, resource: { buffer: this.blasBuffer } }, { binding: 10, resource: { buffer: this.instanceBuffer } }, { binding: 11, resource: { buffer: this.uvBuffer } }, { binding: 12, resource: this.texture.createView({ dimension: "2d-array" }) }, { binding: 13, resource: this.sampler }] }));
  }
  render(t) {
    if (!this.bindGroup) return;
    this.updateFrameBuffer(t);
    const r = Math.ceil(this.canvas.width / 8), n = Math.ceil(this.canvas.height / 8), s = this.device.createCommandEncoder(), o = s.beginComputePass();
    o.setPipeline(this.pipeline), o.setBindGroup(0, this.bindGroup), o.dispatchWorkgroups(r, n), o.end(), s.copyTextureToTexture({ texture: this.renderTarget }, { texture: this.context.getCurrentTexture() }, { width: this.canvas.width, height: this.canvas.height, depthOrArrayLayers: 1 }), this.device.queue.submit([s.finish()]);
  }
}
let c;
function Si(e) {
  const t = c.__externref_table_alloc();
  return c.__wbindgen_externrefs.set(t, e), t;
}
function Ci(e, t) {
  return e = e >>> 0, re().subarray(e / 1, e / 1 + t);
}
let O = null;
function $t() {
  return (O === null || O.buffer.detached === true || O.buffer.detached === void 0 && O.buffer !== c.memory.buffer) && (O = new DataView(c.memory.buffer)), O;
}
function Ve(e, t) {
  return e = e >>> 0, Ui(e, t);
}
let pe = null;
function re() {
  return (pe === null || pe.byteLength === 0) && (pe = new Uint8Array(c.memory.buffer)), pe;
}
function Ii(e, t) {
  try {
    return e.apply(this, t);
  } catch (r) {
    const n = Si(r);
    c.__wbindgen_exn_store(n);
  }
}
function Ht(e) {
  return e == null;
}
function qt(e, t) {
  const r = t(e.length * 1, 1) >>> 0;
  return re().set(e, r / 1), R = e.length, r;
}
function ct(e, t, r) {
  if (r === void 0) {
    const h = ke.encode(e), b = t(h.length, 1) >>> 0;
    return re().subarray(b, b + h.length).set(h), R = h.length, b;
  }
  let n = e.length, s = t(n, 1) >>> 0;
  const o = re();
  let f = 0;
  for (; f < n; f++) {
    const h = e.charCodeAt(f);
    if (h > 127) break;
    o[s + f] = h;
  }
  if (f !== n) {
    f !== 0 && (e = e.slice(f)), s = r(s, n, n = f + e.length * 3, 1) >>> 0;
    const h = re().subarray(s + f, s + n), b = ke.encodeInto(e, h);
    f += b.written, s = r(s, n, f, 1) >>> 0;
  }
  return R = f, s;
}
let Oe = new TextDecoder("utf-8", { ignoreBOM: true, fatal: true });
Oe.decode();
const Mi = 2146435072;
let lt = 0;
function Ui(e, t) {
  return lt += t, lt >= Mi && (Oe = new TextDecoder("utf-8", { ignoreBOM: true, fatal: true }), Oe.decode(), lt = t), Oe.decode(re().subarray(e, e + t));
}
const ke = new TextEncoder();
"encodeInto" in ke || (ke.encodeInto = function(e, t) {
  const r = ke.encode(e);
  return t.set(r), { read: e.length, written: r.length };
});
let R = 0;
typeof FinalizationRegistry > "u" || new FinalizationRegistry((e) => c.__wbg_renderbuffers_free(e >>> 0, 1));
const jt = typeof FinalizationRegistry > "u" ? { register: () => {
}, unregister: () => {
} } : new FinalizationRegistry((e) => c.__wbg_world_free(e >>> 0, 1));
class ut {
  __destroy_into_raw() {
    const t = this.__wbg_ptr;
    return this.__wbg_ptr = 0, jt.unregister(this), t;
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
      return r = s[0], n = s[1], Ve(s[0], s[1]);
    } finally {
      c.__wbindgen_free(r, n, 1);
    }
  }
  load_animation_glb(t) {
    const r = qt(t, c.__wbindgen_malloc), n = R;
    c.world_load_animation_glb(this.__wbg_ptr, r, n);
  }
  get_animation_count() {
    return c.world_get_animation_count(this.__wbg_ptr) >>> 0;
  }
  constructor(t, r, n) {
    const s = ct(t, c.__wbindgen_malloc, c.__wbindgen_realloc), o = R;
    var f = Ht(r) ? 0 : ct(r, c.__wbindgen_malloc, c.__wbindgen_realloc), h = R, b = Ht(n) ? 0 : qt(n, c.__wbindgen_malloc), D = R;
    const st = c.world_new(s, o, f, h, b, D);
    return this.__wbg_ptr = st >>> 0, jt.register(this, this.__wbg_ptr, this), this;
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
Symbol.dispose && (ut.prototype[Symbol.dispose] = ut.prototype.free);
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
    throw new Error(Ve(t, r));
  }, e.wbg.__wbg_error_7534b8e9a36f1ab4 = function(t, r) {
    let n, s;
    try {
      n = t, s = r, console.error(Ve(t, r));
    } finally {
      c.__wbindgen_free(n, s, 1);
    }
  }, e.wbg.__wbg_getRandomValues_1c61fac11405ffdc = function() {
    return Ii(function(t, r) {
      globalThis.crypto.getRandomValues(Ci(t, r));
    }, arguments);
  }, e.wbg.__wbg_log_1d990106d99dacb7 = function(t) {
    console.log(t);
  }, e.wbg.__wbg_new_8a6f238a6ece86ea = function() {
    return new Error();
  }, e.wbg.__wbg_stack_0ed75d68575b0f3c = function(t, r) {
    const n = r.stack, s = ct(n, c.__wbindgen_malloc, c.__wbindgen_realloc), o = R;
    $t().setInt32(t + 4, o, true), $t().setInt32(t + 0, s, true);
  }, e.wbg.__wbindgen_cast_2241b6af4c4b2941 = function(t, r) {
    return Ve(t, r);
  }, e.wbg.__wbindgen_init_externref_table = function() {
    const t = c.__wbindgen_externrefs, r = t.grow(4);
    t.set(0, void 0), t.set(r + 0, void 0), t.set(r + 1, null), t.set(r + 2, true), t.set(r + 3, false);
  }, e;
}
function Li(e, t) {
  return c = e.exports, Zt.__wbindgen_wasm_module = t, O = null, pe = null, c.__wbindgen_start(), c;
}
async function Zt(e) {
  if (c !== void 0) return c;
  typeof e < "u" && (Object.getPrototypeOf(e) === Object.prototype ? { module_or_path: e } = e : console.warn("using deprecated parameters for the initialization function; pass a single object instead")), typeof e > "u" && (e = new URL("/webgpu-raytracer/assets/rust_shader_tools_bg-DSXiDK5e.wasm", import.meta.url));
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
    const t = await Zt();
    this.wasmMemory = t.memory, console.log("Wasm initialized");
  }
  loadScene(t, r, n) {
    this.world && this.world.free(), this.world = new ut(t, r, n);
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
}, _ = (e, t, r, n) => (Tt(e, t, "write to private field"), t.set(e, r), r), d = (e, t, r) => (Tt(e, t, "access private method"), r), Jt = class {
  constructor(e) {
    this.value = e;
  }
}, Bt = class {
  constructor(e) {
    this.value = e;
  }
}, ei = (e) => e < 256 ? 1 : e < 65536 ? 2 : e < 1 << 24 ? 3 : e < 2 ** 32 ? 4 : e < 2 ** 40 ? 5 : 6, Pi = (e) => {
  if (e < 127) return 1;
  if (e < 16383) return 2;
  if (e < (1 << 21) - 1) return 3;
  if (e < (1 << 28) - 1) return 4;
  if (e < 2 ** 35 - 1) return 5;
  if (e < 2 ** 42 - 1) return 6;
  throw new Error("EBML VINT size not supported " + e);
}, Q = (e, t, r) => {
  let n = 0;
  for (let s = t; s < r; s++) {
    let o = Math.floor(s / 8), f = e[o], h = 7 - (s & 7), b = (f & 1 << h) >> h;
    n <<= 1, n |= b;
  }
  return n;
}, Fi = (e, t, r, n) => {
  for (let s = t; s < r; s++) {
    let o = Math.floor(s / 8), f = e[o], h = 7 - (s & 7);
    f &= ~(1 << h), f |= (n & 1 << r - s - 1) >> r - s - 1 << h, e[o] = f;
  }
}, it = class {
}, ti = class extends it {
  constructor() {
    super(...arguments), this.buffer = null;
  }
}, ii = class extends it {
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
}, Di = class extends it {
  constructor(e, t) {
    if (super(), this.stream = e, this.options = t, !(e instanceof FileSystemWritableFileStream)) throw new TypeError("FileSystemWritableFileStreamTarget requires a FileSystemWritableFileStream instance.");
    if (t !== void 0 && typeof t != "object") throw new TypeError("FileSystemWritableFileStreamTarget's options, when provided, must be an object.");
    if (t && t.chunkSize !== void 0 && (!Number.isInteger(t.chunkSize) || t.chunkSize <= 0)) throw new TypeError("options.chunkSize, when provided, must be a positive integer");
  }
}, P, w, ht, ri, ft, ni, _t, si, $e, mt, wt, ai, oi = class {
  constructor() {
    a(this, ht), a(this, ft), a(this, _t), a(this, $e), a(this, wt), this.pos = 0, a(this, P, new Uint8Array(8)), a(this, w, new DataView(i(this, P).buffer)), this.offsets = /* @__PURE__ */ new WeakMap(), this.dataOffsets = /* @__PURE__ */ new WeakMap();
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
    this.write(i(this, P).subarray(0, r));
  }
  writeEBML(e) {
    if (e !== null) if (e instanceof Uint8Array) this.write(e);
    else if (Array.isArray(e)) for (let t of e) this.writeEBML(t);
    else if (this.offsets.set(e, this.pos), d(this, $e, mt).call(this, e.id), Array.isArray(e.data)) {
      let t = this.pos, r = e.size === -1 ? 1 : e.size ?? 4;
      e.size === -1 ? d(this, ht, ri).call(this, 255) : this.seek(this.pos + r);
      let n = this.pos;
      if (this.dataOffsets.set(e, n), this.writeEBML(e.data), e.size !== -1) {
        let s = this.pos - n, o = this.pos;
        this.seek(t), this.writeEBMLVarInt(s, r), this.seek(o);
      }
    } else if (typeof e.data == "number") {
      let t = e.size ?? ei(e.data);
      this.writeEBMLVarInt(t), d(this, $e, mt).call(this, e.data, t);
    } else typeof e.data == "string" ? (this.writeEBMLVarInt(e.data.length), d(this, wt, ai).call(this, e.data)) : e.data instanceof Uint8Array ? (this.writeEBMLVarInt(e.data.byteLength, e.size), this.write(e.data)) : e.data instanceof Jt ? (this.writeEBMLVarInt(4), d(this, ft, ni).call(this, e.data.value)) : e.data instanceof Bt && (this.writeEBMLVarInt(8), d(this, _t, si).call(this, e.data.value));
  }
};
P = /* @__PURE__ */ new WeakMap();
w = /* @__PURE__ */ new WeakMap();
ht = /* @__PURE__ */ new WeakSet();
ri = function(e) {
  i(this, w).setUint8(0, e), this.write(i(this, P).subarray(0, 1));
};
ft = /* @__PURE__ */ new WeakSet();
ni = function(e) {
  i(this, w).setFloat32(0, e, false), this.write(i(this, P).subarray(0, 4));
};
_t = /* @__PURE__ */ new WeakSet();
si = function(e) {
  i(this, w).setFloat64(0, e, false), this.write(i(this, P));
};
$e = /* @__PURE__ */ new WeakSet();
mt = function(e, t = ei(e)) {
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
  this.write(i(this, P).subarray(0, r));
};
wt = /* @__PURE__ */ new WeakSet();
ai = function(e) {
  this.write(new Uint8Array(e.split("").map((t) => t.charCodeAt(0))));
};
var He, X, Ue, qe, gt, Gi = class extends oi {
  constructor(e) {
    super(), a(this, qe), a(this, He, void 0), a(this, X, new ArrayBuffer(2 ** 16)), a(this, Ue, new Uint8Array(i(this, X))), _(this, He, e);
  }
  write(e) {
    d(this, qe, gt).call(this, this.pos + e.byteLength), i(this, Ue).set(e, this.pos), this.pos += e.byteLength;
  }
  finalize() {
    d(this, qe, gt).call(this, this.pos), i(this, He).buffer = i(this, X).slice(0, this.pos);
  }
};
He = /* @__PURE__ */ new WeakMap();
X = /* @__PURE__ */ new WeakMap();
Ue = /* @__PURE__ */ new WeakMap();
qe = /* @__PURE__ */ new WeakSet();
gt = function(e) {
  let t = i(this, X).byteLength;
  for (; t < e; ) t *= 2;
  if (t === i(this, X).byteLength) return;
  let r = new ArrayBuffer(t), n = new Uint8Array(r);
  n.set(i(this, Ue), 0), _(this, X, r), _(this, Ue, n);
};
var J, k, T, $, Pe = class extends oi {
  constructor(e) {
    super(), this.target = e, a(this, J, false), a(this, k, void 0), a(this, T, void 0), a(this, $, void 0);
  }
  write(e) {
    if (!i(this, J)) return;
    let t = this.pos;
    if (t < i(this, T)) {
      if (t + e.byteLength <= i(this, T)) return;
      e = e.subarray(i(this, T) - t), t = 0;
    }
    let r = t + e.byteLength - i(this, T), n = i(this, k).byteLength;
    for (; n < r; ) n *= 2;
    if (n !== i(this, k).byteLength) {
      let s = new Uint8Array(n);
      s.set(i(this, k), 0), _(this, k, s);
    }
    i(this, k).set(e, t - i(this, T)), _(this, $, Math.max(i(this, $), t + e.byteLength));
  }
  startTrackingWrites() {
    _(this, J, true), _(this, k, new Uint8Array(2 ** 10)), _(this, T, this.pos), _(this, $, this.pos);
  }
  getTrackedWrites() {
    if (!i(this, J)) throw new Error("Can't get tracked writes since nothing was tracked.");
    let t = { data: i(this, k).subarray(0, i(this, $) - i(this, T)), start: i(this, T), end: i(this, $) };
    return _(this, k, void 0), _(this, J, false), t;
  }
};
J = /* @__PURE__ */ new WeakMap();
k = /* @__PURE__ */ new WeakMap();
T = /* @__PURE__ */ new WeakMap();
$ = /* @__PURE__ */ new WeakMap();
var Ni = 2 ** 24, Vi = 2, H, ne, Te, be, M, v, Ke, pt, Et, di, St, li, Be, Qe, Ct = class extends Pe {
  constructor(e, t) {
    var _a, _b;
    super(e), a(this, Ke), a(this, Et), a(this, St), a(this, Be), a(this, H, []), a(this, ne, 0), a(this, Te, void 0), a(this, be, void 0), a(this, M, void 0), a(this, v, []), _(this, Te, t), _(this, be, ((_a = e.options) == null ? void 0 : _a.chunked) ?? false), _(this, M, ((_b = e.options) == null ? void 0 : _b.chunkSize) ?? Ni);
  }
  write(e) {
    super.write(e), i(this, H).push({ data: e.slice(), start: this.pos }), this.pos += e.byteLength;
  }
  flush() {
    var _a, _b;
    if (i(this, H).length === 0) return;
    let e = [], t = [...i(this, H)].sort((r, n) => r.start - n.start);
    e.push({ start: t[0].start, size: t[0].data.byteLength });
    for (let r = 1; r < t.length; r++) {
      let n = e[e.length - 1], s = t[r];
      s.start <= n.start + n.size ? n.size = Math.max(n.size, s.start + s.data.byteLength - n.start) : e.push({ start: s.start, size: s.data.byteLength });
    }
    for (let r of e) {
      r.data = new Uint8Array(r.size);
      for (let n of i(this, H)) r.start <= n.start && n.start < r.start + r.size && r.data.set(n.data, n.start - r.start);
      if (i(this, be)) d(this, Ke, pt).call(this, r.data, r.start), d(this, Be, Qe).call(this);
      else {
        if (i(this, Te) && r.start < i(this, ne)) throw new Error("Internal error: Monotonicity violation.");
        (_b = (_a = this.target.options).onData) == null ? void 0 : _b.call(_a, r.data, r.start), _(this, ne, r.start + r.data.byteLength);
      }
    }
    i(this, H).length = 0;
  }
  finalize() {
    i(this, be) && d(this, Be, Qe).call(this, true);
  }
};
H = /* @__PURE__ */ new WeakMap();
ne = /* @__PURE__ */ new WeakMap();
Te = /* @__PURE__ */ new WeakMap();
be = /* @__PURE__ */ new WeakMap();
M = /* @__PURE__ */ new WeakMap();
v = /* @__PURE__ */ new WeakMap();
Ke = /* @__PURE__ */ new WeakSet();
pt = function(e, t) {
  let r = i(this, v).findIndex((h) => h.start <= t && t < h.start + i(this, M));
  r === -1 && (r = d(this, St, li).call(this, t));
  let n = i(this, v)[r], s = t - n.start, o = e.subarray(0, Math.min(i(this, M) - s, e.byteLength));
  n.data.set(o, s);
  let f = { start: s, end: s + o.byteLength };
  if (d(this, Et, di).call(this, n, f), n.written[0].start === 0 && n.written[0].end === i(this, M) && (n.shouldFlush = true), i(this, v).length > Vi) {
    for (let h = 0; h < i(this, v).length - 1; h++) i(this, v)[h].shouldFlush = true;
    d(this, Be, Qe).call(this);
  }
  o.byteLength < e.byteLength && d(this, Ke, pt).call(this, e.subarray(o.byteLength), t + o.byteLength);
};
Et = /* @__PURE__ */ new WeakSet();
di = function(e, t) {
  let r = 0, n = e.written.length - 1, s = -1;
  for (; r <= n; ) {
    let o = Math.floor(r + (n - r + 1) / 2);
    e.written[o].start <= t.start ? (r = o + 1, s = o) : n = o - 1;
  }
  for (e.written.splice(s + 1, 0, t), (s === -1 || e.written[s].end < t.start) && s++; s < e.written.length - 1 && e.written[s].end >= e.written[s + 1].start; ) e.written[s].end = Math.max(e.written[s].end, e.written[s + 1].end), e.written.splice(s + 1, 1);
};
St = /* @__PURE__ */ new WeakSet();
li = function(e) {
  let r = { start: Math.floor(e / i(this, M)) * i(this, M), data: new Uint8Array(i(this, M)), written: [], shouldFlush: false };
  return i(this, v).push(r), i(this, v).sort((n, s) => n.start - s.start), i(this, v).indexOf(r);
};
Be = /* @__PURE__ */ new WeakSet();
Qe = function(e = false) {
  var _a, _b;
  for (let t = 0; t < i(this, v).length; t++) {
    let r = i(this, v)[t];
    if (!(!r.shouldFlush && !e)) {
      for (let n of r.written) {
        if (i(this, Te) && r.start + n.start < i(this, ne)) throw new Error("Internal error: Monotonicity violation.");
        (_b = (_a = this.target.options).onData) == null ? void 0 : _b.call(_a, r.data.subarray(n.start, n.end), r.start + n.start), _(this, ne, r.start + n.end);
      }
      i(this, v).splice(t--, 1);
    }
  }
};
var Oi = class extends Ct {
  constructor(e, t) {
    var _a;
    super(new ii({ onData: (r, n) => e.stream.write({ type: "write", data: r, position: n }), chunked: true, chunkSize: (_a = e.options) == null ? void 0 : _a.chunkSize }), t);
  }
}, ue = 1, Ae = 2, Ze = 3, $i = 1, Hi = 2, qi = 17, ji = 2 ** 15, Ee = 2 ** 13, Yt = "https://github.com/Vanilagy/webm-muxer", ci = 6, ui = 5, Yi = ["strict", "offset", "permissive"], u, l, ze, We, I, he, ee, K, fe, F, se, ae, E, Fe, oe, W, L, q, Se, Ce, de, le, Je, Le, Ie, bt, hi, vt, fi, It, _i, Mt, mi, Ut, wi, At, gi, zt, pi, rt, Wt, nt, Lt, Rt, bi, j, te, Y, ie, yt, vi, xt, yi, ve, je, ye, Ye, Pt, xi, B, C, ce, Re, Me, et, Ft, ki, tt, Dt, xe, Xe, Xi = class {
  constructor(e) {
    a(this, bt), a(this, vt), a(this, It), a(this, Mt), a(this, Ut), a(this, At), a(this, zt), a(this, rt), a(this, nt), a(this, Rt), a(this, j), a(this, Y), a(this, yt), a(this, xt), a(this, ve), a(this, ye), a(this, Pt), a(this, B), a(this, ce), a(this, Me), a(this, Ft), a(this, tt), a(this, xe), a(this, u, void 0), a(this, l, void 0), a(this, ze, void 0), a(this, We, void 0), a(this, I, void 0), a(this, he, void 0), a(this, ee, void 0), a(this, K, void 0), a(this, fe, void 0), a(this, F, void 0), a(this, se, void 0), a(this, ae, void 0), a(this, E, void 0), a(this, Fe, void 0), a(this, oe, 0), a(this, W, []), a(this, L, []), a(this, q, []), a(this, Se, void 0), a(this, Ce, void 0), a(this, de, -1), a(this, le, -1), a(this, Je, -1), a(this, Le, void 0), a(this, Ie, false), d(this, bt, hi).call(this, e), _(this, u, { type: "webm", firstTimestampBehavior: "strict", ...e }), this.target = e.target;
    let t = !!i(this, u).streaming;
    if (e.target instanceof ti) _(this, l, new Gi(e.target));
    else if (e.target instanceof ii) _(this, l, new Ct(e.target, t));
    else if (e.target instanceof Di) _(this, l, new Oi(e.target, t));
    else throw new Error(`Invalid target: ${e.target}`);
    d(this, vt, fi).call(this);
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
    if (d(this, xe, Xe).call(this), !i(this, u).video) throw new Error("No video track declared.");
    i(this, Se) === void 0 && _(this, Se, r), n && d(this, yt, vi).call(this, n);
    let s = d(this, ye, Ye).call(this, e, t, r, ue);
    for (i(this, u).video.codec === "V_VP9" && d(this, xt, yi).call(this, s), _(this, de, s.timestamp); i(this, L).length > 0 && i(this, L)[0].timestamp <= s.timestamp; ) {
      let o = i(this, L).shift();
      d(this, B, C).call(this, o, false);
    }
    !i(this, u).audio || s.timestamp <= i(this, le) ? d(this, B, C).call(this, s, true) : i(this, W).push(s), d(this, ve, je).call(this), d(this, j, te).call(this);
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
    if (d(this, xe, Xe).call(this), !i(this, u).audio) throw new Error("No audio track declared.");
    i(this, Ce) === void 0 && _(this, Ce, r), (n == null ? void 0 : n.decoderConfig) && (i(this, u).streaming ? _(this, F, d(this, ce, Re).call(this, n.decoderConfig.description)) : d(this, Me, et).call(this, i(this, F), n.decoderConfig.description));
    let s = d(this, ye, Ye).call(this, e, t, r, Ae);
    for (_(this, le, s.timestamp); i(this, W).length > 0 && i(this, W)[0].timestamp <= s.timestamp; ) {
      let o = i(this, W).shift();
      d(this, B, C).call(this, o, true);
    }
    !i(this, u).video || s.timestamp <= i(this, de) ? d(this, B, C).call(this, s, !i(this, u).video) : i(this, L).push(s), d(this, ve, je).call(this), d(this, j, te).call(this);
  }
  addSubtitleChunk(e, t, r) {
    if (typeof e != "object" || !e) throw new TypeError("addSubtitleChunk's first argument (chunk) must be an object.");
    if (!(e.body instanceof Uint8Array)) throw new TypeError("body must be an instance of Uint8Array.");
    if (!Number.isFinite(e.timestamp) || e.timestamp < 0) throw new TypeError("timestamp must be a non-negative real number.");
    if (!Number.isFinite(e.duration) || e.duration < 0) throw new TypeError("duration must be a non-negative real number.");
    if (e.additions && !(e.additions instanceof Uint8Array)) throw new TypeError("additions, when present, must be an instance of Uint8Array.");
    if (typeof t != "object") throw new TypeError("addSubtitleChunk's second argument (meta) must be an object.");
    if (d(this, xe, Xe).call(this), !i(this, u).subtitles) throw new Error("No subtitle track declared.");
    (t == null ? void 0 : t.decoderConfig) && (i(this, u).streaming ? _(this, se, d(this, ce, Re).call(this, t.decoderConfig.description)) : d(this, Me, et).call(this, i(this, se), t.decoderConfig.description));
    let n = d(this, ye, Ye).call(this, e.body, "key", r ?? e.timestamp, Ze, e.duration, e.additions);
    _(this, Je, n.timestamp), i(this, q).push(n), d(this, ve, je).call(this), d(this, j, te).call(this);
  }
  finalize() {
    if (i(this, Ie)) throw new Error("Cannot finalize a muxer more than once.");
    for (; i(this, W).length > 0; ) d(this, B, C).call(this, i(this, W).shift(), true);
    for (; i(this, L).length > 0; ) d(this, B, C).call(this, i(this, L).shift(), true);
    for (; i(this, q).length > 0 && i(this, q)[0].timestamp <= i(this, oe); ) d(this, B, C).call(this, i(this, q).shift(), false);
    if (i(this, E) && d(this, tt, Dt).call(this), i(this, l).writeEBML(i(this, ae)), !i(this, u).streaming) {
      let e = i(this, l).pos, t = i(this, l).pos - i(this, Y, ie);
      i(this, l).seek(i(this, l).offsets.get(i(this, ze)) + 4), i(this, l).writeEBMLVarInt(t, ci), i(this, ee).data = new Bt(i(this, oe)), i(this, l).seek(i(this, l).offsets.get(i(this, ee))), i(this, l).writeEBML(i(this, ee)), i(this, I).data[0].data[1].data = i(this, l).offsets.get(i(this, ae)) - i(this, Y, ie), i(this, I).data[1].data[1].data = i(this, l).offsets.get(i(this, We)) - i(this, Y, ie), i(this, I).data[2].data[1].data = i(this, l).offsets.get(i(this, he)) - i(this, Y, ie), i(this, l).seek(i(this, l).offsets.get(i(this, I))), i(this, l).writeEBML(i(this, I)), i(this, l).seek(e);
    }
    d(this, j, te).call(this), i(this, l).finalize(), _(this, Ie, true);
  }
};
u = /* @__PURE__ */ new WeakMap();
l = /* @__PURE__ */ new WeakMap();
ze = /* @__PURE__ */ new WeakMap();
We = /* @__PURE__ */ new WeakMap();
I = /* @__PURE__ */ new WeakMap();
he = /* @__PURE__ */ new WeakMap();
ee = /* @__PURE__ */ new WeakMap();
K = /* @__PURE__ */ new WeakMap();
fe = /* @__PURE__ */ new WeakMap();
F = /* @__PURE__ */ new WeakMap();
se = /* @__PURE__ */ new WeakMap();
ae = /* @__PURE__ */ new WeakMap();
E = /* @__PURE__ */ new WeakMap();
Fe = /* @__PURE__ */ new WeakMap();
oe = /* @__PURE__ */ new WeakMap();
W = /* @__PURE__ */ new WeakMap();
L = /* @__PURE__ */ new WeakMap();
q = /* @__PURE__ */ new WeakMap();
Se = /* @__PURE__ */ new WeakMap();
Ce = /* @__PURE__ */ new WeakMap();
de = /* @__PURE__ */ new WeakMap();
le = /* @__PURE__ */ new WeakMap();
Je = /* @__PURE__ */ new WeakMap();
Le = /* @__PURE__ */ new WeakMap();
Ie = /* @__PURE__ */ new WeakMap();
bt = /* @__PURE__ */ new WeakSet();
hi = function(e) {
  if (typeof e != "object") throw new TypeError("The muxer requires an options object to be passed to its constructor.");
  if (!(e.target instanceof it)) throw new TypeError("The target must be provided and an instance of Target.");
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
fi = function() {
  i(this, l) instanceof Pe && i(this, l).target.options.onHeader && i(this, l).startTrackingWrites(), d(this, It, _i).call(this), i(this, u).streaming || d(this, At, gi).call(this), d(this, zt, pi).call(this), d(this, Mt, mi).call(this), d(this, Ut, wi).call(this), i(this, u).streaming || (d(this, rt, Wt).call(this), d(this, nt, Lt).call(this)), d(this, Rt, bi).call(this), d(this, j, te).call(this);
};
It = /* @__PURE__ */ new WeakSet();
_i = function() {
  let e = { id: 440786851, data: [{ id: 17030, data: 1 }, { id: 17143, data: 1 }, { id: 17138, data: 4 }, { id: 17139, data: 8 }, { id: 17026, data: i(this, u).type ?? "webm" }, { id: 17031, data: 2 }, { id: 17029, data: 2 }] };
  i(this, l).writeEBML(e);
};
Mt = /* @__PURE__ */ new WeakSet();
mi = function() {
  _(this, fe, { id: 236, size: 4, data: new Uint8Array(Ee) }), _(this, F, { id: 236, size: 4, data: new Uint8Array(Ee) }), _(this, se, { id: 236, size: 4, data: new Uint8Array(Ee) });
};
Ut = /* @__PURE__ */ new WeakSet();
wi = function() {
  _(this, K, { id: 21936, data: [{ id: 21937, data: 2 }, { id: 21946, data: 2 }, { id: 21947, data: 2 }, { id: 21945, data: 0 }] });
};
At = /* @__PURE__ */ new WeakSet();
gi = function() {
  const e = new Uint8Array([28, 83, 187, 107]), t = new Uint8Array([21, 73, 169, 102]), r = new Uint8Array([22, 84, 174, 107]);
  _(this, I, { id: 290298740, data: [{ id: 19899, data: [{ id: 21419, data: e }, { id: 21420, size: 5, data: 0 }] }, { id: 19899, data: [{ id: 21419, data: t }, { id: 21420, size: 5, data: 0 }] }, { id: 19899, data: [{ id: 21419, data: r }, { id: 21420, size: 5, data: 0 }] }] });
};
zt = /* @__PURE__ */ new WeakSet();
pi = function() {
  let e = { id: 17545, data: new Bt(0) };
  _(this, ee, e);
  let t = { id: 357149030, data: [{ id: 2807729, data: 1e6 }, { id: 19840, data: Yt }, { id: 22337, data: Yt }, i(this, u).streaming ? null : e] };
  _(this, We, t);
};
rt = /* @__PURE__ */ new WeakSet();
Wt = function() {
  let e = { id: 374648427, data: [] };
  _(this, he, e), i(this, u).video && e.data.push({ id: 174, data: [{ id: 215, data: ue }, { id: 29637, data: ue }, { id: 131, data: $i }, { id: 134, data: i(this, u).video.codec }, i(this, fe), i(this, u).video.frameRate ? { id: 2352003, data: 1e9 / i(this, u).video.frameRate } : null, { id: 224, data: [{ id: 176, data: i(this, u).video.width }, { id: 186, data: i(this, u).video.height }, i(this, u).video.alpha ? { id: 21440, data: 1 } : null, i(this, K)] }] }), i(this, u).audio && (_(this, F, i(this, u).streaming ? i(this, F) || null : { id: 236, size: 4, data: new Uint8Array(Ee) }), e.data.push({ id: 174, data: [{ id: 215, data: Ae }, { id: 29637, data: Ae }, { id: 131, data: Hi }, { id: 134, data: i(this, u).audio.codec }, i(this, F), { id: 225, data: [{ id: 181, data: new Jt(i(this, u).audio.sampleRate) }, { id: 159, data: i(this, u).audio.numberOfChannels }, i(this, u).audio.bitDepth ? { id: 25188, data: i(this, u).audio.bitDepth } : null] }] })), i(this, u).subtitles && e.data.push({ id: 174, data: [{ id: 215, data: Ze }, { id: 29637, data: Ze }, { id: 131, data: qi }, { id: 134, data: i(this, u).subtitles.codec }, i(this, se)] });
};
nt = /* @__PURE__ */ new WeakSet();
Lt = function() {
  let e = { id: 408125543, size: i(this, u).streaming ? -1 : ci, data: [i(this, u).streaming ? null : i(this, I), i(this, We), i(this, he)] };
  if (_(this, ze, e), i(this, l).writeEBML(e), i(this, l) instanceof Pe && i(this, l).target.options.onHeader) {
    let { data: t, start: r } = i(this, l).getTrackedWrites();
    i(this, l).target.options.onHeader(t, r);
  }
};
Rt = /* @__PURE__ */ new WeakSet();
bi = function() {
  _(this, ae, { id: 475249515, data: [] });
};
j = /* @__PURE__ */ new WeakSet();
te = function() {
  i(this, l) instanceof Ct && i(this, l).flush();
};
Y = /* @__PURE__ */ new WeakSet();
ie = function() {
  return i(this, l).dataOffsets.get(i(this, ze));
};
yt = /* @__PURE__ */ new WeakSet();
vi = function(e) {
  if (e.decoderConfig) {
    if (e.decoderConfig.colorSpace) {
      let t = e.decoderConfig.colorSpace;
      if (_(this, Le, t), i(this, K).data = [{ id: 21937, data: { rgb: 1, bt709: 1, bt470bg: 5, smpte170m: 6 }[t.matrix] }, { id: 21946, data: { bt709: 1, smpte170m: 6, "iec61966-2-1": 13 }[t.transfer] }, { id: 21947, data: { bt709: 1, bt470bg: 5, smpte170m: 6 }[t.primaries] }, { id: 21945, data: [1, 2][Number(t.fullRange)] }], !i(this, u).streaming) {
        let r = i(this, l).pos;
        i(this, l).seek(i(this, l).offsets.get(i(this, K))), i(this, l).writeEBML(i(this, K)), i(this, l).seek(r);
      }
    }
    e.decoderConfig.description && (i(this, u).streaming ? _(this, fe, d(this, ce, Re).call(this, e.decoderConfig.description)) : d(this, Me, et).call(this, i(this, fe), e.decoderConfig.description));
  }
};
xt = /* @__PURE__ */ new WeakSet();
yi = function(e) {
  if (e.type !== "key" || !i(this, Le)) return;
  let t = 0;
  if (Q(e.data, 0, 2) !== 2) return;
  t += 2;
  let r = (Q(e.data, t + 1, t + 2) << 1) + Q(e.data, t + 0, t + 1);
  t += 2, r === 3 && t++;
  let n = Q(e.data, t + 0, t + 1);
  if (t++, n) return;
  let s = Q(e.data, t + 0, t + 1);
  if (t++, s !== 0) return;
  t += 2;
  let o = Q(e.data, t + 0, t + 24);
  if (t += 24, o !== 4817730) return;
  r >= 2 && t++;
  let f = { rgb: 7, bt709: 2, bt470bg: 1, smpte170m: 3 }[i(this, Le).matrix];
  Fi(e.data, t + 0, t + 3, f);
};
ve = /* @__PURE__ */ new WeakSet();
je = function() {
  let e = Math.min(i(this, u).video ? i(this, de) : 1 / 0, i(this, u).audio ? i(this, le) : 1 / 0), t = i(this, q);
  for (; t.length > 0 && t[0].timestamp <= e; ) d(this, B, C).call(this, t.shift(), !i(this, u).video && !i(this, u).audio);
};
ye = /* @__PURE__ */ new WeakSet();
Ye = function(e, t, r, n, s, o) {
  let f = d(this, Pt, xi).call(this, r, n);
  return { data: e, additions: o, type: t, timestamp: f, duration: s, trackNumber: n };
};
Pt = /* @__PURE__ */ new WeakSet();
xi = function(e, t) {
  let r = t === ue ? i(this, de) : t === Ae ? i(this, le) : i(this, Je);
  if (t !== Ze) {
    let n = t === ue ? i(this, Se) : i(this, Ce);
    if (i(this, u).firstTimestampBehavior === "strict" && r === -1 && e !== 0) throw new Error(`The first chunk for your media track must have a timestamp of 0 (received ${e}). Non-zero first timestamps are often caused by directly piping frames or audio data from a MediaStreamTrack into the encoder. Their timestamps are typically relative to the age of the document, which is probably what you want.

If you want to offset all timestamps of a track such that the first one is zero, set firstTimestampBehavior: 'offset' in the options.
If you want to allow non-zero first timestamps, set firstTimestampBehavior: 'permissive'.
`);
    i(this, u).firstTimestampBehavior === "offset" && (e -= n);
  }
  if (e < r) throw new Error(`Timestamps must be monotonically increasing (went from ${r} to ${e}).`);
  if (e < 0) throw new Error(`Timestamps must be non-negative (received ${e}).`);
  return e;
};
B = /* @__PURE__ */ new WeakSet();
C = function(e, t) {
  i(this, u).streaming && !i(this, he) && (d(this, rt, Wt).call(this), d(this, nt, Lt).call(this));
  let r = Math.floor(e.timestamp / 1e3), n = r - i(this, Fe), s = t && e.type === "key" && n >= 1e3, o = n >= ji;
  if ((!i(this, E) || s || o) && (d(this, Ft, ki).call(this, r), n = 0), n < 0) return;
  let f = new Uint8Array(4), h = new DataView(f.buffer);
  if (h.setUint8(0, 128 | e.trackNumber), h.setInt16(1, n, false), e.duration === void 0 && !e.additions) {
    h.setUint8(3, +(e.type === "key") << 7);
    let b = { id: 163, data: [f, e.data] };
    i(this, l).writeEBML(b);
  } else {
    let b = Math.floor(e.duration / 1e3), D = { id: 160, data: [{ id: 161, data: [f, e.data] }, e.duration !== void 0 ? { id: 155, data: b } : null, e.additions ? { id: 30113, data: e.additions } : null] };
    i(this, l).writeEBML(D);
  }
  _(this, oe, Math.max(i(this, oe), r));
};
ce = /* @__PURE__ */ new WeakSet();
Re = function(e) {
  return { id: 25506, size: 4, data: new Uint8Array(e) };
};
Me = /* @__PURE__ */ new WeakSet();
et = function(e, t) {
  let r = i(this, l).pos;
  i(this, l).seek(i(this, l).offsets.get(e));
  let n = 6 + t.byteLength, s = Ee - n;
  if (s < 0) {
    let o = t.byteLength + s;
    t instanceof ArrayBuffer ? t = t.slice(0, o) : t = t.buffer.slice(0, o), s = 0;
  }
  e = [d(this, ce, Re).call(this, t), { id: 236, size: 4, data: new Uint8Array(s) }], i(this, l).writeEBML(e), i(this, l).seek(r);
};
Ft = /* @__PURE__ */ new WeakSet();
ki = function(e) {
  i(this, E) && d(this, tt, Dt).call(this), i(this, l) instanceof Pe && i(this, l).target.options.onCluster && i(this, l).startTrackingWrites(), _(this, E, { id: 524531317, size: i(this, u).streaming ? -1 : ui, data: [{ id: 231, data: e }] }), i(this, l).writeEBML(i(this, E)), _(this, Fe, e);
  let t = i(this, l).offsets.get(i(this, E)) - i(this, Y, ie);
  i(this, ae).data.push({ id: 187, data: [{ id: 179, data: e }, i(this, u).video ? { id: 183, data: [{ id: 247, data: ue }, { id: 241, data: t }] } : null, i(this, u).audio ? { id: 183, data: [{ id: 247, data: Ae }, { id: 241, data: t }] } : null] });
};
tt = /* @__PURE__ */ new WeakSet();
Dt = function() {
  if (!i(this, u).streaming) {
    let e = i(this, l).pos - i(this, l).dataOffsets.get(i(this, E)), t = i(this, l).pos;
    i(this, l).seek(i(this, l).offsets.get(i(this, E)) + 4), i(this, l).writeEBMLVarInt(e, ui), i(this, l).seek(t);
  }
  if (i(this, l) instanceof Pe && i(this, l).target.options.onCluster) {
    let { data: e, start: t } = i(this, l).getTrackedWrites();
    i(this, l).target.options.onCluster(e, t, i(this, Fe));
  }
};
xe = /* @__PURE__ */ new WeakSet();
Xe = function() {
  if (i(this, Ie)) throw new Error("Cannot add new video or audio chunks after the file has been finalized.");
};
new TextEncoder();
const Z = document.getElementById("gpu-canvas"), S = document.getElementById("render-btn"), Xt = document.getElementById("scene-select"), Kt = document.getElementById("res-width"), Qt = document.getElementById("res-height"), kt = document.getElementById("obj-file");
kt && (kt.accept = ".obj,.glb,.vrm");
const Ki = document.getElementById("max-depth"), Qi = document.getElementById("spp-frame"), Zi = document.getElementById("recompile-btn"), Ji = document.getElementById("update-interval"), A = document.getElementById("anim-select"), z = document.getElementById("record-btn"), er = document.getElementById("rec-fps"), tr = document.getElementById("rec-duration"), ir = document.getElementById("rec-spp"), rr = document.getElementById("rec-batch"), Gt = document.createElement("div");
Object.assign(Gt.style, { position: "fixed", bottom: "10px", left: "10px", color: "#0f0", background: "rgba(0,0,0,0.7)", padding: "8px", fontFamily: "monospace", fontSize: "14px", pointerEvents: "none", zIndex: "9999", borderRadius: "4px" });
document.body.appendChild(Gt);
let V = 0, x = false, Ge = false, ge = null, Ne = null;
async function nr() {
  const e = new Ei(Z), t = new Ri();
  let r = 0;
  try {
    await e.init(), await t.initWasm();
  } catch (m) {
    alert("Initialization failed: " + m), console.error(m);
    return;
  }
  const n = () => {
    const m = parseInt(Ki.value, 10) || 10, g = parseInt(Qi.value, 10) || 1;
    e.buildPipeline(m, g);
  };
  n();
  const s = () => {
    const m = parseInt(Kt.value, 10) || 720, g = parseInt(Qt.value, 10) || 480;
    e.updateScreenSize(m, g), t.hasWorld && (t.updateCamera(m, g), e.updateCameraBuffer(t.cameraData)), e.recreateBindGroup(), e.resetAccumulation(), V = 0, r = 0;
  }, o = async (m, g = true) => {
    x = false, console.log(`Loading Scene: ${m}...`);
    let p, y;
    m === "viewer" && ge && (Ne === "obj" ? p = ge : Ne === "glb" && (y = new Uint8Array(ge))), t.loadScene(m, p, y), t.printStats(), await e.loadTexturesFromWorld(t), e.updateGeometryBuffer("vertex", t.vertices), e.updateGeometryBuffer("normal", t.normals), e.updateGeometryBuffer("uv", t.uvs), e.updateGeometryBuffer("index", t.indices), e.updateGeometryBuffer("attr", t.attributes), e.updateGeometryBuffer("tlas", t.tlas), e.updateGeometryBuffer("blas", t.blas), e.updateGeometryBuffer("instance", t.instances), s(), st(), g && (x = true, S && (S.textContent = "Stop Rendering"));
  }, f = async () => {
    if (Ge) return;
    x = false, Ge = true, z.textContent = "Initializing...", z.disabled = true, S && (S.textContent = "Resume Rendering");
    const m = parseInt(er.value, 10) || 30, g = parseInt(tr.value, 10) || 3, p = m * g, y = parseInt(ir.value, 10) || 64, Nt = parseInt(rr.value, 10) || 4;
    console.log(`Starting recording: ${p} frames @ ${m}fps (VP9)`), console.log(`Quality: ${y} SPP, Batch: ${Nt}`);
    const at = new Xi({ target: new ti(), video: { codec: "V_VP9", width: Z.width, height: Z.height, frameRate: m } }), _e = new VideoEncoder({ output: (G, ot) => at.addVideoChunk(G, ot), error: (G) => console.error("VideoEncoder Error:", G) });
    _e.configure({ codec: "vp09.00.10.08", width: Z.width, height: Z.height, bitrate: 12e6 });
    try {
      for (let N = 0; N < p; N++) {
        z.textContent = `Rec: ${N}/${p} (${Math.round(N / p * 100)}%)`, await new Promise((De) => setTimeout(De, 0));
        const Ti = N / m;
        t.update(Ti);
        let U = false;
        U || (U = e.updateGeometryBuffer("tlas", t.tlas)), U || (U = e.updateGeometryBuffer("blas", t.blas)), U || (U = e.updateGeometryBuffer("instance", t.instances)), U || (U = e.updateGeometryBuffer("vertex", t.vertices)), U || (U = e.updateGeometryBuffer("normal", t.normals)), U || (U = e.updateGeometryBuffer("index", t.indices)), U || (U = e.updateGeometryBuffer("attr", t.attributes)), U && e.recreateBindGroup(), e.resetAccumulation();
        let me = 0;
        for (; me < y; ) {
          const De = Math.min(Nt, y - me);
          for (let we = 0; we < De; we++) e.render(me + we);
          me += De, await e.device.queue.onSubmittedWorkDone(), me < y && await new Promise((we) => setTimeout(we, 0));
        }
        _e.encodeQueueSize > 5 && await _e.flush();
        const Ot = new VideoFrame(Z, { timestamp: N * 1e6 / m, duration: 1e6 / m });
        _e.encode(Ot, { keyFrame: N % m === 0 }), Ot.close();
      }
      z.textContent = "Finalizing...", await _e.flush(), at.finalize();
      const { buffer: G } = at.target, ot = new Blob([G], { type: "video/webm" }), Vt = URL.createObjectURL(ot), dt = document.createElement("a");
      dt.href = Vt, dt.download = `raytrace_${Date.now()}.webm`, dt.click(), URL.revokeObjectURL(Vt);
    } catch (G) {
      console.error("Recording failed:", G), alert("Recording failed. See console.");
    } finally {
      Ge = false, x = true, z.textContent = "\u25CF Rec", z.disabled = false, S && (S.textContent = "Stop Rendering"), requestAnimationFrame(D);
    }
  };
  let h = performance.now(), b = 0;
  const D = () => {
    if (Ge || (requestAnimationFrame(D), !x || !t.hasWorld)) return;
    let m = parseInt(Ji.value, 10);
    if ((isNaN(m) || m < 0) && (m = 0), m > 0 && V >= m) {
      t.update(r / m / 60);
      let p = false;
      p || (p = e.updateGeometryBuffer("tlas", t.tlas)), p || (p = e.updateGeometryBuffer("blas", t.blas)), p || (p = e.updateGeometryBuffer("instance", t.instances)), p || (p = e.updateGeometryBuffer("vertex", t.vertices)), p || (p = e.updateGeometryBuffer("normal", t.normals)), p || (p = e.updateGeometryBuffer("index", t.indices)), p || (p = e.updateGeometryBuffer("attr", t.attributes)), p && e.recreateBindGroup(), e.resetAccumulation(), V = 0;
    }
    V++, b++, r++, e.render(V);
    const g = performance.now();
    g - h >= 1e3 && (Gt.textContent = `FPS: ${b} | ${(1e3 / b).toFixed(2)}ms | Frame: ${V}`, b = 0, h = g);
  };
  S && S.addEventListener("click", () => {
    x = !x, S.textContent = x ? "Stop Rendering" : "Resume Rendering";
  }), z && z.addEventListener("click", f), Xt.addEventListener("change", (m) => o(m.target.value, false)), Kt.addEventListener("change", s), Qt.addEventListener("change", s), Zi.addEventListener("click", () => {
    x = false, n(), e.recreateBindGroup(), e.resetAccumulation(), V = 0, x = true;
  }), kt.addEventListener("change", async (m) => {
    var _a, _b;
    const g = (_a = m.target.files) == null ? void 0 : _a[0];
    if (!g) return;
    ((_b = g.name.split(".").pop()) == null ? void 0 : _b.toLowerCase()) === "obj" ? (ge = await g.text(), Ne = "obj") : (ge = await g.arrayBuffer(), Ne = "glb"), Xt.value = "viewer", o("viewer", false);
  });
  const st = () => {
    const m = t.getAnimationList();
    if (A.innerHTML = "", m.length === 0) {
      const g = document.createElement("option");
      g.text = "No Anim", A.add(g), A.disabled = true;
      return;
    }
    A.disabled = false, m.forEach((g, p) => {
      const y = document.createElement("option");
      y.text = `[${p}] ${g}`, y.value = p.toString(), A.add(y);
    }), A.value = "0";
  };
  A.addEventListener("change", () => {
    const m = parseInt(A.value, 10);
    t.setAnimation(m);
  }), s(), o("cornell", false), requestAnimationFrame(D);
}
nr().catch(console.error);
