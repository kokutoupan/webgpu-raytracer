var __defProp = Object.defineProperty;
var __defNormalProp = (obj, key, value) => key in obj ? __defProp(obj, key, { enumerable: true, configurable: true, writable: true, value }) : obj[key] = value;
var __publicField = (obj, key, value) => __defNormalProp(obj, typeof key !== "symbol" ? key + "" : key, value);
(async () => {
  (function() {
    const e = document.createElement("link").relList;
    if (e && e.supports && e.supports("modulepreload")) return;
    for (const i of document.querySelectorAll('link[rel="modulepreload"]')) r(i);
    new MutationObserver((i) => {
      for (const s of i) if (s.type === "childList") for (const a of s.addedNodes) a.tagName === "LINK" && a.rel === "modulepreload" && r(a);
    }).observe(document, {
      childList: true,
      subtree: true
    });
    function t(i) {
      const s = {};
      return i.integrity && (s.integrity = i.integrity), i.referrerPolicy && (s.referrerPolicy = i.referrerPolicy), i.crossOrigin === "use-credentials" ? s.credentials = "include" : i.crossOrigin === "anonymous" ? s.credentials = "omit" : s.credentials = "same-origin", s;
    }
    function r(i) {
      if (i.ep) return;
      i.ep = true;
      const s = t(i);
      fetch(i.href, s);
    }
  })();
  const ue = "modulepreload", fe = function(n) {
    return "/webgpu-raytracer/" + n;
  }, K = {}, ie = function(e, t, r) {
    let i = Promise.resolve();
    if (t && t.length > 0) {
      let d = function(u) {
        return Promise.all(u.map((p) => Promise.resolve(p).then((B) => ({
          status: "fulfilled",
          value: B
        }), (B) => ({
          status: "rejected",
          reason: B
        }))));
      };
      document.getElementsByTagName("link");
      const a = document.querySelector("meta[property=csp-nonce]"), l = (a == null ? void 0 : a.nonce) || (a == null ? void 0 : a.getAttribute("nonce"));
      i = d(t.map((u) => {
        if (u = fe(u), u in K) return;
        K[u] = true;
        const p = u.endsWith(".css"), B = p ? '[rel="stylesheet"]' : "";
        if (document.querySelector(`link[href="${u}"]${B}`)) return;
        const y = document.createElement("link");
        if (y.rel = p ? "stylesheet" : ue, p || (y.as = "script"), y.crossOrigin = "", y.href = u, l && y.setAttribute("nonce", l), document.head.appendChild(y), p) return new Promise((le, de) => {
          y.addEventListener("load", le), y.addEventListener("error", () => de(new Error(`Unable to preload CSS for ${u}`)));
        });
      }));
    }
    function s(a) {
      const l = new Event("vite:preloadError", {
        cancelable: true
      });
      if (l.payload = a, window.dispatchEvent(l), !l.defaultPrevented) throw a;
    }
    return i.then((a) => {
      for (const l of a || []) l.status === "rejected" && s(l.reason);
      return e().catch(s);
    });
  }, he = `// =========================================================
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
  class _e {
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
      __publicField(this, "uniformMixedData", new Uint32Array(4));
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
      }), this.sceneUniformBuffer = this.device.createBuffer({
        size: 128,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
      }), this.sampler = this.device.createSampler({
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
      this.defaultTexture = this.device.createTexture({
        size: [
          1,
          1,
          1
        ],
        format: "rgba8unorm",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
      }), this.device.queue.writeTexture({
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
    buildPipeline(e, t) {
      let r = he;
      r = r.replace(/const\s+MAX_DEPTH\s*=\s*\d+u;/, `const MAX_DEPTH = ${e}u;`), r = r.replace(/const\s+SPP\s*=\s*\d+u;/, `const SPP = ${t}u;`);
      const i = this.device.createShaderModule({
        label: "RayTracing",
        code: r
      });
      this.pipeline = this.device.createComputePipeline({
        label: "Main Pipeline",
        layout: "auto",
        compute: {
          module: i,
          entryPoint: "main"
        }
      }), this.bindGroupLayout = this.pipeline.getBindGroupLayout(0);
    }
    updateScreenSize(e, t) {
      this.canvas.width = e, this.canvas.height = t, this.renderTarget && this.renderTarget.destroy(), this.renderTarget = this.device.createTexture({
        size: [
          e,
          t
        ],
        format: "rgba8unorm",
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC
      }), this.renderTargetView = this.renderTarget.createView(), this.bufferSize = e * t * 16, this.accumulateBuffer && this.accumulateBuffer.destroy(), this.accumulateBuffer = this.device.createBuffer({
        size: this.bufferSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
      });
    }
    resetAccumulation() {
      this.accumulateBuffer && this.device.queue.writeBuffer(this.accumulateBuffer, 0, new Float32Array(this.bufferSize / 4));
    }
    async loadTexturesFromWorld(e) {
      const t = e.textureCount;
      if (t === 0) {
        this.createDefaultTexture();
        return;
      }
      console.log(`Loading ${t} textures...`);
      const r = [];
      for (let i = 0; i < t; i++) {
        const s = e.getTexture(i);
        if (s) try {
          const a = new Blob([
            s
          ]), l = await createImageBitmap(a, {
            resizeWidth: 1024,
            resizeHeight: 1024
          });
          r.push(l);
        } catch (a) {
          console.warn(`Failed tex ${i}`, a), r.push(await this.createFallbackBitmap());
        }
        else r.push(await this.createFallbackBitmap());
      }
      this.texture && this.texture.destroy(), this.texture = this.device.createTexture({
        size: [
          1024,
          1024,
          r.length
        ],
        format: "rgba8unorm",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT
      });
      for (let i = 0; i < r.length; i++) this.device.queue.copyExternalImageToTexture({
        source: r[i]
      }, {
        texture: this.texture,
        origin: [
          0,
          0,
          i
        ]
      }, [
        1024,
        1024
      ]);
    }
    async createFallbackBitmap() {
      const e = document.createElement("canvas");
      e.width = 1024, e.height = 1024;
      const t = e.getContext("2d");
      return t.fillStyle = "white", t.fillRect(0, 0, 1024, 1024), await createImageBitmap(e);
    }
    ensureBuffer(e, t, r) {
      if (e && e.size >= t) return e;
      e && e.destroy();
      let i = Math.ceil(t * 1.5);
      return i = i + 3 & -4, i = Math.max(i, 4), this.device.createBuffer({
        label: r,
        size: i,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
      });
    }
    updateBuffer(e, t) {
      const r = t.byteLength;
      let i = false, s;
      return e === "index" ? ((!this.indexBuffer || this.indexBuffer.size < r) && (i = true), this.indexBuffer = this.ensureBuffer(this.indexBuffer, r, "IndexBuffer"), s = this.indexBuffer) : e === "attr" ? ((!this.attrBuffer || this.attrBuffer.size < r) && (i = true), this.attrBuffer = this.ensureBuffer(this.attrBuffer, r, "AttrBuffer"), s = this.attrBuffer) : ((!this.instanceBuffer || this.instanceBuffer.size < r) && (i = true), this.instanceBuffer = this.ensureBuffer(this.instanceBuffer, r, "InstanceBuffer"), s = this.instanceBuffer), this.device.queue.writeBuffer(s, 0, t, 0, t.length), i;
    }
    updateCombinedGeometry(e, t, r) {
      const i = e.byteLength + t.byteLength + r.byteLength;
      let s = false;
      (!this.geometryBuffer || this.geometryBuffer.size < i) && (s = true);
      const a = e.length / 4;
      this.vertexCount = a, this.geometryBuffer = this.ensureBuffer(this.geometryBuffer, i, "GeometryBuffer"), !(r.length >= a * 2) && a > 0 && console.warn(`UV buffer mismatch: V=${a}, UV=${r.length / 2}. Filling 0.`);
      let d = 0;
      return this.device.queue.writeBuffer(this.geometryBuffer, d, e), d += e.byteLength, this.device.queue.writeBuffer(this.geometryBuffer, d, t), d += t.byteLength, this.device.queue.writeBuffer(this.geometryBuffer, d, r), s;
    }
    updateCombinedBVH(e, t) {
      const r = e.byteLength, i = t.byteLength, s = r + i;
      let a = false;
      return (!this.nodesBuffer || this.nodesBuffer.size < s) && (a = true), this.nodesBuffer = this.ensureBuffer(this.nodesBuffer, s, "NodesBuffer"), this.device.queue.writeBuffer(this.nodesBuffer, 0, e), this.device.queue.writeBuffer(this.nodesBuffer, r, t), this.blasOffset = e.length / 8, a;
    }
    updateSceneUniforms(e, t) {
      this.sceneUniformBuffer && (this.device.queue.writeBuffer(this.sceneUniformBuffer, 0, e), this.uniformMixedData[0] = t, this.uniformMixedData[1] = this.blasOffset, this.uniformMixedData[2] = this.vertexCount, this.uniformMixedData[3] = 0, this.device.queue.writeBuffer(this.sceneUniformBuffer, 96, this.uniformMixedData));
    }
    recreateBindGroup() {
      !this.renderTargetView || !this.accumulateBuffer || !this.geometryBuffer || !this.nodesBuffer || !this.sceneUniformBuffer || (this.bindGroup = this.device.createBindGroup({
        layout: this.bindGroupLayout,
        entries: [
          {
            binding: 0,
            resource: this.renderTargetView
          },
          {
            binding: 1,
            resource: {
              buffer: this.accumulateBuffer
            }
          },
          {
            binding: 2,
            resource: {
              buffer: this.sceneUniformBuffer
            }
          },
          {
            binding: 3,
            resource: {
              buffer: this.geometryBuffer
            }
          },
          {
            binding: 4,
            resource: {
              buffer: this.indexBuffer
            }
          },
          {
            binding: 5,
            resource: {
              buffer: this.attrBuffer
            }
          },
          {
            binding: 6,
            resource: {
              buffer: this.nodesBuffer
            }
          },
          {
            binding: 7,
            resource: {
              buffer: this.instanceBuffer
            }
          },
          {
            binding: 8,
            resource: this.texture.createView({
              dimension: "2d-array"
            })
          },
          {
            binding: 9,
            resource: this.sampler
          }
        ]
      }));
    }
    render(e) {
      if (!this.bindGroup) return;
      this.uniformMixedData[0] = e, this.device.queue.writeBuffer(this.sceneUniformBuffer, 96, this.uniformMixedData, 0, 1);
      const t = Math.ceil(this.canvas.width / 8), r = Math.ceil(this.canvas.height / 8), i = this.device.createCommandEncoder(), s = i.beginComputePass();
      s.setPipeline(this.pipeline), s.setBindGroup(0, this.bindGroup), s.dispatchWorkgroups(t, r), s.end(), i.copyTextureToTexture({
        texture: this.renderTarget
      }, {
        texture: this.context.getCurrentTexture()
      }, {
        width: this.canvas.width,
        height: this.canvas.height,
        depthOrArrayLayers: 1
      }), this.device.queue.submit([
        i.finish()
      ]);
    }
  }
  let c;
  function pe(n) {
    const e = c.__externref_table_alloc();
    return c.__wbindgen_externrefs.set(e, n), e;
  }
  function ge(n, e) {
    return n = n >>> 0, E().subarray(n / 1, n / 1 + e);
  }
  let k = null;
  function X() {
    return (k === null || k.buffer.detached === true || k.buffer.detached === void 0 && k.buffer !== c.memory.buffer) && (k = new DataView(c.memory.buffer)), k;
  }
  function U(n, e) {
    return n = n >>> 0, we(n, e);
  }
  let A = null;
  function E() {
    return (A === null || A.byteLength === 0) && (A = new Uint8Array(c.memory.buffer)), A;
  }
  function me(n, e) {
    try {
      return n.apply(this, e);
    } catch (t) {
      const r = pe(t);
      c.__wbindgen_exn_store(r);
    }
  }
  function Q(n) {
    return n == null;
  }
  function Z(n, e) {
    const t = e(n.length * 1, 1) >>> 0;
    return E().set(n, t / 1), R = n.length, t;
  }
  function G(n, e, t) {
    if (t === void 0) {
      const l = D.encode(n), d = e(l.length, 1) >>> 0;
      return E().subarray(d, d + l.length).set(l), R = l.length, d;
    }
    let r = n.length, i = e(r, 1) >>> 0;
    const s = E();
    let a = 0;
    for (; a < r; a++) {
      const l = n.charCodeAt(a);
      if (l > 127) break;
      s[i + a] = l;
    }
    if (a !== r) {
      a !== 0 && (n = n.slice(a)), i = t(i, r, r = a + n.length * 3, 1) >>> 0;
      const l = E().subarray(i + a, i + r), d = D.encodeInto(n, l);
      a += d.written, i = t(i, r, a, 1) >>> 0;
    }
    return R = a, i;
  }
  let L = new TextDecoder("utf-8", {
    ignoreBOM: true,
    fatal: true
  });
  L.decode();
  const be = 2146435072;
  let N = 0;
  function we(n, e) {
    return N += e, N >= be && (L = new TextDecoder("utf-8", {
      ignoreBOM: true,
      fatal: true
    }), L.decode(), N = e), L.decode(E().subarray(n, n + e));
  }
  const D = new TextEncoder();
  "encodeInto" in D || (D.encodeInto = function(n, e) {
    const t = D.encode(n);
    return e.set(t), {
      read: n.length,
      written: t.length
    };
  });
  let R = 0;
  typeof FinalizationRegistry > "u" || new FinalizationRegistry((n) => c.__wbg_renderbuffers_free(n >>> 0, 1));
  const ee = typeof FinalizationRegistry > "u" ? {
    register: () => {
    },
    unregister: () => {
    }
  } : new FinalizationRegistry((n) => c.__wbg_world_free(n >>> 0, 1));
  class V {
    __destroy_into_raw() {
      const e = this.__wbg_ptr;
      return this.__wbg_ptr = 0, ee.unregister(this), e;
    }
    free() {
      const e = this.__destroy_into_raw();
      c.__wbg_world_free(e, 0);
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
    set_animation(e) {
      c.world_set_animation(this.__wbg_ptr, e);
    }
    update_camera(e, t) {
      c.world_update_camera(this.__wbg_ptr, e, t);
    }
    attributes_len() {
      return c.world_attributes_len(this.__wbg_ptr) >>> 0;
    }
    attributes_ptr() {
      return c.world_attributes_ptr(this.__wbg_ptr) >>> 0;
    }
    get_texture_ptr(e) {
      return c.world_get_texture_ptr(this.__wbg_ptr, e) >>> 0;
    }
    get_texture_size(e) {
      return c.world_get_texture_size(this.__wbg_ptr, e) >>> 0;
    }
    get_texture_count() {
      return c.world_get_texture_count(this.__wbg_ptr) >>> 0;
    }
    get_animation_name(e) {
      let t, r;
      try {
        const i = c.world_get_animation_name(this.__wbg_ptr, e);
        return t = i[0], r = i[1], U(i[0], i[1]);
      } finally {
        c.__wbindgen_free(t, r, 1);
      }
    }
    load_animation_glb(e) {
      const t = Z(e, c.__wbindgen_malloc), r = R;
      c.world_load_animation_glb(this.__wbg_ptr, t, r);
    }
    get_animation_count() {
      return c.world_get_animation_count(this.__wbg_ptr) >>> 0;
    }
    constructor(e, t, r) {
      const i = G(e, c.__wbindgen_malloc, c.__wbindgen_realloc), s = R;
      var a = Q(t) ? 0 : G(t, c.__wbindgen_malloc, c.__wbindgen_realloc), l = R, d = Q(r) ? 0 : Z(r, c.__wbindgen_malloc), u = R;
      const p = c.world_new(i, s, a, l, d, u);
      return this.__wbg_ptr = p >>> 0, ee.register(this, this.__wbg_ptr, this), this;
    }
    update(e) {
      c.world_update(this.__wbg_ptr, e);
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
  Symbol.dispose && (V.prototype[Symbol.dispose] = V.prototype.free);
  const ve = /* @__PURE__ */ new Set([
    "basic",
    "cors",
    "default"
  ]);
  async function ye(n, e) {
    if (typeof Response == "function" && n instanceof Response) {
      if (typeof WebAssembly.instantiateStreaming == "function") try {
        return await WebAssembly.instantiateStreaming(n, e);
      } catch (r) {
        if (n.ok && ve.has(n.type) && n.headers.get("Content-Type") !== "application/wasm") console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", r);
        else throw r;
      }
      const t = await n.arrayBuffer();
      return await WebAssembly.instantiate(t, e);
    } else {
      const t = await WebAssembly.instantiate(n, e);
      return t instanceof WebAssembly.Instance ? {
        instance: t,
        module: n
      } : t;
    }
  }
  function Re() {
    const n = {};
    return n.wbg = {}, n.wbg.__wbg___wbindgen_throw_dd24417ed36fc46e = function(e, t) {
      throw new Error(U(e, t));
    }, n.wbg.__wbg_error_7534b8e9a36f1ab4 = function(e, t) {
      let r, i;
      try {
        r = e, i = t, console.error(U(e, t));
      } finally {
        c.__wbindgen_free(r, i, 1);
      }
    }, n.wbg.__wbg_getRandomValues_1c61fac11405ffdc = function() {
      return me(function(e, t) {
        globalThis.crypto.getRandomValues(ge(e, t));
      }, arguments);
    }, n.wbg.__wbg_log_1d990106d99dacb7 = function(e) {
      console.log(e);
    }, n.wbg.__wbg_new_8a6f238a6ece86ea = function() {
      return new Error();
    }, n.wbg.__wbg_stack_0ed75d68575b0f3c = function(e, t) {
      const r = t.stack, i = G(r, c.__wbindgen_malloc, c.__wbindgen_realloc), s = R;
      X().setInt32(e + 4, s, true), X().setInt32(e + 0, i, true);
    }, n.wbg.__wbindgen_cast_2241b6af4c4b2941 = function(e, t) {
      return U(e, t);
    }, n.wbg.__wbindgen_init_externref_table = function() {
      const e = c.__wbindgen_externrefs, t = e.grow(4);
      e.set(0, void 0), e.set(t + 0, void 0), e.set(t + 1, null), e.set(t + 2, true), e.set(t + 3, false);
    }, n;
  }
  function xe(n, e) {
    return c = n.exports, se.__wbindgen_wasm_module = e, k = null, A = null, c.__wbindgen_start(), c;
  }
  async function se(n) {
    if (c !== void 0) return c;
    typeof n < "u" && (Object.getPrototypeOf(n) === Object.prototype ? { module_or_path: n } = n : console.warn("using deprecated parameters for the initialization function; pass a single object instead")), typeof n > "u" && (n = new URL("/webgpu-raytracer/assets/rust_shader_tools_bg-CC5HVMsp.wasm", import.meta.url));
    const e = Re();
    (typeof n == "string" || typeof Request == "function" && n instanceof Request || typeof URL == "function" && n instanceof URL) && (n = fetch(n));
    const { instance: t, module: r } = await ye(await n, e);
    return xe(t, r);
  }
  class Se {
    constructor() {
      __publicField(this, "world", null);
      __publicField(this, "wasmMemory", null);
    }
    async initWasm() {
      const e = await se();
      this.wasmMemory = e.memory, console.log("Wasm initialized");
    }
    loadScene(e, t, r) {
      this.world && this.world.free(), this.world = new V(e, t, r);
    }
    update(e) {
      var _a;
      (_a = this.world) == null ? void 0 : _a.update(e);
    }
    updateCamera(e, t) {
      var _a;
      (_a = this.world) == null ? void 0 : _a.update_camera(e, t);
    }
    loadAnimation(e) {
      var _a;
      (_a = this.world) == null ? void 0 : _a.load_animation_glb(e);
    }
    getAnimationList() {
      if (!this.world) return [];
      const e = this.world.get_animation_count(), t = [];
      for (let r = 0; r < e; r++) t.push(this.world.get_animation_name(r));
      return t;
    }
    setAnimation(e) {
      var _a;
      (_a = this.world) == null ? void 0 : _a.set_animation(e);
    }
    getF32(e, t) {
      return new Float32Array(this.wasmMemory.buffer, e, t);
    }
    getU32(e, t) {
      return new Uint32Array(this.wasmMemory.buffer, e, t);
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
      const t = this.world.get_texture_ptr(e), r = this.world.get_texture_size(e);
      return !t || r === 0 ? null : new Uint8Array(this.wasmMemory.buffer, t, r).slice();
    }
    get hasWorld() {
      return !!this.world;
    }
    printStats() {
      this.world && console.log(`Scene Stats: V=${this.vertices.length / 4}, Tri=${this.indices.length / 3}, BLAS=${this.blas.length / 8}, TLAS=${this.tlas.length / 8}`);
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
      btnHost: "btn-host",
      btnWorker: "btn-worker",
      statusDiv: "status"
    }
  };
  class Be {
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
      this.canvas = this.el(h.ids.canvas), this.btnRender = this.el(h.ids.renderBtn), this.sceneSelect = this.el(h.ids.sceneSelect), this.inputWidth = this.el(h.ids.resWidth), this.inputHeight = this.el(h.ids.resHeight), this.inputFile = this.setupFileInput(), this.inputDepth = this.el(h.ids.maxDepth), this.inputSPP = this.el(h.ids.sppFrame), this.btnRecompile = this.el(h.ids.recompileBtn), this.inputUpdateInterval = this.el(h.ids.updateInterval), this.animSelect = this.el(h.ids.animSelect), this.btnRecord = this.el(h.ids.recordBtn), this.inputRecFps = this.el(h.ids.recFps), this.inputRecDur = this.el(h.ids.recDuration), this.inputRecSpp = this.el(h.ids.recSpp), this.inputRecBatch = this.el(h.ids.recBatch), this.btnHost = this.el(h.ids.btnHost), this.btnWorker = this.el(h.ids.btnWorker), this.statusDiv = this.el(h.ids.statusDiv), this.statsDiv = this.createStatsDiv(), this.bindEvents();
    }
    el(e) {
      const t = document.getElementById(e);
      if (!t) throw new Error(`Element not found: ${e}`);
      return t;
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
      }), this.inputFile.addEventListener("change", (t) => {
        var _a, _b;
        const r = (_a = t.target.files) == null ? void 0 : _a[0];
        r && ((_b = this.onFileSelect) == null ? void 0 : _b.call(this, r));
      }), this.animSelect.addEventListener("change", () => {
        var _a;
        const t = parseInt(this.animSelect.value, 10);
        (_a = this.onAnimSelect) == null ? void 0 : _a.call(this, t);
      }), this.btnRecord.addEventListener("click", () => {
        var _a;
        return (_a = this.onRecordStart) == null ? void 0 : _a.call(this);
      }), this.btnHost.addEventListener("click", () => {
        var _a;
        return (_a = this.onConnectHost) == null ? void 0 : _a.call(this);
      }), this.btnWorker.addEventListener("click", () => {
        var _a;
        return (_a = this.onConnectWorker) == null ? void 0 : _a.call(this);
      });
    }
    updateRenderButton(e) {
      this.btnRender.textContent = e ? "Stop Rendering" : "Resume Rendering";
    }
    updateStats(e, t, r) {
      this.statsDiv.textContent = `FPS: ${e} | ${t.toFixed(2)}ms | Frame: ${r}`;
    }
    setStatus(e) {
      this.statusDiv.textContent = e;
    }
    setConnectionState(e) {
      e === "host" ? (this.btnHost.textContent = "Disconnect", this.btnHost.disabled = false, this.btnWorker.textContent = "Worker", this.btnWorker.disabled = true) : e === "worker" ? (this.btnHost.textContent = "Host", this.btnHost.disabled = true, this.btnWorker.textContent = "Disconnect", this.btnWorker.disabled = false) : (this.btnHost.textContent = "Host", this.btnHost.disabled = false, this.btnWorker.textContent = "Worker", this.btnWorker.disabled = false, this.statusDiv.textContent = "Offline");
    }
    setRecordingState(e, t) {
      e ? (this.btnRecord.disabled = true, this.btnRecord.textContent = t || "Recording...", this.btnRender.textContent = "Resume Rendering") : (this.btnRecord.disabled = false, this.btnRecord.textContent = "\u25CF Rec");
    }
    updateAnimList(e) {
      if (this.animSelect.innerHTML = "", e.length === 0) {
        const t = document.createElement("option");
        t.text = "No Anim", this.animSelect.add(t), this.animSelect.disabled = true;
        return;
      }
      this.animSelect.disabled = false, e.forEach((t, r) => {
        const i = document.createElement("option");
        i.text = `[${r}] ${t}`, i.value = r.toString(), this.animSelect.add(i);
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
        anim: parseInt(this.animSelect.value, 10) || 0
      };
    }
    setRenderConfig(e) {
      this.inputWidth.value = e.width.toString(), this.inputHeight.value = e.height.toString(), this.inputRecFps.value = e.fps.toString(), this.inputRecDur.value = e.duration.toString(), this.inputRecSpp.value = e.spp.toString(), this.inputRecBatch.value = e.batch.toString();
    }
  }
  class ke {
    constructor(e, t, r) {
      __publicField(this, "isRecording", false);
      __publicField(this, "renderer");
      __publicField(this, "worldBridge");
      __publicField(this, "canvas");
      this.renderer = e, this.worldBridge = t, this.canvas = r;
    }
    get recording() {
      return this.isRecording;
    }
    async record(e, t, r) {
      if (this.isRecording) return;
      this.isRecording = true;
      const { Muxer: i, ArrayBufferTarget: s } = await ie(async () => {
        const { Muxer: u, ArrayBufferTarget: p } = await import("./webm-muxer-MLtUgOCn.js");
        return {
          Muxer: u,
          ArrayBufferTarget: p
        };
      }, []), a = Math.ceil(e.fps * e.duration);
      console.log(`Starting recording: ${a} frames @ ${e.fps}fps (VP9)`);
      const l = new i({
        target: new s(),
        video: {
          codec: "V_VP9",
          width: this.canvas.width,
          height: this.canvas.height,
          frameRate: e.fps
        }
      }), d = new VideoEncoder({
        output: (u, p) => l.addVideoChunk(u, p),
        error: (u) => console.error("VideoEncoder Error:", u)
      });
      d.configure({
        codec: "vp09.00.10.08",
        width: this.canvas.width,
        height: this.canvas.height,
        bitrate: 12e6
      });
      try {
        await this.renderAndEncode(a, e, d, t, e.startFrame || 0), await d.flush(), l.finalize();
        const { buffer: u } = l.target, p = new Blob([
          u
        ], {
          type: "video/webm"
        }), B = URL.createObjectURL(p);
        r(B, p);
      } catch (u) {
        throw console.error("Recording failed:", u), u;
      } finally {
        this.isRecording = false;
      }
    }
    async recordChunks(e, t) {
      if (this.isRecording) throw new Error("Already recording");
      this.isRecording = true;
      const r = [], i = Math.ceil(e.fps * e.duration), s = new VideoEncoder({
        output: (a, l) => {
          const d = new Uint8Array(a.byteLength);
          a.copyTo(d), r.push({
            type: a.type,
            timestamp: a.timestamp,
            duration: a.duration,
            data: d.buffer,
            decoderConfig: l == null ? void 0 : l.decoderConfig
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
        return await this.renderAndEncode(i, e, s, t, e.startFrame || 0), await s.flush(), r;
      } finally {
        this.isRecording = false;
      }
    }
    async renderAndEncode(e, t, r, i, s = 0) {
      for (let a = 0; a < e; a++) {
        i(a, e), await new Promise((p) => setTimeout(p, 0));
        const l = s + a, d = l / t.fps;
        this.worldBridge.update(d), await this.updateSceneBuffers(), await this.renderFrame(t.spp, t.batch), r.encodeQueueSize > 5 && await r.flush();
        const u = new VideoFrame(this.canvas, {
          timestamp: l * 1e6 / t.fps,
          duration: 1e6 / t.fps
        });
        r.encode(u, {
          keyFrame: a % t.fps === 0
        }), u.close();
      }
    }
    async updateSceneBuffers() {
      let e = false;
      e || (e = this.renderer.updateCombinedBVH(this.worldBridge.tlas, this.worldBridge.blas)), e || (e = this.renderer.updateBuffer("instance", this.worldBridge.instances)), e || (e = this.renderer.updateCombinedGeometry(this.worldBridge.vertices, this.worldBridge.normals, this.worldBridge.uvs)), e || (e = this.renderer.updateBuffer("index", this.worldBridge.indices)), e || (e = this.renderer.updateBuffer("attr", this.worldBridge.attributes)), this.worldBridge.updateCamera(this.canvas.width, this.canvas.height), this.renderer.updateSceneUniforms(this.worldBridge.cameraData, 0), e && this.renderer.recreateBindGroup(), this.renderer.resetAccumulation();
    }
    async renderFrame(e, t) {
      let r = 0;
      for (; r < e; ) {
        const i = Math.min(t, e - r);
        for (let s = 0; s < i; s++) this.renderer.render(r + s);
        r += i, await this.renderer.device.queue.onSubmittedWorkDone(), r < e && await new Promise((s) => setTimeout(s, 0));
      }
    }
  }
  const Ce = {
    iceServers: [
      {
        urls: "stun:stun.l.google.com:19302"
      }
    ]
  };
  class te {
    constructor(e, t) {
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
      this.remoteId = e, this.sendSignal = t, this.pc = new RTCPeerConnection(Ce), this.pc.onicecandidate = (r) => {
        r.candidate && this.sendSignal({
          type: "candidate",
          candidate: r.candidate.toJSON(),
          targetId: this.remoteId
        });
      };
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
      this.pc.ondatachannel = (r) => {
        this.dc = r.channel, this.setupDataChannel();
      }, await this.pc.setRemoteDescription(new RTCSessionDescription(e));
      const t = await this.pc.createAnswer();
      await this.pc.setLocalDescription(t), this.sendSignal({
        type: "answer",
        sdp: t,
        targetId: this.remoteId
      });
    }
    async handleAnswer(e) {
      await this.pc.setRemoteDescription(new RTCSessionDescription(e));
    }
    async handleCandidate(e) {
      await this.pc.addIceCandidate(new RTCIceCandidate(e));
    }
    async sendScene(e, t, r) {
      if (!this.dc || this.dc.readyState !== "open") return;
      let i;
      typeof e == "string" ? i = new TextEncoder().encode(e) : i = new Uint8Array(e);
      const s = {
        type: "SCENE_INIT",
        totalBytes: i.byteLength,
        config: {
          ...r,
          fileType: t
        }
      };
      this.sendData(s), await this.sendBinaryChunks(i);
    }
    async sendRenderResult(e, t) {
      if (!this.dc || this.dc.readyState !== "open") return;
      let r = 0;
      const i = e.map((l) => {
        const d = l.data.byteLength;
        return r += d, {
          type: l.type,
          timestamp: l.timestamp,
          duration: l.duration,
          size: d,
          decoderConfig: l.decoderConfig
        };
      });
      console.log(`[RTC] Sending Render Result: ${r} bytes, ${e.length} chunks`), this.sendData({
        type: "RENDER_RESULT",
        startFrame: t,
        totalBytes: r,
        chunksMeta: i
      });
      const s = new Uint8Array(r);
      let a = 0;
      for (const l of e) s.set(new Uint8Array(l.data), a), a += l.data.byteLength;
      await this.sendBinaryChunks(s);
    }
    async sendBinaryChunks(e) {
      let r = 0;
      const i = () => new Promise((s) => {
        const a = setInterval(() => {
          (!this.dc || this.dc.bufferedAmount < 65536) && (clearInterval(a), s());
        }, 5);
      });
      for (; r < e.byteLength; ) {
        this.dc && this.dc.bufferedAmount > 256 * 1024 && await i();
        const s = Math.min(r + 16384, e.byteLength);
        if (this.dc) try {
          this.dc.send(e.subarray(r, s));
        } catch {
        }
        r = s, r % (16384 * 5) === 0 && await new Promise((a) => setTimeout(a, 0));
      }
      console.log("[RTC] Transfer Complete");
    }
    setupDataChannel() {
      this.dc && (this.dc.binaryType = "arraybuffer", this.dc.onopen = () => {
        console.log("[RTC] DataChannel Open"), this.onDataChannelOpen && this.onDataChannelOpen();
      }, this.dc.onmessage = (e) => {
        const t = e.data;
        if (typeof t == "string") try {
          const r = JSON.parse(t);
          this.handleControlMessage(r);
        } catch {
        }
        else t instanceof ArrayBuffer && this.handleBinaryChunk(t);
      });
    }
    handleControlMessage(e) {
      var _a, _b;
      e.type === "SCENE_INIT" ? (console.log(`[RTC] Receiving Scene: ${e.config.fileType}, ${e.totalBytes} bytes`), this.sceneMeta = {
        config: e.config,
        totalBytes: e.totalBytes
      }, this.receiveBuffer = new Uint8Array(e.totalBytes), this.receivedBytes = 0) : e.type === "SCENE_ACK" ? (console.log(`[RTC] Scene ACK: ${e.receivedBytes} bytes`), this.onAckReceived && this.onAckReceived(e.receivedBytes)) : e.type === "RENDER_REQUEST" ? (console.log(`[RTC] Render Request: Frame ${e.startFrame}, Count ${e.frameCount}`), (_a = this.onRenderRequest) == null ? void 0 : _a.call(this, e.startFrame, e.frameCount, e.config)) : e.type === "RENDER_RESULT" ? (console.log(`[RTC] Receiving Render Result: ${e.totalBytes} bytes`), this.resultMeta = {
        startFrame: e.startFrame,
        totalBytes: e.totalBytes,
        chunksMeta: e.chunksMeta
      }, this.receiveBuffer = new Uint8Array(e.totalBytes), this.receivedBytes = 0) : e.type === "WORKER_READY" && (console.log("[RTC] Worker Ready Signal Received"), (_b = this.onWorkerReady) == null ? void 0 : _b.call(this));
    }
    handleBinaryChunk(e) {
      var _a, _b;
      try {
        const t = new Uint8Array(e);
        if (this.receivedBytes + t.byteLength > this.receiveBuffer.byteLength) {
          console.error("[RTC] Receive Buffer Overflow!");
          return;
        }
        this.receiveBuffer.set(t, this.receivedBytes), this.receivedBytes += t.byteLength;
      } catch (t) {
        console.error("[RTC] Error handling binary chunk", t);
        return;
      }
      if (this.sceneMeta) {
        if (this.receivedBytes >= this.sceneMeta.totalBytes) {
          console.log("[RTC] Scene Download Complete!");
          let t;
          this.sceneMeta.config.fileType === "obj" ? t = new TextDecoder().decode(this.receiveBuffer) : t = this.receiveBuffer.buffer, (_a = this.onSceneReceived) == null ? void 0 : _a.call(this, t, this.sceneMeta.config), this.sceneMeta = null;
        }
      } else if (this.resultMeta && this.receivedBytes >= this.resultMeta.totalBytes) {
        console.log("[RTC] Render Result Complete!");
        const t = [];
        let r = 0;
        for (const i of this.resultMeta.chunksMeta) {
          const s = this.receiveBuffer.slice(r, r + i.size);
          t.push({
            type: i.type,
            timestamp: i.timestamp,
            duration: i.duration,
            data: s.buffer,
            decoderConfig: i.decoderConfig
          }), r += i.size;
        }
        (_b = this.onRenderResult) == null ? void 0 : _b.call(this, t, this.resultMeta.startFrame), this.resultMeta = null;
      }
    }
    sendData(e) {
      var _a;
      ((_a = this.dc) == null ? void 0 : _a.readyState) === "open" && this.dc.send(JSON.stringify(e));
    }
    sendAck(e) {
      this.sendData({
        type: "SCENE_ACK",
        receivedBytes: e
      });
    }
    sendRenderRequest(e, t, r) {
      const i = {
        type: "RENDER_REQUEST",
        startFrame: e,
        frameCount: t,
        config: r
      };
      this.sendData(i);
    }
    sendWorkerReady() {
      this.sendData({
        type: "WORKER_READY"
      });
    }
    close() {
      this.dc && (this.dc.close(), this.dc = null), this.pc && this.pc.close(), console.log(`[RTC] Connection closed: ${this.remoteId}`);
    }
  }
  class Te {
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
      const t = "xWUaLfXQQkHZ9VmF";
      this.ws = new WebSocket(`${h.signalingServerUrl}?token=${t}`), this.ws.onopen = () => {
        var _a2;
        console.log("WS Connected"), (_a2 = this.onStatusChange) == null ? void 0 : _a2.call(this, `Waiting for Peer (${e.toUpperCase()})`), this.sendSignal({
          type: e === "host" ? "register_host" : "register_worker"
        });
      }, this.ws.onmessage = (r) => {
        const i = JSON.parse(r.data);
        this.handleMessage(i);
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
    async sendRenderResult(e, t) {
      this.hostClient && await this.hostClient.sendRenderResult(e, t);
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
          const t = new te(e.workerId, (r) => this.sendSignal(r));
          this.workers.set(e.workerId, t), t.onDataChannelOpen = () => {
            var _a2;
            console.log(`[Host] Open for ${e.workerId}`), t.sendData({
              type: "HELLO",
              msg: "Hello from Host!"
            }), (_a2 = this.onWorkerJoined) == null ? void 0 : _a2.call(this, e.workerId);
          }, t.onAckReceived = (r) => {
            console.log(`Worker ${e.workerId} ACK: ${r}`);
          }, t.onRenderResult = (r, i) => {
            var _a2;
            console.log(`Received Render Result from ${e.workerId}: ${r.length} chunks`), (_a2 = this.onRenderResult) == null ? void 0 : _a2.call(this, r, i, e.workerId);
          }, t.onWorkerReady = () => {
            var _a2;
            (_a2 = this.onWorkerReady) == null ? void 0 : _a2.call(this, e.workerId);
          }, await t.startAsHost();
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
          e.fromId && (this.hostClient = new te(e.fromId, (t) => this.sendSignal(t)), await this.hostClient.handleOffer(e.sdp), (_a = this.onStatusChange) == null ? void 0 : _a.call(this, "Connected to Host!"), (_b = this.onHostConnected) == null ? void 0 : _b.call(this), this.hostClient.onDataChannelOpen = () => {
            var _a2, _b2;
            (_a2 = this.hostClient) == null ? void 0 : _a2.sendData({
              type: "HELLO",
              msg: "Hello from Worker!"
            }), (_b2 = this.onHostHello) == null ? void 0 : _b2.call(this);
          }, this.hostClient.onSceneReceived = (t, r) => {
            var _a2, _b2;
            (_a2 = this.onSceneReceived) == null ? void 0 : _a2.call(this, t, r);
            const i = typeof t == "string" ? t.length : t.byteLength;
            (_b2 = this.hostClient) == null ? void 0 : _b2.sendAck(i);
          }, this.hostClient.onRenderRequest = (t, r, i) => {
            var _a2;
            (_a2 = this.onRenderRequest) == null ? void 0 : _a2.call(this, t, r, i);
          });
          break;
        case "candidate":
          await ((_c = this.hostClient) == null ? void 0 : _c.handleCandidate(e.candidate));
          break;
      }
    }
    async broadcastScene(e, t, r) {
      const i = Array.from(this.workers.values()).map((s) => s.sendScene(e, t, r));
      await Promise.all(i);
    }
    async sendSceneToWorker(e, t, r, i) {
      const s = this.workers.get(e);
      s && await s.sendScene(t, r, i);
    }
    async sendRenderRequest(e, t, r, i) {
      const s = this.workers.get(e);
      s && await s.sendRenderRequest(t, r, i);
    }
  }
  let m = false, b = null, x = null, v = null, S = [], $ = /* @__PURE__ */ new Map(), P = 0, F = 0, q = 0, T = null, w = /* @__PURE__ */ new Map(), W = /* @__PURE__ */ new Map(), j = false, M = null;
  const ne = 20, o = new Be(), _ = new _e(o.canvas), f = new Se(), H = new ke(_, f, o.canvas), g = new Te();
  let C = 0, J = 0, I = 0, re = performance.now();
  const Ee = () => {
    const n = parseInt(o.inputDepth.value, 10) || h.defaultDepth, e = parseInt(o.inputSPP.value, 10) || h.defaultSPP;
    _.buildPipeline(n, e);
  }, Y = () => {
    const { width: n, height: e } = o.getRenderConfig();
    _.updateScreenSize(n, e), f.hasWorld && (f.updateCamera(n, e), _.updateSceneUniforms(f.cameraData, 0)), _.recreateBindGroup(), _.resetAccumulation(), C = 0, J = 0;
  }, O = async (n, e = true) => {
    m = false, console.log(`Loading Scene: ${n}...`);
    let t, r;
    n === "viewer" && b && (x === "obj" ? t = b : x === "glb" && (r = new Uint8Array(b))), f.loadScene(n, t, r), f.printStats(), await _.loadTexturesFromWorld(f), await Ae(), Y(), o.updateAnimList(f.getAnimationList()), e && (m = true, o.updateRenderButton(true));
  }, Ae = async () => {
    _.updateCombinedGeometry(f.vertices, f.normals, f.uvs), _.updateCombinedBVH(f.tlas, f.blas), _.updateBuffer("index", f.indices), _.updateBuffer("attr", f.attributes), _.updateBuffer("instance", f.instances);
  }, z = () => {
    if (H.recording || (requestAnimationFrame(z), !m || !f.hasWorld)) return;
    let n = parseInt(o.inputUpdateInterval.value, 10) || 0;
    if (n < 0 && (n = 0), n > 0 && C >= n) {
      f.update(J / n / 60);
      let t = false;
      t || (t = _.updateCombinedBVH(f.tlas, f.blas)), t || (t = _.updateBuffer("instance", f.instances)), t || (t = _.updateCombinedGeometry(f.vertices, f.normals, f.uvs)), t || (t = _.updateBuffer("index", f.indices)), t || (t = _.updateBuffer("attr", f.attributes)), f.updateCamera(o.canvas.width, o.canvas.height), _.updateSceneUniforms(f.cameraData, 0), t && _.recreateBindGroup(), _.resetAccumulation(), C = 0;
    }
    C++, I++, J++, _.render(C);
    const e = performance.now();
    e - re >= 1e3 && (o.updateStats(I, 1e3 / I, C), I = 0, re = e);
  }, ae = async (n) => {
    if (!b || !x) return;
    const e = o.getRenderConfig();
    n ? (console.log(`Sending scene to specific worker: ${n}`), w.set(n, "loading"), await g.sendSceneToWorker(n, b, x, e)) : (console.log("Broadcasting scene to all workers..."), g.getWorkerIds().forEach((t) => w.set(t, "loading")), await g.broadcastScene(b, x, e));
  }, oe = async (n) => {
    if (w.get(n) !== "idle") {
      console.log(`Worker ${n} is ${w.get(n)}, skipping assignment.`);
      return;
    }
    if (S.length === 0) return;
    const e = S.shift();
    e && (w.set(n, "busy"), W.set(n, e), console.log(`Assigning Job ${e.start} - ${e.start + e.count} to ${n}`), await g.sendRenderRequest(n, e.start, e.count, {
      ...T,
      fileType: "obj"
    }));
  }, De = async () => {
    const n = Array.from($.keys()).sort((d, u) => d - u), { Muxer: e, ArrayBufferTarget: t } = await ie(async () => {
      const { Muxer: d, ArrayBufferTarget: u } = await import("./webm-muxer-MLtUgOCn.js");
      return {
        Muxer: d,
        ArrayBufferTarget: u
      };
    }, []), r = new e({
      target: new t(),
      video: {
        codec: "V_VP9",
        width: T.width,
        height: T.height,
        frameRate: T.fps
      }
    });
    for (const d of n) {
      const u = $.get(d);
      if (u) for (const p of u) r.addVideoChunk(new EncodedVideoChunk({
        type: p.type,
        timestamp: p.timestamp,
        duration: p.duration,
        data: p.data
      }), {
        decoderConfig: p.decoderConfig
      });
    }
    r.finalize();
    const { buffer: i } = r.target, s = new Blob([
      i
    ], {
      type: "video/webm"
    }), a = URL.createObjectURL(s), l = document.createElement("a");
    l.href = a, l.download = `distributed_trace_${Date.now()}.webm`, document.body.appendChild(l), l.click(), document.body.removeChild(l), URL.revokeObjectURL(a), o.setStatus("Finished!");
  }, ce = async (n, e, t) => {
    console.log(`[Worker] Starting Render: Frames ${n} - ${n + e}`), o.setStatus(`Remote Rendering: ${n}-${n + e}`), m = false;
    const r = {
      ...t,
      startFrame: n,
      duration: e / t.fps
    };
    try {
      o.setRecordingState(true, `Remote: ${e} f`);
      const i = await H.recordChunks(r, (s, a) => o.setRecordingState(true, `Remote: ${s}/${a}`));
      console.log("Sending Chunks back to Host..."), o.setRecordingState(true, "Uploading..."), await g.sendRenderResult(i, n), o.setRecordingState(false), o.setStatus("Idle");
    } catch (i) {
      console.error("Remote Recording Failed", i), o.setStatus("Recording Failed");
    } finally {
      m = true, requestAnimationFrame(z);
    }
  }, We = async () => {
    if (!M) return;
    const { start: n, count: e, config: t } = M;
    M = null, await ce(n, e, t);
  };
  g.onStatusChange = (n) => o.setStatus(`Status: ${n}`);
  g.onWorkerLeft = (n) => {
    console.log(`Worker Left: ${n}`), o.setStatus(`Worker Left: ${n}`), w.delete(n);
    const e = W.get(n);
    e && (console.warn(`Worker ${n} failed job ${e.start}. Re-queueing.`), S.unshift(e), W.delete(n), o.setStatus(`Re-queued Job ${e.start}`));
  };
  g.onWorkerReady = (n) => {
    console.log(`Worker ${n} is READY`), o.setStatus(`Worker ${n} Ready!`), w.set(n, "idle"), v === "host" && S.length > 0 && oe(n);
  };
  g.onWorkerJoined = (n) => {
    o.setStatus(`Worker Joined: ${n}`), w.set(n, "idle"), v === "host" && S.length > 0 && ae(n);
  };
  g.onRenderRequest = async (n, e, t) => {
    if (console.log(`[Worker] Received Render Request: Frames ${n} - ${n + e}`), j) {
      console.log(`[Worker] Scene loading in progress. Queueing Render Request for ${n}`), M = {
        start: n,
        count: e,
        config: t
      };
      return;
    }
    await ce(n, e, t);
  };
  g.onRenderResult = async (n, e, t) => {
    console.log(`[Host] Received ${n.length} chunks for ${e} from ${t}`), $.set(e, n), P++, o.setStatus(`Distributed Progress: ${P} / ${F} jobs`), w.set(t, "idle"), W.delete(t), await oe(t), P >= F && (console.log("All jobs complete. Muxing..."), o.setStatus("Muxing..."), await De());
  };
  g.onSceneReceived = async (n, e) => {
    console.log("Scene received successfully."), j = true, o.setRenderConfig(e), x = e.fileType, e.fileType, b = n, o.sceneSelect.value = "viewer", await O("viewer", false), e.anim !== void 0 && (o.animSelect.value = e.anim.toString(), f.setAnimation(e.anim)), j = false, console.log("Scene Loaded. Sending WORKER_READY."), await g.sendWorkerReady(), We();
  };
  const Ie = () => {
    o.onRenderStart = () => {
      m = true;
    }, o.onRenderStop = () => {
      m = false;
    }, o.onSceneSelect = (n) => O(n, false), o.onResolutionChange = Y, o.onRecompile = (n, e) => {
      m = false, _.buildPipeline(n, e), _.recreateBindGroup(), _.resetAccumulation(), C = 0, m = true;
    }, o.onFileSelect = async (n) => {
      var _a;
      ((_a = n.name.split(".").pop()) == null ? void 0 : _a.toLowerCase()) === "obj" ? (b = await n.text(), x = "obj") : (b = await n.arrayBuffer(), x = "glb"), o.sceneSelect.value = "viewer", O("viewer", false);
    }, o.onAnimSelect = (n) => f.setAnimation(n), o.onRecordStart = async () => {
      if (!H.recording) if (v === "host") {
        const n = g.getWorkerIds();
        if (T = o.getRenderConfig(), q = Math.ceil(T.fps * T.duration), !confirm(`Distribute recording? (Workers: ${n.length})
Auto Scene Sync enabled.`)) return;
        S = [], $.clear(), P = 0, W.clear();
        for (let e = 0; e < q; e += ne) {
          const t = Math.min(ne, q - e);
          S.push({
            start: e,
            count: t
          });
        }
        F = S.length, n.forEach((e) => w.set(e, "idle")), o.setStatus(`Distributed Progress: 0 / ${F} jobs (Waiting for workers...)`), n.length > 0 ? (o.setStatus("Syncing Scene to Workers..."), await ae()) : console.log("No workers yet. Waiting...");
      } else {
        m = false, o.setRecordingState(true);
        const n = o.getRenderConfig();
        try {
          await H.record(n, (e, t) => o.setRecordingState(true, `Rec: ${e}/${t} (${Math.round(e / t * 100)}%)`), (e) => {
            const t = document.createElement("a");
            t.href = e, t.download = `raytrace_${Date.now()}.webm`, t.click(), URL.revokeObjectURL(e);
          });
        } catch {
          alert("Recording failed.");
        } finally {
          o.setRecordingState(false), m = true, o.updateRenderButton(true), requestAnimationFrame(z);
        }
      }
    }, o.onConnectHost = () => {
      v === "host" ? (g.disconnect(), v = null, o.setConnectionState(null)) : (g.connect("host"), v = "host", o.setConnectionState("host"));
    }, o.onConnectWorker = () => {
      v === "worker" ? (g.disconnect(), v = null, o.setConnectionState(null)) : (g.connect("worker"), v = "worker", o.setConnectionState("worker"));
    }, o.setConnectionState(null);
  };
  async function Ue() {
    try {
      await _.init(), await f.initWasm();
    } catch (n) {
      alert("Init failed: " + n);
      return;
    }
    Ie(), Ee(), Y(), O("cornell", false), requestAnimationFrame(z);
  }
  Ue().catch(console.error);
})();
