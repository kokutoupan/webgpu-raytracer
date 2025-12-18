var __defProp = Object.defineProperty;
var __defNormalProp = (obj, key, value) => key in obj ? __defProp(obj, key, { enumerable: true, configurable: true, writable: true, value }) : obj[key] = value;
var __publicField = (obj, key, value) => __defNormalProp(obj, typeof key !== "symbol" ? key + "" : key, value);
(async () => {
  (function() {
    const e = document.createElement("link").relList;
    if (e && e.supports && e.supports("modulepreload")) return;
    for (const r of document.querySelectorAll('link[rel="modulepreload"]')) n(r);
    new MutationObserver((r) => {
      for (const i of r) if (i.type === "childList") for (const o of i.addedNodes) o.tagName === "LINK" && o.rel === "modulepreload" && n(o);
    }).observe(document, {
      childList: true,
      subtree: true
    });
    function t(r) {
      const i = {};
      return r.integrity && (i.integrity = r.integrity), r.referrerPolicy && (i.referrerPolicy = r.referrerPolicy), r.crossOrigin === "use-credentials" ? i.credentials = "include" : r.crossOrigin === "anonymous" ? i.credentials = "omit" : i.credentials = "same-origin", i;
    }
    function n(r) {
      if (r.ep) return;
      r.ep = true;
      const i = t(r);
      fetch(r.href, i);
    }
  })();
  const K = "modulepreload", X = function(s) {
    return "/webgpu-raytracer/" + s;
  }, N = {}, z = function(e, t, n) {
    let r = Promise.resolve();
    if (t && t.length > 0) {
      let p = function(f) {
        return Promise.all(f.map((y) => Promise.resolve(y).then((B) => ({
          status: "fulfilled",
          value: B
        }), (B) => ({
          status: "rejected",
          reason: B
        }))));
      };
      var o = p;
      document.getElementsByTagName("link");
      const c = document.querySelector("meta[property=csp-nonce]"), d = (c == null ? void 0 : c.nonce) || (c == null ? void 0 : c.getAttribute("nonce"));
      r = p(t.map((f) => {
        if (f = X(f), f in N) return;
        N[f] = true;
        const y = f.endsWith(".css"), B = y ? '[rel="stylesheet"]' : "";
        if (document.querySelector(`link[href="${f}"]${B}`)) return;
        const w = document.createElement("link");
        if (w.rel = y ? "stylesheet" : K, y || (w.as = "script"), w.crossOrigin = "", w.href = f, d && w.setAttribute("nonce", d), document.head.appendChild(w), y) return new Promise((J, Y) => {
          w.addEventListener("load", J), w.addEventListener("error", () => Y(new Error(`Unable to preload CSS for ${f}`)));
        });
      }));
    }
    function i(c) {
      const d = new Event("vite:preloadError", {
        cancelable: true
      });
      if (d.payload = c, window.dispatchEvent(d), !d.defaultPrevented) throw c;
    }
    return r.then((c) => {
      for (const d of c || []) d.status === "rejected" && i(d.reason);
      return e().catch(i);
    });
  }, Q = `// =========================================================
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
  class Z {
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
      let n = Q;
      n = n.replace(/const\s+MAX_DEPTH\s*=\s*\d+u;/, `const MAX_DEPTH = ${e}u;`), n = n.replace(/const\s+SPP\s*=\s*\d+u;/, `const SPP = ${t}u;`);
      const r = this.device.createShaderModule({
        label: "RayTracing",
        code: n
      });
      this.pipeline = this.device.createComputePipeline({
        label: "Main Pipeline",
        layout: "auto",
        compute: {
          module: r,
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
      const n = [];
      for (let r = 0; r < t; r++) {
        const i = e.getTexture(r);
        if (i) try {
          const o = new Blob([
            i
          ]), c = await createImageBitmap(o, {
            resizeWidth: 1024,
            resizeHeight: 1024
          });
          n.push(c);
        } catch (o) {
          console.warn(`Failed tex ${r}`, o), n.push(await this.createFallbackBitmap());
        }
        else n.push(await this.createFallbackBitmap());
      }
      this.texture && this.texture.destroy(), this.texture = this.device.createTexture({
        size: [
          1024,
          1024,
          n.length
        ],
        format: "rgba8unorm",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT
      });
      for (let r = 0; r < n.length; r++) this.device.queue.copyExternalImageToTexture({
        source: n[r]
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
    }
    async createFallbackBitmap() {
      const e = document.createElement("canvas");
      e.width = 1024, e.height = 1024;
      const t = e.getContext("2d");
      return t.fillStyle = "white", t.fillRect(0, 0, 1024, 1024), await createImageBitmap(e);
    }
    ensureBuffer(e, t, n) {
      if (e && e.size >= t) return e;
      e && e.destroy();
      let r = Math.ceil(t * 1.5);
      return r = r + 3 & -4, r = Math.max(r, 4), this.device.createBuffer({
        label: n,
        size: r,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
      });
    }
    updateBuffer(e, t) {
      const n = t.byteLength;
      let r = false, i;
      return e === "index" ? ((!this.indexBuffer || this.indexBuffer.size < n) && (r = true), this.indexBuffer = this.ensureBuffer(this.indexBuffer, n, "IndexBuffer"), i = this.indexBuffer) : e === "attr" ? ((!this.attrBuffer || this.attrBuffer.size < n) && (r = true), this.attrBuffer = this.ensureBuffer(this.attrBuffer, n, "AttrBuffer"), i = this.attrBuffer) : ((!this.instanceBuffer || this.instanceBuffer.size < n) && (r = true), this.instanceBuffer = this.ensureBuffer(this.instanceBuffer, n, "InstanceBuffer"), i = this.instanceBuffer), this.device.queue.writeBuffer(i, 0, t, 0, t.length), r;
    }
    updateCombinedGeometry(e, t, n) {
      const r = e.byteLength + t.byteLength + n.byteLength;
      let i = false;
      (!this.geometryBuffer || this.geometryBuffer.size < r) && (i = true);
      const o = e.length / 4;
      this.vertexCount = o, this.geometryBuffer = this.ensureBuffer(this.geometryBuffer, r, "GeometryBuffer"), !(n.length >= o * 2) && o > 0 && console.warn(`UV buffer mismatch: V=${o}, UV=${n.length / 2}. Filling 0.`);
      let d = 0;
      return this.device.queue.writeBuffer(this.geometryBuffer, d, e), d += e.byteLength, this.device.queue.writeBuffer(this.geometryBuffer, d, t), d += t.byteLength, this.device.queue.writeBuffer(this.geometryBuffer, d, n), i;
    }
    updateCombinedBVH(e, t) {
      const n = e.byteLength, r = t.byteLength, i = n + r;
      let o = false;
      return (!this.nodesBuffer || this.nodesBuffer.size < i) && (o = true), this.nodesBuffer = this.ensureBuffer(this.nodesBuffer, i, "NodesBuffer"), this.device.queue.writeBuffer(this.nodesBuffer, 0, e), this.device.queue.writeBuffer(this.nodesBuffer, n, t), this.blasOffset = e.length / 8, o;
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
      const t = Math.ceil(this.canvas.width / 8), n = Math.ceil(this.canvas.height / 8), r = this.device.createCommandEncoder(), i = r.beginComputePass();
      i.setPipeline(this.pipeline), i.setBindGroup(0, this.bindGroup), i.dispatchWorkgroups(t, n), i.end(), r.copyTextureToTexture({
        texture: this.renderTarget
      }, {
        texture: this.context.getCurrentTexture()
      }, {
        width: this.canvas.width,
        height: this.canvas.height,
        depthOrArrayLayers: 1
      }), this.device.queue.submit([
        r.finish()
      ]);
    }
  }
  function ee(s) {
    return new Worker("/webgpu-raytracer/assets/wasm-worker-hoHDk6V5.js", {
      name: s == null ? void 0 : s.name
    });
  }
  class te {
    constructor() {
      __publicField(this, "worker");
      __publicField(this, "resolveReady", null);
      __publicField(this, "_vertices", new Float32Array(0));
      __publicField(this, "_normals", new Float32Array(0));
      __publicField(this, "_uvs", new Float32Array(0));
      __publicField(this, "_indices", new Uint32Array(0));
      __publicField(this, "_attributes", new Float32Array(0));
      __publicField(this, "_tlas", new Float32Array(0));
      __publicField(this, "_blas", new Float32Array(0));
      __publicField(this, "_instances", new Float32Array(0));
      __publicField(this, "_cameraData", new Float32Array(24));
      __publicField(this, "_textureCount", 0);
      __publicField(this, "_textures", []);
      __publicField(this, "_animations", []);
      __publicField(this, "hasNewData", false);
      __publicField(this, "pendingUpdate", false);
      __publicField(this, "resolveSceneLoad", null);
      __publicField(this, "updateResolvers", []);
      __publicField(this, "lastWidth", -1);
      __publicField(this, "lastHeight", -1);
      this.worker = new ee(), this.worker.onmessage = this.handleMessage.bind(this);
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
      const t = e.data;
      switch (t.type) {
        case "READY":
          console.log("Main: Worker Ready"), (_a = this.resolveReady) == null ? void 0 : _a.call(this);
          break;
        case "SCENE_LOADED":
          this._vertices = t.vertices, this._normals = t.normals, this._uvs = t.uvs, this._indices = t.indices, this._attributes = t.attributes, this._tlas = t.tlas, this._blas = t.blas, this._instances = t.instances, this._cameraData = t.camera, this._textureCount = t.textureCount, this._textures = t.textures || [], this._animations = t.animations || [], this.hasNewData = true, (_b = this.resolveSceneLoad) == null ? void 0 : _b.call(this);
          break;
        case "UPDATE_RESULT":
          this._tlas = t.tlas, this._blas = t.blas, this._instances = t.instances, this._cameraData = t.camera, this._vertices = t.vertices, this._normals = t.normals, this._uvs = t.uvs, this._indices = t.indices, this._attributes = t.attributes, this.hasNewData = true, this.pendingUpdate = false, this.updateResolvers.forEach((n) => n()), this.updateResolvers = [];
          break;
      }
    }
    getAnimationList() {
      return this._animations;
    }
    getTexture(e) {
      return e >= 0 && e < this._textures.length ? this._textures[e] : null;
    }
    loadScene(e, t, n) {
      return this.lastWidth = -1, this.lastHeight = -1, new Promise((r) => {
        this.resolveSceneLoad = r, this.worker.postMessage({
          type: "LOAD_SCENE",
          sceneName: e,
          objSource: t,
          glbData: n
        }, n ? [
          n.buffer
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
    updateCamera(e, t) {
      this.lastWidth === e && this.lastHeight === t || (this.lastWidth = e, this.lastHeight = t, this.worker.postMessage({
        type: "UPDATE_CAMERA",
        width: e,
        height: t
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
    get indices() {
      return this._indices;
    }
    get attributes() {
      return this._attributes;
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
      console.log(`Scene Stats (Worker Proxy): V=${this.vertices.length / 4}, Tri=${this.indices.length / 3}`);
    }
  }
  const u = {
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
  class ne {
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
      this.canvas = this.el(u.ids.canvas), this.btnRender = this.el(u.ids.renderBtn), this.sceneSelect = this.el(u.ids.sceneSelect), this.inputWidth = this.el(u.ids.resWidth), this.inputHeight = this.el(u.ids.resHeight), this.inputFile = this.setupFileInput(), this.inputDepth = this.el(u.ids.maxDepth), this.inputSPP = this.el(u.ids.sppFrame), this.btnRecompile = this.el(u.ids.recompileBtn), this.inputUpdateInterval = this.el(u.ids.updateInterval), this.animSelect = this.el(u.ids.animSelect), this.btnRecord = this.el(u.ids.recordBtn), this.inputRecFps = this.el(u.ids.recFps), this.inputRecDur = this.el(u.ids.recDuration), this.inputRecSpp = this.el(u.ids.recSpp), this.inputRecBatch = this.el(u.ids.recBatch), this.btnHost = this.el(u.ids.btnHost), this.btnWorker = this.el(u.ids.btnWorker), this.statusDiv = this.el(u.ids.statusDiv), this.statsDiv = this.createStatsDiv(), this.bindEvents();
    }
    el(e) {
      const t = document.getElementById(e);
      if (!t) throw new Error(`Element not found: ${e}`);
      return t;
    }
    setupFileInput() {
      const e = this.el(u.ids.objFile);
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
        return (_a = this.onResolutionChange) == null ? void 0 : _a.call(this, parseInt(this.inputWidth.value) || u.defaultWidth, parseInt(this.inputHeight.value) || u.defaultHeight);
      };
      this.inputWidth.addEventListener("change", e), this.inputHeight.addEventListener("change", e), this.btnRecompile.addEventListener("click", () => {
        var _a;
        return (_a = this.onRecompile) == null ? void 0 : _a.call(this, parseInt(this.inputDepth.value) || 10, parseInt(this.inputSPP.value) || 1);
      }), this.inputFile.addEventListener("change", (t) => {
        var _a, _b;
        const n = (_a = t.target.files) == null ? void 0 : _a[0];
        n && ((_b = this.onFileSelect) == null ? void 0 : _b.call(this, n));
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
    updateStats(e, t, n) {
      this.statsDiv.textContent = `FPS: ${e} | ${t.toFixed(2)}ms | Frame: ${n}`;
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
      this.animSelect.disabled = false, e.forEach((t, n) => {
        const r = document.createElement("option");
        r.text = `[${n}] ${t}`, r.value = n.toString(), this.animSelect.add(r);
      }), this.animSelect.value = "0";
    }
    getRenderConfig() {
      return {
        width: parseInt(this.inputWidth.value, 10) || u.defaultWidth,
        height: parseInt(this.inputHeight.value, 10) || u.defaultHeight,
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
  class re {
    constructor(e, t, n) {
      __publicField(this, "isRecording", false);
      __publicField(this, "renderer");
      __publicField(this, "worldBridge");
      __publicField(this, "canvas");
      this.renderer = e, this.worldBridge = t, this.canvas = n;
    }
    get recording() {
      return this.isRecording;
    }
    async record(e, t, n) {
      if (this.isRecording) return;
      this.isRecording = true;
      const { Muxer: r, ArrayBufferTarget: i } = await z(async () => {
        const { Muxer: p, ArrayBufferTarget: f } = await import("./webm-muxer-MLtUgOCn.js");
        return {
          Muxer: p,
          ArrayBufferTarget: f
        };
      }, []), o = Math.ceil(e.fps * e.duration);
      console.log(`Starting recording: ${o} frames @ ${e.fps}fps (VP9)`);
      const c = new r({
        target: new i(),
        video: {
          codec: "V_VP9",
          width: this.canvas.width,
          height: this.canvas.height,
          frameRate: e.fps
        }
      }), d = new VideoEncoder({
        output: (p, f) => c.addVideoChunk(p, f),
        error: (p) => console.error("VideoEncoder Error:", p)
      });
      d.configure({
        codec: "vp09.00.10.08",
        width: this.canvas.width,
        height: this.canvas.height,
        bitrate: 12e6
      });
      try {
        await this.renderAndEncode(o, e, d, t, e.startFrame || 0), await d.flush(), c.finalize();
        const { buffer: p } = c.target, f = new Blob([
          p
        ], {
          type: "video/webm"
        }), y = URL.createObjectURL(f);
        n(y, f);
      } catch (p) {
        throw console.error("Recording failed:", p), p;
      } finally {
        this.isRecording = false;
      }
    }
    async recordChunks(e, t) {
      if (this.isRecording) throw new Error("Already recording");
      this.isRecording = true;
      const n = [], r = Math.ceil(e.fps * e.duration), i = new VideoEncoder({
        output: (o, c) => {
          const d = new Uint8Array(o.byteLength);
          o.copyTo(d), n.push({
            type: o.type,
            timestamp: o.timestamp,
            duration: o.duration,
            data: d.buffer,
            decoderConfig: c == null ? void 0 : c.decoderConfig
          });
        },
        error: (o) => console.error("VideoEncoder Error:", o)
      });
      i.configure({
        codec: "vp09.00.10.08",
        width: this.canvas.width,
        height: this.canvas.height,
        bitrate: 12e6
      });
      try {
        return await this.renderAndEncode(r, e, i, t, e.startFrame || 0), await i.flush(), n;
      } finally {
        this.isRecording = false;
      }
    }
    async renderAndEncode(e, t, n, r, i = 0) {
      for (let o = 0; o < e; o++) {
        r(o, e), await new Promise((f) => setTimeout(f, 0));
        const c = i + o, d = c / t.fps;
        this.worldBridge.update(d), await this.worldBridge.waitForNextUpdate(), await this.updateSceneBuffers(), await this.renderFrame(t.spp, t.batch), n.encodeQueueSize > 5 && await n.flush();
        const p = new VideoFrame(this.canvas, {
          timestamp: c * 1e6 / t.fps,
          duration: 1e6 / t.fps
        });
        n.encode(p, {
          keyFrame: o % t.fps === 0
        }), p.close();
      }
    }
    async updateSceneBuffers() {
      let e = false;
      e || (e = this.renderer.updateCombinedBVH(this.worldBridge.tlas, this.worldBridge.blas)), e || (e = this.renderer.updateBuffer("instance", this.worldBridge.instances)), e || (e = this.renderer.updateCombinedGeometry(this.worldBridge.vertices, this.worldBridge.normals, this.worldBridge.uvs)), e || (e = this.renderer.updateBuffer("index", this.worldBridge.indices)), e || (e = this.renderer.updateBuffer("attr", this.worldBridge.attributes)), this.worldBridge.updateCamera(this.canvas.width, this.canvas.height), this.renderer.updateSceneUniforms(this.worldBridge.cameraData, 0), e && this.renderer.recreateBindGroup(), this.renderer.resetAccumulation();
    }
    async renderFrame(e, t) {
      let n = 0;
      for (; n < e; ) {
        const r = Math.min(t, e - n);
        for (let i = 0; i < r; i++) this.renderer.render(n + i);
        n += r, await this.renderer.device.queue.onSubmittedWorkDone(), n < e && await new Promise((i) => setTimeout(i, 0));
      }
    }
  }
  const se = {
    iceServers: [
      {
        urls: "stun:stun.l.google.com:19302"
      }
    ]
  };
  class F {
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
      this.remoteId = e, this.sendSignal = t, this.pc = new RTCPeerConnection(se), this.pc.onicecandidate = (n) => {
        n.candidate && this.sendSignal({
          type: "candidate",
          candidate: n.candidate.toJSON(),
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
      this.pc.ondatachannel = (n) => {
        this.dc = n.channel, this.setupDataChannel();
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
    async sendScene(e, t, n) {
      if (!this.dc || this.dc.readyState !== "open") return;
      let r;
      typeof e == "string" ? r = new TextEncoder().encode(e) : r = new Uint8Array(e);
      const i = {
        type: "SCENE_INIT",
        totalBytes: r.byteLength,
        config: {
          ...n,
          fileType: t
        }
      };
      this.sendData(i), await this.sendBinaryChunks(r);
    }
    async sendRenderResult(e, t) {
      if (!this.dc || this.dc.readyState !== "open") return;
      let n = 0;
      const r = e.map((c) => {
        const d = c.data.byteLength;
        return n += d, {
          type: c.type,
          timestamp: c.timestamp,
          duration: c.duration,
          size: d,
          decoderConfig: c.decoderConfig
        };
      });
      console.log(`[RTC] Sending Render Result: ${n} bytes, ${e.length} chunks`), this.sendData({
        type: "RENDER_RESULT",
        startFrame: t,
        totalBytes: n,
        chunksMeta: r
      });
      const i = new Uint8Array(n);
      let o = 0;
      for (const c of e) i.set(new Uint8Array(c.data), o), o += c.data.byteLength;
      await this.sendBinaryChunks(i);
    }
    async sendBinaryChunks(e) {
      let n = 0;
      const r = () => new Promise((i) => {
        const o = setInterval(() => {
          (!this.dc || this.dc.bufferedAmount < 65536) && (clearInterval(o), i());
        }, 5);
      });
      for (; n < e.byteLength; ) {
        this.dc && this.dc.bufferedAmount > 256 * 1024 && await r();
        const i = Math.min(n + 16384, e.byteLength);
        if (this.dc) try {
          this.dc.send(e.subarray(n, i));
        } catch {
        }
        n = i, n % (16384 * 5) === 0 && await new Promise((o) => setTimeout(o, 0));
      }
      console.log("[RTC] Transfer Complete");
    }
    setupDataChannel() {
      this.dc && (this.dc.binaryType = "arraybuffer", this.dc.onopen = () => {
        console.log("[RTC] DataChannel Open"), this.onDataChannelOpen && this.onDataChannelOpen();
      }, this.dc.onmessage = (e) => {
        const t = e.data;
        if (typeof t == "string") try {
          const n = JSON.parse(t);
          this.handleControlMessage(n);
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
        let n = 0;
        for (const r of this.resultMeta.chunksMeta) {
          const i = this.receiveBuffer.slice(n, n + r.size);
          t.push({
            type: r.type,
            timestamp: r.timestamp,
            duration: r.duration,
            data: i.buffer,
            decoderConfig: r.decoderConfig
          }), n += r.size;
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
    sendRenderRequest(e, t, n) {
      const r = {
        type: "RENDER_REQUEST",
        startFrame: e,
        frameCount: t,
        config: n
      };
      this.sendData(r);
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
  class ie {
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
      this.ws = new WebSocket(`${u.signalingServerUrl}?token=${t}`), this.ws.onopen = () => {
        var _a2;
        console.log("WS Connected"), (_a2 = this.onStatusChange) == null ? void 0 : _a2.call(this, `Waiting for Peer (${e.toUpperCase()})`), this.sendSignal({
          type: e === "host" ? "register_host" : "register_worker"
        });
      }, this.ws.onmessage = (n) => {
        const r = JSON.parse(n.data);
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
      var _a, _b, _c, _d, _e;
      switch (e.type) {
        case "worker_joined":
          console.log(`Worker joined: ${e.workerId}`);
          const t = new F(e.workerId, (n) => this.sendSignal(n));
          this.workers.set(e.workerId, t), t.onDataChannelOpen = () => {
            var _a2;
            console.log(`[Host] Open for ${e.workerId}`), t.sendData({
              type: "HELLO",
              msg: "Hello from Host!"
            }), (_a2 = this.onWorkerJoined) == null ? void 0 : _a2.call(this, e.workerId);
          }, t.onAckReceived = (n) => {
            console.log(`Worker ${e.workerId} ACK: ${n}`);
          }, t.onRenderResult = (n, r) => {
            var _a2;
            console.log(`Received Render Result from ${e.workerId}: ${n.length} chunks`), (_a2 = this.onRenderResult) == null ? void 0 : _a2.call(this, n, r, e.workerId);
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
          e.workerId && ((_e = this.onWorkerReady) == null ? void 0 : _e.call(this, e.workerId));
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
          e.fromId && (this.hostClient = new F(e.fromId, (t) => this.sendSignal(t)), await this.hostClient.handleOffer(e.sdp), (_a = this.onStatusChange) == null ? void 0 : _a.call(this, "Connected to Host!"), (_b = this.onHostConnected) == null ? void 0 : _b.call(this), this.hostClient.onDataChannelOpen = () => {
            var _a2, _b2;
            (_a2 = this.hostClient) == null ? void 0 : _a2.sendData({
              type: "HELLO",
              msg: "Hello from Worker!"
            }), (_b2 = this.onHostHello) == null ? void 0 : _b2.call(this);
          }, this.hostClient.onSceneReceived = (t, n) => {
            var _a2, _b2;
            (_a2 = this.onSceneReceived) == null ? void 0 : _a2.call(this, t, n);
            const r = typeof t == "string" ? t.length : t.byteLength;
            (_b2 = this.hostClient) == null ? void 0 : _b2.sendAck(r);
          }, this.hostClient.onRenderRequest = (t, n, r) => {
            var _a2;
            (_a2 = this.onRenderRequest) == null ? void 0 : _a2.call(this, t, n, r);
          });
          break;
        case "candidate":
          await ((_c = this.hostClient) == null ? void 0 : _c.handleCandidate(e.candidate));
          break;
      }
    }
    async broadcastScene(e, t, n) {
      const r = Array.from(this.workers.values()).map((i) => i.sendScene(e, t, n));
      await Promise.all(r);
    }
    async sendSceneToWorker(e, t, n, r) {
      const i = this.workers.get(e);
      i && await i.sendScene(t, n, r);
    }
    async sendRenderRequest(e, t, n, r) {
      const i = this.workers.get(e);
      i && await i.sendRenderRequest(t, n, r);
    }
  }
  let m = false, b = null, k = null, _ = null, R = [], D = /* @__PURE__ */ new Map(), A = 0, U = 0, L = 0, S = null, v = /* @__PURE__ */ new Map(), C = /* @__PURE__ */ new Map(), M = false, E = null;
  const O = 20, a = new ne(), h = new Z(a.canvas), l = new te(), I = new re(h, l, a.canvas), g = new ie();
  let x = 0, $ = 0, T = 0, q = performance.now();
  const ae = () => {
    const s = parseInt(a.inputDepth.value, 10) || u.defaultDepth, e = parseInt(a.inputSPP.value, 10) || u.defaultSPP;
    h.buildPipeline(s, e);
  }, H = () => {
    const { width: s, height: e } = a.getRenderConfig();
    h.updateScreenSize(s, e), l.hasWorld && (l.updateCamera(s, e), h.updateSceneUniforms(l.cameraData, 0)), h.recreateBindGroup(), h.resetAccumulation(), x = 0, $ = 0;
  }, P = async (s, e = true) => {
    m = false, console.log(`Loading Scene: ${s}...`);
    let t, n;
    s === "viewer" && b && (k === "obj" ? t = b : k === "glb" && (n = new Uint8Array(b).slice(0))), await l.loadScene(s, t, n), l.printStats(), await h.loadTexturesFromWorld(l), await oe(), H(), a.updateAnimList(l.getAnimationList()), e && (m = true, a.updateRenderButton(true));
  }, oe = async () => {
    h.updateCombinedGeometry(l.vertices, l.normals, l.uvs), h.updateCombinedBVH(l.tlas, l.blas), h.updateBuffer("index", l.indices), h.updateBuffer("attr", l.attributes), h.updateBuffer("instance", l.instances), h.updateSceneUniforms(l.cameraData, 0);
  }, W = () => {
    if (I.recording || (requestAnimationFrame(W), !m || !l.hasWorld)) return;
    let s = parseInt(a.inputUpdateInterval.value, 10) || 0;
    if (s > 0 && x >= s && l.update($ / (s || 1) / 60), l.hasNewData) {
      let t = false;
      t || (t = h.updateCombinedBVH(l.tlas, l.blas)), t || (t = h.updateBuffer("instance", l.instances)), t || (t = h.updateCombinedGeometry(l.vertices, l.normals, l.uvs)), t || (t = h.updateBuffer("index", l.indices)), t || (t = h.updateBuffer("attr", l.attributes)), l.updateCamera(a.canvas.width, a.canvas.height), h.updateSceneUniforms(l.cameraData, 0), t && h.recreateBindGroup(), h.resetAccumulation(), x = 0, l.hasNewData = false;
    }
    x++, T++, $++, h.render(x);
    const e = performance.now();
    e - q >= 1e3 && (a.updateStats(T, 1e3 / T, x), T = 0, q = e);
  }, G = async (s) => {
    const e = a.sceneSelect.value, t = e !== "viewer";
    if (!t && (!b || !k)) return;
    const n = a.getRenderConfig(), r = t ? e : void 0, i = t ? "DUMMY" : b, o = t ? "obj" : k;
    n.sceneName = r, s ? (console.log(`Sending scene to specific worker: ${s}`), v.set(s, "loading"), await g.sendSceneToWorker(s, i, o, n)) : (console.log("Broadcasting scene to all workers..."), g.getWorkerIds().forEach((c) => v.set(c, "loading")), await g.broadcastScene(i, o, n));
  }, V = async (s) => {
    if (v.get(s) !== "idle") {
      console.log(`Worker ${s} is ${v.get(s)}, skipping assignment.`);
      return;
    }
    if (R.length === 0) return;
    const e = R.shift();
    e && (v.set(s, "busy"), C.set(s, e), console.log(`Assigning Job ${e.start} - ${e.start + e.count} to ${s}`), await g.sendRenderRequest(s, e.start, e.count, {
      ...S,
      fileType: "obj"
    }));
  }, ce = async () => {
    const s = Array.from(D.keys()).sort((d, p) => d - p), { Muxer: e, ArrayBufferTarget: t } = await z(async () => {
      const { Muxer: d, ArrayBufferTarget: p } = await import("./webm-muxer-MLtUgOCn.js");
      return {
        Muxer: d,
        ArrayBufferTarget: p
      };
    }, []), n = new e({
      target: new t(),
      video: {
        codec: "V_VP9",
        width: S.width,
        height: S.height,
        frameRate: S.fps
      }
    });
    for (const d of s) {
      const p = D.get(d);
      if (p) for (const f of p) n.addVideoChunk(new EncodedVideoChunk({
        type: f.type,
        timestamp: f.timestamp,
        duration: f.duration,
        data: f.data
      }), {
        decoderConfig: f.decoderConfig
      });
    }
    n.finalize();
    const { buffer: r } = n.target, i = new Blob([
      r
    ], {
      type: "video/webm"
    }), o = URL.createObjectURL(i), c = document.createElement("a");
    c.href = o, c.download = `distributed_trace_${Date.now()}.webm`, document.body.appendChild(c), c.click(), document.body.removeChild(c), URL.revokeObjectURL(o), a.setStatus("Finished!");
  }, j = async (s, e, t) => {
    console.log(`[Worker] Starting Render: Frames ${s} - ${s + e}`), a.setStatus(`Remote Rendering: ${s}-${s + e}`), m = false;
    const n = {
      ...t,
      startFrame: s,
      duration: e / t.fps
    };
    try {
      a.setRecordingState(true, `Remote: ${e} f`);
      const r = await I.recordChunks(n, (i, o) => a.setRecordingState(true, `Remote: ${i}/${o}`));
      console.log("Sending Chunks back to Host..."), a.setRecordingState(true, "Uploading..."), await g.sendRenderResult(r, s), a.setRecordingState(false), a.setStatus("Idle");
    } catch (r) {
      console.error("Remote Recording Failed", r), a.setStatus("Recording Failed");
    } finally {
      m = true, requestAnimationFrame(W);
    }
  }, le = async () => {
    if (!E) return;
    const { start: s, count: e, config: t } = E;
    E = null, await j(s, e, t);
  };
  g.onStatusChange = (s) => a.setStatus(`Status: ${s}`);
  g.onWorkerLeft = (s) => {
    console.log(`Worker Left: ${s}`), a.setStatus(`Worker Left: ${s}`), v.delete(s);
    const e = C.get(s);
    e && (console.warn(`Worker ${s} failed job ${e.start}. Re-queueing.`), R.unshift(e), C.delete(s), a.setStatus(`Re-queued Job ${e.start}`));
  };
  g.onWorkerReady = (s) => {
    console.log(`Worker ${s} is READY`), a.setStatus(`Worker ${s} Ready!`), v.set(s, "idle"), _ === "host" && R.length > 0 && V(s);
  };
  g.onWorkerJoined = (s) => {
    a.setStatus(`Worker Joined: ${s}`), v.set(s, "idle"), _ === "host" && R.length > 0 && G(s);
  };
  g.onRenderRequest = async (s, e, t) => {
    if (console.log(`[Worker] Received Render Request: Frames ${s} - ${s + e}`), M) {
      console.log(`[Worker] Scene loading in progress. Queueing Render Request for ${s}`), E = {
        start: s,
        count: e,
        config: t
      };
      return;
    }
    await j(s, e, t);
  };
  g.onRenderResult = async (s, e, t) => {
    console.log(`[Host] Received ${s.length} chunks for ${e} from ${t}`), D.set(e, s), A++, a.setStatus(`Distributed Progress: ${A} / ${U} jobs`), v.set(t, "idle"), C.delete(t), await V(t), A >= U && (console.log("All jobs complete. Muxing..."), a.setStatus("Muxing..."), await ce());
  };
  g.onSceneReceived = async (s, e) => {
    console.log("Scene received successfully."), M = true, a.setRenderConfig(e), k = e.fileType, e.fileType, b = s, a.sceneSelect.value = e.sceneName || "viewer", await P(e.sceneName || "viewer", false), e.anim !== void 0 && (a.animSelect.value = e.anim.toString(), l.setAnimation(e.anim)), M = false, console.log("Scene Loaded. Sending WORKER_READY."), await g.sendWorkerReady(), le();
  };
  const de = () => {
    a.onRenderStart = () => {
      m = true;
    }, a.onRenderStop = () => {
      m = false;
    }, a.onSceneSelect = (s) => P(s, false), a.onResolutionChange = H, a.onRecompile = (s, e) => {
      m = false, h.buildPipeline(s, e), h.recreateBindGroup(), h.resetAccumulation(), x = 0, m = true;
    }, a.onFileSelect = async (s) => {
      var _a;
      ((_a = s.name.split(".").pop()) == null ? void 0 : _a.toLowerCase()) === "obj" ? (b = await s.text(), k = "obj") : (b = await s.arrayBuffer(), k = "glb"), a.sceneSelect.value = "viewer", P("viewer", false);
    }, a.onAnimSelect = (s) => l.setAnimation(s), a.onRecordStart = async () => {
      if (!I.recording) if (_ === "host") {
        const s = g.getWorkerIds();
        if (S = a.getRenderConfig(), L = Math.ceil(S.fps * S.duration), !confirm(`Distribute recording? (Workers: ${s.length})
Auto Scene Sync enabled.`)) return;
        R = [], D.clear(), A = 0, C.clear();
        for (let e = 0; e < L; e += O) {
          const t = Math.min(O, L - e);
          R.push({
            start: e,
            count: t
          });
        }
        U = R.length, s.forEach((e) => v.set(e, "idle")), a.setStatus(`Distributed Progress: 0 / ${U} jobs (Waiting for workers...)`), s.length > 0 ? (a.setStatus("Syncing Scene to Workers..."), await G()) : console.log("No workers yet. Waiting...");
      } else {
        m = false, a.setRecordingState(true);
        const s = a.getRenderConfig();
        try {
          await I.record(s, (e, t) => a.setRecordingState(true, `Rec: ${e}/${t} (${Math.round(e / t * 100)}%)`), (e) => {
            const t = document.createElement("a");
            t.href = e, t.download = `raytrace_${Date.now()}.webm`, t.click(), URL.revokeObjectURL(e);
          });
        } catch {
          alert("Recording failed.");
        } finally {
          a.setRecordingState(false), m = true, a.updateRenderButton(true), requestAnimationFrame(W);
        }
      }
    }, a.onConnectHost = () => {
      _ === "host" ? (g.disconnect(), _ = null, a.setConnectionState(null)) : (g.connect("host"), _ = "host", a.setConnectionState("host"));
    }, a.onConnectWorker = () => {
      _ === "worker" ? (g.disconnect(), _ = null, a.setConnectionState(null)) : (g.connect("worker"), _ = "worker", a.setConnectionState("worker"));
    }, a.setConnectionState(null);
  };
  async function ue() {
    try {
      await h.init(), await l.initWasm();
    } catch (s) {
      alert("Init failed: " + s);
      return;
    }
    de(), ae(), H(), P("cornell", false), requestAnimationFrame(W);
  }
  ue().catch(console.error);
})();
