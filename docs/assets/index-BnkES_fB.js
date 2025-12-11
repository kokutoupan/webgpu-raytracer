(function() {
  const e = document.createElement("link").relList;
  if (e && e.supports && e.supports("modulepreload")) return;
  for (const i of document.querySelectorAll('link[rel="modulepreload"]')) a(i);
  new MutationObserver((i) => {
    for (const l of i) if (l.type === "childList") for (const c of l.addedNodes) c.tagName === "LINK" && c.rel === "modulepreload" && a(c);
  }).observe(document, { childList: true, subtree: true });
  function r(i) {
    const l = {};
    return i.integrity && (l.integrity = i.integrity), i.referrerPolicy && (l.referrerPolicy = i.referrerPolicy), i.crossOrigin === "use-credentials" ? l.credentials = "include" : i.crossOrigin === "anonymous" ? l.credentials = "omit" : l.credentials = "same-origin", l;
  }
  function a(i) {
    if (i.ep) return;
    i.ep = true;
    const l = r(i);
    fetch(i.href, l);
  }
})();
const Te = `// =========================================================
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
let n;
function ke(t) {
  const e = n.__externref_table_alloc();
  return n.__wbindgen_externrefs.set(e, t), e;
}
function Be(t, e) {
  return t = t >>> 0, A().subarray(t / 1, t / 1 + e);
}
let B = null;
function pe() {
  return (B === null || B.buffer.detached === true || B.buffer.detached === void 0 && B.buffer !== n.memory.buffer) && (B = new DataView(n.memory.buffer)), B;
}
function be(t, e) {
  return t = t >>> 0, Re(t, e);
}
let O = null;
function A() {
  return (O === null || O.byteLength === 0) && (O = new Uint8Array(n.memory.buffer)), O;
}
function Se(t, e) {
  try {
    return t.apply(this, e);
  } catch (r) {
    const a = ke(r);
    n.__wbindgen_exn_store(a);
  }
}
function me(t) {
  return t == null;
}
function Pe(t, e) {
  const r = e(t.length * 1, 1) >>> 0;
  return A().set(t, r / 1), S = t.length, r;
}
function re(t, e, r) {
  if (r === void 0) {
    const d = G.encode(t), b = e(d.length, 1) >>> 0;
    return A().subarray(b, b + d.length).set(d), S = d.length, b;
  }
  let a = t.length, i = e(a, 1) >>> 0;
  const l = A();
  let c = 0;
  for (; c < a; c++) {
    const d = t.charCodeAt(c);
    if (d > 127) break;
    l[i + c] = d;
  }
  if (c !== a) {
    c !== 0 && (t = t.slice(c)), i = r(i, a, a = c + t.length * 3, 1) >>> 0;
    const d = A().subarray(i + c, i + a), b = G.encodeInto(t, d);
    c += b.written, i = r(i, a, c, 1) >>> 0;
  }
  return S = c, i;
}
let X = new TextDecoder("utf-8", { ignoreBOM: true, fatal: true });
X.decode();
const Ae = 2146435072;
let ne = 0;
function Re(t, e) {
  return ne += e, ne >= Ae && (X = new TextDecoder("utf-8", { ignoreBOM: true, fatal: true }), X.decode(), ne = e), X.decode(A().subarray(t, t + e));
}
const G = new TextEncoder();
"encodeInto" in G || (G.encodeInto = function(t, e) {
  const r = G.encode(t);
  return e.set(r), { read: t.length, written: r.length };
});
let S = 0;
const we = typeof FinalizationRegistry > "u" ? { register: () => {
}, unregister: () => {
} } : new FinalizationRegistry((t) => n.__wbg_world_free(t >>> 0, 1));
class ie {
  __destroy_into_raw() {
    const e = this.__wbg_ptr;
    return this.__wbg_ptr = 0, we.unregister(this), e;
  }
  free() {
    const e = this.__destroy_into_raw();
    n.__wbg_world_free(e, 0);
  }
  camera_ptr() {
    return n.world_camera_ptr(this.__wbg_ptr) >>> 0;
  }
  joints_len() {
    return n.world_joints_len(this.__wbg_ptr) >>> 0;
  }
  joints_ptr() {
    return n.world_joints_ptr(this.__wbg_ptr) >>> 0;
  }
  indices_len() {
    return n.world_indices_len(this.__wbg_ptr) >>> 0;
  }
  indices_ptr() {
    return n.world_indices_ptr(this.__wbg_ptr) >>> 0;
  }
  normals_len() {
    return n.world_normals_len(this.__wbg_ptr) >>> 0;
  }
  normals_ptr() {
    return n.world_normals_ptr(this.__wbg_ptr) >>> 0;
  }
  weights_len() {
    return n.world_weights_len(this.__wbg_ptr) >>> 0;
  }
  weights_ptr() {
    return n.world_weights_ptr(this.__wbg_ptr) >>> 0;
  }
  vertices_len() {
    return n.world_vertices_len(this.__wbg_ptr) >>> 0;
  }
  vertices_ptr() {
    return n.world_vertices_ptr(this.__wbg_ptr) >>> 0;
  }
  instances_len() {
    return n.world_instances_len(this.__wbg_ptr) >>> 0;
  }
  instances_ptr() {
    return n.world_instances_ptr(this.__wbg_ptr) >>> 0;
  }
  update_camera(e, r) {
    n.world_update_camera(this.__wbg_ptr, e, r);
  }
  attributes_len() {
    return n.world_attributes_len(this.__wbg_ptr) >>> 0;
  }
  attributes_ptr() {
    return n.world_attributes_ptr(this.__wbg_ptr) >>> 0;
  }
  constructor(e, r, a) {
    const i = re(e, n.__wbindgen_malloc, n.__wbindgen_realloc), l = S;
    var c = me(r) ? 0 : re(r, n.__wbindgen_malloc, n.__wbindgen_realloc), d = S, b = me(a) ? 0 : Pe(a, n.__wbindgen_malloc), m = S;
    const y = n.world_new(i, l, c, d, b, m);
    return this.__wbg_ptr = y >>> 0, we.register(this, this.__wbg_ptr, this), this;
  }
  update(e) {
    n.world_update(this.__wbg_ptr, e);
  }
  blas_len() {
    return n.world_blas_len(this.__wbg_ptr) >>> 0;
  }
  blas_ptr() {
    return n.world_blas_ptr(this.__wbg_ptr) >>> 0;
  }
  tlas_len() {
    return n.world_tlas_len(this.__wbg_ptr) >>> 0;
  }
  tlas_ptr() {
    return n.world_tlas_ptr(this.__wbg_ptr) >>> 0;
  }
}
Symbol.dispose && (ie.prototype[Symbol.dispose] = ie.prototype.free);
const Ie = /* @__PURE__ */ new Set(["basic", "cors", "default"]);
async function Ee(t, e) {
  if (typeof Response == "function" && t instanceof Response) {
    if (typeof WebAssembly.instantiateStreaming == "function") try {
      return await WebAssembly.instantiateStreaming(t, e);
    } catch (a) {
      if (t.ok && Ie.has(t.type) && t.headers.get("Content-Type") !== "application/wasm") console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", a);
      else throw a;
    }
    const r = await t.arrayBuffer();
    return await WebAssembly.instantiate(r, e);
  } else {
    const r = await WebAssembly.instantiate(t, e);
    return r instanceof WebAssembly.Instance ? { instance: r, module: t } : r;
  }
}
function ze() {
  const t = {};
  return t.wbg = {}, t.wbg.__wbg___wbindgen_throw_dd24417ed36fc46e = function(e, r) {
    throw new Error(be(e, r));
  }, t.wbg.__wbg_error_7534b8e9a36f1ab4 = function(e, r) {
    let a, i;
    try {
      a = e, i = r, console.error(be(e, r));
    } finally {
      n.__wbindgen_free(a, i, 1);
    }
  }, t.wbg.__wbg_getRandomValues_1c61fac11405ffdc = function() {
    return Se(function(e, r) {
      globalThis.crypto.getRandomValues(Be(e, r));
    }, arguments);
  }, t.wbg.__wbg_new_8a6f238a6ece86ea = function() {
    return new Error();
  }, t.wbg.__wbg_stack_0ed75d68575b0f3c = function(e, r) {
    const a = r.stack, i = re(a, n.__wbindgen_malloc, n.__wbindgen_realloc), l = S;
    pe().setInt32(e + 4, l, true), pe().setInt32(e + 0, i, true);
  }, t.wbg.__wbindgen_init_externref_table = function() {
    const e = n.__wbindgen_externrefs, r = e.grow(4);
    e.set(0, void 0), e.set(r + 0, void 0), e.set(r + 1, null), e.set(r + 2, true), e.set(r + 3, false);
  }, t;
}
function Le(t, e) {
  return n = t.exports, xe.__wbindgen_wasm_module = e, B = null, O = null, n.__wbindgen_start(), n;
}
async function xe(t) {
  if (n !== void 0) return n;
  typeof t < "u" && (Object.getPrototypeOf(t) === Object.prototype ? { module_or_path: t } = t : console.warn("using deprecated parameters for the initialization function; pass a single object instead")), typeof t > "u" && (t = new URL("/webgpu-raytracer/assets/rust_shader_tools_bg-C3ONkwZv.wasm", import.meta.url));
  const e = ze();
  (typeof t == "string" || typeof Request == "function" && t instanceof Request || typeof URL == "function" && t instanceof URL) && (t = fetch(t));
  const { instance: r, module: a } = await Ee(await t, e);
  return Le(r, a);
}
const u = document.getElementById("gpu-canvas"), F = document.getElementById("render-btn"), ve = document.getElementById("scene-select"), he = document.getElementById("res-width"), ye = document.getElementById("res-height"), se = document.getElementById("obj-file");
se && (se.accept = ".obj,.glb,.vrm");
const Ue = document.getElementById("update-interval"), Ce = document.getElementById("max-depth"), De = document.getElementById("spp-frame"), Ne = document.getElementById("recompile-btn"), ae = document.createElement("div");
Object.assign(ae.style, { position: "fixed", bottom: "10px", left: "10px", color: "#0f0", background: "rgba(0,0,0,0.7)", padding: "8px", fontFamily: "monospace", fontSize: "14px", pointerEvents: "none", zIndex: "9999", borderRadius: "4px" });
document.body.appendChild(ae);
let M = 0, v = false, s = null, h = null;
async function Fe() {
  if (!navigator.gpu) {
    alert("WebGPU not supported.");
    return;
  }
  const t = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!t) throw new Error("No adapter");
  const e = await t.requestDevice(), r = u.getContext("webgpu");
  r.configure({ device: e, format: "rgba8unorm", usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT }), h = (await xe()).memory, console.log("Wasm initialized");
  let i, l;
  const c = () => {
    const o = parseInt(Ce.value, 10) || 10, f = parseInt(De.value, 10) || 1;
    console.log(`Recompiling Shader... Depth:${o}, SPP:${f}`);
    let _ = Te;
    _ = _.replace(/const\s+MAX_DEPTH\s*=\s*\d+u;/, `const MAX_DEPTH = ${o}u;`), _ = _.replace(/const\s+SPP\s*=\s*\d+u;/, `const SPP = ${f}u;`);
    const g = e.createShaderModule({ label: "RayTracing", code: _ });
    i = e.createComputePipeline({ label: "Main Pipeline", layout: "auto", compute: { module: g, entryPoint: "main" } }), l = i.getBindGroupLayout(0);
  };
  c();
  let d, b, m, y, x, R, I, E, z, T, L, P, Y, K = 0;
  const V = () => {
    if (!m) return;
    const o = new Float32Array(K / 4);
    e.queue.writeBuffer(m, 0, o), M = 0;
  }, oe = () => {
    let o = parseInt(he.value, 10), f = parseInt(ye.value, 10);
    (isNaN(o) || o < 1) && (o = 720), (isNaN(f) || f < 1) && (f = 480), u.width = o, u.height = f, d && d.destroy(), d = e.createTexture({ size: [u.width, u.height], format: "rgba8unorm", usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC }), b = d.createView(), K = u.width * u.height * 16, m && m.destroy(), m = e.createBuffer({ size: K, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }), y || (y = e.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }));
  }, Z = () => {
    !b || !m || !y || !x || !R || !E || !z || !I || !T || !L || !P || (Y = e.createBindGroup({ layout: l, entries: [{ binding: 0, resource: b }, { binding: 1, resource: { buffer: m } }, { binding: 2, resource: { buffer: y } }, { binding: 3, resource: { buffer: x } }, { binding: 4, resource: { buffer: R } }, { binding: 5, resource: { buffer: E } }, { binding: 6, resource: { buffer: z } }, { binding: 7, resource: { buffer: T } }, { binding: 8, resource: { buffer: I } }, { binding: 9, resource: { buffer: L } }, { binding: 10, resource: { buffer: P } }] }));
  }, J = (o, f = true) => {
    console.log(`Loading Scene: ${o}...`), v = false, s && s.free();
    let _, g;
    if (o === "viewer" && U && (W === "obj" ? _ = U : W === "glb" && (g = new Uint8Array(U))), console.time("Rust Build"), s = new ie(o, _, g), console.timeEnd("Rust Build"), !h) return;
    const p = (k, N) => new Float32Array(h.buffer, k, N), Q = (k, N) => new Uint32Array(h.buffer, k, N), C = p(s.vertices_ptr(), s.vertices_len()), _e = p(s.normals_ptr(), s.normals_len()), $ = Q(s.indices_ptr(), s.indices_len()), ee = p(s.attributes_ptr(), s.attributes_len()), j = p(s.tlas_ptr(), s.tlas_len()), D = p(s.blas_ptr(), s.blas_len()), H = p(s.instances_ptr(), s.instances_len());
    console.log(`Scene Stats: 
      Vertices: ${C.length / 4}
      Triangles: ${$.length / 3}
      TLAS Nodes: ${j.length / 8}
      BLAS Nodes: ${D.length / 8}
      Instances: ${H.length / 36}
    `);
    const w = (k) => {
      const N = Math.max(k.byteLength, 4), ge = e.createBuffer({ size: N, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      return k.byteLength > 0 && e.queue.writeBuffer(ge, 0, k), ge;
    };
    R && R.destroy(), R = w(C), I && I.destroy(), I = w(_e), E && E.destroy(), E = w($), z && z.destroy(), z = w(ee), T && T.destroy(), T = w(j), L && L.destroy(), L = w(D), P && P.destroy(), P = w(H), x || (x = e.createBuffer({ size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })), s.update_camera(u.width, u.height);
    const te = p(s.camera_ptr(), 24);
    e.queue.writeBuffer(x, 0, te), Z(), V(), f ? (v = true, F.textContent = "Stop Rendering") : (v = false, F.textContent = "Render Start");
  }, ce = new Uint32Array(1), le = { texture: null };
  let de = performance.now(), q = 0;
  const ue = () => {
    if (requestAnimationFrame(ue), !v || !Y || !s) return;
    let o = parseInt(Ue.value, 10);
    if ((isNaN(o) || o < 0) && (o = 0), o > 0 && M >= o) {
      const $ = performance.now() / 1e3;
      s.update($);
      const ee = s.tlas_ptr(), j = s.tlas_len(), D = new Float32Array(h.buffer, ee, j);
      T.size >= D.byteLength ? e.queue.writeBuffer(T, 0, D) : console.warn("TLAS buffer size mismatch during update. Animation might glitch.");
      const H = s.instances_ptr(), w = s.instances_len(), te = new Float32Array(h.buffer, H, w);
      e.queue.writeBuffer(P, 0, te), V();
    }
    const f = Math.ceil(u.width / 8), _ = Math.ceil(u.height / 8);
    M++, q++, ce[0] = M, e.queue.writeBuffer(y, 0, ce), le.texture = r.getCurrentTexture();
    const g = e.createCommandEncoder(), p = g.beginComputePass();
    p.setPipeline(i), p.setBindGroup(0, Y), p.dispatchWorkgroups(f, _), p.end();
    const Q = { width: u.width, height: u.height, depthOrArrayLayers: 1 };
    g.copyTextureToTexture({ texture: d }, le, Q), e.queue.submit([g.finish()]);
    const C = performance.now();
    C - de >= 1e3 && (ae.textContent = `FPS: ${q} | ${(1e3 / q).toFixed(2)}ms | Frame: ${M} | Res: ${u.width}x${u.height}`, q = 0, de = C);
  };
  let U = null, W = null;
  F.addEventListener("click", () => {
    v = !v, F.textContent = v ? "Stop Rendering" : "Resume Rendering";
  }), ve.addEventListener("change", (o) => {
    const f = o.target;
    J(f.value, false);
  }), se.addEventListener("change", async (o) => {
    var _a, _b;
    const f = o.target, _ = (_a = f.files) == null ? void 0 : _a[0];
    if (!_) return;
    console.log(`Reading ${_.name}...`);
    const g = (_b = _.name.split(".").pop()) == null ? void 0 : _b.toLowerCase();
    if (g === "obj") U = await _.text(), W = "obj";
    else if (g === "glb" || g === "vrm") U = await _.arrayBuffer(), W = "glb";
    else {
      alert("Unsupported file format");
      return;
    }
    ve.value = "viewer", J("viewer", false), f.value = "";
  });
  const fe = () => {
    if (oe(), s && h && x) {
      s.update_camera(u.width, u.height);
      const o = new Float32Array(h.buffer, s.camera_ptr(), 24);
      e.queue.writeBuffer(x, 0, o);
    }
    Z(), V();
  };
  he.addEventListener("change", fe), ye.addEventListener("change", fe), Ne.addEventListener("click", () => {
    v = false, c(), Z(), V(), v = true, F.textContent = "Stop Rendering";
  }), oe(), J("cornell", false), requestAnimationFrame(ue);
}
Fe().catch(console.error);
