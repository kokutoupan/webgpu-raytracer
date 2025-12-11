(function() {
  const e = document.createElement("link").relList;
  if (e && e.supports && e.supports("modulepreload")) return;
  for (const i of document.querySelectorAll('link[rel="modulepreload"]')) a(i);
  new MutationObserver((i) => {
    for (const c of i) if (c.type === "childList") for (const s of c.addedNodes) s.tagName === "LINK" && s.rel === "modulepreload" && a(s);
  }).observe(document, { childList: true, subtree: true });
  function n(i) {
    const c = {};
    return i.integrity && (c.integrity = i.integrity), i.referrerPolicy && (c.referrerPolicy = i.referrerPolicy), i.crossOrigin === "use-credentials" ? c.credentials = "include" : i.crossOrigin === "anonymous" ? c.credentials = "omit" : c.credentials = "same-origin", c;
  }
  function a(i) {
    if (i.ep) return;
    i.ep = true;
    const c = n(i);
    fetch(i.href, c);
  }
})();
const Te = `// =========================================================
//   WebGPU Ray Tracer (Indexed Geometry & Full Tessellation)
// =========================================================

// --- Constants ---
const PI = 3.141592653589793;
const T_MIN = 0.001;
const T_MAX = 1e30;
// \u3053\u308C\u3089\u306E\u5B9A\u6570\u306FTypeScript\u5074\u304B\u3089\u7F6E\u63DB\u3055\u308C\u307E\u3059
const MAX_DEPTH = 10u;
const SPP = 1u;

// --- Bindings ---
// Group 0: Main resources
@group(0) @binding(0) var outputTex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<storage, read_write> accumulateBuffer: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> frame: FrameInfo;
@group(0) @binding(3) var<uniform> camera: Camera;

// New Geometry Buffers
@group(0) @binding(4) var<storage, read> vertices: array<vec4<f32>>;        // [x, y, z, pad]
@group(0) @binding(5) var<storage, read> indices: array<u32>;               // [i0, i1, i2, i3...]
@group(0) @binding(6) var<storage, read> attributes: array<TriangleAttributes>; // [Color, Mat, Extra...]
@group(0) @binding(7) var<storage, read> bvh_nodes: array<BVHNode>;
@group(0) @binding(8) var<storage, read> normals: array<vec4<f32>>; // \u2605\u8FFD\u52A0

// --- Structs ---

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
    left_first: f32, // Leaf: Index of first triangle, Internal: Index of left child
    max_b: vec3<f32>,
    tri_count: f32   // Leaf: Number of triangles, Internal: 0
}

// 8 floats = 32 bytes (Rust\u5074\u306E attributes \u914D\u5217\u3068\u4E00\u81F4\u3055\u305B\u308B)
struct TriangleAttributes {
    data0: vec4<f32>, // xyz: Color, w: MaterialType (bits)
    data1: vec4<f32>, // x: Extra, yzw: Padding
}

// --- Random Utilities ---
fn init_rng(pixel_idx: u32, frame: u32) -> u32 {
    var seed = pixel_idx + frame * 719393u;
    seed ^= 2747636419u; seed *= 2654435769u; seed ^= (seed >> 16u);
    seed *= 2654435769u; seed ^= (seed >> 16u); seed *= 2654435769u;
    return seed;
}
fn rand_pcg(state: ptr<function, u32>) -> f32 {
    let old = *state;
    *state = old * 747796405u + 2891336453u;
    let word = ((*state) >> ((old >> 28u) + 4u)) ^ (*state);
    return f32((word >> 22u) ^ word) / 4294967295.0;
}
fn random_unit_vector(rng: ptr<function, u32>) -> vec3<f32> {
    let z = rand_pcg(rng) * 2.0 - 1.0;
    let a = rand_pcg(rng) * 2.0 * PI;
    let r = sqrt(max(0.0, 1.0 - z * z));
    return vec3<f32>(r * cos(a), r * sin(a), z);
}
fn random_in_unit_disk(rng: ptr<function, u32>) -> vec3<f32> {
    let r = sqrt(rand_pcg(rng));
    let theta = 2.0 * PI * rand_pcg(rng);
    return vec3<f32>(r * cos(theta), r * sin(theta), 0.0);
}

// --- Physics Helpers ---
fn reflectance(cosine: f32, ref_idx: f32) -> f32 {
    var r0 = (1.0 - ref_idx) / (1.0 + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0 - r0) * pow((1.0 - cosine), 5.0);
}

// --- Intersection Logic ---

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
    let e1 = v1 - v0;
    let e2 = v2 - v0;
    let h = cross(r.direction, e2);
    let a = dot(e1, h);
    if abs(a) < 1e-4 { return -1.0; }
    let f = 1.0 / a;
    let s = r.origin - v0;
    let u = f * dot(s, h);
    if u < 0.0 || u > 1.0 { return -1.0; }
    let q = cross(s, e1);
    let v = f * dot(r.direction, q);
    if v < 0.0 || u + v > 1.0 { return -1.0; }
    let t = f * dot(e2, q);
    if t > t_min && t < t_max { return t; }
    return -1.0;
}

// \u91CD\u5FC3\u5EA7\u6A19 (u, v) \u3092\u8A08\u7B97\u3059\u308B\u30D8\u30EB\u30D1\u30FC 
fn get_triangle_barycentrics(v0: vec3<f32>, v1: vec3<f32>, v2: vec3<f32>, r: Ray) -> vec2<f32> {
    let e1 = v1 - v0;
    let e2 = v2 - v0;
    let h = cross(r.direction, e2);
    let a = dot(e1, h);
    let f = 1.0 / a;
    let s = r.origin - v0;
    let u = f * dot(s, h);
    let q = cross(s, e1);
    let v = f * dot(r.direction, q);
    return vec2<f32>(u, v);
}

// \u623B\u308A\u5024: x=t, y=triangle_index
fn hit_bvh(r: Ray, t_min: f32, t_max: f32) -> vec2<f32> {
    var closest_t = t_max;
    var hit_idx = -1.0;

    let inv_d = 1.0 / r.direction;
    var stack: array<u32, 32>;
    var stackptr = 0u;

    // \u30EB\u30FC\u30C8\u30CE\u30FC\u30C9\u4EA4\u5DEE\u5224\u5B9A
    if intersect_aabb(bvh_nodes[0].min_b, bvh_nodes[0].max_b, r, inv_d, t_min, closest_t) < 1e30 {
        stack[stackptr] = 0u;
        stackptr++;
    }

    while stackptr > 0u {
        stackptr--;
        let node_idx = stack[stackptr];
        let node = bvh_nodes[node_idx];

        let count = u32(node.tri_count);

        if count > 0u {
            // Leaf Node
            let first = u32(node.left_first); // \u30BD\u30FC\u30C8\u6E08\u307F\u30A4\u30F3\u30C7\u30C3\u30AF\u30B9\u914D\u5217\u4E0A\u306E\u958B\u59CB\u4F4D\u7F6E

            for (var i = 0u; i < count; i++) {
                let tri_id = first + i; // \u5C5E\u6027\u914D\u5217\u306E\u30A4\u30F3\u30C7\u30C3\u30AF\u30B9\u3067\u3082\u3042\u308B
                let base_idx = tri_id * 3u;

                // \u30A4\u30F3\u30C7\u30C3\u30AF\u30B9\u30D0\u30C3\u30D5\u30A1\u304B\u3089\u9802\u70B9ID\u3092\u53D6\u5F97
                let i0 = indices[base_idx];
                let i1 = indices[base_idx + 1u];
                let i2 = indices[base_idx + 2u];

                // \u9802\u70B9\u30D0\u30C3\u30D5\u30A1\u304B\u3089\u5EA7\u6A19\u3092\u53D6\u5F97
                let v0 = vertices[i0].xyz;
                let v1 = vertices[i1].xyz;
                let v2 = vertices[i2].xyz;

                let t = hit_triangle_raw(v0, v1, v2, r, t_min, closest_t);
                if t > 0.0 {
                    closest_t = t;
                    hit_idx = f32(tri_id); // \u30D2\u30C3\u30C8\u3057\u305F\u4E09\u89D2\u5F62\u306EID (\u5C5E\u6027\u53C2\u7167\u7528)
                }
            }
        } else {
            // Internal Node
            let left_idx = u32(node.left_first);
            let right_idx = left_idx + 1u;
            let node_l = bvh_nodes[left_idx];
            let node_r = bvh_nodes[right_idx];

            let dist_l = intersect_aabb(node_l.min_b, node_l.max_b, r, inv_d, t_min, closest_t);
            let dist_r = intersect_aabb(node_r.min_b, node_r.max_b, r, inv_d, t_min, closest_t);

            let hit_l = dist_l < 1e30;
            let hit_r = dist_r < 1e30;

            if hit_l && hit_r {
                if dist_l < dist_r {
                    stack[stackptr] = right_idx; stackptr++;
                    stack[stackptr] = left_idx;  stackptr++;
                } else {
                    stack[stackptr] = left_idx;  stackptr++;
                    stack[stackptr] = right_idx; stackptr++;
                }
            } else if hit_l {
                stack[stackptr] = left_idx; stackptr++;
            } else if hit_r {
                stack[stackptr] = right_idx; stackptr++;
            }
        }
    }
    return vec2<f32>(closest_t, hit_idx);
}

// --- Main Tracer ---
fn ray_color(r_in: Ray, rng: ptr<function, u32>) -> vec3<f32> {
    var ray = r_in;
    var throughput = vec3<f32>(1.0);

    for (var depth = 0u; depth < MAX_DEPTH; depth++) {
        let hit = hit_bvh(ray, T_MIN, T_MAX);
        let t = hit.x;
        let tri_idx_f = hit.y;

        if tri_idx_f < 0.0 {
            // Miss: Black Background
            return vec3<f32>(0.0);
        }

        // Hit: Fetch Attributes
        let tri_idx = u32(tri_idx_f);
        let base_idx = tri_idx * 3u;

        let i0 = indices[base_idx];
        let i1 = indices[base_idx + 1u];
        let i2 = indices[base_idx + 2u];

        let v0 = vertices[i0].xyz;
        let v1 = vertices[i1].xyz;
        let v2 = vertices[i2].xyz;

        // \u2605\u3053\u3053\u3067UV\u3092\u518D\u8A08\u7B97\u3059\u308B
        let uv = get_triangle_barycentrics(v0, v1, v2, ray);
        let u = uv.x;
        let v = uv.y;
        let w = 1.0 - u - v;

        // \u6CD5\u7DDA\u88DC\u9593 (Smooth Shading)
        let n0 = normals[i0].xyz;
        let n1 = normals[i1].xyz;
        let n2 = normals[i2].xyz;
        
        // \u88DC\u9593\u5F8C\u306E\u6CD5\u7DDA\u3092\u6B63\u898F\u5316
        var normal = normalize(n0 * w + n1 * u + n2 * v);
        var front_face = dot(ray.direction, normal) < 0.0;
        normal = select(-normal, normal, front_face);

        // 2. \u30DE\u30C6\u30EA\u30A2\u30EB\u5C5E\u6027\u3092\u53D6\u5F97
        let attr = attributes[tri_idx];
        let albedo = attr.data0.rgb;
        let mat_bits = bitcast<u32>(attr.data0.w); // f32 -> u32
        let extra = attr.data1.x;

        let p = ray.origin + t * ray.direction;

        // Emission (Light = 3)
        if mat_bits == 3u {
            // \u88CF\u9762\u306F\u5149\u3089\u306A\u3044\u3088\u3046\u306B\u3059\u308B\u3001\u3042\u308B\u3044\u306F\u4E21\u9762\u767A\u5149\u306B\u3059\u308B
            // \u3053\u3053\u3067\u306F\u8868\u9762\u306E\u307F\u767A\u5149\u3068\u3057\u3066\u304A\u304F
            return select(vec3<f32>(0.0), throughput * albedo, front_face);
        }

        var scat = vec3<f32>(0.0);

        if mat_bits == 0u { // Lambertian
            scat = normal + random_unit_vector(rng);
            if length(scat) < 1e-6 { scat = normal; }
        
        } else if mat_bits == 1u { // Metal
            scat = reflect(ray.direction, normal) + extra * random_unit_vector(rng);
            if dot(scat, normal) <= 0.0 { return vec3<f32>(0.0); }
        
        } else { // Dielectric (2u)
            let ir = extra; // index of refraction
            let ratio = select(ir, 1.0 / ir, front_face);
            let unit = normalize(ray.direction);
            let cos_t = min(dot(-unit, normal), 1.0);
            let sin_t = sqrt(1.0 - cos_t * cos_t);

            let cannot = ratio * sin_t > 1.0;
            if cannot || reflectance(cos_t, ratio) > rand_pcg(rng) {
                scat = reflect(unit, normal);
            } else {
                let perp = ratio * (unit + cos_t * normal);
                let para = -sqrt(abs(1.0 - dot(perp, perp))) * normal;
                scat = perp + para;
            }
        }

        let next_origin = p + scat * 1e-4;
        ray = Ray(next_origin, scat);
        throughput *= albedo;

        // Russian Roulette
        if depth > 2u {
            let p_rr = max(throughput.r, max(throughput.g, throughput.b));
            if rand_pcg(rng) > p_rr { break; }
            throughput /= p_rr;
        }
    }
    return vec3<f32>(0.0);
}

// --- Entry Point ---
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(outputTex);
    if id.x >= dims.x || id.y >= dims.y { return; }

    let pixel_idx = id.y * dims.x + id.x;
    var rng = init_rng(pixel_idx, frame.frame_count);

    var accum_color = vec3<f32>(0.0);

    for (var s = 0u; s < SPP; s++) {
        var ro = camera.origin;
        var offset = vec3<f32>(0.0);
        if camera.lens_radius > 0.0 {
            let rd = camera.lens_radius * random_in_unit_disk(&rng);
            offset = camera.u * rd.x + camera.v * rd.y;
            ro += offset;
        }
        let u = (f32(id.x) + rand_pcg(&rng)) / f32(dims.x);
        let v = 1.0 - (f32(id.y) + rand_pcg(&rng)) / f32(dims.y);
        let dir = camera.lower_left_corner + u * camera.horizontal + v * camera.vertical - camera.origin - offset;

        accum_color += ray_color(Ray(ro, dir), &rng);
    }

    let col = accum_color / f32(SPP);

    var acc = vec4<f32>(0.0);
    if frame.frame_count > 1u { acc = accumulateBuffer[pixel_idx]; }
    let new_acc = acc + vec4<f32>(col, 1.0);
    accumulateBuffer[pixel_idx] = new_acc;

    var final_col = new_acc.rgb / f32(frame.frame_count);
    final_col = sqrt(clamp(final_col, vec3<f32>(0.0), vec3<f32>(1.0)));
    textureStore(outputTex, vec2<i32>(id.xy), vec4<f32>(final_col, 1.0));
}
`;
let r;
function Se(t) {
  const e = r.__externref_table_alloc();
  return r.__wbindgen_externrefs.set(e, t), e;
}
function Re(t, e) {
  return t = t >>> 0, U().subarray(t / 1, t / 1 + e);
}
let x = null;
function ce() {
  return (x === null || x.buffer.detached === true || x.buffer.detached === void 0 && x.buffer !== r.memory.buffer) && (x = new DataView(r.memory.buffer)), x;
}
function le(t, e) {
  return t = t >>> 0, ze(t, e);
}
let I = null;
function U() {
  return (I === null || I.byteLength === 0) && (I = new Uint8Array(r.memory.buffer)), I;
}
function Ee(t, e) {
  try {
    return t.apply(this, e);
  } catch (n) {
    const a = Se(n);
    r.__wbindgen_exn_store(a);
  }
}
function ue(t) {
  return t == null;
}
function Ue(t, e) {
  const n = e(t.length * 1, 1) >>> 0;
  return U().set(t, n / 1), B = t.length, n;
}
function H(t, e, n) {
  if (n === void 0) {
    const u = L.encode(t), m = e(u.length, 1) >>> 0;
    return U().subarray(m, m + u.length).set(u), B = u.length, m;
  }
  let a = t.length, i = e(a, 1) >>> 0;
  const c = U();
  let s = 0;
  for (; s < a; s++) {
    const u = t.charCodeAt(s);
    if (u > 127) break;
    c[i + s] = u;
  }
  if (s !== a) {
    s !== 0 && (t = t.slice(s)), i = n(i, a, a = s + t.length * 3, 1) >>> 0;
    const u = U().subarray(i + s, i + a), m = L.encodeInto(t, u);
    s += m.written, i = n(i, a, s, 1) >>> 0;
  }
  return B = s, i;
}
let M = new TextDecoder("utf-8", { ignoreBOM: true, fatal: true });
M.decode();
const Ae = 2146435072;
let j = 0;
function ze(t, e) {
  return j += e, j >= Ae && (M = new TextDecoder("utf-8", { ignoreBOM: true, fatal: true }), M.decode(), j = e), M.decode(U().subarray(t, t + e));
}
const L = new TextEncoder();
"encodeInto" in L || (L.encodeInto = function(t, e) {
  const n = L.encode(t);
  return e.set(n), { read: t.length, written: n.length };
});
let B = 0;
const fe = typeof FinalizationRegistry > "u" ? { register: () => {
}, unregister: () => {
} } : new FinalizationRegistry((t) => r.__wbg_world_free(t >>> 0, 1));
class X {
  __destroy_into_raw() {
    const e = this.__wbg_ptr;
    return this.__wbg_ptr = 0, fe.unregister(this), e;
  }
  free() {
    const e = this.__destroy_into_raw();
    r.__wbg_world_free(e, 0);
  }
  camera_ptr() {
    return r.world_camera_ptr(this.__wbg_ptr) >>> 0;
  }
  joints_len() {
    return r.world_joints_len(this.__wbg_ptr) >>> 0;
  }
  joints_ptr() {
    return r.world_joints_ptr(this.__wbg_ptr) >>> 0;
  }
  indices_len() {
    return r.world_indices_len(this.__wbg_ptr) >>> 0;
  }
  indices_ptr() {
    return r.world_indices_ptr(this.__wbg_ptr) >>> 0;
  }
  normals_len() {
    return r.world_normals_len(this.__wbg_ptr) >>> 0;
  }
  normals_ptr() {
    return r.world_normals_ptr(this.__wbg_ptr) >>> 0;
  }
  weights_len() {
    return r.world_weights_len(this.__wbg_ptr) >>> 0;
  }
  weights_ptr() {
    return r.world_weights_ptr(this.__wbg_ptr) >>> 0;
  }
  vertices_len() {
    return r.world_vertices_len(this.__wbg_ptr) >>> 0;
  }
  vertices_ptr() {
    return r.world_vertices_ptr(this.__wbg_ptr) >>> 0;
  }
  update_camera(e, n) {
    r.world_update_camera(this.__wbg_ptr, e, n);
  }
  attributes_len() {
    return r.world_attributes_len(this.__wbg_ptr) >>> 0;
  }
  attributes_ptr() {
    return r.world_attributes_ptr(this.__wbg_ptr) >>> 0;
  }
  constructor(e, n, a) {
    const i = H(e, r.__wbindgen_malloc, r.__wbindgen_realloc), c = B;
    var s = ue(n) ? 0 : H(n, r.__wbindgen_malloc, r.__wbindgen_realloc), u = B, m = ue(a) ? 0 : Ue(a, r.__wbindgen_malloc), b = B;
    const h = r.world_new(i, c, s, u, m, b);
    return this.__wbg_ptr = h >>> 0, fe.register(this, this.__wbg_ptr, this), this;
  }
  bvh_len() {
    return r.world_bvh_len(this.__wbg_ptr) >>> 0;
  }
  bvh_ptr() {
    return r.world_bvh_ptr(this.__wbg_ptr) >>> 0;
  }
}
Symbol.dispose && (X.prototype[Symbol.dispose] = X.prototype.free);
const Ie = /* @__PURE__ */ new Set(["basic", "cors", "default"]);
async function Le(t, e) {
  if (typeof Response == "function" && t instanceof Response) {
    if (typeof WebAssembly.instantiateStreaming == "function") try {
      return await WebAssembly.instantiateStreaming(t, e);
    } catch (a) {
      if (t.ok && Ie.has(t.type) && t.headers.get("Content-Type") !== "application/wasm") console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", a);
      else throw a;
    }
    const n = await t.arrayBuffer();
    return await WebAssembly.instantiate(n, e);
  } else {
    const n = await WebAssembly.instantiate(t, e);
    return n instanceof WebAssembly.Instance ? { instance: n, module: t } : n;
  }
}
function Ge() {
  const t = {};
  return t.wbg = {}, t.wbg.__wbg___wbindgen_throw_dd24417ed36fc46e = function(e, n) {
    throw new Error(le(e, n));
  }, t.wbg.__wbg_error_7534b8e9a36f1ab4 = function(e, n) {
    let a, i;
    try {
      a = e, i = n, console.error(le(e, n));
    } finally {
      r.__wbindgen_free(a, i, 1);
    }
  }, t.wbg.__wbg_getRandomValues_1c61fac11405ffdc = function() {
    return Ee(function(e, n) {
      globalThis.crypto.getRandomValues(Re(e, n));
    }, arguments);
  }, t.wbg.__wbg_new_8a6f238a6ece86ea = function() {
    return new Error();
  }, t.wbg.__wbg_stack_0ed75d68575b0f3c = function(e, n) {
    const a = n.stack, i = H(a, r.__wbindgen_malloc, r.__wbindgen_realloc), c = B;
    ce().setInt32(e + 4, c, true), ce().setInt32(e + 0, i, true);
  }, t.wbg.__wbindgen_init_externref_table = function() {
    const e = r.__wbindgen_externrefs, n = e.grow(4);
    e.set(0, void 0), e.set(n + 0, void 0), e.set(n + 1, null), e.set(n + 2, true), e.set(n + 3, false);
  }, t;
}
function ke(t, e) {
  return r = t.exports, me.__wbindgen_wasm_module = e, x = null, I = null, r.__wbindgen_start(), r;
}
async function me(t) {
  if (r !== void 0) return r;
  typeof t < "u" && (Object.getPrototypeOf(t) === Object.prototype ? { module_or_path: t } = t : console.warn("using deprecated parameters for the initialization function; pass a single object instead")), typeof t > "u" && (t = new URL("/webgpu-raytracer/assets/rust_shader_tools_bg-BBBmYXEw.wasm", import.meta.url));
  const e = Ge();
  (typeof t == "string" || typeof Request == "function" && t instanceof Request || typeof URL == "function" && t instanceof URL) && (t = fetch(t));
  const { instance: n, module: a } = await Le(await t, e);
  return ke(n, a);
}
const d = document.getElementById("gpu-canvas"), z = document.getElementById("render-btn"), de = document.getElementById("scene-select"), _e = document.getElementById("res-width"), ge = document.getElementById("res-height"), J = document.getElementById("obj-file");
J && (J.accept = ".obj,.glb,.vrm");
const Ce = document.getElementById("max-depth"), De = document.getElementById("spp-frame"), Oe = document.getElementById("recompile-btn"), K = document.createElement("div");
Object.assign(K.style, { position: "fixed", bottom: "10px", left: "10px", color: "#0f0", background: "rgba(0,0,0,0.7)", padding: "8px", fontFamily: "monospace", fontSize: "14px", pointerEvents: "none", zIndex: "9999", borderRadius: "4px" });
document.body.appendChild(K);
let O = 0, v = false, l = null, p = null;
async function Me() {
  if (!navigator.gpu) {
    alert("WebGPU not supported.");
    return;
  }
  const t = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!t) throw new Error("No adapter");
  const e = await t.requestDevice(), n = d.getContext("webgpu");
  if (!n) throw new Error("No context");
  n.configure({ device: e, format: "rgba8unorm", usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT }), p = (await me()).memory, console.log("Wasm initialized");
  let i, c;
  const s = () => {
    const o = parseInt(Ce.value, 10) || 10, _ = parseInt(De.value, 10) || 1;
    console.log(`Recompiling Shader... Depth:${o}, SPP:${_}`);
    let f = Te;
    f = f.replace(/const\s+MAX_DEPTH\s*=\s*\d+u;/, `const MAX_DEPTH = ${o}u;`), f = f.replace(/const\s+SPP\s*=\s*\d+u;/, `const SPP = ${_}u;`);
    const g = e.createShaderModule({ label: "RayTracing", code: f });
    i = e.createComputePipeline({ label: "Main Pipeline", layout: "auto", compute: { module: g, entryPoint: "main" } }), c = i.getBindGroupLayout(0);
  };
  s();
  let u, m, b, h, y, P, T, S, R, E, F, q = 0;
  const N = () => {
    if (!b) return;
    const o = new Float32Array(q / 4);
    e.queue.writeBuffer(b, 0, o), O = 0;
  }, Q = () => {
    let o = parseInt(_e.value, 10), _ = parseInt(ge.value, 10);
    (isNaN(o) || o < 1) && (o = 720), (isNaN(_) || _ < 1) && (_ = 480), d.width = o, d.height = _, u && u.destroy(), u = e.createTexture({ size: [d.width, d.height], format: "rgba8unorm", usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC }), m = u.createView(), q = d.width * d.height * 16, b && b.destroy(), b = e.createBuffer({ size: q, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }), h || (h = e.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }));
  }, V = () => {
    !m || !b || !h || !y || !P || !S || !R || !E || !T || (F = e.createBindGroup({ layout: c, entries: [{ binding: 0, resource: m }, { binding: 1, resource: { buffer: b } }, { binding: 2, resource: { buffer: h } }, { binding: 3, resource: { buffer: y } }, { binding: 4, resource: { buffer: P } }, { binding: 5, resource: { buffer: S } }, { binding: 6, resource: { buffer: R } }, { binding: 7, resource: { buffer: E } }, { binding: 8, resource: { buffer: T } }] }));
  }, W = (o, _ = true) => {
    console.log(`Loading Scene: ${o}... (Rust)`), v = false, l && l.free();
    let f, g;
    if (o === "viewer" && A && (k === "obj" ? f = A : k === "glb" && (g = new Uint8Array(A))), console.time("Rust Build"), l = new X(o, f, g), console.timeEnd("Rust Build"), !p) return;
    const w = l.vertices_ptr(), C = l.vertices_len(), D = new Float32Array(p.buffer, w, C), pe = l.normals_ptr(), be = l.normals_len(), ie = new Float32Array(p.buffer, pe, be), ve = l.indices_ptr(), ae = l.indices_len(), Y = new Uint32Array(p.buffer, ve, ae), we = l.attributes_ptr(), he = l.attributes_len(), $ = new Float32Array(p.buffer, we, he), ye = l.bvh_ptr(), se = l.bvh_len(), oe = new Float32Array(p.buffer, ye, se);
    console.log(`Scene Stats: Verts:${C / 4}, Tris:${ae / 3}, Nodes:${se / 8}`), P && P.destroy(), P = e.createBuffer({ size: D.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }), e.queue.writeBuffer(P, 0, D), T && T.destroy(), T = e.createBuffer({ size: ie.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }), e.queue.writeBuffer(T, 0, ie), S && S.destroy();
    const xe = Math.max(Y.byteLength, 4);
    S = e.createBuffer({ size: xe, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }), Y.byteLength > 0 && e.queue.writeBuffer(S, 0, Y), R && R.destroy();
    const Be = Math.max($.byteLength, 4);
    R = e.createBuffer({ size: Be, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }), $.byteLength > 0 && e.queue.writeBuffer(R, 0, $), E && E.destroy(), E = e.createBuffer({ size: oe.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }), e.queue.writeBuffer(E, 0, oe), y || (y = e.createBuffer({ size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })), l.update_camera(d.width, d.height);
    const Pe = new Float32Array(p.buffer, l.camera_ptr(), 24);
    e.queue.writeBuffer(y, 0, Pe), V(), N(), _ ? (v = true, z.textContent = "Stop Rendering") : (v = false, z.textContent = "Render Start");
  }, Z = new Uint32Array(1), ee = { texture: null };
  let te = performance.now(), G = 0;
  const ne = () => {
    if (requestAnimationFrame(ne), !v || !F) return;
    const o = Math.ceil(d.width / 8), _ = Math.ceil(d.height / 8), f = performance.now();
    O++, G++, Z[0] = O, e.queue.writeBuffer(h, 0, Z), ee.texture = n.getCurrentTexture();
    const g = e.createCommandEncoder(), w = g.beginComputePass();
    w.setPipeline(i), w.setBindGroup(0, F), w.dispatchWorkgroups(o, _), w.end();
    const C = { width: d.width, height: d.height, depthOrArrayLayers: 1 }, D = { texture: u };
    g.copyTextureToTexture(D, ee, C), e.queue.submit([g.finish()]), f - te >= 1e3 && (K.textContent = `FPS: ${G} | ${(1e3 / G).toFixed(2)}ms | Frame: ${O} | Res: ${d.width}x${d.height}`, G = 0, te = f);
  };
  let A = null, k = null;
  z.addEventListener("click", () => {
    v = !v, z.textContent = v ? "Stop Rendering" : "Resume Rendering";
  }), de.addEventListener("change", (o) => {
    const _ = o.target;
    W(_.value, false);
  }), J.addEventListener("change", async (o) => {
    var _a, _b;
    const _ = o.target, f = (_a = _.files) == null ? void 0 : _a[0];
    if (!f) return;
    console.log(`Reading ${f.name}...`);
    const g = (_b = f.name.split(".").pop()) == null ? void 0 : _b.toLowerCase();
    try {
      if (g === "obj") A = await f.text(), k = "obj";
      else if (g === "glb" || g === "vrm") A = await f.arrayBuffer(), k = "glb";
      else {
        alert("Unsupported file format");
        return;
      }
      de.value = "viewer", W("viewer", false);
    } catch (w) {
      console.error("Failed to load OBJ:", w), alert("Failed to load OBJ file.");
    }
    _.value = "";
  });
  const re = () => {
    if (Q(), l && p && y) {
      l.update_camera(d.width, d.height);
      const o = new Float32Array(p.buffer, l.camera_ptr(), 24);
      e.queue.writeBuffer(y, 0, o);
    }
    V(), N();
  };
  _e.addEventListener("change", re), ge.addEventListener("change", re), Oe.addEventListener("click", () => {
    v = false, s(), V(), N(), v = true, z.textContent = "Stop Rendering";
  }), Q(), W("cornell", false), requestAnimationFrame(ne);
}
Me().catch(console.error);
