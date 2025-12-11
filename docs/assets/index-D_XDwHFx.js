(function() {
  const e = document.createElement("link").relList;
  if (e && e.supports && e.supports("modulepreload")) return;
  for (const r of document.querySelectorAll('link[rel="modulepreload"]')) a(r);
  new MutationObserver((r) => {
    for (const c of r) if (c.type === "childList") for (const s of c.addedNodes) s.tagName === "LINK" && s.rel === "modulepreload" && a(s);
  }).observe(document, { childList: true, subtree: true });
  function n(r) {
    const c = {};
    return r.integrity && (c.integrity = r.integrity), r.referrerPolicy && (c.referrerPolicy = r.referrerPolicy), r.crossOrigin === "use-credentials" ? c.credentials = "include" : r.crossOrigin === "anonymous" ? c.credentials = "omit" : c.credentials = "same-origin", c;
  }
  function a(r) {
    if (r.ep) return;
    r.ep = true;
    const c = n(r);
    fetch(r.href, c);
  }
})();
const ye = `// =========================================================
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
let i;
function xe(t) {
  const e = i.__externref_table_alloc();
  return i.__wbindgen_externrefs.set(e, t), e;
}
function Pe(t, e) {
  return t = t >>> 0, A().subarray(t / 1, t / 1 + e);
}
let y = null;
function se() {
  return (y === null || y.buffer.detached === true || y.buffer.detached === void 0 && y.buffer !== i.memory.buffer) && (y = new DataView(i.memory.buffer)), y;
}
function oe(t, e) {
  return t = t >>> 0, Re(t, e);
}
let U = null;
function A() {
  return (U === null || U.byteLength === 0) && (U = new Uint8Array(i.memory.buffer)), U;
}
function Be(t, e) {
  try {
    return t.apply(this, e);
  } catch (n) {
    const a = xe(n);
    i.__wbindgen_exn_store(a);
  }
}
function Te(t) {
  return t == null;
}
function H(t, e, n) {
  if (n === void 0) {
    const l = z.encode(t), b = e(l.length, 1) >>> 0;
    return A().subarray(b, b + l.length).set(l), I = l.length, b;
  }
  let a = t.length, r = e(a, 1) >>> 0;
  const c = A();
  let s = 0;
  for (; s < a; s++) {
    const l = t.charCodeAt(s);
    if (l > 127) break;
    c[r + s] = l;
  }
  if (s !== a) {
    s !== 0 && (t = t.slice(s)), r = n(r, a, a = s + t.length * 3, 1) >>> 0;
    const l = A().subarray(r + s, r + a), b = z.encodeInto(t, l);
    s += b.written, r = n(r, a, s, 1) >>> 0;
  }
  return I = s, r;
}
let k = new TextDecoder("utf-8", { ignoreBOM: true, fatal: true });
k.decode();
const Se = 2146435072;
let Y = 0;
function Re(t, e) {
  return Y += e, Y >= Se && (k = new TextDecoder("utf-8", { ignoreBOM: true, fatal: true }), k.decode(), Y = e), k.decode(A().subarray(t, t + e));
}
const z = new TextEncoder();
"encodeInto" in z || (z.encodeInto = function(t, e) {
  const n = z.encode(t);
  return e.set(n), { read: t.length, written: n.length };
});
let I = 0;
const ce = typeof FinalizationRegistry > "u" ? { register: () => {
}, unregister: () => {
} } : new FinalizationRegistry((t) => i.__wbg_world_free(t >>> 0, 1));
class X {
  __destroy_into_raw() {
    const e = this.__wbg_ptr;
    return this.__wbg_ptr = 0, ce.unregister(this), e;
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
  update_camera(e, n) {
    i.world_update_camera(this.__wbg_ptr, e, n);
  }
  attributes_len() {
    return i.world_attributes_len(this.__wbg_ptr) >>> 0;
  }
  attributes_ptr() {
    return i.world_attributes_ptr(this.__wbg_ptr) >>> 0;
  }
  constructor(e, n) {
    const a = H(e, i.__wbindgen_malloc, i.__wbindgen_realloc), r = I;
    var c = Te(n) ? 0 : H(n, i.__wbindgen_malloc, i.__wbindgen_realloc), s = I;
    const l = i.world_new(a, r, c, s);
    return this.__wbg_ptr = l >>> 0, ce.register(this, this.__wbg_ptr, this), this;
  }
  bvh_len() {
    return i.world_bvh_len(this.__wbg_ptr) >>> 0;
  }
  bvh_ptr() {
    return i.world_bvh_ptr(this.__wbg_ptr) >>> 0;
  }
}
Symbol.dispose && (X.prototype[Symbol.dispose] = X.prototype.free);
const Ee = /* @__PURE__ */ new Set(["basic", "cors", "default"]);
async function Ue(t, e) {
  if (typeof Response == "function" && t instanceof Response) {
    if (typeof WebAssembly.instantiateStreaming == "function") try {
      return await WebAssembly.instantiateStreaming(t, e);
    } catch (a) {
      if (t.ok && Ee.has(t.type) && t.headers.get("Content-Type") !== "application/wasm") console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", a);
      else throw a;
    }
    const n = await t.arrayBuffer();
    return await WebAssembly.instantiate(n, e);
  } else {
    const n = await WebAssembly.instantiate(t, e);
    return n instanceof WebAssembly.Instance ? { instance: n, module: t } : n;
  }
}
function Ae() {
  const t = {};
  return t.wbg = {}, t.wbg.__wbg___wbindgen_throw_dd24417ed36fc46e = function(e, n) {
    throw new Error(oe(e, n));
  }, t.wbg.__wbg_error_7534b8e9a36f1ab4 = function(e, n) {
    let a, r;
    try {
      a = e, r = n, console.error(oe(e, n));
    } finally {
      i.__wbindgen_free(a, r, 1);
    }
  }, t.wbg.__wbg_getRandomValues_1c61fac11405ffdc = function() {
    return Be(function(e, n) {
      globalThis.crypto.getRandomValues(Pe(e, n));
    }, arguments);
  }, t.wbg.__wbg_new_8a6f238a6ece86ea = function() {
    return new Error();
  }, t.wbg.__wbg_stack_0ed75d68575b0f3c = function(e, n) {
    const a = n.stack, r = H(a, i.__wbindgen_malloc, i.__wbindgen_realloc), c = I;
    se().setInt32(e + 4, c, true), se().setInt32(e + 0, r, true);
  }, t.wbg.__wbindgen_init_externref_table = function() {
    const e = i.__wbindgen_externrefs, n = e.grow(4);
    e.set(0, void 0), e.set(n + 0, void 0), e.set(n + 1, null), e.set(n + 2, true), e.set(n + 3, false);
  }, t;
}
function ze(t, e) {
  return i = t.exports, de.__wbindgen_wasm_module = e, y = null, U = null, i.__wbindgen_start(), i;
}
async function de(t) {
  if (i !== void 0) return i;
  typeof t < "u" && (Object.getPrototypeOf(t) === Object.prototype ? { module_or_path: t } = t : console.warn("using deprecated parameters for the initialization function; pass a single object instead")), typeof t > "u" && (t = new URL("/webgpu-raytracer/assets/rust_shader_tools_bg-C9j29YNZ.wasm", import.meta.url));
  const e = Ae();
  (typeof t == "string" || typeof Request == "function" && t instanceof Request || typeof URL == "function" && t instanceof URL) && (t = fetch(t));
  const { instance: n, module: a } = await Ue(await t, e);
  return ze(n, a);
}
const f = document.getElementById("gpu-canvas"), E = document.getElementById("render-btn"), ue = document.getElementById("scene-select"), le = document.getElementById("res-width"), fe = document.getElementById("res-height"), Ie = document.getElementById("obj-file"), Ge = document.getElementById("max-depth"), Le = document.getElementById("spp-frame"), Oe = document.getElementById("recompile-btn"), j = document.createElement("div");
Object.assign(j.style, { position: "fixed", bottom: "10px", left: "10px", color: "#0f0", background: "rgba(0,0,0,0.7)", padding: "8px", fontFamily: "monospace", fontSize: "14px", pointerEvents: "none", zIndex: "9999", borderRadius: "4px" });
document.body.appendChild(j);
let O = 0, p = false, u = null, m = null, $;
async function ke() {
  if (!navigator.gpu) {
    alert("WebGPU not supported.");
    return;
  }
  const t = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!t) throw new Error("No adapter");
  const e = await t.requestDevice(), n = f.getContext("webgpu");
  if (!n) throw new Error("No context");
  n.configure({ device: e, format: "rgba8unorm", usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT }), m = (await de()).memory, console.log("Wasm initialized");
  let r, c;
  const s = () => {
    const o = parseInt(Ge.value, 10) || 10, d = parseInt(Le.value, 10) || 1;
    console.log(`Recompiling Shader... Depth:${o}, SPP:${d}`);
    let _ = ye;
    _ = _.replace(/const\s+MAX_DEPTH\s*=\s*\d+u;/, `const MAX_DEPTH = ${o}u;`), _ = _.replace(/const\s+SPP\s*=\s*\d+u;/, `const SPP = ${d}u;`);
    const g = e.createShaderModule({ label: "RayTracing", code: _ });
    r = e.createComputePipeline({ label: "Main Pipeline", layout: "auto", compute: { module: g, entryPoint: "main" } }), c = r.getBindGroupLayout(0);
  };
  s();
  let l, b, v, R, h, x, P, B, T, S, C, D = 0;
  const M = () => {
    if (!v) return;
    const o = new Float32Array(D / 4);
    e.queue.writeBuffer(v, 0, o), O = 0;
  }, J = () => {
    let o = parseInt(le.value, 10), d = parseInt(fe.value, 10);
    (isNaN(o) || o < 1) && (o = 720), (isNaN(d) || d < 1) && (d = 480), f.width = o, f.height = d, l && l.destroy(), l = e.createTexture({ size: [f.width, f.height], format: "rgba8unorm", usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC }), b = l.createView(), D = f.width * f.height * 16, v && v.destroy(), v = e.createBuffer({ size: D, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }), R || (R = e.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }));
  }, F = () => {
    !b || !v || !R || !h || !x || !B || !T || !S || !P || (C = e.createBindGroup({ layout: c, entries: [{ binding: 0, resource: b }, { binding: 1, resource: { buffer: v } }, { binding: 2, resource: { buffer: R } }, { binding: 3, resource: { buffer: h } }, { binding: 4, resource: { buffer: x } }, { binding: 5, resource: { buffer: B } }, { binding: 6, resource: { buffer: T } }, { binding: 7, resource: { buffer: S } }, { binding: 8, resource: { buffer: P } }] }));
  }, q = (o, d = true) => {
    console.log(`Loading Scene: ${o}... (Rust)`), p = false, u && u.free();
    const _ = o === "viewer" && $ ? $ : void 0;
    if (console.time("Rust Build"), u = new X(o, _), console.timeEnd("Rust Build"), !m) return;
    const g = u.vertices_ptr(), w = u.vertices_len(), L = new Float32Array(m.buffer, g, w), N = u.normals_ptr(), _e = u.normals_len(), ne = new Float32Array(m.buffer, N, _e), ge = u.indices_ptr(), re = u.indices_len(), V = new Uint32Array(m.buffer, ge, re), me = u.attributes_ptr(), be = u.attributes_len(), W = new Float32Array(m.buffer, me, be), pe = u.bvh_ptr(), ie = u.bvh_len(), ae = new Float32Array(m.buffer, pe, ie);
    console.log(`Scene Stats: Verts:${w / 4}, Tris:${re / 3}, Nodes:${ie / 8}`), x && x.destroy(), x = e.createBuffer({ size: L.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }), e.queue.writeBuffer(x, 0, L), P && P.destroy(), P = e.createBuffer({ size: ne.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }), e.queue.writeBuffer(P, 0, ne), B && B.destroy();
    const ve = Math.max(V.byteLength, 4);
    B = e.createBuffer({ size: ve, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }), V.byteLength > 0 && e.queue.writeBuffer(B, 0, V), T && T.destroy();
    const he = Math.max(W.byteLength, 4);
    T = e.createBuffer({ size: he, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }), W.byteLength > 0 && e.queue.writeBuffer(T, 0, W), S && S.destroy(), S = e.createBuffer({ size: ae.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }), e.queue.writeBuffer(S, 0, ae), h || (h = e.createBuffer({ size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })), u.update_camera(f.width, f.height);
    const we = new Float32Array(m.buffer, u.camera_ptr(), 24);
    e.queue.writeBuffer(h, 0, we), F(), M(), d ? (p = true, E.textContent = "Stop Rendering") : (p = false, E.textContent = "Render Start");
  }, K = new Uint32Array(1), Z = { texture: null };
  let Q = performance.now(), G = 0;
  const ee = () => {
    if (requestAnimationFrame(ee), !p || !C) return;
    const o = Math.ceil(f.width / 8), d = Math.ceil(f.height / 8), _ = performance.now();
    O++, G++, K[0] = O, e.queue.writeBuffer(R, 0, K), Z.texture = n.getCurrentTexture();
    const g = e.createCommandEncoder(), w = g.beginComputePass();
    w.setPipeline(r), w.setBindGroup(0, C), w.dispatchWorkgroups(o, d), w.end();
    const L = { width: f.width, height: f.height, depthOrArrayLayers: 1 }, N = { texture: l };
    g.copyTextureToTexture(N, Z, L), e.queue.submit([g.finish()]), _ - Q >= 1e3 && (j.textContent = `FPS: ${G} | ${(1e3 / G).toFixed(2)}ms | Frame: ${O} | Res: ${f.width}x${f.height}`, G = 0, Q = _);
  };
  E.addEventListener("click", () => {
    p = !p, E.textContent = p ? "Stop Rendering" : "Resume Rendering";
  }), ue.addEventListener("change", (o) => {
    const d = o.target;
    q(d.value, false);
  }), Ie.addEventListener("change", async (o) => {
    var _a;
    const d = o.target, _ = (_a = d.files) == null ? void 0 : _a[0];
    if (_) {
      console.log(`Reading ${_.name}...`);
      try {
        $ = await _.text(), ue.value = "viewer", q("viewer", false);
      } catch (g) {
        console.error("Failed to load OBJ:", g), alert("Failed to load OBJ file.");
      }
      d.value = "";
    }
  });
  const te = () => {
    if (J(), u && m && h) {
      u.update_camera(f.width, f.height);
      const o = new Float32Array(m.buffer, u.camera_ptr(), 24);
      e.queue.writeBuffer(h, 0, o);
    }
    F(), M();
  };
  le.addEventListener("change", te), fe.addEventListener("change", te), Oe.addEventListener("click", () => {
    p = false, s(), F(), M(), p = true, E.textContent = "Stop Rendering";
  }), J(), q("cornell", false), requestAnimationFrame(ee);
}
ke().catch(console.error);
