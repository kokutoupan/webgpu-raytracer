(function() {
  const e = document.createElement("link").relList;
  if (e && e.supports && e.supports("modulepreload")) return;
  for (const r of document.querySelectorAll('link[rel="modulepreload"]')) a(r);
  new MutationObserver((r) => {
    for (const s of r) if (s.type === "childList") for (const o of s.addedNodes) o.tagName === "LINK" && o.rel === "modulepreload" && a(o);
  }).observe(document, { childList: true, subtree: true });
  function n(r) {
    const s = {};
    return r.integrity && (s.integrity = r.integrity), r.referrerPolicy && (s.referrerPolicy = r.referrerPolicy), r.crossOrigin === "use-credentials" ? s.credentials = "include" : r.crossOrigin === "anonymous" ? s.credentials = "omit" : s.credentials = "same-origin", s;
  }
  function a(r) {
    if (r.ep) return;
    r.ep = true;
    const s = n(r);
    fetch(r.href, s);
  }
})();
const ce = `// =========================================================
//   WebGPU Ray Tracer (Final Refactored Version)
// =========================================================

// --- Constants ---
const PI = 3.141592653589793;
const T_MIN = 0.001;
const T_MAX = 1e30;
const MAX_DEPTH = 10u; // \u30AC\u30E9\u30B9\u306A\u3069\u3092\u7DBA\u9E97\u306B\u898B\u305B\u308B\u305F\u3081\u5C11\u3057\u5897\u3084\u3059
const SPP = 1u;

// Binding Stride
const SPHERES_STRIDE = 3u;
const TRIANGLES_STRIDE = 4u;

// --- Bindings ---
@group(0) @binding(0) var outputTex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<storage, read_write> accumulateBuffer: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> frame: FrameInfo;
@group(0) @binding(3) var<uniform> camera: Camera;

// Scene Data
// \u7D71\u5408\u30D7\u30EA\u30DF\u30C6\u30A3\u30D6\u30D0\u30C3\u30D5\u30A1 (Binding 4)
// vec4 x 4 = 64 bytes stride
@group(0) @binding(4) var<storage, read> scene_primitives: array<UnifiedPrimitive>;

// Binding 5: BVH Nodes
@group(0) @binding(5) var<storage, read> bvh_nodes: array<BVHNode>;

// --- Structs ---
struct UnifiedPrimitive {
    data0: vec4<f32>, // [Tri:V0+Extra] [Sph:Center+Radius]
    data1: vec4<f32>, // [Tri:V1+Mat]   [Sph:Unused+Mat]
    data2: vec4<f32>, // [Tri:V2+ObjType] [Sph:Unused+ObjType]
    data3: vec4<f32>, // [Tri:Col+Unused] [Sph:Col+Extra]
}
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
    left_first: f32,
    max_b: vec3<f32>,
    tri_count: f32
}

// --- RNG Helpers ---
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
fn hit_sphere_raw(center: vec3<f32>, radius: f32, r: Ray, t_min: f32, t_max: f32) -> f32 {
    let oc = r.origin - center;
    let a = dot(r.direction, r.direction);
    let h = dot(r.direction, oc);
    let c = dot(oc, oc) - radius * radius;
    let disc = h * h - a * c;
    if disc < 0.0 { return -1.0; }
    let sqrtd = sqrt(disc);
    var root = (-h - sqrtd) / a;
    if root <= t_min || t_max <= root {
        root = (-h + sqrtd) / a;
        if root <= t_min || t_max <= root { return -1.0; }
    }
    return root;
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

fn intersect_aabb(min_b: vec3<f32>, max_b: vec3<f32>, r: Ray, inv_d: vec3<f32>, t_min: f32, t_max: f32) -> f32 {
    let t0s = (min_b - r.origin) * inv_d;
    let t1s = (max_b - r.origin) * inv_d;
    let t_small = min(t0s, t1s);
    let t_big = max(t0s, t1s);
    let tmin = max(t_min, max(t_small.x, max(t_small.y, t_small.z)));
    let tmax = min(t_max, min(t_big.x, min(t_big.y, t_big.z)));
    return select(1e30, tmin, tmin <= tmax);
}

// --- BVH Traversal ---
fn hit_bvh(r: Ray, t_min: f32, t_max: f32) -> vec4<f32> {
    var closest_t = t_max;
    var hit_idx = -1.0;
    var hit_type = 0.0;

    //[DEBUG] \u30B3\u30B9\u30C8\u30AB\u30A6\u30F3\u30C8
    //var steps = 0.0;


    let inv_d = 1.0 / r.direction;

    var stack: array<u32, 32>;
    var stackptr = 0u;


    // \u30EB\u30FC\u30C8\u30CE\u30FC\u30C9(0)\u306E\u5224\u5B9A
    let root_dist = intersect_aabb(bvh_nodes[0].min_b, bvh_nodes[0].max_b, r, inv_d, t_min, closest_t);

    // \u30D2\u30C3\u30C8\u3057\u305F\u6642\u3060\u3051\u7A4D\u3080
    if root_dist < 1e30 {
        stack[stackptr] = 0u;
        stackptr++;
    }

    while stackptr > 0u {
        stackptr--;
        let node_idx = stack[stackptr];
        let node = bvh_nodes[node_idx];

        //steps += 1.0;

        let count = u32(node.tri_count);
        let first = u32(node.left_first);

        if count > 0u {
            // Leaf Node
           // steps += f32(count);

            for (var i = 0u; i < count; i++) {
                let idx = first + i;
                let prim = scene_primitives[idx];
                let obj_type = prim.data2.w;
                var t = -1.0;

                if obj_type < 1.5 { // Sphere
                    t = hit_sphere_raw(prim.data0.xyz, prim.data0.w, r, t_min, closest_t);
                } else { // Triangle
                    t = hit_triangle_raw(prim.data0.xyz, prim.data1.xyz, prim.data2.xyz, r, t_min, closest_t);
                }

                if t > 0.0 {
                    closest_t = t;
                    hit_idx = f32(idx);
                    hit_type = obj_type;
                }
            }
        } else {
            // Internal Node (Front-to-Back)
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
    //return vec4<f32>(closest_t, hit_idx, hit_type, steps);
    return vec4<f32>(closest_t, hit_idx, hit_type, 0.0);
}

// --- Main Tracer ---
fn ray_color(r_in: Ray, rng: ptr<function, u32>) -> vec3<f32> {
    var ray = r_in;
    var throughput = vec3<f32>(1.0);

    for (var depth = 0u; depth < MAX_DEPTH; depth++) {
        // BVH Traversal
        let hit = hit_bvh(ray, T_MIN, T_MAX);
        let t = hit.x;
        let idx = u32(hit.y);
        let type_id = u32(hit.z);
//        let cost = hit.w; // \u2605\u30B3\u30B9\u30C8\u53D6\u5F97
//
//        // --- \u30D2\u30FC\u30C8\u30DE\u30C3\u30D7\u8868\u793A\u30E2\u30FC\u30C9 ---
//        // \u30B3\u30B9\u30C8 0\u301C100 \u3092 \u9752\u301C\u8D64 \u306B\u30DE\u30C3\u30D4\u30F3\u30B0\u3059\u308B\u7C21\u6613\u8868\u793A
//        // 100\u56DE\u4EE5\u4E0A\u306E\u5224\u5B9A\u306F\u771F\u3063\u8D64\u306B\u306A\u308B
//        let heat = cost / 100.0; 
//    
//        // \u8679\u8272\u30DE\u30C3\u30D7\u3063\u307D\u3044\u30B0\u30E9\u30C7\u30FC\u30B7\u30E7\u30F3
//        let r = smoothstep(0.5, 1.0, heat);
//        let g = sin(heat * PI);
//        let b = smoothstep(0.5, 0.0, heat);
//
//        return vec3<f32>(r, g, b);

        if type_id == 0u {
            // Miss: Black Background
            return vec3<f32>(0.);
        }

        // Hit Data Unpacking
        let prim = scene_primitives[idx];
        var p = ray.origin + t * ray.direction;
        var normal = vec3<f32>(0.0);
        var mat = 0.0;
        var col = vec3<f32>(0.0);
        var ext = 0.0;
        var front_face = true;

        if type_id == 1u { // Sphere
            let center = prim.data0.xyz;
            let radius = prim.data0.w;
            let outward_n = (p - center) / radius;
            front_face = dot(ray.direction, outward_n) < 0.0;
            normal = select(-outward_n, outward_n, front_face);

            mat = prim.data1.w; // Data1.w
            col = prim.data3.xyz; // Data3.xyz
            ext = prim.data3.w;   // Data3.w
        } else { // Triangle
            let v0 = prim.data0.xyz;
            let v1 = prim.data1.xyz;
            let v2 = prim.data2.xyz;
            let e1 = v1 - v0;
            let e2 = v2 - v0;
            let outward_n = normalize(cross(e1, e2));
            front_face = dot(ray.direction, outward_n) < 0.0;
            normal = select(-outward_n, outward_n, front_face);

            mat = prim.data1.w;   // Data1.w
            col = prim.data3.xyz; // Data3.xyz
            ext = prim.data3.w;   // Data0.w (Triangle extra is stored here)
        }

        // Emission
        if mat > 2.5 { return select(vec3<f32>(0.0), throughput * col, front_face || ext > 0.5); }

        var scat = vec3<f32>(0.0);

        // Material Scatter
        if mat < 0.5 { // Lambertian
            scat = normal + random_unit_vector(rng);
            if length(scat) < 1e-6 { scat = normal; }
        
        } else if mat < 1.5 { // Metal
            scat = reflect(ray.direction, normal) + ext * random_unit_vector(rng);
            // \u2605\u91CD\u8981: \u5185\u5074\u3078\u306E\u53CD\u5C04\u306F\u5438\u53CE\uFF08\u9ED2\uFF09
            if dot(scat, normal) <= 0.0 { return vec3<f32>(0.0); }
        
        } else { // Dielectric
            let ratio = select(ext, 1.0 / ext, front_face);
            let unit = normalize(ray.direction);
            let cos_t = min(dot(-unit, normal), 1.0);
            let sin_t = sqrt(1.0 - cos_t * cos_t);
            let cannot = ratio * sin_t > 1.0;
            if cannot || reflectance(cos_t, ratio) > rand_pcg(rng) {
                scat = reflect(unit, normal);
            } else {
                // Refract manual
                let perp = ratio * (unit + cos_t * normal);
                let para = -sqrt(abs(1.0 - dot(perp, perp))) * normal;
                scat = perp + para;
            }
        }

        // \u2605\u91CD\u8981: Shadow Acne \u5BFE\u7B56 (\u5C11\u3057\u6D6E\u304B\u305B\u3066\u518D\u767A\u5C04)
        let next_origin = p + scat * 1e-4;
        ray = Ray(next_origin, scat);
        throughput *= col;

        // Russian Roulette
        if depth > 2u {
            let p_rr = max(throughput.r, max(throughput.g, throughput.b));
            if rand_pcg(rng) > p_rr { break; }
            throughput /= p_rr;
        }
    }
    return vec3<f32>(0.);
}

// --- Entry Point ---
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(outputTex);
    if id.x >= dims.x || id.y >= dims.y { return; }

    let pixel_idx = id.y * dims.x + id.x;
    var rng = init_rng(pixel_idx, frame.frame_count);

    // \u2605\u5909\u66F4: SPP\u30EB\u30FC\u30D7\u3067\u5E73\u5747\u8272\u3092\u8A08\u7B97
    var accum_color = vec3<f32>(0.0);

    for (var s = 0u; s < SPP; s++) {
        // \u30EC\u30A4\u751F\u6210 (Jittering) \u3092\u30EB\u30FC\u30D7\u5185\u3067\u884C\u3046\u3053\u3068\u3067AA\u52B9\u679C\u3092\u5F97\u308B
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
    
    // \u3053\u306E\u30D5\u30EC\u30FC\u30E0\u3067\u306E\u5E73\u5747\u8272
    let col = accum_color / f32(SPP);

    // Accumulate
    var acc = vec4<f32>(0.0);
    if frame.frame_count > 1u { acc = accumulateBuffer[pixel_idx]; }
    let new_acc = acc + vec4<f32>(col, 1.0);
    accumulateBuffer[pixel_idx] = new_acc;

    // Display
    var final_col = new_acc.rgb / f32(frame.frame_count);
    final_col = sqrt(clamp(final_col, vec3<f32>(0.0), vec3<f32>(1.0)));
    textureStore(outputTex, vec2<i32>(id.xy), vec4<f32>(final_col, 1.0));
}
`;
let i;
function le(t) {
  const e = i.__externref_table_alloc();
  return i.__wbindgen_externrefs.set(e, t), e;
}
function fe(t, e) {
  return t = t >>> 0, R().subarray(t / 1, t / 1 + e);
}
let h = null;
function Z() {
  return (h === null || h.buffer.detached === true || h.buffer.detached === void 0 && h.buffer !== i.memory.buffer) && (h = new DataView(i.memory.buffer)), h;
}
function Q(t, e) {
  return t = t >>> 0, me(t, e);
}
let B = null;
function R() {
  return (B === null || B.byteLength === 0) && (B = new Uint8Array(i.memory.buffer)), B;
}
function ue(t, e) {
  try {
    return t.apply(this, e);
  } catch (n) {
    const a = le(n);
    i.__wbindgen_exn_store(a);
  }
}
function de(t) {
  return t == null;
}
function N(t, e, n) {
  if (n === void 0) {
    const l = E.encode(t), m = e(l.length, 1) >>> 0;
    return R().subarray(m, m + l.length).set(l), U = l.length, m;
  }
  let a = t.length, r = e(a, 1) >>> 0;
  const s = R();
  let o = 0;
  for (; o < a; o++) {
    const l = t.charCodeAt(o);
    if (l > 127) break;
    s[r + o] = l;
  }
  if (o !== a) {
    o !== 0 && (t = t.slice(o)), r = n(r, a, a = o + t.length * 3, 1) >>> 0;
    const l = R().subarray(r + o, r + a), m = E.encodeInto(t, l);
    o += m.written, r = n(r, a, o, 1) >>> 0;
  }
  return U = o, r;
}
let z = new TextDecoder("utf-8", { ignoreBOM: true, fatal: true });
z.decode();
const _e = 2146435072;
let F = 0;
function me(t, e) {
  return F += e, F >= _e && (z = new TextDecoder("utf-8", { ignoreBOM: true, fatal: true }), z.decode(), F = e), z.decode(R().subarray(t, t + e));
}
const E = new TextEncoder();
"encodeInto" in E || (E.encodeInto = function(t, e) {
  const n = E.encode(t);
  return e.set(n), { read: t.length, written: n.length };
});
let U = 0;
const ee = typeof FinalizationRegistry > "u" ? { register: () => {
}, unregister: () => {
} } : new FinalizationRegistry((t) => i.__wbg_world_free(t >>> 0, 1));
class V {
  __destroy_into_raw() {
    const e = this.__wbg_ptr;
    return this.__wbg_ptr = 0, ee.unregister(this), e;
  }
  free() {
    const e = this.__destroy_into_raw();
    i.__wbg_world_free(e, 0);
  }
  camera_ptr() {
    return i.world_camera_ptr(this.__wbg_ptr) >>> 0;
  }
  update_camera(e, n) {
    i.world_update_camera(this.__wbg_ptr, e, n);
  }
  constructor(e, n) {
    const a = N(e, i.__wbindgen_malloc, i.__wbindgen_realloc), r = U;
    var s = de(n) ? 0 : N(n, i.__wbindgen_malloc, i.__wbindgen_realloc), o = U;
    const l = i.world_new(a, r, s, o);
    return this.__wbg_ptr = l >>> 0, ee.register(this, this.__wbg_ptr, this), this;
  }
  bvh_len() {
    return i.world_bvh_len(this.__wbg_ptr) >>> 0;
  }
  bvh_ptr() {
    return i.world_bvh_ptr(this.__wbg_ptr) >>> 0;
  }
  prim_len() {
    return i.world_prim_len(this.__wbg_ptr) >>> 0;
  }
  prim_ptr() {
    return i.world_prim_ptr(this.__wbg_ptr) >>> 0;
  }
}
Symbol.dispose && (V.prototype[Symbol.dispose] = V.prototype.free);
const pe = /* @__PURE__ */ new Set(["basic", "cors", "default"]);
async function ge(t, e) {
  if (typeof Response == "function" && t instanceof Response) {
    if (typeof WebAssembly.instantiateStreaming == "function") try {
      return await WebAssembly.instantiateStreaming(t, e);
    } catch (a) {
      if (t.ok && pe.has(t.type) && t.headers.get("Content-Type") !== "application/wasm") console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", a);
      else throw a;
    }
    const n = await t.arrayBuffer();
    return await WebAssembly.instantiate(n, e);
  } else {
    const n = await WebAssembly.instantiate(t, e);
    return n instanceof WebAssembly.Instance ? { instance: n, module: t } : n;
  }
}
function be() {
  const t = {};
  return t.wbg = {}, t.wbg.__wbg___wbindgen_throw_dd24417ed36fc46e = function(e, n) {
    throw new Error(Q(e, n));
  }, t.wbg.__wbg_error_7534b8e9a36f1ab4 = function(e, n) {
    let a, r;
    try {
      a = e, r = n, console.error(Q(e, n));
    } finally {
      i.__wbindgen_free(a, r, 1);
    }
  }, t.wbg.__wbg_getRandomValues_1c61fac11405ffdc = function() {
    return ue(function(e, n) {
      globalThis.crypto.getRandomValues(fe(e, n));
    }, arguments);
  }, t.wbg.__wbg_new_8a6f238a6ece86ea = function() {
    return new Error();
  }, t.wbg.__wbg_stack_0ed75d68575b0f3c = function(e, n) {
    const a = n.stack, r = N(a, i.__wbindgen_malloc, i.__wbindgen_realloc), s = U;
    Z().setInt32(e + 4, s, true), Z().setInt32(e + 0, r, true);
  }, t.wbg.__wbindgen_init_externref_table = function() {
    const e = i.__wbindgen_externrefs, n = e.grow(4);
    e.set(0, void 0), e.set(n + 0, void 0), e.set(n + 1, null), e.set(n + 2, true), e.set(n + 3, false);
  }, t;
}
function he(t, e) {
  return i = t.exports, ie.__wbindgen_wasm_module = e, h = null, B = null, i.__wbindgen_start(), i;
}
async function ie(t) {
  if (i !== void 0) return i;
  typeof t < "u" && (Object.getPrototypeOf(t) === Object.prototype ? { module_or_path: t } = t : console.warn("using deprecated parameters for the initialization function; pass a single object instead")), typeof t > "u" && (t = new URL("/webgpu-raytracer/assets/rust_shader_tools_bg-CZfcX3KL.wasm", import.meta.url));
  const e = be();
  (typeof t == "string" || typeof Request == "function" && t instanceof Request || typeof URL == "function" && t instanceof URL) && (t = fetch(t));
  const { instance: n, module: a } = await ge(await t, e);
  return he(n, a);
}
const d = document.getElementById("gpu-canvas"), T = document.getElementById("render-btn"), te = document.getElementById("scene-select"), ne = document.getElementById("res-width"), re = document.getElementById("res-height"), ve = document.getElementById("obj-file"), we = document.getElementById("max-depth"), ye = document.getElementById("spp-frame"), xe = document.getElementById("recompile-btn"), W = document.createElement("div");
Object.assign(W.style, { position: "fixed", bottom: "10px", left: "10px", color: "#0f0", background: "rgba(0,0,0,0.7)", padding: "8px", fontFamily: "monospace", fontSize: "14px", pointerEvents: "none", zIndex: "9999", borderRadius: "4px" });
document.body.appendChild(W);
let I = 0, p = false, g = null, S = null, q = null;
async function Pe() {
  if (!navigator.gpu) {
    alert("WebGPU not supported.");
    return;
  }
  const t = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!t) throw new Error("No adapter");
  const e = await t.requestDevice(), n = d.getContext("webgpu");
  if (!n) throw new Error("No context");
  n.configure({ device: e, format: "rgba8unorm", usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT }), S = (await ie()).memory, console.log("Wasm initialized");
  let r, s;
  const o = () => {
    const c = parseInt(we.value, 10) || 10, f = parseInt(ye.value, 10) || 1;
    console.log(`Recompiling Shader... Depth:${c}, SPP:${f}`);
    let u = ce;
    u = u.replace(/const\s+MAX_DEPTH\s*=\s*\d+u;/, `const MAX_DEPTH = ${c}u;`), u = u.replace(/const\s+SPP\s*=\s*\d+u;/, `const SPP = ${f}u;`);
    const _ = e.createShaderModule({ label: "RayTracing", code: u });
    r = e.createComputePipeline({ label: "Main Pipeline", layout: "auto", compute: { module: _, entryPoint: "main" } }), s = r.getBindGroupLayout(0);
  };
  o();
  let l, m, b, x, P, v, w, k, C = 0;
  const O = () => {
    if (!b) return;
    const c = new Float32Array(C / 4);
    e.queue.writeBuffer(b, 0, c), I = 0;
  }, H = () => {
    let c = parseInt(ne.value, 10), f = parseInt(re.value, 10);
    (isNaN(c) || c < 1) && (c = 720), (isNaN(f) || f < 1) && (f = 480), d.width = c, d.height = f, l && l.destroy(), l = e.createTexture({ size: [d.width, d.height], format: "rgba8unorm", usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC }), m = l.createView(), C = d.width * d.height * 16, b && b.destroy(), b = e.createBuffer({ size: C, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }), x || (x = e.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }));
  }, M = () => {
    !m || !b || !x || !P || !v || !w || (k = e.createBindGroup({ layout: s, entries: [{ binding: 0, resource: m }, { binding: 1, resource: { buffer: b } }, { binding: 2, resource: { buffer: x } }, { binding: 3, resource: { buffer: P } }, { binding: 4, resource: { buffer: v } }, { binding: 5, resource: { buffer: w } }] }));
  }, G = (c, f = true) => {
    console.log(`Loading Scene: ${c}... (Rust)`), p = false, g && g.free();
    const u = c === "viewer" && q ? q : void 0;
    if (console.time("Rust Build"), g = new V(c, u), console.timeEnd("Rust Build"), !S) return;
    const _ = g.bvh_ptr(), y = g.bvh_len(), D = new Float32Array(S.buffer, _, y), L = g.prim_ptr(), ae = g.prim_len(), K = new Float32Array(S.buffer, L, ae), oe = g.camera_ptr(), se = new Float32Array(S.buffer, oe, 24);
    v && v.destroy(), v = e.createBuffer({ size: K.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }), e.queue.writeBuffer(v, 0, K), w && w.destroy(), w = e.createBuffer({ size: D.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }), e.queue.writeBuffer(w, 0, D), P || (P = e.createBuffer({ size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })), e.queue.writeBuffer(P, 0, se), M(), O(), f ? (p = true, T.textContent = "Stop Rendering") : (p = false, T.textContent = "Render Start");
  }, $ = new Uint32Array(1), j = { texture: null };
  let X = performance.now(), A = 0;
  const Y = () => {
    if (requestAnimationFrame(Y), !p || !k) return;
    const c = Math.ceil(d.width / 8), f = Math.ceil(d.height / 8), u = performance.now();
    I++, A++, $[0] = I, e.queue.writeBuffer(x, 0, $), j.texture = n.getCurrentTexture();
    const _ = e.createCommandEncoder(), y = _.beginComputePass();
    y.setPipeline(r), y.setBindGroup(0, k), y.dispatchWorkgroups(c, f), y.end();
    const D = { width: d.width, height: d.height, depthOrArrayLayers: 1 }, L = { texture: l };
    _.copyTextureToTexture(L, j, D), e.queue.submit([_.finish()]), u - X >= 1e3 && (W.textContent = `FPS: ${A} | ${(1e3 / A).toFixed(2)}ms | Frame: ${I} | Res: ${d.width}x${d.height}`, A = 0, X = u);
  };
  T.addEventListener("click", () => {
    p = !p, T.textContent = p ? "Stop Rendering" : "Resume Rendering";
  }), te.addEventListener("change", (c) => {
    const f = c.target;
    G(f.value, false);
  }), ve.addEventListener("change", async (c) => {
    var _a;
    const f = c.target, u = (_a = f.files) == null ? void 0 : _a[0];
    if (u) {
      console.log(`Reading ${u.name}...`);
      try {
        q = await u.text(), te.value = "viewer", G("viewer", false);
      } catch (_) {
        console.error("Failed to load OBJ:", _), alert("Failed to load OBJ file.");
      }
      f.value = "";
    }
  });
  const J = () => {
    H(), M(), O();
  };
  ne.addEventListener("change", J), re.addEventListener("change", J), xe.addEventListener("click", () => {
    p = false, o(), M(), O(), p = true, T.textContent = "Stop Rendering";
  }), H(), G("cornell", false), requestAnimationFrame(Y);
}
Pe().catch(console.error);
