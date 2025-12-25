var __defProp = Object.defineProperty;
var __defNormalProp = (obj, key, value) => key in obj ? __defProp(obj, key, { enumerable: true, configurable: true, writable: true, value }) : obj[key] = value;
var __publicField = (obj, key, value) => __defNormalProp(obj, typeof key !== "symbol" ? key + "" : key, value);
(async () => {
  (function() {
    const e = document.createElement("link").relList;
    if (e && e.supports && e.supports("modulepreload")) return;
    for (const r of document.querySelectorAll('link[rel="modulepreload"]')) n(r);
    new MutationObserver((r) => {
      for (const s of r) if (s.type === "childList") for (const o of s.addedNodes) o.tagName === "LINK" && o.rel === "modulepreload" && n(o);
    }).observe(document, {
      childList: true,
      subtree: true
    });
    function t(r) {
      const s = {};
      return r.integrity && (s.integrity = r.integrity), r.referrerPolicy && (s.referrerPolicy = r.referrerPolicy), r.crossOrigin === "use-credentials" ? s.credentials = "include" : r.crossOrigin === "anonymous" ? s.credentials = "omit" : s.credentials = "same-origin", s;
    }
    function n(r) {
      if (r.ep) return;
      r.ep = true;
      const s = t(r);
      fetch(r.href, s);
    }
  })();
  const X = "modulepreload", K = function(i) {
    return "/webgpu-raytracer/" + i;
  }, $ = {}, z = function(e, t, n) {
    let r = Promise.resolve();
    if (t && t.length > 0) {
      let h = function(u) {
        return Promise.all(u.map((g) => Promise.resolve(g).then((b) => ({
          status: "fulfilled",
          value: b
        }), (b) => ({
          status: "rejected",
          reason: b
        }))));
      };
      var o = h;
      document.getElementsByTagName("link");
      const l = document.querySelector("meta[property=csp-nonce]"), c = (l == null ? void 0 : l.nonce) || (l == null ? void 0 : l.getAttribute("nonce"));
      r = h(t.map((u) => {
        if (u = K(u), u in $) return;
        $[u] = true;
        const g = u.endsWith(".css"), b = g ? '[rel="stylesheet"]' : "";
        if (document.querySelector(`link[href="${u}"]${b}`)) return;
        const v = document.createElement("link");
        if (v.rel = g ? "stylesheet" : X, g || (v.as = "script"), v.crossOrigin = "", v.href = u, c && v.setAttribute("nonce", c), document.head.appendChild(v), g) return new Promise((J, Y) => {
          v.addEventListener("load", J), v.addEventListener("error", () => Y(new Error(`Unable to preload CSS for ${u}`)));
        });
      }));
    }
    function s(l) {
      const c = new Event("vite:preloadError", {
        cancelable: true
      });
      if (c.payload = l, window.dispatchEvent(c), !c.defaultPrevented) throw l;
    }
    return r.then((l) => {
      for (const c of l || []) c.status === "rejected" && s(c.reason);
      return e().catch(s);
    });
  }, Q = `// =========================================================
//   WebGPU Ray Tracer (Raytracer.wgsl)
// =========================================================

const PI = 3.141592653589793;
const T_MIN = 0.001;
const T_MAX = 1e30;
// These are replaced by TypeScript before compilation
const MAX_DEPTH = 10u;
const SPP = 1u;

// =========================================================
//   Structs
// =========================================================

struct Camera {
    origin: vec4<f32>, // w: lens_radius
    lower_left_corner: vec4<f32>,
    horizontal: vec4<f32>,
    vertical: vec4<f32>,
    u: vec4<f32>,
    v: vec4<f32>
}

struct SceneUniforms {
    camera: Camera,
    prev_camera: Camera,
    frame_count: u32,
    blas_base_idx: u32,
    vertex_count: u32,
    rand_seed: u32,
    light_count: u32,
    width: u32,
    height: u32,
    pad: u32,
    jitter: vec2<f32>,
    prev_jitter: vec2<f32>
}

struct MeshTopology {
    v0: u32,
    v1: u32,
    v2: u32,
    pad: u32,
    data0: vec4<f32>, // rgb: BaseColor, w: MaterialType (bitcast)
    data1: vec4<f32>, // x: Metallic, y: Roughness, z: IOR, w: 0.0
    data2: vec4<f32>, // x: BaseTex, y: MetRoughTex, z: NormalTex, w: EmissiveTex
    data3: vec4<f32>  // rgb: EmissiveColor, w: OcclusionTex
}

struct PackedLight {
    position: vec3<f32>,
    area: f32,
    emission: vec3<f32>,
    pad: f32,
    u_edge: vec3<f32>,
    pad_u: f32,
    v_edge: vec3<f32>,
    pad_v: f32,
    normal: vec3<f32>,
    pad_n: f32
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

struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>
}

struct HitResult {
    t: f32,
    tri_idx: f32,
    inst_idx: i32
}

struct ONB {
    u: vec3<f32>,
    v: vec3<f32>,
    w: vec3<f32>,
}

struct LightSample {
    L: vec3<f32>,       // Radiance
    dir: vec3<f32>,     // Direction to light
    dist: f32,          // Distance to light
    pdf: f32,           // PDF of sampling this point
}

struct ScatterResult {
    dir: vec3<f32>,
    pdf: f32,
    throughput: vec3<f32>,
    is_specular: bool
}

// ReSTIR\u7528\u306E\u30EA\u30B6\u30FC\u30D0 (\u5019\u88DC\u3092\u4FDD\u6301\u3059\u308B\u7BB1)
struct Reservoir {
    w_sum: f32,      // \u91CD\u307F\u306E\u5408\u8A08
    M: f32,          // \u898B\u3066\u304D\u305F\u5019\u88DC\u6570
    W: f32,          // \u6700\u7D42\u7684\u306AUnbiased\u30A6\u30A7\u30A4\u30C8
    light_idx: u32,  // \u52DD\u3061\u6B8B\u3063\u305F\u30E9\u30A4\u30C8\u306EID
    // \u9078\u3070\u308C\u305F\u30E9\u30A4\u30C8\u4E0A\u306E\u30B5\u30F3\u30D7\u30EA\u30F3\u30B0\u4F4D\u7F6E\u3092\u518D\u73FE\u3059\u308B\u305F\u3081\u306E\u4E71\u6570
    r_u: f32,
    r_v: f32
}

struct ReservoirData {
    w_sum: f32,
    M: f32,     // \u5019\u88DC\u6570 (u32\u3067\u3082\u3044\u3044\u304C\u8A08\u7B97\u4E0Af32\u304C\u697D)
    W: f32,     // Unbiased Weight
    light_idx: u32,
    r_u: f32,   // \u30B5\u30F3\u30D7\u30EA\u30F3\u30B0\u4E71\u6570\u5FA9\u5143\u7528
    r_v: f32,
    pad1: f32,  // \u30A2\u30E9\u30A4\u30E1\u30F3\u30C8\u7528
    pad2: f32
}

// \u7C21\u6613\u7684\u306A\u30E9\u30A4\u30C8\u60C5\u5831\u306E\u5165\u308C\u7269\uFF08\u30B7\u30E3\u30C9\u30A6\u30EC\u30A4\u524D\uFF09
struct LightCandidate {
    L: vec3<f32>,       // \u767A\u5149\u5F37\u5EA6
    pos: vec3<f32>,     // \u30E9\u30A4\u30C8\u4E0A\u306E\u70B9\u306E\u4F4D\u7F6E
    normal: vec3<f32>,  // \u30E9\u30A4\u30C8\u4E0A\u306E\u70B9\u306E\u6CD5\u7DDA
    pdf: f32            // \u9078\u3070\u308C\u308B\u78BA\u7387
}

// =========================================================
//   Bindings
// =========================================================

@group(0) @binding(1) var<storage, read_write> accumulateBuffer: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> scene: SceneUniforms;

@group(0) @binding(3) var<storage, read> geometry: array<f32>; // [Pos(4)... | Norm(4)... | UV(2)...]
@group(0) @binding(4) var<storage, read> topology: array<MeshTopology>;
@group(0) @binding(5) var<storage, read> nodes: array<BVHNode>; 
@group(0) @binding(6) var<storage, read> instances: array<Instance>;
@group(0) @binding(9) var<storage, read> lights: array<PackedLight>;

@group(0) @binding(7) var tex: texture_2d_array<f32>;
@group(0) @binding(8) var smp: sampler;

@group(0) @binding(10) var<storage, read_write> reservoirs: array<ReservoirData>;

// =========================================================
//   Buffer Accessors
// =========================================================

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

fn get_transform(inst: Instance) -> mat4x4<f32> {
    return mat4x4<f32>(inst.transform_0, inst.transform_1, inst.transform_2, inst.transform_3);
}

fn get_inv_transform(inst: Instance) -> mat4x4<f32> {
    return mat4x4<f32>(inst.inv_0, inst.inv_1, inst.inv_2, inst.inv_3);
}

// =========================================================
//   Math & RNG Helpers
// =========================================================

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

fn random_unit_vector(onb: ONB, rng: ptr<function, u32>) -> vec3<f32> {
    let r1 = rand_pcg(rng);
    let r2 = rand_pcg(rng);
    let phi = 2.0 * PI * r1;
    let cos_theta = sqrt(1.0 - r2);
    let sin_theta = sqrt(r2);
    let local_dir = vec3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
    return local_to_world(onb, local_dir);
}

fn random_in_unit_disk(rng: ptr<function, u32>) -> vec3<f32> {
    let r = sqrt(rand_pcg(rng));
    let theta = 2.0 * PI * rand_pcg(rng);
    return vec3<f32>(r * cos(theta), r * sin(theta), 0.0);
}

fn build_onb(n: vec3<f32>) -> ONB {
    let sign = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (sign + n.z);
    let b = n.x * n.y * a;
    let u = vec3(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x);
    let v = vec3(b, sign + n.y * n.y * a, -n.y);
    return ONB(u, v, n);
}

fn local_to_world(onb: ONB, a: vec3<f32>) -> vec3<f32> {
    return a.x * onb.u + a.y * onb.v + a.z * onb.w;
}

// =========================================================
//   BSDF Functions
// =========================================================

fn eval_diffuse(albedo: vec3<f32>) -> vec3<f32> {
    return albedo / PI;
}

fn sample_diffuse(normal: vec3<f32>, albedo: vec3<f32>, rng: ptr<function, u32>) -> ScatterResult {
    let onb = build_onb(normal);
    let dir = random_unit_vector(onb, rng);
    let cos_theta = max(dot(normal, dir), 0.0);
    return ScatterResult(dir, cos_theta / PI, albedo, false);
}

// GGX
fn ggx_d(n_dot_h: f32, a2: f32) -> f32 {
    let d = (n_dot_h * a2 - n_dot_h) * n_dot_h + 1.0;
    return a2 / (PI * d * d);
}

fn ggx_g(n_dot_v: f32, n_dot_l: f32, a2: f32) -> f32 {
    let g1_v = 2.0 * n_dot_v / (n_dot_v + sqrt(a2 + (1.0 - a2) * n_dot_v * n_dot_v));
    let g1_l = 2.0 * n_dot_l / (n_dot_l + sqrt(a2 + (1.0 - a2) * n_dot_l * n_dot_l));
    return g1_v * g1_l;
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

fn eval_ggx(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, roughness: f32, f0: vec3<f32>) -> vec3<f32> {
    let h = normalize(v + l);
    let n_dot_v = max(dot(n, v), 1e-4);
    let n_dot_l = max(dot(n, l), 1e-4);
    let n_dot_h = max(dot(n, h), 1e-4);
    let v_dot_h = max(dot(v, h), 1e-4);

    let a2 = roughness * roughness;
    let d = ggx_d(n_dot_h, a2);
    let g = ggx_g(n_dot_v, n_dot_l, a2);
    let f = fresnel_schlick(v_dot_h, f0);

    return (d * g * f) / (4.0 * n_dot_v * n_dot_l);
}

fn sample_ggx(n: vec3<f32>, v: vec3<f32>, roughness: f32, f0: vec3<f32>, rng: ptr<function, u32>) -> ScatterResult {
    let a = roughness;
    let u = vec2(rand_pcg(rng), rand_pcg(rng));

    let phi = 2.0 * PI * u.x;
    let cos_theta = sqrt(max(0.0, (1.0 - u.y) / (1.0 + (a * a - 1.0) * u.y)));
    let sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));

    let h_local = vec3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
    let onb = build_onb(n);
    let h = local_to_world(onb, h_local);
    let l = reflect(-v, h);

    if dot(n, l) <= 0.0 {
        return ScatterResult(vec3(0.0), 0.0, vec3(0.0), false);
    }

    let n_dot_v = max(dot(n, v), 1e-4);
    let n_dot_l = max(dot(n, l), 1e-4);
    let n_dot_h = max(dot(n, h), 1e-4);
    let v_dot_h = max(dot(v, h), 1e-4);

    let a2 = a * a;
    let d = ggx_d(n_dot_h, a2);
    let g = ggx_g(n_dot_v, n_dot_l, a2);
    let f = fresnel_schlick(v_dot_h, f0);

    let pdf = (d * n_dot_h) / (4.0 * v_dot_h);
    let throughput = bsdf_to_throughput(d, g, f, n_dot_v, n_dot_l, n_dot_h, v_dot_h, pdf);

    return ScatterResult(l, pdf, throughput, false);
}

fn bsdf_to_throughput(d: f32, g: f32, f: vec3<f32>, n_dot_v: f32, n_dot_l: f32, n_dot_h: f32, v_dot_h: f32, pdf: f32) -> vec3<f32> {
    if pdf <= 0.0 { return vec3(0.0); }
    return (d * g * f) / (4.0 * n_dot_v * n_dot_l) * n_dot_l / pdf;
}

// Dielectric
fn reflectance_dielectric(cosine: f32, ref_idx: f32) -> f32 {
    var r0 = (1.0 - ref_idx) / (1.0 + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0 - r0) * pow(1.0 - cosine, 5.0);
}

fn sample_dielectric(dir: vec3<f32>, normal: vec3<f32>, ior: f32, albedo: vec3<f32>, rng: ptr<function, u32>) -> ScatterResult {
    let front_face = dot(dir, normal) < 0.0;
    let refraction_ratio = select(ior, 1.0 / ior, front_face);
    let n = select(-normal, normal, front_face);

    let unit_dir = normalize(dir);
    let cos_theta = min(dot(-unit_dir, n), 1.0);
    let sin_theta = sqrt(1.0 - cos_theta * cos_theta);

    let cannot_refract = refraction_ratio * sin_theta > 1.0;
    var direction: vec3<f32>;

    if cannot_refract || reflectance_dielectric(cos_theta, refraction_ratio) > rand_pcg(rng) {
        direction = reflect(unit_dir, n);
    } else {
        direction = refract(unit_dir, n, refraction_ratio);
    }

    return ScatterResult(direction, 1.0, albedo, true);
}

// =========================================================
//   Direct Light Sampling
// =========================================================


// \u65E2\u5B58\u306E\u30EA\u30B6\u30FC\u30D0(target)\u306B\u3001\u5225\u306E\u30EA\u30B6\u30FC\u30D0(src)\u3092\u30DE\u30FC\u30B8\u3059\u308B
fn merge_reservoir(
    target_reservoir: ptr<function, Reservoir>,
    src: Reservoir,
    p_hat_src: f32, // src\u306E\u52DD\u8005\u30E9\u30A4\u30C8\u306E\u8A55\u4FA1\u5024
    rng: ptr<function, u32>
) -> bool {
    // \u30DE\u30FC\u30B8\u6642\u306E\u91CD\u307F: src.W * src.M * p_hat
    // \u76F4\u611F\u7684\u306B\u306F\u300Csrc\u304C\u6301\u3063\u3066\u3044\u308B\u60C5\u5831\u306E\u4FE1\u983C\u5EA6\u300D
    
    // M\u306E\u4E0A\u9650\u30AD\u30E3\u30C3\u30D7\uFF08\u91CD\u8981\uFF01\uFF09
    // \u3053\u308C\u304C\u306A\u3044\u3068M\u304C\u7121\u9650\u306B\u5897\u3048\u3066\u3001\u65B0\u3057\u3044\u60C5\u5831\u304C\u5165\u308A\u306B\u304F\u304F\u306A\u308B\uFF08Temporal Lag\u306E\u539F\u56E0\uFF09
    let M_clamped = min(src.M, 20.0);

    let weight = p_hat_src * src.W * M_clamped;

    (*target_reservoir).w_sum += weight;
    (*target_reservoir).M += M_clamped;

    if rand_pcg(rng) * (*target_reservoir).w_sum < weight {
        (*target_reservoir).light_idx = src.light_idx;
        (*target_reservoir).r_u = src.r_u;
        (*target_reservoir).r_v = src.r_v;
        return true;
    }
    return false;
}

// \u30EA\u30B6\u30FC\u30D0\u3092\u66F4\u65B0\u3059\u308B\u95A2\u6570
// light_idx: \u5019\u88DC\u306E\u30E9\u30A4\u30C8ID
// weight: \u305D\u306E\u5019\u88DC\u306E\u91CD\u307F (p_hat / source_pdf)
// c: \u5019\u88DC\u306E\u500B\u6570 (\u901A\u5E38\u306F1.0)
fn update_reservoir(res: ptr<function, Reservoir>, light_idx: u32, r_u: f32, r_v: f32, weight: f32, c: f32, rng: ptr<function, u32>) -> bool {
    (*res).w_sum += weight;
    (*res).M += c;

    // \u78BA\u7387\u7684\u306B\u5165\u308C\u66FF\u3048\u308B (\u91CD\u3044\u307B\u3069\u9078\u3070\u308C\u3084\u3059\u3044)
    if rand_pcg(rng) * (*res).w_sum < weight {
        (*res).light_idx = light_idx;
        (*res).r_u = r_u;
        (*res).r_v = r_v;
        return true;
    }
    return false;
}


// \u6307\u5B9A\u3057\u305F\u30E9\u30A4\u30C8ID\u3068\u4E71\u6570(r_u, r_v)\u3092\u4F7F\u3063\u3066\u3001\u30E9\u30A4\u30C8\u4E0A\u306E\u70B9\u3068\u660E\u308B\u3055\u3092\u8A08\u7B97\u3059\u308B
// \u203B\u3053\u3053\u3067\u306F\u300C\u53EF\u8996\u6027(\u58C1\u306E\u88CF\u304B\u3069\u3046\u304B)\u300D\u306F\u30C1\u30A7\u30C3\u30AF\u3057\u307E\u305B\u3093\uFF01
fn evaluate_light_sample(light_idx: u32, r_u: f32, r_v: f32) -> LightCandidate {
    let lp = lights[light_idx];

    // \u91CD\u5FC3\u5EA7\u6A19\u8A08\u7B97 (r_u, r_v \u304B\u3089 u, v, w \u3092\u4F5C\u308B)
    let sqrt_r1 = sqrt(r_u);
    let u = 1.0 - sqrt_r1;
    let v = r_v * sqrt_r1;

    // \u30EF\u30FC\u30EB\u30C9\u7A7A\u9593\u4F4D\u7F6E
    let p = lp.position + lp.u_edge * u + lp.v_edge * v;
    
    // \u9762\u6CD5\u7DDA\u3068\u9762\u7A4D\u3001\u304A\u3088\u3073\u4E8B\u524D\u8A08\u7B97\u6E08\u307F\u767A\u5149\u8272
    let n = lp.normal;
    let area = lp.area;
    let L = lp.emission;
    
    // PDF = 1 / (Area * TotalLights)
    let pdf = 1.0 / (area * f32(scene.light_count));

    return LightCandidate(L, p, n, pdf);
}

fn evaluate_p_hat(hit_p: vec3<f32>, normal: vec3<f32>, light: LightCandidate) -> f32 {
    let l_vec = light.pos - hit_p;
    let dist_sq = dot(l_vec, l_vec);
    let dist = sqrt(dist_sq);
    let dir = l_vec / dist;

    let cos_light = max(dot(light.normal, -dir), 0.0);
    let cos_surf = max(dot(normal, dir), 0.0);
    
    // \u8F1D\u5EA6
    let intensity = dot(light.L, vec3(0.299, 0.587, 0.114));

    return (intensity * cos_light * cos_surf) / max(dist_sq, 1e-4);
}

// RIS (Resampled Importance Sampling) \u3092\u4F7F\u3063\u305F\u30E9\u30A4\u30C8\u30B5\u30F3\u30D7\u30EA\u30F3\u30B0
// hit_p: \u30EC\u30A4\u304C\u5F53\u305F\u3063\u305F\u5834\u6240
// normal: \u30EC\u30A4\u304C\u5F53\u305F\u3063\u305F\u5834\u6240\u306E\u6CD5\u7DDA
fn sample_lights_restir_reuse(hit_p: vec3<f32>, normal: vec3<f32>, p_idx: u32, rng: ptr<function, u32>) -> LightSample {
    let stride = scene.width * scene.height;
    // --- Phase 1: \u521D\u671F\u5019\u88DC (RIS) ---
    // \u524D\u56DE\u306E\u30B3\u30FC\u30C9\u3068\u540C\u3058\uFF0832\u56DE\u30EB\u30FC\u30D7\u3057\u30661\u500B\u9078\u3076\uFF09
    // \u7D50\u679C\u3092 \`state\` (Reservoir\u578B) \u306B\u4FDD\u6301
    let frame_mod = scene.frame_count % 2u;

    let read_offset = frame_mod * stride;           // 0 or stride
    let write_offset = (1u - frame_mod) * stride;

    var state: Reservoir;
    state.w_sum = 0.0; state.M = 0.0; state.W = 0.0;

    let CANDIDATE_COUNT = 4u; // \u518D\u5229\u7528\u3059\u308B\u306A\u3089\u521D\u671F\u5019\u88DC\u306F\u6E1B\u3089\u3057\u3066\u3082OK (32 -> 4)
    for (var i = 0u; i < CANDIDATE_COUNT; i++) {
        let light_idx = u32(rand_pcg(rng) * f32(scene.light_count));
        let r_u = rand_pcg(rng);
        let r_v = rand_pcg(rng);
        let candidate = evaluate_light_sample(light_idx, r_u, r_v);

        let p_hat = evaluate_p_hat(hit_p, normal, candidate);
        let weight = p_hat / max(candidate.pdf, 1e-6);

        update_reservoir(&state, light_idx, r_u, r_v, weight, 1.0, rng);
    }
    
    // RIS\u3067\u9078\u3070\u308C\u305F\u5019\u88DC\u306E p_hat \u3092\u8A08\u7B97\u3057\u3066\u304A\u304F\uFF08\u5F8C\u3067\u4F7F\u3046\uFF09
    // \u203B\u6700\u9069\u5316: update_reservoir\u5185\u3067\u4FDD\u5B58\u3057\u3066\u304A\u304F\u3068\u901F\u3044
    var p_hat_current = 0.0;
        {
        let winner = evaluate_light_sample(state.light_idx, state.r_u, state.r_v);
        p_hat_current = evaluate_p_hat(hit_p, normal, winner);
        
        // RIS\u6BB5\u968E\u3067\u306E W \u3092\u4EEE\u8A08\u7B97
        if p_hat_current > 0.0 {
            state.W = state.w_sum / (state.M * p_hat_current);
        } else {
            state.W = 0.0;
        }
    }


    // --- Phase 2: \u6642\u9593\u7684\u518D\u5229\u7528 (Temporal Reuse) ---
    // \u30D0\u30C3\u30D5\u30A1\u304B\u3089\u524D\u56DE\u306E\u81EA\u5206\u3092\u8AAD\u307F\u8FBC\u3080

    let prev_idx = read_offset + p_idx;
    let prev_data = reservoirs[prev_idx];
    
    // \u524D\u56DE\u306E\u30C7\u30FC\u30BF\u304C\u6709\u52B9\u304B\u30C1\u30A7\u30C3\u30AF\uFF08\u30AB\u30E1\u30E9\u304C\u52D5\u3044\u3066\u3044\u306A\u3044\u524D\u63D0\u306A\u3089\u305D\u306E\u307E\u307E\u4F7F\u3048\u308B\uFF09
    // \u203B\u53B3\u5BC6\u306B\u306F\u3053\u3053\u3067\u30EA\u30D7\u30ED\u30B8\u30A7\u30AF\u30B7\u30E7\u30F3\u3084\u3001\u6CD5\u7DDA\u30FB\u6DF1\u5EA6\u306E\u985E\u4F3C\u5EA6\u30C1\u30A7\u30C3\u30AF\u304C\u5FC5\u8981
    // \u4ECA\u56DE\u306F\u300C\u306A\u3057\u300D\u3067\u7A81\u3063\u8FBC\u3080\uFF08\u591A\u5C11\u306E\u6B8B\u50CF\u306F\u8A31\u5BB9\uFF09

    var prev_res: Reservoir;
    prev_res.light_idx = prev_data.light_idx;
    prev_res.W = prev_data.W;
    prev_res.M = prev_data.M;
    prev_res.r_u = prev_data.r_u;
    prev_res.r_v = prev_data.r_v;
    
    // \u524D\u56DE\u306E\u52DD\u8005\u304C\u3001\u4ECA\u306E\u81EA\u5206\u306B\u3068\u3063\u3066\u3069\u308C\u304F\u3089\u3044\u5B09\u3057\u3044\u304B (p_hat) \u3092\u518D\u8A55\u4FA1
    let prev_winner = evaluate_light_sample(prev_res.light_idx, prev_res.r_u, prev_res.r_v);
    let p_hat_prev = evaluate_p_hat(hit_p, normal, prev_winner);
    
    // \u5F71\u30C1\u30A7\u30C3\u30AF\u306F\u307E\u3060\u3057\u306A\u3044\uFF01p_hat > 0 \u306A\u3089\u30DE\u30FC\u30B8\u3059\u308B
    if p_hat_prev > 0.0 {
        merge_reservoir(&state, prev_res, p_hat_prev, rng);
        // \u52DD\u8005\u304C\u5165\u308C\u66FF\u308F\u3063\u305F\u304B\u3082\u3057\u308C\u306A\u3044\u306E\u3067 p_hat_current \u3092\u66F4\u65B0\u3057\u305F\u3044\u304C\u3001
        // \u53B3\u5BC6\u306B\u306F\u6700\u5F8C\u306B1\u56DE\u3084\u308C\u3070\u3044\u3044
    }


    // --- Phase 3: \u7A7A\u9593\u7684\u518D\u5229\u7528 (Spatial Reuse) ---
    // \u96A3\u306E\u30D4\u30AF\u30BB\u30EB\u3092\u30E9\u30F3\u30C0\u30E0\u306B\u9078\u3093\u3067\u30DE\u30FC\u30B8

    let SPATIAL_COUNT = 2u; // 2\u301C3\u8FD1\u508D\u3092\u898B\u308B
    let RADIUS = 30.0; // \u534A\u5F8430\u30D4\u30AF\u30BB\u30EB\u304F\u3089\u3044\u5E83\u3081\u306B\u63A2\u3059

    for (var i = 0u; i < SPATIAL_COUNT; i++) {
        // \u30E9\u30F3\u30C0\u30E0\u306A\u30AA\u30D5\u30BB\u30C3\u30C8
        let offset = random_in_unit_disk(rng) * RADIUS;
        let nx = i32(f32(scene.width) * 0.0 + f32(p_idx % scene.width) + offset.x); // \u7C21\u6613\u8A08\u7B97
        let ny = i32(f32(scene.height) * 0.0 + f32(p_idx / scene.width) + offset.y);
        
        // \u753B\u9762\u5916\u30C1\u30A7\u30C3\u30AF
        if nx < 0 || nx >= i32(scene.width) || ny < 0 || ny >= i32(scene.height) { continue; }

        let n_idx = u32(ny * i32(scene.width) + nx);
        let neighbor_data = reservoirs[read_offset + n_idx];

        var n_res: Reservoir;
        n_res.light_idx = neighbor_data.light_idx;
        n_res.W = neighbor_data.W;
        n_res.M = neighbor_data.M;
        n_res.r_u = neighbor_data.r_u;
        n_res.r_v = neighbor_data.r_v;
        
        // \u5E7E\u4F55\u5B66\u7684\u985E\u4F3C\u5EA6\u30C1\u30A7\u30C3\u30AF\uFF08\u6CD5\u7DDA\u304C\u9055\u3044\u3059\u304E\u308B\u306A\u3089\u30DE\u30FC\u30B8\u3057\u306A\u3044\uFF09
        // \u203B\u30D0\u30C3\u30D5\u30A1\u306B\u6CD5\u7DDA\u304C\u5165\u3063\u3066\u3044\u306A\u3044\u306E\u3067\u3001\u4ECA\u56DE\u306F\u30B9\u30AD\u30C3\u30D7\u3057\u3066\u300C\u5168\u90E8\u6DF7\u305C\u308B\u300D
        // \uFF08\u89D2\u3067\u5149\u304C\u6F0F\u308C\u308B\u539F\u56E0\u306B\u306A\u308B\u304C\u3001\u307E\u305A\u306F\u52D5\u304B\u3059\u512A\u5148\uFF09

        let n_winner = evaluate_light_sample(n_res.light_idx, n_res.r_u, n_res.r_v);
        let p_hat_n = evaluate_p_hat(hit_p, normal, n_winner);

        if p_hat_n > 0.0 {
            merge_reservoir(&state, n_res, p_hat_n, rng);
        }
    }


    // --- Phase 4: \u6700\u7D42\u6C7A\u5B9A\u3068\u4FDD\u5B58 ---
    
    // \u6700\u7D42\u7684\u306A\u52DD\u8005\u306E p_hat
    let final_winner = evaluate_light_sample(state.light_idx, state.r_u, state.r_v);
    let p_hat_final = evaluate_p_hat(hit_p, normal, final_winner);
    
    // Unbiased Weight (W)
    if p_hat_final > 0.0 {
        state.W = state.w_sum / (state.M * p_hat_final);
    } else {
        state.W = 0.0;
    }
    
    // \u30D0\u30C3\u30D5\u30A1\u306B\u4FDD\u5B58\uFF08\u6B21\u30D5\u30EC\u30FC\u30E0\u7528\uFF09
    var store_data: ReservoirData;
    store_data.w_sum = state.w_sum; // \u203B\u5B9F\u306F\u4FDD\u5B58\u4E0D\u8981\u3060\u304C\u30C7\u30D0\u30C3\u30B0\u7528\u306B
    store_data.M = state.M;
    store_data.W = state.W;
    store_data.light_idx = state.light_idx;
    store_data.r_u = state.r_u;
    store_data.r_v = state.r_v;
    let out_idx = write_offset + p_idx;
    reservoirs[out_idx] = store_data;

    // --- Phase 5: \u30B7\u30E3\u30C9\u30A6\u30EC\u30A4\u7528\u306E\u30B5\u30F3\u30D7\u30EB\u8FD4\u5374 ---
    
    // \u3053\u3053\u3067\u521D\u3081\u3066\u300C\u53EF\u8996\u6027\uFF08Shadow\uFF09\u300D\u3092\u30C1\u30A7\u30C3\u30AF\u3055\u308C\u308B\u3053\u3068\u306B\u306A\u308B
    // \u8FD4\u308A\u5024\u306FLightSample\u578B
    var sample_out: LightSample;
    sample_out.L = final_winner.L;
    let l_vec_final = final_winner.pos - hit_p;
    let dist_sq_final = dot(l_vec_final, l_vec_final);
    let dist_final = sqrt(dist_sq_final);
    
    // \u30BC\u30ED\u9664\u7B97\u5BFE\u7B56
    if dist_final > 1e-6 {
        sample_out.dir = l_vec_final / dist_final;
        sample_out.dist = dist_final;
    } else {
        sample_out.dir = vec3(0.0, 1.0, 0.0); // \u30C0\u30DF\u30FC
        sample_out.dist = 0.0;
    }
    
    // PDF\u30C8\u30EA\u30C3\u30AF:
    // \u5F93\u6765\u306E ray_color \u306F (L / pdf) \u3067\u8A08\u7B97\u3057\u3066\u3044\u308B\u3002
    // ReSTIR\u306F (L * W) \u3067\u8A08\u7B97\u3057\u305F\u3044\u3002
    // \u3064\u307E\u308A pdf = 1.0 / W \u3068\u507D\u308C\u3070\u3001\u65E2\u5B58\u30B3\u30FC\u30C9\u3092\u5909\u3048\u305A\u306B\u6E08\u3080\u3002

    if state.W > 0.0 {
        sample_out.pdf = 1.0 / state.W;
    } else {
        sample_out.pdf = 0.0;
    }

    return sample_out;
}

fn get_light_pdf(origin: vec3<f32>, tri_idx: u32, inst_idx: u32, t: f32, l_dir: vec3<f32>) -> f32 {
    let tri = topology[tri_idx];
    let inst = instances[inst_idx];
    let m = get_transform(inst);

    let v0 = (m * vec4(get_pos(tri.v0), 1.0)).xyz;
    let v1 = (m * vec4(get_pos(tri.v1), 1.0)).xyz;
    let v2 = (m * vec4(get_pos(tri.v2), 1.0)).xyz;

    let edge1 = v1 - v0;
    let edge2 = v2 - v0;
    let area = length(cross(edge1, edge2)) * 0.5;
    let normal = normalize(cross(edge1, edge2));

    let cos_theta_l = max(dot(normal, -l_dir), 0.0);
    if cos_theta_l < 1e-4 { return 0.0; }

    let light_count = scene.light_count;
    let dist_sq = t * t;
    return (dist_sq / (cos_theta_l * area)) / f32(light_count);
}

fn power_heuristic(pdf_a: f32, pdf_b: f32) -> f32 {
    let a2 = pdf_a * pdf_a;
    let b2 = pdf_b * pdf_b;
    return a2 / (a2 + b2);
}

// =========================================================
//   Intersection Functions
// =========================================================

fn intersect_aabb(min_b: vec3<f32>, max_b: vec3<f32>, origin: vec3<f32>, inv_d: vec3<f32>, t_min: f32, t_max: f32) -> f32 {
    let t0s = (min_b - origin) * inv_d;
    let t1s = (max_b - origin) * inv_d;
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
    var stack: array<u32, 24>;
    var stackptr = 0u;

    var idx = node_start_idx;
    if intersect_aabb(nodes[idx].min_b, nodes[idx].max_b, r.origin, inv_d, t_min, closest_t) < 1e30 {
        stack[stackptr] = idx; stackptr++;
    }

    while stackptr > 0u {
        stackptr--;
        idx = stack[stackptr];
        let node = nodes[idx];
        let count = u32(node.tri_count);

        if count > 0u {
            let first = u32(node.left_first);
            for (var i = 0u; i < count; i++) {
                let tri_id = first + i;
                let tri = topology[tri_id];
                let v0 = get_pos(tri.v0);
                let v1 = get_pos(tri.v1);
                let v2 = get_pos(tri.v2);
                let t = hit_triangle_raw(v0, v1, v2, r, t_min, closest_t);
                if t > 0.0 { closest_t = t; hit_idx = f32(tri_id); }
            }
        } else {
            let l = u32(node.left_first) + node_start_idx;
            let r_idx = l + 1u;
            let nl = nodes[l];
            let nr = nodes[r_idx];
            let dl = intersect_aabb(nl.min_b, nl.max_b, r.origin, inv_d, t_min, closest_t);
            let dr = intersect_aabb(nr.min_b, nr.max_b, r.origin, inv_d, t_min, closest_t);

            if dl < 1e30 && dr < 1e30 {
                if dl < dr { stack[stackptr] = r_idx; stackptr++; stack[stackptr] = l; stackptr++; } else { stack[stackptr] = l; stackptr++; stack[stackptr] = r_idx; stackptr++; }
            } else if dl < 1e30 { stack[stackptr] = l; stackptr++; } else if dr < 1e30 { stack[stackptr] = r_idx; stackptr++; }
        }
    }
    return vec2<f32>(closest_t, hit_idx);
}

fn intersect_tlas(r: Ray, t_min: f32, t_max: f32) -> HitResult {
    var res: HitResult; res.t = t_max; res.tri_idx = -1.0; res.inst_idx = -1;
    if scene.blas_base_idx == 0u { return res; }

    let inv_d = 1.0 / r.direction;
    var stack: array<u32, 16>;
    var stackptr = 0u;

    if intersect_aabb(nodes[0].min_b, nodes[0].max_b, r.origin, inv_d, t_min, res.t) < 1e30 {
        stack[stackptr] = 0u; stackptr++;
    }

    while stackptr > 0u {
        stackptr--;
        let idx = stack[stackptr];
        let node = nodes[idx];

        if node.tri_count > 0.5 { // Leaf
            let inst_idx = u32(node.left_first);
            let inst = instances[inst_idx];
            let inv = get_inv_transform(inst);
            let r_local = Ray((inv * vec4(r.origin, 1.0)).xyz, (inv * vec4(r.direction, 0.0)).xyz);
            let blas_start = scene.blas_base_idx + inst.blas_node_offset;
            let blas = intersect_blas(r_local, t_min, res.t, blas_start);
            if blas.y > -0.5 {
                res.t = blas.x; res.tri_idx = blas.y; res.inst_idx = i32(inst_idx);
            }
        } else {
            let l = u32(node.left_first);
            let r_idx = l + 1u;
            let nl = nodes[l];
            let nr = nodes[r_idx];
            let dl = intersect_aabb(nl.min_b, nl.max_b, r.origin, inv_d, t_min, res.t);
            let dr = intersect_aabb(nr.min_b, nr.max_b, r.origin, inv_d, t_min, res.t);
            if dl < 1e30 && dr < 1e30 {
                if dl < dr { stack[stackptr] = r_idx; stackptr++; stack[stackptr] = l; stackptr++; } else { stack[stackptr] = l; stackptr++; stack[stackptr] = r_idx; stackptr++; }
            } else if dl < 1e30 { stack[stackptr] = l; stackptr++; } else if dr < 1e30 { stack[stackptr] = r_idx; stackptr++; }
        }
    }
    return res;
}

// =========================================================
//   Main Path Tracer Loop
// =========================================================

fn ray_color(r_in: Ray, p_idx: u32, rng: ptr<function, u32>) -> vec3<f32> {
    var ray = r_in;
    var throughput = vec3(1.0);
    var radiance = vec3(0.0);

    var prev_bsdf_pdf = 0.0;
    var specular_bounce = true;

    for (var depth = 0u; depth < MAX_DEPTH; depth++) {
        let hit = intersect_tlas(ray, T_MIN, T_MAX);
        if hit.inst_idx < 0 { break; }

        let tri_idx = u32(hit.tri_idx);
        let inst_idx = u32(hit.inst_idx);
        let tri = topology[tri_idx];
        let mat_type = bitcast<u32>(tri.data0.w);
        
        // --- Geometry ---
        let inst = instances[inst_idx];
        let inv = get_inv_transform(inst);
        let v0_pos = get_pos(tri.v0);
        let v1_pos = get_pos(tri.v1);
        let v2_pos = get_pos(tri.v2);

        let r_local = Ray((inv * vec4(ray.origin, 1.)).xyz, (inv * vec4(ray.direction, 0.)).xyz);
        let s = r_local.origin - v0_pos;
        let e1 = v1_pos - v0_pos;
        let e2 = v2_pos - v0_pos;
        let h_val = cross(r_local.direction, e2);
        let a = dot(e1, h_val);
        let f_val = 1.0 / a;
        let u_bar = f_val * dot(s, h_val);
        let q = cross(s, e1);
        let v_bar = f_val * dot(r_local.direction, q);
        let w_bar = 1.0 - u_bar - v_bar;

        // Normals
        let n0 = get_normal(tri.v0);
        let n1 = get_normal(tri.v1);
        let n2 = get_normal(tri.v2);
        let ln = normalize(n0 * w_bar + n1 * u_bar + n2 * v_bar);
        var normal = normalize((vec4(ln, 0.0) * inv).xyz);

        let hit_p = ray.origin + ray.direction * hit.t;
        
        // Textures & Attributes
        let uv0 = get_uv(tri.v0);
        let uv1 = get_uv(tri.v1);
        let uv2 = get_uv(tri.v2);
        let tex_uv = uv0 * w_bar + uv1 * u_bar + uv2 * v_bar;
        
        // 1. Base Color
        var albedo = tri.data0.rgb;
        let base_tex = tri.data2.x;
        if base_tex > -0.5 {
            albedo *= textureSampleLevel(tex, smp, tex_uv, i32(base_tex), 0.0).rgb;
        }

        // 2. Normal Map
        let normal_tex = tri.data2.z;
        if normal_tex > -0.5 {
            let n_map = textureSampleLevel(tex, smp, tex_uv, i32(normal_tex), 0.0).rgb * 2.0 - 1.0;
            let T = normalize(e1);
            let B = normalize(cross(ln, T));
            let ln_mapped = normalize(T * n_map.x + B * n_map.y + ln * n_map.z);
            normal = normalize((vec4(ln_mapped, 0.0) * inv).xyz);
        }

        let front_face = dot(ray.direction, normal) < 0.0;
        normal = select(-normal, normal, front_face);

        // 3. Metallic / Roughness
        var metallic = tri.data1.x;
        var roughness = tri.data1.y;
        let met_rough_tex = tri.data2.y;
        if met_rough_tex > -0.5 {
            let mr = textureSampleLevel(tex, smp, tex_uv, i32(met_rough_tex), 0.0).rgb;
            metallic *= mr.b;
            roughness *= mr.g;
        }
        roughness = max(roughness, 0.005);

        // 4. Emissive
        var emissive = tri.data3.rgb;
        let em_tex = tri.data2.w;
        if em_tex > -0.5 {
            emissive *= textureSampleLevel(tex, smp, tex_uv, i32(em_tex), 0.0).rgb;
        }

        // --- Emission ---
        if mat_type == 3u || length(emissive) > 1e-4 {
            let em_val = select(emissive, albedo, mat_type == 3u);
            if specular_bounce {
                radiance += throughput * em_val;
            } else {
                let light_pdf_val = get_light_pdf(ray.origin, tri_idx, inst_idx, hit.t, ray.direction);
                let weight = power_heuristic(prev_bsdf_pdf, light_pdf_val);
                radiance += throughput * em_val * weight;
            }
            if mat_type == 3u { break; }
        }

        // --- Material Setup ---
        var f0 = mix(vec3(0.04), albedo, metallic);
        var is_specular = (mat_type == 2u) || (metallic > 0.9 && roughness < 0.3); 

        // --- NEE ---
        if !is_specular && mat_type != 2u {
            let light_s = sample_lights_restir_reuse(hit_p, normal, p_idx, rng);
            if light_s.pdf > 0.0 {
                let shadow_ray = Ray(hit_p + normal * 1e-4, light_s.dir);
                let shadow_hit = intersect_tlas(shadow_ray, T_MIN, light_s.dist - 2e-4);

                if shadow_hit.inst_idx == -1 {
                    var bsdf_val = vec3(0.0);
                    var bsdf_pdf_val = 0.0;
                    let L = light_s.dir;
                    let V = -ray.direction;

                    if mat_type == 0u {
                        bsdf_val = eval_diffuse(albedo);
                        bsdf_pdf_val = max(dot(normal, L), 0.0) / PI;
                    } else if mat_type == 1u {
                        bsdf_val = eval_ggx(normal, V, L, roughness, f0);
                        let H = normalize(V + L);
                        bsdf_pdf_val = (ggx_d(dot(normal, H), roughness * roughness) * max(dot(normal, H), 0.0)) / (4.0 * max(dot(V, H), 0.0));
                    }

                    if bsdf_pdf_val > 0.0 {
                        let weight = power_heuristic(light_s.pdf, bsdf_pdf_val);
                        radiance += throughput * bsdf_val * light_s.L * weight * max(dot(normal, L), 0.0) / light_s.pdf;
                    }
                }
            }
        }

        // --- BSDF Sampling ---
        var scatter: ScatterResult;
        if mat_type == 0u {
            scatter = sample_diffuse(normal, albedo, rng);
        } else if mat_type == 1u {
            scatter = sample_ggx(normal, -ray.direction, roughness, f0, rng);
        } else {
            let ior = tri.data1.z;
            scatter = sample_dielectric(ray.direction, normal, ior, albedo, rng);
        }

        if scatter.pdf <= 0.0 || length(scatter.throughput) <= 0.0 { break; }

        throughput *= scatter.throughput;
        ray = Ray(hit_p + normal * 1e-4, scatter.dir);
        prev_bsdf_pdf = scatter.pdf;
        specular_bounce = scatter.is_specular;

        // RR
        if depth > 3u {
            let p = max(throughput.r, max(throughput.g, throughput.b));
            if rand_pcg(rng) > p { break; }
            throughput /= p;
        }
    }
    return radiance;
}

// =========================================================
//   Compute Main
// =========================================================

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= scene.width || id.y >= scene.height { return; }
    let p_idx = id.y * scene.width + id.x;

    var col = vec3(0.);
    for (var i = 0u; i < SPP; i++) {
        var rng = init_rng(p_idx, scene.frame_count * SPP + i);

        var off = vec3(0.);
        if scene.camera.origin.w > 0. {
            let rd = scene.camera.origin.w * random_in_unit_disk(&rng);
            off = scene.camera.u.xyz * rd.x + scene.camera.v.xyz * rd.y;
        }

        let u = (f32(id.x) + 0.5 + scene.jitter.x * f32(scene.width)) / f32(scene.width);
        let v = 1. - (f32(id.y) + 0.5 + scene.jitter.y * f32(scene.height)) / f32(scene.height);
        let d = scene.camera.lower_left_corner.xyz + u * scene.camera.horizontal.xyz + v * scene.camera.vertical.xyz - scene.camera.origin.xyz - off;

        col += ray_color(Ray(scene.camera.origin.xyz + off, d), p_idx, &rng);
    }
    col /= f32(SPP);
    accumulateBuffer[p_idx] = vec4(col, 1.0);
}
`, Z = `// =========================================================
//   Post Process (PostProcess.wgsl)
// =========================================================

struct Camera {
    origin: vec4<f32>, // w: lens_radius
    lower_left_corner: vec4<f32>,
    horizontal: vec4<f32>,
    vertical: vec4<f32>,
    u: vec4<f32>,
    v: vec4<f32>
}

struct SceneUniforms {
    camera: Camera,
    prev_camera: Camera,
    frame_count: u32,
    blas_base_idx: u32,
    vertex_count: u32,
    rand_seed: u32,
    light_count: u32,
    width: u32,
    height: u32,
    pad: u32,
    jitter: vec2<f32>,
    prev_jitter: vec2<f32>
}

@group(0) @binding(0) var outputTex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<storage, read> accumulateBuffer: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> scene: SceneUniforms;
@group(0) @binding(3) var historyTex: texture_2d<f32>;
@group(0) @binding(4) var smp: sampler;
@group(0) @binding(5) var historyOutput: texture_storage_2d<rgba16float, write>;

fn aces_tone_mapping(color: vec3<f32>) -> vec3<f32> {
    let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), vec3(0.0), vec3(1.0));
}

fn get_radiance(coord: vec2<i32>) -> vec3<f32> {
    let c = clamp(coord, vec2<i32>(0), vec2<i32>(i32(scene.width) - 1, i32(scene.height) - 1));
    let p_idx = u32(c.y) * scene.width + u32(c.x);
    let acc = accumulateBuffer[p_idx];
    if acc.a <= 0.0 { return vec3(0.0); }
    return acc.rgb / acc.a;
}

fn get_radiance_clean(coord: vec2<i32>) -> vec3<f32> {
    let center = get_radiance(coord);

    var min_nb = vec3(1e6);
    var max_nb = vec3(-1e6);
    
    // 3x3 box (excluding center) for more stable suppression
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            if x == 0 && y == 0 { continue; }
            let nb = get_radiance(coord + vec2<i32>(x, y));
            min_nb = min(min_nb, nb);
            max_nb = max(max_nb, nb);
        }
    }
    
    // Firefly suppression: clamp center to neighborhood range with some headroom
    let threshold = 3.0;
    return clamp(center, vec3(0.0), max_nb * threshold + 0.1);
}

// Bilinear sampling for un-jittering
fn get_radiance_bilinear(uv: vec2<f32>) -> vec3<f32> {
    let dims = vec2<f32>(f32(scene.width), f32(scene.height));
    let f_coord = uv * dims - 0.5;
    let i_coord = vec2<i32>(floor(f_coord));
    let f = f_coord - vec2<f32>(i_coord);

    let c00 = get_radiance_clean(i_coord + vec2<i32>(0, 0));
    let c10 = get_radiance_clean(i_coord + vec2<i32>(1, 0));
    let c01 = get_radiance_clean(i_coord + vec2<i32>(0, 1));
    let c11 = get_radiance_clean(i_coord + vec2<i32>(1, 1));

    return mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);
}

fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= scene.width || id.y >= scene.height { return; }

    let dims = vec2<f32>(f32(scene.width), f32(scene.height));
    let uv = (vec2<f32>(id.xy) + 0.5) / dims;

    // 1. Un-jittered Current Frame Radiance (Now cleaned internally)
    let center_color = get_radiance_bilinear(uv - scene.jitter);

    // 2. Bilateral Filter
    let SIGMA_S = 0.5;
    let SIGMA_R = 0.1;
    let RADIUS = 1;

    var filtered_sum = vec3(0.0);
    var total_weight = 0.0;
    for (var dy = -RADIUS; dy <= RADIUS; dy++) {
        for (var dx = -RADIUS; dx <= RADIUS; dx++) {
            let samp_uv = uv + vec2<f32>(f32(dx), f32(dy)) / dims - scene.jitter;
            let neighbor_color = get_radiance_bilinear(samp_uv);

            let w_s = exp(-f32(dx * dx + dy * dy) / (2.0 * SIGMA_S * SIGMA_S));
            let color_diff = neighbor_color - center_color;
            let w_r = exp(-dot(color_diff, color_diff) / (2.0 * SIGMA_R * f32(RADIUS) * f32(RADIUS)));
            let w = w_s * w_r;
            filtered_sum += neighbor_color * w;
            total_weight += w;
        }
    }
    let denoised_hdr = filtered_sum / max(total_weight, 1e-4);
    
    // 3. TAA Blend (HDR Feedback)
    let samples_history = textureSampleLevel(historyTex, smp, uv, 0.0).rgb;
    
    // Neighborhood Clamping
    var m1 = vec3(0.0);
    var m2 = vec3(0.0);
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let nb_uv = uv + vec2<f32>(f32(dx), f32(dy)) / dims - scene.jitter;
            let c = get_radiance_bilinear(nb_uv);
            m1 += c;
            m2 += c * c;
        }
    }
    let mean = m1 / 9.0;
    let stddev = sqrt(max(m2 / 9.0 - mean * mean, vec3(0.0)));
    
    // Adaptive clamping
    var k = 1.5;
    if scene.frame_count > 16u { k = 60.0; } // Effectively disable clamping when static for full convergence
    let clamped_history = clamp(samples_history, mean - stddev * k, mean + stddev * k);

    // Adaptive alpha for progressive refinement
    var alpha = 1.0 / f32(scene.frame_count);
    alpha = max(alpha, 0.0001); // Even deeper convergence (10000 frames)

    let final_hdr = mix(clamped_history, denoised_hdr, alpha);

    // Store un-jittered result
    textureStore(historyOutput, vec2<i32>(id.xy), vec4(final_hdr, 1.0));

    // 4. Output
    let mapped = aces_tone_mapping(final_hdr);
    let edge_detect = center_color - denoised_hdr;
    let sharpened = mapped + aces_tone_mapping(edge_detect) * 0.3;

    let ldr_out = pow(clamp(sharpened, vec3(0.0), vec3(1.0)), vec3<f32>(1.0 / 2.2));
    textureStore(outputTex, vec2<i32>(id.xy), vec4(ldr_out, 1.0));
}`;
  class ee {
    constructor(e) {
      __publicField(this, "device");
      __publicField(this, "context");
      __publicField(this, "pipeline");
      __publicField(this, "postprocessPipeline");
      __publicField(this, "bindGroupLayout");
      __publicField(this, "postprocessBindGroupLayout");
      __publicField(this, "bindGroup");
      __publicField(this, "postprocessBindGroup");
      __publicField(this, "renderTarget");
      __publicField(this, "renderTargetView");
      __publicField(this, "accumulateBuffer");
      __publicField(this, "sceneUniformBuffer");
      __publicField(this, "geometryBuffer");
      __publicField(this, "nodesBuffer");
      __publicField(this, "topologyBuffer");
      __publicField(this, "instanceBuffer");
      __publicField(this, "lightsBuffer");
      __publicField(this, "texture");
      __publicField(this, "defaultTexture");
      __publicField(this, "sampler");
      __publicField(this, "reservoirBuffer");
      __publicField(this, "bufferSize", 0);
      __publicField(this, "canvas");
      __publicField(this, "blasOffset", 0);
      __publicField(this, "vertexCount", 0);
      __publicField(this, "seed", Math.floor(Math.random() * 16777215));
      __publicField(this, "historyTextures", []);
      __publicField(this, "historyTextureViews", []);
      __publicField(this, "historyIndex", 0);
      __publicField(this, "prevCameraData", new Float32Array(24));
      __publicField(this, "jitter", {
        x: 0,
        y: 0
      });
      __publicField(this, "prevJitter", {
        x: 0,
        y: 0
      });
      __publicField(this, "uniformMixedData", new Uint32Array(12));
      __publicField(this, "totalFrames", 0);
      __publicField(this, "lightCount", 0);
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
        size: 256,
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
      const s = this.device.createShaderModule({
        label: "PostProcess",
        code: Z
      });
      this.postprocessPipeline = this.device.createComputePipeline({
        label: "PostProcess Pipeline",
        layout: "auto",
        compute: {
          module: s,
          entryPoint: "main"
        }
      }), this.postprocessBindGroupLayout = this.postprocessPipeline.getBindGroupLayout(0);
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
      for (let r = 0; r < 2; r++) this.historyTextures[r] && this.historyTextures[r].destroy(), this.historyTextures[r] = this.device.createTexture({
        size: [
          e,
          t
        ],
        format: "rgba16float",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST
      }), this.historyTextureViews[r] = this.historyTextures[r].createView();
      const n = e * t * 32 * 2;
      this.reservoirBuffer && this.reservoirBuffer.destroy(), this.reservoirBuffer = this.device.createBuffer({
        label: "ReservoirBuffer",
        size: n,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
      });
    }
    resetAccumulation() {
      !this.accumulateBuffer || !this.reservoirBuffer || (this.device.queue.writeBuffer(this.accumulateBuffer, 0, new Float32Array(this.bufferSize / 4)), this.device.queue.writeBuffer(this.reservoirBuffer, 0, new Float32Array(this.canvas.width * this.canvas.height * 32 * 2 / 4)));
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
        const s = e.getTexture(r);
        if (s) try {
          const o = new Blob([
            s
          ]), l = await createImageBitmap(o, {
            resizeWidth: 1024,
            resizeHeight: 1024
          });
          n.push(l);
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
      return r = r + 3 & -4, r = Math.max(r, 16), this.device.createBuffer({
        label: n,
        size: r,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
      });
    }
    updateBuffer(e, t) {
      const n = t.byteLength;
      let r = false, s;
      return e === "topology" ? ((!this.topologyBuffer || this.topologyBuffer.size < n) && (r = true), this.topologyBuffer = this.ensureBuffer(this.topologyBuffer, n, "TopologyBuffer"), s = this.topologyBuffer) : e === "instance" ? ((!this.instanceBuffer || this.instanceBuffer.size < n) && (r = true), this.instanceBuffer = this.ensureBuffer(this.instanceBuffer, n, "InstanceBuffer"), s = this.instanceBuffer) : ((!this.lightsBuffer || this.lightsBuffer.size < n) && (r = true), this.lightsBuffer = this.ensureBuffer(this.lightsBuffer, n, "LightsBuffer"), s = this.lightsBuffer), this.device.queue.writeBuffer(s, 0, t, 0, t.length), r;
    }
    updateCombinedGeometry(e, t, n) {
      const r = e.byteLength + t.byteLength + n.byteLength;
      let s = false;
      (!this.geometryBuffer || this.geometryBuffer.size < r) && (s = true);
      const o = e.length / 4;
      this.vertexCount = o, this.geometryBuffer = this.ensureBuffer(this.geometryBuffer, r, "GeometryBuffer"), !(n.length >= o * 2) && o > 0 && console.warn(`UV buffer mismatch: V=${o}, UV=${n.length / 2}. Filling 0.`);
      let c = 0;
      return this.device.queue.writeBuffer(this.geometryBuffer, c, e), c += e.byteLength, this.device.queue.writeBuffer(this.geometryBuffer, c, t), c += t.byteLength, this.device.queue.writeBuffer(this.geometryBuffer, c, n), s;
    }
    updateCombinedBVH(e, t) {
      const n = e.byteLength, r = t.byteLength, s = n + r;
      let o = false;
      return (!this.nodesBuffer || this.nodesBuffer.size < s) && (o = true), this.nodesBuffer = this.ensureBuffer(this.nodesBuffer, s, "NodesBuffer"), this.device.queue.writeBuffer(this.nodesBuffer, 0, e), this.device.queue.writeBuffer(this.nodesBuffer, n, t), this.blasOffset = e.length / 8, o;
    }
    updateSceneUniforms(e, t, n) {
      if (this.lightCount = n, !this.sceneUniformBuffer) return;
      const r = (c, h) => {
        let u = 1, g = 0;
        for (; c > 0; ) u = u / h, g = g + u * (c % h), c = Math.floor(c / h);
        return g;
      }, s = r(t % 16 + 1, 2) - 0.5, o = r(t % 16 + 1, 3) - 0.5;
      this.jitter = {
        x: s / this.canvas.width,
        y: o / this.canvas.height
      }, this.device.queue.writeBuffer(this.sceneUniformBuffer, 0, e), this.device.queue.writeBuffer(this.sceneUniformBuffer, 96, this.prevCameraData), this.uniformMixedData[0] = t, this.uniformMixedData[1] = this.blasOffset, this.uniformMixedData[2] = this.vertexCount, this.uniformMixedData[3] = this.seed, this.uniformMixedData[4] = n, this.uniformMixedData[5] = this.canvas.width, this.uniformMixedData[6] = this.canvas.height, this.uniformMixedData[7] = 0;
      const l = new Float32Array(this.uniformMixedData.buffer);
      l[8] = this.jitter.x, l[9] = this.jitter.y, this.device.queue.writeBuffer(this.sceneUniformBuffer, 192, this.uniformMixedData), this.prevCameraData.set(e);
    }
    recreateBindGroup() {
      !this.renderTargetView || !this.accumulateBuffer || !this.geometryBuffer || !this.nodesBuffer || !this.sceneUniformBuffer || !this.lightsBuffer || !this.reservoirBuffer || (this.bindGroup = this.device.createBindGroup({
        layout: this.bindGroupLayout,
        entries: [
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
              buffer: this.topologyBuffer
            }
          },
          {
            binding: 5,
            resource: {
              buffer: this.nodesBuffer
            }
          },
          {
            binding: 6,
            resource: {
              buffer: this.instanceBuffer
            }
          },
          {
            binding: 7,
            resource: this.texture.createView({
              dimension: "2d-array"
            })
          },
          {
            binding: 8,
            resource: this.sampler
          },
          {
            binding: 9,
            resource: {
              buffer: this.lightsBuffer
            }
          },
          {
            binding: 10,
            resource: {
              buffer: this.reservoirBuffer
            }
          }
        ]
      }), this.postprocessBindGroup = this.device.createBindGroup({
        layout: this.postprocessBindGroupLayout,
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
            resource: this.historyTextureViews[1 - this.historyIndex]
          },
          {
            binding: 4,
            resource: this.sampler
          },
          {
            binding: 5,
            resource: this.historyTextureViews[this.historyIndex]
          }
        ]
      }));
    }
    compute(e) {
      if (!this.bindGroup || !this.postprocessBindGroup) return;
      this.totalFrames++;
      const t = (u, g) => {
        let b = 1, v = 0;
        for (; u > 0; ) b = b / g, v = v + b * (u % g), u = Math.floor(u / g);
        return v;
      }, n = t(this.totalFrames % 16 + 1, 2) - 0.5, r = t(this.totalFrames % 16 + 1, 3) - 0.5;
      this.prevJitter.x = this.jitter.x, this.prevJitter.y = this.jitter.y, this.jitter = {
        x: n / this.canvas.width,
        y: r / this.canvas.height
      }, this.uniformMixedData[0] = e, this.uniformMixedData[1] = this.blasOffset, this.uniformMixedData[2] = this.vertexCount, this.uniformMixedData[3] = this.seed, this.uniformMixedData[4] = this.lightCount, this.uniformMixedData[5] = this.canvas.width, this.uniformMixedData[6] = this.canvas.height, this.uniformMixedData[7] = 0;
      const s = new Float32Array(this.uniformMixedData.buffer);
      s[8] = this.jitter.x, s[9] = this.jitter.y, s[10] = this.prevJitter.x, s[11] = this.prevJitter.y, this.device.queue.writeBuffer(this.sceneUniformBuffer, 192, this.uniformMixedData);
      const o = Math.ceil(this.canvas.width / 8), l = Math.ceil(this.canvas.height / 8), c = this.device.createCommandEncoder(), h = c.beginComputePass();
      h.setPipeline(this.pipeline), h.setBindGroup(0, this.bindGroup), h.dispatchWorkgroups(o, l), h.end(), this.device.queue.submit([
        c.finish()
      ]);
    }
    present() {
      if (!this.renderTarget || !this.postprocessBindGroup) return;
      const e = Math.ceil(this.canvas.width / 8), t = Math.ceil(this.canvas.height / 8), n = this.device.createCommandEncoder(), r = n.beginComputePass();
      r.setPipeline(this.postprocessPipeline), r.setBindGroup(0, this.postprocessBindGroup), r.dispatchWorkgroups(e, t), r.end(), n.copyTextureToTexture({
        texture: this.renderTarget
      }, {
        texture: this.context.getCurrentTexture()
      }, {
        width: this.canvas.width,
        height: this.canvas.height,
        depthOrArrayLayers: 1
      }), this.device.queue.submit([
        n.finish()
      ]), this.historyIndex = 1 - this.historyIndex, this.recreateBindGroup();
    }
  }
  function te(i) {
    return new Worker("/webgpu-raytracer/assets/wasm-worker-BESrBICk.js", {
      name: i == null ? void 0 : i.name
    });
  }
  class ne {
    constructor() {
      __publicField(this, "worker");
      __publicField(this, "resolveReady", null);
      __publicField(this, "_vertices", new Float32Array(0));
      __publicField(this, "_normals", new Float32Array(0));
      __publicField(this, "_uvs", new Float32Array(0));
      __publicField(this, "_mesh_topology", new Uint32Array(0));
      __publicField(this, "_lights", new Uint32Array(0));
      __publicField(this, "_tlas", new Float32Array(0));
      __publicField(this, "_blas", new Float32Array(0));
      __publicField(this, "_instances", new Float32Array(0));
      __publicField(this, "_cameraData", new Float32Array(24));
      __publicField(this, "_textureCount", 0);
      __publicField(this, "_textures", []);
      __publicField(this, "_animations", []);
      __publicField(this, "hasNewData", false);
      __publicField(this, "hasNewGeometry", false);
      __publicField(this, "pendingUpdate", false);
      __publicField(this, "resolveSceneLoad", null);
      __publicField(this, "updateResolvers", []);
      __publicField(this, "lastWidth", -1);
      __publicField(this, "lastHeight", -1);
      this.worker = new te(), this.worker.onmessage = this.handleMessage.bind(this);
    }
    get lights() {
      return this._lights;
    }
    get lightCount() {
      return this._lights.length / 2;
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
          this._vertices = t.vertices, this._normals = t.normals, this._uvs = t.uvs, this._mesh_topology = t.mesh_topology, this._lights = t.lights, this._tlas = t.tlas, this._blas = t.blas, this._instances = t.instances, this._cameraData = t.camera, this._textureCount = t.textureCount, this._textures = t.textures || [], this._animations = t.animations || [], this.hasNewData = true, this.hasNewGeometry = true, (_b = this.resolveSceneLoad) == null ? void 0 : _b.call(this);
          break;
        case "UPDATE_RESULT":
          this._tlas = t.tlas, this._blas = t.blas, this._instances = t.instances, this._lights = t.lights, this._cameraData = t.camera, t.vertices && (this._vertices = t.vertices, this.hasNewGeometry = true), t.normals && (this._normals = t.normals), t.uvs && (this._uvs = t.uvs), t.mesh_topology && (this._mesh_topology = t.mesh_topology), this.hasNewData = true, this.pendingUpdate = false, this.updateResolvers.forEach((n) => n()), this.updateResolvers = [];
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
    get mesh_topology() {
      return this._mesh_topology;
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
      console.log(`Scene Stats (Worker Proxy): V=${this.vertices.length / 4}, Topo=${this.mesh_topology.length / 12}, I=${this.instances.length / 16}, TLAS=${this.tlas.length / 8}, BLAS=${this.blas.length / 8}, Anim=${this._animations.length}, Lights=${this._lights.length / 2}`);
    }
  }
  const _ = {
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
  class re {
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
      this.canvas = this.el(_.ids.canvas), this.btnRender = this.el(_.ids.renderBtn), this.sceneSelect = this.el(_.ids.sceneSelect), this.inputWidth = this.el(_.ids.resWidth), this.inputHeight = this.el(_.ids.resHeight), this.inputFile = this.setupFileInput(), this.inputDepth = this.el(_.ids.maxDepth), this.inputSPP = this.el(_.ids.sppFrame), this.btnRecompile = this.el(_.ids.recompileBtn), this.inputUpdateInterval = this.el(_.ids.updateInterval), this.animSelect = this.el(_.ids.animSelect), this.btnRecord = this.el(_.ids.recordBtn), this.inputRecFps = this.el(_.ids.recFps), this.inputRecDur = this.el(_.ids.recDuration), this.inputRecSpp = this.el(_.ids.recSpp), this.inputRecBatch = this.el(_.ids.recBatch), this.btnHost = this.el(_.ids.btnHost), this.btnWorker = this.el(_.ids.btnWorker), this.statusDiv = this.el(_.ids.statusDiv), this.statsDiv = this.createStatsDiv(), this.bindEvents();
    }
    el(e) {
      const t = document.getElementById(e);
      if (!t) throw new Error(`Element not found: ${e}`);
      return t;
    }
    setupFileInput() {
      const e = this.el(_.ids.objFile);
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
        return (_a = this.onResolutionChange) == null ? void 0 : _a.call(this, parseInt(this.inputWidth.value) || _.defaultWidth, parseInt(this.inputHeight.value) || _.defaultHeight);
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
        width: parseInt(this.inputWidth.value, 10) || _.defaultWidth,
        height: parseInt(this.inputHeight.value, 10) || _.defaultHeight,
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
  class ie {
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
      const { Muxer: r, ArrayBufferTarget: s } = await z(async () => {
        const { Muxer: h, ArrayBufferTarget: u } = await import("./webm-muxer-MLtUgOCn.js");
        return {
          Muxer: h,
          ArrayBufferTarget: u
        };
      }, []), o = Math.ceil(e.fps * e.duration);
      console.log(`Starting recording: ${o} frames @ ${e.fps}fps (VP9)`);
      const l = new r({
        target: new s(),
        video: {
          codec: "V_VP9",
          width: this.canvas.width,
          height: this.canvas.height,
          frameRate: e.fps
        }
      }), c = new VideoEncoder({
        output: (h, u) => l.addVideoChunk(h, u),
        error: (h) => console.error("VideoEncoder Error:", h)
      });
      c.configure({
        codec: "vp09.00.10.08",
        width: this.canvas.width,
        height: this.canvas.height,
        bitrate: 12e6
      });
      try {
        await this.renderAndEncode(o, e, c, t, e.startFrame || 0), await c.flush(), l.finalize();
        const { buffer: h } = l.target, u = new Blob([
          h
        ], {
          type: "video/webm"
        }), g = URL.createObjectURL(u);
        n(g, u);
      } catch (h) {
        throw console.error("Recording failed:", h), h;
      } finally {
        this.isRecording = false;
      }
    }
    async recordChunks(e, t) {
      if (this.isRecording) throw new Error("Already recording");
      this.isRecording = true;
      const n = [], r = Math.ceil(e.fps * e.duration), s = new VideoEncoder({
        output: (o, l) => {
          const c = new Uint8Array(o.byteLength);
          o.copyTo(c), n.push({
            type: o.type,
            timestamp: o.timestamp,
            duration: o.duration,
            data: c.buffer,
            decoderConfig: l == null ? void 0 : l.decoderConfig
          });
        },
        error: (o) => console.error("VideoEncoder Error:", o)
      });
      s.configure({
        codec: "vp09.00.10.08",
        width: this.canvas.width,
        height: this.canvas.height,
        bitrate: 12e6
      });
      try {
        return await this.renderAndEncode(r, e, s, t, e.startFrame || 0), await s.flush(), n;
      } finally {
        this.isRecording = false;
      }
    }
    async renderAndEncode(e, t, n, r, s = 0) {
      const o = s;
      this.worldBridge.update(o / t.fps), await this.worldBridge.waitForNextUpdate();
      for (let l = 0; l < e; l++) {
        r(l, e), await new Promise((u) => setTimeout(u, 0)), await this.updateSceneBuffers();
        let c = null;
        if (l < e - 1) {
          const u = s + l + 1;
          this.worldBridge.update(u / t.fps), c = this.worldBridge.waitForNextUpdate();
        }
        await this.renderFrame(t.spp, t.batch), n.encodeQueueSize > 5 && await n.flush();
        const h = new VideoFrame(this.canvas, {
          timestamp: (s + l) * 1e6 / t.fps,
          duration: 1e6 / t.fps
        });
        n.encode(h, {
          keyFrame: l % t.fps === 0
        }), h.close(), c && await c;
      }
    }
    async updateSceneBuffers() {
      let e = false;
      e || (e = this.renderer.updateCombinedBVH(this.worldBridge.tlas, this.worldBridge.blas)), e || (e = this.renderer.updateBuffer("instance", this.worldBridge.instances)), e || (e = this.renderer.updateCombinedGeometry(this.worldBridge.vertices, this.worldBridge.normals, this.worldBridge.uvs)), e || (e = this.renderer.updateBuffer("topology", this.worldBridge.mesh_topology)), e || (e = this.renderer.updateBuffer("lights", this.worldBridge.lights)), this.worldBridge.updateCamera(this.canvas.width, this.canvas.height), this.renderer.updateSceneUniforms(this.worldBridge.cameraData, 0, this.worldBridge.lightCount), e && this.renderer.recreateBindGroup(), this.renderer.resetAccumulation();
    }
    async renderFrame(e, t) {
      let n = 0;
      for (; n < e; ) {
        const r = Math.min(t, e - n);
        for (let s = 0; s < r; s++) this.renderer.compute(n + s);
        n += r, this.renderer.present(), await this.renderer.device.queue.onSubmittedWorkDone();
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
  class H {
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
      const s = {
        type: "SCENE_INIT",
        totalBytes: r.byteLength,
        config: {
          ...n,
          fileType: t
        }
      };
      this.sendData(s), await this.sendBinaryChunks(r);
    }
    async sendRenderResult(e, t) {
      if (!this.dc || this.dc.readyState !== "open") return;
      let n = 0;
      const r = e.map((l) => {
        const c = l.data.byteLength;
        return n += c, {
          type: l.type,
          timestamp: l.timestamp,
          duration: l.duration,
          size: c,
          decoderConfig: l.decoderConfig
        };
      });
      console.log(`[RTC] Sending Render Result: ${n} bytes, ${e.length} chunks`), this.sendData({
        type: "RENDER_RESULT",
        startFrame: t,
        totalBytes: n,
        chunksMeta: r
      });
      const s = new Uint8Array(n);
      let o = 0;
      for (const l of e) s.set(new Uint8Array(l.data), o), o += l.data.byteLength;
      await this.sendBinaryChunks(s);
    }
    async sendBinaryChunks(e) {
      let n = 0;
      const r = () => new Promise((s) => {
        const o = setInterval(() => {
          (!this.dc || this.dc.bufferedAmount < 65536) && (clearInterval(o), s());
        }, 5);
      });
      for (; n < e.byteLength; ) {
        this.dc && this.dc.bufferedAmount > 256 * 1024 && await r();
        const s = Math.min(n + 16384, e.byteLength);
        if (this.dc) try {
          this.dc.send(e.subarray(n, s));
        } catch {
        }
        n = s, n % (16384 * 5) === 0 && await new Promise((o) => setTimeout(o, 0));
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
          const s = this.receiveBuffer.slice(n, n + r.size);
          t.push({
            type: r.type,
            timestamp: r.timestamp,
            duration: r.duration,
            data: s.buffer,
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
  class ae {
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
      this.ws = new WebSocket(`${_.signalingServerUrl}?token=${t}`), this.ws.onopen = () => {
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
          const t = new H(e.workerId, (n) => this.sendSignal(n));
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
          e.fromId && (this.hostClient = new H(e.fromId, (t) => this.sendSignal(t)), await this.hostClient.handleOffer(e.sdp), (_a = this.onStatusChange) == null ? void 0 : _a.call(this, "Connected to Host!"), (_b = this.onHostConnected) == null ? void 0 : _b.call(this), this.hostClient.onDataChannelOpen = () => {
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
      const r = Array.from(this.workers.values()).map((s) => s.sendScene(e, t, n));
      await Promise.all(r);
    }
    async sendSceneToWorker(e, t, n, r) {
      const s = this.workers.get(e);
      s && await s.sendScene(t, n, r);
    }
    async sendRenderRequest(e, t, n, r) {
      const s = this.workers.get(e);
      s && await s.sendRenderRequest(t, n, r);
    }
  }
  let m = false, w = null, k = null, x = null, R = [], P = /* @__PURE__ */ new Map(), D = 0, M = 0, L = 0, B = null, y = /* @__PURE__ */ new Map(), C = /* @__PURE__ */ new Map(), W = false, A = null;
  const G = 20, a = new re(), f = new ee(a.canvas), d = new ne(), E = new ie(f, d, a.canvas), p = new ae();
  let S = 0, N = 0, T = 0, O = performance.now();
  const oe = () => {
    const i = parseInt(a.inputDepth.value, 10) || _.defaultDepth, e = parseInt(a.inputSPP.value, 10) || _.defaultSPP;
    f.buildPipeline(i, e);
  }, F = () => {
    const { width: i, height: e } = a.getRenderConfig();
    f.updateScreenSize(i, e), d.hasWorld && (d.updateCamera(i, e), f.updateSceneUniforms(d.cameraData, 0, d.lightCount)), f.recreateBindGroup(), f.resetAccumulation(), S = 0, N = 0;
  }, U = async (i, e = true) => {
    m = false, console.log(`Loading Scene: ${i}...`);
    let t, n;
    i === "viewer" && w && (k === "obj" ? t = w : k === "glb" && (n = new Uint8Array(w).slice(0))), await d.loadScene(i, t, n), d.printStats(), await f.loadTexturesFromWorld(d), await le(), F(), a.updateAnimList(d.getAnimationList()), e && (m = true, a.updateRenderButton(true));
  }, le = async () => {
    f.updateCombinedGeometry(d.vertices, d.normals, d.uvs), f.updateCombinedBVH(d.tlas, d.blas), f.updateBuffer("topology", d.mesh_topology), f.updateBuffer("instance", d.instances), f.updateBuffer("lights", d.lights), f.updateSceneUniforms(d.cameraData, 0, d.lightCount);
  }, I = () => {
    if (E.recording || (requestAnimationFrame(I), !m || !d.hasWorld)) return;
    let i = parseInt(a.inputUpdateInterval.value, 10) || 0;
    if (i > 0 && S >= i && d.update(N / (i || 1) / 60), d.hasNewData) {
      let t = false;
      t || (t = f.updateCombinedBVH(d.tlas, d.blas)), t || (t = f.updateBuffer("instance", d.instances)), d.hasNewGeometry && (t || (t = f.updateCombinedGeometry(d.vertices, d.normals, d.uvs)), t || (t = f.updateBuffer("topology", d.mesh_topology)), t || (t = f.updateBuffer("lights", d.lights)), d.hasNewGeometry = false), d.updateCamera(a.canvas.width, a.canvas.height), f.updateSceneUniforms(d.cameraData, 0, d.lightCount), t && f.recreateBindGroup(), f.resetAccumulation(), S = 0, d.hasNewData = false;
    }
    S++, T++, N++, f.compute(S), f.present();
    const e = performance.now();
    e - O >= 1e3 && (a.updateStats(T, 1e3 / T, S), T = 0, O = e);
  }, q = async (i) => {
    const e = a.sceneSelect.value, t = e !== "viewer";
    if (!t && (!w || !k)) return;
    const n = a.getRenderConfig(), r = t ? e : void 0, s = t ? "DUMMY" : w, o = t ? "obj" : k;
    n.sceneName = r, i ? (console.log(`Sending scene to specific worker: ${i}`), y.set(i, "loading"), await p.sendSceneToWorker(i, s, o, n)) : (console.log("Broadcasting scene to all workers..."), p.getWorkerIds().forEach((l) => y.set(l, "loading")), await p.broadcastScene(s, o, n));
  }, j = async (i) => {
    if (y.get(i) !== "idle") {
      console.log(`Worker ${i} is ${y.get(i)}, skipping assignment.`);
      return;
    }
    if (R.length === 0) return;
    const e = R.shift();
    e && (y.set(i, "busy"), C.set(i, e), console.log(`Assigning Job ${e.start} - ${e.start + e.count} to ${i}`), await p.sendRenderRequest(i, e.start, e.count, {
      ...B,
      fileType: "obj"
    }));
  }, ce = async () => {
    const i = Array.from(P.keys()).sort((c, h) => c - h), { Muxer: e, ArrayBufferTarget: t } = await z(async () => {
      const { Muxer: c, ArrayBufferTarget: h } = await import("./webm-muxer-MLtUgOCn.js");
      return {
        Muxer: c,
        ArrayBufferTarget: h
      };
    }, []), n = new e({
      target: new t(),
      video: {
        codec: "V_VP9",
        width: B.width,
        height: B.height,
        frameRate: B.fps
      }
    });
    for (const c of i) {
      const h = P.get(c);
      if (h) for (const u of h) n.addVideoChunk(new EncodedVideoChunk({
        type: u.type,
        timestamp: u.timestamp,
        duration: u.duration,
        data: u.data
      }), {
        decoderConfig: u.decoderConfig
      });
    }
    n.finalize();
    const { buffer: r } = n.target, s = new Blob([
      r
    ], {
      type: "video/webm"
    }), o = URL.createObjectURL(s), l = document.createElement("a");
    l.href = o, l.download = `distributed_trace_${Date.now()}.webm`, document.body.appendChild(l), l.click(), document.body.removeChild(l), URL.revokeObjectURL(o), a.setStatus("Finished!");
  }, V = async (i, e, t) => {
    console.log(`[Worker] Starting Render: Frames ${i} - ${i + e}`), a.setStatus(`Remote Rendering: ${i}-${i + e}`), m = false;
    const n = {
      ...t,
      startFrame: i,
      duration: e / t.fps
    };
    try {
      a.setRecordingState(true, `Remote: ${e} f`);
      const r = await E.recordChunks(n, (s, o) => a.setRecordingState(true, `Remote: ${s}/${o}`));
      console.log("Sending Chunks back to Host..."), a.setRecordingState(true, "Uploading..."), await p.sendRenderResult(r, i), a.setRecordingState(false), a.setStatus("Idle");
    } catch (r) {
      console.error("Remote Recording Failed", r), a.setStatus("Recording Failed");
    } finally {
      m = true, requestAnimationFrame(I);
    }
  }, de = async () => {
    if (!A) return;
    const { start: i, count: e, config: t } = A;
    A = null, await V(i, e, t);
  };
  p.onStatusChange = (i) => a.setStatus(`Status: ${i}`);
  p.onWorkerLeft = (i) => {
    console.log(`Worker Left: ${i}`), a.setStatus(`Worker Left: ${i}`), y.delete(i);
    const e = C.get(i);
    e && (console.warn(`Worker ${i} failed job ${e.start}. Re-queueing.`), R.unshift(e), C.delete(i), a.setStatus(`Re-queued Job ${e.start}`));
  };
  p.onWorkerReady = (i) => {
    console.log(`Worker ${i} is READY`), a.setStatus(`Worker ${i} Ready!`), y.set(i, "idle"), x === "host" && R.length > 0 && j(i);
  };
  p.onWorkerJoined = (i) => {
    a.setStatus(`Worker Joined: ${i}`), y.set(i, "idle"), x === "host" && R.length > 0 && q(i);
  };
  p.onRenderRequest = async (i, e, t) => {
    if (console.log(`[Worker] Received Render Request: Frames ${i} - ${i + e}`), W) {
      console.log(`[Worker] Scene loading in progress. Queueing Render Request for ${i}`), A = {
        start: i,
        count: e,
        config: t
      };
      return;
    }
    await V(i, e, t);
  };
  p.onRenderResult = async (i, e, t) => {
    console.log(`[Host] Received ${i.length} chunks for ${e} from ${t}`), P.set(e, i), D++, a.setStatus(`Distributed Progress: ${D} / ${M} jobs`), y.set(t, "idle"), C.delete(t), await j(t), D >= M && (console.log("All jobs complete. Muxing..."), a.setStatus("Muxing..."), await ce());
  };
  p.onSceneReceived = async (i, e) => {
    console.log("Scene received successfully."), W = true, a.setRenderConfig(e), k = e.fileType, e.fileType, w = i, a.sceneSelect.value = e.sceneName || "viewer", await U(e.sceneName || "viewer", false), e.anim !== void 0 && (a.animSelect.value = e.anim.toString(), d.setAnimation(e.anim)), W = false, console.log("Scene Loaded. Sending WORKER_READY."), await p.sendWorkerReady(), de();
  };
  const ue = () => {
    a.onRenderStart = () => {
      m = true;
    }, a.onRenderStop = () => {
      m = false;
    }, a.onSceneSelect = (i) => U(i, false), a.onResolutionChange = F, a.onRecompile = (i, e) => {
      m = false, f.buildPipeline(i, e), f.recreateBindGroup(), f.resetAccumulation(), S = 0, m = true;
    }, a.onFileSelect = async (i) => {
      var _a;
      ((_a = i.name.split(".").pop()) == null ? void 0 : _a.toLowerCase()) === "obj" ? (w = await i.text(), k = "obj") : (w = await i.arrayBuffer(), k = "glb"), a.sceneSelect.value = "viewer", U("viewer", false);
    }, a.onAnimSelect = (i) => d.setAnimation(i), a.onRecordStart = async () => {
      if (!E.recording) if (x === "host") {
        const i = p.getWorkerIds();
        if (B = a.getRenderConfig(), L = Math.ceil(B.fps * B.duration), !confirm(`Distribute recording? (Workers: ${i.length})
Auto Scene Sync enabled.`)) return;
        R = [], P.clear(), D = 0, C.clear();
        for (let e = 0; e < L; e += G) {
          const t = Math.min(G, L - e);
          R.push({
            start: e,
            count: t
          });
        }
        M = R.length, i.forEach((e) => y.set(e, "idle")), a.setStatus(`Distributed Progress: 0 / ${M} jobs (Waiting for workers...)`), i.length > 0 ? (a.setStatus("Syncing Scene to Workers..."), await q()) : console.log("No workers yet. Waiting...");
      } else {
        m = false, a.setRecordingState(true);
        const i = a.getRenderConfig();
        try {
          const e = performance.now();
          await E.record(i, (t, n) => a.setRecordingState(true, `Rec: ${t}/${n} (${Math.round(t / n * 100)}%)`), (t) => {
            const n = document.createElement("a");
            n.href = t, n.download = `raytrace_${Date.now()}.webm`, n.click(), URL.revokeObjectURL(t);
          }), console.log(`Recording took ${performance.now() - e}[ms]`);
        } catch {
          alert("Recording failed.");
        } finally {
          a.setRecordingState(false), m = true, a.updateRenderButton(true), requestAnimationFrame(I);
        }
      }
    }, a.onConnectHost = () => {
      x === "host" ? (p.disconnect(), x = null, a.setConnectionState(null)) : (p.connect("host"), x = "host", a.setConnectionState("host"));
    }, a.onConnectWorker = () => {
      x === "worker" ? (p.disconnect(), x = null, a.setConnectionState(null)) : (p.connect("worker"), x = "worker", a.setConnectionState("worker"));
    }, a.setConnectionState(null);
  };
  async function he() {
    try {
      await f.init(), await d.initWasm();
    } catch (i) {
      alert("Init failed: " + i);
      return;
    }
    ue(), oe(), F(), U("cornell", false), requestAnimationFrame(I);
  }
  he().catch(console.error);
})();
