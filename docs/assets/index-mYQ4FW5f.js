var __defProp = Object.defineProperty;
var __defNormalProp = (obj, key, value) => key in obj ? __defProp(obj, key, { enumerable: true, configurable: true, writable: true, value }) : obj[key] = value;
var __publicField = (obj, key, value) => __defNormalProp(obj, typeof key !== "symbol" ? key + "" : key, value);
(async () => {
  (function() {
    const e = document.createElement("link").relList;
    if (e && e.supports && e.supports("modulepreload")) return;
    for (const r of document.querySelectorAll('link[rel="modulepreload"]')) n(r);
    new MutationObserver((r) => {
      for (const i of r) if (i.type === "childList") for (const s of i.addedNodes) s.tagName === "LINK" && s.rel === "modulepreload" && n(s);
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
  const H = "modulepreload", G = function(o) {
    return "/webgpu-raytracer/" + o;
  }, E = {}, M = function(e, t, n) {
    let r = Promise.resolve();
    if (t && t.length > 0) {
      let l = function(d) {
        return Promise.all(d.map((g) => Promise.resolve(g).then((b) => ({
          status: "fulfilled",
          value: b
        }), (b) => ({
          status: "rejected",
          reason: b
        }))));
      };
      var s = l;
      document.getElementsByTagName("link");
      const a = document.querySelector("meta[property=csp-nonce]"), c = (a == null ? void 0 : a.nonce) || (a == null ? void 0 : a.getAttribute("nonce"));
      r = l(t.map((d) => {
        if (d = G(d), d in E) return;
        E[d] = true;
        const g = d.endsWith(".css"), b = g ? '[rel="stylesheet"]' : "";
        if (document.querySelector(`link[href="${d}"]${b}`)) return;
        const m = document.createElement("link");
        if (m.rel = g ? "stylesheet" : H, g || (m.as = "script"), m.crossOrigin = "", m.href = d, c && m.setAttribute("nonce", c), document.head.appendChild(m), g) return new Promise((U, z) => {
          m.addEventListener("load", U), m.addEventListener("error", () => z(new Error(`Unable to preload CSS for ${d}`)));
        });
      }));
    }
    function i(a) {
      const c = new Event("vite:preloadError", {
        cancelable: true
      });
      if (c.payload = a, window.dispatchEvent(c), !c.defaultPrevented) throw a;
    }
    return r.then((a) => {
      for (const c of a || []) c.status === "rejected" && i(c.reason);
      return e().catch(i);
    });
  }, $ = `// =========================================================
//   WebGPU Ray Tracer (Raytracer.wgsl)
// =========================================================

const PI = 3.141592653589793;
const T_MIN = 0.001;
const T_MAX = 1e30;
const SPATIAL_COUNT = 2u;      // \u4F55\u500B\u306E\u8FD1\u508D\u3092\u63A2\u3059\u304B (2\u301C5\u304F\u3089\u3044)
const SPATIAL_RADIUS = 30.0;   // \u63A2\u7D22\u534A\u5F84 (\u30D4\u30AF\u30BB\u30EB\u5358\u4F4D)
// These are replaced by Pipeline Overrides
override MAX_DEPTH: u32;
override SPP: u32;

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

struct LightRef {
    inst_idx: u32,
    tri_idx: u32
}

struct BVHNode {
    min_b: vec4<f32>, // w: left_first
    max_b: vec4<f32>, // w: tri_count
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

// =========================================================
//   ReSTIR GI Structures
// =========================================================

// \u516B\u9762\u4F53\u30A8\u30F3\u30B3\u30FC\u30C7\u30A3\u30F3\u30B0\u306B\u3088\u308B\u6CD5\u7DDA\u5727\u7E2E (vec3 -> vec2)
fn pack_normal(n: vec3<f32>) -> vec2<f32> {
    let p = n.xy * (1.0 / (abs(n.x) + abs(n.y) + abs(n.z)));
    return select(p, (1.0 - abs(p.yx)) * select(vec2(-1.0), vec2(1.0), p.xy >= vec2(0.0)), n.z < 0.0);
}

fn unpack_normal(p: vec2<f32>) -> vec3<f32> {
    var n = vec3(p, 1.0 - abs(p.x) - abs(p.y));
    let t = saturate(-n.z);
    n.x += select(t, -t, n.x >= 0.0);
    n.y += select(t, -t, n.y >= 0.0);
    return normalize(n);
}

struct GIReservoir {
    // 64 bytes (4 x vec4)
    data0: vec4<f32>, // [dir.x, dir.y, dir.z, dist]
    data1: vec4<f32>, // [radiance.x, radiance.y, radiance.z, w_sum]
    data2: vec4<f32>, // [pos.x, pos.y, pos.z, M]
    data3: vec4<f32>, // [norm_packed.x, norm_packed.y, W, mat_id (bitcast)]
}

fn get_gi_sample_dir(res: GIReservoir) -> vec3<f32> { return res.data0.xyz; }
fn get_gi_sample_dist(res: GIReservoir) -> f32 { return res.data0.w; }
fn get_gi_sample_radiance(res: GIReservoir) -> vec3<f32> { return res.data1.xyz; }
fn get_gi_w_sum(res: GIReservoir) -> f32 { return res.data1.w; }
fn get_gi_creation_pos(res: GIReservoir) -> vec3<f32> { return res.data2.xyz; }
fn get_gi_M(res: GIReservoir) -> f32 { return res.data2.w; }
fn get_gi_creation_normal(res: GIReservoir) -> vec3<f32> { return unpack_normal(res.data3.xy); }
fn get_gi_W(res: GIReservoir) -> f32 { return res.data3.z; }
fn get_gi_creation_mat_id(res: GIReservoir) -> u32 { return bitcast<u32>(res.data3.w); }

fn set_gi_W(res: ptr<function, GIReservoir>, W: f32) { (*res).data3.z = W; }
fn set_gi_M(res: ptr<function, GIReservoir>, M: f32) { (*res).data2.w = M; }
fn set_gi_w_sum(res: ptr<function, GIReservoir>, w_sum: f32) { (*res).data1.w = w_sum; }


// =========================================================
//   Bindings
// =========================================================

@group(0) @binding(1) var<storage, read_write> accumulateBuffer: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> scene: SceneUniforms;

@group(0) @binding(3) var<storage, read> geometry_pos: array<vec4<f32>>;
@group(0) @binding(11) var<storage, read> geometry_norm: array<vec4<f32>>;
@group(0) @binding(12) var<storage, read> geometry_uv: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read> topology: array<MeshTopology>;
@group(0) @binding(5) var<storage, read> nodes: array<BVHNode>; 
@group(0) @binding(6) var<storage, read> instances: array<Instance>;
@group(0) @binding(9) var<storage, read> lights: array<LightRef>;

@group(0) @binding(7) var tex: texture_2d_array<f32>;
@group(0) @binding(8) var smp: sampler;

@group(0) @binding(10) var<storage, read_write> gi_reservoir: array<GIReservoir>;

// =========================================================
//   Buffer Accessors
// =========================================================

fn get_pos(idx: u32) -> vec3<f32> {
    return geometry_pos[idx].xyz;
}

fn get_normal(idx: u32) -> vec3<f32> {
    return geometry_norm[idx].xyz;
}

fn get_uv(idx: u32) -> vec2<f32> {
    return geometry_uv[idx];
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
//   ReSTIR Helpers
// =========================================================

// \u73FE\u5728\u306E\u30D5\u30EC\u30FC\u30E0\u7528\u306E\u30A4\u30F3\u30C7\u30C3\u30AF\u30B9\u3068\u3001\u904E\u53BB\u30D5\u30EC\u30FC\u30E0\u7528\u306E\u30A4\u30F3\u30C7\u30C3\u30AF\u30B9\u3092\u8A08\u7B97\u3059\u308B
fn get_reservoir_offsets(pixel_idx: u32) -> vec2<u32> {
    let page_size = scene.width * scene.height;
    
    // frame_count \u304C\u5076\u6570\u306E\u3068\u304D: Curr=0\u9762, Prev=1\u9762
    // frame_count \u304C\u5947\u6570\u306E\u3068\u304D: Curr=1\u9762, Prev=0\u9762
    let current_page = (scene.frame_count % 2u) * page_size;
    let prev_page = ((scene.frame_count + 1u) % 2u) * page_size;

    return vec2(current_page + pixel_idx, prev_page + pixel_idx);
}

// \u30BF\u30FC\u30B2\u30C3\u30C8\u95A2\u6570 p_hat (\u8F1D\u5EA6\u30D9\u30FC\u30B9 + BSDF\u8FD1\u4F3C)
fn evaluate_p_hat(radiance: vec3<f32>, albedo: vec3<f32>, cos_theta: f32) -> f32 {
    // \u8F1D\u5EA6(Luminance)\u3092\u5B9F\u8CEA\u7684\u306A\u5BC4\u4E0E(radiance * albedo * cos)\u3067\u8A55\u4FA1
    let contribution = radiance * albedo * cos_theta;
    
    // [Firefly\u5BFE\u7B56] \u975E\u5E38\u306B\u9AD8\u3044\u8F1D\u5EA6\u3092\u30AF\u30E9\u30F3\u30D7\u3057\u3066\u3001\u7279\u5B9A\u306E\u30B5\u30F3\u30D7\u30EB\u304C\u30EA\u30B6\u30FC\u30D0\u3092\u652F\u914D\u3057\u7D9A\u3051\u306A\u3044\u3088\u3046\u306B\u3059\u308B
    let clamped_contribution = min(contribution, vec3(50.0));
    return dot(clamped_contribution, vec3(0.2126, 0.7152, 0.0722));
}

// \u30DE\u30FC\u30B8\u7528: \u53CD\u5C04\u7387\u3068\u4F59\u5F26\u3092\u8003\u616E\u3057\u305F\u91CD\u307F\u4ED8\u3051
fn merge_reservoir_refined(
    dest: ptr<function, GIReservoir>,
    src: GIReservoir,
    p_hat_dest: f32, // \u79FB\u52D5\u5148\u306E\u70B9\u3067\u306E\u91CD\u8981\u5EA6
    albedo: vec3<f32>,
    normal: vec3<f32>,
    rng: ptr<function, u32>
) -> bool {
    let M = src.data2.w;
    let weight = p_hat_dest * src.data3.z * M; // RIS weight (src.W is data3.z)

    (*dest).data1.w += weight; // w_sum
    (*dest).data2.w += M;      // M

    if rand_pcg(rng) < (weight / (*dest).data1.w) {
        (*dest).data0 = src.data0;
        (*dest).data1 = vec4(src.data1.xyz, (*dest).data1.w);
        return true;
    }
    return false;
}

// \u4E92\u63DB\u6027\u306E\u305F\u3081\u306E\u65E2\u5B58\u306E\u30EA\u30B6\u30FC\u30D0\u66F4\u65B0 (RIS Update\u7528)
fn update_reservoir(
    res: ptr<function, GIReservoir>,
    dir: vec3<f32>,
    radiance: vec3<f32>,
    dist: f32,
    weight: f32,
    rng: ptr<function, u32>
) -> bool {
    (*res).data1.w += weight; // w_sum
    (*res).data2.w += 1.0;    // M

    if rand_pcg(rng) < (weight / (*res).data1.w) {
        (*res).data0 = vec4(dir, dist);
        (*res).data1 = vec4(radiance, (*res).data1.w);
        return true;
    }
    return false;
}

// 1. Initial Candidate Generation
fn generate_initial_gi_candidate(
    hit_p: vec3<f32>,
    normal: vec3<f32>,
    albedo: vec3<f32>,
    rng: ptr<function, u32>
) -> GIReservoir {
    var res: GIReservoir;
    res.data0 = vec4(0.0);
    res.data1 = vec4(0.0);
    res.data2 = vec4(0.0);
    res.data3 = vec4(0.0);

    let scatter = sample_diffuse(normal, albedo, rng);
    if scatter.pdf <= 0.0 { return res; }

    let bounce_ray = Ray(hit_p + normal * 1e-4, scatter.dir);
    let bounce_hit = intersect_tlas(bounce_ray, T_MIN, T_MAX);

    var Li = vec3(0.0);
    var dist = 0.0;

    if bounce_hit.inst_idx >= 0 {
        dist = bounce_hit.t;
        let b_tri = topology[u32(bounce_hit.tri_idx)];
        let b_inst = instances[u32(bounce_hit.inst_idx)];
        let b_inv = get_inv_transform(b_inst);

        let b_pos0 = get_pos(b_tri.v0);
        let b_pos1 = get_pos(b_tri.v1);
        let b_pos2 = get_pos(b_tri.v2);
        let b_r_local = Ray((b_inv * vec4(bounce_ray.origin, 1.)).xyz, (b_inv * vec4(bounce_ray.direction, 0.)).xyz);
        let b_e1 = b_pos1 - b_pos0; let b_e2 = b_pos2 - b_pos0;
        let b_s = b_r_local.origin - b_pos0;
        let b_h = cross(b_r_local.direction, b_e2);
        let b_f = 1.0 / dot(b_e1, b_h);
        let b_u = b_f * dot(b_s, b_h);
        let b_q = cross(b_s, b_e1);
        let b_v = b_f * dot(b_r_local.direction, b_q);
        let b_w = 1.0 - b_u - b_v;

        let b_uv = get_uv(b_tri.v0) * b_w + get_uv(b_tri.v1) * b_u + get_uv(b_tri.v2) * b_v;
        let b_ln = normalize(get_normal(b_tri.v0) * b_w + get_normal(b_tri.v1) * b_u + get_normal(b_tri.v2) * b_v);
        let b_normal = normalize((vec4(b_ln, 0.0) * b_inv).xyz);
        let b_p = bounce_ray.origin + bounce_ray.direction * bounce_hit.t;

        var b_em = b_tri.data3.rgb;
        if b_tri.data2.w > -0.5 { b_em *= textureSampleLevel(tex, smp, b_uv, i32(b_tri.data2.w), 0.0).rgb; }
        Li = b_em;

        let b_mat_type = bitcast<u32>(b_tri.data0.w);
        if b_mat_type != 3u && b_mat_type != 2u {
            let b_ls = sample_light_source(b_p, rng);
            if b_ls.pdf > 0.0 {
                let b_sr = Ray(b_p + b_normal * 1e-4, b_ls.dir);
                if intersect_tlas(b_sr, T_MIN, b_ls.dist - 2e-4).inst_idx == -1 {
                    var b_alb = b_tri.data0.rgb;
                    if b_tri.data2.x > -0.5 { b_alb *= textureSampleLevel(tex, smp, b_uv, i32(b_tri.data2.x), 0.0).rgb; }
                    Li += eval_diffuse(b_alb) * b_ls.L * max(dot(b_normal, b_ls.dir), 0.0) / b_ls.pdf;
                }
            }
        }
    }
    if length(Li) < 1e-4 { Li = albedo * 0.05; }
    let ph = evaluate_p_hat(Li, albedo, max(dot(normal, scatter.dir), 0.0));
    update_reservoir(&res, scatter.dir, Li, dist, select(0.0, ph / max(scatter.pdf, 1e-3), scatter.pdf > 1e-8), rng);
    return res;
}

// 2. Temporal Reuse
fn apply_temporal_reuse_gi(
    state: ptr<function, GIReservoir>,
    prev_res_idx: u32,
    normal: vec3<f32>,
    albedo: vec3<f32>,
    mat_type: u32,
    rng: ptr<function, u32>
) {
    let pr = gi_reservoir[prev_res_idx];
    if dot(get_gi_creation_normal(pr), normal) > 0.9 && get_gi_creation_mat_id(pr) == mat_type {
        let ph = evaluate_p_hat(get_gi_sample_radiance(pr), albedo, max(dot(normal, get_gi_sample_dir(pr)), 0.0));
        merge_reservoir_refined(state, pr, ph, albedo, normal, rng);
        let m = get_gi_M(*state);
        if m > 20.0 { set_gi_w_sum(state, get_gi_w_sum(*state) * (20.0 / m)); set_gi_M(state, 20.0); }
    }
}

// 3. Spatial Reuse
fn apply_spatial_reuse_gi(
    state: ptr<function, GIReservoir>,
    coord: vec2<u32>,
    normal: vec3<f32>,
    albedo: vec3<f32>,
    mat_type: u32,
    hit_p: vec3<f32>,
    rng: ptr<function, u32>
) {
    for (var i = 0u; i < SPATIAL_COUNT; i++) {
        let rad = sqrt(rand_pcg(rng)) * SPATIAL_RADIUS;
        let ang = 2.0 * PI * rand_pcg(rng);
        let nc = vec2<i32>(vec2<f32>(coord) + vec2(rad * cos(ang), rad * sin(ang)));
        if nc.x >= 0 && nc.x < i32(scene.width) && nc.y >= 0 && nc.y < i32(scene.height) {
            let nr = gi_reservoir[get_reservoir_offsets(u32(nc.y) * scene.width + u32(nc.x)).y];
            if get_gi_creation_mat_id(nr) == mat_type && dot(get_gi_creation_normal(nr), normal) > 0.9 && dot(get_gi_creation_pos(nr) - hit_p, get_gi_creation_pos(nr) - hit_p) < 0.1 {
                let ph = evaluate_p_hat(get_gi_sample_radiance(nr), albedo, max(dot(normal, get_gi_sample_dir(nr)), 0.0));
                merge_reservoir_refined(state, nr, ph, albedo, normal, rng);
            }
        }
    }
    let m = get_gi_M(*state);
    if m > 50.0 { set_gi_w_sum(state, get_gi_w_sum(*state) * (50.0 / m)); set_gi_M(state, 50.0); }
}

// 4. Finalize
fn finalize_gi_reservoir(
    state: ptr<function, GIReservoir>,
    hit_p: vec3<f32>,
    normal: vec3<f32>,
    albedo: vec3<f32>,
    mat_type: u32,
    curr_res_idx: u32
) {
    let ph = evaluate_p_hat(get_gi_sample_radiance(*state), albedo, max(dot(normal, get_gi_sample_dir(*state)), 0.0));
    let m = get_gi_M(*state);
    let w = select(0.0, get_gi_w_sum(*state) / (m * ph + 1e-6), ph > 1e-6);
    set_gi_W(state, min(w, 10.0));
    (*state).data2 = vec4(hit_p, m);
    let pn = pack_normal(normal);
    (*state).data3.x = pn.x; (*state).data3.y = pn.y;
    (*state).data3.w = bitcast<f32>(mat_type);
    gi_reservoir[curr_res_idx] = *state;
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

    let treat_as_specular = roughness < 0.4;

    return ScatterResult(l, pdf, throughput, treat_as_specular);
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

fn sample_light_source(hit_p: vec3<f32>, rng: ptr<function, u32>) -> LightSample {
    let light_count = scene.light_count;
    if light_count == 0u {
        return LightSample(vec3(0.0), vec3(0.0), 0.0, 0.0);
    }

    let light_pick_idx = u32(rand_pcg(rng) * f32(light_count));
    let l_ref = lights[light_pick_idx];

    let tri = topology[l_ref.tri_idx];
    let inst = instances[l_ref.inst_idx];
    let m = get_transform(inst);

    let v0 = (m * vec4(get_pos(tri.v0), 1.0)).xyz;
    let v1 = (m * vec4(get_pos(tri.v1), 1.0)).xyz;
    let v2 = (m * vec4(get_pos(tri.v2), 1.0)).xyz;

    let r1 = rand_pcg(rng);
    let r2 = rand_pcg(rng);
    let sqrt_r1 = sqrt(r1);
    let u = 1.0 - sqrt_r1;
    let v = r2 * sqrt_r1;
    let w = 1.0 - u - v;

    let p = v0 * u + v1 * v + v2 * w;
    let edge1 = v1 - v0;
    let edge2 = v2 - v0;
    let n_raw = normalize(cross(edge1, edge2));
    let area = length(cross(edge1, edge2)) * 0.5;

    let l_dir = p - hit_p;
    let dist_sq = dot(l_dir, l_dir);
    let dist = sqrt(dist_sq);
    let unit_l = l_dir / dist;

    let cos_theta_l = max(dot(n_raw, -unit_l), 0.0);
    if cos_theta_l < 1e-6 {
        return LightSample(vec3(0.0), vec3(0.0), 0.0, 0.0);
    }

    // Albedo if light
    let uv0 = get_uv(tri.v0);
    let uv1 = get_uv(tri.v1);
    let uv2 = get_uv(tri.v2);
    let tex_uv = uv0 * u + uv1 * v + uv2 * w;
    var L = tri.data0.rgb;
    let base_tex = tri.data2.x;
    if base_tex > -0.5 {
        L *= textureSampleLevel(tex, smp, tex_uv, i32(base_tex), 0.0).rgb;
    }

    let pdf = (dist_sq / (cos_theta_l * area)) / f32(light_count);

    return LightSample(L, unit_l, dist, pdf);
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
    let t1 = (min_b - origin) * inv_d;
    let t2 = (max_b - origin) * inv_d;
    let t_near = min(t1, t2);
    let t_far = max(t1, t2);
    let tm_near = max(t_min, max(t_near.x, max(t_near.y, t_near.z)));
    let tm_far = min(t_max, min(t_far.x, min(t_far.y, t_far.z)));
    return select(T_MAX, tm_near, tm_near <= tm_far);
}

fn hit_triangle_raw(v0: vec3<f32>, v1: vec3<f32>, v2: vec3<f32>, r: Ray, t_min: f32, t_max: f32) -> f32 {
    let e1 = v1 - v0; let e2 = v2 - v0;
    let h = cross(r.direction, e2); let a = dot(e1, h);
    if abs(a) < 1e-6 { return -1.0; } // Increased epsilon
    let f = 1.0 / a; let s = r.origin - v0; let u = f * dot(s, h);
    if u < 0.0 || u > 1.0 { return -1.0; }
    let q = cross(s, e1); let v = f * dot(r.direction, q);
    if v < 0.0 || u + v > 1.0 { return -1.0; }
    let t = f * dot(e2, q);
    return select(-1.0, t, t > t_min && t < t_max);
}

fn intersect_blas(r: Ray, t_min: f32, t_max: f32, node_start_idx: u32) -> vec2<f32> {
    var closest_t = t_max;
    var hit_idx = -1.0;
    let inv_d = 1.0 / r.direction;
    var stack: array<u32, 32>; // Reduced stack size
    var stackptr = 0u;

    if intersect_aabb(nodes[node_start_idx].min_b.xyz, nodes[node_start_idx].max_b.xyz, r.origin, inv_d, t_min, closest_t) < T_MAX {
        stack[0] = node_start_idx; stackptr = 1u;
    }

    while stackptr > 0u {
        stackptr--;
        let idx = stack[stackptr];
        let node = nodes[idx];
        let count = u32(node.max_b.w);

        if count > 0u {
            let first = u32(node.min_b.w);
            for (var i = 0u; i < count; i++) {
                let tri_id = first + i;
                let tr = topology[tri_id];
                let t = hit_triangle_raw(get_pos(tr.v0), get_pos(tr.v1), get_pos(tr.v2), r, t_min, closest_t);
                if t > 0.0 { closest_t = t; hit_idx = f32(tri_id); }
            }
        } else {
            let l = u32(node.min_b.w) + node_start_idx;
            let r_idx = l + 1u;
            let dl = intersect_aabb(nodes[l].min_b.xyz, nodes[l].max_b.xyz, r.origin, inv_d, t_min, closest_t);
            let dr = intersect_aabb(nodes[r_idx].min_b.xyz, nodes[r_idx].max_b.xyz, r.origin, inv_d, t_min, closest_t);

            if dl < T_MAX && dr < T_MAX {
                if dl < dr { stack[stackptr] = r_idx; stack[stackptr + 1u] = l; } else { stack[stackptr] = l; stack[stackptr + 1u] = r_idx; }
                stackptr += 2u;
            } else if dl < T_MAX { stack[stackptr] = l; stackptr++; } else if dr < T_MAX { stack[stackptr] = r_idx; stackptr++; }
        }
    }
    return vec2<f32>(closest_t, hit_idx);
}

fn intersect_tlas(r: Ray, t_min: f32, t_max: f32) -> HitResult {
    var res: HitResult; res.t = t_max; res.tri_idx = -1.0; res.inst_idx = -1;
    if scene.blas_base_idx == 0u { return res; }

    let inv_d = 1.0 / r.direction;
    var stack: array<u32, 16>; // TLAS is usually shallow
    var stackptr = 0u;

    if intersect_aabb(nodes[0].min_b.xyz, nodes[0].max_b.xyz, r.origin, inv_d, t_min, res.t) < T_MAX {
        stack[0] = 0u; stackptr = 1u;
    }

    while stackptr > 0u {
        stackptr--;
        let node = nodes[stack[stackptr]];

        if node.max_b.w > 0.5 { // Leaf
            let inst_idx = u32(node.min_b.w);
            let inst = instances[inst_idx];
            let r_local = Ray((get_inv_transform(inst) * vec4(r.origin, 1.0)).xyz, (get_inv_transform(inst) * vec4(r.direction, 0.0)).xyz);
            let blas = intersect_blas(r_local, t_min, res.t, scene.blas_base_idx + inst.blas_node_offset);
            if blas.y > -0.5 { res.t = blas.x; res.tri_idx = blas.y; res.inst_idx = i32(inst_idx); }
        } else {
            let l = u32(node.min_b.w);
            let r_idx = l + 1u;
            let dl = intersect_aabb(nodes[l].min_b.xyz, nodes[l].max_b.xyz, r.origin, inv_d, t_min, res.t);
            let dr = intersect_aabb(nodes[r_idx].min_b.xyz, nodes[r_idx].max_b.xyz, r.origin, inv_d, t_min, res.t);
            if dl < T_MAX && dr < T_MAX {
                if dl < dr { stack[stackptr] = r_idx; stack[stackptr + 1u] = l; } else { stack[stackptr] = l; stack[stackptr + 1u] = r_idx; }
                stackptr += 2u;
            } else if dl < T_MAX { stack[stackptr] = l; stackptr++; } else if dr < T_MAX { stack[stackptr] = r_idx; stackptr++; }
        }
    }
    return res;
}

// =========================================================
//   Main Path Tracer Loop
// =========================================================

fn ray_color(r_in: Ray, rng: ptr<function, u32>, coord: vec2<u32>) -> vec3<f32> {
    var ray = r_in;
    var throughput = vec3(1.0);
    var radiance = vec3(0.0);

    let pixel_idx = coord.y * scene.width + coord.x;
    let offsets = get_reservoir_offsets(pixel_idx);
    let curr_res_idx = offsets.x;
    let prev_res_idx = offsets.y;

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
        let f_val = 1.0 / dot(e1, h_val);
        let u_bar = f_val * dot(s, h_val);
        let q = cross(s, e1);
        let v_bar = f_val * dot(r_local.direction, q);
        let w_bar = 1.0 - u_bar - v_bar;

        let hit_p = ray.origin + ray.direction * hit.t;
        let uv0 = get_uv(tri.v0);
        let uv1 = get_uv(tri.v1);
        let uv2 = get_uv(tri.v2);
        let tex_uv = uv0 * w_bar + uv1 * u_bar + uv2 * v_bar;

        let n0 = get_normal(tri.v0);
        let n1 = get_normal(tri.v1);
        let n2 = get_normal(tri.v2);
        let ln = normalize(n0 * w_bar + n1 * u_bar + n2 * v_bar);
        var normal = normalize((vec4(ln, 0.0) * inv).xyz);

        var albedo = tri.data0.rgb;
        if tri.data2.x > -0.5 { albedo *= textureSampleLevel(tex, smp, tex_uv, i32(tri.data2.x), 0.0).rgb; }

        if tri.data2.z > -0.5 {
            let n_map = textureSampleLevel(tex, smp, tex_uv, i32(tri.data2.z), 0.0).rgb * 2.0 - 1.0;
            let T = normalize(e1);
            let B = normalize(cross(ln, T));
            let ln_mapped = normalize(T * n_map.x + B * n_map.y + ln * n_map.z);
            normal = normalize((vec4(ln_mapped, 0.0) * inv).xyz);
        }

        normal = select(-normal, normal, dot(ray.direction, normal) < 0.0);

        var metallic = tri.data1.x;
        var roughness = tri.data1.y;
        if tri.data2.y > -0.5 {
            let mr = textureSampleLevel(tex, smp, tex_uv, i32(tri.data2.y), 0.0).rgb;
            metallic *= mr.b; roughness *= mr.g;
        }
        roughness = max(roughness, 0.005);

        var emissive = tri.data3.rgb;
        if tri.data2.w > -0.5 { emissive *= textureSampleLevel(tex, smp, tex_uv, i32(tri.data2.w), 0.0).rgb; }

        if mat_type == 3u || length(emissive) > 1e-4 {
            let em_val = select(emissive, albedo, mat_type == 3u);
            if specular_bounce { radiance += throughput * em_val; } else { radiance += throughput * em_val * power_heuristic(prev_bsdf_pdf, get_light_pdf(ray.origin, tri_idx, inst_idx, hit.t, ray.direction)); }
            if mat_type == 3u { break; }
        }

        let f0 = mix(vec3(0.04), albedo, metallic);
        let is_specular = (mat_type == 2u) || (metallic > 0.9 && roughness < 0.1);
        let is_shiny_metallic = (mat_type == 1u) && (metallic > 0.1) && (roughness < 0.4);
        let use_restir = (mat_type == 0u) || (!is_shiny_metallic && !is_specular && mat_type != 2u);

        // ReSTIR GI (Indirect Lighting)
        if specular_bounce && use_restir {
            var state = generate_initial_gi_candidate(hit_p, normal, albedo, rng);
            apply_temporal_reuse_gi(&state, prev_res_idx, normal, albedo, mat_type, rng);
            apply_spatial_reuse_gi(&state, coord, normal, albedo, mat_type, hit_p, rng);
            finalize_gi_reservoir(&state, hit_p, normal, albedo, mat_type, curr_res_idx);

            radiance += throughput * get_gi_sample_radiance(state) * eval_diffuse(albedo) * max(dot(normal, get_gi_sample_dir(state)), 0.0) * get_gi_W(state);
        }

        // Direct Light (NEE)
        if !is_specular && mat_type != 2u {
            let light_s = sample_light_source(hit_p, rng);
            if light_s.pdf > 0.0 {
                if intersect_tlas(Ray(hit_p + normal * 1e-4, light_s.dir), T_MIN, light_s.dist - 2e-4).inst_idx == -1 {
                    var bsdf_val = vec3(0.0); var bsdf_pdf_val = 0.0;
                    if mat_type == 0u { bsdf_val = eval_diffuse(albedo); bsdf_pdf_val = max(dot(normal, light_s.dir), 0.0) / PI; } else if mat_type == 1u {
                        bsdf_val = eval_ggx(normal, -ray.direction, light_s.dir, roughness, f0);
                        let H = normalize(-ray.direction + light_s.dir);
                        bsdf_pdf_val = (ggx_d(dot(normal, H), roughness * roughness) * max(dot(normal, H), 0.0)) / (4.0 * max(dot(-ray.direction, H), 0.0));
                    }
                    if bsdf_pdf_val > 0.0 { radiance += throughput * bsdf_val * light_s.L * power_heuristic(light_s.pdf, bsdf_pdf_val) * max(dot(normal, light_s.dir), 0.0) / light_s.pdf; }
                }
            }
        }

        if specular_bounce && use_restir { break; }

        var scatter: ScatterResult;
        if mat_type == 0u { scatter = sample_diffuse(normal, albedo, rng); } else if mat_type == 1u { scatter = sample_ggx(normal, -ray.direction, roughness, f0, rng); } else { scatter = sample_dielectric(ray.direction, normal, tri.data1.z, albedo, rng); }

        if scatter.pdf <= 0.0 || length(scatter.throughput) <= 0.0 { break; }
        throughput *= scatter.throughput;
        ray = Ray(hit_p + normal * 1e-4, scatter.dir);
        prev_bsdf_pdf = scatter.pdf;
        specular_bounce = scatter.is_specular;

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
        col += ray_color(Ray(scene.camera.origin.xyz + off, d), &rng, id.xy);
    }
    col /= f32(SPP);
    
    // Proper accumulation
    var acc_val = vec4<f32>(col, 1.0);
    if scene.frame_count > 1u {
        acc_val = accumulateBuffer[p_idx] + vec4<f32>(col, 1.0);
    }
    accumulateBuffer[p_idx] = acc_val;
}
`, N = `// =========================================================
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
    return clamp((color * (a * color + vec3<f32>(b))) / (color * (c * color + vec3<f32>(d)) + vec3<f32>(e)), vec3<f32>(0.0), vec3<f32>(1.0));
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
    return clamp(center, vec3<f32>(0.0), max_nb * threshold + vec3<f32>(0.1));
}

// Bilinear sampling for un-jittering
fn get_radiance_bilinear(uv: vec2<f32>) -> vec3<f32> {
    let dims = vec2<f32>(f32(scene.width), f32(scene.height));
    let f_coord = uv * dims - vec2<f32>(0.5);
    let i_coord = vec2<i32>(i32(floor(f_coord.x)), i32(floor(f_coord.y)));
    let f = f_coord - vec2<f32>(floor(f_coord.x), floor(f_coord.y));

    let c00 = get_radiance_clean(i_coord + vec2<i32>(0, 0));
    let c10 = get_radiance_clean(i_coord + vec2<i32>(1, 0));
    let c01 = get_radiance_clean(i_coord + vec2<i32>(0, 1));
    let c11 = get_radiance_clean(i_coord + vec2<i32>(1, 1));

    return mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);
}

// \u2605 Enhanced un-jittering: only fully active when frame_count is low.
// As accumulation progresses, the buffer naturally centers itself.
fn get_radiance_nearest(coord: vec2<i32>) -> vec3<f32> {
    if scene.frame_count > 16u {
        return get_radiance_clean(coord);
    }

    let dims = vec2<f32>(f32(scene.width), f32(scene.height));
    let uv = (vec2<f32>(f32(coord.x), f32(coord.y)) + vec2<f32>(0.5)) / dims;
    
    // Fade out un-jittering as accumulation averages out the jitter
    let weight = clamp(1.0 - f32(scene.frame_count - 1u) / 15.0, 0.0, 1.0);
    return get_radiance_bilinear(uv - scene.jitter * weight);
}

fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= scene.width || id.y >= scene.height { return; }

    let dims = vec2<f32>(f32(scene.width), f32(scene.height));
    let uv = (vec2<f32>(f32(id.x), f32(id.y)) + vec2<f32>(0.5)) / dims;

    // 1. Un-jittered Current Frame Radiance (Now cleaned internally)
    let center_color = get_radiance_nearest(vec2<i32>(i32(id.x), i32(id.y)));

    // 2. Bilateral Filter
    let SIGMA_S = 0.5;
    let SIGMA_R = 0.1;
    let RADIUS = 1;

    var filtered_sum = vec3(0.0);
    var total_weight = 0.0;
    for (var dy = -RADIUS; dy <= RADIUS; dy++) {
        for (var dx = -RADIUS; dx <= RADIUS; dx++) {
            let neighbor_pos = vec2<i32>(i32(id.x), i32(id.y)) + vec2<i32>(dx, dy);
            let neighbor_color = get_radiance_nearest(neighbor_pos);

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
    var m1 = vec3<f32>(0.0);
    var m2 = vec3<f32>(0.0);
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let neighbor_pos = vec2<i32>(i32(id.x), i32(id.y)) + vec2<i32>(dx, dy);
            let neighbor_color = get_radiance_nearest(neighbor_pos);
            m1 += neighbor_color;
            m2 += neighbor_color * neighbor_color;
        }
    }
    let mean = m1 / 9.0;
    let stddev = sqrt(max(m2 / 9.0 - mean * mean, vec3<f32>(0.0)));
    
    // Adaptive clamping
    var k = 1.0; // Tighter clamping for better stability during animation
    if scene.frame_count > 16u { k = 60.0; } // Effectively disable clamping when static for full convergence
    let clamped_history = clamp(samples_history, mean - stddev * k, mean + stddev * k);
 
    // Adaptive alpha for progressive refinement
    var alpha = 1.0 / f32(scene.frame_count);
    if scene.frame_count == 1u {
        alpha = 0.1; // Enable TAA even on first frame after update to hide jitter
    }
    alpha = max(alpha, 0.0001); // Even deeper convergence (10000 frames)

    let final_hdr = mix(clamped_history, denoised_hdr, alpha);

    // Store un-jittered result
    textureStore(historyOutput, vec2<i32>(i32(id.x), i32(id.y)), vec4<f32>(final_hdr, 1.0));

    // 4. Output
    let mapped = aces_tone_mapping(final_hdr);
    let edge_detect = center_color - denoised_hdr;
    let sharpened = mapped + aces_tone_mapping(edge_detect) * 0.3;

    let ldr_out = pow(clamp(sharpened, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / 2.2));
    textureStore(outputTex, vec2<i32>(i32(id.x), i32(id.y)), vec4<f32>(ldr_out, 1.0));
}`;
  class O {
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
      __publicField(this, "normOffset", 0);
      __publicField(this, "uvOffset", 0);
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
      const n = this.device.createShaderModule({
        label: "RayTracing",
        code: $
      });
      this.pipeline = this.device.createComputePipeline({
        label: "Main Pipeline",
        layout: "auto",
        compute: {
          module: n,
          entryPoint: "main",
          constants: {
            MAX_DEPTH: e,
            SPP: t
          }
        }
      }), this.bindGroupLayout = this.pipeline.getBindGroupLayout(0);
      const r = this.device.createShaderModule({
        label: "PostProcess",
        code: N
      });
      this.postprocessPipeline = this.device.createComputePipeline({
        label: "PostProcess Pipeline",
        layout: "auto",
        compute: {
          module: r,
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
      for (let i = 0; i < 2; i++) this.historyTextures[i] && this.historyTextures[i].destroy(), this.historyTextures[i] = this.device.createTexture({
        size: [
          e,
          t
        ],
        format: "rgba16float",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST
      }), this.historyTextureViews[i] = this.historyTextures[i].createView();
      const r = e * t * 64 * 2;
      this.reservoirBuffer && this.reservoirBuffer.destroy(), this.reservoirBuffer = this.device.createBuffer({
        label: "GI Reservoir Buffer (Double)",
        size: r,
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
          const s = new Blob([
            i
          ]), a = await createImageBitmap(s, {
            resizeWidth: 1024,
            resizeHeight: 1024
          });
          n.push(a);
        } catch (s) {
          console.warn(`Failed tex ${r}`, s), n.push(await this.createFallbackBitmap());
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
      await this.device.queue.onSubmittedWorkDone();
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
      let r = false, i;
      return e === "topology" ? ((!this.topologyBuffer || this.topologyBuffer.size < n) && (r = true), this.topologyBuffer = this.ensureBuffer(this.topologyBuffer, n, "TopologyBuffer"), i = this.topologyBuffer) : e === "instance" ? ((!this.instanceBuffer || this.instanceBuffer.size < n) && (r = true), this.instanceBuffer = this.ensureBuffer(this.instanceBuffer, n, "InstanceBuffer"), i = this.instanceBuffer) : ((!this.lightsBuffer || this.lightsBuffer.size < n) && (r = true), this.lightsBuffer = this.ensureBuffer(this.lightsBuffer, n, "LightsBuffer"), i = this.lightsBuffer), this.device.queue.writeBuffer(i, 0, t, 0, t.length), r;
    }
    updateCombinedGeometry(e, t, n) {
      const i = e.byteLength;
      this.normOffset = Math.ceil(i / 256) * 256;
      const s = t.byteLength;
      this.uvOffset = Math.ceil((this.normOffset + s) / 256) * 256;
      const a = this.uvOffset + n.byteLength;
      let c = false;
      (!this.geometryBuffer || this.geometryBuffer.size < a) && (c = true);
      const l = e.length / 4;
      return this.vertexCount = l, this.geometryBuffer = this.ensureBuffer(this.geometryBuffer, a, "GeometryBuffer"), !(n.length >= l * 2) && l > 0 && console.warn(`UV buffer mismatch: V=${l}, UV=${n.length / 2}. Filling 0.`), this.device.queue.writeBuffer(this.geometryBuffer, 0, e), this.device.queue.writeBuffer(this.geometryBuffer, this.normOffset, t), this.device.queue.writeBuffer(this.geometryBuffer, this.uvOffset, n), c;
    }
    updateCombinedBVH(e, t) {
      const n = e.byteLength, r = t.byteLength, i = n + r;
      let s = false;
      return (!this.nodesBuffer || this.nodesBuffer.size < i) && (s = true), this.nodesBuffer = this.ensureBuffer(this.nodesBuffer, i, "NodesBuffer"), this.device.queue.writeBuffer(this.nodesBuffer, 0, e), this.device.queue.writeBuffer(this.nodesBuffer, n, t), this.blasOffset = e.length / 8, s;
    }
    updateSceneUniforms(e, t, n) {
      if (this.lightCount = n, !this.sceneUniformBuffer) return;
      const r = (c, l) => {
        let d = 1, g = 0;
        for (; c > 0; ) d = d / l, g = g + d * (c % l), c = Math.floor(c / l);
        return g;
      }, i = r(t % 16 + 1, 2) - 0.5, s = r(t % 16 + 1, 3) - 0.5;
      this.jitter = {
        x: i / this.canvas.width,
        y: s / this.canvas.height
      }, this.device.queue.writeBuffer(this.sceneUniformBuffer, 0, e), this.device.queue.writeBuffer(this.sceneUniformBuffer, 96, this.prevCameraData), this.uniformMixedData[0] = t, this.uniformMixedData[1] = this.blasOffset, this.uniformMixedData[2] = this.vertexCount, this.uniformMixedData[3] = this.seed, this.uniformMixedData[4] = n, this.uniformMixedData[5] = this.canvas.width, this.uniformMixedData[6] = this.canvas.height, this.uniformMixedData[7] = 0;
      const a = new Float32Array(this.uniformMixedData.buffer);
      a[8] = this.jitter.x, a[9] = this.jitter.y, this.device.queue.writeBuffer(this.sceneUniformBuffer, 192, this.uniformMixedData), this.prevCameraData.set(e);
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
              buffer: this.geometryBuffer,
              offset: 0,
              size: this.vertexCount * 16
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
          },
          {
            binding: 11,
            resource: {
              buffer: this.geometryBuffer,
              offset: this.normOffset,
              size: this.vertexCount * 16
            }
          },
          {
            binding: 12,
            resource: {
              buffer: this.geometryBuffer,
              offset: this.uvOffset,
              size: this.vertexCount * 8
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
      const t = (d, g) => {
        let b = 1, m = 0;
        for (; d > 0; ) b = b / g, m = m + b * (d % g), d = Math.floor(d / g);
        return m;
      }, n = t(this.totalFrames % 16 + 1, 2) - 0.5, r = t(this.totalFrames % 16 + 1, 3) - 0.5;
      this.prevJitter.x = this.jitter.x, this.prevJitter.y = this.jitter.y, this.jitter = {
        x: n / this.canvas.width,
        y: r / this.canvas.height
      }, this.uniformMixedData[0] = e, this.uniformMixedData[1] = this.blasOffset, this.uniformMixedData[2] = this.vertexCount, this.uniformMixedData[3] = this.seed, this.uniformMixedData[4] = this.lightCount, this.uniformMixedData[5] = this.canvas.width, this.uniformMixedData[6] = this.canvas.height, this.uniformMixedData[7] = 0;
      const i = new Float32Array(this.uniformMixedData.buffer);
      i[8] = this.jitter.x, i[9] = this.jitter.y, i[10] = this.prevJitter.x, i[11] = this.prevJitter.y, this.device.queue.writeBuffer(this.sceneUniformBuffer, 192, this.uniformMixedData);
      const s = Math.ceil(this.canvas.width / 8), a = Math.ceil(this.canvas.height / 8), c = this.device.createCommandEncoder(), l = c.beginComputePass();
      l.setPipeline(this.pipeline), l.setBindGroup(0, this.bindGroup), l.dispatchWorkgroups(s, a), l.end(), this.device.queue.submit([
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
  function F(o) {
    return new Worker("/webgpu-raytracer/assets/wasm-worker-D4DQfpKg.js", {
      name: o == null ? void 0 : o.name
    });
  }
  class q {
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
      this.worker = new F(), this.worker.onmessage = this.handleMessage.bind(this);
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
  class j {
    constructor(e, t, n) {
      __publicField(this, "isRecording", false);
      __publicField(this, "renderer");
      __publicField(this, "worldBridge");
      __publicField(this, "canvas");
      __publicField(this, "currentBatchSize", 0);
      this.renderer = e, this.worldBridge = t, this.canvas = n;
    }
    get recording() {
      return this.isRecording;
    }
    async record(e, t, n) {
      if (this.isRecording) return;
      this.isRecording = true;
      const { Muxer: r, ArrayBufferTarget: i } = await M(async () => {
        const { Muxer: l, ArrayBufferTarget: d } = await import("./webm-muxer-MLtUgOCn.js");
        return {
          Muxer: l,
          ArrayBufferTarget: d
        };
      }, []), s = Math.ceil(e.fps * e.duration);
      console.log(`Starting recording: ${s} frames @ ${e.fps}fps (VP9)`);
      const a = new r({
        target: new i(),
        video: {
          codec: "V_VP9",
          width: this.canvas.width,
          height: this.canvas.height,
          frameRate: e.fps
        }
      }), c = new VideoEncoder({
        output: (l, d) => a.addVideoChunk(l, d),
        error: (l) => console.error("VideoEncoder Error:", l)
      });
      c.configure({
        codec: "vp09.00.10.08",
        width: this.canvas.width,
        height: this.canvas.height,
        bitrate: 12e6
      });
      try {
        await this.renderAndEncode(s, e, c, t, e.startFrame || 0), await c.flush(), a.finalize();
        const { buffer: l } = a.target, d = new Blob([
          l
        ], {
          type: "video/webm"
        }), g = URL.createObjectURL(d);
        n(g, d);
      } catch (l) {
        throw console.error("Recording failed:", l), l;
      } finally {
        this.isRecording = false;
      }
    }
    async recordChunks(e, t, n) {
      if (this.isRecording) throw new Error("Already recording");
      this.isRecording = true;
      const r = [], i = Math.ceil(e.fps * e.duration), s = new VideoEncoder({
        output: (a, c) => {
          const l = new Uint8Array(a.byteLength);
          a.copyTo(l), r.push({
            type: a.type,
            timestamp: a.timestamp,
            duration: a.duration,
            data: l.buffer,
            decoderConfig: c == null ? void 0 : c.decoderConfig
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
        return await this.renderAndEncode(i, e, s, t, e.startFrame || 0, n), await s.flush(), r;
      } finally {
        this.isRecording = false;
      }
    }
    async renderAndEncode(e, t, n, r, i = 0, s) {
      if (s == null ? void 0 : s.aborted) throw new Error("Aborted");
      this.currentBatchSize = t.batch;
      const a = i;
      this.worldBridge.update(a / t.fps), await this.worldBridge.waitForNextUpdate();
      for (let c = 0; c < e; c++) {
        if (s == null ? void 0 : s.aborted) throw new Error("Aborted");
        r(c, e), await new Promise((g) => setTimeout(g, 0)), await this.updateSceneBuffers();
        let l = null;
        if (c < e - 1) {
          const g = i + c + 1;
          this.worldBridge.update(g / t.fps), l = this.worldBridge.waitForNextUpdate();
        }
        await this.renderFrame(t.spp), n.encodeQueueSize > 5 && await n.flush();
        const d = new VideoFrame(this.canvas, {
          timestamp: (i + c) * 1e6 / t.fps,
          duration: 1e6 / t.fps
        });
        n.encode(d, {
          keyFrame: c % t.fps === 0
        }), d.close(), l && await l;
      }
    }
    async updateSceneBuffers() {
      let e = false;
      e || (e = this.renderer.updateCombinedBVH(this.worldBridge.tlas, this.worldBridge.blas)), e || (e = this.renderer.updateBuffer("instance", this.worldBridge.instances)), e || (e = this.renderer.updateCombinedGeometry(this.worldBridge.vertices, this.worldBridge.normals, this.worldBridge.uvs)), e || (e = this.renderer.updateBuffer("topology", this.worldBridge.mesh_topology)), e || (e = this.renderer.updateBuffer("lights", this.worldBridge.lights)), this.worldBridge.updateCamera(this.canvas.width, this.canvas.height), this.renderer.updateSceneUniforms(this.worldBridge.cameraData, 0, this.worldBridge.lightCount), e && this.renderer.recreateBindGroup(), this.renderer.resetAccumulation();
    }
    async renderFrame(e) {
      let t = 0;
      for (; t < e; ) {
        const n = Math.min(this.currentBatchSize, e - t), r = performance.now();
        for (let d = 0; d < n; d++) this.renderer.compute(t + d);
        t += n, this.renderer.present(), await this.renderer.device.queue.onSubmittedWorkDone();
        const s = performance.now() - r, a = s > 0 ? 100 / s : 2, c = Math.round(this.currentBatchSize * (0.8 + 0.2 * a)), l = this.currentBatchSize;
        this.currentBatchSize = Math.max(1, Math.min(e, c)), this.currentBatchSize !== l && Math.abs(s - 100) > 20 && console.log(`[Worker] Batch Tuned: ${l} -> ${this.currentBatchSize} (Elapsed: ${s.toFixed(1)}ms, Target: 100ms)`);
      }
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
  }, J = {
    iceServers: [
      {
        urls: "stun:stun.l.google.com:19302"
      }
    ]
  };
  class P {
    constructor(e, t) {
      __publicField(this, "pc");
      __publicField(this, "dc", null);
      __publicField(this, "remoteId");
      __publicField(this, "sendSignal");
      __publicField(this, "transferLock", Promise.resolve());
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
      __publicField(this, "onConnectionFailure", null);
      __publicField(this, "onWorkerStatus", null);
      __publicField(this, "onStopRender", null);
      __publicField(this, "onSceneLoaded", null);
      this.remoteId = e, this.sendSignal = t, this.pc = new RTCPeerConnection(J), this.pc.onconnectionstatechange = () => {
        console.log(`[RTC] State Change: ${this.pc.connectionState}`), (this.pc.connectionState === "failed" || this.pc.connectionState === "disconnected") && this.onConnectionFailure && this.onConnectionFailure();
      }, this.pc.onicecandidate = (n) => {
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
      if (!this.dc || this.dc.readyState !== "open") throw new Error("DataChannel is not open");
      await (this.transferLock = this.transferLock.then(async () => {
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
        await this.sendData(i), await this.sendBinaryChunks(r);
      }).catch((r) => {
        throw console.error("[RTC] sendScene failed:", r), r;
      }));
    }
    async sendRenderResult(e, t) {
      if (!this.dc || this.dc.readyState !== "open") throw new Error("DataChannel is not open");
      await (this.transferLock = this.transferLock.then(async () => {
        let n = 0;
        const r = e.map((a) => {
          const c = a.data.byteLength;
          return n += c, {
            type: a.type,
            timestamp: a.timestamp,
            duration: a.duration,
            size: c,
            decoderConfig: a.decoderConfig
          };
        });
        console.log(`[RTC] Sending Render Result: ${n} bytes, ${e.length} chunks`), await this.sendData({
          type: "RENDER_RESULT",
          startFrame: t,
          totalBytes: n,
          chunksMeta: r
        });
        const i = new Uint8Array(n);
        let s = 0;
        for (const a of e) i.set(new Uint8Array(a.data), s), s += a.data.byteLength;
        await this.sendBinaryChunks(i);
      }).catch((n) => {
        throw console.error("[RTC] sendRenderResult failed:", n), n;
      }));
    }
    async sendBinaryChunks(e) {
      let n = 0;
      const r = () => new Promise((i) => {
        const s = setInterval(() => {
          (!this.dc || this.dc.bufferedAmount < 65536) && (clearInterval(s), i());
        }, 5);
      });
      for (; n < e.byteLength; ) {
        this.dc && this.dc.bufferedAmount > 256 * 1024 && await r();
        const i = Math.min(n + 16384, e.byteLength);
        if (this.dc) try {
          this.dc.send(e.subarray(n, i));
        } catch {
        }
        n = i, n % (16384 * 5) === 0 && await new Promise((s) => setTimeout(s, 0));
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
      var _a, _b, _c, _d, _e, _f;
      e.type === "SCENE_INIT" ? (console.log(`[RTC] Receiving Scene: ${e.config.fileType}, ${e.totalBytes} bytes`), this.sceneMeta = {
        config: e.config,
        totalBytes: e.totalBytes
      }, this.receiveBuffer = new Uint8Array(e.totalBytes), this.receivedBytes = 0) : e.type === "SCENE_ACK" ? (console.log(`[RTC] Scene ACK: ${e.receivedBytes} bytes`), this.onAckReceived && this.onAckReceived(e.receivedBytes)) : e.type === "RENDER_REQUEST" ? (console.log(`[RTC] Render Request: Frame ${e.startFrame}, Count ${e.frameCount}`), (_a = this.onRenderRequest) == null ? void 0 : _a.call(this, e.startFrame, e.frameCount, e.config)) : e.type === "RENDER_RESULT" ? (console.log(`[RTC] Receiving Render Result: ${e.totalBytes} bytes`), this.resultMeta = {
        startFrame: e.startFrame,
        totalBytes: e.totalBytes,
        chunksMeta: e.chunksMeta
      }, this.receiveBuffer = new Uint8Array(e.totalBytes), this.receivedBytes = 0) : e.type === "WORKER_READY" ? (console.log("[RTC] Worker Ready Signal Received"), (_b = this.onWorkerReady) == null ? void 0 : _b.call(this)) : e.type === "WORKER_STATUS" ? (console.log(`[RTC] Worker Status Received: hasScene=${e.hasScene}, job=${(_c = e.currentJob) == null ? void 0 : _c.start}`), (_d = this.onWorkerStatus) == null ? void 0 : _d.call(this, e.hasScene, e.currentJob)) : e.type === "STOP_RENDER" ? (console.log("[RTC] Stop Render Signal Received"), (_e = this.onStopRender) == null ? void 0 : _e.call(this)) : e.type === "SCENE_LOADED" && (console.log("[RTC] Scene Loaded Signal Received"), (_f = this.onSceneLoaded) == null ? void 0 : _f.call(this));
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
    async sendData(e) {
      var _a;
      ((_a = this.dc) == null ? void 0 : _a.readyState) === "open" && (this.dc.bufferedAmount > 1024 * 1024 && await new Promise((t) => {
        const n = setInterval(() => {
          (!this.dc || this.dc.bufferedAmount < 524288) && (clearInterval(n), t());
        }, 10);
      }), this.dc.send(JSON.stringify(e)));
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
    sendWorkerStatus(e, t) {
      this.sendData({
        type: "WORKER_STATUS",
        hasScene: e,
        currentJob: t
      });
    }
    sendStopRender() {
      this.sendData({
        type: "STOP_RENDER"
      });
    }
    sendSceneLoaded() {
      this.sendData({
        type: "SCENE_LOADED"
      });
    }
    close() {
      this.dc && (this.dc.close(), this.dc = null), this.pc && this.pc.close(), console.log(`[RTC] Connection closed: ${this.remoteId}`);
    }
  }
  class V {
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
      __publicField(this, "onWorkerStatus", null);
      __publicField(this, "onStopRender", null);
      __publicField(this, "onSceneLoaded", null);
    }
    connect(e) {
      var _a;
      if (this.ws) return;
      this.myRole = e, (_a = this.onStatusChange) == null ? void 0 : _a.call(this, `Connecting as ${e.toUpperCase()}...`);
      const t = "xWUaLfXQQkHZ9VmF";
      this.ws = new WebSocket(`${_.signalingServerUrl}?token=${t}`), this.ws.onopen = () => {
        var _a2;
        if (console.log("WS Connected"), (_a2 = this.onStatusChange) == null ? void 0 : _a2.call(this, `Waiting for Peer (${e.toUpperCase()})`), e === "worker") {
          const n = sessionStorage.getItem("raytracer_session_id"), r = sessionStorage.getItem("raytracer_session_token");
          this.sendSignal({
            type: "register_worker",
            sessionId: n || void 0,
            sessionToken: r || void 0
          });
        } else this.sendSignal({
          type: "register_host"
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
      if (this.hostClient) await this.hostClient.sendRenderResult(e, t);
      else throw new Error("No Host Connection");
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
          const t = (r) => {
            const i = new P(r, (s) => this.sendSignal(s));
            return this.workers.set(r, i), i.onDataChannelOpen = () => {
              var _a2;
              console.log(`[Host] Open for ${r}`), i.sendData({
                type: "HELLO",
                msg: "Hello from Host!"
              }), (_a2 = this.onWorkerJoined) == null ? void 0 : _a2.call(this, r);
            }, i.onAckReceived = (s) => {
              console.log(`Worker ${r} ACK: ${s}`);
            }, i.onRenderResult = (s, a) => {
              var _a2;
              console.log(`Received Render Result from ${r}: ${s.length} chunks`), (_a2 = this.onRenderResult) == null ? void 0 : _a2.call(this, s, a, r);
            }, i.onWorkerReady = () => {
              var _a2;
              (_a2 = this.onWorkerReady) == null ? void 0 : _a2.call(this, r);
            }, i.onWorkerStatus = (s, a) => {
              var _a2;
              (_a2 = this.onWorkerStatus) == null ? void 0 : _a2.call(this, r, s, a);
            }, i.onStopRender = () => {
              var _a2;
              (_a2 = this.onStopRender) == null ? void 0 : _a2.call(this);
            }, i.onSceneLoaded = () => {
              var _a2;
              (_a2 = this.onSceneLoaded) == null ? void 0 : _a2.call(this, r);
            }, i.onConnectionFailure = () => {
              console.warn(`[Host] Connection failed for ${r}. Retrying...`), i.close(), setTimeout(() => {
                var _a2;
                this.workers.has(r) && (t(r), (_a2 = this.workers.get(r)) == null ? void 0 : _a2.startAsHost());
              }, 2e3);
            }, i;
          };
          await t(e.workerId).startAsHost();
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
    async sendWorkerStatus(e, t) {
      this.hostClient && this.hostClient.sendWorkerStatus(e, t);
    }
    async sendSceneLoaded() {
      this.hostClient && this.hostClient.sendSceneLoaded();
    }
    async handleWorkerMessage(e) {
      var _a;
      switch (e.type) {
        case "session_info":
          console.log(`[Worker] Session Info Received: ${e.sessionId}`), sessionStorage.setItem("raytracer_session_id", e.sessionId), sessionStorage.setItem("raytracer_session_token", e.sessionToken);
          break;
        case "offer":
          e.fromId && await ((r) => {
            var _a2, _b;
            return this.hostClient && this.hostClient.close(), this.hostClient = new P(r, (i) => this.sendSignal(i)), (_a2 = this.onStatusChange) == null ? void 0 : _a2.call(this, "Connected to Host!"), (_b = this.onHostConnected) == null ? void 0 : _b.call(this), this.hostClient.onDataChannelOpen = () => {
              var _a3, _b2;
              (_a3 = this.hostClient) == null ? void 0 : _a3.sendData({
                type: "HELLO",
                msg: "Hello from Worker!"
              }), (_b2 = this.onHostHello) == null ? void 0 : _b2.call(this);
            }, this.hostClient.onSceneReceived = (i, s) => {
              var _a3, _b2;
              (_a3 = this.onSceneReceived) == null ? void 0 : _a3.call(this, i, s);
              const a = typeof i == "string" ? i.length : i.byteLength;
              (_b2 = this.hostClient) == null ? void 0 : _b2.sendAck(a);
            }, this.hostClient.onRenderRequest = (i, s, a) => {
              var _a3;
              (_a3 = this.onRenderRequest) == null ? void 0 : _a3.call(this, i, s, a);
            }, this.hostClient.onStopRender = () => {
              var _a3;
              (_a3 = this.onStopRender) == null ? void 0 : _a3.call(this);
            }, this.hostClient.onSceneLoaded = () => {
              var _a3;
              (_a3 = this.onSceneLoaded) == null ? void 0 : _a3.call(this, "host");
            }, this.hostClient.onConnectionFailure = () => {
              var _a3;
              console.warn(`[Worker] Connection failed for host ${r}.`), this.hostClient && (this.hostClient.close(), this.hostClient = null), (_a3 = this.onStatusChange) == null ? void 0 : _a3.call(this, "Disconnected from Host (Reconnecting...)");
            }, this.hostClient;
          })(e.fromId).handleOffer(e.sdp);
          break;
        case "candidate":
          await ((_a = this.hostClient) == null ? void 0 : _a.handleCandidate(e.candidate));
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
    sendStopRender(e) {
      const t = this.workers.get(e);
      t && t.sendStopRender();
    }
    sendRenderStart() {
      this.sendSignal({
        type: "render_start"
      });
    }
    sendRenderStop() {
      this.sendSignal({
        type: "render_stop"
      });
    }
  }
  class X {
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
        anim: parseInt(this.animSelect.value, 10) || 0,
        maxDepth: parseInt(this.inputDepth.value, 10) || _.defaultDepth,
        shaderSpp: parseInt(this.inputSPP.value, 10) || _.defaultSPP
      };
    }
    setRenderConfig(e) {
      this.inputWidth.value = e.width.toString(), this.inputHeight.value = e.height.toString(), this.inputRecFps.value = e.fps.toString(), this.inputRecDur.value = e.duration.toString(), this.inputRecSpp.value = e.spp.toString(), this.inputRecBatch.value = e.batch.toString(), e.maxDepth !== void 0 && (this.inputDepth.value = e.maxDepth.toString()), e.shaderSpp !== void 0 && (this.inputSPP.value = e.shaderSpp.toString());
    }
  }
  class Y {
    constructor(e, t) {
      __publicField(this, "jobQueue", []);
      __publicField(this, "pendingChunks", /* @__PURE__ */ new Map());
      __publicField(this, "completedJobs", 0);
      __publicField(this, "totalJobs", 0);
      __publicField(this, "totalRenderFrames", 0);
      __publicField(this, "distributedConfig", null);
      __publicField(this, "workerStatus", /* @__PURE__ */ new Map());
      __publicField(this, "activeJobs", /* @__PURE__ */ new Map());
      __publicField(this, "signaling");
      __publicField(this, "ui");
      __publicField(this, "disconnectedWorkers", /* @__PURE__ */ new Map());
      __publicField(this, "GRACE_PERIOD_MS", 3e4);
      this.signaling = e, this.ui = t, this.setupSignaling();
    }
    setupSignaling() {
      this.signaling.onWorkerLeft = (e) => this.onWorkerLeft(e), this.signaling.onWorkerReady = (e) => this.onWorkerReady(e), this.signaling.onWorkerJoined = (e) => this.onWorkerJoined(e), this.signaling.onWorkerStatus = (e, t, n) => this.onWorkerStatus(e, t, n), this.signaling.onSceneLoaded = (e) => this.onSceneLoaded(e), this.signaling.onRenderResult = (e, t, n) => this.onRenderResult(e, t, n);
    }
    async sendSceneHelper(e, t, n) {
      const r = this.ui.sceneSelect.value, i = r !== "viewer";
      if (!i && (!e || !t)) return;
      const s = this.ui.getRenderConfig(), a = i ? r : void 0, c = i ? "DUMMY" : e, l = i ? "obj" : t;
      s.sceneName = a, s.fileType = l, n ? (console.log(`[Host] Sending scene to specific worker: ${n}`), this.workerStatus.set(n, "loading"), await this.signaling.sendSceneToWorker(n, c, l, s)) : (console.log("[Host] Broadcasting scene to all workers..."), this.signaling.getWorkerIds().forEach((d) => this.workerStatus.set(d, "loading")), await this.signaling.broadcastScene(c, l, s));
    }
    async assignJob(e) {
      if (this.workerStatus.get(e) !== "idle" || this.jobQueue.length === 0) return;
      if (!this.distributedConfig) {
        console.warn("[Host] Distributed config is missing. Cannot assign job.");
        return;
      }
      const t = this.jobQueue.shift();
      this.workerStatus.set(e, "busy"), this.activeJobs.set(e, t), console.log(`[Host] Assigning job to ${e}: Frames ${t.start} - ${t.start + t.count}`);
      try {
        await this.signaling.sendRenderRequest(e, t.start, t.count, this.distributedConfig);
      } catch (n) {
        console.error(`[Host] Failed to send job to ${e}, re-queuing`, n), this.jobQueue.push(t), this.workerStatus.set(e, "idle"), this.activeJobs.delete(e), setTimeout(() => this.assignJob(e), 2e3);
      }
    }
    triggerAssignments() {
      for (const [e, t] of this.workerStatus.entries()) t === "idle" && this.assignJob(e);
    }
    onWorkerLeft(e) {
      console.log(`[Host] Worker ${e} left.`);
      const t = this.activeJobs.get(e);
      if (t) {
        console.log(`[Host] Worker ${e} had active job. Starting grace period.`);
        const n = window.setTimeout(() => {
          console.log(`[Host] Grace period expired for ${e}. Re-queuing job.`), this.jobQueue.push(t), this.disconnectedWorkers.delete(e), this.activeJobs.delete(e), this.workerStatus.delete(e), this.triggerAssignments();
        }, this.GRACE_PERIOD_MS);
        this.disconnectedWorkers.set(e, {
          job: t,
          timeoutId: n
        });
      } else this.workerStatus.delete(e), this.activeJobs.delete(e);
    }
    onWorkerReady(e) {
      console.log(`[Host] Worker ${e} is ready (Manual Signal).`);
    }
    onWorkerJoined(e) {
      console.log(`[Host] Worker ${e} joined.`), this.workerStatus.set(e, "loading");
      const t = this.disconnectedWorkers.get(e);
      t && (console.log(`[Host] Worker ${e} re-joined. Resuming job.`), clearTimeout(t.timeoutId), this.activeJobs.set(e, t.job), this.disconnectedWorkers.delete(e));
    }
    async onWorkerStatus(e, t, n) {
      if (console.log(`[Host] Worker ${e} status update: hasScene=${t}`, n), !t) {
        if (this.workerStatus.get(e) === "loading") {
          console.log(`[Host] Worker ${e} has no scene but is already loading. Skipping redundant send.`);
          return;
        }
        return this.workerStatus.get(e) === "busy" && console.warn(`[Host] Worker ${e} reports no scene while host thinks it is busy. Re-syncing.`), console.log(`[Host] Worker ${e} has no scene. Syncing...`), "NEED_SCENE";
      }
      !n && this.workerStatus.get(e) !== "busy" ? this.workerStatus.get(e) === "loading" ? console.log(`[Host] Worker ${e} is still loading scene.`) : (this.workerStatus.set(e, "idle"), await this.assignJob(e)) : n && (this.workerStatus.set(e, "busy"), this.activeJobs.set(e, n));
    }
    async onSceneLoaded(e) {
      if (this.workerStatus.get(e) !== "loading") {
        console.log(`[Host] Ignore redundant SCENE_LOADED from ${e} (Status: ${this.workerStatus.get(e)})`);
        return;
      }
      console.log(`[Host] Worker ${e} loaded the scene.`), this.workerStatus.set(e, "idle"), await this.assignJob(e);
    }
    async onRenderResult(e, t, n) {
      if (this.pendingChunks.has(t)) {
        console.warn(`[Host] Ignore duplicate result for ${t} from ${n}`), this.workerStatus.set(n, "idle"), this.activeJobs.delete(n), await this.assignJob(n);
        return;
      }
      if (console.log(`[Host] Received ${e.length} chunks for ${t} from ${n}`), this.pendingChunks.set(t, e), this.completedJobs++, this.ui.setStatus(`Distributed Progress: ${this.completedJobs} / ${this.totalJobs} jobs`), this.workerStatus.set(n, "idle"), this.activeJobs.delete(n), await this.assignJob(n), this.completedJobs >= this.totalJobs) return console.log("[Host] All jobs complete. Triggering Muxing Callback."), "ALL_COMPLETE";
    }
    async muxAndDownload() {
      const e = Array.from(this.pendingChunks.keys()).sort((l, d) => l - d), { Muxer: t, ArrayBufferTarget: n } = await M(async () => {
        const { Muxer: l, ArrayBufferTarget: d } = await import("./webm-muxer-MLtUgOCn.js");
        return {
          Muxer: l,
          ArrayBufferTarget: d
        };
      }, []), r = new t({
        target: new n(),
        video: {
          codec: "V_VP9",
          width: this.distributedConfig.width,
          height: this.distributedConfig.height,
          frameRate: this.distributedConfig.fps
        }
      });
      for (const l of e) {
        const d = this.pendingChunks.get(l);
        if (d) for (const g of d) r.addVideoChunk(new EncodedVideoChunk({
          type: g.type,
          timestamp: g.timestamp,
          duration: g.duration,
          data: g.data
        }), {
          decoderConfig: g.decoderConfig
        });
      }
      r.finalize();
      const { buffer: i } = r.target, s = new Blob([
        i
      ], {
        type: "video/webm"
      }), a = URL.createObjectURL(s), c = document.createElement("a");
      c.href = a, c.download = `distributed_render_${Date.now()}.webm`, c.click(), URL.revokeObjectURL(a), this.ui.setStatus("Distributed Render Complete."), this.signaling.sendRenderStop();
    }
  }
  class Q {
    constructor(e, t, n, r) {
      __publicField(this, "isSceneLoading", false);
      __publicField(this, "isDistributedSceneLoaded", false);
      __publicField(this, "pendingRenderRequest", null);
      __publicField(this, "currentWorkerJob", null);
      __publicField(this, "signaling");
      __publicField(this, "renderer");
      __publicField(this, "ui");
      __publicField(this, "recorder");
      __publicField(this, "bufferedResults", []);
      __publicField(this, "currentWorkerAbortController", null);
      this.signaling = e, this.renderer = t, this.ui = n, this.recorder = r, this.setupSignaling();
    }
    setupSignaling() {
      this.signaling.onHostHello = () => this.onHostHello(), this.signaling.onRenderRequest = (e, t, n) => this.onRenderRequest(e, t, n), this.signaling.onStopRender = () => this.onStopRender(), this.signaling.onSceneReceived = (e, t) => this.onSceneReceived(e, t);
    }
    async executeWorkerRender(e, t, n) {
      if (this.recorder.isRecording) {
        console.warn("[Worker] Already recording/rendering, skipping request.");
        return;
      }
      this.currentWorkerAbortController && this.currentWorkerAbortController.abort(), this.currentWorkerAbortController = new AbortController();
      const r = this.currentWorkerAbortController.signal;
      if (this.isSceneLoading || !this.isDistributedSceneLoaded) {
        console.log(`[Worker] Scene loading (or not synced) in progress. Queueing Render Request for ${e}`), this.pendingRenderRequest = {
          start: e,
          count: t,
          config: n
        };
        return;
      }
      this.currentWorkerJob = {
        start: e,
        count: t
      }, console.log(`[Worker] Starting Render: Frames ${e} - ${e + t}`), this.ui.setStatus(`Remote Rendering: ${e}-${e + t}`), n.maxDepth !== void 0 && n.shaderSpp !== void 0 && (console.log(`[Worker] Updating Shader Pipeline: Depth=${n.maxDepth}, SPP=${n.shaderSpp}`), this.renderer.buildPipeline(n.maxDepth, n.shaderSpp));
      const i = {
        ...n,
        startFrame: e,
        duration: t / n.fps
      };
      try {
        this.ui.setRecordingState(true, `Remote: ${t} f`);
        const s = await this.recorder.recordChunks(i, (a, c) => this.ui.setRecordingState(true, `Remote: ${a}/${c}`), r);
        console.log(`[Worker] Render Finished for ${e}. Sending results.`), await this.signaling.sendRenderResult(s, e), this.currentWorkerJob = null;
      } catch (s) {
        s.name === "AbortError" ? console.log(`[Worker] Render Aborted for ${e}`) : (console.error("[Worker] Remote Recording Failed", s), this.ui.setStatus("Recording Failed"));
      } finally {
        this.currentWorkerJob = null, this.currentWorkerAbortController = null, this.ui.updateRenderButton(false), this.ui.setRecordingState(false);
      }
    }
    async trySendBufferedResults() {
      if (this.bufferedResults.length === 0) return;
      console.log(`[Worker] Retrying to send ${this.bufferedResults.length} buffered results...`);
      const e = [];
      for (const t of this.bufferedResults) try {
        await this.signaling.sendRenderResult(t.chunks, t.startFrame);
      } catch {
        e.push(t);
      }
      this.bufferedResults = e;
    }
    handlePendingRenderRequest() {
      if (this.pendingRenderRequest) {
        console.log(`[Worker] Processing Pending Render Request: ${this.pendingRenderRequest.start}`);
        const e = this.pendingRenderRequest;
        this.pendingRenderRequest = null, this.executeWorkerRender(e.start, e.count, e.config);
      }
    }
    onHostHello() {
      console.log("[Worker] Host Hello received."), this.signaling.sendWorkerStatus(this.isDistributedSceneLoaded, this.currentWorkerJob || void 0);
    }
    onRenderRequest(e, t, n) {
      this.executeWorkerRender(e, t, n);
    }
    onStopRender() {
      console.log("[Worker] Stop Render received."), this.currentWorkerAbortController && this.currentWorkerAbortController.abort();
    }
    async onSceneReceived(e, t) {
      return console.log("[Worker] Scene received successfully."), this.isSceneLoading = true, this.ui.setRenderConfig(t), t.maxDepth !== void 0 && t.shaderSpp !== void 0 && (console.log(`[Worker] Syncing Shader settings: Depth=${t.maxDepth}, SPP=${t.shaderSpp}`), this.renderer.buildPipeline(t.maxDepth, t.shaderSpp)), {
        data: e,
        config: t
      };
    }
  }
  let y = false, x = null, R = null, w = null;
  const I = 20, u = new X(), f = new O(u.canvas), h = new q(), B = new j(f, h, u.canvas), v = new V(), p = new Y(v, u), k = new Q(v, f, u, B);
  let S = 0, D = 0, C = 0, L = performance.now();
  const K = () => {
    const o = parseInt(u.inputDepth.value, 10) || _.defaultDepth, e = parseInt(u.inputSPP.value, 10) || _.defaultSPP;
    f.buildPipeline(o, e);
  }, A = () => {
    const { width: o, height: e } = u.getRenderConfig();
    f.updateScreenSize(o, e), h.hasWorld && (h.updateCamera(o, e), f.updateSceneUniforms(h.cameraData, 0, h.lightCount)), f.recreateBindGroup(), f.resetAccumulation(), S = 0, D = 0;
  }, T = async (o, e = true) => {
    y = false, console.log(`Loading Scene: ${o}...`);
    let t, n;
    o === "viewer" && x && (R === "obj" ? t = x : R === "glb" && (n = new Uint8Array(x).slice(0))), await h.loadScene(o, t, n), h.printStats(), await f.loadTexturesFromWorld(h), await Z(), A(), u.updateAnimList(h.getAnimationList()), e && (y = true, u.updateRenderButton(true));
  }, Z = async () => {
    f.updateCombinedGeometry(h.vertices, h.normals, h.uvs), f.updateCombinedBVH(h.tlas, h.blas), f.updateBuffer("topology", h.mesh_topology), f.updateBuffer("instance", h.instances), f.updateBuffer("lights", h.lights), f.updateSceneUniforms(h.cameraData, 0, h.lightCount), await f.device.queue.onSubmittedWorkDone();
  }, W = () => {
    if (B.recording || (requestAnimationFrame(W), !y || !h.hasWorld)) return;
    let o = parseInt(u.inputUpdateInterval.value, 10) || 0;
    if (o > 0 && S >= o && h.update(D / (o || 1) / 60), h.hasNewData) {
      let t = false;
      t || (t = f.updateCombinedBVH(h.tlas, h.blas)), t || (t = f.updateBuffer("instance", h.instances)), h.hasNewGeometry && (t || (t = f.updateCombinedGeometry(h.vertices, h.normals, h.uvs)), t || (t = f.updateBuffer("topology", h.mesh_topology)), t || (t = f.updateBuffer("lights", h.lights)), h.hasNewGeometry = false), h.updateCamera(u.canvas.width, u.canvas.height), f.updateSceneUniforms(h.cameraData, 0, h.lightCount), t && f.recreateBindGroup(), f.resetAccumulation(), S = 0, h.hasNewData = false;
    }
    S++, C++, D++, f.compute(S), f.present();
    const e = performance.now();
    e - L >= 1e3 && (u.updateStats(C, 1e3 / C, S), C = 0, L = e);
  };
  v.onStatusChange = (o) => u.setStatus(`Status: ${o}`);
  v.onWorkerStatus = async (o, e, t) => {
    await p.onWorkerStatus(o, e, t) === "NEED_SCENE" && (console.log(`[Host] Worker ${o} needs scene. Syncing...`), await p.sendSceneHelper(x, R, o));
  };
  v.onRenderResult = async (o, e, t) => {
    await p.onRenderResult(o, e, t) === "ALL_COMPLETE" && (console.log("[Host] All jobs complete. Muxing and downloading..."), u.setStatus("Muxing..."), await p.muxAndDownload());
  };
  v.onSceneReceived = async (o, e) => {
    console.log("[Worker] Received Scene from Host."), await k.onSceneReceived(o, e), R = e.fileType, e.fileType, x = o, u.sceneSelect.value = e.sceneName || "viewer", await T(e.sceneName || "viewer", false), e.anim !== void 0 && (u.animSelect.value = e.anim.toString(), h.setAnimation(e.anim)), k.isDistributedSceneLoaded = true, k.isSceneLoading = false, console.log("[Worker] Distributed Scene Loaded. Signaling Host."), await v.sendSceneLoaded(), k.handlePendingRenderRequest();
  };
  const ee = () => {
    u.onRenderStart = () => {
      y = true;
    }, u.onRenderStop = () => {
      y = false;
    }, u.onSceneSelect = (o) => T(o, false), u.onResolutionChange = A, u.onRecompile = (o, e) => {
      y = false, f.buildPipeline(o, e), f.recreateBindGroup(), f.resetAccumulation(), S = 0, y = true;
    }, u.onFileSelect = async (o) => {
      var _a;
      ((_a = o.name.split(".").pop()) == null ? void 0 : _a.toLowerCase()) === "obj" ? (x = await o.text(), R = "obj") : (x = await o.arrayBuffer(), R = "glb"), u.sceneSelect.value = "viewer", T("viewer", false);
    }, u.onAnimSelect = (o) => h.setAnimation(o), u.onRecordStart = async () => {
      if (!B.recording) if (w === "host") {
        const o = v.getWorkerIds();
        p.distributedConfig = u.getRenderConfig();
        const e = Math.ceil(p.distributedConfig.fps * p.distributedConfig.duration);
        if (!confirm(`Distribute recording? (Workers: ${o.length})
Auto Scene Sync enabled.`)) return;
        p.jobQueue = [], p.pendingChunks.clear(), p.completedJobs = 0, p.activeJobs.clear();
        for (let t = 0; t < e; t += I) {
          const n = Math.min(I, e - t);
          p.jobQueue.push({
            start: t,
            count: n
          });
        }
        p.totalJobs = p.jobQueue.length, o.forEach((t) => p.workerStatus.set(t, "idle")), u.setStatus(`Distributed Progress: 0 / ${p.totalJobs} jobs (Waiting for workers...)`), o.length > 0 ? (u.setStatus("Syncing Scene to Workers..."), v.sendRenderStart(), await p.sendSceneHelper(x, R)) : console.log("No workers yet. Waiting...");
      } else {
        y = false, u.setRecordingState(true);
        const o = u.getRenderConfig();
        try {
          const e = performance.now();
          await B.record(o, (t, n) => u.setRecordingState(true, `Rec: ${t}/${n} (${Math.round(t / n * 100)}%)`), (t) => {
            const n = document.createElement("a");
            n.href = t, n.download = `raytrace_${Date.now()}.webm`, n.click(), URL.revokeObjectURL(t);
          }), console.log(`Recording took ${performance.now() - e}[ms]`);
        } catch {
          alert("Recording failed.");
        } finally {
          u.setRecordingState(false), y = false, u.updateRenderButton(false), requestAnimationFrame(W);
        }
      }
    }, u.onConnectHost = () => {
      w === "host" ? (v.disconnect(), w = null, u.setConnectionState(null)) : (v.connect("host"), w = "host", u.setConnectionState("host"));
    }, u.onConnectWorker = () => {
      w === "worker" ? (v.disconnect(), w = null, u.setConnectionState(null)) : (v.connect("worker"), w = "worker", u.setConnectionState("worker"));
    }, u.setConnectionState(null);
  };
  async function te() {
    try {
      await f.init(), await h.initWasm();
    } catch (o) {
      alert("Init failed: " + o);
      return;
    }
    ee(), K(), A(), T("cornell", false), requestAnimationFrame(W);
  }
  te().catch(console.error);
})();
