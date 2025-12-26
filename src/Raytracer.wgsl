// =========================================================
//   WebGPU Ray Tracer (Raytracer.wgsl)
// =========================================================

const PI = 3.141592653589793;
const T_MIN = 0.001;
const T_MAX = 1e30;
const SPATIAL_COUNT = 2u;      // 何個の近傍を探すか (2〜5くらい)
const SPATIAL_RADIUS = 30.0;   // 探索半径 (ピクセル単位)
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

// 八面体エンコーディングによる法線圧縮 (vec3 -> vec2)
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

// 現在のフレーム用のインデックスと、過去フレーム用のインデックスを計算する
fn get_reservoir_offsets(pixel_idx: u32) -> vec2<u32> {
    let page_size = scene.width * scene.height;
    
    // frame_count が偶数のとき: Curr=0面, Prev=1面
    // frame_count が奇数のとき: Curr=1面, Prev=0面
    let current_page = (scene.frame_count % 2u) * page_size;
    let prev_page = ((scene.frame_count + 1u) % 2u) * page_size;

    return vec2(current_page + pixel_idx, prev_page + pixel_idx);
}

// ターゲット関数 p_hat (輝度ベース + BSDF近似)
fn evaluate_p_hat(radiance: vec3<f32>, albedo: vec3<f32>, cos_theta: f32) -> f32 {
    // 輝度(Luminance)を実質的な寄与(radiance * albedo * cos)で評価
    let contribution = radiance * albedo * cos_theta;
    
    // [Firefly対策] 非常に高い輝度をクランプして、特定のサンプルがリザーバを支配し続けないようにする
    let clamped_contribution = min(contribution, vec3(50.0));
    return dot(clamped_contribution, vec3(0.2126, 0.7152, 0.0722));
}

// マージ用: 反射率と余弦を考慮した重み付け
fn merge_reservoir_refined(
    dest: ptr<function, GIReservoir>,
    src: GIReservoir,
    p_hat_dest: f32, // 移動先の点での重要度
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

// 互換性のための既存のリザーバ更新 (RIS Update用)
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
