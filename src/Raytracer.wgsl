// =========================================================
//   WebGPU Ray Tracer (Raytracer.wgsl)
// =========================================================

const PI = 3.141592653589793;
const T_MIN = 0.001;
const T_MAX = 1e30;
const SPATIAL_COUNT = 2u;      // 何個の近傍を探すか (2〜5くらい)
const SPATIAL_RADIUS = 30.0;   // 探索半径 (ピクセル単位)
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

struct GIReservoir {
    // [Slot 0] 16 bytes
    sample_dir: vec3<f32>,       // 12 bytes
    sample_dist: f32,            // 4 bytes

    // [Slot 1] 16 bytes
    sample_radiance: vec3<f32>,  // 12 bytes
    w_sum: f32,                  // 4 bytes

    // [Slot 2] 16 bytes
    creation_pos: vec3<f32>,     // 12 bytes
    M: f32,                      // 4 bytes

    // [Slot 3] 16 bytes
    creation_normal: vec3<f32>,  // 12 bytes
    W: f32,                      // 4 bytes
    // [Slot 4] 16 bytes (Alignment padding)
    creation_mat_id: u32,        // 4 bytes
    pad0: u32,                   // 12 bytes padding to align struct to 16 bytes
    pad1: u32,
    pad2: u32,
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
@group(0) @binding(9) var<storage, read> lights: array<LightRef>;

@group(0) @binding(7) var tex: texture_2d_array<f32>;
@group(0) @binding(8) var smp: sampler;

@group(0) @binding(10) var<storage, read_write> gi_reservoir: array<GIReservoir>;

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

// ターゲット関数 p_hat (輝度ベース)
fn evaluate_p_hat(radiance: vec3<f32>) -> f32 {
    // 輝度(Luminance)を重要度とする
    return dot(radiance, vec3(0.2126, 0.7152, 0.0722));
}

// RIS Update: 候補をリザーバに結合する
fn update_reservoir(
    res: ptr<function, GIReservoir>,
    dir: vec3<f32>,
    radiance: vec3<f32>,
    dist: f32,
    weight: f32,
    rng: ptr<function, u32>
) -> bool {
    (*res).w_sum += weight;
    (*res).M += 1.0;

    // 確率的に置換 (weight / w_sum)
    if rand_pcg(rng) < (weight / (*res).w_sum) {
        (*res).sample_dir = dir;
        (*res).sample_radiance = radiance;
        (*res).sample_dist = dist;
        return true;
    }
    return false;
}

// マージ用: 別のリザーバを自分に結合する
fn merge_reservoir(
    dest: ptr<function, GIReservoir>,
    src: GIReservoir,
    p_hat_src: f32,
    rng: ptr<function, u32>
) -> bool {
    let M = src.M;
    let weight = p_hat_src * src.W * M; // RIS weight

    (*dest).w_sum += weight;
    (*dest).M += M;

    if rand_pcg(rng) < (weight / (*dest).w_sum) {
        (*dest).sample_dir = src.sample_dir;
        (*dest).sample_radiance = src.sample_radiance;
        (*dest).sample_dist = src.sample_dist;
        return true;
    }
    return false;
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
    let tmin = max(t_min, max(max(min(t1.x, t2.x), min(t1.y, t2.y)), min(t1.z, t2.z)));
    let tmax = min(t_max, min(min(max(t1.x, t2.x), max(t1.y, t2.y)), max(t1.z, t2.z)));
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
    var stack: array<u32, 64>;
    var stackptr = 0u;

    var idx = node_start_idx;
    if intersect_aabb(nodes[idx].min_b.xyz, nodes[idx].max_b.xyz, r.origin, inv_d, t_min, closest_t) < 1e30 {
        stack[stackptr] = idx; stackptr++;
    }

    while stackptr > 0u {
        stackptr--;
        idx = stack[stackptr];
        let node = nodes[idx];
        let count = u32(node.max_b.w);

        if count > 0u {
            let first = u32(node.min_b.w);
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
            let l = u32(node.min_b.w) + node_start_idx;
            let r_idx = l + 1u;
            let nl = nodes[l];
            let nr = nodes[r_idx];
            let dl = intersect_aabb(nl.min_b.xyz, nl.max_b.xyz, r.origin, inv_d, t_min, closest_t);
            let dr = intersect_aabb(nr.min_b.xyz, nr.max_b.xyz, r.origin, inv_d, t_min, closest_t);

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
    var stack: array<u32, 64>;
    var stackptr = 0u;

    if intersect_aabb(nodes[0].min_b.xyz, nodes[0].max_b.xyz, r.origin, inv_d, t_min, res.t) < 1e30 {
        stack[stackptr] = 0u; stackptr++;
    }

    while stackptr > 0u {
        stackptr--;
        let idx = stack[stackptr];
        let node = nodes[idx];

        if node.max_b.w > 0.5 { // Leaf
            let inst_idx = u32(node.min_b.w);
            let inst = instances[inst_idx];
            let inv = get_inv_transform(inst);
            let r_local = Ray((inv * vec4(r.origin, 1.0)).xyz, (inv * vec4(r.direction, 0.0)).xyz);
            let blas_start = scene.blas_base_idx + inst.blas_node_offset;
            let blas = intersect_blas(r_local, t_min, res.t, blas_start);
            if blas.y > -0.5 {
                res.t = blas.x; res.tri_idx = blas.y; res.inst_idx = i32(inst_idx);
            }
        } else {
            let l = u32(node.min_b.w);
            let r_idx = l + 1u;
            let nl = nodes[l];
            let nr = nodes[r_idx];
            let dl = intersect_aabb(nl.min_b.xyz, nl.max_b.xyz, r.origin, inv_d, t_min, res.t);
            let dr = intersect_aabb(nr.min_b.xyz, nr.max_b.xyz, r.origin, inv_d, t_min, res.t);
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

fn ray_color(r_in: Ray, rng: ptr<function, u32>, coord: vec2<u32>) -> vec3<f32> {
    var ray = r_in;
    var throughput = vec3(1.0);
    var radiance = vec3(0.0);

    // ReSTIR用のバッファオフセット計算
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
        var is_specular = (mat_type == 2u) || (metallic > 0.9 && roughness < 0.1); 

        // ----------------------------------------------------------------
        // ★ ReSTIR GI Logic (Depth == 0 and Diffuse-like)
        // ----------------------------------------------------------------
        // 修正: 金属（Metallicが高いもの）は ReSTIR GI (Diffuse) の対象外にする
        let is_metallic = (mat_type == 1u) && (metallic > 0.1);
        
        // 「拡散反射成分が支配的なもの」だけを対象にする
        // Specular (鏡) や Metallic (金属) は除外
        let use_restir = (mat_type == 0u) || (!is_metallic && !is_specular && mat_type != 2u);

        if specular_bounce && use_restir {
            var state: GIReservoir;
            // 初期化
            state.w_sum = 0.0;
            state.M = 0.0;
            state.W = 0.0;
            
            // 1. Initial Candidate Generation (初期候補)
            // 通常通りBSDFサンプリングして、1バウンス先のライティングを計算する
            // ※ ここでは簡易的に「次の1バウンス」だけを評価します

            var initial_dir: vec3<f32>;
            var initial_Li = vec3(0.0);
            var initial_dist = 0.0;
            var initial_pdf = 0.0;

            // BSDFサンプリング
            let scatter = sample_diffuse(normal, albedo, rng); // Diffuseのみと仮定
            if scatter.pdf > 0.0 {
                initial_dir = scatter.dir;
                initial_pdf = scatter.pdf;
                
                // レイを飛ばしてLiを取得 (再帰の代わりに1ステップだけトレース)
                let bounce_ray = Ray(hit_p + normal * 1e-4, initial_dir);
                let bounce_hit = intersect_tlas(bounce_ray, T_MIN, T_MAX);

                if bounce_hit.inst_idx >= 0 {
                    initial_dist = bounce_hit.t;
                    // ヒット先の色を取得（簡易版：Emissive + BaseColor）
                    // 本来はここでNEEや更なるバウンス評価が必要ですが、まずはヒット先のEmissive等で近似
                    let b_tri = topology[u32(bounce_hit.tri_idx)];
                    initial_Li = b_tri.data3.rgb; // Emissive
                    // もしEmissiveが弱ければ、環境光や簡易ライティングを足す
                    if length(initial_Li) < 1e-4 {
                         // 簡易的にヒット面のAlbedoを間接光として扱う（本来はNGだがデバッグ用）
                        initial_Li = b_tri.data0.rgb * 0.1;
                    }
                } else {
                    // Sky color (もしあれば)
                    initial_Li = vec3(0.0);
                }
            }

            // 初期候補をリザーバに追加
            let p_hat_initial = evaluate_p_hat(initial_Li);
            // RIS Weight = p_hat / pdf (BSDF重点サンプリングの場合)
            let w_initial = select(0.0, p_hat_initial / initial_pdf, initial_pdf > 1e-6);

            update_reservoir(&state, initial_dir, initial_Li, initial_dist, w_initial, rng);

            // 2. Temporal Reuse (時間的再利用)
            // 前フレームのリザーバを読み込む
            // ※ 本来はバックプロジェクションで「前の位置」のピクセルを読むべきだが、
            //    まずは「同じピクセル位置」を読む (Camera固定ならこれで動く)
            let prev_res = gi_reservoir[prev_res_idx];
            
            // 有効性チェック（法線とマテリアルID）
            let normal_check = dot(prev_res.creation_normal, normal) > 0.9;
            let mat_check = prev_res.creation_mat_id == bitcast<u32>(tri.data0.w); // mat_type

            if normal_check && mat_check {
                // 過去の候補の評価値 (保存されたRadianceを使用)
                let p_hat_prev = evaluate_p_hat(prev_res.sample_radiance);
                
                // マージ
                merge_reservoir(&state, prev_res, p_hat_prev, rng);
                
                // [修正後] w_sum も道連れにして比率を保つ
                if state.M > 20.0 {
                    state.w_sum *= (20.0 / state.M);
                    state.M = 20.0;
                }
            }

            // ----------------------------------------------------------------
            // 3. Spatial Reuse (空間的再利用)
            // ----------------------------------------------------------------
            // 前フレームのデータ(prev_res_idx)から近傍を探してマージします
            // ※同じパス内で現在の近傍(curr_res_idx)を読むと、まだ計算されていない可能性があり危険なため

            for (var i = 0u; i < SPATIAL_COUNT; i++) {
                // ランダムな近傍ピクセルを選択
                let r_radius = sqrt(rand_pcg(rng)) * SPATIAL_RADIUS;
                let r_angle = 2.0 * PI * rand_pcg(rng);
                let offset = vec2<f32>(r_radius * cos(r_angle), r_radius * sin(r_angle));

                let neighbor_coord = vec2<i32>(vec2<f32>(coord) + offset);
                
                // 画面内チェック
                if neighbor_coord.x >= 0 && neighbor_coord.x < i32(scene.width) && neighbor_coord.y >= 0 && neighbor_coord.y < i32(scene.height) {

                    let n_idx = u32(neighbor_coord.y) * scene.width + u32(neighbor_coord.x);
                    
                    // 近傍のリザーバを取得 (Prevフレームのページから読む)
                    let n_offsets = get_reservoir_offsets(n_idx);
                    let neighbor_res = gi_reservoir[n_offsets.y]; // .y が Prev
                    
                    // --- 幾何的類似度チェック (Geometry Validation) ---
                    // これをやらないと、壁の裏や別オブジェクトの光が漏れてくる(Light Leaking)
                    
                    // 1. マテリアルIDチェック
                    let mat_ok = neighbor_res.creation_mat_id == bitcast<u32>(tri.data0.w);
                    
                    // 2. 法線チェック (似た向きの面か)
                    let norm_ok = dot(neighbor_res.creation_normal, normal) > 0.9;
                    
                    // 3. 位置(深度)チェック (距離が離れすぎていないか)
                    // 簡易的に距離の二乗で判定。シーンのスケールに合わせて閾値調整が必要(例: 1.0)
                    let dist_sq = dot(neighbor_res.creation_pos - hit_p, neighbor_res.creation_pos - hit_p);
                    let pos_ok = dist_sq < 0.1; // シーンに合わせて調整してください

                    if mat_ok && norm_ok && pos_ok {
                        // 近傍の p_hat を再評価
                        // (ReSTIR GIでは近傍のRadianceをそのまま使う近似が一般的)
                        let p_hat_neighbor = evaluate_p_hat(neighbor_res.sample_radiance);
                        
                        // マージ実行
                        merge_reservoir(&state, neighbor_res, p_hat_neighbor, rng);
                    }
                }
            }
            
            // クランプ (Spatial Reuse後はMが増えすぎるので再度抑える)
            if state.M > 50.0 {
                state.w_sum *= (50.0 / state.M);
                state.M = 50.0;
            };

            // 3. Finalize W (Unbiased Weight calculation)
            let p_hat_final = evaluate_p_hat(state.sample_radiance);
            if p_hat_final > 0.0 {
                state.W = state.w_sum / (state.M * p_hat_final);
            } else {
                state.W = 0.0;
            }

            if state.W > 20.0 {
                state.W = 20.0;
            }

            // 4. Store Reservoir (次フレームのために保存)
            state.creation_pos = hit_p;
            state.creation_normal = normal;
            state.creation_mat_id = bitcast<u32>(tri.data0.w);
            gi_reservoir[curr_res_idx] = state;

            // 5. Shading (選ばれた候補を使って色を決定)
            // Lo = Li * BSDF * cos / PDF
            // ReSTIRでは: Lo = Li * BSDF * cos * W

            let bsdf_val = eval_diffuse(albedo); // Diffuse BSDF = albedo / PI
            let cos_theta = max(dot(normal, state.sample_dir), 0.0);

            // [修正] radiance += throughput * state.sample_radiance * bsdf_val * cos_theta * state.W * state.M; 
            // Standard ReSTIR estimator: Lo = Li * BSDF * cos * W. 
            // Our W already includes normalization (w_sum / (M * p_hat)).
            radiance += throughput * state.sample_radiance * bsdf_val * cos_theta * state.W;

            // ReSTIRで決まったのでループ終了 (GI計算済み)
            break;
        }


        // --- NEE ---
        if !is_specular && mat_type != 2u {
            let light_s = sample_light_source(hit_p, rng);
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
