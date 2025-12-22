// =========================================================
//   WebGPU Ray Tracer (TLAS & BLAS)
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
    blas_base_idx: u32,
    vertex_count: u32,
    rand_seed: u32,
    light_count: u32
}

struct MeshTopology {
    v0: u32,
    v1: u32,
    v2: u32,
    pad: u32,
    data0: vec4<f32>, // rgb: Albedo, w: MaterialType (0:Diffuse, 1:Metal, 2:Glass, 3:Light)
    data1: vec4<f32>  // x: Roughness/IOR, y: TexIdx
}

struct LightRef {
    inst_idx: u32,
    tri_idx: u32
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

// =========================================================
//   Bindings
// =========================================================

@group(0) @binding(0) var outputTex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<storage, read_write> accumulateBuffer: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> scene: SceneUniforms;

@group(0) @binding(3) var<storage, read> geometry: array<f32>; // [Pos(4)... | Norm(4)... | UV(2)...]
@group(0) @binding(4) var<storage, read> topology: array<MeshTopology>;
@group(0) @binding(5) var<storage, read> nodes: array<BVHNode>; 
@group(0) @binding(6) var<storage, read> instances: array<Instance>;
@group(0) @binding(9) var<storage, read> lights: array<LightRef>;

@group(0) @binding(7) var tex: texture_2d_array<f32>;
@group(0) @binding(8) var smp: sampler;

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
//   BSDF: Diffuse
// =========================================================

fn eval_diffuse(albedo: vec3<f32>) -> vec3<f32> {
    return albedo / PI;
}

fn sample_diffuse(normal: vec3<f32>, albedo: vec3<f32>, rng: ptr<function, u32>) -> ScatterResult {
    // Cosine weighted sampling
    let r1 = rand_pcg(rng);
    let r2 = rand_pcg(rng);
    let z = sqrt(1.0 - r2);
    let phi = 2.0 * PI * r1;
    let x = cos(phi) * sqrt(r2);
    let y = sin(phi) * sqrt(r2);

    let onb = build_onb(normal);
    let local_dir = vec3(x, y, z);
    let world_dir = local_to_world(onb, local_dir);

    let cos_theta = max(local_dir.z, 0.0); // local_dir.z is cos(theta)
    let pdf = cos_theta / PI;

    // throughput = bsdf * cosine / pdf
    //            = (albedo/PI) * cos_theta / (cos_theta/PI)
    //            = albedo

    return ScatterResult(world_dir, pdf, albedo, false);
}

// =========================================================
//   BSDF: GGX / Metal
// =========================================================

fn ggx_d(n_dot_h: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let d = (n_dot_h * n_dot_h) * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

fn ggx_g1(v_dot_n: f32, k: f32) -> f32 {
    return v_dot_n / (v_dot_n * (1.0 - k) + k);
}

fn ggx_g(n_dot_l: f32, n_dot_v: f32, alpha: f32) -> f32 {
    let k = (alpha + 1.0) * (alpha + 1.0) / 8.0;
    return ggx_g1(n_dot_l, k) * ggx_g1(n_dot_v, k);
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

fn eval_ggx(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, roughness: f32, f0: vec3<f32>) -> vec3<f32> {
    let h = normalize(v + l);
    let n_dot_l = max(dot(n, l), 0.0);
    let n_dot_v = max(dot(n, v), 0.0);

    if n_dot_l == 0.0 || n_dot_v == 0.0 { return vec3(0.0); }

    let alpha = roughness * roughness;
    let D = ggx_d(dot(n, h), alpha);
    let G = ggx_g(n_dot_l, n_dot_v, alpha);
    let F = fresnel_schlick(dot(h, v), f0);

    return (D * G * F) / (4.0 * n_dot_v * n_dot_l);
}

fn sample_ggx(n: vec3<f32>, v: vec3<f32>, roughness: f32, f0: vec3<f32>, rng: ptr<function, u32>) -> ScatterResult {
    let r1 = rand_pcg(rng);
    let r2 = rand_pcg(rng);
    let alpha = roughness * roughness;

    // Sample Microfacet Normal (H)
    let phi = 2.0 * PI * r1;
    let cos_theta = sqrt((1.0 - r2) / (r2 * (alpha * alpha - 1.0) + 1.0));
    let sin_theta = sqrt(1.0 - cos_theta * cos_theta);

    let local_h = vec3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
    let onb = build_onb(n);
    let h = local_to_world(onb, local_h);

    let l = reflect(-v, h);

    if dot(n, l) <= 0.0 {
        // Absorbed
        return ScatterResult(vec3(0.0), 0.0, vec3(0.0), true);
    }

    let n_dot_l = max(dot(n, l), 0.0);
    let n_dot_v = max(dot(n, v), 0.0);
    let n_dot_h = max(dot(n, h), 0.0);
    let v_dot_h = max(dot(v, h), 0.0);

    let D = ggx_d(n_dot_h, alpha);
    let G = ggx_g(n_dot_l, n_dot_v, alpha);
    let F = fresnel_schlick(v_dot_h, f0);

    let pdf = (D * n_dot_h) / (4.0 * v_dot_h);
    let bsdf = (D * G * F) / (4.0 * n_dot_v * n_dot_l);
    
    // throughput = bsdf * cosine / pdf
    let throughput = bsdf * n_dot_l / pdf;

    return ScatterResult(l, pdf, throughput, false); // GGX treated as non-specular for MIS? Usually yes for rough.
    // However original code had: specular_bounce = false; // GGX is proper for MIS
}

// =========================================================
//   BSDF: Dielectric / Glass
// =========================================================

fn reflectance(cosine: f32, ref_idx: f32) -> f32 {
    let r0_sqrt = (1.0 - ref_idx) / (1.0 + ref_idx);
    let r0 = r0_sqrt * r0_sqrt;
    return r0 + (1.0 - r0) * pow((1.0 - cosine), 5.0);
}

fn sample_dielectric(in_dir: vec3<f32>, normal: vec3<f32>, ior: f32, albedo: vec3<f32>, rng: ptr<function, u32>) -> ScatterResult {
    let front_face = dot(in_dir, normal) < 0.0;
    let n_eff = select(-normal, normal, front_face);
    let ratio = select(ior, 1.0 / ior, front_face);

    let unit_in = normalize(in_dir);
    let cos_t = min(dot(-unit_in, n_eff), 1.0);
    let sin_t = sqrt(1.0 - cos_t * cos_t);

    var out_dir = vec3(0.0);

    if (ratio * sin_t > 1.0) || (reflectance(cos_t, ratio) > rand_pcg(rng)) {
        out_dir = reflect(unit_in, n_eff);
    } else {
        out_dir = ratio * (unit_in + cos_t * n_eff) - sqrt(abs(1.0 - (1.0 - cos_t * cos_t) * ratio * ratio)) * n_eff;
    }

    return ScatterResult(out_dir, 1.0, albedo, true); // Specular bounce (Delta)
}

// =========================================================
//   Direct Light Sampling (NEE)
// =========================================================

fn power_heuristic(pdf_f: f32, pdf_g: f32) -> f32 {
    let f2 = pdf_f * pdf_f;
    let g2 = pdf_g * pdf_g;
    return f2 / (f2 + g2);
}

fn sample_light_source(origin: vec3<f32>, rng: ptr<function, u32>) -> LightSample {
    var res: LightSample;
    res.L = vec3(0.0);
    res.pdf = 0.0;

    let num_lights = scene.light_count;
    if num_lights == 0u { return res; }

    // Pick a random light
    let idx = u32(rand_pcg(rng) * f32(num_lights));
    let light_ref = lights[idx];
    let inst = instances[light_ref.inst_idx];
    let tri = topology[light_ref.tri_idx];

    let transform = get_transform(inst);
    let v0 = (transform * vec4(get_pos(tri.v0), 1.0)).xyz;
    let v1 = (transform * vec4(get_pos(tri.v1), 1.0)).xyz;
    let v2 = (transform * vec4(get_pos(tri.v2), 1.0)).xyz;

    // Sample triangle uniformly
    let r1 = rand_pcg(rng);
    let r2 = rand_pcg(rng);
    let sqrt_r1 = sqrt(r1);
    let u = 1.0 - sqrt_r1;
    let v = sqrt_r1 * (1.0 - r2);
    let w = sqrt_r1 * r2;

    let light_pos = u * v0 + v * v1 + w * v2;
    let to_light = light_pos - origin;
    let dist_sq = dot(to_light, to_light);
    res.dist = sqrt(dist_sq);
    res.dir = to_light / res.dist;

    let cross_v = cross(v1 - v0, v2 - v0);
    let area = 0.5 * length(cross_v);
    let light_normal = normalize(cross_v);

    let cos_light = dot(-res.dir, light_normal);

    if cos_light <= 1e-4 { return res; }

    res.pdf = dist_sq / (area * cos_light * f32(num_lights));
    res.L = tri.data0.rgb * 1.0;

    return res;
}

fn get_light_pdf(origin: vec3<f32>, hit_tri_idx: u32, hit_inst_idx: u32, hit_t: f32, ray_dir: vec3<f32>) -> f32 {
    let num_lights = scene.light_count;
    if num_lights == 0u { return 0.0; }

    let tri = topology[hit_tri_idx];
    let inst = instances[hit_inst_idx];
    let transform = get_transform(inst);
    let v0 = (transform * vec4(get_pos(tri.v0), 1.0)).xyz;
    let v1 = (transform * vec4(get_pos(tri.v1), 1.0)).xyz;
    let v2 = (transform * vec4(get_pos(tri.v2), 1.0)).xyz;

    let cross_v = cross(v1 - v0, v2 - v0);
    let area = 0.5 * length(cross_v);
    let light_normal = normalize(cross_v);

    let cos_light = max(dot(-ray_dir, light_normal), 0.0);
    if cos_light < 1e-4 { return 0.0; }

    let dist_sq = hit_t * hit_t;
    return dist_sq / (area * cos_light * f32(num_lights));
}

// =========================================================
//   Intersection
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
    var stack: array<u32, 32>;
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
//   Main Path Tracer
// =========================================================

fn ray_color(r_in: Ray, rng: ptr<function, u32>) -> vec3<f32> {
    var ray = r_in;
    var throughput = vec3(1.0);
    var radiance = vec3(0.0);

    var prev_bsdf_pdf = 0.0;
    var specular_bounce = true;

    for (var depth = 0u; depth < MAX_DEPTH; depth++) {
        let hit = intersect_tlas(ray, T_MIN, T_MAX);
        if hit.inst_idx < 0 { 
            // Environment lighting could go here
            break;
        }

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
        
        // Barycentric coords
        let r_local = Ray((inv * vec4(ray.origin, 1.)).xyz, (inv * vec4(ray.direction, 0.)).xyz);
        let s = r_local.origin - v0_pos;
        let e1 = v1_pos - v0_pos;
        let e2 = v2_pos - v0_pos;
        let h_val = cross(r_local.direction, e2);
        let a = dot(e1, h_val);
        let f_val = 1.0 / a;
        let u = f_val * dot(s, h_val);
        let q = cross(s, e1);
        let v = f_val * dot(r_local.direction, q);
        let w = 1.0 - u - v;

        // Normals
        let n0 = get_normal(tri.v0);
        let n1 = get_normal(tri.v1);
        let n2 = get_normal(tri.v2);
        let ln = normalize(n0 * w + n1 * u + n2 * v);
        let wn = normalize((vec4(ln, 0.0) * inv).xyz);
        var normal = wn;
        let front_face = dot(ray.direction, normal) < 0.0;
        normal = select(-normal, normal, front_face);

        let hit_p = ray.origin + ray.direction * hit.t;
        
        // Textures
        let uv0 = get_uv(tri.v0);
        let uv1 = get_uv(tri.v1);
        let uv2 = get_uv(tri.v2);
        let tex_uv = uv0 * w + uv1 * u + uv2 * v;
        let tex_idx = tri.data1.y;
        var albedo = tri.data0.rgb;
        if tex_idx > -0.5 {
            albedo *= textureSampleLevel(tex, smp, tex_uv, i32(tex_idx), 0.0).rgb;
        }

        // --- Emission ---
        if mat_type == 3u {
            let emitted = albedo * 1.0;
            if specular_bounce {
                radiance += min(throughput * emitted, vec3(5.0));
            } else {
                let light_pdf_val = get_light_pdf(ray.origin, tri_idx, inst_idx, hit.t, ray.direction);
                let weight = power_heuristic(prev_bsdf_pdf, light_pdf_val);
                radiance += min(throughput * emitted * weight, vec3(3.0));
            }
            break;
        }

        // --- Material Setup ---
        var is_specular = (mat_type == 2u) || (mat_type == 1u && tri.data1.x < 0.1);
        let roughness = max(tri.data1.x, 0.004);
        var f0 = vec3(0.04);
        if mat_type == 1u { f0 = albedo; } else if mat_type == 2u { f0 = albedo; } // Glass uses albedo for color/f0 tint

        // --- NEE ---
        if !is_specular && mat_type != 2u {
            let light_s = sample_light_source(hit_p, rng);
            if light_s.pdf > 0.0 {
                let shadow_ray = Ray(hit_p + normal * 1e-4, light_s.dir);
                let shadow_hit = intersect_tlas(shadow_ray, T_MIN, light_s.dist - 1e-4);

                if shadow_hit.inst_idx == -1 {
                    var bsdf_val = vec3(0.0);
                    var bsdf_pdf_val = 0.0;

                    let L = light_s.dir;
                    let V = -ray.direction;

                    if mat_type == 0u { // Diffuse
                        bsdf_val = eval_diffuse(albedo);
                        bsdf_pdf_val = max(dot(normal, L), 0.0) / PI;
                    } else if mat_type == 1u { // Metal
                        bsdf_val = eval_ggx(normal, V, L, roughness, f0);
                        // PDF approximation for GGX
                        let H = normalize(V + L);
                        let a2 = roughness * roughness;
                        let D = ggx_d(dot(normal, H), a2);
                        bsdf_pdf_val = (D * max(dot(normal, H), 0.0)) / (4.0 * max(dot(V, H), 0.0));
                    }

                    if bsdf_pdf_val > 0.0 {
                        let weight = power_heuristic(light_s.pdf, bsdf_pdf_val);
                        let contribution = throughput * bsdf_val * light_s.L * weight * max(dot(normal, L), 0.0) / light_s.pdf;
                        radiance += min(contribution, vec3(3.0));
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
        } else { // Glass
            scatter = sample_dielectric(ray.direction, normal, tri.data1.x, albedo, rng);
        }

        if scatter.pdf == 0.0 || (scatter.throughput.x + scatter.throughput.y + scatter.throughput.z) == 0.0 {
            break;
        }

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
//   Tone Mapping
// =========================================================

fn aces_tone_mapping(color: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), vec3(0.0), vec3(1.0));
}

// =========================================================
//   Compute Main
// =========================================================

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(outputTex);
    if id.x >= dims.x || id.y >= dims.y { return; }
    let p_idx = id.y * dims.x + id.x;
    var rng = init_rng(p_idx, scene.rand_seed);

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
    let hdr_color = new_acc.rgb / new_acc.a;
    let mapped = aces_tone_mapping(hdr_color);
    let out = pow(mapped, vec3(1.0 / 2.2));
    textureStore(outputTex, vec2<i32>(id.xy), vec4(out, 1.));
}
