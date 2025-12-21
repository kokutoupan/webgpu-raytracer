// =========================================================
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
    rand_seed: u32,
    light_count: u32
}

struct MeshTopology {
    v0: u32,
    v1: u32,
    v2: u32,
    pad: u32,
    data0: vec4<f32>,
    data1: vec4<f32>
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

// --- Bindings (Consolidated) ---

@group(0) @binding(0) var outputTex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<storage, read_write> accumulateBuffer: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> scene: SceneUniforms;

@group(0) @binding(3) var<storage, read> geometry: array<f32>; // [Pos(4)... | Norm(4)... | UV(2)...]
@group(0) @binding(4) var<storage, read> topology: array<MeshTopology>;
@group(0) @binding(5) var<storage, read> nodes: array<BVHNode>; // Merged TLAS/BLAS
@group(0) @binding(6) var<storage, read> instances: array<Instance>;
@group(0) @binding(9) var<storage, read> lights: array<LightRef>;

@group(0) @binding(7) var tex: texture_2d_array<f32>;
@group(0) @binding(8) var smp: sampler;

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
    var r0 = (1.0 - ref_idx) / (1.0 + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0 - r0) * pow((1.0 - cosine), 5.0);
}

// 1.0を超えても綺麗に収める関数 (ACES近似)
fn aces_tone_mapping(color: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), vec3(0.0), vec3(1.0));
}

// --- Intersection ---

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

    // Root Node (Global Index in 'nodes' array)
    let root_idx = node_start_idx;
    let root = nodes[root_idx];

    if intersect_aabb(root.min_b, root.max_b, r.origin, inv_d, t_min, closest_t) < 1e30 {
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

                let tri = topology[tri_id];
                let v0 = get_pos(tri.v0);
                let v1 = get_pos(tri.v1);
                let v2 = get_pos(tri.v2);

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

            let dl = intersect_aabb(nl.min_b, nl.max_b, r.origin, inv_d, t_min, closest_t);
            let dr = intersect_aabb(nr.min_b, nr.max_b, r.origin, inv_d, t_min, closest_t);

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
    var stack: array<u32, 16>;
    var stackptr = 0u;

    if intersect_aabb(nodes[0].min_b, nodes[0].max_b, r.origin, inv_d, t_min, res.t) < 1e30 {
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
            let dl = intersect_aabb(nl.min_b, nl.max_b, r.origin, inv_d, t_min, res.t);
            let dr = intersect_aabb(nr.min_b, nr.max_b, r.origin, inv_d, t_min, res.t);
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

fn get_transform(inst: Instance) -> mat4x4<f32> {
    return mat4x4<f32>(inst.transform_0, inst.transform_1, inst.transform_2, inst.transform_3);
}

fn sample_lights(origin: vec3<f32>, normal: vec3<f32>, rng: ptr<function, u32>) -> vec3<f32> {
    let num_lights = scene.light_count;
    if num_lights == 0u { return vec3(0.0); }

    // Pick random light
    let idx = u32(rand_pcg(rng) * f32(num_lights));
    let light_ref = lights[idx];

    let inst = instances[light_ref.inst_idx];
    let tri = topology[light_ref.tri_idx];

    // Get Triangle Vertices (Local)
    let v0_local = get_pos(tri.v0);
    let v1_local = get_pos(tri.v1);
    let v2_local = get_pos(tri.v2);

    // Transform to World
    let transform = get_transform(inst);
    let v0 = (transform * vec4(v0_local, 1.0)).xyz;
    let v1 = (transform * vec4(v1_local, 1.0)).xyz;
    let v2 = (transform * vec4(v2_local, 1.0)).xyz;

    // Sample Point on Triangle
    let r1 = rand_pcg(rng);
    let r2 = rand_pcg(rng);
    let sqrt_r1 = sqrt(r1);
    let u = 1.0 - sqrt_r1;
    let v = sqrt_r1 * (1.0 - r2);
    let w = sqrt_r1 * r2;

    let light_pos = u * v0 + v * v1 + w * v2;
    let light_vec = light_pos - origin;
    let dist_sq = dot(light_vec, light_vec);
    let dist = sqrt(dist_sq);
    let dir = light_vec / dist;

    // Visibility Check
    if dot(dir, normal) <= 0.0 { return vec3(0.0); }

    let shadow_ray = Ray(origin + normal * 1e-3, dir);
    let hit = intersect_tlas(shadow_ray, T_MIN, dist - 1e-3);

    if hit.inst_idx != -1 {
         // Occluded
        return vec3(0.0);
    }

    // Geometry Factor
    let light_normal = normalize(cross(v1 - v0, v2 - v0)); // Assuming consistent winding
    let cos_light_raw = dot(-dir, light_normal);
    if cos_light_raw <= 0.0 { return vec3(0.0); } // ★裏面なら光らない
    let cos_theta_light = cos_light_raw;
    let cos_theta_surf = max(dot(normal, dir), 0.0);
    let area = 0.5 * length(cross(v1 - v0, v2 - v0));

    // Emission (Hardcoded multiplier for now to match ray_color)
    let emission = tri.data0.rgb * 1.0; 

    // PDF = 1 / (Area * num_lights)
    // Contribution = Le * G * pdf_inv
    // G = (cos_surf * cos_light) / dist_sq
    // pdf_inv = Area * num_lights

    let G = (cos_theta_surf * cos_theta_light) / dist_sq;
    let weight = G * area * f32(num_lights);

    return emission * weight;
}

fn ray_color(r_in: Ray, rng: ptr<function, u32>) -> vec3<f32> {
    var ray = r_in;
    var throughput = vec3<f32>(1.0);
    var radiance = vec3<f32>(0.0);
    
    // ★追加: 直前の反射がスペキュラ（鏡面/屈折）だったか？
    // 初期値 true にすることで、カメラから直接見えるライトは描画される
    var specular_bounce = true;

    for (var depth = 0u; depth < MAX_DEPTH; depth++) {
        let hit = intersect_tlas(ray, T_MIN, T_MAX);
        if hit.inst_idx < 0 { break; }

        let inst = instances[u32(hit.inst_idx)];
        let tri_idx = u32(hit.tri_idx);
        let tri = topology[tri_idx];

        let i0 = tri.v0;
        let i1 = tri.v1;
        let i2 = tri.v2;
        let v0_pos = get_pos(i0);
        let v1_pos = get_pos(i1);
        let v2_pos = get_pos(i2);

        let inv = get_inv_transform(inst);
        let r_local = Ray((inv * vec4(ray.origin, 1.)).xyz, (inv * vec4(ray.direction, 0.)).xyz);
        let s = r_local.origin - v0_pos;
        let e1 = v1_pos - v0_pos;
        let e2 = v2_pos - v0_pos;
        let h = cross(r_local.direction, e2);
        let a = dot(e1, h);
        let f = 1.0 / a;
        let u = f * dot(s, h);
        let q = cross(s, e1);
        let v = f * dot(r_local.direction, q);
        let w = 1.0 - u - v;

        let n0 = get_normal(i0);
        let n1 = get_normal(i1);
        let n2 = get_normal(i2);
        let ln = normalize(n0 * w + n1 * u + n2 * v);
        let wn = normalize((vec4(ln, 0.0) * inv).xyz);

        var normal = wn;
        let front = dot(ray.direction, normal) < 0.0;
        normal = select(-normal, normal, front);

        let uv0 = get_uv(i0);
        let uv1 = get_uv(i1);
        let uv2 = get_uv(i2);
        let tex_uv = uv0 * w + uv1 * u + uv2 * v;
        let hit_p = ray.origin + ray.direction * hit.t;

        let albedo_color = tri.data0.rgb;
        let mat_type = bitcast<u32>(tri.data0.w);

        let tex_idx = tri.data1.y;
        var tex_color = vec3(1.0);
        if tex_idx > -0.5 {
            tex_color = textureSampleLevel(tex, smp, tex_uv, i32(tex_idx), 0.0).rgb;
        }
        let albedo = albedo_color * tex_color;

        // ★修正1: Emissionの計算
        // ここでの判定には depth や mat_type を使わず、フラグを見る
        if mat_type == 3u {
            if specular_bounce {
                let emitted = albedo * 1.; // 明るさ20

                if depth == 0u {
                    radiance += throughput * emitted;
                } else {
                    // ここも上限 5.0 程度にクランプ
                    radiance += min(throughput * emitted, vec3(3.0));
                }
            }
            break;
        }

        // 2. NEE (Diffuse only)
        if mat_type == 0u {
            let Ld = sample_lights(hit_p, normal, rng);
            let brdf = albedo * 0.318309886; // albedo / PI
            let contribution = throughput * Ld * brdf;
            let clamped = min(contribution, vec3(3.0));
            radiance += clamped;

            // ★重要: NEEを行ったので、次のバウンスでライトに当たっても発光を加算しない
            specular_bounce = false;
        } else {
            // 鏡面反射やガラスの場合はNEEが効かないので、
            // 次のバウンスでライトに当たったら発光を加算する
            specular_bounce = true;
        }

        // 3. Scatter
        var scattered_dir: vec3<f32>;
        if mat_type == 0u {
            let target_p = hit_p + normal + random_unit_vector(rng);
            scattered_dir = normalize(target_p - hit_p);
            throughput *= albedo;
        } else if mat_type == 1u {
            let reflected = reflect(ray.direction, normal);
            let fuzz = tri.data1.x;
            scattered_dir = normalize(reflected + fuzz * random_unit_vector(rng));
            if dot(scattered_dir, normal) <= 0.0 { break; }
            throughput *= albedo;
        } else {
            // Dielectric
            let ir = tri.data1.x; // index of refraction (屈折率)
            let ratio = select(ir, 1.0 / ir, front);
            let unit = normalize(ray.direction);
            let cos_t = min(dot(-unit, normal), 1.0);
            let sin_t = sqrt(1.0 - cos_t * cos_t);

            // 完全に反射するか (全反射)、フレネル反射するかを判定
            if (ratio * sin_t > 1.0) || (reflectance(cos_t, ratio) > rand_pcg(rng)) {
                scattered_dir = reflect(unit, normal);
            } else {
                // 屈折 (Refract)
                scattered_dir = ratio * (unit + cos_t * normal) - sqrt(abs(1.0 - (1.0 - cos_t * cos_t) * ratio * ratio)) * normal;
            }
            
            // 減衰なし（透明）
            throughput *= albedo;
        }

        ray = Ray(hit_p + normal * 1e-4, scattered_dir);

        // RR (変更なし)
        if depth > 3u {
            let p = max(throughput.r, max(throughput.g, throughput.b));
            if rand_pcg(rng) > p { break; }
            throughput /= p;
        }
    }
    return radiance;
}

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
