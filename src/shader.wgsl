// =========================================================
//   WebGPU Ray Tracer (Indexed Geometry & Full Tessellation)
// =========================================================

// --- Constants ---
const PI = 3.141592653589793;
const T_MIN = 0.001;
const T_MAX = 1e30;
// これらの定数はTypeScript側から置換されます
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

// 8 floats = 32 bytes (Rust側の attributes 配列と一致させる)
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

// 戻り値: x=t, y=triangle_index
fn hit_bvh(r: Ray, t_min: f32, t_max: f32) -> vec2<f32> {
    var closest_t = t_max;
    var hit_idx = -1.0;

    let inv_d = 1.0 / r.direction;
    var stack: array<u32, 32>;
    var stackptr = 0u;

    // ルートノード交差判定
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
            let first = u32(node.left_first); // ソート済みインデックス配列上の開始位置

            for (var i = 0u; i < count; i++) {
                let tri_id = first + i; // 属性配列のインデックスでもある
                let base_idx = tri_id * 3u;

                // インデックスバッファから頂点IDを取得
                let i0 = indices[base_idx];
                let i1 = indices[base_idx + 1u];
                let i2 = indices[base_idx + 2u];

                // 頂点バッファから座標を取得
                let v0 = vertices[i0].xyz;
                let v1 = vertices[i1].xyz;
                let v2 = vertices[i2].xyz;

                let t = hit_triangle_raw(v0, v1, v2, r, t_min, closest_t);
                if t > 0.0 {
                    closest_t = t;
                    hit_idx = f32(tri_id); // ヒットした三角形のID (属性参照用)
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
        
        // 1. 頂点法線を計算 (スムーズシェーディングする場合はここで頂点法線を補間するが、今回はフラット)
        let base_idx = tri_idx * 3u;
        let i0 = indices[base_idx];
        let i1 = indices[base_idx + 1u];
        let i2 = indices[base_idx + 2u];
        let v0 = vertices[i0].xyz;
        let v1 = vertices[i1].xyz;
        let v2 = vertices[i2].xyz;

        let e1 = v1 - v0;
        let e2 = v2 - v0;
        var normal = normalize(cross(e1, e2));
        var front_face = dot(ray.direction, normal) < 0.0;
        normal = select(-normal, normal, front_face);

        // 2. マテリアル属性を取得
        let attr = attributes[tri_idx];
        let albedo = attr.data0.rgb;
        let mat_bits = bitcast<u32>(attr.data0.w); // f32 -> u32
        let extra = attr.data1.x;

        let p = ray.origin + t * ray.direction;

        // Emission (Light = 3)
        if mat_bits == 3u {
            // 裏面は光らないようにする、あるいは両面発光にする
            // ここでは表面のみ発光としておく
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
