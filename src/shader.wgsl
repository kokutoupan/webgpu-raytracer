// =========================================================
//   WebGPU Ray Tracer (Final Refactored Version)
// =========================================================

// --- Constants ---
const PI = 3.141592653589793;
const T_MIN = 0.001;
const T_MAX = 1e30;
const MAX_DEPTH = 10u; // ガラスなどを綺麗に見せるため少し増やす

// Binding Stride
const SPHERES_STRIDE = 3u;
const TRIANGLES_STRIDE = 4u;

// --- Bindings ---
@group(0) @binding(0) var outputTex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<storage, read_write> accumulateBuffer: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> frame: FrameInfo;
@group(0) @binding(3) var<uniform> camera: Camera;

// Scene Data
// 統合プリミティブバッファ (Binding 4)
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

    let inv_d = 1.0 / r.direction;

    var stack: array<u32, 32>;
    var stackptr = 0u;


    // ルートノード(0)の判定
    let root_dist = intersect_aabb(bvh_nodes[0].min_b, bvh_nodes[0].max_b, r, inv_d, t_min, closest_t);

    // ヒットした時だけ積む
    if root_dist < 1e30 {
        stack[stackptr] = 0u;
        stackptr++;
    }

    while stackptr > 0u {
        stackptr--;
        let node_idx = stack[stackptr];
        let node = bvh_nodes[node_idx];

        let count = u32(node.tri_count);
        let first = u32(node.left_first);

        if count > 0u {
            // Leaf Node
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
        if mat > 2.5 { return throughput * col; }

        var scat = vec3<f32>(0.0);

        // Material Scatter
        if mat < 0.5 { // Lambertian
            scat = normal + random_unit_vector(rng);
            if length(scat) < 1e-6 { scat = normal; }
        
        } else if mat < 1.5 { // Metal
            scat = reflect(ray.direction, normal) + ext * random_unit_vector(rng);
            // ★重要: 内側への反射は吸収（黒）
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

        // ★重要: Shadow Acne 対策 (少し浮かせて再発射)
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

    // Jittered Camera Ray
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
    
    // Trace
    let col = ray_color(Ray(ro, dir), &rng);

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
