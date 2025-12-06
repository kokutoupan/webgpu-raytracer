// =========================================================
//   WebGPU Ray Tracer (Refactored)
// =========================================================

// --- Constants ---
const PI = 3.141592653589793;
const T_MIN = 0.001;
const T_MAX = 1e30;
const MAX_DEPTH = 5u;

// Binding Stride (in vec4 units)
const SPHERES_BINDING_STRIDE = 3u;   // 1 sphere = 3 vec4s
const TRIANGLES_BINDING_STRIDE = 4u; // 1 triangle = 4 vec4s

// --- Bindings ---
@group(0) @binding(0) var outputTex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<storage, read_write> accumulateBuffer: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> frame: FrameInfo;
@group(0) @binding(3) var<uniform> camera: Camera;

// Packed Data Buffers (vec4 arrays to ensure alignment)
@group(0) @binding(4) var<storage, read> scene_spheres_packed: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> scene_triangles_packed: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> bvh_nodes: array<BVHNode>;

// Binding 7: Primitive References (新規追加)
// vec2<u32> = { type, index }
@group(0) @binding(7) var<storage, read> primitive_refs: array<vec2<u32>>;

// --- Structs ---

struct FrameInfo {
    frame_count: u32,
}

struct Camera {
    origin: vec3<f32>,
    lens_radius: f32,
    lower_left_corner: vec3<f32>,
    horizontal: vec3<f32>,
    vertical: vec3<f32>,
    u: vec3<f32>,
    v: vec3<f32>,
}

struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
}

struct Sphere {
    center: vec3<f32>,
    radius: f32,
    color: vec3<f32>,
    mat_type: f32,
    extra: f32,
}

struct Triangle {
    v0: vec3<f32>,
    v1: vec3<f32>,
    v2: vec3<f32>,
    color: vec3<f32>,
    mat_type: f32,
    extra: f32,
}

struct BVHNode {
    min_b: vec3<f32>,
    left_first: f32, // index
    max_b: vec3<f32>,
    tri_count: f32,
}

// --- Helper Functions: Data Access ---

fn get_sphere(index: u32) -> Sphere {
    let i = index * SPHERES_BINDING_STRIDE;
    let d1 = scene_spheres_packed[i];      // center(xyz), radius(w)
    let d2 = scene_spheres_packed[i + 1u]; // color(xyz), mat_type(w)
    let d3 = scene_spheres_packed[i + 2u]; // extra(x), padding...
    return Sphere(d1.xyz, d1.w, d2.xyz, d2.w, d3.x);
}

fn get_triangle(index: u32) -> Triangle {
    let i = index * TRIANGLES_BINDING_STRIDE;
    let d0 = scene_triangles_packed[i];      // v0(xyz), extra(w)
    let d1 = scene_triangles_packed[i + 1u]; // v1(xyz), pad
    let d2 = scene_triangles_packed[i + 2u]; // v2(xyz), pad
    let d3 = scene_triangles_packed[i + 3u]; // color(xyz), mat_type(w)
    return Triangle(d0.xyz, d1.xyz, d2.xyz, d3.xyz, d3.w, d0.w);
}

// --- Helper Functions: RNG (PCG Hash) ---

fn init_rng(pixel_idx: u32, frame: u32) -> u32 {
    var seed = pixel_idx + frame * 719393u;
    seed = seed ^ 2747636419u;
    seed = seed * 2654435769u;
    seed = seed ^ (seed >> 16u);
    seed = seed * 2654435769u;
    seed = seed ^ (seed >> 16u);
    seed = seed * 2654435769u;
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

// --- Helper Functions: Physics ---

fn reflectance(cosine: f32, ref_idx: f32) -> f32 {
    var r0 = (1.0 - ref_idx) / (1.0 + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0 - r0) * pow((1.0 - cosine), 5.0);
}

// --- Intersection Functions ---

fn hit_sphere_t(s: Sphere, r: Ray, t_min: f32, t_max: f32) -> f32 {
    let oc = r.origin - s.center;
    let a = dot(r.direction, r.direction);
    let h = dot(r.direction, oc);
    let c = dot(oc, oc) - s.radius * s.radius;
    let discriminant = h * h - a * c;

    if discriminant < 0.0 { return -1.0; }

    let sqrtd = sqrt(discriminant);
    var root = (-h - sqrtd) / a;
    if root <= t_min || t_max <= root {
        root = (-h + sqrtd) / a;
        if root <= t_min || t_max <= root { return -1.0; }
    }
    return root;
}

fn hit_triangle_t(tri: Triangle, r: Ray, t_min: f32, t_max: f32) -> f32 {
    let edge1 = tri.v1 - tri.v0;
    let edge2 = tri.v2 - tri.v0;
    let h = cross(r.direction, edge2);
    let a = dot(edge1, h);

    if abs(a) < 0.0001 { return -1.0; }

    let f = 1.0 / a;
    let s = r.origin - tri.v0;
    let u = f * dot(s, h);

    if u < 0.0 || u > 1.0 { return -1.0; }

    let q = cross(s, edge1);
    let v = f * dot(r.direction, q);

    if v < 0.0 || u + v > 1.0 { return -1.0; }

    let t = f * dot(edge2, q);

    if t > t_min && t < t_max {
        return t;
    }
    return -1.0;
}

// --- BVH Logic ---

// AABB Intersection (Slab Method)
fn hit_aabb(min_b: vec3<f32>, max_b: vec3<f32>, r: Ray, inv_d: vec3<f32>, t_min: f32, t_max: f32) -> bool {
    let t0s = (min_b - r.origin) * inv_d;
    let t1s = (max_b - r.origin) * inv_d;

    let t_smaller = min(t0s, t1s);
    let t_bigger = max(t0s, t1s);

    let t_min_final = max(t_min, max(t_smaller.x, max(t_smaller.y, t_smaller.z)));
    let t_max_final = min(t_max, min(t_bigger.x, min(t_bigger.y, t_bigger.z)));

    return t_min_final <= t_max_final;
}

// BVH Traversal
fn hit_bvh(r: Ray, t_min: f32, t_current_closest: f32) -> vec4<f32> {
    var closest_t = t_current_closest;
    var hit_idx = -1.0;
    var hit_type = 0.0;

    // ★最適化: inv_d はレイごとに不変なので、ループの外で一度だけ計算する
    let inv_d = vec3<f32>(
        select(1.0 / r.direction.x, 1e30, abs(r.direction.x) < 1e-20),
        select(1.0 / r.direction.y, 1e30, abs(r.direction.y) < 1e-20),
        select(1.0 / r.direction.z, 1e30, abs(r.direction.z) < 1e-20)
    );

    var stack: array<u32, 32>;
    var stack_ptr = 0u;

    // Push Root
    stack[stack_ptr] = 0u;
    stack_ptr++;

    while stack_ptr > 0u {
        stack_ptr--;
        let node_idx = stack[stack_ptr];
        let node = bvh_nodes[node_idx];

        if hit_aabb(node.min_b, node.max_b, r, inv_d, t_min, closest_t) {
            let count = u32(node.tri_count);
            let first = u32(node.left_first);

            if count > 0u {
                // --- Leaf Node ---
                for (var i = 0u; i < count; i++) {
                    let ref_idx = first + i;
                    let refp = primitive_refs[ref_idx];

                    let type_id = refp.x;
                    let obj_idx = refp.y;

                    var t = -1.0;

                    if type_id == 0u {
                        // Sphere
                        let s = get_sphere(obj_idx);
                        t = hit_sphere_t(s, r, t_min, closest_t);
                    } else {
                        // Triangle
                        let tri = get_triangle(obj_idx);
                        t = hit_triangle_t(tri, r, t_min, closest_t);
                    }

                    if t > 0.0 {
                        closest_t = t;
                        hit_idx = f32(obj_idx);
                        hit_type = f32(type_id + 1u); // 0:Noneなので +1 して区別
                    }
                }
            } else {
                // --- Internal Node ---
                // Push children
                stack[stack_ptr] = first;
                stack_ptr++;
                stack[stack_ptr] = first + 1u;
                stack_ptr++;
            }
        }
    }

    return vec4<f32>(closest_t, hit_idx, hit_type, 0.);
}

// --- Main Ray Tracing Loop ---

fn ray_color(r_in: Ray, rng: ptr<function, u32>) -> vec3<f32> {
    var ray = r_in;
    var throughput = vec3<f32>(1.0);
    let sphere_count = arrayLength(&scene_spheres_packed) / SPHERES_BINDING_STRIDE;

    for (var depth = 0u; depth < MAX_DEPTH; depth++) {
        var hit_anything = false;
        var closest_t = T_MAX;
        var hit_idx = 0u;
        var hit_type = 0u; // 0:None, 1:Sphere, 2:Triangle


        // 1. BVH
        let bvh_result = hit_bvh(ray, T_MIN, closest_t);
        let t_hit = bvh_result.x;
        let idx_hit = u32(bvh_result.y);
        let type_hit = u32(bvh_result.z); // 1:Sphere, 2:Triangle

        if type_hit > 0u {
            hit_anything = true;
            closest_t = t_hit;
            hit_idx = idx_hit;
            hit_type = type_hit;
        }

        // Shading
        if hit_anything {
            var p = vec3<f32>(0.0);
            var normal = vec3<f32>(0.0);
            var mat_type = 0.0;
            var color = vec3<f32>(0.0);
            var extra = 0.0;
            var front_face = true;

            // Retrieve Hit Data
            if hit_type == 1u { // Sphere
                let s = get_sphere(hit_idx);
                p = ray.origin + closest_t * ray.direction;
                let outward_normal = (p - s.center) / s.radius;
                front_face = dot(ray.direction, outward_normal) < 0.0;
                normal = select(-outward_normal, outward_normal, front_face);
                mat_type = s.mat_type;
                color = s.color;
                extra = s.extra;
            } else { // Triangle
                let tri = get_triangle(hit_idx);
                p = ray.origin + closest_t * ray.direction;
                let e1 = tri.v1 - tri.v0;
                let e2 = tri.v2 - tri.v0;
                let outward_normal = normalize(cross(e1, e2));
                front_face = dot(ray.direction, outward_normal) < 0.0;
                normal = select(-outward_normal, outward_normal, front_face);
                mat_type = tri.mat_type;
                color = tri.color;
                extra = tri.extra;
            }

            // Emissive Material (Light)
            if mat_type > 2.5 {
                return throughput * color; // Boost emission
            }

            // Scatter
            var scattered_dir = vec3<f32>(0.0);

            if mat_type < 0.5 { 
                // Lambertian
                scattered_dir = normal + random_unit_vector(rng);
                if length(scattered_dir) < 1e-6 { scattered_dir = normal; }
            } else if mat_type < 1.5 { 
                // Metal
                let reflected = reflect(ray.direction, normal);
                scattered_dir = reflected + extra * random_unit_vector(rng);
            } else { 
                // Dielectric
                let ref_ratio = select(extra, 1.0 / extra, front_face);
                let unit_dir = normalize(ray.direction);
                let cos_theta = min(dot(-unit_dir, normal), 1.0);
                let sin_theta = sqrt(1.0 - cos_theta * cos_theta);

                let cannot_refract = ref_ratio * sin_theta > 1.0;
                let do_reflect = cannot_refract || (reflectance(cos_theta, ref_ratio) > rand_pcg(rng));

                if do_reflect {
                    scattered_dir = reflect(unit_dir, normal);
                } else {
                    let r_out_perp = ref_ratio * (unit_dir + cos_theta * normal);
                    let r_out_parallel = -sqrt(abs(1.0 - dot(r_out_perp, r_out_perp))) * normal;
                    scattered_dir = r_out_perp + r_out_parallel;
                }
            }

            ray = Ray(p, scattered_dir);
            throughput *= color;

            // Russian Roulette
            let p_rr = max(throughput.r, max(throughput.g, throughput.b));
            if rand_pcg(rng) > p_rr { break; }
            throughput /= p_rr;
        } else {
            // Miss (Background)
            return vec3<f32>(0.0); // Black background for Cornell Box
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

    // Camera
    var ray_orig = camera.origin;
    var ray_off = vec3<f32>(0.0);

    if camera.lens_radius > 0.0 {
        let rd = camera.lens_radius * random_in_unit_disk(&rng);
        ray_off = camera.u * rd.x + camera.v * rd.y;
        ray_orig += ray_off;
    }

    let u_jitter = rand_pcg(&rng);
    let v_jitter = rand_pcg(&rng);
    let s = (f32(id.x) + u_jitter) / f32(dims.x);
    let t = 1.0 - ((f32(id.y) + v_jitter) / f32(dims.y));

    let direction = camera.lower_left_corner + s * camera.horizontal + t * camera.vertical - camera.origin - ray_off;

    let r = Ray(ray_orig, direction);

    // Trace
    let pixel_color = ray_color(r, &rng);

    // Accumulate
    var acc_color = vec4<f32>(0.0);
    if frame.frame_count > 1u {
        acc_color = accumulateBuffer[pixel_idx];
    }
    let new_acc_color = acc_color + vec4<f32>(pixel_color, 1.0);
    accumulateBuffer[pixel_idx] = new_acc_color;

    // Output
    var final_color = new_acc_color.rgb / f32(frame.frame_count);
    final_color = sqrt(clamp(final_color, vec3<f32>(0.0), vec3<f32>(1.0)));

    textureStore(outputTex, vec2<i32>(id.xy), vec4<f32>(final_color, 1.0));
}
