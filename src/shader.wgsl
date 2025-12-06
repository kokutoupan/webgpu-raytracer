// =========================================================
//   WebGPU Ray Tracer (Refactored)
// =========================================================

// --- Constants ---
const PI = 3.141592653589793;
const T_MIN = 0.001;
const T_MAX = 1e30;
const MAX_DEPTH = 5u;
const SPHERES_BINDING_STRIDE = 3u; // 1球あたり vec4 が 3つ
const TRIANGLES_BINDING_STRIDE = 4u;

// --- Bindings ---
@group(0) @binding(0) var outputTex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<storage, read_write> accumulateBuffer: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> frame: FrameInfo;
@group(0) @binding(3) var<uniform> camera: Camera;
// vec4配列として読み込むことでアライメント問題を回避
@group(0) @binding(4) var<storage, read> scene_spheres_packed: array<vec4<f32>>;

// --- Binding 5: Packed Triangles ---
// vec4 x 4 = 64 bytes stride
// 0: v0(xyz) + extra(w)  <-- ★ここにextraを入れる
// 1: v1(xyz) + pad
// 2: v2(xyz) + pad
// 3: color(rgb) + mat_type(w)
@group(0) @binding(5) var<storage, read> scene_triangles_packed: array<vec4<f32>>;


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

// 論理的な球データ構造 (GPUメモリ上では packed 配列)
struct Sphere {
    center: vec3<f32>,
    radius: f32,
    color: vec3<f32>,
    mat_type: f32, // 0:Lambertian, 1:Metal, 2:Dielectric
    extra: f32,    // Fuzz or IOR
}


struct Triangle {
    v0: vec3<f32>,
    v1: vec3<f32>,
    v2: vec3<f32>,
    color: vec3<f32>,
    mat_type: f32,
    extra: f32, // Fuzz or IOR
}


// --- Helper Functions: Data Access ---

// パッキングされたバッファから球データを復元
// 
fn get_sphere(index: u32) -> Sphere {
    let i = index * SPHERES_BINDING_STRIDE;
    let d1 = scene_spheres_packed[i];      // center(xyz), radius(w)
    let d2 = scene_spheres_packed[i + 1u]; // color(xyz), mat_type(w)
    let d3 = scene_spheres_packed[i + 2u]; // extra(x), padding...
    return Sphere(d1.xyz, d1.w, d2.xyz, d2.w, d3.x);
}

fn get_triangle(index: u32) -> Triangle {
    let i = index * 4u; // 1つの三角形につき vec4 が 4つ
    let d0 = scene_triangles_packed[i];
    let d1 = scene_triangles_packed[i + 1u];
    let d2 = scene_triangles_packed[i + 2u];
    let d3 = scene_triangles_packed[i + 3u];

    return Triangle(
        d0.xyz, // v0
        d1.xyz, // v1
        d2.xyz, // v2
        d3.xyz, // color
        d3.w,   // mat_type
        d0.w    // extra (v0の隙間に格納されていたものを取り出す)
    );
}
// --- Helper Functions: RNG (PCG Hash) ---

// ピクセル座標とフレーム数からユニークなシードを生成
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

// 0.0 ~ 1.0 の乱数を生成
fn rand_pcg(state: ptr<function, u32>) -> f32 {
    let old = *state;
    *state = old * 747796405u + 2891336453u;
    let word = ((*state) >> ((old >> 28u) + 4u)) ^ (*state);
    return f32((word >> 22u) ^ word) / 4294967295.0;
}

// 球面上のランダムな点 (Lambertian/Metal用)
fn random_unit_vector(rng: ptr<function, u32>) -> vec3<f32> {
    let z = rand_pcg(rng) * 2.0 - 1.0;
    let a = rand_pcg(rng) * 2.0 * PI;
    let r = sqrt(max(0.0, 1.0 - z * z));
    return vec3<f32>(r * cos(a), r * sin(a), z);
}

// 単位円盤内のランダムな点 (被写界深度用)
fn random_in_unit_disk(rng: ptr<function, u32>) -> vec3<f32> {
    let r = sqrt(rand_pcg(rng));
    let theta = 2.0 * PI * rand_pcg(rng);
    return vec3<f32>(r * cos(theta), r * sin(theta), 0.0);
}

// --- Helper Functions: Physics ---

// Schlickの近似式 (フレネル反射率)
fn reflectance(cosine: f32, ref_idx: f32) -> f32 {
    var r0 = (1.0 - ref_idx) / (1.0 + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0 - r0) * pow((1.0 - cosine), 5.0);
}

// 球とレイの交差判定 (距離tを返す。ヒットしなければ -1.0)
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

// --- 三角形の交差判定 (Möller–Trumbore algorithm) ---
fn hit_triangle_t(tri: Triangle, r: Ray, t_min: f32, t_max: f32) -> f32 {
    let edge1 = tri.v1 - tri.v0;
    let edge2 = tri.v2 - tri.v0;
    let h = cross(r.direction, edge2);
    let a = dot(edge1, h);

    // レイが三角形と平行に近い場合
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

// --- Main Ray Tracing Logic ---

fn ray_color(r_in: Ray, rng: ptr<function, u32>) -> vec3<f32> {
    var ray = r_in;
    var throughput = vec3<f32>(1.0);
    let sphere_count = arrayLength(&scene_spheres_packed) / SPHERES_BINDING_STRIDE;
    let triangle_count = arrayLength(&scene_triangles_packed) / TRIANGLES_BINDING_STRIDE;

    for (var depth = 0u; depth < MAX_DEPTH; depth++) {
        var hit_anything = false;
        var closest_t = T_MAX;
        var hit_idx = 0u;
        var hit_type = 0u; // 0:None, 1:Sphere, 2:Triangle

        // 全球探索 (単純ループ)
        for (var i = 0u; i < sphere_count; i++) {
            let s = get_sphere(i);
            let t = hit_sphere_t(s, ray, T_MIN, closest_t);
            if t > 0.0 {
                hit_anything = true;
                closest_t = t;
                hit_idx = i;
                hit_type = 1u;
            }
        }

        // 三角形
        for (var i = 0u; i < triangle_count; i++) {
            let tri = get_triangle(i);
            let t = hit_triangle_t(tri, ray, T_MIN, closest_t);
            if t > 0.0 {
                hit_anything = true;
                closest_t = t;
                hit_idx = i;
                hit_type = 2u;
            }
        }

        if hit_anything {
            var p = vec3<f32>(0.0);
            var normal = vec3<f32>(0.0);
            var mat_type = 0.0;
            var color = vec3<f32>(0.0);
            var extra = 0.0;
            var front_face = true;

            if hit_type == 1u {
                // Sphere
                let s = get_sphere(hit_idx);
                p = ray.origin + closest_t * ray.direction;
                let outward_normal = (p - s.center) / s.radius;
                front_face = dot(ray.direction, outward_normal) < 0.0;
                normal = select(-outward_normal, outward_normal, front_face);
                mat_type = s.mat_type;
                color = s.color;
                extra = s.extra;
            } else {
                // Triangle
                let tri = get_triangle(hit_idx);
                p = ray.origin + closest_t * ray.direction;
                // 法線はエッジの外積（簡易計算）
                let e1 = tri.v1 - tri.v0;
                let e2 = tri.v2 - tri.v0;
                let outward_normal = normalize(cross(e1, e2));
                front_face = dot(ray.direction, outward_normal) < 0.0;
                normal = select(-outward_normal, outward_normal, front_face);
                mat_type = tri.mat_type;
                color = tri.color;
                extra = tri.extra;
            }

            var scattered_dir = vec3<f32>(0.0);

            if mat_type > 2.5 {
                return throughput * color; // 強烈に光らせる
            }

            // Material Handling
            if mat_type < 0.5 { 
                // --- Lambertian (Diffuse) ---
                scattered_dir = normal + random_unit_vector(rng);
                // 縮退（ゼロベクトル）対策
                if length(scattered_dir) < 1e-6 { scattered_dir = normal; }
            } else if mat_type < 1.5 { 
                // --- Metal ---
                let reflected = reflect(ray.direction, normal);
                scattered_dir = reflected + extra * random_unit_vector(rng);
            } else { 
                // --- Dielectric (Glass) ---
                let ref_ratio = select(extra, 1.0 / extra, front_face);
                let unit_dir = normalize(ray.direction);

                let cos_theta = min(dot(-unit_dir, normal), 1.0);
                let sin_theta = sqrt(1.0 - cos_theta * cos_theta);
                
                // 全反射条件 または フレネル反射
                let cannot_refract = ref_ratio * sin_theta > 1.0;
                let do_reflect = cannot_refract || (reflectance(cos_theta, ref_ratio) > rand_pcg(rng));

                if do_reflect {
                    scattered_dir = reflect(unit_dir, normal);
                } else {
                    // Refract (Snell's Law)
                    let r_out_perp = ref_ratio * (unit_dir + cos_theta * normal);
                    let r_out_parallel = -sqrt(abs(1.0 - dot(r_out_perp, r_out_perp))) * normal;
                    scattered_dir = r_out_perp + r_out_parallel;
                }
            }

            ray = Ray(p, scattered_dir);
            throughput *= color;

            // --- Russian Roulette (Path Termination) ---
            // 寄与率が低いパスを確率的に打ち切って高速化
            let p_rr = max(throughput.r, max(throughput.g, throughput.b));
            // 常に少しは確率を残すため、例えば0.001以下でも完全には切らない工夫も可能だが
            // ここでは単純なスループットベースで行う
            if rand_pcg(rng) > p_rr { break; }
            throughput /= p_rr; // 生き残ったレイのエネルギーを補正
        } else {
            // // --- Sky Color (Miss) ---
            // let unit_dir = normalize(ray.direction);
            // let t = 0.5 * (unit_dir.y + 1.0);
            // let sky = mix(vec3<f32>(1.0), vec3<f32>(0.5, 0.7, 1.0), t);
            // return throughput * sky;
            return vec3<f32>(.0);
        }
    }
    
    // 最大深度を超えた、またはロシアンルーレットで吸収された場合
    return vec3<f32>(0.0);
}

// --- Compute Shader Entry Point ---

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(outputTex);
    
    // Bounds Check: 画面外スレッドは即時終了
    if id.x >= dims.x || id.y >= dims.y { return; }

    let pixel_idx = id.y * dims.x + id.x;
    
    // RNG Initialization
    var rng = init_rng(pixel_idx, frame.frame_count);

    // Camera Setup & Ray Generation (with Defocus Blur & Jitter)
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

    // Ray Tracing
    let pixel_color = ray_color(r, &rng);

    // Accumulation
    var acc_color = vec4<f32>(0.0);
    if frame.frame_count > 1u {
        acc_color = accumulateBuffer[pixel_idx];
    }

    let new_acc_color = acc_color + vec4<f32>(pixel_color, 1.0);
    accumulateBuffer[pixel_idx] = new_acc_color;

    // Output Processing (Average & Gamma Correction)
    var final_color = new_acc_color.rgb / f32(frame.frame_count);
    // Linear to sRGB approximation (Gamma 2.0/2.2)
    final_color = sqrt(clamp(final_color, vec3<f32>(0.0), vec3<f32>(1.0)));

    textureStore(outputTex, vec2<i32>(id.xy), vec4<f32>(final_color, 1.0));
}
