// --- 0. Bindings & Uniforms ---

// Binding 0: 表示用テクスチャ (sRGB)
@group(0) @binding(0) var outputTex: texture_storage_2d<rgba8unorm, write>;

// Binding 1: 蓄積用バッファ (Linear, Read-Write)
@group(0) @binding(1) var<storage, read_write> accumulateBuffer: array<vec4<f32>>;

// Binding 2: フレーム情報 (毎フレーム更新)
struct FrameInfo {
    frame_count: u32,
    // padding...
}
@group(0) @binding(2) var<uniform> frame: FrameInfo;

// Binding 3: カメラ情報 (固定/操作時のみ更新)
// vec3 は 16byte アライメントされるため、TypeScript側もそれに合わせる
struct Camera {
    origin: vec3<f32>,
    lower_left_corner: vec3<f32>,
    horizontal: vec3<f32>,
    vertical: vec3<f32>,
}
@group(0) @binding(3) var<uniform> camera: Camera;


// --- 1. Constants & Structs ---

const PI = 3.141592653589793;

struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
}

struct Sphere {
    center: vec3<f32>,
    radius: f32,
}

struct HitRecord {
    p: vec3<f32>,
    normal: vec3<f32>,
    t: f32,
    front_face: bool,
}

struct RandomGenerator {
    state: u32,
}

// --- 2. Random Functions ---

fn rand_pcg(rng: ptr<function, RandomGenerator>) -> f32 {
    let old_state = (*rng).state;
    (*rng).state = old_state * 747796405u + 2891336453u;
    let word = ((*rng).state >> ((old_state >> 28u) + 4u)) ^ (*rng).state;
    let result = word * 277803737u;
    let final_hash = (result >> 22u) ^ result;
    return f32(final_hash) / 4294967295.0;
}

fn random_unit_vector(rng: ptr<function, RandomGenerator>) -> vec3<f32> {
    let z = rand_pcg(rng) * 2.0 - 1.0;
    let a = rand_pcg(rng) * 2.0 * PI;
    let r = sqrt(max(0.0, 1.0 - z * z));
    let x = r * cos(a);
    let y = r * sin(a);
    return vec3<f32>(x, y, z);
}

// --- 3. Geometry Functions ---

fn ray_at(r: Ray, t: f32) -> vec3<f32> {
    return r.origin + t * r.direction;
}

fn hit_sphere(s: Sphere, r: Ray, t_min: f32, t_max: f32, rec: ptr<function, HitRecord>) -> bool {
    let oc = r.origin - s.center;
    let a = dot(r.direction, r.direction);
    let h = dot(r.direction, oc);
    let c = dot(oc, oc) - s.radius * s.radius;
    let discriminant = h * h - a * c;

    if (discriminant < 0.0) { return false; }

    let sqrtd = sqrt(discriminant);
    var root = (-h - sqrtd) / a;
    if (root <= t_min || t_max <= root) {
        root = (-h + sqrtd) / a;
        if (root <= t_min || t_max <= root) {
            return false;
        }
    }

    (*rec).t = root;
    (*rec).p = ray_at(r, root);
    let outward_normal = ((*rec).p - s.center) / s.radius;
    (*rec).front_face = dot(r.direction, outward_normal) < 0.0;
    (*rec).normal = select(-outward_normal, outward_normal, (*rec).front_face);

    return true;
}

// --- 4. Ray Tracing Logic ---

fn ray_color(r_in: Ray, rng: ptr<function, RandomGenerator>) -> vec3<f32> {
    var spheres = array<Sphere, 2>(
        Sphere(vec3<f32>(0.0, 0.0, -1.0), 0.5),
        Sphere(vec3<f32>(0.0, -100.5, -1.0), 100.0)
    );

    var ray = r_in;
    var color = vec3<f32>(0.0);
    var throughput = vec3<f32>(1.0);

    for (var depth = 0u; depth < 10u; depth++) {
        var rec: HitRecord;
        var hit_anything = false;
        var closest_so_far = 9999999.0;
        let t_min = 0.001;

        for (var i = 0u; i < 2u; i++) {
            if (hit_sphere(spheres[i], ray, t_min, closest_so_far, &rec)) {
                hit_anything = true;
                closest_so_far = rec.t;
            }
        }

        if (hit_anything) {
            let new_dir = rec.normal + random_unit_vector(rng);
            ray = Ray(rec.p, new_dir);
            throughput *= 0.5;
            if (length(throughput) < 0.001) { break; }
        } else {
            let unit_direction = normalize(ray.direction);
            let t = 0.5 * (unit_direction.y + 1.0);
            let sky_color = mix(vec3<f32>(1.0), vec3<f32>(0.5, 0.7, 1.0), t);
            color += throughput * sky_color;
            break;
        }
    }
    return color;
}

// --- 5. Main Compute Shader ---

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(outputTex);
    if (id.x >= dims.x || id.y >= dims.y) { return; }

    let pixel_idx = id.y * dims.x + id.x;

    // 乱数初期化 (ピクセル位置 + フレーム数でシード変化)
    var rng = RandomGenerator(pixel_idx * 719393u + frame.frame_count * 51234u);

    // --- カメラレイの生成 (Uniform使用) ---
    // CPUで計算したパラメータを使用するため、ここではUV計算とRay生成のみ
    
    // アンチエイリアス用ジッタリング
    let u_offset = rand_pcg(&rng);
    let v_offset = rand_pcg(&rng);
    let u = (f32(id.x) + u_offset) / f32(dims.x);
    let v = 1.0 - ((f32(id.y) + v_offset) / f32(dims.y)); // 上下反転

    // Ray = Origin + u*Horizontal + v*Vertical - Origin (方向のみ)
    // lower_left_cornerには既に (origin - horizontal/2 - vertical/2 - w) が入っている
    let direction = camera.lower_left_corner 
                  + (u * camera.horizontal) 
                  + (v * camera.vertical) 
                  - camera.origin;

    let r = Ray(camera.origin, direction);

    // --- 色計算 ---
    // 蓄積レンダリング時は spp=1 で十分
    let pixel_color_linear = ray_color(r, &rng);

    // --- 蓄積処理 (Linear Space) ---
    var accumulated = vec4<f32>(0.0);
    if (frame.frame_count > 1u) {
        accumulated = accumulateBuffer[pixel_idx];
    }
    
    let new_accumulated = accumulated + vec4<f32>(pixel_color_linear, 1.0);
    accumulateBuffer[pixel_idx] = new_accumulated;

    // --- 表示処理 (Gamma Correction) ---
    let frame_f32 = f32(frame.frame_count);
    var final_color = new_accumulated.rgb / frame_f32;

    // Linear -> sRGB (簡易ガンマ 2.2近似 = sqrt)
    final_color = sqrt(final_color);
    final_color = clamp(final_color, vec3<f32>(0.0), vec3<f32>(1.0));

    textureStore(outputTex, vec2<i32>(id.xy), vec4<f32>(final_color, 1.0));
}
