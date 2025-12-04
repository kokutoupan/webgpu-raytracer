// shader.wgsl

// 出力先テクスチャ
@group(0) @binding(0) var outputTex: texture_storage_2d<rgba8unorm, write>;

// --- 1. Rayの定義 ---
struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
}

struct Sphere {
    center: vec3<f32>,
    radius: f32,
    // マテリアルIDなどは後で追加
}

struct HitRecord{
    p: vec3<f32>,
    normal: vec3<f32>,
    t: f32,
    front_face: bool
}

// ある地点 P(t) = A + tb を求める関数
fn ray_at(r: Ray, t: f32) -> vec3<f32> {
    return r.origin + t * r.direction;
}


fn hit_sphere(s: Sphere, r: Ray, t_min: f32, t_max: f32, rec: ptr<function, HitRecord>) -> bool {
    let oc = r.origin - s.center;
    let a = dot(r.direction, r.direction);
    let h = dot(r.direction, oc);
    let c = dot(oc, oc) - s.radius * s.radius;
    let discriminant = h * h - a * c;

    if (discriminant < 0.0) {
        return false;
    }

    let sqrtd = sqrt(discriminant);
    
    // 近くの交点 (-sqrt) を試す
    var root = (-h - sqrtd) / a;
    if (root <= t_min || t_max <= root) {
        // ダメなら遠くの交点 (+sqrt) を試す
        root = (-h + sqrtd) / a;
        if (root <= t_min || t_max <= root) {
            return false;
        }
    }

    // ヒット確定！ recを書き換える
    (*rec).t = root;
    (*rec).p = ray_at(r, root);
    let outward_normal = ((*rec).p - s.center) / s.radius;
    
    // set_face_normal のロジック (内側から当たったかの判定)
    (*rec).front_face = dot(r.direction, outward_normal) < 0.0;
    (*rec).normal = select(-outward_normal, outward_normal, (*rec).front_face);

    return true;
}

fn ray_color(r_in: Ray) -> vec3<f32> {
    // --- 世界の定義 (ハードコード) ---
    var spheres = array<Sphere, 2>(
        Sphere(vec3<f32>(0.0, 0.0, -1.0), 0.5),    // 真ん中の球
        Sphere(vec3<f32>(0.0, -100.5, -1.0), 100.0) // 地面（巨大な球）
    );

    // --- ループ処理 ---
    var rec: HitRecord;
    var hit_anything = false;
    var closest_so_far = 9999999.0; // 無限大の代わり
    let t_min = 0.001;

    // 配列をループ (C++の hittable_list::hit と同じロジック)
    for (var i = 0u; i < 2u; i++) {
        let s = spheres[i];
        // &rec でポインタを渡す
        if (hit_sphere(s, r_in, t_min, closest_so_far, &rec)) {
            hit_anything = true;
            closest_so_far = rec.t;
        }
    }

    // --- 色の決定 ---
    if (hit_anything) {
        // 法線を色にする (-1~1 -> 0~1)
        return 0.5 * (rec.normal + vec3<f32>(1.0, 1.0, 1.0));
    }

    // 背景 (空)
    let unit_direction = normalize(r_in.direction);
    let t = 0.5 * (unit_direction.y + 1.0);
    return mix(vec3<f32>(1.0), vec3<f32>(0.5, 0.7, 1.0), t);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(outputTex);
    let x = id.x;
    let y = id.y;

    if x >= dims.x || y >= dims.y {
        return;
    }

    // 画像のアスペクト比を計算
    let aspect_ratio = f32(dims.x) / f32(dims.y);

    // --- 2. シンプルなカメラのセットアップ ---
    // 週末レイトレーシングの定義値
    let viewport_height = 2.0;
    let viewport_width = aspect_ratio * viewport_height;
    let focal_length = 1.0;

    let origin = vec3<f32>(0.0, 0.0, 0.0);
    let horizontal = vec3<f32>(viewport_width, 0.0, 0.0);
    let vertical = vec3<f32>(0.0, viewport_height, 0.0);

    // 左下隅の座標を計算
    let lower_left_corner = origin - (horizontal / 2.0) - (vertical / 2.0) - vec3<f32>(0.0, 0.0, focal_length);

    // --- UV座標の計算 ---
    // u: 0 (左) -> 1 (右)
    let u = f32(x) / f32(dims.x - 1u);
    
    // v: 0 (下) -> 1 (上)
    // WebGPUのテクスチャ座標(y)は「上が0」なので、
    // C++のレイトレ座標系(左下が原点)に合わせるために反転させます
    let v = 1.0 - (f32(y) / f32(dims.y - 1u));

    // --- レイの生成 ---
    // direction = lower_left_corner + u*horizontal + v*vertical - origin
    let direction = lower_left_corner + (u * horizontal) + (v * vertical) - origin;
    let r = Ray(origin, direction);

    // 色を計算
    let pixel_color = ray_color(r);

    let coords = vec2<u32>(x, y);
    // 書き込み
    textureStore(outputTex, vec2<i32>(coords), vec4<f32>(pixel_color, 1.0));
}
