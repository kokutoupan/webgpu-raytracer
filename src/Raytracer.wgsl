// =========================================================
//   WebGPU Ray Tracer (Raytracer.wgsl)
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

// ReSTIR用のリザーバ (候補を保持する箱)
struct Reservoir {
    w_sum: f32,      // 重みの合計
    M: f32,          // 見てきた候補数
    W: f32,          // 最終的なUnbiasedウェイト
    light_idx: u32,  // 勝ち残ったライトのID
    // 選ばれたライト上のサンプリング位置を再現するための乱数
    r_u: f32,
    r_v: f32
}

struct ReservoirData {
    w_sum: f32,
    M: f32,     // 候補数 (u32でもいいが計算上f32が楽)
    W: f32,     // Unbiased Weight
    light_idx: u32,
    r_u: f32,   // サンプリング乱数復元用
    r_v: f32,
    pad1: f32,  // アライメント用
    pad2: f32
}

// 簡易的なライト情報の入れ物（シャドウレイ前）
struct LightCandidate {
    L: vec3<f32>,       // 発光強度
    pos: vec3<f32>,     // ライト上の点の位置
    normal: vec3<f32>,  // ライト上の点の法線
    pdf: f32            // 選ばれる確率
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

@group(0) @binding(10) var<storage, read_write> reservoirs: array<ReservoirData>;

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

    return ScatterResult(l, pdf, throughput, false);
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


// 既存のリザーバ(target)に、別のリザーバ(src)をマージする
fn merge_reservoir(
    target_reservoir: ptr<function, Reservoir>,
    src: Reservoir,
    p_hat_src: f32, // srcの勝者ライトの評価値
    rng: ptr<function, u32>
) -> bool {
    // マージ時の重み: src.W * src.M * p_hat
    // 直感的には「srcが持っている情報の信頼度」
    
    // Mの上限キャップ（重要！）
    // これがないとMが無限に増えて、新しい情報が入りにくくなる（Temporal Lagの原因）
    let M_clamped = min(src.M, 20.0);

    let weight = p_hat_src * src.W * M_clamped;

    (*target_reservoir).w_sum += weight;
    (*target_reservoir).M += M_clamped;

    if rand_pcg(rng) * (*target_reservoir).w_sum < weight {
        (*target_reservoir).light_idx = src.light_idx;
        (*target_reservoir).r_u = src.r_u;
        (*target_reservoir).r_v = src.r_v;
        return true;
    }
    return false;
}

// リザーバを更新する関数
// light_idx: 候補のライトID
// weight: その候補の重み (p_hat / source_pdf)
// c: 候補の個数 (通常は1.0)
fn update_reservoir(res: ptr<function, Reservoir>, light_idx: u32, r_u: f32, r_v: f32, weight: f32, c: f32, rng: ptr<function, u32>) -> bool {
    (*res).w_sum += weight;
    (*res).M += c;

    // 確率的に入れ替える (重いほど選ばれやすい)
    if rand_pcg(rng) * (*res).w_sum < weight {
        (*res).light_idx = light_idx;
        (*res).r_u = r_u;
        (*res).r_v = r_v;
        return true;
    }
    return false;
}


// 指定したライトIDと乱数(r_u, r_v)を使って、ライト上の点と明るさを計算する
// ※ここでは「可視性(壁の裏かどうか)」はチェックしません！
fn evaluate_light_sample(light_idx: u32, r_u: f32, r_v: f32) -> LightCandidate {
    let l_ref = lights[light_idx];
    let tri = topology[l_ref.tri_idx];
    let inst = instances[l_ref.inst_idx];
    let m = get_transform(inst);

    // 頂点座標
    let v0 = (m * vec4(get_pos(tri.v0), 1.0)).xyz;
    let v1 = (m * vec4(get_pos(tri.v1), 1.0)).xyz;
    let v2 = (m * vec4(get_pos(tri.v2), 1.0)).xyz;

    // 重心座標計算 (r_u, r_v から u, v, w を作る)
    let sqrt_r1 = sqrt(r_u);
    let u = 1.0 - sqrt_r1;
    let v = r_v * sqrt_r1;
    let w = 1.0 - u - v;

    let p = v0 * u + v1 * v + v2 * w;
    
    // 法線と面積
    let edge1 = v1 - v0;
    let edge2 = v2 - v0;
    let cross_e = cross(edge1, edge2);
    let area = length(cross_e) * 0.5;
    let n = normalize(cross_e);

    // 発光色取得 (テクスチャ対応)
    let mat_type = bitcast<u32>(tri.data0.w);
    var albedo = tri.data0.rgb;
    var emissive = tri.data3.rgb;

    // マテリアルタイプに応じて発光色を決定
    // mat_type 3u (Light) なら Albedo を、それ以外なら Emissive を使う
    var L = select(emissive, albedo, mat_type == 3u);

    let base_tex = tri.data2.x;
    if base_tex > -0.5 {
        let uv0 = get_uv(tri.v0);
        let uv1 = get_uv(tri.v1);
        let uv2 = get_uv(tri.v2);
        let tex_uv = uv0 * u + uv1 * v + uv2 * w;
        L *= textureSampleLevel(tex, smp, tex_uv, i32(base_tex), 0.0).rgb;
    }
    
    // PDF = 1 / (Area * TotalLights)
    // 面積によるPDF。ライト選択確率(1/N)も含める
    let pdf = 1.0 / (area * f32(scene.light_count));

    return LightCandidate(L, p, n, pdf);
}

fn evaluate_p_hat(hit_p: vec3<f32>, normal: vec3<f32>, light: LightCandidate) -> f32 {
    let l_vec = light.pos - hit_p;
    let dist_sq = dot(l_vec, l_vec);
    let dist = sqrt(dist_sq);
    let dir = l_vec / dist;

    let cos_light = max(dot(light.normal, -dir), 0.0);
    let cos_surf = max(dot(normal, dir), 0.0);
    
    // 輝度
    let intensity = dot(light.L, vec3(0.299, 0.587, 0.114));

    return (intensity * cos_light * cos_surf) / max(dist_sq, 1e-4);
}

// RIS (Resampled Importance Sampling) を使ったライトサンプリング
// hit_p: レイが当たった場所
// normal: レイが当たった場所の法線
fn sample_lights_restir_reuse(hit_p: vec3<f32>, normal: vec3<f32>, p_idx: u32, rng: ptr<function, u32>) -> LightSample {
    let stride = scene.width * scene.height;
    // --- Phase 1: 初期候補 (RIS) ---
    // 前回のコードと同じ（32回ループして1個選ぶ）
    // 結果を `state` (Reservoir型) に保持
    let frame_mod = scene.frame_count % 2u;

    let read_offset = frame_mod * stride;           // 0 or stride
    let write_offset = (1u - frame_mod) * stride;

    var state: Reservoir;
    state.w_sum = 0.0; state.M = 0.0; state.W = 0.0;

    let CANDIDATE_COUNT = 4u; // 再利用するなら初期候補は減らしてもOK (32 -> 4)
    for (var i = 0u; i < CANDIDATE_COUNT; i++) {
        let light_idx = u32(rand_pcg(rng) * f32(scene.light_count));
        let r_u = rand_pcg(rng);
        let r_v = rand_pcg(rng);
        let candidate = evaluate_light_sample(light_idx, r_u, r_v);

        let p_hat = evaluate_p_hat(hit_p, normal, candidate);
        let weight = p_hat / max(candidate.pdf, 1e-6);

        update_reservoir(&state, light_idx, r_u, r_v, weight, 1.0, rng);
    }
    
    // RISで選ばれた候補の p_hat を計算しておく（後で使う）
    // ※最適化: update_reservoir内で保存しておくと速い
    var p_hat_current = 0.0;
        {
        let winner = evaluate_light_sample(state.light_idx, state.r_u, state.r_v);
        p_hat_current = evaluate_p_hat(hit_p, normal, winner);
        
        // RIS段階での W を仮計算
        if p_hat_current > 0.0 {
            state.W = state.w_sum / (state.M * p_hat_current);
        } else {
            state.W = 0.0;
        }
    }


    // --- Phase 2: 時間的再利用 (Temporal Reuse) ---
    // バッファから前回の自分を読み込む

    let prev_idx = read_offset + p_idx;
    let prev_data = reservoirs[prev_idx];
    
    // 前回のデータが有効かチェック（カメラが動いていない前提ならそのまま使える）
    // ※厳密にはここでリプロジェクションや、法線・深度の類似度チェックが必要
    // 今回は「なし」で突っ込む（多少の残像は許容）

    var prev_res: Reservoir;
    prev_res.light_idx = prev_data.light_idx;
    prev_res.W = prev_data.W;
    prev_res.M = prev_data.M;
    prev_res.r_u = prev_data.r_u;
    prev_res.r_v = prev_data.r_v;
    
    // 前回の勝者が、今の自分にとってどれくらい嬉しいか (p_hat) を再評価
    let prev_winner = evaluate_light_sample(prev_res.light_idx, prev_res.r_u, prev_res.r_v);
    let p_hat_prev = evaluate_p_hat(hit_p, normal, prev_winner);
    
    // 影チェックはまだしない！p_hat > 0 ならマージする
    if p_hat_prev > 0.0 {
        merge_reservoir(&state, prev_res, p_hat_prev, rng);
        // 勝者が入れ替わったかもしれないので p_hat_current を更新したいが、
        // 厳密には最後に1回やればいい
    }


    // --- Phase 3: 空間的再利用 (Spatial Reuse) ---
    // 隣のピクセルをランダムに選んでマージ

    let SPATIAL_COUNT = 2u; // 2〜3近傍を見る
    let RADIUS = 30.0; // 半径30ピクセルくらい広めに探す

    for (var i = 0u; i < SPATIAL_COUNT; i++) {
        // ランダムなオフセット
        let offset = random_in_unit_disk(rng) * RADIUS;
        let nx = i32(f32(scene.width) * 0.0 + f32(p_idx % scene.width) + offset.x); // 簡易計算
        let ny = i32(f32(scene.height) * 0.0 + f32(p_idx / scene.width) + offset.y);
        
        // 画面外チェック
        if nx < 0 || nx >= i32(scene.width) || ny < 0 || ny >= i32(scene.height) { continue; }

        let n_idx = u32(ny * i32(scene.width) + nx);
        let neighbor_data = reservoirs[read_offset + n_idx];

        var n_res: Reservoir;
        n_res.light_idx = neighbor_data.light_idx;
        n_res.W = neighbor_data.W;
        n_res.M = neighbor_data.M;
        n_res.r_u = neighbor_data.r_u;
        n_res.r_v = neighbor_data.r_v;
        
        // 幾何学的類似度チェック（法線が違いすぎるならマージしない）
        // ※バッファに法線が入っていないので、今回はスキップして「全部混ぜる」
        // （角で光が漏れる原因になるが、まずは動かす優先）

        let n_winner = evaluate_light_sample(n_res.light_idx, n_res.r_u, n_res.r_v);
        let p_hat_n = evaluate_p_hat(hit_p, normal, n_winner);

        if p_hat_n > 0.0 {
            merge_reservoir(&state, n_res, p_hat_n, rng);
        }
    }


    // --- Phase 4: 最終決定と保存 ---
    
    // 最終的な勝者の p_hat
    let final_winner = evaluate_light_sample(state.light_idx, state.r_u, state.r_v);
    let p_hat_final = evaluate_p_hat(hit_p, normal, final_winner);
    
    // Unbiased Weight (W)
    if p_hat_final > 0.0 {
        state.W = state.w_sum / (state.M * p_hat_final);
    } else {
        state.W = 0.0;
    }
    
    // バッファに保存（次フレーム用）
    var store_data: ReservoirData;
    store_data.w_sum = state.w_sum; // ※実は保存不要だがデバッグ用に
    store_data.M = state.M;
    store_data.W = state.W;
    store_data.light_idx = state.light_idx;
    store_data.r_u = state.r_u;
    store_data.r_v = state.r_v;
    let out_idx = write_offset + p_idx;
    reservoirs[out_idx] = store_data;

    // --- Phase 5: シャドウレイ用のサンプル返却 ---
    
    // ここで初めて「可視性（Shadow）」をチェックされることになる
    // 返り値はLightSample型
    var sample_out: LightSample;
    sample_out.L = final_winner.L;
    let l_vec_final = final_winner.pos - hit_p;
    let dist_sq_final = dot(l_vec_final, l_vec_final);
    let dist_final = sqrt(dist_sq_final);
    
    // ゼロ除算対策
    if dist_final > 1e-6 {
        sample_out.dir = l_vec_final / dist_final;
        sample_out.dist = dist_final;
    } else {
        sample_out.dir = vec3(0.0, 1.0, 0.0); // ダミー
        sample_out.dist = 0.0;
    }
    
    // PDFトリック:
    // 従来の ray_color は (L / pdf) で計算している。
    // ReSTIRは (L * W) で計算したい。
    // つまり pdf = 1.0 / W と偽れば、既存コードを変えずに済む。

    if state.W > 0.0 {
        sample_out.pdf = 1.0 / state.W;
    } else {
        sample_out.pdf = 0.0;
    }

    return sample_out;
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
    var stack: array<u32, 24>;
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
//   Main Path Tracer Loop
// =========================================================

fn ray_color(r_in: Ray, p_idx: u32, rng: ptr<function, u32>) -> vec3<f32> {
    var ray = r_in;
    var throughput = vec3(1.0);
    var radiance = vec3(0.0);

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
        var is_specular = (mat_type == 2u) || (metallic > 0.9 && roughness < 0.3); 

        // --- NEE ---
        if !is_specular && mat_type != 2u {
            let light_s = sample_lights_restir_reuse(hit_p, normal, p_idx, rng);
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
    var rng = init_rng(p_idx, scene.rand_seed);

    var col = vec3(0.);
    for (var s = 0u; s < SPP; s++) {
        var off = vec3(0.);
        if scene.camera.origin.w > 0. {
            let rd = scene.camera.origin.w * random_in_unit_disk(&rng);
            off = scene.camera.u.xyz * rd.x + scene.camera.v.xyz * rd.y;
        }
        let u = (f32(id.x) + 0.5 + scene.jitter.x * f32(scene.width)) / f32(scene.width);
        let v = 1. - (f32(id.y) + 0.5 + scene.jitter.y * f32(scene.height)) / f32(scene.height);
        let d = scene.camera.lower_left_corner.xyz + u * scene.camera.horizontal.xyz + v * scene.camera.vertical.xyz - scene.camera.origin.xyz - off;
        col += ray_color(Ray(scene.camera.origin.xyz + off, d), p_idx, &rng);
    }
    col /= f32(SPP);

    var acc = vec4(0.);
    if scene.frame_count > 1u { acc = accumulateBuffer[p_idx]; }
    accumulateBuffer[p_idx] = acc + vec4(col, 1.0);
}
