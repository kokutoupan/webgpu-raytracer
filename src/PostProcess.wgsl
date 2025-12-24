// =========================================================
//   Post Process (PostProcess.wgsl)
// =========================================================

struct SceneUniforms {
    camera_pad: array<vec4<f32>, 6>, // Skip camera data (96 bytes)
    frame_count: u32,
    blas_base_idx: u32,
    vertex_count: u32,
    rand_seed: u32,
    light_count: u32,
    width: u32,
    height: u32
}

@group(0) @binding(0) var outputTex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<storage, read> accumulateBuffer: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> scene: SceneUniforms;

fn aces_tone_mapping(color: vec3<f32>) -> vec3<f32> {
    let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), vec3(0.0), vec3(1.0));
}

fn get_radiance(coord: vec2<u32>) -> vec3<f32> {
    if coord.x >= scene.width || coord.y >= scene.height { return vec3(0.0); }
    let p_idx = coord.y * scene.width + coord.x;
    let acc = accumulateBuffer[p_idx];
    // サンプル数が0なら黒を返す（ゼロ除算防止）
    if acc.a <= 0.0 { return vec3(0.0); }
    return acc.rgb / acc.a;
}

// 輝度計算用ヘルパー
fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= scene.width || id.y >= scene.height { return; }

    let center_pos = id.xy;
    var center_color = get_radiance(center_pos);

    // ----------------------------------------------------------------
    // 1. Firefly Removal (Neighborhood Clamping)
    // バイラテラルフィルタの前に、異常な輝点を取り除く
    // ----------------------------------------------------------------
    var max_neighbor_lum = 0.0;
    
    // 3x3の近傍をチェック
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            if dx == 0 && dy == 0 { continue; } // 自分は含めない

            let np = vec2<i32>(center_pos) + vec2<i32>(dx, dy);
            // 簡易境界チェック（厳密には範囲外は除外すべきだが、get_radianceが黒を返すので安全）
            let neighbor_color = get_radiance(vec2<u32>(np));

            max_neighbor_lum = max(max_neighbor_lum, luminance(neighbor_color));
        }
    }

    let center_lum = luminance(center_color);
    // 「周囲の最大輝度の1.2倍」よりも明るければ、強制的に暗くする
    // シーンが真っ暗な場合のゼロ除算/誤動作防止のために max(..., 1.0) を入れる
    let threshold = max(max_neighbor_lum * 1.2, 1.0);

    if center_lum > threshold {
        // 色味を保ったまま輝度を下げる
        center_color *= (threshold / center_lum);
    }

    // ----------------------------------------------------------------
    // 2. Bilateral Filter (Denoising)
    // 修正された center_color を基準にしてフィルタリングを行う
    // ----------------------------------------------------------------
    
    // Filter Parameters
    let SIGMA_S = 1.0;    // Spatial sigma (少し広げました: 0.8 -> 1.0)
    let SIGMA_R = 0.2;    // Range sigma (色は厳しく判定)
    let RADIUS = 2;       // 5x5 window

    var weighted_sum = vec3(0.0);
    var total_weight = 0.0;

    for (var dy = -RADIUS; dy <= RADIUS; dy++) {
        for (var dx = -RADIUS; dx <= RADIUS; dx++) {
            let neighbor_pos = vec2<i32>(center_pos) + vec2<i32>(dx, dy);

            if neighbor_pos.x < 0 || neighbor_pos.x >= i32(scene.width) || neighbor_pos.y < 0 || neighbor_pos.y >= i32(scene.height) {
                continue;
            }

            // 注: ここで取得するneighborはClampされていない「生の輝度」です。
            // 厳密にはここもClamp済みを使うべきですが、シェーダーの構造上(Read-Write競合)できないため
            // 「中心ピクセルがClampされている」ことで十分重みが調整され、効果が出ます。
            let neighbor_color = get_radiance(vec2<u32>(neighbor_pos));

            // Spatial weight
            let spatial_dist_sq = f32(dx * dx + dy * dy);
            let w_s = exp(-spatial_dist_sq / (2.0 * SIGMA_S * SIGMA_S));

            // Range weight (Clampされた中心色との差分を見るのが重要！)
            let color_diff = neighbor_color - center_color;
            let color_dist_sq = dot(color_diff, color_diff);
            let w_r = exp(-color_dist_sq / (2.0 * SIGMA_R * SIGMA_R));

            let w = w_s * w_r;
            weighted_sum += neighbor_color * w;
            total_weight += w;
        }
    }

    // 重みの合計が極小になった場合のゼロ除算対策
    let hdr_color = weighted_sum / max(total_weight, 1e-4);
    
    // 3. Tone Mapping & Gamma
    let mapped = aces_tone_mapping(hdr_color);
    let out = pow(mapped, vec3(1.0 / 2.2));

    textureStore(outputTex, vec2<i32>(id.xy), vec4(out, 1.));
}