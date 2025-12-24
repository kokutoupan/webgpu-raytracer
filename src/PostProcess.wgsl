// =========================================================
//   Post Process (PostProcess.wgsl)
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
    pad2: vec2<f32>
}

@group(0) @binding(0) var outputTex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<storage, read> accumulateBuffer: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> scene: SceneUniforms;
@group(0) @binding(3) var historyTex: texture_2d<f32>;
@group(0) @binding(4) var smp: sampler;
@group(0) @binding(5) var historyOutput: texture_storage_2d<rgba16float, write>;

fn aces_tone_mapping(color: vec3<f32>) -> vec3<f32> {
    let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), vec3(0.0), vec3(1.0));
}

fn get_radiance(coord: vec2<u32>) -> vec3<f32> {
    if coord.x >= scene.width || coord.y >= scene.height { return vec3(0.0); }
    let p_idx = coord.y * scene.width + coord.x;
    let acc = accumulateBuffer[p_idx];
    if acc.a <= 0.0 { return vec3(0.0); }
    return acc.rgb / acc.a;
}

fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= scene.width || id.y >= scene.height { return; }

    let center_pos = id.xy;
    let center_color_raw = get_radiance(center_pos);

    // 1. Firefly Removal
    var max_neighbor_lum = 0.0;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            if dx == 0 && dy == 0 { continue; }
            let nx = i32(center_pos.x) + dx;
            let ny = i32(center_pos.y) + dy;
            if nx < 0 || nx >= i32(scene.width) || ny < 0 || ny >= i32(scene.height) { continue; }
            let neighbor_color = get_radiance(vec2<u32>(u32(nx), u32(ny)));
            max_neighbor_lum = max(max_neighbor_lum, luminance(neighbor_color));
        }
    }
    let center_lum = luminance(center_color_raw);
    let threshold = max(max_neighbor_lum * 2.5, 1.0); // Slightly more relaxed for sharpness
    var center_color = center_color_raw;
    if center_lum > threshold {
        center_color *= (threshold / center_lum);
    }

    // 2. Bilateral Filter (Sharpened/Reduced radius)
    let SIGMA_S = 0.5; // Tighter spatial filter
    let SIGMA_R = 0.1; // Tighter range filter
    let RADIUS = 1;    // 3x3 only

    var filtered_sum = vec3(0.0);
    var total_weight = 0.0;

    for (var dy = -RADIUS; dy <= RADIUS; dy++) {
        for (var dx = -RADIUS; dx <= RADIUS; dx++) {
            let neighbor_pos = vec2<i32>(center_pos) + vec2<i32>(dx, dy);
            if neighbor_pos.x < 0 || neighbor_pos.x >= i32(scene.width) || neighbor_pos.y < 0 || neighbor_pos.y >= i32(scene.height) {
                continue;
            }

            let neighbor_color = get_radiance(vec2<u32>(u32(neighbor_pos.x), u32(neighbor_pos.y)));
            let w_s = exp(-f32(dx * dx + dy * dy) / (2.0 * SIGMA_S * SIGMA_S));
            let color_diff = neighbor_color - center_color;
            let w_r = exp(-dot(color_diff, color_diff) / (2.0 * SIGMA_R * SIGMA_R));
            let w = w_s * w_r;
            filtered_sum += neighbor_color * w;
            total_weight += w;
        }
    }

    let denoised_hdr = filtered_sum / max(total_weight, 1e-4);
    
    // 3. TAA Blend (HDR Feedback)
    let dims = vec2<f32>(f32(scene.width), f32(scene.height));
    let uv = (vec2<f32>(id.xy) + 0.5) / dims;
    let samples_history = textureSampleLevel(historyTex, smp, uv, 0.0).rgb;
    
    // Neighborhood Clamping
    var min_c = center_color;
    var max_c = center_color;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let nx = i32(center_pos.x) + x;
            let ny = i32(center_pos.y) + y;
            if nx < 0 || nx >= i32(scene.width) || ny < 0 || ny >= i32(scene.height) { continue; }
            let nb = get_radiance(vec2<u32>(u32(nx), u32(ny)));
            min_c = min(min_c, nb);
            max_c = max(max_c, nb);
        }
    }
    let clamped_history = clamp(samples_history, min_c, max_c);

    var alpha = 0.15; // Increased alpha for more sharpness/current frame detail
    if scene.frame_count <= 1u { alpha = 1.0; }

    let final_hdr = mix(clamped_history, denoised_hdr, alpha);

    // Save HDR for next frame
    textureStore(historyOutput, vec2<i32>(id.xy), vec4(final_hdr, 1.0));

    // 4. Tonemapping and Sharpening Filter
    let mapped = aces_tone_mapping(final_hdr);

    // Simple cross-sharpening (Unsharp mask)
    // Sample cross neighbors from the tonemapped result would be expensive (re-calculating neighbors)
    // So we just do a simple pass on 'mapped' itself using derived neighborhood if possible,
    // but better to just output here. 
    // Wait, let's just use a basic sharpen calculation:
    // We already have 'center_color' and 'denoised_hdr'.
    let edge_detect = center_color - denoised_hdr;
    let sharpened = mapped + aces_tone_mapping(edge_detect) * 0.3;

    let ldr_out = pow(clamp(sharpened, vec3(0.0), vec3(1.0)), vec3<f32>(1.0 / 2.2));
    textureStore(outputTex, vec2<i32>(id.xy), vec4(ldr_out, 1.0));
}