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
    prev_jitter: vec2<f32>
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

fn get_radiance(coord: vec2<i32>) -> vec3<f32> {
    let c = clamp(coord, vec2<i32>(0), vec2<i32>(i32(scene.width) - 1, i32(scene.height) - 1));
    let p_idx = u32(c.y) * scene.width + u32(c.x);
    let acc = accumulateBuffer[p_idx];
    if acc.a <= 0.0 { return vec3(0.0); }
    return acc.rgb / acc.a;
}

// Bilinear sampling for un-jittering
fn get_radiance_bilinear(uv: vec2<f32>) -> vec3<f32> {
    let dims = vec2<f32>(f32(scene.width), f32(scene.height));
    let f_coord = uv * dims - 0.5;
    let i_coord = vec2<i32>(floor(f_coord));
    let f = f_coord - vec2<f32>(i_coord);

    let c00 = get_radiance(i_coord + vec2<i32>(0, 0));
    let c10 = get_radiance(i_coord + vec2<i32>(1, 0));
    let c01 = get_radiance(i_coord + vec2<i32>(0, 1));
    let c11 = get_radiance(i_coord + vec2<i32>(1, 1));

    return mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);
}

fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= scene.width || id.y >= scene.height { return; }

    let dims = vec2<f32>(f32(scene.width), f32(scene.height));
    let uv = (vec2<f32>(id.xy) + 0.5) / dims;

    // 1. Un-jittered Current Frame Radiance
    let center_color_raw = get_radiance_bilinear(uv - scene.jitter);

    // Firefly Removal (Check un-jittered neighborhood)
    var max_neighbor_lum = 0.0;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            if dx == 0 && dy == 0 { continue; }
            let nb_uv = uv + vec2<f32>(f32(dx), f32(dy)) / dims - scene.jitter;
            let neighbor_color = get_radiance_bilinear(nb_uv);
            max_neighbor_lum = max(max_neighbor_lum, dot(neighbor_color, vec3(0.299, 0.587, 0.114)));
        }
    }
    let threshold = max(max_neighbor_lum * 3.0, 1.0);
    let center_lum = dot(center_color_raw, vec3(0.299, 0.587, 0.114));
    let center_color = select(center_color_raw, center_color_raw * (threshold / max(center_lum, 1e-4)), center_lum > threshold);

    // 2. Bilateral Filter
    let SIGMA_S = 0.5;
    let SIGMA_R = 0.1;
    let RADIUS = 1;

    var filtered_sum = vec3(0.0);
    var total_weight = 0.0;
    for (var dy = -RADIUS; dy <= RADIUS; dy++) {
        for (var dx = -RADIUS; dx <= RADIUS; dx++) {
            let samp_uv = uv + vec2<f32>(f32(dx), f32(dy)) / dims - scene.jitter;
            let neighbor_color = get_radiance_bilinear(samp_uv);

            let w_s = exp(-f32(dx * dx + dy * dy) / (2.0 * SIGMA_S * SIGMA_S));
            let color_diff = neighbor_color - center_color;
            let w_r = exp(-dot(color_diff, color_diff) / (2.0 * SIGMA_R * f32(RADIUS) * f32(RADIUS)));
            let w = w_s * w_r;
            filtered_sum += neighbor_color * w;
            total_weight += w;
        }
    }
    let denoised_hdr = filtered_sum / max(total_weight, 1e-4);
    
    // 3. TAA Blend (HDR Feedback)
    let samples_history = textureSampleLevel(historyTex, smp, uv, 0.0).rgb;
    
    // Neighborhood Clamping
    var m1 = vec3(0.0);
    var m2 = vec3(0.0);
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let nb_uv = uv + vec2<f32>(f32(dx), f32(dy)) / dims - scene.jitter;
            let c = get_radiance_bilinear(nb_uv);
            m1 += c;
            m2 += c * c;
        }
    }
    let mean = m1 / 9.0;
    let stddev = sqrt(max(m2 / 9.0 - mean * mean, vec3(0.0)));
    
    // Loosen clamping as we converge to avoid clipping true history through single-frame noise
    var k = 2.0;
    if scene.frame_count > 10u { k = 8.0; }
    let clamped_history = clamp(samples_history, mean - stddev * k, mean + stddev * k);

    // Adaptive alpha for progressive refinement
    var alpha = 1.0 / f32(scene.frame_count);
    alpha = max(alpha, 0.0005); // Allow much deeper convergence

    let final_hdr = mix(clamped_history, denoised_hdr, alpha);

    // Store un-jittered result
    textureStore(historyOutput, vec2<i32>(id.xy), vec4(final_hdr, 1.0));

    // 4. Output
    let mapped = aces_tone_mapping(final_hdr);
    let edge_detect = center_color - denoised_hdr;
    let sharpened = mapped + aces_tone_mapping(edge_detect) * 0.3;

    let ldr_out = pow(clamp(sharpened, vec3(0.0), vec3(1.0)), vec3<f32>(1.0 / 2.2));
    textureStore(outputTex, vec2<i32>(id.xy), vec4(ldr_out, 1.0));
}