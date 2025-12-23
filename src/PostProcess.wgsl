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

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= scene.width || id.y >= scene.height { return; }
    let p_idx = id.y * scene.width + id.x;

    let acc = accumulateBuffer[p_idx];
    if acc.a <= 0.0 {
        textureStore(outputTex, vec2<i32>(id.xy), vec4(0., 0., 0., 1.));
        return;
    }

    let hdr_color = acc.rgb / acc.a;
    let mapped = aces_tone_mapping(hdr_color);
    let out = pow(mapped, vec3(1.0 / 2.2));
    textureStore(outputTex, vec2<i32>(id.xy), vec4(out, 1.));
}
