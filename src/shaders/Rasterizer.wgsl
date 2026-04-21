struct Camera {
    origin: vec4<f32>,
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

struct MeshTopology {
    v0: u32,
    v1: u32,
    v2: u32,
    pad: u32,
    data0: vec4<f32>,
    data1: vec4<f32>,
    data2: vec4<f32>,
    data3: vec4<f32>
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

@group(0) @binding(2) var<uniform> scene: SceneUniforms;
@group(0) @binding(3) var<storage, read> geometry_pos: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> topology: array<MeshTopology>;
@group(0) @binding(6) var<storage, read> instances: array<Instance>;
@group(0) @binding(11) var<storage, read> geometry_norm: array<vec4<f32>>;
@group(0) @binding(12) var<storage, read> geometry_uv: array<vec2<f32>>;
@group(0) @binding(7) var tex: texture_2d_array<f32>;
@group(0) @binding(8) var smp: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) tex_id: f32,
    @location(4) @interpolate(flat) instance_id: u32,
    @location(5) @interpolate(flat) tri_idx: u32,
}

// 八面体エンコーディングによる法線圧縮 (vec3 -> vec2)
fn pack_normal(n: vec3<f32>) -> vec2<f32> {
    let p = n.xy * (1.0 / (abs(n.x) + abs(n.y) + abs(n.z)));
    return select(p, (1.0 - abs(p.yx)) * select(vec2(-1.0), vec2(1.0), p.xy >= vec2(0.0)), n.z < 0.0);
}

struct GBufferOutput {
    @location(0) albedo: vec4<f32>,
    @location(1) normal_and_id: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index : u32, @builtin(instance_index) instance_index : u32) -> VertexOutput {
    let tri_idx = vertex_index / 3u;
    let local_idx = vertex_index % 3u;

    // Out of bounds safety
    let tri = topology[tri_idx];
    let inst = instances[instance_index];

    var v_idx: u32;
    if (local_idx == 0u) {
        v_idx = tri.v0;
    } else if (local_idx == 1u) {
        v_idx = tri.v1;
    } else {
        v_idx = tri.v2;
    }

    let local_pos = geometry_pos[v_idx].xyz;
    let local_norm = geometry_norm[v_idx].xyz;
    let local_uv = geometry_uv[v_idx];

    // Apply instance transform
    let mat = mat4x4<f32>(inst.transform_0, inst.transform_1, inst.transform_2, inst.transform_3);
    let inv_mat = mat4x4<f32>(inst.inv_0, inst.inv_1, inst.inv_2, inst.inv_3);

    let pos = (mat * vec4<f32>(local_pos, 1.0)).xyz;
    let norm = normalize((vec4<f32>(local_norm, 0.0) * inv_mat).xyz);

    // Exact View-Projection matching the Raytracer's image plane
    let eye = scene.camera.origin.xyz;
    let horizontal = scene.camera.horizontal.xyz;
    let vertical = scene.camera.vertical.xyz;
    let lower_left = scene.camera.lower_left_corner.xyz;
    
    let center = lower_left + horizontal * 0.5 + vertical * 0.5;
    let forward_vec = center - eye;
    let focal_length = length(forward_vec);
    let forward = normalize(forward_vec);
    
    let plane_w = length(horizontal);
    let plane_h = length(vertical);
    
    let right = normalize(horizontal);
    let up = normalize(vertical);

    let view_dir = pos - eye;
    let z_view = dot(view_dir, normalize(center - eye));
    let x_view = dot(view_dir, right);
    let y_view = dot(view_dir, up);

    let z_near = 0.001;
    let z_far = 1000.0;
    
    // Depth mapping to [0, 1] for WebGPU
    let z_clip = z_view * (z_far / (z_far - z_near)) - (z_far * z_near) / (z_far - z_near);

    let proj_pos = vec4<f32>(
        x_view * (focal_length / (plane_w * 0.5)),
        y_view * (focal_length / (plane_h * 0.5)),
        z_clip,
        z_view
    );

    var out: VertexOutput;
    out.position = proj_pos;
    
    out.color = tri.data0.rgb; // BaseColor
    out.normal = norm;
    out.uv = local_uv;
    out.tex_id = tri.data2.x; // BaseTex ID
    out.instance_id = inst.instance_id; // Using actual instance_id from instance struct
    out.tri_idx = tri_idx;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> GBufferOutput {
    var albedo = in.color;
    if (in.tex_id > -0.5) {
        let tex_col = textureSample(tex, smp, in.uv, i32(in.tex_id)).rgb;
        albedo *= tex_col;
    }

    var out: GBufferOutput;
    out.albedo = vec4<f32>(albedo, 1.0);
    out.normal_and_id = vec4<f32>(pack_normal(normalize(in.normal)), bitcast<f32>(in.tri_idx), bitcast<f32>(in.instance_id));
    return out;
}