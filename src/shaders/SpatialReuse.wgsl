struct Camera {
    origin: vec4<f32>,
    lower_left_corner: vec4<f32>,
    horizontal: vec4<f32>,
    vertical: vec4<f32>,
    u: vec4<f32>,
    v: vec4<f32>,
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
    average_jitter: vec2<f32>
}

struct Sample {
    hit_p: vec4<f32>,    // xyz: pos, w: dir.x
    normal: vec4<f32>,   // xyz: normal, w: dir.y
    radiance: vec4<f32>, // xyz: radiance, w: dir.z
}

struct Reservoir {
    sample: Sample,
    w_sum: f32,
    W: f32,
    M: u32,
    padding: f32,
}

struct MeshTopology {
    v0: u32,
    v1: u32,
    v2: u32,
    pad: u32,
    data0: vec4<f32>, // rgb: BaseColor, w: MaterialType (cast)
    data1: vec4<f32>, // x: Metallic, y: Roughness, z: IOR, w: 0.0
    data2: vec4<f32>, // x: BaseTex, y: MetRoughTex, z: NormalTex, w: EmissiveTex
    data3: vec4<f32>, // rgb: Emissive, w: 0.0
}

struct BVHNode {
    min_b: vec4<f32>, // w: skip_pointer
    max_b: vec4<f32>, // w: data (internal: 0, leaf: (left_first << 3) | tri_count)
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
    direction: vec3<f32>,
    inv_d: vec3<f32>,
    origin_inv_d: vec3<f32>
}

fn make_ray(origin: vec3<f32>, direction: vec3<f32>) -> Ray {
    let inv_d = 1.0 / direction;
    return Ray(origin, direction, inv_d, origin * inv_d);
}

@group(0) @binding(2) var<uniform> scene : SceneUniforms;
@group(0) @binding(3) var<storage, read> geometry_pos : array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> topology : array<MeshTopology>;
@group(0) @binding(5) var<storage, read> nodes : array<BVHNode>; 
@group(0) @binding(6) var<storage, read> instances : array<Instance>;
@group(0) @binding(14) var g_normal : texture_2d<f32>;
@group(0) @binding(15) var g_depth : texture_depth_2d;
@group(0) @binding(16) var<storage, read_write> reservoirsBuffer : array<Reservoir>;
@group(0) @binding(17) var<storage, read_write> spatialReservoirsBuffer : array<Reservoir>;

fn get_pos(idx: u32) -> vec3<f32> {
    return geometry_pos[idx].xyz;
}

fn get_inv_transform(inst: Instance) -> mat4x4<f32> {
    return mat4x4<f32>(inst.inv_0, inst.inv_1, inst.inv_2, inst.inv_3);
}

fn intersect_aabb(min_b: vec3<f32>, max_b: vec3<f32>, r: Ray, t_min: f32, t_max: f32) -> f32 {
    let t1 = min_b * r.inv_d - r.origin_inv_d;
    let t2 = max_b * r.inv_d - r.origin_inv_d;
    let t_near = min(t1, t2);
    let t_far = max(t1, t2);
    let tm_near = max(t_min, max(t_near.x, max(t_near.y, t_near.z)));
    let tm_far = min(t_max, min(t_far.x, min(t_far.y, t_far.z)));
    return select(1e30, tm_near, tm_near <= tm_far);
}

fn hit_triangle_raw(v0: vec3<f32>, v1: vec3<f32>, v2: vec3<f32>, r: Ray, t_min: f32, t_max: f32) -> f32 {
    let e1 = v1 - v0; let e2 = v2 - v0;
    let h = cross(r.direction, e2); let a = dot(e1, h);
    if abs(a) < 1e-6 { return -1.0; } 
    let f = 1.0 / a; let s = r.origin - v0; let u = f * dot(s, h);
    if u < 0.0 || u > 1.0 { return -1.0; }
    let q = cross(s, e1); let v = f * dot(r.direction, q);
    if v < 0.0 || u + v > 1.0 { return -1.0; }
    let t = f * dot(e2, q);
    return select(-1.0, t, t > t_min && t < t_max);
}

fn intersect_blas_shadow(r: Ray, t_min: f32, t_max: f32, node_start_idx: u32) -> bool {
    let end_node = node_start_idx + bitcast<u32>(nodes[node_start_idx].min_b.w);
    var curr = node_start_idx;
    while curr < end_node {
        let node = nodes[curr];
        if intersect_aabb(node.min_b.xyz, node.max_b.xyz, r, t_min, t_max) < 1e30 {
            let data = bitcast<u32>(node.max_b.w);
            if data != 0u {
                let first = data >> 3u;
                let count = data & 7u;
                for (var i = 0u; i < count; i++) {
                    let tr = topology[first + i];
                    if hit_triangle_raw(get_pos(tr.v0), get_pos(tr.v1), get_pos(tr.v2), r, t_min, t_max) > 0.0 { return true; }
                }
                curr = node_start_idx + bitcast<u32>(node.min_b.w);
            } else { curr = curr + 1u; }
        } else { curr = node_start_idx + bitcast<u32>(node.min_b.w); }
    }
    return false;
}

fn intersect_tlas_shadow(r: Ray, t_min: f32, t_max: f32) -> bool {
    if scene.blas_base_idx == 0u { return false; }
    var curr = 0u;
    let end_node = bitcast<u32>(nodes[0].min_b.w);
    while curr < end_node {
        let node = nodes[curr];
        if intersect_aabb(node.min_b.xyz, node.max_b.xyz, r, t_min, t_max) < 1e30 {
            let data = bitcast<u32>(node.max_b.w);
            if data != 0u {
                let inst = instances[data >> 3u];
                let r_local = make_ray((get_inv_transform(inst) * vec4(r.origin, 1.0)).xyz, (get_inv_transform(inst) * vec4(r.direction, 0.0)).xyz);
                if intersect_blas_shadow(r_local, t_min, t_max, scene.blas_base_idx + inst.blas_node_offset) { return true; }
                curr = bitcast<u32>(node.min_b.w);
            } else { curr = curr + 1u; }
        } else { curr = bitcast<u32>(node.min_b.w); }
    }
    return false;
}

const PI: f32 = 3.14159265359;

// =========================================================
//   Math & Helpers
// =========================================================

fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn init_rng(pixel_idx: u32, frame_count: u32) -> u32 {
    var seed = pixel_idx + frame_count * 719393u;
    seed ^= 2747636419u; seed *= 2654435769u; seed ^= (seed >> 16u);
    seed *= 2654435769u; seed ^= (seed >> 16u); seed *= 2654435769u;
    return seed;
}

fn rand_pcg(rng: ptr<function, u32>) -> f32 {
    let state = *rng;
    *rng = state * 747796405u + 2891336453u;
    var word: u32 = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    word = (word >> 22u) ^ word;
    return f32(word) / 4294967296.0;
}

fn unpack_normal(p: vec2<f32>) -> vec3<f32> {
    var n = vec3(p, 1.0 - abs(p.x) - abs(p.y));
    let t = saturate(-n.z);
    n.x += select(t, -t, n.x >= 0.0);
    n.y += select(t, -t, n.y >= 0.0);
    return normalize(n);
}

// =========================================================
//   BRDF Helpers
// =========================================================

fn ggx_d(n_dot_h: f32, a2: f32) -> f32 {
    let d = (n_dot_h * a2 - n_dot_h) * n_dot_h + 1.0;
    return a2 / (PI * d * d);
}

fn ggx_g(n_dot_v: f32, n_dot_l: f32, a2: f32) -> f32 {
    let g_v = n_dot_v + sqrt((-n_dot_v * a2 + n_dot_v) * n_dot_v + a2);
    let g_l = n_dot_l + sqrt((-n_dot_l * a2 + n_dot_l) * n_dot_l + a2);
    return 2.0 * n_dot_v * n_dot_l / (g_v * g_l);
}

fn fresnel_schlick(v_dot_h: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (1.0 - f0) * pow(1.0 - v_dot_h, 5.0);
}

fn eval_brdf_cos(w_o: vec3<f32>, w_i: vec3<f32>, normal: vec3<f32>, mat_type: u32, roughness: f32, f0: vec3<f32>, albedo: vec3<f32>) -> vec3<f32> {
    let n_dot_l = max(dot(normal, w_i), 1e-4);
    let n_dot_v = max(dot(normal, w_o), 1e-4);

    if mat_type == 0u {
        return (albedo / PI) * n_dot_l;
    } else if mat_type == 1u {
        let h = normalize(w_o + w_i);
        let n_dot_h = max(dot(normal, h), 1e-4);
        let v_dot_h = max(dot(w_o, h), 1e-4);
        let a2 = roughness * roughness;
        let d = ggx_d(n_dot_h, a2);
        let g = ggx_g(n_dot_v, n_dot_l, a2);
        let f = fresnel_schlick(v_dot_h, f0);
        return (d * g * f) / (4.0 * n_dot_v);
    } else { 
        return vec3<f32>(0.0);
    }
}

// =========================================================
//   ReSTIR Logic
// =========================================================

fn update_reservoir(r: ptr<function, Reservoir>, s: Sample, weight: f32, rng: ptr<function, u32>) {
    if weight <= 0.0 { return; }
    r.w_sum += weight;
    if rand_pcg(rng) < (weight / r.w_sum) {
        r.sample = s;
    }
}

fn get_world_pos(id: vec2<u32>, depth_val: f32) -> vec3<f32> {
    let u_cam = (f32(id.x) + 0.5 + scene.jitter.x * f32(scene.width)) / f32(scene.width);
    let v_cam = 1.0 - (f32(id.y) + 0.5 + scene.jitter.y * f32(scene.height)) / f32(scene.height);
    let ray_dir = normalize(scene.camera.lower_left_corner.xyz + u_cam * scene.camera.horizontal.xyz + v_cam * scene.camera.vertical.xyz - scene.camera.origin.xyz);
    
    // Reverse non-linear Z to view-space Z
    let z_near = 0.001;
    let z_far = 10000.0;
    let z_view = (z_far * z_near) / (z_far - depth_val * (z_far - z_near));
    
    // View-space Z to ray distance t
    let eye = scene.camera.origin.xyz;
    let center = scene.camera.lower_left_corner.xyz + scene.camera.horizontal.xyz * 0.5 + scene.camera.vertical.xyz * 0.5;
    let forward = normalize(center - eye);
    let t = z_view / dot(ray_dir, forward);
    
    return eye + ray_dir * t;
}

@compute @workgroup_size(8, 8)
fn spatial_reuse(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= scene.width || id.y >= scene.height { return; }
    let p_idx = id.y * scene.width + id.x;
    var rng = init_rng(p_idx, scene.frame_count + 2000u);

    let g_normal_val = textureLoad(g_normal, id.xy, 0);
    let depth_val = textureLoad(g_depth, id.xy, 0);
    if depth_val >= 1.0 { 
        spatialReservoirsBuffer[p_idx] = reservoirsBuffer[p_idx];
        return; 
    }

    let tri_idx = bitcast<u32>(g_normal_val.z);
    let tri = topology[tri_idx];
    let mat_type = u32(tri.data0.w + 0.5);
    let albedo = tri.data0.rgb;
    let metallic = tri.data1.x;
    let roughness = max(tri.data1.y, 0.005);
    let f0 = mix(vec3(0.04), albedo, metallic);

    let u_cam = (f32(id.x) + 0.5 + scene.jitter.x * f32(scene.width)) / f32(scene.width);
    let v_cam = 1.0 - (f32(id.y) + 0.5 + scene.jitter.y * f32(scene.height)) / f32(scene.height);
    let ray_dir = normalize(scene.camera.lower_left_corner.xyz + u_cam * scene.camera.horizontal.xyz + v_cam * scene.camera.vertical.xyz - scene.camera.origin.xyz);
    let w_o = -ray_dir;

    var normal = unpack_normal(g_normal_val.xy);
    normal = select(-normal, normal, dot(w_o, normal) > 0.0);
    
    // Reconstruct world hit point
    let curr_hit_p = get_world_pos(id.xy, depth_val);

    let is_delta = (mat_type == 2u) || (mat_type == 1u && metallic > 0.9 && roughness < 0.01);

    if is_delta {
        spatialReservoirsBuffer[p_idx] = reservoirsBuffer[p_idx];
        return;
    }

    var r_curr = reservoirsBuffer[p_idx];
    
    // Initialize w_sum for the merge
    let w_i_curr = vec3(r_curr.sample.hit_p.w, r_curr.sample.normal.w, r_curr.sample.radiance.w);
    var p_hat_curr = 0.0;
    let brdf_curr = eval_brdf_cos(w_o, w_i_curr, normal, mat_type, roughness, f0, albedo);
    p_hat_curr = luminance(r_curr.sample.radiance.xyz * brdf_curr);
    r_curr.w_sum = r_curr.W * f32(r_curr.M) * p_hat_curr;

    // Spatial Reuse Parameters
    const num_neighbors: u32 = 4u;
    let base_radius = 20.0;
    var dynamic_radius = mix(1.0, 10.0, roughness);
    dynamic_radius = select(base_radius, dynamic_radius,  mat_type == 1u);


    for (var i = 0u; i < num_neighbors; i++) {
        let angle = rand_pcg(&rng) * 2.0 * PI;
        let dist = rand_pcg(&rng) * dynamic_radius;
        let offset = vec2<i32>(i32(cos(angle) * dist), i32(sin(angle) * dist));
        let neighbor_coord = vec2<i32>(id.xy) + offset;

        if neighbor_coord.x < 0 || neighbor_coord.x >= i32(scene.width) ||
           neighbor_coord.y < 0 || neighbor_coord.y >= i32(scene.height) {
            continue;
        }

        let n_idx = u32(neighbor_coord.y) * scene.width + u32(neighbor_coord.x);
        let neighbor_normal_val = textureLoad(g_normal, neighbor_coord, 0);
        let neighbor_depth = textureLoad(g_depth, neighbor_coord, 0);
        let neighbor_normal = unpack_normal(neighbor_normal_val.xy);
        
        if dot(normal, neighbor_normal) < 0.9 || abs(depth_val - neighbor_depth) > 0.1 * depth_val {
            continue;
        }

        let r_neighbor = reservoirsBuffer[n_idx];
        let light_p = r_neighbor.sample.hit_p.xyz;
        let light_n = r_neighbor.sample.normal.xyz;
        
        // Reconnection Shift: New direction from current point to neighbor's light sample
        let v_curr = light_p - curr_hit_p;
        let dist_curr2 = dot(v_curr, v_curr);
        let dist_curr = sqrt(dist_curr2);
        let w_i_new = v_curr / dist_curr;

        // Jacobian calculation
        // J = (|cos_theta_L'| * dist_neighbor^2) / (|cos_theta_L| * dist_curr^2)
        let neighbor_hit_p = get_world_pos(vec2<u32>(neighbor_coord), neighbor_depth);
        
        let v_neighbor = light_p - neighbor_hit_p;
        let dist_neighbor2 = dot(v_neighbor, v_neighbor);
        let w_i_old = v_neighbor / sqrt(dist_neighbor2);

        let cos_L_curr = max(dot(light_n, -w_i_new), 0.0);
        let cos_L_old = max(dot(light_n, -w_i_old), 0.0);
        
        var jacobian = 1.0;
        if cos_L_old > 1e-6 {
            jacobian = (cos_L_curr * dist_neighbor2) / (cos_L_old * dist_curr2);
        }
        // jacobian = clamp(jacobian, 0.1, 10.0);

        // Visibility Check: Shadow ray from current hit point to neighbor's light position
        let shadow_ray = make_ray(curr_hit_p + normal * 1e-4, w_i_new);
        if intersect_tlas_shadow(shadow_ray, 0.001, dist_curr - 2e-4) {
            continue;
        }

        var p_hat_new = 0.0;
        let brdf_new = eval_brdf_cos(w_o, w_i_new, normal, mat_type, roughness, f0, albedo);
        p_hat_new = luminance(r_neighbor.sample.radiance.xyz * brdf_new);

        if p_hat_new > 1e-6 {
            let weight = p_hat_new * r_neighbor.W * f32(r_neighbor.M) * jacobian;
            var shifted_sample = r_neighbor.sample;
            shifted_sample.hit_p.w = w_i_new.x;
            shifted_sample.normal.w = w_i_new.y;
            shifted_sample.radiance.w = w_i_new.z;
            
            update_reservoir(&r_curr, shifted_sample, weight, &rng);
            r_curr.M += r_neighbor.M;
        }
    }

    let w_i_final = vec3(r_curr.sample.hit_p.w, r_curr.sample.normal.w, r_curr.sample.radiance.w);
    var p_hat_final = 0.0;
    let brdf_final = eval_brdf_cos(w_o, w_i_final, normal, mat_type, roughness, f0, albedo);
    p_hat_final = luminance(r_curr.sample.radiance.xyz * brdf_final);

    if p_hat_final > 1e-6 {
        r_curr.W = r_curr.w_sum / (f32(r_curr.M) * p_hat_final);
    } else {
        r_curr.W = 0.0;
    }

    // r_curr.W = min(r_curr.W, 1000.0);
    spatialReservoirsBuffer[p_idx] = r_curr;
}
