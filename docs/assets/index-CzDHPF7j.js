var __defProp = Object.defineProperty;
var __defNormalProp = (obj, key, value) => key in obj ? __defProp(obj, key, { enumerable: true, configurable: true, writable: true, value }) : obj[key] = value;
var __publicField = (obj, key, value) => __defNormalProp(obj, typeof key !== "symbol" ? key + "" : key, value);
(async () => {
  (function() {
    const e = document.createElement("link").relList;
    if (e && e.supports && e.supports("modulepreload")) return;
    for (const r of document.querySelectorAll('link[rel="modulepreload"]')) n(r);
    new MutationObserver((r) => {
      for (const i of r) if (i.type === "childList") for (const o of i.addedNodes) o.tagName === "LINK" && o.rel === "modulepreload" && n(o);
    }).observe(document, {
      childList: true,
      subtree: true
    });
    function t(r) {
      const i = {};
      return r.integrity && (i.integrity = r.integrity), r.referrerPolicy && (i.referrerPolicy = r.referrerPolicy), r.crossOrigin === "use-credentials" ? i.credentials = "include" : r.crossOrigin === "anonymous" ? i.credentials = "omit" : i.credentials = "same-origin", i;
    }
    function n(r) {
      if (r.ep) return;
      r.ep = true;
      const i = t(r);
      fetch(r.href, i);
    }
  })();
  const K = "modulepreload", X = function(s) {
    return "/webgpu-raytracer/" + s;
  }, H = {}, G = function(e, t, n) {
    let r = Promise.resolve();
    if (t && t.length > 0) {
      let p = function(u) {
        return Promise.all(u.map((b) => Promise.resolve(b).then((B) => ({
          status: "fulfilled",
          value: B
        }), (B) => ({
          status: "rejected",
          reason: B
        }))));
      };
      var o = p;
      document.getElementsByTagName("link");
      const l = document.querySelector("meta[property=csp-nonce]"), d = (l == null ? void 0 : l.nonce) || (l == null ? void 0 : l.getAttribute("nonce"));
      r = p(t.map((u) => {
        if (u = X(u), u in H) return;
        H[u] = true;
        const b = u.endsWith(".css"), B = b ? '[rel="stylesheet"]' : "";
        if (document.querySelector(`link[href="${u}"]${B}`)) return;
        const w = document.createElement("link");
        if (w.rel = b ? "stylesheet" : K, b || (w.as = "script"), w.crossOrigin = "", w.href = u, d && w.setAttribute("nonce", d), document.head.appendChild(w), b) return new Promise((J, Y) => {
          w.addEventListener("load", J), w.addEventListener("error", () => Y(new Error(`Unable to preload CSS for ${u}`)));
        });
      }));
    }
    function i(l) {
      const d = new Event("vite:preloadError", {
        cancelable: true
      });
      if (d.payload = l, window.dispatchEvent(d), !d.defaultPrevented) throw l;
    }
    return r.then((l) => {
      for (const d of l || []) d.status === "rejected" && i(d.reason);
      return e().catch(i);
    });
  }, Q = `// =========================================================
//   WebGPU Ray Tracer (TLAS & BLAS)
// =========================================================

const PI = 3.141592653589793;
const T_MIN = 0.001;
const T_MAX = 1e30;
// These are replaced by TypeScript
const MAX_DEPTH = 10u;
const SPP = 1u;

// --- Structs ---

struct Camera {
    origin: vec3<f32>,
    lens_radius: f32,
    lower_left_corner: vec3<f32>,
    horizontal: vec3<f32>,
    vertical: vec3<f32>,
    u: vec3<f32>,
    v: vec3<f32>
}

struct SceneUniforms {
    camera: Camera,
    frame_count: u32,
    blas_base_idx: u32, // Start index of BLAS nodes in 'nodes' array
    vertex_count: u32,
    rand_seed: u32,
    light_count: u32
}

struct MeshTopology {
    v0: u32,
    v1: u32,
    v2: u32,
    pad: u32,
    data0: vec4<f32>,
    data1: vec4<f32>
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

// --- Bindings (Consolidated) ---

@group(0) @binding(0) var outputTex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<storage, read_write> accumulateBuffer: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> scene: SceneUniforms;

@group(0) @binding(3) var<storage, read> geometry: array<f32>; // [Pos(4)... | Norm(4)... | UV(2)...]
@group(0) @binding(4) var<storage, read> topology: array<MeshTopology>;
@group(0) @binding(5) var<storage, read> nodes: array<BVHNode>; // Merged TLAS/BLAS
@group(0) @binding(6) var<storage, read> instances: array<Instance>;
@group(0) @binding(9) var<storage, read> lights: array<LightRef>;

@group(0) @binding(7) var tex: texture_2d_array<f32>;
@group(0) @binding(8) var smp: sampler;

// --- Helpers ---

// Accessors
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

struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>
}

struct HitResult {
    t: f32,
    tri_idx: f32,
    inst_idx: i32
}

fn get_inv_transform(inst: Instance) -> mat4x4<f32> {
    return mat4x4<f32>(inst.inv_0, inst.inv_1, inst.inv_2, inst.inv_3);
}

// --- Random ---
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
fn random_unit_vector(rng: ptr<function, u32>) -> vec3<f32> {
    let z = rand_pcg(rng) * 2.0 - 1.0; let a = rand_pcg(rng) * 2.0 * PI;
    let r = sqrt(max(0.0, 1.0 - z * z)); return vec3<f32>(r * cos(a), r * sin(a), z);
}
fn random_in_unit_disk(rng: ptr<function, u32>) -> vec3<f32> {
    let r = sqrt(rand_pcg(rng)); let theta = 2.0 * PI * rand_pcg(rng);
    return vec3<f32>(r * cos(theta), r * sin(theta), 0.0);
}
fn reflectance(cosine: f32, ref_idx: f32) -> f32 {
    var r0 = (1.0 - ref_idx) / (1.0 + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0 - r0) * pow((1.0 - cosine), 5.0);
}

// 1.0\u3092\u8D85\u3048\u3066\u3082\u7DBA\u9E97\u306B\u53CE\u3081\u308B\u95A2\u6570 (ACES\u8FD1\u4F3C)
fn aces_tone_mapping(color: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), vec3(0.0), vec3(1.0));
}

// --- Intersection ---

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
    var stack: array<u32, 32>;
    var stackptr = 0u;

    // Root Node (Global Index in 'nodes' array)
    let root_idx = node_start_idx;
    let root = nodes[root_idx];

    if intersect_aabb(root.min_b, root.max_b, r.origin, inv_d, t_min, closest_t) < 1e30 {
        stack[stackptr] = root_idx;
        stackptr++;
    }

    while stackptr > 0u {
        stackptr--;
        let idx = stack[stackptr];
        let node = nodes[idx];
        let count = u32(node.tri_count);

        if count > 0u {
            // Leaf Node
            let first = u32(node.left_first); // Triangle Index (Sorted)
            for (var i = 0u; i < count; i++) {
                let tri_id = first + i;
                let b = tri_id * 3u;

                let tri = topology[tri_id];
                let v0 = get_pos(tri.v0);
                let v1 = get_pos(tri.v1);
                let v2 = get_pos(tri.v2);

                let t = hit_triangle_raw(v0, v1, v2, r, t_min, closest_t);
                if t > 0.0 { closest_t = t; hit_idx = f32(tri_id); }
            }
        } else {
            // Internal Node
            // node.left_first is RELATIVE to the start of this BLAS.
            let l = u32(node.left_first) + node_start_idx;
            let r_node_idx = l + 1u;

            let nl = nodes[l];
            let nr = nodes[r_node_idx];

            let dl = intersect_aabb(nl.min_b, nl.max_b, r.origin, inv_d, t_min, closest_t);
            let dr = intersect_aabb(nr.min_b, nr.max_b, r.origin, inv_d, t_min, closest_t);

            let hl = dl < 1e30; let hr = dr < 1e30;
            if hl && hr {
                if dl < dr { stack[stackptr] = r_node_idx; stackptr++; stack[stackptr] = l; stackptr++; } else { stack[stackptr] = l; stackptr++; stack[stackptr] = r_node_idx; stackptr++; }
            } else if hl { stack[stackptr] = l; stackptr++; } else if hr { stack[stackptr] = r_node_idx; stackptr++; }
        }
    }
    return vec2<f32>(closest_t, hit_idx);
}

fn intersect_tlas(r: Ray, t_min: f32, t_max: f32) -> HitResult {
    var res: HitResult; res.t = t_max; res.tri_idx = -1.0; res.inst_idx = -1;
    // Need at least one node.
    // Ideally we pass tlas_count, but checking arrayLength of storage buffer 'nodes' is not precise for sub-allocations.
    // Assuming root is at 0.
    // If scene.blas_base_idx == 0, then tlas is empty or something weird.
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

        if node.tri_count > 0.5 { // Leaf (Instance)
            let inst_idx = u32(node.left_first);
            let inst = instances[inst_idx];
            let inv = get_inv_transform(inst);

            let r_local = Ray((inv * vec4(r.origin, 1.0)).xyz, (inv * vec4(r.direction, 0.0)).xyz);
            
            // BLAS start = global BLAS base + instance specific offset
            let blas_start = scene.blas_base_idx + inst.blas_node_offset;
            let blas = intersect_blas(r_local, t_min, res.t, blas_start);

            if blas.y > -0.5 {
                res.t = blas.x;
                res.tri_idx = blas.y;
                res.inst_idx = i32(inst_idx);
            }
        } else {
            // Internal (TLAS)
            let l = u32(node.left_first);
            let r_idx = l + 1u;
            let nl = nodes[l];
            let nr = nodes[r_idx];
            let dl = intersect_aabb(nl.min_b, nl.max_b, r.origin, inv_d, t_min, res.t);
            let dr = intersect_aabb(nr.min_b, nr.max_b, r.origin, inv_d, t_min, res.t);
            if dl < 1e30 && dr < 1e30 {
                if dl < dr {
                    stack[stackptr] = r_idx; stackptr++; stack[stackptr] = l; stackptr++;
                } else {
                    stack[stackptr] = l; stackptr++; stack[stackptr] = r_idx; stackptr++;
                }
            } else if dl < 1e30 {
                stack[stackptr] = l; stackptr++;
            } else if dr < 1e30 {
                stack[stackptr] = r_idx; stackptr++;
            }
        }
    }
    return res;
}

fn get_transform(inst: Instance) -> mat4x4<f32> {
    return mat4x4<f32>(inst.transform_0, inst.transform_1, inst.transform_2, inst.transform_3);
}

fn sample_lights(origin: vec3<f32>, normal: vec3<f32>, rng: ptr<function, u32>) -> vec3<f32> {
    let num_lights = scene.light_count;
    if num_lights == 0u { return vec3(0.0); }

    // Pick random light
    let idx = u32(rand_pcg(rng) * f32(num_lights));
    let light_ref = lights[idx];

    let inst = instances[light_ref.inst_idx];
    let tri = topology[light_ref.tri_idx];

    // Get Triangle Vertices (Local)
    let v0_local = get_pos(tri.v0);
    let v1_local = get_pos(tri.v1);
    let v2_local = get_pos(tri.v2);

    // Transform to World
    let transform = get_transform(inst);
    let v0 = (transform * vec4(v0_local, 1.0)).xyz;
    let v1 = (transform * vec4(v1_local, 1.0)).xyz;
    let v2 = (transform * vec4(v2_local, 1.0)).xyz;

    // Sample Point on Triangle
    let r1 = rand_pcg(rng);
    let r2 = rand_pcg(rng);
    let sqrt_r1 = sqrt(r1);
    let u = 1.0 - sqrt_r1;
    let v = sqrt_r1 * (1.0 - r2);
    let w = sqrt_r1 * r2;

    let light_pos = u * v0 + v * v1 + w * v2;
    let light_vec = light_pos - origin;
    let dist_sq = dot(light_vec, light_vec);
    let dist = sqrt(dist_sq);
    let dir = light_vec / dist;

    // Visibility Check
    if dot(dir, normal) <= 0.0 { return vec3(0.0); }

    let shadow_ray = Ray(origin + normal * 1e-3, dir);
    let hit = intersect_tlas(shadow_ray, T_MIN, dist - 1e-3);

    if hit.inst_idx != -1 {
         // Occluded
        return vec3(0.0);
    }

    // Geometry Factor
    let light_normal = normalize(cross(v1 - v0, v2 - v0)); // Assuming consistent winding
    let cos_light_raw = dot(-dir, light_normal);
    if cos_light_raw <= 0.0 { return vec3(0.0); } // \u2605\u88CF\u9762\u306A\u3089\u5149\u3089\u306A\u3044
    let cos_theta_light = cos_light_raw;
    let cos_theta_surf = max(dot(normal, dir), 0.0);
    let area = 0.5 * length(cross(v1 - v0, v2 - v0));

    // Emission (Hardcoded multiplier for now to match ray_color)
    let emission = tri.data0.rgb * 2.0; 

    // PDF = 1 / (Area * num_lights)
    // Contribution = Le * G * pdf_inv
    // G = (cos_surf * cos_light) / dist_sq
    // pdf_inv = Area * num_lights

    let G = (cos_theta_surf * cos_theta_light) / dist_sq;
    let weight = G * area * f32(num_lights);

    return emission * weight;
}

fn ray_color(r_in: Ray, rng: ptr<function, u32>) -> vec3<f32> {
    var ray = r_in;
    var throughput = vec3<f32>(1.0);
    var radiance = vec3<f32>(0.0);
    
    // \u2605\u8FFD\u52A0: \u76F4\u524D\u306E\u53CD\u5C04\u304C\u30B9\u30DA\u30AD\u30E5\u30E9\uFF08\u93E1\u9762/\u5C48\u6298\uFF09\u3060\u3063\u305F\u304B\uFF1F
    // \u521D\u671F\u5024 true \u306B\u3059\u308B\u3053\u3068\u3067\u3001\u30AB\u30E1\u30E9\u304B\u3089\u76F4\u63A5\u898B\u3048\u308B\u30E9\u30A4\u30C8\u306F\u63CF\u753B\u3055\u308C\u308B
    var specular_bounce = true;

    for (var depth = 0u; depth < MAX_DEPTH; depth++) {
        let hit = intersect_tlas(ray, T_MIN, T_MAX);
        if hit.inst_idx < 0 { break; }

        let inst = instances[u32(hit.inst_idx)];
        let tri_idx = u32(hit.tri_idx);
        let tri = topology[tri_idx];

        let i0 = tri.v0;
        let i1 = tri.v1;
        let i2 = tri.v2;
        let v0_pos = get_pos(i0);
        let v1_pos = get_pos(i1);
        let v2_pos = get_pos(i2);

        let inv = get_inv_transform(inst);
        let r_local = Ray((inv * vec4(ray.origin, 1.)).xyz, (inv * vec4(ray.direction, 0.)).xyz);
        let s = r_local.origin - v0_pos;
        let e1 = v1_pos - v0_pos;
        let e2 = v2_pos - v0_pos;
        let h = cross(r_local.direction, e2);
        let a = dot(e1, h);
        let f = 1.0 / a;
        let u = f * dot(s, h);
        let q = cross(s, e1);
        let v = f * dot(r_local.direction, q);
        let w = 1.0 - u - v;

        let n0 = get_normal(i0);
        let n1 = get_normal(i1);
        let n2 = get_normal(i2);
        let ln = normalize(n0 * w + n1 * u + n2 * v);
        let wn = normalize((vec4(ln, 0.0) * inv).xyz);

        var normal = wn;
        let front = dot(ray.direction, normal) < 0.0;
        normal = select(-normal, normal, front);

        let uv0 = get_uv(i0);
        let uv1 = get_uv(i1);
        let uv2 = get_uv(i2);
        let tex_uv = uv0 * w + uv1 * u + uv2 * v;
        let hit_p = ray.origin + ray.direction * hit.t;

        let albedo_color = tri.data0.rgb;
        let mat_type = bitcast<u32>(tri.data0.w);

        let tex_idx = tri.data1.y;
        var tex_color = vec3(1.0);
        if tex_idx > -0.5 {
            tex_color = textureSampleLevel(tex, smp, tex_uv, i32(tex_idx), 0.0).rgb;
        }
        let albedo = albedo_color * tex_color;

        // \u2605\u4FEE\u6B631: Emission\u306E\u8A08\u7B97
        // \u3053\u3053\u3067\u306E\u5224\u5B9A\u306B\u306F depth \u3084 mat_type \u3092\u4F7F\u308F\u305A\u3001\u30D5\u30E9\u30B0\u3092\u898B\u308B
        if mat_type == 3u {
            if specular_bounce {
                let emitted = albedo * 15.0; // \u5F37\u5EA615.0 (\u8ABF\u6574\u53EF\u80FD)
                radiance += throughput * emitted;
            }
            break; // \u30E9\u30A4\u30C8\u306B\u5F53\u305F\u3063\u305F\u3089\u7D42\u4E86
        }

        // 2. NEE (Diffuse only)
        if mat_type == 0u {
            let Ld = sample_lights(hit_p, normal, rng);
            let brdf = albedo * 0.318309886; // albedo / PI
            radiance += throughput * Ld * brdf;

            // \u2605\u91CD\u8981: NEE\u3092\u884C\u3063\u305F\u306E\u3067\u3001\u6B21\u306E\u30D0\u30A6\u30F3\u30B9\u3067\u30E9\u30A4\u30C8\u306B\u5F53\u305F\u3063\u3066\u3082\u767A\u5149\u3092\u52A0\u7B97\u3057\u306A\u3044
            specular_bounce = false;
        } else {
            // \u93E1\u9762\u53CD\u5C04\u3084\u30AC\u30E9\u30B9\u306E\u5834\u5408\u306FNEE\u304C\u52B9\u304B\u306A\u3044\u306E\u3067\u3001
            // \u6B21\u306E\u30D0\u30A6\u30F3\u30B9\u3067\u30E9\u30A4\u30C8\u306B\u5F53\u305F\u3063\u305F\u3089\u767A\u5149\u3092\u52A0\u7B97\u3059\u308B
            specular_bounce = true;
        }

        // 3. Scatter
        var scattered_dir: vec3<f32>;
        if mat_type == 0u {
            let target_p = hit_p + normal + random_unit_vector(rng);
            scattered_dir = normalize(target_p - hit_p);
            throughput *= albedo;
        } else if mat_type == 1u {
            let reflected = reflect(ray.direction, normal);
            let fuzz = tri.data1.x;
            scattered_dir = normalize(reflected + fuzz * random_in_unit_disk(rng));
            if dot(scattered_dir, normal) <= 0.0 { break; }
            throughput *= albedo;
        } else {
            // Dielectric
            scattered_dir = reflect(ray.direction, normal);
            throughput *= albedo; 
            // \u203B\u672C\u6765\u306F\u5C48\u6298\u51E6\u7406\u304C\u5FC5\u8981\u3067\u3059\u304C\u3001\u65E2\u5B58\u30B3\u30FC\u30C9\u306B\u5408\u308F\u305B\u3066\u7701\u7565
        }

        ray = Ray(hit_p + normal * 1e-4, scattered_dir);

        // RR (\u5909\u66F4\u306A\u3057)
        if depth > 3u {
            let p = max(throughput.r, max(throughput.g, throughput.b));
            if rand_pcg(rng) > p { break; }
            throughput /= p;
        }
    }
    return radiance;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(outputTex);
    if id.x >= dims.x || id.y >= dims.y { return; }
    let p_idx = id.y * dims.x + id.x;
    var rng = init_rng(p_idx, scene.rand_seed);

    var col = vec3(0.);
    for (var s = 0u; s < SPP; s++) {
        var off = vec3(0.);
        if scene.camera.lens_radius > 0. {
            let rd = scene.camera.lens_radius * random_in_unit_disk(&rng);
            off = scene.camera.u * rd.x + scene.camera.v * rd.y;
        }
        let u = (f32(id.x) + rand_pcg(&rng)) / f32(dims.x);
        let v = 1. - (f32(id.y) + rand_pcg(&rng)) / f32(dims.y);
        let d = scene.camera.lower_left_corner + u * scene.camera.horizontal + v * scene.camera.vertical - scene.camera.origin - off;
        col += ray_color(Ray(scene.camera.origin + off, d), &rng);
    }
    col /= f32(SPP);

    var acc = vec4(0.);
    if scene.frame_count > 1u { acc = accumulateBuffer[p_idx]; }
    let new_acc = acc + vec4(col, 1.0);
    accumulateBuffer[p_idx] = new_acc;
    let hdr_color = new_acc.rgb / new_acc.a;
    let mapped = aces_tone_mapping(hdr_color);
    let out = pow(mapped, vec3(1.0 / 2.2));
    textureStore(outputTex, vec2<i32>(id.xy), vec4(out, 1.));
}
`;
  class Z {
    constructor(e) {
      __publicField(this, "device");
      __publicField(this, "context");
      __publicField(this, "pipeline");
      __publicField(this, "bindGroupLayout");
      __publicField(this, "bindGroup");
      __publicField(this, "renderTarget");
      __publicField(this, "renderTargetView");
      __publicField(this, "accumulateBuffer");
      __publicField(this, "sceneUniformBuffer");
      __publicField(this, "geometryBuffer");
      __publicField(this, "nodesBuffer");
      __publicField(this, "topologyBuffer");
      __publicField(this, "instanceBuffer");
      __publicField(this, "lightsBuffer");
      __publicField(this, "texture");
      __publicField(this, "defaultTexture");
      __publicField(this, "sampler");
      __publicField(this, "bufferSize", 0);
      __publicField(this, "canvas");
      __publicField(this, "blasOffset", 0);
      __publicField(this, "vertexCount", 0);
      __publicField(this, "seed", Math.floor(Math.random() * 16777215));
      __publicField(this, "uniformMixedData", new Uint32Array(8));
      this.canvas = e;
    }
    async init() {
      if (!navigator.gpu) throw new Error("WebGPU not supported.");
      const e = await navigator.gpu.requestAdapter({
        powerPreference: "high-performance"
      });
      if (!e) throw new Error("No adapter");
      console.log("Max Storage Buffers Per Shader Stage:", e.limits.maxStorageBuffersPerShaderStage), this.device = await e.requestDevice({
        requiredLimits: {
          maxStorageBuffersPerShaderStage: 10
        }
      }), this.context = this.canvas.getContext("webgpu"), this.context.configure({
        device: this.device,
        format: "rgba8unorm",
        usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT
      }), this.sceneUniformBuffer = this.device.createBuffer({
        size: 128,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
      }), this.sampler = this.device.createSampler({
        magFilter: "linear",
        minFilter: "linear",
        mipmapFilter: "linear",
        addressModeU: "repeat",
        addressModeV: "repeat"
      }), this.createDefaultTexture(), this.texture = this.defaultTexture;
    }
    createDefaultTexture() {
      const e = new Uint8Array([
        255,
        255,
        255,
        255
      ]);
      this.defaultTexture = this.device.createTexture({
        size: [
          1,
          1,
          1
        ],
        format: "rgba8unorm",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
      }), this.device.queue.writeTexture({
        texture: this.defaultTexture,
        origin: [
          0,
          0,
          0
        ]
      }, e, {
        bytesPerRow: 256,
        rowsPerImage: 1
      }, [
        1,
        1
      ]);
    }
    buildPipeline(e, t) {
      let n = Q;
      n = n.replace(/const\s+MAX_DEPTH\s*=\s*\d+u;/, `const MAX_DEPTH = ${e}u;`), n = n.replace(/const\s+SPP\s*=\s*\d+u;/, `const SPP = ${t}u;`);
      const r = this.device.createShaderModule({
        label: "RayTracing",
        code: n
      });
      this.pipeline = this.device.createComputePipeline({
        label: "Main Pipeline",
        layout: "auto",
        compute: {
          module: r,
          entryPoint: "main"
        }
      }), this.bindGroupLayout = this.pipeline.getBindGroupLayout(0);
    }
    updateScreenSize(e, t) {
      this.canvas.width = e, this.canvas.height = t, this.renderTarget && this.renderTarget.destroy(), this.renderTarget = this.device.createTexture({
        size: [
          e,
          t
        ],
        format: "rgba8unorm",
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC
      }), this.renderTargetView = this.renderTarget.createView(), this.bufferSize = e * t * 16, this.accumulateBuffer && this.accumulateBuffer.destroy(), this.accumulateBuffer = this.device.createBuffer({
        size: this.bufferSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
      });
    }
    resetAccumulation() {
      this.accumulateBuffer && this.device.queue.writeBuffer(this.accumulateBuffer, 0, new Float32Array(this.bufferSize / 4));
    }
    async loadTexturesFromWorld(e) {
      const t = e.textureCount;
      if (t === 0) {
        this.createDefaultTexture();
        return;
      }
      console.log(`Loading ${t} textures...`);
      const n = [];
      for (let r = 0; r < t; r++) {
        const i = e.getTexture(r);
        if (i) try {
          const o = new Blob([
            i
          ]), l = await createImageBitmap(o, {
            resizeWidth: 1024,
            resizeHeight: 1024
          });
          n.push(l);
        } catch (o) {
          console.warn(`Failed tex ${r}`, o), n.push(await this.createFallbackBitmap());
        }
        else n.push(await this.createFallbackBitmap());
      }
      this.texture && this.texture.destroy(), this.texture = this.device.createTexture({
        size: [
          1024,
          1024,
          n.length
        ],
        format: "rgba8unorm",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT
      });
      for (let r = 0; r < n.length; r++) this.device.queue.copyExternalImageToTexture({
        source: n[r]
      }, {
        texture: this.texture,
        origin: [
          0,
          0,
          r
        ]
      }, [
        1024,
        1024
      ]);
    }
    async createFallbackBitmap() {
      const e = document.createElement("canvas");
      e.width = 1024, e.height = 1024;
      const t = e.getContext("2d");
      return t.fillStyle = "white", t.fillRect(0, 0, 1024, 1024), await createImageBitmap(e);
    }
    ensureBuffer(e, t, n) {
      if (e && e.size >= t) return e;
      e && e.destroy();
      let r = Math.ceil(t * 1.5);
      return r = r + 3 & -4, r = Math.max(r, 16), this.device.createBuffer({
        label: n,
        size: r,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
      });
    }
    updateBuffer(e, t) {
      const n = t.byteLength;
      let r = false, i;
      return e === "topology" ? ((!this.topologyBuffer || this.topologyBuffer.size < n) && (r = true), this.topologyBuffer = this.ensureBuffer(this.topologyBuffer, n, "TopologyBuffer"), i = this.topologyBuffer) : e === "instance" ? ((!this.instanceBuffer || this.instanceBuffer.size < n) && (r = true), this.instanceBuffer = this.ensureBuffer(this.instanceBuffer, n, "InstanceBuffer"), i = this.instanceBuffer) : ((!this.lightsBuffer || this.lightsBuffer.size < n) && (r = true), this.lightsBuffer = this.ensureBuffer(this.lightsBuffer, n, "LightsBuffer"), i = this.lightsBuffer), this.device.queue.writeBuffer(i, 0, t, 0, t.length), r;
    }
    updateCombinedGeometry(e, t, n) {
      const r = e.byteLength + t.byteLength + n.byteLength;
      let i = false;
      (!this.geometryBuffer || this.geometryBuffer.size < r) && (i = true);
      const o = e.length / 4;
      this.vertexCount = o, this.geometryBuffer = this.ensureBuffer(this.geometryBuffer, r, "GeometryBuffer"), !(n.length >= o * 2) && o > 0 && console.warn(`UV buffer mismatch: V=${o}, UV=${n.length / 2}. Filling 0.`);
      let d = 0;
      return this.device.queue.writeBuffer(this.geometryBuffer, d, e), d += e.byteLength, this.device.queue.writeBuffer(this.geometryBuffer, d, t), d += t.byteLength, this.device.queue.writeBuffer(this.geometryBuffer, d, n), i;
    }
    updateCombinedBVH(e, t) {
      const n = e.byteLength, r = t.byteLength, i = n + r;
      let o = false;
      return (!this.nodesBuffer || this.nodesBuffer.size < i) && (o = true), this.nodesBuffer = this.ensureBuffer(this.nodesBuffer, i, "NodesBuffer"), this.device.queue.writeBuffer(this.nodesBuffer, 0, e), this.device.queue.writeBuffer(this.nodesBuffer, n, t), this.blasOffset = e.length / 8, o;
    }
    updateSceneUniforms(e, t, n) {
      this.sceneUniformBuffer && (this.device.queue.writeBuffer(this.sceneUniformBuffer, 0, e), this.uniformMixedData[0] = t, this.uniformMixedData[1] = this.blasOffset, this.uniformMixedData[2] = this.vertexCount, this.uniformMixedData[3] = 0, this.uniformMixedData[4] = n, this.device.queue.writeBuffer(this.sceneUniformBuffer, 96, this.uniformMixedData));
    }
    recreateBindGroup() {
      !this.renderTargetView || !this.accumulateBuffer || !this.geometryBuffer || !this.nodesBuffer || !this.sceneUniformBuffer || !this.lightsBuffer || (this.bindGroup = this.device.createBindGroup({
        layout: this.bindGroupLayout,
        entries: [
          {
            binding: 0,
            resource: this.renderTargetView
          },
          {
            binding: 1,
            resource: {
              buffer: this.accumulateBuffer
            }
          },
          {
            binding: 2,
            resource: {
              buffer: this.sceneUniformBuffer
            }
          },
          {
            binding: 3,
            resource: {
              buffer: this.geometryBuffer
            }
          },
          {
            binding: 4,
            resource: {
              buffer: this.topologyBuffer
            }
          },
          {
            binding: 5,
            resource: {
              buffer: this.nodesBuffer
            }
          },
          {
            binding: 6,
            resource: {
              buffer: this.instanceBuffer
            }
          },
          {
            binding: 7,
            resource: this.texture.createView({
              dimension: "2d-array"
            })
          },
          {
            binding: 8,
            resource: this.sampler
          },
          {
            binding: 9,
            resource: {
              buffer: this.lightsBuffer
            }
          }
        ]
      }));
    }
    compute(e) {
      if (!this.bindGroup) return;
      this.seed++, this.uniformMixedData[0] = e, this.uniformMixedData[3] = this.seed, this.device.queue.writeBuffer(this.sceneUniformBuffer, 96, this.uniformMixedData);
      const t = Math.ceil(this.canvas.width / 8), n = Math.ceil(this.canvas.height / 8), r = this.device.createCommandEncoder(), i = r.beginComputePass();
      i.setPipeline(this.pipeline), i.setBindGroup(0, this.bindGroup), i.dispatchWorkgroups(t, n), i.end(), this.device.queue.submit([
        r.finish()
      ]);
    }
    present() {
      if (!this.renderTarget) return;
      const e = this.device.createCommandEncoder();
      e.copyTextureToTexture({
        texture: this.renderTarget
      }, {
        texture: this.context.getCurrentTexture()
      }, {
        width: this.canvas.width,
        height: this.canvas.height,
        depthOrArrayLayers: 1
      }), this.device.queue.submit([
        e.finish()
      ]);
    }
  }
  function ee(s) {
    return new Worker("/webgpu-raytracer/assets/wasm-worker-BwHKOs2c.js", {
      name: s == null ? void 0 : s.name
    });
  }
  class te {
    constructor() {
      __publicField(this, "worker");
      __publicField(this, "resolveReady", null);
      __publicField(this, "_vertices", new Float32Array(0));
      __publicField(this, "_normals", new Float32Array(0));
      __publicField(this, "_uvs", new Float32Array(0));
      __publicField(this, "_mesh_topology", new Uint32Array(0));
      __publicField(this, "_lights", new Uint32Array(0));
      __publicField(this, "_tlas", new Float32Array(0));
      __publicField(this, "_blas", new Float32Array(0));
      __publicField(this, "_instances", new Float32Array(0));
      __publicField(this, "_cameraData", new Float32Array(24));
      __publicField(this, "_textureCount", 0);
      __publicField(this, "_textures", []);
      __publicField(this, "_animations", []);
      __publicField(this, "hasNewData", false);
      __publicField(this, "hasNewGeometry", false);
      __publicField(this, "pendingUpdate", false);
      __publicField(this, "resolveSceneLoad", null);
      __publicField(this, "updateResolvers", []);
      __publicField(this, "lastWidth", -1);
      __publicField(this, "lastHeight", -1);
      this.worker = new ee(), this.worker.onmessage = this.handleMessage.bind(this);
    }
    get lights() {
      return this._lights;
    }
    get lightCount() {
      return this._lights.length / 2;
    }
    async initWasm() {
      return new Promise((e) => {
        this.resolveReady = e, this.worker.postMessage({
          type: "INIT"
        });
      });
    }
    handleMessage(e) {
      var _a, _b;
      const t = e.data;
      switch (t.type) {
        case "READY":
          console.log("Main: Worker Ready"), (_a = this.resolveReady) == null ? void 0 : _a.call(this);
          break;
        case "SCENE_LOADED":
          this._vertices = t.vertices, this._normals = t.normals, this._uvs = t.uvs, this._mesh_topology = t.mesh_topology, this._lights = t.lights, this._tlas = t.tlas, this._blas = t.blas, this._instances = t.instances, this._cameraData = t.camera, this._textureCount = t.textureCount, this._textures = t.textures || [], this._animations = t.animations || [], this.hasNewData = true, this.hasNewGeometry = true, (_b = this.resolveSceneLoad) == null ? void 0 : _b.call(this);
          break;
        case "UPDATE_RESULT":
          this._tlas = t.tlas, this._blas = t.blas, this._instances = t.instances, this._lights = t.lights, this._cameraData = t.camera, t.vertices && (this._vertices = t.vertices, this.hasNewGeometry = true), t.normals && (this._normals = t.normals), t.uvs && (this._uvs = t.uvs), t.mesh_topology && (this._mesh_topology = t.mesh_topology), this.hasNewData = true, this.pendingUpdate = false, this.updateResolvers.forEach((n) => n()), this.updateResolvers = [];
          break;
      }
    }
    getAnimationList() {
      return this._animations;
    }
    getTexture(e) {
      return e >= 0 && e < this._textures.length ? this._textures[e] : null;
    }
    loadScene(e, t, n) {
      return this.lastWidth = -1, this.lastHeight = -1, new Promise((r) => {
        this.resolveSceneLoad = r, this.worker.postMessage({
          type: "LOAD_SCENE",
          sceneName: e,
          objSource: t,
          glbData: n
        }, n ? [
          n.buffer
        ] : []);
      });
    }
    waitForNextUpdate() {
      return new Promise((e) => {
        this.updateResolvers.push(e);
      });
    }
    update(e) {
      this.pendingUpdate || (this.pendingUpdate = true, this.worker.postMessage({
        type: "UPDATE",
        time: e
      }));
    }
    updateCamera(e, t) {
      this.lastWidth === e && this.lastHeight === t || (this.lastWidth = e, this.lastHeight = t, this.worker.postMessage({
        type: "UPDATE_CAMERA",
        width: e,
        height: t
      }));
    }
    loadAnimation(e) {
      this.worker.postMessage({
        type: "LOAD_ANIMATION",
        data: e
      }, [
        e.buffer
      ]);
    }
    setAnimation(e) {
      this.worker.postMessage({
        type: "SET_ANIMATION",
        index: e
      });
    }
    get vertices() {
      return this._vertices;
    }
    get normals() {
      return this._normals;
    }
    get uvs() {
      return this._uvs;
    }
    get mesh_topology() {
      return this._mesh_topology;
    }
    get tlas() {
      return this._tlas;
    }
    get blas() {
      return this._blas;
    }
    get instances() {
      return this._instances;
    }
    get cameraData() {
      return this._cameraData;
    }
    get textureCount() {
      return this._textureCount;
    }
    get hasWorld() {
      return this._vertices.length > 0;
    }
    printStats() {
      console.log(`Scene Stats (Worker Proxy): V=${this.vertices.length / 4}, Topo=${this.mesh_topology.length / 12}, I=${this.instances.length / 16}, TLAS=${this.tlas.length / 8}, BLAS=${this.blas.length / 8}, Anim=${this._animations.length}, Lights=${this._lights.length / 2}`);
    }
  }
  const f = {
    defaultWidth: 720,
    defaultHeight: 480,
    defaultDepth: 10,
    defaultSPP: 1,
    signalingServerUrl: "ws://localhost:8080",
    rtcConfig: {
      iceServers: JSON.parse('[{"urls": "stun:stun.l.google.com:19302"}]')
    },
    ids: {
      canvas: "gpu-canvas",
      renderBtn: "render-btn",
      sceneSelect: "scene-select",
      resWidth: "res-width",
      resHeight: "res-height",
      objFile: "obj-file",
      maxDepth: "max-depth",
      sppFrame: "spp-frame",
      recompileBtn: "recompile-btn",
      updateInterval: "update-interval",
      animSelect: "anim-select",
      recordBtn: "record-btn",
      recFps: "rec-fps",
      recDuration: "rec-duration",
      recSpp: "rec-spp",
      recBatch: "rec-batch",
      btnHost: "btn-host",
      btnWorker: "btn-worker",
      statusDiv: "status"
    }
  };
  class ne {
    constructor() {
      __publicField(this, "canvas");
      __publicField(this, "btnRender");
      __publicField(this, "sceneSelect");
      __publicField(this, "inputWidth");
      __publicField(this, "inputHeight");
      __publicField(this, "inputFile");
      __publicField(this, "inputDepth");
      __publicField(this, "inputSPP");
      __publicField(this, "btnRecompile");
      __publicField(this, "inputUpdateInterval");
      __publicField(this, "animSelect");
      __publicField(this, "btnRecord");
      __publicField(this, "inputRecFps");
      __publicField(this, "inputRecDur");
      __publicField(this, "inputRecSpp");
      __publicField(this, "inputRecBatch");
      __publicField(this, "btnHost");
      __publicField(this, "btnWorker");
      __publicField(this, "statusDiv");
      __publicField(this, "statsDiv");
      __publicField(this, "onRenderStart", null);
      __publicField(this, "onRenderStop", null);
      __publicField(this, "onSceneSelect", null);
      __publicField(this, "onResolutionChange", null);
      __publicField(this, "onRecompile", null);
      __publicField(this, "onFileSelect", null);
      __publicField(this, "onAnimSelect", null);
      __publicField(this, "onRecordStart", null);
      __publicField(this, "onConnectHost", null);
      __publicField(this, "onConnectWorker", null);
      this.canvas = this.el(f.ids.canvas), this.btnRender = this.el(f.ids.renderBtn), this.sceneSelect = this.el(f.ids.sceneSelect), this.inputWidth = this.el(f.ids.resWidth), this.inputHeight = this.el(f.ids.resHeight), this.inputFile = this.setupFileInput(), this.inputDepth = this.el(f.ids.maxDepth), this.inputSPP = this.el(f.ids.sppFrame), this.btnRecompile = this.el(f.ids.recompileBtn), this.inputUpdateInterval = this.el(f.ids.updateInterval), this.animSelect = this.el(f.ids.animSelect), this.btnRecord = this.el(f.ids.recordBtn), this.inputRecFps = this.el(f.ids.recFps), this.inputRecDur = this.el(f.ids.recDuration), this.inputRecSpp = this.el(f.ids.recSpp), this.inputRecBatch = this.el(f.ids.recBatch), this.btnHost = this.el(f.ids.btnHost), this.btnWorker = this.el(f.ids.btnWorker), this.statusDiv = this.el(f.ids.statusDiv), this.statsDiv = this.createStatsDiv(), this.bindEvents();
    }
    el(e) {
      const t = document.getElementById(e);
      if (!t) throw new Error(`Element not found: ${e}`);
      return t;
    }
    setupFileInput() {
      const e = this.el(f.ids.objFile);
      return e && (e.accept = ".obj,.glb,.vrm"), e;
    }
    createStatsDiv() {
      const e = document.createElement("div");
      return Object.assign(e.style, {
        position: "fixed",
        bottom: "10px",
        left: "10px",
        color: "#0f0",
        background: "rgba(0,0,0,0.7)",
        padding: "8px",
        fontFamily: "monospace",
        fontSize: "14px",
        pointerEvents: "none",
        zIndex: "9999",
        borderRadius: "4px"
      }), document.body.appendChild(e), e;
    }
    bindEvents() {
      this.btnRender.addEventListener("click", () => {
        var _a, _b;
        this.btnRender.textContent === "Render Start" || this.btnRender.textContent === "Resume Rendering" ? ((_a = this.onRenderStart) == null ? void 0 : _a.call(this), this.updateRenderButton(true)) : ((_b = this.onRenderStop) == null ? void 0 : _b.call(this), this.updateRenderButton(false));
      }), this.sceneSelect.addEventListener("change", () => {
        var _a;
        return (_a = this.onSceneSelect) == null ? void 0 : _a.call(this, this.sceneSelect.value);
      });
      const e = () => {
        var _a;
        return (_a = this.onResolutionChange) == null ? void 0 : _a.call(this, parseInt(this.inputWidth.value) || f.defaultWidth, parseInt(this.inputHeight.value) || f.defaultHeight);
      };
      this.inputWidth.addEventListener("change", e), this.inputHeight.addEventListener("change", e), this.btnRecompile.addEventListener("click", () => {
        var _a;
        return (_a = this.onRecompile) == null ? void 0 : _a.call(this, parseInt(this.inputDepth.value) || 10, parseInt(this.inputSPP.value) || 1);
      }), this.inputFile.addEventListener("change", (t) => {
        var _a, _b;
        const n = (_a = t.target.files) == null ? void 0 : _a[0];
        n && ((_b = this.onFileSelect) == null ? void 0 : _b.call(this, n));
      }), this.animSelect.addEventListener("change", () => {
        var _a;
        const t = parseInt(this.animSelect.value, 10);
        (_a = this.onAnimSelect) == null ? void 0 : _a.call(this, t);
      }), this.btnRecord.addEventListener("click", () => {
        var _a;
        return (_a = this.onRecordStart) == null ? void 0 : _a.call(this);
      }), this.btnHost.addEventListener("click", () => {
        var _a;
        return (_a = this.onConnectHost) == null ? void 0 : _a.call(this);
      }), this.btnWorker.addEventListener("click", () => {
        var _a;
        return (_a = this.onConnectWorker) == null ? void 0 : _a.call(this);
      });
    }
    updateRenderButton(e) {
      this.btnRender.textContent = e ? "Stop Rendering" : "Resume Rendering";
    }
    updateStats(e, t, n) {
      this.statsDiv.textContent = `FPS: ${e} | ${t.toFixed(2)}ms | Frame: ${n}`;
    }
    setStatus(e) {
      this.statusDiv.textContent = e;
    }
    setConnectionState(e) {
      e === "host" ? (this.btnHost.textContent = "Disconnect", this.btnHost.disabled = false, this.btnWorker.textContent = "Worker", this.btnWorker.disabled = true) : e === "worker" ? (this.btnHost.textContent = "Host", this.btnHost.disabled = true, this.btnWorker.textContent = "Disconnect", this.btnWorker.disabled = false) : (this.btnHost.textContent = "Host", this.btnHost.disabled = false, this.btnWorker.textContent = "Worker", this.btnWorker.disabled = false, this.statusDiv.textContent = "Offline");
    }
    setRecordingState(e, t) {
      e ? (this.btnRecord.disabled = true, this.btnRecord.textContent = t || "Recording...", this.btnRender.textContent = "Resume Rendering") : (this.btnRecord.disabled = false, this.btnRecord.textContent = "\u25CF Rec");
    }
    updateAnimList(e) {
      if (this.animSelect.innerHTML = "", e.length === 0) {
        const t = document.createElement("option");
        t.text = "No Anim", this.animSelect.add(t), this.animSelect.disabled = true;
        return;
      }
      this.animSelect.disabled = false, e.forEach((t, n) => {
        const r = document.createElement("option");
        r.text = `[${n}] ${t}`, r.value = n.toString(), this.animSelect.add(r);
      }), this.animSelect.value = "0";
    }
    getRenderConfig() {
      return {
        width: parseInt(this.inputWidth.value, 10) || f.defaultWidth,
        height: parseInt(this.inputHeight.value, 10) || f.defaultHeight,
        fps: parseInt(this.inputRecFps.value, 10) || 30,
        duration: parseFloat(this.inputRecDur.value) || 3,
        spp: parseInt(this.inputRecSpp.value, 10) || 64,
        batch: parseInt(this.inputRecBatch.value, 10) || 4,
        anim: parseInt(this.animSelect.value, 10) || 0
      };
    }
    setRenderConfig(e) {
      this.inputWidth.value = e.width.toString(), this.inputHeight.value = e.height.toString(), this.inputRecFps.value = e.fps.toString(), this.inputRecDur.value = e.duration.toString(), this.inputRecSpp.value = e.spp.toString(), this.inputRecBatch.value = e.batch.toString();
    }
  }
  class re {
    constructor(e, t, n) {
      __publicField(this, "isRecording", false);
      __publicField(this, "renderer");
      __publicField(this, "worldBridge");
      __publicField(this, "canvas");
      this.renderer = e, this.worldBridge = t, this.canvas = n;
    }
    get recording() {
      return this.isRecording;
    }
    async record(e, t, n) {
      if (this.isRecording) return;
      this.isRecording = true;
      const { Muxer: r, ArrayBufferTarget: i } = await G(async () => {
        const { Muxer: p, ArrayBufferTarget: u } = await import("./webm-muxer-MLtUgOCn.js");
        return {
          Muxer: p,
          ArrayBufferTarget: u
        };
      }, []), o = Math.ceil(e.fps * e.duration);
      console.log(`Starting recording: ${o} frames @ ${e.fps}fps (VP9)`);
      const l = new r({
        target: new i(),
        video: {
          codec: "V_VP9",
          width: this.canvas.width,
          height: this.canvas.height,
          frameRate: e.fps
        }
      }), d = new VideoEncoder({
        output: (p, u) => l.addVideoChunk(p, u),
        error: (p) => console.error("VideoEncoder Error:", p)
      });
      d.configure({
        codec: "vp09.00.10.08",
        width: this.canvas.width,
        height: this.canvas.height,
        bitrate: 12e6
      });
      try {
        await this.renderAndEncode(o, e, d, t, e.startFrame || 0), await d.flush(), l.finalize();
        const { buffer: p } = l.target, u = new Blob([
          p
        ], {
          type: "video/webm"
        }), b = URL.createObjectURL(u);
        n(b, u);
      } catch (p) {
        throw console.error("Recording failed:", p), p;
      } finally {
        this.isRecording = false;
      }
    }
    async recordChunks(e, t) {
      if (this.isRecording) throw new Error("Already recording");
      this.isRecording = true;
      const n = [], r = Math.ceil(e.fps * e.duration), i = new VideoEncoder({
        output: (o, l) => {
          const d = new Uint8Array(o.byteLength);
          o.copyTo(d), n.push({
            type: o.type,
            timestamp: o.timestamp,
            duration: o.duration,
            data: d.buffer,
            decoderConfig: l == null ? void 0 : l.decoderConfig
          });
        },
        error: (o) => console.error("VideoEncoder Error:", o)
      });
      i.configure({
        codec: "vp09.00.10.08",
        width: this.canvas.width,
        height: this.canvas.height,
        bitrate: 12e6
      });
      try {
        return await this.renderAndEncode(r, e, i, t, e.startFrame || 0), await i.flush(), n;
      } finally {
        this.isRecording = false;
      }
    }
    async renderAndEncode(e, t, n, r, i = 0) {
      const o = i;
      this.worldBridge.update(o / t.fps), await this.worldBridge.waitForNextUpdate();
      for (let l = 0; l < e; l++) {
        r(l, e), await new Promise((u) => setTimeout(u, 0)), await this.updateSceneBuffers();
        let d = null;
        if (l < e - 1) {
          const u = i + l + 1;
          this.worldBridge.update(u / t.fps), d = this.worldBridge.waitForNextUpdate();
        }
        await this.renderFrame(t.spp, t.batch), n.encodeQueueSize > 5 && await n.flush();
        const p = new VideoFrame(this.canvas, {
          timestamp: (i + l) * 1e6 / t.fps,
          duration: 1e6 / t.fps
        });
        n.encode(p, {
          keyFrame: l % t.fps === 0
        }), p.close(), d && await d;
      }
    }
    async updateSceneBuffers() {
      let e = false;
      e || (e = this.renderer.updateCombinedBVH(this.worldBridge.tlas, this.worldBridge.blas)), e || (e = this.renderer.updateBuffer("instance", this.worldBridge.instances)), e || (e = this.renderer.updateCombinedGeometry(this.worldBridge.vertices, this.worldBridge.normals, this.worldBridge.uvs)), e || (e = this.renderer.updateBuffer("topology", this.worldBridge.mesh_topology)), e || (e = this.renderer.updateBuffer("lights", this.worldBridge.lights)), this.worldBridge.updateCamera(this.canvas.width, this.canvas.height), this.renderer.updateSceneUniforms(this.worldBridge.cameraData, 0, this.worldBridge.lightCount), e && this.renderer.recreateBindGroup(), this.renderer.resetAccumulation();
    }
    async renderFrame(e, t) {
      let n = 0;
      for (; n < e; ) {
        const r = Math.min(t, e - n);
        for (let i = 0; i < r; i++) this.renderer.compute(n + i);
        n += r, this.renderer.present(), await this.renderer.device.queue.onSubmittedWorkDone();
      }
    }
  }
  const se = {
    iceServers: [
      {
        urls: "stun:stun.l.google.com:19302"
      }
    ]
  };
  class F {
    constructor(e, t) {
      __publicField(this, "pc");
      __publicField(this, "dc", null);
      __publicField(this, "remoteId");
      __publicField(this, "sendSignal");
      __publicField(this, "receiveBuffer", new Uint8Array(0));
      __publicField(this, "receivedBytes", 0);
      __publicField(this, "sceneMeta", null);
      __publicField(this, "resultMeta", null);
      __publicField(this, "onSceneReceived", null);
      __publicField(this, "onRenderRequest", null);
      __publicField(this, "onRenderResult", null);
      __publicField(this, "onDataChannelOpen", null);
      __publicField(this, "onAckReceived", null);
      __publicField(this, "onWorkerReady", null);
      this.remoteId = e, this.sendSignal = t, this.pc = new RTCPeerConnection(se), this.pc.onicecandidate = (n) => {
        n.candidate && this.sendSignal({
          type: "candidate",
          candidate: n.candidate.toJSON(),
          targetId: this.remoteId
        });
      };
    }
    async startAsHost() {
      this.dc = this.pc.createDataChannel("render-channel"), this.setupDataChannel();
      const e = await this.pc.createOffer();
      await this.pc.setLocalDescription(e), this.sendSignal({
        type: "offer",
        sdp: e,
        targetId: this.remoteId
      });
    }
    async handleOffer(e) {
      this.pc.ondatachannel = (n) => {
        this.dc = n.channel, this.setupDataChannel();
      }, await this.pc.setRemoteDescription(new RTCSessionDescription(e));
      const t = await this.pc.createAnswer();
      await this.pc.setLocalDescription(t), this.sendSignal({
        type: "answer",
        sdp: t,
        targetId: this.remoteId
      });
    }
    async handleAnswer(e) {
      await this.pc.setRemoteDescription(new RTCSessionDescription(e));
    }
    async handleCandidate(e) {
      await this.pc.addIceCandidate(new RTCIceCandidate(e));
    }
    async sendScene(e, t, n) {
      if (!this.dc || this.dc.readyState !== "open") return;
      let r;
      typeof e == "string" ? r = new TextEncoder().encode(e) : r = new Uint8Array(e);
      const i = {
        type: "SCENE_INIT",
        totalBytes: r.byteLength,
        config: {
          ...n,
          fileType: t
        }
      };
      this.sendData(i), await this.sendBinaryChunks(r);
    }
    async sendRenderResult(e, t) {
      if (!this.dc || this.dc.readyState !== "open") return;
      let n = 0;
      const r = e.map((l) => {
        const d = l.data.byteLength;
        return n += d, {
          type: l.type,
          timestamp: l.timestamp,
          duration: l.duration,
          size: d,
          decoderConfig: l.decoderConfig
        };
      });
      console.log(`[RTC] Sending Render Result: ${n} bytes, ${e.length} chunks`), this.sendData({
        type: "RENDER_RESULT",
        startFrame: t,
        totalBytes: n,
        chunksMeta: r
      });
      const i = new Uint8Array(n);
      let o = 0;
      for (const l of e) i.set(new Uint8Array(l.data), o), o += l.data.byteLength;
      await this.sendBinaryChunks(i);
    }
    async sendBinaryChunks(e) {
      let n = 0;
      const r = () => new Promise((i) => {
        const o = setInterval(() => {
          (!this.dc || this.dc.bufferedAmount < 65536) && (clearInterval(o), i());
        }, 5);
      });
      for (; n < e.byteLength; ) {
        this.dc && this.dc.bufferedAmount > 256 * 1024 && await r();
        const i = Math.min(n + 16384, e.byteLength);
        if (this.dc) try {
          this.dc.send(e.subarray(n, i));
        } catch {
        }
        n = i, n % (16384 * 5) === 0 && await new Promise((o) => setTimeout(o, 0));
      }
      console.log("[RTC] Transfer Complete");
    }
    setupDataChannel() {
      this.dc && (this.dc.binaryType = "arraybuffer", this.dc.onopen = () => {
        console.log("[RTC] DataChannel Open"), this.onDataChannelOpen && this.onDataChannelOpen();
      }, this.dc.onmessage = (e) => {
        const t = e.data;
        if (typeof t == "string") try {
          const n = JSON.parse(t);
          this.handleControlMessage(n);
        } catch {
        }
        else t instanceof ArrayBuffer && this.handleBinaryChunk(t);
      });
    }
    handleControlMessage(e) {
      var _a, _b;
      e.type === "SCENE_INIT" ? (console.log(`[RTC] Receiving Scene: ${e.config.fileType}, ${e.totalBytes} bytes`), this.sceneMeta = {
        config: e.config,
        totalBytes: e.totalBytes
      }, this.receiveBuffer = new Uint8Array(e.totalBytes), this.receivedBytes = 0) : e.type === "SCENE_ACK" ? (console.log(`[RTC] Scene ACK: ${e.receivedBytes} bytes`), this.onAckReceived && this.onAckReceived(e.receivedBytes)) : e.type === "RENDER_REQUEST" ? (console.log(`[RTC] Render Request: Frame ${e.startFrame}, Count ${e.frameCount}`), (_a = this.onRenderRequest) == null ? void 0 : _a.call(this, e.startFrame, e.frameCount, e.config)) : e.type === "RENDER_RESULT" ? (console.log(`[RTC] Receiving Render Result: ${e.totalBytes} bytes`), this.resultMeta = {
        startFrame: e.startFrame,
        totalBytes: e.totalBytes,
        chunksMeta: e.chunksMeta
      }, this.receiveBuffer = new Uint8Array(e.totalBytes), this.receivedBytes = 0) : e.type === "WORKER_READY" && (console.log("[RTC] Worker Ready Signal Received"), (_b = this.onWorkerReady) == null ? void 0 : _b.call(this));
    }
    handleBinaryChunk(e) {
      var _a, _b;
      try {
        const t = new Uint8Array(e);
        if (this.receivedBytes + t.byteLength > this.receiveBuffer.byteLength) {
          console.error("[RTC] Receive Buffer Overflow!");
          return;
        }
        this.receiveBuffer.set(t, this.receivedBytes), this.receivedBytes += t.byteLength;
      } catch (t) {
        console.error("[RTC] Error handling binary chunk", t);
        return;
      }
      if (this.sceneMeta) {
        if (this.receivedBytes >= this.sceneMeta.totalBytes) {
          console.log("[RTC] Scene Download Complete!");
          let t;
          this.sceneMeta.config.fileType === "obj" ? t = new TextDecoder().decode(this.receiveBuffer) : t = this.receiveBuffer.buffer, (_a = this.onSceneReceived) == null ? void 0 : _a.call(this, t, this.sceneMeta.config), this.sceneMeta = null;
        }
      } else if (this.resultMeta && this.receivedBytes >= this.resultMeta.totalBytes) {
        console.log("[RTC] Render Result Complete!");
        const t = [];
        let n = 0;
        for (const r of this.resultMeta.chunksMeta) {
          const i = this.receiveBuffer.slice(n, n + r.size);
          t.push({
            type: r.type,
            timestamp: r.timestamp,
            duration: r.duration,
            data: i.buffer,
            decoderConfig: r.decoderConfig
          }), n += r.size;
        }
        (_b = this.onRenderResult) == null ? void 0 : _b.call(this, t, this.resultMeta.startFrame), this.resultMeta = null;
      }
    }
    sendData(e) {
      var _a;
      ((_a = this.dc) == null ? void 0 : _a.readyState) === "open" && this.dc.send(JSON.stringify(e));
    }
    sendAck(e) {
      this.sendData({
        type: "SCENE_ACK",
        receivedBytes: e
      });
    }
    sendRenderRequest(e, t, n) {
      const r = {
        type: "RENDER_REQUEST",
        startFrame: e,
        frameCount: t,
        config: n
      };
      this.sendData(r);
    }
    sendWorkerReady() {
      this.sendData({
        type: "WORKER_READY"
      });
    }
    close() {
      this.dc && (this.dc.close(), this.dc = null), this.pc && this.pc.close(), console.log(`[RTC] Connection closed: ${this.remoteId}`);
    }
  }
  class ie {
    constructor() {
      __publicField(this, "ws", null);
      __publicField(this, "myRole", null);
      __publicField(this, "workers", /* @__PURE__ */ new Map());
      __publicField(this, "hostClient", null);
      __publicField(this, "onStatusChange", null);
      __publicField(this, "onWorkerJoined", null);
      __publicField(this, "onWorkerLeft", null);
      __publicField(this, "onHostConnected", null);
      __publicField(this, "onWorkerReady", null);
      __publicField(this, "onSceneReceived", null);
      __publicField(this, "onHostHello", null);
      __publicField(this, "onRenderResult", null);
      __publicField(this, "onRenderRequest", null);
    }
    connect(e) {
      var _a;
      if (this.ws) return;
      this.myRole = e, (_a = this.onStatusChange) == null ? void 0 : _a.call(this, `Connecting as ${e.toUpperCase()}...`);
      const t = "xWUaLfXQQkHZ9VmF";
      this.ws = new WebSocket(`${f.signalingServerUrl}?token=${t}`), this.ws.onopen = () => {
        var _a2;
        console.log("WS Connected"), (_a2 = this.onStatusChange) == null ? void 0 : _a2.call(this, `Waiting for Peer (${e.toUpperCase()})`), this.sendSignal({
          type: e === "host" ? "register_host" : "register_worker"
        });
      }, this.ws.onmessage = (n) => {
        const r = JSON.parse(n.data);
        this.handleMessage(r);
      }, this.ws.onclose = () => {
        var _a2;
        (_a2 = this.onStatusChange) == null ? void 0 : _a2.call(this, "Disconnected"), this.ws = null;
      };
    }
    disconnect() {
      var _a;
      this.ws && (this.ws.close(), this.ws = null), this.workers.forEach((e) => e.close()), this.workers.clear(), this.hostClient && (this.hostClient.close(), this.hostClient = null), (_a = this.onStatusChange) == null ? void 0 : _a.call(this, "Disconnected");
    }
    getWorkerCount() {
      return this.workers.size;
    }
    getWorkerIds() {
      return Array.from(this.workers.keys());
    }
    async sendRenderResult(e, t) {
      this.hostClient && await this.hostClient.sendRenderResult(e, t);
    }
    sendSignal(e) {
      var _a;
      ((_a = this.ws) == null ? void 0 : _a.readyState) === WebSocket.OPEN && this.ws.send(JSON.stringify(e));
    }
    async handleMessage(e) {
      this.myRole === "host" ? await this.handleHostMessage(e) : await this.handleWorkerMessage(e);
    }
    async handleHostMessage(e) {
      var _a, _b, _c, _d, _e;
      switch (e.type) {
        case "worker_joined":
          console.log(`Worker joined: ${e.workerId}`);
          const t = new F(e.workerId, (n) => this.sendSignal(n));
          this.workers.set(e.workerId, t), t.onDataChannelOpen = () => {
            var _a2;
            console.log(`[Host] Open for ${e.workerId}`), t.sendData({
              type: "HELLO",
              msg: "Hello from Host!"
            }), (_a2 = this.onWorkerJoined) == null ? void 0 : _a2.call(this, e.workerId);
          }, t.onAckReceived = (n) => {
            console.log(`Worker ${e.workerId} ACK: ${n}`);
          }, t.onRenderResult = (n, r) => {
            var _a2;
            console.log(`Received Render Result from ${e.workerId}: ${n.length} chunks`), (_a2 = this.onRenderResult) == null ? void 0 : _a2.call(this, n, r, e.workerId);
          }, t.onWorkerReady = () => {
            var _a2;
            (_a2 = this.onWorkerReady) == null ? void 0 : _a2.call(this, e.workerId);
          }, await t.startAsHost();
          break;
        case "worker_left":
          console.log(`Worker left: ${e.workerId}`), (_a = this.workers.get(e.workerId)) == null ? void 0 : _a.close(), this.workers.delete(e.workerId), (_b = this.onWorkerLeft) == null ? void 0 : _b.call(this, e.workerId);
          break;
        case "answer":
          e.fromId && await ((_c = this.workers.get(e.fromId)) == null ? void 0 : _c.handleAnswer(e.sdp));
          break;
        case "candidate":
          e.fromId && await ((_d = this.workers.get(e.fromId)) == null ? void 0 : _d.handleCandidate(e.candidate));
          break;
        case "host_exists":
          alert("Host already exists!");
          break;
        case "WORKER_READY":
          e.workerId && ((_e = this.onWorkerReady) == null ? void 0 : _e.call(this, e.workerId));
          break;
      }
    }
    async sendWorkerReady() {
      this.hostClient && this.hostClient.sendWorkerReady();
    }
    async handleWorkerMessage(e) {
      var _a, _b, _c;
      switch (e.type) {
        case "offer":
          e.fromId && (this.hostClient = new F(e.fromId, (t) => this.sendSignal(t)), await this.hostClient.handleOffer(e.sdp), (_a = this.onStatusChange) == null ? void 0 : _a.call(this, "Connected to Host!"), (_b = this.onHostConnected) == null ? void 0 : _b.call(this), this.hostClient.onDataChannelOpen = () => {
            var _a2, _b2;
            (_a2 = this.hostClient) == null ? void 0 : _a2.sendData({
              type: "HELLO",
              msg: "Hello from Worker!"
            }), (_b2 = this.onHostHello) == null ? void 0 : _b2.call(this);
          }, this.hostClient.onSceneReceived = (t, n) => {
            var _a2, _b2;
            (_a2 = this.onSceneReceived) == null ? void 0 : _a2.call(this, t, n);
            const r = typeof t == "string" ? t.length : t.byteLength;
            (_b2 = this.hostClient) == null ? void 0 : _b2.sendAck(r);
          }, this.hostClient.onRenderRequest = (t, n, r) => {
            var _a2;
            (_a2 = this.onRenderRequest) == null ? void 0 : _a2.call(this, t, n, r);
          });
          break;
        case "candidate":
          await ((_c = this.hostClient) == null ? void 0 : _c.handleCandidate(e.candidate));
          break;
      }
    }
    async broadcastScene(e, t, n) {
      const r = Array.from(this.workers.values()).map((i) => i.sendScene(e, t, n));
      await Promise.all(r);
    }
    async sendSceneToWorker(e, t, n, r) {
      const i = this.workers.get(e);
      i && await i.sendScene(t, n, r);
    }
    async sendRenderRequest(e, t, n, r) {
      const i = this.workers.get(e);
      i && await i.sendRenderRequest(t, n, r);
    }
  }
  let m = false, y = null, k = null, v = null, R = [], D = /* @__PURE__ */ new Map(), E = 0, L = 0, W = 0, x = null, _ = /* @__PURE__ */ new Map(), C = /* @__PURE__ */ new Map(), M = false, A = null;
  const O = 20, a = new ne(), h = new Z(a.canvas), c = new te(), U = new re(h, c, a.canvas), g = new ie();
  let S = 0, $ = 0, T = 0, q = performance.now();
  const ae = () => {
    const s = parseInt(a.inputDepth.value, 10) || f.defaultDepth, e = parseInt(a.inputSPP.value, 10) || f.defaultSPP;
    h.buildPipeline(s, e);
  }, N = () => {
    const { width: s, height: e } = a.getRenderConfig();
    h.updateScreenSize(s, e), c.hasWorld && (c.updateCamera(s, e), h.updateSceneUniforms(c.cameraData, 0, c.lightCount)), h.recreateBindGroup(), h.resetAccumulation(), S = 0, $ = 0;
  }, P = async (s, e = true) => {
    m = false, console.log(`Loading Scene: ${s}...`);
    let t, n;
    s === "viewer" && y && (k === "obj" ? t = y : k === "glb" && (n = new Uint8Array(y).slice(0))), await c.loadScene(s, t, n), c.printStats(), await h.loadTexturesFromWorld(c), await oe(), N(), a.updateAnimList(c.getAnimationList()), e && (m = true, a.updateRenderButton(true));
  }, oe = async () => {
    h.updateCombinedGeometry(c.vertices, c.normals, c.uvs), h.updateCombinedBVH(c.tlas, c.blas), h.updateBuffer("topology", c.mesh_topology), h.updateBuffer("instance", c.instances), h.updateBuffer("lights", c.lights), h.updateSceneUniforms(c.cameraData, 0, c.lightCount);
  }, I = () => {
    if (U.recording || (requestAnimationFrame(I), !m || !c.hasWorld)) return;
    let s = parseInt(a.inputUpdateInterval.value, 10) || 0;
    if (s > 0 && S >= s && c.update($ / (s || 1) / 60), c.hasNewData) {
      let t = false;
      t || (t = h.updateCombinedBVH(c.tlas, c.blas)), t || (t = h.updateBuffer("instance", c.instances)), c.hasNewGeometry && (t || (t = h.updateCombinedGeometry(c.vertices, c.normals, c.uvs)), t || (t = h.updateBuffer("topology", c.mesh_topology)), t || (t = h.updateBuffer("lights", c.lights)), c.hasNewGeometry = false), c.updateCamera(a.canvas.width, a.canvas.height), h.updateSceneUniforms(c.cameraData, 0, c.lightCount), t && h.recreateBindGroup(), h.resetAccumulation(), S = 0, c.hasNewData = false;
    }
    S++, T++, $++, h.compute(S), h.present();
    const e = performance.now();
    e - q >= 1e3 && (a.updateStats(T, 1e3 / T, S), T = 0, q = e);
  }, z = async (s) => {
    const e = a.sceneSelect.value, t = e !== "viewer";
    if (!t && (!y || !k)) return;
    const n = a.getRenderConfig(), r = t ? e : void 0, i = t ? "DUMMY" : y, o = t ? "obj" : k;
    n.sceneName = r, s ? (console.log(`Sending scene to specific worker: ${s}`), _.set(s, "loading"), await g.sendSceneToWorker(s, i, o, n)) : (console.log("Broadcasting scene to all workers..."), g.getWorkerIds().forEach((l) => _.set(l, "loading")), await g.broadcastScene(i, o, n));
  }, V = async (s) => {
    if (_.get(s) !== "idle") {
      console.log(`Worker ${s} is ${_.get(s)}, skipping assignment.`);
      return;
    }
    if (R.length === 0) return;
    const e = R.shift();
    e && (_.set(s, "busy"), C.set(s, e), console.log(`Assigning Job ${e.start} - ${e.start + e.count} to ${s}`), await g.sendRenderRequest(s, e.start, e.count, {
      ...x,
      fileType: "obj"
    }));
  }, le = async () => {
    const s = Array.from(D.keys()).sort((d, p) => d - p), { Muxer: e, ArrayBufferTarget: t } = await G(async () => {
      const { Muxer: d, ArrayBufferTarget: p } = await import("./webm-muxer-MLtUgOCn.js");
      return {
        Muxer: d,
        ArrayBufferTarget: p
      };
    }, []), n = new e({
      target: new t(),
      video: {
        codec: "V_VP9",
        width: x.width,
        height: x.height,
        frameRate: x.fps
      }
    });
    for (const d of s) {
      const p = D.get(d);
      if (p) for (const u of p) n.addVideoChunk(new EncodedVideoChunk({
        type: u.type,
        timestamp: u.timestamp,
        duration: u.duration,
        data: u.data
      }), {
        decoderConfig: u.decoderConfig
      });
    }
    n.finalize();
    const { buffer: r } = n.target, i = new Blob([
      r
    ], {
      type: "video/webm"
    }), o = URL.createObjectURL(i), l = document.createElement("a");
    l.href = o, l.download = `distributed_trace_${Date.now()}.webm`, document.body.appendChild(l), l.click(), document.body.removeChild(l), URL.revokeObjectURL(o), a.setStatus("Finished!");
  }, j = async (s, e, t) => {
    console.log(`[Worker] Starting Render: Frames ${s} - ${s + e}`), a.setStatus(`Remote Rendering: ${s}-${s + e}`), m = false;
    const n = {
      ...t,
      startFrame: s,
      duration: e / t.fps
    };
    try {
      a.setRecordingState(true, `Remote: ${e} f`);
      const r = await U.recordChunks(n, (i, o) => a.setRecordingState(true, `Remote: ${i}/${o}`));
      console.log("Sending Chunks back to Host..."), a.setRecordingState(true, "Uploading..."), await g.sendRenderResult(r, s), a.setRecordingState(false), a.setStatus("Idle");
    } catch (r) {
      console.error("Remote Recording Failed", r), a.setStatus("Recording Failed");
    } finally {
      m = true, requestAnimationFrame(I);
    }
  }, ce = async () => {
    if (!A) return;
    const { start: s, count: e, config: t } = A;
    A = null, await j(s, e, t);
  };
  g.onStatusChange = (s) => a.setStatus(`Status: ${s}`);
  g.onWorkerLeft = (s) => {
    console.log(`Worker Left: ${s}`), a.setStatus(`Worker Left: ${s}`), _.delete(s);
    const e = C.get(s);
    e && (console.warn(`Worker ${s} failed job ${e.start}. Re-queueing.`), R.unshift(e), C.delete(s), a.setStatus(`Re-queued Job ${e.start}`));
  };
  g.onWorkerReady = (s) => {
    console.log(`Worker ${s} is READY`), a.setStatus(`Worker ${s} Ready!`), _.set(s, "idle"), v === "host" && R.length > 0 && V(s);
  };
  g.onWorkerJoined = (s) => {
    a.setStatus(`Worker Joined: ${s}`), _.set(s, "idle"), v === "host" && R.length > 0 && z(s);
  };
  g.onRenderRequest = async (s, e, t) => {
    if (console.log(`[Worker] Received Render Request: Frames ${s} - ${s + e}`), M) {
      console.log(`[Worker] Scene loading in progress. Queueing Render Request for ${s}`), A = {
        start: s,
        count: e,
        config: t
      };
      return;
    }
    await j(s, e, t);
  };
  g.onRenderResult = async (s, e, t) => {
    console.log(`[Host] Received ${s.length} chunks for ${e} from ${t}`), D.set(e, s), E++, a.setStatus(`Distributed Progress: ${E} / ${L} jobs`), _.set(t, "idle"), C.delete(t), await V(t), E >= L && (console.log("All jobs complete. Muxing..."), a.setStatus("Muxing..."), await le());
  };
  g.onSceneReceived = async (s, e) => {
    console.log("Scene received successfully."), M = true, a.setRenderConfig(e), k = e.fileType, e.fileType, y = s, a.sceneSelect.value = e.sceneName || "viewer", await P(e.sceneName || "viewer", false), e.anim !== void 0 && (a.animSelect.value = e.anim.toString(), c.setAnimation(e.anim)), M = false, console.log("Scene Loaded. Sending WORKER_READY."), await g.sendWorkerReady(), ce();
  };
  const de = () => {
    a.onRenderStart = () => {
      m = true;
    }, a.onRenderStop = () => {
      m = false;
    }, a.onSceneSelect = (s) => P(s, false), a.onResolutionChange = N, a.onRecompile = (s, e) => {
      m = false, h.buildPipeline(s, e), h.recreateBindGroup(), h.resetAccumulation(), S = 0, m = true;
    }, a.onFileSelect = async (s) => {
      var _a;
      ((_a = s.name.split(".").pop()) == null ? void 0 : _a.toLowerCase()) === "obj" ? (y = await s.text(), k = "obj") : (y = await s.arrayBuffer(), k = "glb"), a.sceneSelect.value = "viewer", P("viewer", false);
    }, a.onAnimSelect = (s) => c.setAnimation(s), a.onRecordStart = async () => {
      if (!U.recording) if (v === "host") {
        const s = g.getWorkerIds();
        if (x = a.getRenderConfig(), W = Math.ceil(x.fps * x.duration), !confirm(`Distribute recording? (Workers: ${s.length})
Auto Scene Sync enabled.`)) return;
        R = [], D.clear(), E = 0, C.clear();
        for (let e = 0; e < W; e += O) {
          const t = Math.min(O, W - e);
          R.push({
            start: e,
            count: t
          });
        }
        L = R.length, s.forEach((e) => _.set(e, "idle")), a.setStatus(`Distributed Progress: 0 / ${L} jobs (Waiting for workers...)`), s.length > 0 ? (a.setStatus("Syncing Scene to Workers..."), await z()) : console.log("No workers yet. Waiting...");
      } else {
        m = false, a.setRecordingState(true);
        const s = a.getRenderConfig();
        try {
          const e = performance.now();
          await U.record(s, (t, n) => a.setRecordingState(true, `Rec: ${t}/${n} (${Math.round(t / n * 100)}%)`), (t) => {
            const n = document.createElement("a");
            n.href = t, n.download = `raytrace_${Date.now()}.webm`, n.click(), URL.revokeObjectURL(t);
          }), console.log(`Recording took ${performance.now() - e}[ms]`);
        } catch {
          alert("Recording failed.");
        } finally {
          a.setRecordingState(false), m = true, a.updateRenderButton(true), requestAnimationFrame(I);
        }
      }
    }, a.onConnectHost = () => {
      v === "host" ? (g.disconnect(), v = null, a.setConnectionState(null)) : (g.connect("host"), v = "host", a.setConnectionState("host"));
    }, a.onConnectWorker = () => {
      v === "worker" ? (g.disconnect(), v = null, a.setConnectionState(null)) : (g.connect("worker"), v = "worker", a.setConnectionState("worker"));
    }, a.setConnectionState(null);
  };
  async function ue() {
    try {
      await h.init(), await c.initWasm();
    } catch (s) {
      alert("Init failed: " + s);
      return;
    }
    de(), ae(), N(), P("cornell", false), requestAnimationFrame(I);
  }
  ue().catch(console.error);
})();
