(function(){const t=document.createElement("link").relList;if(t&&t.supports&&t.supports("modulepreload"))return;for(const r of document.querySelectorAll('link[rel="modulepreload"]'))s(r);new MutationObserver(r=>{for(const o of r)if(o.type==="childList")for(const i of o.addedNodes)i.tagName==="LINK"&&i.rel==="modulepreload"&&s(i)}).observe(document,{childList:!0,subtree:!0});function n(r){const o={};return r.integrity&&(o.integrity=r.integrity),r.referrerPolicy&&(o.referrerPolicy=r.referrerPolicy),r.crossOrigin==="use-credentials"?o.credentials="include":r.crossOrigin==="anonymous"?o.credentials="omit":o.credentials="same-origin",o}function s(r){if(r.ep)return;r.ep=!0;const o=n(r);fetch(r.href,o)}})();const J=`// =========================================================
//   WebGPU Ray Tracer (Final Refactored Version)
// =========================================================

// --- Constants ---
const PI = 3.141592653589793;
const T_MIN = 0.001;
const T_MAX = 1e30;
const MAX_DEPTH = 10u; // ガラスなどを綺麗に見せるため少し増やす

// Binding Stride
const SPHERES_STRIDE = 3u;
const TRIANGLES_STRIDE = 4u;

// --- Bindings ---
@group(0) @binding(0) var outputTex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<storage, read_write> accumulateBuffer: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> frame: FrameInfo;
@group(0) @binding(3) var<uniform> camera: Camera;

// Scene Data
@group(0) @binding(4) var<storage, read> scene_spheres_packed: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> scene_triangles_packed: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> bvh_nodes: array<BVHNode>;
@group(0) @binding(7) var<storage, read> primitive_refs: array<vec2<u32>>;

// --- Structs ---
struct FrameInfo {
    frame_count: u32
}
struct Camera {
    origin: vec3<f32>,
    lens_radius: f32,
    lower_left_corner: vec3<f32>,
    horizontal: vec3<f32>,
    vertical: vec3<f32>,
    u: vec3<f32>,
    v: vec3<f32>
}
struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>
}
struct Sphere {
    center: vec3<f32>,
    radius: f32,
    color: vec3<f32>,
    mat_type: f32,
    extra: f32
}
struct Triangle {
    v0: vec3<f32>,
    v1: vec3<f32>,
    v2: vec3<f32>,
    color: vec3<f32>,
    mat_type: f32,
    extra: f32
}
struct BVHNode {
    min_b: vec3<f32>,
    left_first: f32,
    max_b: vec3<f32>,
    tri_count: f32
}

// --- Data Access Helpers ---
fn get_sphere(index: u32) -> Sphere {
    let i = index * SPHERES_STRIDE;
    let d1 = scene_spheres_packed[i];
    let d2 = scene_spheres_packed[i + 1u];
    let d3 = scene_spheres_packed[i + 2u];
    return Sphere(d1.xyz, d1.w, d2.xyz, d2.w, d3.x);
}

fn get_triangle(index: u32) -> Triangle {
    let i = index * TRIANGLES_STRIDE;
    let d0 = scene_triangles_packed[i];
    let d1 = scene_triangles_packed[i + 1u];
    let d2 = scene_triangles_packed[i + 2u];
    let d3 = scene_triangles_packed[i + 3u];
    return Triangle(d0.xyz, d1.xyz, d2.xyz, d3.xyz, d3.w, d0.w);
}

// --- RNG Helpers ---
fn init_rng(pixel_idx: u32, frame: u32) -> u32 {
    var seed = pixel_idx + frame * 719393u;
    seed ^= 2747636419u; seed *= 2654435769u; seed ^= (seed >> 16u);
    seed *= 2654435769u; seed ^= (seed >> 16u); seed *= 2654435769u;
    return seed;
}
fn rand_pcg(state: ptr<function, u32>) -> f32 {
    let old = *state;
    *state = old * 747796405u + 2891336453u;
    let word = ((*state) >> ((old >> 28u) + 4u)) ^ (*state);
    return f32((word >> 22u) ^ word) / 4294967295.0;
}
fn random_unit_vector(rng: ptr<function, u32>) -> vec3<f32> {
    let z = rand_pcg(rng) * 2.0 - 1.0;
    let a = rand_pcg(rng) * 2.0 * PI;
    let r = sqrt(max(0.0, 1.0 - z * z));
    return vec3<f32>(r * cos(a), r * sin(a), z);
}
fn random_in_unit_disk(rng: ptr<function, u32>) -> vec3<f32> {
    let r = sqrt(rand_pcg(rng));
    let theta = 2.0 * PI * rand_pcg(rng);
    return vec3<f32>(r * cos(theta), r * sin(theta), 0.0);
}

// --- Physics Helpers ---
fn reflectance(cosine: f32, ref_idx: f32) -> f32 {
    var r0 = (1.0 - ref_idx) / (1.0 + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0 - r0) * pow((1.0 - cosine), 5.0);
}

// --- Intersection Logic ---
fn hit_sphere_t(s: Sphere, r: Ray, t_min: f32, t_max: f32) -> f32 {
    let oc = r.origin - s.center;
    let a = dot(r.direction, r.direction);
    let h = dot(r.direction, oc);
    let c = dot(oc, oc) - s.radius * s.radius;
    let disc = h * h - a * c;
    if disc < 0.0 { return -1.0; }
    let sqrtd = sqrt(disc);
    var root = (-h - sqrtd) / a;
    if root <= t_min || t_max <= root {
        root = (-h + sqrtd) / a;
        if root <= t_min || t_max <= root { return -1.0; }
    }
    return root;
}

fn hit_triangle_t(tri: Triangle, r: Ray, t_min: f32, t_max: f32) -> f32 {
    let e1 = tri.v1 - tri.v0;
    let e2 = tri.v2 - tri.v0;
    let h = cross(r.direction, e2);
    let a = dot(e1, h);
    if abs(a) < 1e-4 { return -1.0; }
    let f = 1.0 / a;
    let s = r.origin - tri.v0;
    let u = f * dot(s, h);
    if u < 0.0 || u > 1.0 { return -1.0; }
    let q = cross(s, e1);
    let v = f * dot(r.direction, q);
    if v < 0.0 || u + v > 1.0 { return -1.0; }
    let t = f * dot(e2, q);
    if t > t_min && t < t_max { return t; }
    return -1.0;
}

fn hit_aabb(min_b: vec3<f32>, max_b: vec3<f32>, r: Ray, inv_d: vec3<f32>, t_min: f32, t_max: f32) -> bool {
    let t0s = (min_b - r.origin) * inv_d;
    let t1s = (max_b - r.origin) * inv_d;
    let t_small = min(t0s, t1s);
    let t_big = max(t0s, t1s);
    let tmin = max(t_min, max(t_small.x, max(t_small.y, t_small.z)));
    let tmax = min(t_max, min(t_big.x, min(t_big.y, t_big.z)));
    return tmin <= tmax;
}

// --- BVH Traversal ---
fn hit_bvh(r: Ray, t_min: f32, t_max: f32) -> vec4<f32> {
    var closest_t = t_max;
    var hit_idx = -1.0;
    var hit_type = 0.0; // 0:None, 1:Sphere, 2:Triangle

    // Precompute inverse direction (Safe for zero components)
    let inv_d = vec3<f32>(
        select(1.0 / r.direction.x, 1e30, abs(r.direction.x) < 1e-20),
        select(1.0 / r.direction.y, 1e30, abs(r.direction.y) < 1e-20),
        select(1.0 / r.direction.z, 1e30, abs(r.direction.z) < 1e-20)
    );

    var stack: array<u32, 32>;
    var stackptr = 0u;
    stack[stackptr] = 0u; stackptr++;

    while stackptr > 0u {
        stackptr--;
        let node_idx = stack[stackptr];
        let node = bvh_nodes[node_idx];

        if hit_aabb(node.min_b, node.max_b, r, inv_d, t_min, closest_t) {
            let count = u32(node.tri_count);
            let first = u32(node.left_first);

            if count > 0u {
                // Leaf: Check primitives
                for (var i = 0u; i < count; i++) {
                    let pref = primitive_refs[first + i];
                    let type_id = pref.x;
                    let obj_idx = pref.y;
                    var t = -1.0;

                    if type_id == 0u { // Sphere
                        t = hit_sphere_t(get_sphere(obj_idx), r, t_min, closest_t);
                    } else { // Triangle
                        t = hit_triangle_t(get_triangle(obj_idx), r, t_min, closest_t);
                    }

                    if t > 0.0 {
                        closest_t = t;
                        hit_idx = f32(obj_idx);
                        hit_type = f32(type_id + 1u);
                    }
                }
            } else {
                // Internal: Push children
                stack[stackptr] = first; stackptr++;
                stack[stackptr] = first + 1u; stackptr++;
            }
        }
    }
    return vec4<f32>(closest_t, hit_idx, hit_type, 0.0);
}

// --- Main Tracer ---
fn ray_color(r_in: Ray, rng: ptr<function, u32>) -> vec3<f32> {
    var ray = r_in;
    var throughput = vec3<f32>(1.0);

    for (var depth = 0u; depth < MAX_DEPTH; depth++) {
        // BVH Traversal
        let hit = hit_bvh(ray, T_MIN, T_MAX);
        let t = hit.x;
        let idx = u32(hit.y);
        let type_id = u32(hit.z);

        if type_id == 0u {
            // Miss: Black Background
            return vec3<f32>(0.0);
        }

        // Shading Variables
        var p = ray.origin + t * ray.direction;
        var normal = vec3<f32>(0.0);
        var mat = 0.0;
        var col = vec3<f32>(0.0);
        var ext = 0.0;
        var front_face = true;

        if type_id == 1u { // Sphere
            let s = get_sphere(idx);
            let out_n = (p - s.center) / s.radius;
            front_face = dot(ray.direction, out_n) < 0.0;
            normal = select(-out_n, out_n, front_face);
            mat = s.mat_type; col = s.color; ext = s.extra;
        } else { // Triangle
            let tri = get_triangle(idx);
            let e1 = tri.v1 - tri.v0;
            let e2 = tri.v2 - tri.v0;
            let out_n = normalize(cross(e1, e2));
            front_face = dot(ray.direction, out_n) < 0.0;
            normal = select(-out_n, out_n, front_face);
            mat = tri.mat_type; col = tri.color; ext = tri.extra;
        }

        // Emission
        if mat > 2.5 { return throughput * col; }

        var scat = vec3<f32>(0.0);

        // Material Scatter
        if mat < 0.5 { // Lambertian
            scat = normal + random_unit_vector(rng);
            if length(scat) < 1e-6 { scat = normal; }
        
        } else if mat < 1.5 { // Metal
            scat = reflect(ray.direction, normal) + ext * random_unit_vector(rng);
            // ★重要: 内側への反射は吸収（黒）
            if dot(scat, normal) <= 0.0 { return vec3<f32>(0.0); }
        
        } else { // Dielectric
            let ratio = select(ext, 1.0 / ext, front_face);
            let unit = normalize(ray.direction);
            let cos_t = min(dot(-unit, normal), 1.0);
            let sin_t = sqrt(1.0 - cos_t * cos_t);
            let cannot = ratio * sin_t > 1.0;
            if cannot || reflectance(cos_t, ratio) > rand_pcg(rng) {
                scat = reflect(unit, normal);
            } else {
                // Refract manual
                let perp = ratio * (unit + cos_t * normal);
                let para = -sqrt(abs(1.0 - dot(perp, perp))) * normal;
                scat = perp + para;
            }
        }

        // ★重要: Shadow Acne 対策 (少し浮かせて再発射)
        let next_origin = p + scat * 1e-4;
        ray = Ray(next_origin, scat);
        throughput *= col;

        // Russian Roulette
        if depth > 2u {
            let p_rr = max(throughput.r, max(throughput.g, throughput.b));
            if rand_pcg(rng) > p_rr { break; }
            throughput /= p_rr;
        }
    }
    return vec3<f32>(0.0);
}

// --- Entry Point ---
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(outputTex);
    if id.x >= dims.x || id.y >= dims.y { return; }

    let pixel_idx = id.y * dims.x + id.x;
    var rng = init_rng(pixel_idx, frame.frame_count);

    // Jittered Camera Ray
    var ro = camera.origin;
    var offset = vec3<f32>(0.0);
    if camera.lens_radius > 0.0 {
        let rd = camera.lens_radius * random_in_unit_disk(&rng);
        offset = camera.u * rd.x + camera.v * rd.y;
        ro += offset;
    }
    let u = (f32(id.x) + rand_pcg(&rng)) / f32(dims.x);
    let v = 1.0 - (f32(id.y) + rand_pcg(&rng)) / f32(dims.y);
    let dir = camera.lower_left_corner + u * camera.horizontal + v * camera.vertical - camera.origin - offset;
    
    // Trace
    let col = ray_color(Ray(ro, dir), &rng);

    // Accumulate
    var acc = vec4<f32>(0.0);
    if frame.frame_count > 1u { acc = accumulateBuffer[pixel_idx]; }
    let new_acc = acc + vec4<f32>(col, 1.0);
    accumulateBuffer[pixel_idx] = new_acc;

    // Display
    var final_col = new_acc.rgb / f32(frame.frame_count);
    final_col = sqrt(clamp(final_col, vec3<f32>(0.0), vec3<f32>(1.0)));
    textureStore(outputTex, vec2<i32>(id.xy), vec4<f32>(final_col, 1.0));
}
`,p={create:(e,t,n)=>({x:e,y:t,z:n}),sub:(e,t)=>({x:e.x-t.x,y:e.y-t.y,z:e.z-t.z}),add:(e,t)=>({x:e.x+t.x,y:e.y+t.y,z:e.z+t.z}),scale:(e,t)=>({x:e.x*t,y:e.y*t,z:e.z*t}),cross:(e,t)=>({x:e.y*t.z-e.z*t.y,y:e.z*t.x-e.x*t.z,z:e.x*t.y-e.y*t.x}),normalize:e=>{const t=Math.sqrt(e.x*e.x+e.y*e.y+e.z*e.z);return t===0?{x:0,y:0,z:0}:{x:e.x/t,y:e.y/t,z:e.z/t}},len:e=>Math.sqrt(e.x*e.x+e.y*e.y+e.z*e.z)},C=()=>Math.random(),O=(e,t)=>e+(t-e)*Math.random(),m={Lambertian:0,Metal:1,Dielectric:2,Light:3};function K(e,t){const n=e.vfov*Math.PI/180,r=2*Math.tan(n/2)*e.focusDist,o=r*t,i=p.normalize(p.sub(e.lookfrom,e.lookat)),d=p.normalize(p.cross(e.vup,i)),a=p.cross(i,d),c=p.scale(d,o),l=p.scale(a,r),u=p.sub(p.sub(p.sub(e.lookfrom,p.scale(c,.5)),p.scale(l,.5)),p.scale(i,e.focusDist)),f=e.focusDist*Math.tan(e.defocusAngle*Math.PI/360);return new Float32Array([e.lookfrom.x,e.lookfrom.y,e.lookfrom.z,f,u.x,u.y,u.z,0,c.x,c.y,c.z,0,l.x,l.y,l.z,0,d.x,d.y,d.z,0,a.x,a.y,a.z,0])}function j(e,t,n,s,r,o=0){return[e.x,e.y,e.z,o,t.x,t.y,t.z,0,n.x,n.y,n.z,0,s.x,s.y,s.z,r]}function b(e,t,n,s,r,o,i,d=0){e.push(...j(t,n,s,o,i,d)),e.push(...j(t,s,r,o,i,d))}function T(e,t,n,s,r){return[e.x,e.y,e.z,t,n.x,n.y,n.z,s,r,0,0,0]}function Q(){return new Float32Array([0,-9999,0,0,0,0,0,0,0,0,0,0])}function Z(){return new Float32Array([0,-9999,0,0,0,-9998,0,0,0,-9997,0,0,0,0,0,0])}function k(e,t,n,s,r,o,i=0){const d=s*Math.PI/180,a=Math.cos(d),c=Math.sin(d),l=M=>({x:M.x*a+M.z*c,y:M.y,z:-M.x*c+M.z*a}),u=n.x/2,f=n.y/2,x=n.z/2,_={x:-u,y:-f,z:x},R={x:u,y:-f,z:x},S={x:u,y:f,z:x},D={x:-u,y:f,z:x},A={x:-u,y:-f,z:-x},g={x:u,y:-f,z:-x},z={x:u,y:f,z:-x},v={x:-u,y:f,z:-x},P=M=>{const F=l(M);return{x:F.x+t.x,y:F.y+t.y,z:F.z+t.z}},L=P(_),E=P(R),w=P(S),B=P(D),y=P(A),q=P(g),U=P(z),I=P(v);b(e,L,E,w,B,r,o,i),b(e,q,y,I,U,r,o,i),b(e,y,L,B,I,r,o,i),b(e,E,q,U,w,r,o,i),b(e,B,w,U,I,r,o,i),b(e,L,y,q,E,r,o,i)}function ee(){const e=[],t={x:.73,y:.73,z:.73},n={x:.65,y:.05,z:.05},s={x:.12,y:.45,z:.15},r={x:50,y:50,z:50},o=555,i=(f,x,_)=>({x:f/o*2-1,y:x/o*2,z:_/o*2-1}),d=(f,x,_)=>({x:f/o*2,y:x/o*2,z:_/o*2});b(e,i(0,0,0),i(555,0,0),i(555,0,555),i(0,0,555),t,m.Lambertian),b(e,i(0,555,0),i(0,555,555),i(555,555,555),i(555,555,0),t,m.Lambertian),b(e,i(0,0,555),i(555,0,555),i(555,555,555),i(0,555,555),t,m.Lambertian),b(e,i(0,0,0),i(0,555,0),i(0,555,555),i(0,0,555),s,m.Lambertian),b(e,i(555,0,0),i(555,0,555),i(555,555,555),i(555,555,0),n,m.Lambertian);const a=i(213,554,227),c=i(343,554,227),l=i(343,554,332),u=i(213,554,332);return b(e,a,u,l,c,r,m.Light),k(e,i(265+82.5-50,165,296+82.5),d(165,330,165),-15,t,m.Lambertian),k(e,i(130+82.5+20,82.5,65+82.5),d(165,165,165),18,t,m.Lambertian),{camera:{lookfrom:{x:0,y:1,z:-2.4},lookat:{x:0,y:1,z:0},vup:{x:0,y:1,z:0},vfov:60,defocusAngle:0,focusDist:2.4},spheres:Q(),triangles:new Float32Array(e)}}function te(){const e=[];e.push(...T({x:0,y:-1e3,z:0},1e3,{x:.5,y:.5,z:.5},m.Lambertian,0)),e.push(...T({x:-50,y:50,z:-50},30,{x:10,y:10,z:10},m.Light,0));for(let t=-11;t<11;t++)for(let n=-11;n<11;n++){const s=C(),r={x:t+.9*C(),y:.2,z:n+.9*C()};if(p.len(p.sub(r,{x:4,y:.2,z:0}))>.9)if(s<.8){const i={x:C()*C(),y:C()*C(),z:C()*C()};e.push(...T(r,.2,i,m.Lambertian,0))}else if(s<.95){const i={x:O(.5,1),y:O(.5,1),z:O(.5,1)};e.push(...T(r,.2,i,m.Metal,O(0,.5)))}else e.push(...T(r,.2,{x:1,y:1,z:1},m.Dielectric,1.5))}return e.push(...T({x:0,y:1,z:0},1,{x:1,y:1,z:1},m.Dielectric,1.5)),e.push(...T({x:-4,y:1,z:0},1,{x:.4,y:.2,z:.1},m.Lambertian,0)),e.push(...T({x:4,y:1,z:0},1,{x:.7,y:.6,z:.5},m.Metal,0)),{camera:{lookfrom:{x:13,y:2,z:3},lookat:{x:0,y:0,z:0},vup:{x:0,y:1,z:0},vfov:20,defocusAngle:.6,focusDist:10},spheres:new Float32Array(e),triangles:Z()}}function ne(){const e=[],t=[];k(e,{x:0,y:-1,z:0},{x:40,y:2,z:40},0,{x:.1,y:.1,z:.1},m.Metal,.05);const s={x:40,y:30,z:10},r={x:-4,y:8,z:4};b(e,{x:r.x,y:r.y,z:r.z},{x:r.x+2,y:r.y,z:r.z},{x:r.x+2,y:r.y,z:r.z+2},{x:r.x,y:r.y,z:r.z+2},s,m.Light);const o={x:5,y:10,z:20},i={x:4,y:6,z:-4};b(e,{x:i.x,y:i.y,z:i.z},{x:i.x+3,y:i.y,z:i.z},{x:i.x+3,y:i.y-3,z:i.z},{x:i.x,y:i.y-3,z:i.z},o,m.Light),k(e,{x:0,y:.5,z:0},{x:2,y:1,z:2},45,{x:.8,y:.6,z:.2},m.Metal,.1),t.push(...T({x:0,y:1.8,z:0},.8,{x:1,y:1,z:1},m.Dielectric,1.5)),t.push(...T({x:0,y:1.8,z:0},-.7,{x:1,y:1,z:1},m.Dielectric,1)),k(e,{x:0,y:3,z:0},{x:.8,y:.8,z:.8},15,{x:.9,y:.1,z:.1},m.Metal,.2);const c=12,l=4;for(let u=0;u<c;u++){const f=u/c*Math.PI*2,x=Math.cos(f)*l,_=Math.sin(f)*l,R=1+Math.sin(f*3)*.5;if(u%2===0){const S={x:.8,y:.8,z:.8};t.push(...T({x,y:R,z:_},.4,S,m.Metal,0))}else{const S=.5+.5*Math.cos(u),D=.5+.5*Math.sin(u);k(e,{x,y:R,z:_},{x:.6,y:.6,z:.6},u*20,{x:S,y:D,z:.8},m.Lambertian)}}return k(e,{x:-4,y:3,z:-6},{x:1,y:6,z:1},10,{x:.2,y:.2,z:.3},m.Lambertian),k(e,{x:4,y:2,z:-5},{x:1,y:4,z:1},-20,{x:.2,y:.2,z:.3},m.Lambertian),{camera:{lookfrom:{x:0,y:3.5,z:9},lookat:{x:0,y:1.5,z:0},vup:{x:0,y:1,z:0},vfov:40,defocusAngle:.3,focusDist:9},spheres:new Float32Array(t),triangles:new Float32Array(e)}}function re(e){switch(e){case"spheres":return te();case"mixed":return ne();case"cornell":default:return ee()}}class ie{nodes=[];sortedPrims=[];build(t,n){this.nodes=[],this.sortedPrims=[];const s=12;for(let a=0;a<t.length/s;a++){const c=a*s,l=t[c+3];if(l<=0)continue;const u=t[c],f=t[c+1],x=t[c+2];this.sortedPrims.push({min:{x:u-l,y:f-l,z:x-l},max:{x:u+l,y:f+l,z:x+l},center:{x:u,y:f,z:x},type:0,originalIndex:a})}const r=16;for(let a=0;a<n.length/r;a++){const c=a*r,l=n[c],u=n[c+1],f=n[c+2],x=n[c+4],_=n[c+5],R=n[c+6],S=n[c+8],D=n[c+9],A=n[c+10],g={x:Math.min(l,x,S),y:Math.min(u,_,D),z:Math.min(f,R,A)},z={x:Math.max(l,x,S),y:Math.max(u,_,D),z:Math.max(f,R,A)},v=.001;z.x-g.x<v&&(g.x-=v,z.x+=v),z.y-g.y<v&&(g.y-=v,z.y+=v),z.z-g.z<v&&(g.z-=v,z.z+=v);const P={x:(g.x+z.x)*.5,y:(g.y+z.y)*.5,z:(g.z+z.z)*.5};this.sortedPrims.push({min:g,max:z,center:P,type:1,originalIndex:a})}const o={min:{x:0,y:0,z:0},max:{x:0,y:0,z:0},leftFirst:0,triCount:this.sortedPrims.length};this.nodes.push(o),this.updateNodeBounds(0),this.subdivide(0);const i=this.packNodes(),d=new Uint32Array(this.sortedPrims.length*2);for(let a=0;a<this.sortedPrims.length;a++)d[a*2]=this.sortedPrims[a].type,d[a*2+1]=this.sortedPrims[a].originalIndex;return{bvhNodes:i,primitiveRefs:d}}updateNodeBounds(t){const n=this.nodes[t];let s=!0;for(let r=0;r<n.triCount;r++){const o=this.sortedPrims[n.leftFirst+r];s?(n.min={...o.min},n.max={...o.max},s=!1):(n.min.x=Math.min(n.min.x,o.min.x),n.min.y=Math.min(n.min.y,o.min.y),n.min.z=Math.min(n.min.z,o.min.z),n.max.x=Math.max(n.max.x,o.max.x),n.max.y=Math.max(n.max.y,o.max.y),n.max.z=Math.max(n.max.z,o.max.z))}}subdivide(t){const n=this.nodes[t];if(n.triCount<=4)return;const s={x:n.max.x-n.min.x,y:n.max.y-n.min.y,z:n.max.z-n.min.z};let r=0;s.y>s.x&&(r=1),s.z>s.x&&s.z>s.y&&(r=2);const o=r===0?n.min.x+s.x*.5:r===1?n.min.y+s.y*.5:n.min.z+s.z*.5;let i=n.leftFirst,d=i+n.triCount-1;for(;i<=d;){const l=this.sortedPrims[i];if((r===0?l.center.x:r===1?l.center.y:l.center.z)<o)i++;else{const f=this.sortedPrims[i];this.sortedPrims[i]=this.sortedPrims[d],this.sortedPrims[d]=f,d--}}const a=i-n.leftFirst;if(a===0||a===n.triCount)return;const c=this.nodes.length;this.nodes.push({min:{x:0,y:0,z:0},max:{x:0,y:0,z:0},leftFirst:n.leftFirst,triCount:a}),this.nodes.push({min:{x:0,y:0,z:0},max:{x:0,y:0,z:0},leftFirst:i,triCount:n.triCount-a}),n.leftFirst=c,n.triCount=0,this.updateNodeBounds(c),this.updateNodeBounds(c+1),this.subdivide(c),this.subdivide(c+1)}packNodes(){const t=new Float32Array(this.nodes.length*8);for(let n=0;n<this.nodes.length;n++){const s=this.nodes[n],r=n*8;t[r]=s.min.x,t[r+1]=s.min.y,t[r+2]=s.min.z,t[r+3]=s.leftFirst,t[r+4]=s.max.x,t[r+5]=s.max.y,t[r+6]=s.max.z,t[r+7]=s.triCount}return t}}const W=1,h=document.getElementById("gpu-canvas"),N=document.getElementById("render-btn"),oe=document.getElementById("scene-select"),Y=document.createElement("div");Object.assign(Y.style,{position:"fixed",top:"10px",left:"10px",color:"#0f0",background:"rgba(0,0,0,0.7)",padding:"8px",fontFamily:"monospace",fontSize:"14px",pointerEvents:"none",zIndex:"9999"});document.body.appendChild(Y);let H=0,G=!1,X=null;async function se(){if(!navigator.gpu){alert("WebGPU not supported.");return}const e=await navigator.gpu.requestAdapter({powerPreference:"high-performance"});if(!e)throw new Error("No adapter");const t=await e.requestDevice(),n=h.getContext("webgpu");if(!n)throw new Error("No context");(()=>{h.width=h.clientWidth*W,h.height=h.clientHeight*W})(),n.configure({device:t,format:"rgba8unorm",usage:GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT});const r=t.createTexture({size:[h.width,h.height],format:"rgba8unorm",usage:GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.COPY_SRC}),o=r.createView(),i=h.width*h.height*16,d=t.createBuffer({size:i,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),a=t.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),c=t.createShaderModule({label:"RayTracing",code:J}),l=t.createComputePipeline({label:"Main Pipeline",layout:"auto",compute:{module:c,entryPoint:"main"}}),u=l.getBindGroupLayout(0);let f,x;const _=(w,B=!0)=>{console.log(`Loading Scene: ${w}...`),G=!1;const y=re(w);X=y,console.log(X);const U=new ie().build(y.spheres,y.triangles),I=t.createBuffer({size:y.spheres.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});t.queue.writeBuffer(I,0,y.spheres);const M=t.createBuffer({size:y.triangles.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});t.queue.writeBuffer(M,0,y.triangles);const F=t.createBuffer({size:U.bvhNodes.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});t.queue.writeBuffer(F,0,U.bvhNodes);const V=t.createBuffer({size:U.primitiveRefs.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});t.queue.writeBuffer(V,0,U.primitiveRefs),x||(x=t.createBuffer({size:96,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}));const $=K(y.camera,h.width/h.height);t.queue.writeBuffer(x,0,$),f=t.createBindGroup({layout:u,entries:[{binding:0,resource:o},{binding:1,resource:{buffer:d}},{binding:2,resource:{buffer:a}},{binding:3,resource:{buffer:x}},{binding:4,resource:{buffer:I}},{binding:5,resource:{buffer:M}},{binding:6,resource:{buffer:F}},{binding:7,resource:{buffer:V}}]}),R(),B?(G=!0,N.textContent="Stop Rendering"):(G=!1,N.textContent="Render Start"),console.log("Scene Loaded.")},R=()=>{const w=new Float32Array(i/4);t.queue.writeBuffer(d,0,w),H=0},S=new Uint32Array(1),D=Math.ceil(h.width/8),A=Math.ceil(h.height/8),g={width:h.width,height:h.height,depthOrArrayLayers:1},z={texture:r},v={texture:null};let P=performance.now(),L=0;const E=()=>{if(!G||!f){requestAnimationFrame(E);return}const w=performance.now();H++,L++,S[0]=H,t.queue.writeBuffer(a,0,S),v.texture=n.getCurrentTexture();const B=t.createCommandEncoder(),y=B.beginComputePass();y.setPipeline(l),y.setBindGroup(0,f),y.dispatchWorkgroups(D,A),y.end(),B.copyTextureToTexture(z,v,g),t.queue.submit([B.finish()]),w-P>=1e3&&(Y.textContent=`FPS: ${L} | ${(1e3/L).toFixed(2)}ms | Frame: ${H}`,L=0,P=w),requestAnimationFrame(E)};N.addEventListener("click",()=>{G?(G=!1,N.textContent="Resume Rendering"):(G=!0,N.textContent="Stop Rendering")}),oe.addEventListener("change",w=>{const B=w.target;_(B.value,!1)}),_("cornell",!1),requestAnimationFrame(E)}se().catch(console.error);
