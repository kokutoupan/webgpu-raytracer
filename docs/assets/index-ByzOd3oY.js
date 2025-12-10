(function(){const t=document.createElement("link").relList;if(t&&t.supports&&t.supports("modulepreload"))return;for(const i of document.querySelectorAll('link[rel="modulepreload"]'))n(i);new MutationObserver(i=>{for(const o of i)if(o.type==="childList")for(const s of o.addedNodes)s.tagName==="LINK"&&s.rel==="modulepreload"&&n(s)}).observe(document,{childList:!0,subtree:!0});function r(i){const o={};return i.integrity&&(o.integrity=i.integrity),i.referrerPolicy&&(o.referrerPolicy=i.referrerPolicy),i.crossOrigin==="use-credentials"?o.credentials="include":i.crossOrigin==="anonymous"?o.credentials="omit":o.credentials="same-origin",o}function n(i){if(i.ep)return;i.ep=!0;const o=r(i);fetch(i.href,o)}})();const Z=`// =========================================================
//   WebGPU Ray Tracer (Final Refactored Version)
// =========================================================

// --- Constants ---
const PI = 3.141592653589793;
const T_MIN = 0.001;
const T_MAX = 1e30;
const MAX_DEPTH = 10u; // ガラスなどを綺麗に見せるため少し増やす
const SPP = 1u;

// Binding Stride
const SPHERES_STRIDE = 3u;
const TRIANGLES_STRIDE = 4u;

// --- Bindings ---
@group(0) @binding(0) var outputTex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<storage, read_write> accumulateBuffer: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> frame: FrameInfo;
@group(0) @binding(3) var<uniform> camera: Camera;

// Scene Data
// 統合プリミティブバッファ (Binding 4)
// vec4 x 4 = 64 bytes stride
@group(0) @binding(4) var<storage, read> scene_primitives: array<UnifiedPrimitive>;

// Binding 5: BVH Nodes
@group(0) @binding(5) var<storage, read> bvh_nodes: array<BVHNode>;

// --- Structs ---
struct UnifiedPrimitive {
    data0: vec4<f32>, // [Tri:V0+Extra] [Sph:Center+Radius]
    data1: vec4<f32>, // [Tri:V1+Mat]   [Sph:Unused+Mat]
    data2: vec4<f32>, // [Tri:V2+ObjType] [Sph:Unused+ObjType]
    data3: vec4<f32>, // [Tri:Col+Unused] [Sph:Col+Extra]
}
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
struct BVHNode {
    min_b: vec3<f32>,
    left_first: f32,
    max_b: vec3<f32>,
    tri_count: f32
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
fn hit_sphere_raw(center: vec3<f32>, radius: f32, r: Ray, t_min: f32, t_max: f32) -> f32 {
    let oc = r.origin - center;
    let a = dot(r.direction, r.direction);
    let h = dot(r.direction, oc);
    let c = dot(oc, oc) - radius * radius;
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

fn hit_triangle_raw(v0: vec3<f32>, v1: vec3<f32>, v2: vec3<f32>, r: Ray, t_min: f32, t_max: f32) -> f32 {
    let e1 = v1 - v0;
    let e2 = v2 - v0;
    let h = cross(r.direction, e2);
    let a = dot(e1, h);
    if abs(a) < 1e-4 { return -1.0; }
    let f = 1.0 / a;
    let s = r.origin - v0;
    let u = f * dot(s, h);
    if u < 0.0 || u > 1.0 { return -1.0; }
    let q = cross(s, e1);
    let v = f * dot(r.direction, q);
    if v < 0.0 || u + v > 1.0 { return -1.0; }
    let t = f * dot(e2, q);
    if t > t_min && t < t_max { return t; }
    return -1.0;
}

fn intersect_aabb(min_b: vec3<f32>, max_b: vec3<f32>, r: Ray, inv_d: vec3<f32>, t_min: f32, t_max: f32) -> f32 {
    let t0s = (min_b - r.origin) * inv_d;
    let t1s = (max_b - r.origin) * inv_d;
    let t_small = min(t0s, t1s);
    let t_big = max(t0s, t1s);
    let tmin = max(t_min, max(t_small.x, max(t_small.y, t_small.z)));
    let tmax = min(t_max, min(t_big.x, min(t_big.y, t_big.z)));
    return select(1e30, tmin, tmin <= tmax);
}

// --- BVH Traversal ---
fn hit_bvh(r: Ray, t_min: f32, t_max: f32) -> vec4<f32> {
    var closest_t = t_max;
    var hit_idx = -1.0;
    var hit_type = 0.0;

    let inv_d = 1.0 / r.direction;

    var stack: array<u32, 32>;
    var stackptr = 0u;


    // ルートノード(0)の判定
    let root_dist = intersect_aabb(bvh_nodes[0].min_b, bvh_nodes[0].max_b, r, inv_d, t_min, closest_t);

    // ヒットした時だけ積む
    if root_dist < 1e30 {
        stack[stackptr] = 0u;
        stackptr++;
    }

    while stackptr > 0u {
        stackptr--;
        let node_idx = stack[stackptr];
        let node = bvh_nodes[node_idx];

        let count = u32(node.tri_count);
        let first = u32(node.left_first);

        if count > 0u {
            // Leaf Node
            for (var i = 0u; i < count; i++) {
                let idx = first + i;
                let prim = scene_primitives[idx];
                let obj_type = prim.data2.w;
                var t = -1.0;

                if obj_type < 1.5 { // Sphere
                    t = hit_sphere_raw(prim.data0.xyz, prim.data0.w, r, t_min, closest_t);
                } else { // Triangle
                    t = hit_triangle_raw(prim.data0.xyz, prim.data1.xyz, prim.data2.xyz, r, t_min, closest_t);
                }

                if t > 0.0 {
                    closest_t = t;
                    hit_idx = f32(idx);
                    hit_type = obj_type;
                }
            }
        } else {
            // Internal Node (Front-to-Back)
            let left_idx = u32(node.left_first);
            let right_idx = left_idx + 1u;

            let node_l = bvh_nodes[left_idx];
            let node_r = bvh_nodes[right_idx];

            let dist_l = intersect_aabb(node_l.min_b, node_l.max_b, r, inv_d, t_min, closest_t);
            let dist_r = intersect_aabb(node_r.min_b, node_r.max_b, r, inv_d, t_min, closest_t);

            let hit_l = dist_l < 1e30;
            let hit_r = dist_r < 1e30;

            if hit_l && hit_r {
                if dist_l < dist_r {
                    stack[stackptr] = right_idx; stackptr++;
                    stack[stackptr] = left_idx;  stackptr++;
                } else {
                    stack[stackptr] = left_idx;  stackptr++;
                    stack[stackptr] = right_idx; stackptr++;
                }
            } else if hit_l {
                stack[stackptr] = left_idx; stackptr++;
            } else if hit_r {
                stack[stackptr] = right_idx; stackptr++;
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
            return vec3<f32>(0.);
        }

        // Hit Data Unpacking
        let prim = scene_primitives[idx];
        var p = ray.origin + t * ray.direction;
        var normal = vec3<f32>(0.0);
        var mat = 0.0;
        var col = vec3<f32>(0.0);
        var ext = 0.0;
        var front_face = true;

        if type_id == 1u { // Sphere
            let center = prim.data0.xyz;
            let radius = prim.data0.w;
            let outward_n = (p - center) / radius;
            front_face = dot(ray.direction, outward_n) < 0.0;
            normal = select(-outward_n, outward_n, front_face);

            mat = prim.data1.w; // Data1.w
            col = prim.data3.xyz; // Data3.xyz
            ext = prim.data3.w;   // Data3.w
        } else { // Triangle
            let v0 = prim.data0.xyz;
            let v1 = prim.data1.xyz;
            let v2 = prim.data2.xyz;
            let e1 = v1 - v0;
            let e2 = v2 - v0;
            let outward_n = normalize(cross(e1, e2));
            front_face = dot(ray.direction, outward_n) < 0.0;
            normal = select(-outward_n, outward_n, front_face);

            mat = prim.data1.w;   // Data1.w
            col = prim.data3.xyz; // Data3.xyz
            ext = prim.data3.w;   // Data0.w (Triangle extra is stored here)
        }

        // Emission
        if mat > 2.5 { return select(vec3<f32>(0.0), throughput * col, front_face || ext > 0.5); }

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
    return vec3<f32>(0.);
}

// --- Entry Point ---
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(outputTex);
    if id.x >= dims.x || id.y >= dims.y { return; }

    let pixel_idx = id.y * dims.x + id.x;
    var rng = init_rng(pixel_idx, frame.frame_count);

    // ★変更: SPPループで平均色を計算
    var accum_color = vec3<f32>(0.0);

    for (var s = 0u; s < SPP; s++) {
        // レイ生成 (Jittering) をループ内で行うことでAA効果を得る
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

        accum_color += ray_color(Ray(ro, dir), &rng);
    }
    
    // このフレームでの平均色
    let col = accum_color / f32(SPP);

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
`,g={create:(e,t,r)=>({x:e,y:t,z:r}),sub:(e,t)=>({x:e.x-t.x,y:e.y-t.y,z:e.z-t.z}),add:(e,t)=>({x:e.x+t.x,y:e.y+t.y,z:e.z+t.z}),scale:(e,t)=>({x:e.x*t,y:e.y*t,z:e.z*t}),cross:(e,t)=>({x:e.y*t.z-e.z*t.y,y:e.z*t.x-e.x*t.z,z:e.x*t.y-e.y*t.x}),normalize:e=>{const t=Math.sqrt(e.x*e.x+e.y*e.y+e.z*e.z);return t===0?{x:0,y:0,z:0}:{x:e.x/t,y:e.y/t,z:e.z/t}},len:e=>Math.sqrt(e.x*e.x+e.y*e.y+e.z*e.z)},R=()=>Math.random(),N=(e,t)=>e+(t-e)*Math.random(),W=16;function q(e,t){return{x:e.x+t.x,y:e.y+t.y,z:e.z+t.z}}function O(e,t,r){return{x:e.x*t+e.z*r,y:e.y,z:-e.x*r+e.z*t}}class z{center;radius;color;matType;extra;constructor(t,r,n,i,o=0){this.center={...t},this.radius=r,this.color={...n},this.matType=i,this.extra=o}getAABB(){const{x:t,y:r,z:n}=this.center,i=this.radius;return{min:{x:t-i,y:r-i,z:n-i},max:{x:t+i,y:r+i,z:n+i},center:this.center}}pack(t,r){const n=r,{x:i,y:o,z:s}=this.center,c=this.color.x,a=this.color.y,d=this.color.z;t[n+0]=i,t[n+1]=o,t[n+2]=s,t[n+3]=this.radius,t[n+4]=0,t[n+5]=0,t[n+6]=0,t[n+7]=this.matType,t[n+8]=0,t[n+9]=0,t[n+10]=0,t[n+11]=1,t[n+12]=c,t[n+13]=a,t[n+14]=d,t[n+15]=this.extra}clone(){return new z(this.center,this.radius,this.color,this.matType,this.extra)}translate(t){this.center=q(this.center,t)}rotateY(t){const r=t*Math.PI/180,n=Math.cos(r),i=Math.sin(r);this.center=O(this.center,n,i)}}class I{v0;v1;v2;color;matType;extra;constructor(t,r,n,i,o,s=0){this.v0={...t},this.v1={...r},this.v2={...n},this.color={...i},this.matType=o,this.extra=s}getAABB(){const t={x:Math.min(this.v0.x,this.v1.x,this.v2.x),y:Math.min(this.v0.y,this.v1.y,this.v2.y),z:Math.min(this.v0.z,this.v1.z,this.v2.z)},r={x:Math.max(this.v0.x,this.v1.x,this.v2.x),y:Math.max(this.v0.y,this.v1.y,this.v2.y),z:Math.max(this.v0.z,this.v1.z,this.v2.z)},n=.001;r.x-t.x<n&&(t.x-=n,r.x+=n),r.y-t.y<n&&(t.y-=n,r.y+=n),r.z-t.z<n&&(t.z-=n,r.z+=n);const i={x:(t.x+r.x)*.5,y:(t.y+r.y)*.5,z:(t.z+r.z)*.5};return{min:t,max:r,center:i}}pack(t,r){const n=r,{x:i,y:o,z:s}=this.color;t[n+0]=this.v0.x,t[n+1]=this.v0.y,t[n+2]=this.v0.z,t[n+3]=0,t[n+4]=this.v1.x,t[n+5]=this.v1.y,t[n+6]=this.v1.z,t[n+7]=this.matType,t[n+8]=this.v2.x,t[n+9]=this.v2.y,t[n+10]=this.v2.z,t[n+11]=2,t[n+12]=i,t[n+13]=o,t[n+14]=s,t[n+15]=this.extra}clone(){return new I(this.v0,this.v1,this.v2,this.color,this.matType,this.extra)}translate(t){this.v0=q(this.v0,t),this.v1=q(this.v1,t),this.v2=q(this.v2,t)}rotateY(t){const r=t*Math.PI/180,n=Math.cos(r),i=Math.sin(r);this.v0=O(this.v0,n,i),this.v1=O(this.v1,n,i),this.v2=O(this.v2,n,i)}}const l={Lambertian:0,Metal:1,Dielectric:2,Light:3};function Q(e,t){const r=e.vfov*Math.PI/180,i=2*Math.tan(r/2)*e.focusDist,o=i*t,s=g.normalize(g.sub(e.lookfrom,e.lookat)),c=g.normalize(g.cross(e.vup,s)),a=g.cross(s,c),d=g.scale(c,o),u=g.scale(a,i),m=g.sub(g.sub(g.sub(e.lookfrom,g.scale(d,.5)),g.scale(u,.5)),g.scale(s,e.focusDist)),x=e.focusDist*Math.tan(e.defocusAngle*Math.PI/360);return new Float32Array([e.lookfrom.x,e.lookfrom.y,e.lookfrom.z,x,m.x,m.y,m.z,0,d.x,d.y,d.z,0,u.x,u.y,u.z,0,c.x,c.y,c.z,0,a.x,a.y,a.z,0])}function k(e,t,r,n=0){const i=[],o=e.x/2,s=e.y/2,c=e.z/2,a={x:-o,y:-s,z:c},d={x:o,y:-s,z:c},u={x:o,y:s,z:c},m={x:-o,y:s,z:c},x={x:-o,y:-s,z:-c},w={x:o,y:-s,z:-c},_={x:o,y:s,z:-c},h={x:-o,y:s,z:-c},f=(T,D,P,B)=>{i.push(new I(T,D,P,t,r,n)),i.push(new I(T,P,B,t,r,n))};return f(a,d,u,m),f(w,x,h,_),f(x,a,m,h),f(d,w,_,u),f(m,u,_,h),f(a,x,w,d),i}function S(e,t,r,n){for(const i of t){const o=i.clone();n!==0&&o.rotateY(n),o.translate(r),e.push(o)}}function b(e,t,r,n,i,o,s,c=0){e.push(new I(t,r,n,o,s,c)),e.push(new I(t,n,i,o,s,c))}function tt(){const e=[],t={x:.73,y:.73,z:.73},r={x:.65,y:.05,z:.05},n={x:.12,y:.45,z:.15},i={x:20,y:20,z:20},o=555,s=(_,h,f)=>({x:_/o*2-1,y:h/o*2,z:f/o*2-1}),c=(_,h,f)=>({x:_/o*2,y:h/o*2,z:f/o*2});b(e,s(0,0,0),s(555,0,0),s(555,0,555),s(0,0,555),t,l.Lambertian),b(e,s(0,555,0),s(0,555,555),s(555,555,555),s(555,555,0),t,l.Lambertian),b(e,s(0,0,555),s(555,0,555),s(555,555,555),s(0,555,555),t,l.Lambertian),b(e,s(0,0,0),s(0,555,0),s(0,555,555),s(0,0,555),n,l.Lambertian),b(e,s(555,0,0),s(555,0,555),s(555,555,555),s(555,555,0),r,l.Lambertian);const a=s(213,554,227),d=s(343,554,227),u=s(343,554,332),m=s(213,554,332);b(e,a,d,u,m,i,l.Light,0);const x=k(c(165,330,165),t,l.Lambertian);S(e,x,s(297.5,165,378.5),-15);const w=k(c(165,165,165),t,l.Lambertian);return S(e,w,s(232.5,82.5,147.5),18),{camera:{lookfrom:{x:0,y:1,z:-2.4},lookat:{x:0,y:1,z:0},vup:{x:0,y:1,z:0},vfov:60,defocusAngle:0,focusDist:2.4},primitives:e}}function et(){const e=[];e.push(new z({x:0,y:-1e3,z:0},1e3,{x:.5,y:.5,z:.5},l.Lambertian,0)),e.push(new z({x:-50,y:50,z:-50},30,{x:3,y:2.7,z:2.7},l.Light,0));for(let t=-11;t<11;t++)for(let r=-11;r<11;r++){const n=R(),i={x:t+.9*R(),y:.2,z:r+.9*R()};if(g.len(g.sub(i,{x:4,y:.2,z:0}))>.9)if(n<.8){const s={x:R()*R(),y:R()*R(),z:R()*R()};e.push(new z(i,.2,s,l.Lambertian,0))}else if(n<.95){const s={x:N(.5,1),y:N(.5,1),z:N(.5,1)};e.push(new z(i,.2,s,l.Metal,N(0,.5)))}else e.push(new z(i,.2,{x:1,y:1,z:1},l.Dielectric,1.5))}return e.push(new z({x:0,y:1,z:0},1,{x:1,y:1,z:1},l.Dielectric,1.5)),e.push(new z({x:-4,y:1,z:0},1,{x:.4,y:.2,z:.1},l.Lambertian,0)),e.push(new z({x:4,y:1,z:0},1,{x:.7,y:.6,z:.5},l.Metal,0)),{camera:{lookfrom:{x:13,y:2,z:3},lookat:{x:0,y:0,z:0},vup:{x:0,y:1,z:0},vfov:20,defocusAngle:.6,focusDist:10},primitives:e}}function nt(){const e=[],r=k({x:40,y:2,z:40},{x:.1,y:.1,z:.1},l.Metal,.05);S(e,r,{x:0,y:-1,z:0},0);const n={x:40,y:30,z:10},i={x:-4,y:8,z:4};b(e,{x:i.x,y:i.y,z:i.z},{x:i.x+2,y:i.y,z:i.z},{x:i.x+2,y:i.y,z:i.z+2},{x:i.x,y:i.y,z:i.z+2},n,l.Light);const o={x:5,y:10,z:20},s={x:4,y:6,z:-4};b(e,{x:s.x,y:s.y,z:s.z},{x:s.x+3,y:s.y,z:s.z},{x:s.x+3,y:s.y-3,z:s.z},{x:s.x,y:s.y-3,z:s.z},o,l.Light);const a=k({x:2,y:1,z:2},{x:.8,y:.6,z:.2},l.Metal,.1);S(e,a,{x:0,y:.5,z:0},0),e.push(new z({x:0,y:1.8,z:0},.8,{x:1,y:1,z:1},l.Dielectric,1.5)),e.push(new z({x:0,y:1.8,z:0},-.7,{x:1,y:1,z:1},l.Dielectric,1)),S(e,k({x:.8,y:.8,z:.8},{x:.9,y:.1,z:.1},l.Metal,.2),{x:0,y:3.2,z:0},15);const u=12,m=4;for(let h=0;h<u;h++){const f=h/u*Math.PI*2,T=Math.cos(f)*m,D=Math.sin(f)*m,P=1+Math.sin(f*3)*.5;if(h%2===0){const B={x:.8,y:.8,z:.8};e.push(new z({x:T,y:P,z:D},.4,B,l.Metal,0))}else{const B=.5+.5*Math.cos(h),A=.5+.5*Math.sin(h);S(e,k({x:.6,y:.6,z:.6},{x:B,y:A,z:.8},l.Lambertian),{x:T,y:P,z:D},h*20)}}const x={x:.2,y:.2,z:.3},w=k({x:1,y:6,z:1},x,l.Lambertian);S(e,w,{x:-4,y:3,z:-6},10);const _=k({x:1,y:4,z:1},x,l.Lambertian);return S(e,_,{x:4,y:2,z:-5},-20),{camera:{lookfrom:{x:0,y:3.5,z:9},lookat:{x:0,y:1.5,z:0},vup:{x:0,y:1,z:0},vfov:40,defocusAngle:.3,focusDist:9},primitives:e}}function rt(){const e=[],t={x:.73,y:.73,z:.73},r={x:.65,y:.05,z:.05},n={x:.12,y:.45,z:.15},i={x:20,y:20,z:20},o={x:.1,y:.1,z:10},s={x:.95,y:.95,z:.95},c=555,a=(P,B,A)=>({x:P/c*2-1,y:B/c*2,z:A/c*2-1}),d=(P,B,A)=>({x:P/c*2,y:B/c*2,z:A/c*2});b(e,a(0,0,0),a(555,0,0),a(555,0,555),a(0,0,555),t,l.Metal,.1),b(e,a(0,555,0),a(0,555,555),a(555,555,555),a(555,555,0),t,l.Lambertian),b(e,a(0,0,555),a(555,0,555),a(555,555,555),a(0,555,555),t,l.Lambertian),b(e,a(0,0,0),a(0,555,0),a(0,555,555),a(0,0,555),n,l.Lambertian),b(e,a(555,0,0),a(555,0,555),a(555,555,555),a(555,555,0),r,l.Lambertian);const u=a(213,554,227),m=a(343,554,227),x=a(343,554,332),w=a(213,554,332);b(e,u,m,x,w,i,l.Light);const _=a(366,165,383),h=k(d(165,331,165),s,l.Dielectric,1.5);S(e,h,_,15);const f=a(183,82.5,209),T=k(d(165,165,165),{x:.73,y:.73,z:.73},l.Metal,.2);S(e,T,f,-18);const D=[new z({x:0,y:0,z:0},60/c*1,o,l.Light)];return S(e,D,_,0),{camera:{lookfrom:{x:0,y:1,z:-3.9},lookat:{x:0,y:1,z:0},vup:{x:0,y:1,z:0},vfov:40,defocusAngle:0,focusDist:2.4},primitives:e}}function it(e){switch(e){case"spheres":return et();case"mixed":return nt();case"special":return rt();case"cornell":default:return tt()}}class st{nodes=[];sortedPrims=[];build(t){this.nodes=[],this.sortedPrims=[...t];const r={min:{x:0,y:0,z:0},max:{x:0,y:0,z:0},leftFirst:0,triCount:this.sortedPrims.length};this.nodes.push(r),this.updateNodeBounds(0),this.subdivide(0);const n=this.packNodes(),i=new Float32Array(this.sortedPrims.length*W);for(let o=0;o<this.sortedPrims.length;o++)this.sortedPrims[o].pack(i,o*W);return{bvhNodes:n,unifiedPrimitives:i}}updateNodeBounds(t){const r=this.nodes[t];let n=!0;for(let i=0;i<r.triCount;i++){const o=this.sortedPrims[r.leftFirst+i].getAABB();n?(r.min={...o.min},r.max={...o.max},n=!1):(r.min.x=Math.min(r.min.x,o.min.x),r.min.y=Math.min(r.min.y,o.min.y),r.min.z=Math.min(r.min.z,o.min.z),r.max.x=Math.max(r.max.x,o.max.x),r.max.y=Math.max(r.max.y,o.max.y),r.max.z=Math.max(r.max.z,o.max.z))}}subdivide(t){const r=this.nodes[t];if(r.triCount<=4)return;const n={x:r.max.x-r.min.x,y:r.max.y-r.min.y,z:r.max.z-r.min.z};let i=0;n.y>n.x&&(i=1),n.z>n.x&&n.z>n.y&&(i=2);const o=i===0?r.min.x+n.x*.5:i===1?r.min.y+n.y*.5:r.min.z+n.z*.5;let s=r.leftFirst,c=s+r.triCount-1;for(;s<=c;){const u=this.sortedPrims[s].getAABB().center;if((i===0?u.x:i===1?u.y:u.z)<o)s++;else{const x=this.sortedPrims[s];this.sortedPrims[s]=this.sortedPrims[c],this.sortedPrims[c]=x,c--}}const a=s-r.leftFirst;if(a===0||a===r.triCount)return;const d=this.nodes.length;this.nodes.push({min:{x:0,y:0,z:0},max:{x:0,y:0,z:0},leftFirst:r.leftFirst,triCount:a}),this.nodes.push({min:{x:0,y:0,z:0},max:{x:0,y:0,z:0},leftFirst:s,triCount:r.triCount-a}),r.leftFirst=d,r.triCount=0,this.updateNodeBounds(d),this.updateNodeBounds(d+1),this.subdivide(d),this.subdivide(d+1)}packNodes(){const t=new Float32Array(this.nodes.length*8);for(let r=0;r<this.nodes.length;r++){const n=this.nodes[r],i=r*8;t[i]=n.min.x,t[i+1]=n.min.y,t[i+2]=n.min.z,t[i+3]=n.leftFirst,t[i+4]=n.max.x,t[i+5]=n.max.y,t[i+6]=n.max.z,t[i+7]=n.triCount}return t}}const p=document.getElementById("gpu-canvas"),G=document.getElementById("render-btn"),ot=document.getElementById("scene-select"),J=document.getElementById("res-width"),K=document.getElementById("res-height"),at=document.getElementById("max-depth"),ct=document.getElementById("spp-frame"),lt=document.getElementById("recompile-btn"),Y=document.createElement("div");Object.assign(Y.style,{position:"fixed",bottom:"10px",left:"10px",color:"#0f0",background:"rgba(0,0,0,0.7)",padding:"8px",fontFamily:"monospace",fontSize:"14px",pointerEvents:"none",zIndex:"9999",borderRadius:"4px"});document.body.appendChild(Y);let F=0,C=!1,j=null;async function dt(){if(!navigator.gpu){alert("WebGPU not supported.");return}const e=await navigator.gpu.requestAdapter({powerPreference:"high-performance"});if(!e)throw new Error("No adapter");const t=await e.requestDevice(),r=p.getContext("webgpu");if(!r)throw new Error("No context");r.configure({device:t,format:"rgba8unorm",usage:GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT});let n,i;const o=()=>{const y=parseInt(at.value,10)||10,v=parseInt(ct.value,10)||1;console.log(`Recompiling Shader... Depth:${y}, SPP:${v}`);let M=Z;M=M.replace(/const\s+MAX_DEPTH\s*=\s*\d+u;/,`const MAX_DEPTH = ${y}u;`),M=M.replace(/const\s+SPP\s*=\s*\d+u;/,`const SPP = ${v}u;`);const E=t.createShaderModule({label:"RayTracing",code:M});n=t.createComputePipeline({label:"Main Pipeline",layout:"auto",compute:{module:E,entryPoint:"main"}}),i=n.getBindGroupLayout(0)};o();let s,c,a,d,u,m,x,w,_=0;const h=()=>{if(!a)return;const y=new Float32Array(_/4);t.queue.writeBuffer(a,0,y),F=0},f=()=>{let y=parseInt(J.value,10),v=parseInt(K.value,10);(isNaN(y)||y<1)&&(y=720),(isNaN(v)||v<1)&&(v=480),p.width=y,p.height=v,s&&s.destroy(),s=t.createTexture({size:[p.width,p.height],format:"rgba8unorm",usage:GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.COPY_SRC}),c=s.createView(),_=p.width*p.height*16,a&&a.destroy(),a=t.createBuffer({size:_,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),d||(d=t.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}))},T=()=>{!c||!a||!d||!u||!m||!x||(w=t.createBindGroup({layout:i,entries:[{binding:0,resource:c},{binding:1,resource:{buffer:a}},{binding:2,resource:{buffer:d}},{binding:3,resource:{buffer:u}},{binding:4,resource:{buffer:m}},{binding:5,resource:{buffer:x}}]}))},D=(y,v=!0)=>{console.log(`Loading Scene: ${y}...`),C=!1;const M=it(y);j=M;const L=new st().build(M.primitives);m&&m.destroy(),m=t.createBuffer({size:L.unifiedPrimitives.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),t.queue.writeBuffer(m,0,L.unifiedPrimitives),x&&x.destroy(),x=t.createBuffer({size:L.bvhNodes.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),t.queue.writeBuffer(x,0,L.bvhNodes),u||(u=t.createBuffer({size:96,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}));const H=p.width/p.height,V=Q(M.camera,H);t.queue.writeBuffer(u,0,V),T(),h(),v?(C=!0,G.textContent="Stop Rendering"):(C=!1,G.textContent="Render Start")},P=new Uint32Array(1),B={texture:null};let A=performance.now(),U=0;const $=()=>{if(requestAnimationFrame($),!C||!w)return;const y=Math.ceil(p.width/8),v=Math.ceil(p.height/8),M=performance.now();F++,U++,P[0]=F,t.queue.writeBuffer(d,0,P),B.texture=r.getCurrentTexture();const E=t.createCommandEncoder(),L=E.beginComputePass();L.setPipeline(n),L.setBindGroup(0,w),L.dispatchWorkgroups(y,v),L.end();const H={width:p.width,height:p.height,depthOrArrayLayers:1},V={texture:s};E.copyTextureToTexture(V,B,H),t.queue.submit([E.finish()]),M-A>=1e3&&(Y.textContent=`FPS: ${U} | ${(1e3/U).toFixed(2)}ms | Frame: ${F} | Res: ${p.width}x${p.height}`,U=0,A=M)};G.addEventListener("click",()=>{C=!C,G.textContent=C?"Stop Rendering":"Resume Rendering"}),ot.addEventListener("change",y=>{const v=y.target;D(v.value,!1)});const X=()=>{if(f(),j&&u){const y=p.width/p.height,v=Q(j.camera,y);t.queue.writeBuffer(u,0,v)}T(),h()};J.addEventListener("change",X),K.addEventListener("change",X),lt.addEventListener("click",()=>{C=!1,o(),T(),h(),C=!0,G.textContent="Stop Rendering"}),f(),D("cornell",!1),requestAnimationFrame($)}dt().catch(console.error);
