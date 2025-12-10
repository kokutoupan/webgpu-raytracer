(function(){const t=document.createElement("link").relList;if(t&&t.supports&&t.supports("modulepreload"))return;for(const i of document.querySelectorAll('link[rel="modulepreload"]'))r(i);new MutationObserver(i=>{for(const o of i)if(o.type==="childList")for(const s of o.addedNodes)s.tagName==="LINK"&&s.rel==="modulepreload"&&r(s)}).observe(document,{childList:!0,subtree:!0});function n(i){const o={};return i.integrity&&(o.integrity=i.integrity),i.referrerPolicy&&(o.referrerPolicy=i.referrerPolicy),i.crossOrigin==="use-credentials"?o.credentials="include":i.crossOrigin==="anonymous"?o.credentials="omit":o.credentials="same-origin",o}function r(i){if(i.ep)return;i.ep=!0;const o=n(i);fetch(i.href,o)}})();const st=`// =========================================================
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

    //[DEBUG] コストカウント
    //var steps = 0.0;


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

        //steps += 1.0;

        let count = u32(node.tri_count);
        let first = u32(node.left_first);

        if count > 0u {
            // Leaf Node
           // steps += f32(count);

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
    //return vec4<f32>(closest_t, hit_idx, hit_type, steps);
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
//        let cost = hit.w; // ★コスト取得
//
//        // --- ヒートマップ表示モード ---
//        // コスト 0〜100 を 青〜赤 にマッピングする簡易表示
//        // 100回以上の判定は真っ赤になる
//        let heat = cost / 100.0; 
//    
//        // 虹色マップっぽいグラデーション
//        let r = smoothstep(0.5, 1.0, heat);
//        let g = sin(heat * PI);
//        let b = smoothstep(0.5, 0.0, heat);
//
//        return vec3<f32>(r, g, b);

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
`,A={create:(e,t,n)=>({x:e,y:t,z:n}),sub:(e,t)=>({x:e.x-t.x,y:e.y-t.y,z:e.z-t.z}),add:(e,t)=>({x:e.x+t.x,y:e.y+t.y,z:e.z+t.z}),scale:(e,t)=>({x:e.x*t,y:e.y*t,z:e.z*t}),cross:(e,t)=>({x:e.y*t.z-e.z*t.y,y:e.z*t.x-e.x*t.z,z:e.x*t.y-e.y*t.x}),normalize:e=>{const t=Math.sqrt(e.x*e.x+e.y*e.y+e.z*e.z);return t===0?{x:0,y:0,z:0}:{x:e.x/t,y:e.y/t,z:e.z/t}},len:e=>Math.sqrt(e.x*e.x+e.y*e.y+e.z*e.z)},E=()=>Math.random(),O=(e,t)=>e+(t-e)*Math.random(),K=16;function H(e,t){return{x:e.x+t.x,y:e.y+t.y,z:e.z+t.z}}function j(e,t,n){return{x:e.x*t+e.z*n,y:e.y,z:-e.x*n+e.z*t}}class w{center;radius;color;matType;extra;constructor(t,n,r,i,o=0){this.center={...t},this.radius=n,this.color={...r},this.matType=i,this.extra=o}getAABB(){const{x:t,y:n,z:r}=this.center,i=this.radius;return{min:{x:t-i,y:n-i,z:r-i},max:{x:t+i,y:n+i,z:r+i},center:this.center}}pack(t,n){const r=n,{x:i,y:o,z:s}=this.center,c=this.color.x,a=this.color.y,x=this.color.z;t[r+0]=i,t[r+1]=o,t[r+2]=s,t[r+3]=this.radius,t[r+4]=0,t[r+5]=0,t[r+6]=0,t[r+7]=this.matType,t[r+8]=0,t[r+9]=0,t[r+10]=0,t[r+11]=1,t[r+12]=c,t[r+13]=a,t[r+14]=x,t[r+15]=this.extra}clone(){return new w(this.center,this.radius,this.color,this.matType,this.extra)}translate(t){this.center=H(this.center,t)}rotateY(t){const n=t*Math.PI/180,r=Math.cos(n),i=Math.sin(n);this.center=j(this.center,r,i)}}class G{v0;v1;v2;color;matType;extra;constructor(t,n,r,i,o,s=0){this.v0={...t},this.v1={...n},this.v2={...r},this.color={...i},this.matType=o,this.extra=s}getAABB(){const t={x:Math.min(this.v0.x,this.v1.x,this.v2.x),y:Math.min(this.v0.y,this.v1.y,this.v2.y),z:Math.min(this.v0.z,this.v1.z,this.v2.z)},n={x:Math.max(this.v0.x,this.v1.x,this.v2.x),y:Math.max(this.v0.y,this.v1.y,this.v2.y),z:Math.max(this.v0.z,this.v1.z,this.v2.z)},r=.001;n.x-t.x<r&&(t.x-=r,n.x+=r),n.y-t.y<r&&(t.y-=r,n.y+=r),n.z-t.z<r&&(t.z-=r,n.z+=r);const i={x:(t.x+n.x)*.5,y:(t.y+n.y)*.5,z:(t.z+n.z)*.5};return{min:t,max:n,center:i}}pack(t,n){const r=n,{x:i,y:o,z:s}=this.color;t[r+0]=this.v0.x,t[r+1]=this.v0.y,t[r+2]=this.v0.z,t[r+3]=0,t[r+4]=this.v1.x,t[r+5]=this.v1.y,t[r+6]=this.v1.z,t[r+7]=this.matType,t[r+8]=this.v2.x,t[r+9]=this.v2.y,t[r+10]=this.v2.z,t[r+11]=2,t[r+12]=i,t[r+13]=o,t[r+14]=s,t[r+15]=this.extra}clone(){return new G(this.v0,this.v1,this.v2,this.color,this.matType,this.extra)}translate(t){this.v0=H(this.v0,t),this.v1=H(this.v1,t),this.v2=H(this.v2,t)}rotateY(t){const n=t*Math.PI/180,r=Math.cos(n),i=Math.sin(n);this.v0=j(this.v0,r,i),this.v1=j(this.v1,r,i),this.v2=j(this.v2,r,i)}}class it{vertices=[];indices=[];constructor(t){this.parseObj(t)}parseObj(t){const n=t.split(`
`),r=[];for(const i of n){const o=i.trim().split(/\s+/),s=o[0];if(s==="v")r.push({x:parseFloat(o[1]),y:parseFloat(o[2]),z:parseFloat(o[3])});else if(s==="f"){const c=[];for(let a=1;a<o.length;a++){const x=o[a].split("/"),d=parseInt(x[0]);isNaN(d)||c.push(d-1)}for(let a=1;a<c.length-1;a++)this.indices.push(c[0]),this.indices.push(c[a]),this.indices.push(c[a+1])}}this.vertices=r}normalize(){if(this.vertices.length===0)return;let t={x:1/0,y:1/0,z:1/0},n={x:-1/0,y:-1/0,z:-1/0};for(const c of this.vertices)t.x=Math.min(t.x,c.x),t.y=Math.min(t.y,c.y),t.z=Math.min(t.z,c.z),n.x=Math.max(n.x,c.x),n.y=Math.max(n.y,c.y),n.z=Math.max(n.z,c.z);const r={x:n.x-t.x,y:n.y-t.y,z:n.z-t.z},i={x:(t.x+n.x)*.5,y:(t.y+n.y)*.5,z:(t.z+n.z)*.5},o=Math.max(r.x,Math.max(r.y,r.z)),s=o>0?2/o:1;for(let c=0;c<this.vertices.length;c++){const a=this.vertices[c];this.vertices[c]={x:(a.x-i.x)*s,y:(a.y-i.y)*s,z:(a.z-i.z)*s}}}createInstance(t,n,r,i,o,s=0){const c=[],a=this.getTransformFn(t,n,r);for(let x=0;x<this.indices.length;x+=3){const d=this.indices[x],h=this.indices[x+1],m=this.indices[x+2];if(!this.vertices[d]||!this.vertices[h]||!this.vertices[m])continue;const _=a({...this.vertices[d]}),p=a({...this.vertices[h]}),y=a({...this.vertices[m]});c.push(new G(_,p,y,i,o,s))}return c}getTransformFn(t,n,r){const i=r*Math.PI/180,o=Math.cos(i),s=Math.sin(i);return c=>{let a=c.x*n,x=c.y*n,d=c.z*n;const h=a*o+d*s,m=-a*s+d*o;return{x:h+t.x,y:x+t.y,z:m+t.z}}}}const l={Lambertian:0,Metal:1,Dielectric:2,Light:3};function Z(e,t){const n=e.vfov*Math.PI/180,i=2*Math.tan(n/2)*e.focusDist,o=i*t,s=A.normalize(A.sub(e.lookfrom,e.lookat)),c=A.normalize(A.cross(e.vup,s)),a=A.cross(s,c),x=A.scale(c,o),d=A.scale(a,i),h=A.sub(A.sub(A.sub(e.lookfrom,A.scale(x,.5)),A.scale(d,.5)),A.scale(s,e.focusDist)),m=e.focusDist*Math.tan(e.defocusAngle*Math.PI/360);return new Float32Array([e.lookfrom.x,e.lookfrom.y,e.lookfrom.z,m,h.x,h.y,h.z,0,x.x,x.y,x.z,0,d.x,d.y,d.z,0,c.x,c.y,c.z,0,a.x,a.y,a.z,0])}function R(e,t,n,r=0){const i=[],o=e.x/2,s=e.y/2,c=e.z/2,a={x:-o,y:-s,z:c},x={x:o,y:-s,z:c},d={x:o,y:s,z:c},h={x:-o,y:s,z:c},m={x:-o,y:-s,z:-c},_={x:o,y:-s,z:-c},p={x:o,y:s,z:-c},y={x:-o,y:s,z:-c},z=(g,M,P,b)=>{i.push(new G(g,M,P,t,n,r)),i.push(new G(g,P,b,t,n,r))};return z(a,x,d,h),z(_,m,y,p),z(m,a,h,y),z(x,_,p,d),z(h,d,p,y),z(a,m,_,x),i}function k(e,t,n,r){for(const i of t){const o=i.clone();r!==0&&o.rotateY(r),o.translate(n),e.push(o)}}function D(e,t,n,r,i,o,s,c=0){e.push(new G(t,n,r,o,s,c)),e.push(new G(t,r,i,o,s,c))}function ot(){const e=[],t={x:.73,y:.73,z:.73},n={x:.65,y:.05,z:.05},r={x:.12,y:.45,z:.15},i={x:20,y:20,z:20},o=555,s=(p,y,z)=>({x:p/o*2-1,y:y/o*2,z:z/o*2-1}),c=(p,y,z)=>({x:p/o*2,y:y/o*2,z:z/o*2});D(e,s(0,0,0),s(555,0,0),s(555,0,555),s(0,0,555),t,l.Lambertian),D(e,s(0,555,0),s(0,555,555),s(555,555,555),s(555,555,0),t,l.Lambertian),D(e,s(0,0,555),s(555,0,555),s(555,555,555),s(0,555,555),t,l.Lambertian),D(e,s(0,0,0),s(0,555,0),s(0,555,555),s(0,0,555),r,l.Lambertian),D(e,s(555,0,0),s(555,0,555),s(555,555,555),s(555,555,0),n,l.Lambertian);const a=s(213,554,227),x=s(343,554,227),d=s(343,554,332),h=s(213,554,332);D(e,a,x,d,h,i,l.Light,0);const m=R(c(165,330,165),t,l.Lambertian);k(e,m,s(297.5,165,378.5),-15);const _=R(c(165,165,165),t,l.Lambertian);return k(e,_,s(232.5,82.5,147.5),18),{camera:{lookfrom:{x:0,y:1,z:-2.4},lookat:{x:0,y:1,z:0},vup:{x:0,y:1,z:0},vfov:60,defocusAngle:0,focusDist:2.4},primitives:e}}function at(){const e=[];e.push(new w({x:0,y:-1e3,z:0},1e3,{x:.5,y:.5,z:.5},l.Lambertian,0)),e.push(new w({x:-50,y:50,z:-50},30,{x:3,y:2.7,z:2.7},l.Light,0));for(let t=-11;t<11;t++)for(let n=-11;n<11;n++){const r=E(),i={x:t+.9*E(),y:.2,z:n+.9*E()};if(A.len(A.sub(i,{x:4,y:.2,z:0}))>.9)if(r<.8){const s={x:E()*E(),y:E()*E(),z:E()*E()};e.push(new w(i,.2,s,l.Lambertian,0))}else if(r<.95){const s={x:O(.5,1),y:O(.5,1),z:O(.5,1)};e.push(new w(i,.2,s,l.Metal,O(0,.5)))}else e.push(new w(i,.2,{x:1,y:1,z:1},l.Dielectric,1.5))}return e.push(new w({x:0,y:1,z:0},1,{x:1,y:1,z:1},l.Dielectric,1.5)),e.push(new w({x:-4,y:1,z:0},1,{x:.4,y:.2,z:.1},l.Lambertian,0)),e.push(new w({x:4,y:1,z:0},1,{x:.7,y:.6,z:.5},l.Metal,0)),{camera:{lookfrom:{x:13,y:2,z:3},lookat:{x:0,y:0,z:0},vup:{x:0,y:1,z:0},vfov:20,defocusAngle:.6,focusDist:10},primitives:e}}function ct(){const e=[],n=R({x:40,y:2,z:40},{x:.1,y:.1,z:.1},l.Metal,.05);k(e,n,{x:0,y:-1,z:0},0);const r={x:40,y:30,z:10},i={x:-4,y:8,z:4};D(e,{x:i.x,y:i.y,z:i.z},{x:i.x+2,y:i.y,z:i.z},{x:i.x+2,y:i.y,z:i.z+2},{x:i.x,y:i.y,z:i.z+2},r,l.Light);const o={x:5,y:10,z:20},s={x:4,y:6,z:-4};D(e,{x:s.x,y:s.y,z:s.z},{x:s.x+3,y:s.y,z:s.z},{x:s.x+3,y:s.y-3,z:s.z},{x:s.x,y:s.y-3,z:s.z},o,l.Light);const a=R({x:2,y:1,z:2},{x:.8,y:.6,z:.2},l.Metal,.1);k(e,a,{x:0,y:.5,z:0},0),e.push(new w({x:0,y:1.8,z:0},.8,{x:1,y:1,z:1},l.Dielectric,1.5)),e.push(new w({x:0,y:1.8,z:0},-.7,{x:1,y:1,z:1},l.Dielectric,1)),k(e,R({x:.8,y:.8,z:.8},{x:.9,y:.1,z:.1},l.Metal,.2),{x:0,y:3.2,z:0},15);const d=12,h=4;for(let y=0;y<d;y++){const z=y/d*Math.PI*2,g=Math.cos(z)*h,M=Math.sin(z)*h,P=1+Math.sin(z*3)*.5;if(y%2===0){const b={x:.8,y:.8,z:.8};e.push(new w({x:g,y:P,z:M},.4,b,l.Metal,0))}else{const b=.5+.5*Math.cos(y),u=.5+.5*Math.sin(y);k(e,R({x:.6,y:.6,z:.6},{x:b,y:u,z:.8},l.Lambertian),{x:g,y:P,z:M},y*20)}}const m={x:.2,y:.2,z:.3},_=R({x:1,y:6,z:1},m,l.Lambertian);k(e,_,{x:-4,y:3,z:-6},10);const p=R({x:1,y:4,z:1},m,l.Lambertian);return k(e,p,{x:4,y:2,z:-5},-20),{camera:{lookfrom:{x:0,y:3.5,z:9},lookat:{x:0,y:1.5,z:0},vup:{x:0,y:1,z:0},vfov:40,defocusAngle:.3,focusDist:9},primitives:e}}function lt(){const e=[],t={x:.73,y:.73,z:.73},n={x:.65,y:.05,z:.05},r={x:.12,y:.45,z:.15},i={x:20,y:20,z:20},o={x:.1,y:.1,z:10},s={x:.95,y:.95,z:.95},c=555,a=(P,b,u)=>({x:P/c*2-1,y:b/c*2,z:u/c*2-1}),x=(P,b,u)=>({x:P/c*2,y:b/c*2,z:u/c*2});D(e,a(0,0,0),a(555,0,0),a(555,0,555),a(0,0,555),t,l.Metal,.1),D(e,a(0,555,0),a(0,555,555),a(555,555,555),a(555,555,0),t,l.Lambertian),D(e,a(0,0,555),a(555,0,555),a(555,555,555),a(0,555,555),t,l.Lambertian),D(e,a(0,0,0),a(0,555,0),a(0,555,555),a(0,0,555),r,l.Lambertian),D(e,a(555,0,0),a(555,0,555),a(555,555,555),a(555,555,0),n,l.Lambertian);const d=a(213,554,227),h=a(343,554,227),m=a(343,554,332),_=a(213,554,332);D(e,d,h,m,_,i,l.Light);const p=a(366,165,383),y=R(x(165,330,165),s,l.Dielectric,1.5);k(e,y,p,15);const z=a(183,82.5,209),g=R(x(165,165,165),{x:.73,y:.73,z:.73},l.Metal,.2);k(e,g,z,-18);const M=[new w({x:0,y:0,z:0},60/c*1,o,l.Light)];return k(e,M,p,0),{camera:{lookfrom:{x:0,y:1,z:-3.9},lookat:{x:0,y:1,z:0},vup:{x:0,y:1,z:0},vfov:40,defocusAngle:0,focusDist:2.4},primitives:e}}const ut=`
v -1 -1  1
v  1 -1  1
v -1  1  1
v  1  1  1
v -1 -1 -1
v  1 -1 -1
v -1  1 -1
v  1  1 -1
f 1 2 4 3
f 3 4 8 7
f 7 8 6 5
f 5 6 2 1
f 3 7 5 1
f 8 4 2 6
`;function dt(){const e=[],t=new it(ut);e.push(new w({x:0,y:-1e3,z:0},1e3,{x:.5,y:.5,z:.5},l.Lambertian)),e.push(...t.createInstance({x:-2,y:1,z:0},1,45,{x:.8,y:.2,z:.2},l.Metal,.2)),e.push(...t.createInstance({x:0,y:1,z:1.5},1.2,0,{x:1,y:1,z:1},l.Dielectric,1.5));for(let n=0;n<5;n++)e.push(...t.createInstance({x:2+n*.5,y:.5+n*.5,z:-n},.5,n*30,{x:.2,y:.4,z:.8},l.Lambertian));return e.push(new w({x:0,y:10,z:0},3,{x:10,y:10,z:10},l.Light)),{camera:{lookfrom:{x:0,y:3,z:6},lookat:{x:0,y:1,z:0},vup:{x:0,y:1,z:0},vfov:40,defocusAngle:0,focusDist:6},primitives:e}}function xt(e){const t=[],n={x:.2,y:.2,z:.2};return t.push(new w({x:0,y:-1e3,z:0},1e3,n,l.Lambertian)),t.push(new w({x:5,y:10,z:5},3,{x:15,y:15,z:15},l.Light)),t.push(new w({x:-5,y:5,z:5},1,{x:3,y:3,z:5},l.Light)),e?(t.push(...e.createInstance({x:0,y:1,z:0},1,0,{x:.8,y:.8,z:.8},l.Lambertian)),t.push(...e.createInstance({x:-2.5,y:1,z:-1},.8,30,{x:1,y:1,z:1},l.Dielectric,1.5)),t.push(...e.createInstance({x:2.5,y:1,z:-1},.8,-30,{x:.8,y:.6,z:.2},l.Metal,.1))):t.push(new w({x:0,y:1,z:0},1,{x:1,y:0,z:1},l.Lambertian)),{camera:{lookfrom:{x:0,y:3,z:6},lookat:{x:0,y:1,z:0},vup:{x:0,y:1,z:0},vfov:35,defocusAngle:0,focusDist:6},primitives:t}}function ht(e,t=null){switch(e){case"spheres":return at();case"mixed":return ct();case"special":return lt();case"mesh":return dt();case"viewer":return xt(t);case"cornell":default:return ot()}}function J(){return{min:{x:1/0,y:1/0,z:1/0},max:{x:-1/0,y:-1/0,z:-1/0},center:{x:0,y:0,z:0}}}function X(e,t){e.min.x=Math.min(e.min.x,t.min.x),e.min.y=Math.min(e.min.y,t.min.y),e.min.z=Math.min(e.min.z,t.min.z),e.max.x=Math.max(e.max.x,t.max.x),e.max.y=Math.max(e.max.y,t.max.y),e.max.z=Math.max(e.max.z,t.max.z)}function tt(e){const t=e.max.x-e.min.x,n=e.max.y-e.min.y,r=e.max.z-e.min.z;return t<0||n<0||r<0?0:2*(t*n+n*r+r*t)}class mt{nodes=[];sortedPrims=[];build(t){this.nodes=[],this.sortedPrims=[...t];const n={min:{x:0,y:0,z:0},max:{x:0,y:0,z:0},leftFirst:0,triCount:this.sortedPrims.length};this.nodes.push(n),this.updateNodeBounds(0),console.time("BVH Build"),this.subdivide(0),console.timeEnd("BVH Build");const r=this.packNodes(),i=new Float32Array(this.sortedPrims.length*K);for(let o=0;o<this.sortedPrims.length;o++)this.sortedPrims[o].pack(i,o*K);return{bvhNodes:r,unifiedPrimitives:i}}updateNodeBounds(t){const n=this.nodes[t];let r=!0;for(let i=0;i<n.triCount;i++){const o=this.sortedPrims[n.leftFirst+i].getAABB();r?(n.min={...o.min},n.max={...o.max},r=!1):(n.min.x=Math.min(n.min.x,o.min.x),n.min.y=Math.min(n.min.y,o.min.y),n.min.z=Math.min(n.min.z,o.min.z),n.max.x=Math.max(n.max.x,o.max.x),n.max.y=Math.max(n.max.y,o.max.y),n.max.z=Math.max(n.max.z,o.max.z))}}subdivide(t){const n=this.nodes[t];if(n.triCount<=4)return;const r={x:n.max.x-n.min.x,y:n.max.y-n.min.y,z:n.max.z-n.min.z};let i=0;r.y>r.x&&(i=1),r.z>r.x&&r.z>r.y&&(i=2);const o=i===0?n.min.x:i===1?n.min.y:n.min.z,s=i===0?r.x:i===1?r.y:r.z;if(s<1e-6)return;const c=16,a=[];for(let u=0;u<c;u++)a.push({bounds:J(),count:0});for(let u=0;u<n.triCount;u++){const T=this.sortedPrims[n.leftFirst+u],F=T.getAABB().center,C=i===0?F.x:i===1?F.y:F.z;let f=Math.floor((C-o)/s*c);f>=c&&(f=c-1),f<0&&(f=0),a[f].count++,X(a[f].bounds,T.getAABB())}const x=new Float32Array(c),d=new Float32Array(c),h=new Float32Array(c),m=new Float32Array(c);let _=J(),p=0;for(let u=0;u<c;u++)a[u].count>0&&(X(_,a[u].bounds),p+=a[u].count),x[u]=tt(_),d[u]=p;_=J(),p=0;for(let u=c-1;u>=0;u--)a[u].count>0&&(X(_,a[u].bounds),p+=a[u].count),h[u]=tt(_),m[u]=p;let y=1/0,z=-1;for(let u=0;u<c-1;u++){if(d[u]===0||m[u+1]===0)continue;const T=x[u]*d[u]+h[u+1]*m[u+1];T<y&&(y=T,z=u)}if(z===-1)return;let g=n.leftFirst,M=g+n.triCount-1;for(;g<=M;){const T=this.sortedPrims[g].getAABB().center,F=i===0?T.x:i===1?T.y:T.z;let C=Math.floor((F-o)/s*c);if(C>=c&&(C=c-1),C<0&&(C=0),C<=z)g++;else{const v=this.sortedPrims[M].getAABB().center,B=i===0?v.x:i===1?v.y:v.z;let S=Math.floor((B-o)/s*c);if(S>=c&&(S=c-1),S<0&&(S=0),S>z)M--;else{const L=this.sortedPrims[g];this.sortedPrims[g]=this.sortedPrims[M],this.sortedPrims[M]=L,g++,M--}}}const P=g-n.leftFirst;if(P===0||P===n.triCount)return;const b=this.nodes.length;this.nodes.push({min:{x:0,y:0,z:0},max:{x:0,y:0,z:0},leftFirst:n.leftFirst,triCount:P}),this.nodes.push({min:{x:0,y:0,z:0},max:{x:0,y:0,z:0},leftFirst:g,triCount:n.triCount-P}),n.leftFirst=b,n.triCount=0,this.updateNodeBounds(b),this.updateNodeBounds(b+1),this.subdivide(b),this.subdivide(b+1)}packNodes(){const t=new Float32Array(this.nodes.length*8);for(let n=0;n<this.nodes.length;n++){const r=this.nodes[n],i=n*8;t[i]=r.min.x,t[i+1]=r.min.y,t[i+2]=r.min.z,t[i+3]=r.leftFirst,t[i+4]=r.max.x,t[i+5]=r.max.y,t[i+6]=r.max.z,t[i+7]=r.triCount}return t}}const I=document.getElementById("gpu-canvas"),N=document.getElementById("render-btn"),et=document.getElementById("scene-select"),nt=document.getElementById("res-width"),rt=document.getElementById("res-height"),ft=document.getElementById("obj-file"),yt=document.getElementById("max-depth"),pt=document.getElementById("spp-frame"),zt=document.getElementById("recompile-btn"),Q=document.createElement("div");Object.assign(Q.style,{position:"fixed",bottom:"10px",left:"10px",color:"#0f0",background:"rgba(0,0,0,0.7)",padding:"8px",fontFamily:"monospace",fontSize:"14px",pointerEvents:"none",zIndex:"9999",borderRadius:"4px"});document.body.appendChild(Q);let q=0,U=!1,W=null,V=null;async function vt(){if(!navigator.gpu){alert("WebGPU not supported.");return}const e=await navigator.gpu.requestAdapter({powerPreference:"high-performance"});if(!e)throw new Error("No adapter");const t=await e.requestDevice(),n=I.getContext("webgpu");if(!n)throw new Error("No context");n.configure({device:t,format:"rgba8unorm",usage:GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT});let r,i;const o=()=>{const f=parseInt(yt.value,10)||10,v=parseInt(pt.value,10)||1;console.log(`Recompiling Shader... Depth:${f}, SPP:${v}`);let B=st;B=B.replace(/const\s+MAX_DEPTH\s*=\s*\d+u;/,`const MAX_DEPTH = ${f}u;`),B=B.replace(/const\s+SPP\s*=\s*\d+u;/,`const SPP = ${v}u;`);const S=t.createShaderModule({label:"RayTracing",code:B});r=t.createComputePipeline({label:"Main Pipeline",layout:"auto",compute:{module:S,entryPoint:"main"}}),i=r.getBindGroupLayout(0)};o();let s,c,a,x,d,h,m,_,p=0;const y=()=>{if(!a)return;const f=new Float32Array(p/4);t.queue.writeBuffer(a,0,f),q=0},z=()=>{let f=parseInt(nt.value,10),v=parseInt(rt.value,10);(isNaN(f)||f<1)&&(f=720),(isNaN(v)||v<1)&&(v=480),I.width=f,I.height=v,s&&s.destroy(),s=t.createTexture({size:[I.width,I.height],format:"rgba8unorm",usage:GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.COPY_SRC}),c=s.createView(),p=I.width*I.height*16,a&&a.destroy(),a=t.createBuffer({size:p,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),x||(x=t.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}))},g=()=>{!c||!a||!x||!d||!h||!m||(_=t.createBindGroup({layout:i,entries:[{binding:0,resource:c},{binding:1,resource:{buffer:a}},{binding:2,resource:{buffer:x}},{binding:3,resource:{buffer:d}},{binding:4,resource:{buffer:h}},{binding:5,resource:{buffer:m}}]}))},M=(f,v=!0)=>{console.log(`Loading Scene: ${f}...`),U=!1;const B=ht(f,V);W=B;const L=new mt().build(B.primitives);h&&h.destroy(),h=t.createBuffer({size:L.unifiedPrimitives.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),t.queue.writeBuffer(h,0,L.unifiedPrimitives),m&&m.destroy(),m=t.createBuffer({size:L.bvhNodes.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),t.queue.writeBuffer(m,0,L.bvhNodes),d||(d=t.createBuffer({size:96,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}));const $=I.width/I.height,Y=Z(B.camera,$);t.queue.writeBuffer(d,0,Y),g(),y(),v?(U=!0,N.textContent="Stop Rendering"):(U=!1,N.textContent="Render Start")},P=new Uint32Array(1),b={texture:null};let u=performance.now(),T=0;const F=()=>{if(requestAnimationFrame(F),!U||!_)return;const f=Math.ceil(I.width/8),v=Math.ceil(I.height/8),B=performance.now();q++,T++,P[0]=q,t.queue.writeBuffer(x,0,P),b.texture=n.getCurrentTexture();const S=t.createCommandEncoder(),L=S.beginComputePass();L.setPipeline(r),L.setBindGroup(0,_),L.dispatchWorkgroups(f,v),L.end();const $={width:I.width,height:I.height,depthOrArrayLayers:1},Y={texture:s};S.copyTextureToTexture(Y,b,$),t.queue.submit([S.finish()]),B-u>=1e3&&(Q.textContent=`FPS: ${T} | ${(1e3/T).toFixed(2)}ms | Frame: ${q} | Res: ${I.width}x${I.height}`,T=0,u=B)};N.addEventListener("click",()=>{U=!U,N.textContent=U?"Stop Rendering":"Resume Rendering"}),et.addEventListener("change",f=>{const v=f.target;M(v.value,!1)}),ft.addEventListener("change",async f=>{const v=f.target,B=v.files?.[0];if(B){console.log(`Reading ${B.name}...`);try{const S=await B.text();V=new it(S),V.normalize(),et.value="viewer",M("viewer",!0),console.log(`Loaded Mesh: ${V.vertices.length} vertices`)}catch(S){console.error("Failed to load OBJ:",S),alert("Failed to load OBJ file.")}v.value=""}});const C=()=>{if(z(),W&&d){const f=I.width/I.height,v=Z(W.camera,f);t.queue.writeBuffer(d,0,v)}g(),y()};nt.addEventListener("change",C),rt.addEventListener("change",C),zt.addEventListener("click",()=>{U=!1,o(),g(),y(),U=!0,N.textContent="Stop Rendering"}),z(),M("cornell",!1),requestAnimationFrame(F)}vt().catch(console.error);
