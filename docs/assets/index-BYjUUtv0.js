(function(){const e=document.createElement("link").relList;if(e&&e.supports&&e.supports("modulepreload"))return;for(const r of document.querySelectorAll('link[rel="modulepreload"]'))s(r);new MutationObserver(r=>{for(const o of r)if(o.type==="childList")for(const a of o.addedNodes)a.tagName==="LINK"&&a.rel==="modulepreload"&&s(a)}).observe(document,{childList:!0,subtree:!0});function n(r){const o={};return r.integrity&&(o.integrity=r.integrity),r.referrerPolicy&&(o.referrerPolicy=r.referrerPolicy),r.crossOrigin==="use-credentials"?o.credentials="include":r.crossOrigin==="anonymous"?o.credentials="omit":o.credentials="same-origin",o}function s(r){if(r.ep)return;r.ep=!0;const o=n(r);fetch(r.href,o)}})();const j=`// =========================================================
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
`,z={create:(t,e,n)=>({x:t,y:e,z:n}),sub:(t,e)=>({x:t.x-e.x,y:t.y-e.y,z:t.z-e.z}),add:(t,e)=>({x:t.x+e.x,y:t.y+e.y,z:t.z+e.z}),scale:(t,e)=>({x:t.x*e,y:t.y*e,z:t.z*e}),cross:(t,e)=>({x:t.y*e.z-t.z*e.y,y:t.z*e.x-t.x*e.z,z:t.x*e.y-t.y*e.x}),normalize:t=>{const e=Math.sqrt(t.x*t.x+t.y*t.y+t.z*t.z);return e===0?{x:0,y:0,z:0}:{x:t.x/e,y:t.y/e,z:t.z/e}},len:t=>Math.sqrt(t.x*t.x+t.y*t.y+t.z*t.z)},C={Lambertian:0,Metal:1,Dielectric:2};function W(t,e,n,s,r,o,a){const i=s*Math.PI/180,u=2*Math.tan(i/2)*a,m=u*r,l=z.normalize(z.sub(t,e)),f=z.normalize(z.cross(n,l)),d=z.cross(l,f),p=t,x=z.scale(f,m),b=z.scale(d,u),B=z.sub(z.sub(z.sub(p,z.scale(x,.5)),z.scale(b,.5)),z.scale(l,a)),T=a*Math.tan(o*Math.PI/360);return new Float32Array([p.x,p.y,p.z,T,B.x,B.y,B.z,0,x.x,x.y,x.z,0,b.x,b.y,b.z,0,f.x,f.y,f.z,0,d.x,d.y,d.z,0])}function V(t,e,n,s,r,o=0){return[t.x,t.y,t.z,o,e.x,e.y,e.z,0,n.x,n.y,n.z,0,s.x,s.y,s.z,r]}function P(t,e,n,s,r,o,a,i=0){t.push(...V(e,n,s,o,a,i)),t.push(...V(e,s,r,o,a,i))}function Y(t,e,n,s,r,o,a=0){const i=s*Math.PI/180,c=Math.cos(i),u=Math.sin(i),m=w=>({x:w.x*c+w.z*u,y:w.y,z:-w.x*u+w.z*c}),l=n.x/2,f=n.y/2,d=n.z/2,p={x:-l,y:-f,z:d},x={x:l,y:-f,z:d},b={x:l,y:f,z:d},B={x:-l,y:f,z:d},T={x:-l,y:-f,z:-d},_={x:l,y:-f,z:-d},g={x:l,y:f,z:-d},y={x:-l,y:f,z:-d},v=w=>{const G=m(w);return{x:G.x+e.x,y:G.y+e.y,z:G.z+e.z}},D=v(p),k=v(x),A=v(b),I=v(B),U=v(T),R=v(_),S=v(g),M=v(y);P(t,D,k,A,I,r,o,a),P(t,R,U,M,S,r,o,a),P(t,U,D,I,M,r,o,a),P(t,k,R,S,A,r,o,a),P(t,I,A,S,M,r,o,a),P(t,D,U,R,k,r,o,a)}function X(){const t=[],e={x:.73,y:.73,z:.73},n={x:.65,y:.05,z:.05},s={x:.12,y:.45,z:.15},r={x:15,y:15,z:15},o={x:1,y:1,z:1},a=555,i=(d,p,x)=>({x:d/a*2-1,y:p/a*2,z:x/a*2-1}),c=(d,p,x)=>({x:d/a*2,y:p/a*2,z:x/a*2});P(t,i(0,0,0),i(555,0,0),i(555,0,555),i(0,0,555),e,C.Lambertian),P(t,i(0,555,0),i(0,555,555),i(555,555,555),i(555,555,0),e,C.Lambertian),P(t,i(0,0,555),i(555,0,555),i(555,555,555),i(0,555,555),e,C.Lambertian),P(t,i(0,0,0),i(0,555,0),i(0,555,555),i(0,0,555),s,C.Lambertian),P(t,i(555,0,0),i(555,0,555),i(555,555,555),i(555,555,0),n,C.Lambertian);const u=i(213,554,227),m=i(343,554,227),l=i(343,554,332),f=i(213,554,332);return P(t,u,f,l,m,r,3),Y(t,i(265+165/2-50,165,296+165/2),c(165,330,165),-15,e,C.Metal),Y(t,i(130+165/2+20,82.5,65+165/2),c(165,165,165),18,o,C.Dielectric,1.5),new Float32Array(t)}class J{nodes=[];sortedPrims=[];build(e,n){this.nodes=[],this.sortedPrims=[];const s=12;for(let c=0;c<e.length/s;c++){const u=c*s,m=e[u+3];if(m<=0)continue;const l=e[u],f=e[u+1],d=e[u+2];this.sortedPrims.push({min:{x:l-m,y:f-m,z:d-m},max:{x:l+m,y:f+m,z:d+m},center:{x:l,y:f,z:d},type:0,originalIndex:c})}const r=16;for(let c=0;c<n.length/r;c++){const u=c*r,m=n[u],l=n[u+1],f=n[u+2],d=n[u+4],p=n[u+5],x=n[u+6],b=n[u+8],B=n[u+9],T=n[u+10],_={x:Math.min(m,d,b),y:Math.min(l,p,B),z:Math.min(f,x,T)},g={x:Math.max(m,d,b),y:Math.max(l,p,B),z:Math.max(f,x,T)},y=.001;g.x-_.x<y&&(_.x-=y,g.x+=y),g.y-_.y<y&&(_.y-=y,g.y+=y),g.z-_.z<y&&(_.z-=y,g.z+=y);const v={x:(_.x+g.x)*.5,y:(_.y+g.y)*.5,z:(_.z+g.z)*.5};this.sortedPrims.push({min:_,max:g,center:v,type:1,originalIndex:c})}const o={min:{x:0,y:0,z:0},max:{x:0,y:0,z:0},leftFirst:0,triCount:this.sortedPrims.length};this.nodes.push(o),this.updateNodeBounds(0),this.subdivide(0);const a=this.packNodes(),i=new Uint32Array(this.sortedPrims.length*2);for(let c=0;c<this.sortedPrims.length;c++)i[c*2]=this.sortedPrims[c].type,i[c*2+1]=this.sortedPrims[c].originalIndex;return{bvhNodes:a,primitiveRefs:i}}updateNodeBounds(e){const n=this.nodes[e];let s=!0;for(let r=0;r<n.triCount;r++){const o=this.sortedPrims[n.leftFirst+r];s?(n.min={...o.min},n.max={...o.max},s=!1):(n.min.x=Math.min(n.min.x,o.min.x),n.min.y=Math.min(n.min.y,o.min.y),n.min.z=Math.min(n.min.z,o.min.z),n.max.x=Math.max(n.max.x,o.max.x),n.max.y=Math.max(n.max.y,o.max.y),n.max.z=Math.max(n.max.z,o.max.z))}}subdivide(e){const n=this.nodes[e];if(n.triCount<=4)return;const s={x:n.max.x-n.min.x,y:n.max.y-n.min.y,z:n.max.z-n.min.z};let r=0;s.y>s.x&&(r=1),s.z>s.x&&s.z>s.y&&(r=2);const o=r===0?n.min.x+s.x*.5:r===1?n.min.y+s.y*.5:n.min.z+s.z*.5;let a=n.leftFirst,i=a+n.triCount-1;for(;a<=i;){const m=this.sortedPrims[a];if((r===0?m.center.x:r===1?m.center.y:m.center.z)<o)a++;else{const f=this.sortedPrims[a];this.sortedPrims[a]=this.sortedPrims[i],this.sortedPrims[i]=f,i--}}const c=a-n.leftFirst;if(c===0||c===n.triCount)return;const u=this.nodes.length;this.nodes.push({min:{x:0,y:0,z:0},max:{x:0,y:0,z:0},leftFirst:n.leftFirst,triCount:c}),this.nodes.push({min:{x:0,y:0,z:0},max:{x:0,y:0,z:0},leftFirst:a,triCount:n.triCount-c}),n.leftFirst=u,n.triCount=0,this.updateNodeBounds(u),this.updateNodeBounds(u+1),this.subdivide(u),this.subdivide(u+1)}packNodes(){const e=new Float32Array(this.nodes.length*8);for(let n=0;n<this.nodes.length;n++){const s=this.nodes[n],r=n*8;e[r]=s.min.x,e[r+1]=s.min.y,e[r+2]=s.min.z,e[r+3]=s.leftFirst,e[r+4]=s.max.x,e[r+5]=s.max.y,e[r+6]=s.max.z,e[r+7]=s.triCount}return e}}const F=1,h=document.getElementById("gpu-canvas"),L=document.getElementById("render-btn"),O=document.createElement("div");Object.assign(O.style,{position:"fixed",top:"10px",left:"10px",color:"#0f0",background:"rgba(0, 0, 0, 0.7)",padding:"8px",fontFamily:"monospace",fontSize:"14px",pointerEvents:"none",zIndex:"9999"});document.body.appendChild(O);async function K(){if(!navigator.gpu){alert("WebGPU not supported.");return}const t=await navigator.gpu.requestAdapter({powerPreference:"high-performance"});if(!t)throw new Error("No adapter found");const e=await t.requestDevice(),n=h.getContext("webgpu");if(!n)throw new Error("WebGPU context not found");(()=>{h.width=h.clientWidth*F,h.height=h.clientHeight*F})(),console.log(`DPR: ${F}, Buffer: ${h.width}x${h.height}`),n.configure({device:e,format:"rgba8unorm",usage:GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT});const r=e.createTexture({size:[h.width,h.height],format:"rgba8unorm",usage:GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.COPY_SRC}),o=r.createView(),a=h.width*h.height*16,i=e.createBuffer({size:a,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),c=e.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),u=e.createBuffer({size:96,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});{const E=W({x:0,y:1,z:-2.4},{x:0,y:1,z:0},{x:0,y:1,z:0},60,h.width/h.height,0,2.4);e.queue.writeBuffer(u,0,E)}const m=new Float32Array([0,-9999,0,0,0,0,0,0,0,0,0,0]),l=e.createBuffer({size:m.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});e.queue.writeBuffer(l,0,m);const f=X(),d=e.createBuffer({size:f.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});e.queue.writeBuffer(d,0,f);const x=new J().build(m,f);console.log(`BVH Nodes: ${x.bvhNodes.length/8}, Refs: ${x.primitiveRefs.length/2}`);const b=e.createBuffer({size:x.bvhNodes.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});e.queue.writeBuffer(b,0,x.bvhNodes);const B=e.createBuffer({size:x.primitiveRefs.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});e.queue.writeBuffer(B,0,x.primitiveRefs);const T=e.createShaderModule({label:"RayTracing",code:j}),_=e.createComputePipeline({label:"Main Pipeline",layout:"auto",compute:{module:T,entryPoint:"main"}}),g=_.getBindGroupLayout(0),y=e.createBindGroup({layout:g,entries:[{binding:0,resource:o},{binding:1,resource:{buffer:i}},{binding:2,resource:{buffer:c}},{binding:3,resource:{buffer:u}},{binding:4,resource:{buffer:l}},{binding:5,resource:{buffer:d}},{binding:6,resource:{buffer:b}},{binding:7,resource:{buffer:B}}]}),v=new Uint32Array(1),D=Math.ceil(h.width/8),k=Math.ceil(h.height/8),A={width:h.width,height:h.height,depthOrArrayLayers:1},I={texture:r},U={texture:null};let R=0,S=!1,M=performance.now(),w=0;const G=()=>{if(!S)return;const E=performance.now();R++,w++,v[0]=R,e.queue.writeBuffer(c,0,v),U.texture=n.getCurrentTexture();const q=e.createCommandEncoder(),N=q.beginComputePass();if(N.setPipeline(_),N.setBindGroup(0,y),N.dispatchWorkgroups(D,k),N.end(),q.copyTextureToTexture(I,U,A),e.queue.submit([q.finish()]),E-M>=1e3){const H=w,$=(1e3/H).toFixed(2);O.textContent=`FPS: ${H} | Frame Time: ${$}ms`,w=0,M=E}requestAnimationFrame(G)};L.addEventListener("click",()=>{if(S)console.log("Stop Rendering"),S=!1,L.textContent="Restart Rendering";else{console.log("Reset & Start Rendering"),R=0;const E=new Float32Array(a/4);e.queue.writeBuffer(i,0,E),S=!0,G(),L.textContent="Stop Rendering"}})}K().catch(console.error);
