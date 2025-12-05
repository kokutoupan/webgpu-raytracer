(function(){const n=document.createElement("link").relList;if(n&&n.supports&&n.supports("modulepreload"))return;for(const r of document.querySelectorAll('link[rel="modulepreload"]'))c(r);new MutationObserver(r=>{for(const o of r)if(o.type==="childList")for(const i of o.addedNodes)i.tagName==="LINK"&&i.rel==="modulepreload"&&c(i)}).observe(document,{childList:!0,subtree:!0});function t(r){const o={};return r.integrity&&(o.integrity=r.integrity),r.referrerPolicy&&(o.referrerPolicy=r.referrerPolicy),r.crossOrigin==="use-credentials"?o.credentials="include":r.crossOrigin==="anonymous"?o.credentials="omit":o.credentials="same-origin",o}function c(r){if(r.ep)return;r.ep=!0;const o=t(r);fetch(r.href,o)}})();const S=`// --- 0. Bindings & Uniforms ---

// Binding 0: 表示用テクスチャ (sRGB)
@group(0) @binding(0) var outputTex: texture_storage_2d<rgba8unorm, write>;

// Binding 1: 蓄積用バッファ (Linear, Read-Write)
@group(0) @binding(1) var<storage, read_write> accumulateBuffer: array<vec4<f32>>;

// Binding 2: フレーム情報 (毎フレーム更新)
struct FrameInfo {
    frame_count: u32,
    // padding...
}
@group(0) @binding(2) var<uniform> frame: FrameInfo;

// カメラ構造体 (Defocus Blur対応版)
struct Camera {
    origin: vec3<f32>,
    lens_radius: f32,       // Originのw成分
    lower_left_corner: vec3<f32>,
    horizontal: vec3<f32>,
    vertical: vec3<f32>,
    u: vec3<f32>,           // カメラの右方向 (ボケ用)
    v: vec3<f32>,           // カメラの上方向 (ボケ用)
}
@group(0) @binding(3) var<uniform> camera: Camera;


@group(0) @binding(4) var<storage, read> scene_spheres: array<Sphere>;


// --- 1. Constants & Structs ---

const PI = 3.141592653589793;

struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
}

struct Sphere {
    center: vec3<f32>,
    radius: f32,
    color: vec3<f32>,   // マテリアルの色 (Albedo)
    mat_type: f32,      // 0: Lambertian, 1: Metal
    extra: f32,         // Fuzz など
    // padding... (WGSLでは明示しなくてもアライメントが合っていればOK)
}

struct HitRecord {
    p: vec3<f32>,
    normal: vec3<f32>,
    t: f32,
    front_face: bool,
}

struct RandomGenerator {
    state: u32,
}

// --- 2. Random Functions ---

fn rand_pcg(rng: ptr<function, RandomGenerator>) -> f32 {
    let old_state = (*rng).state;
    (*rng).state = old_state * 747796405u + 2891336453u;
    let word = ((*rng).state >> ((old_state >> 28u) + 4u)) ^ (*rng).state;
    let result = word * 277803737u;
    let final_hash = (result >> 22u) ^ result;
    return f32(final_hash) / 4294967295.0;
}

fn random_unit_vector(rng: ptr<function, RandomGenerator>) -> vec3<f32> {
    let z = rand_pcg(rng) * 2.0 - 1.0;
    let a = rand_pcg(rng) * 2.0 * PI;
    let r = sqrt(max(0.0, 1.0 - z * z));
    let x = r * cos(a);
    let y = r * sin(a);
    return vec3<f32>(x, y, z);
}

// 単位円盤内のランダムな点 (Defocus Blur用)
fn random_in_unit_disk(rng: ptr<function, RandomGenerator>) -> vec3<f32> {
    // 棄却法
    for (var i = 0; i < 10; i++) {
        let p = vec3<f32>(rand_pcg(rng) * 2.0 - 1.0, rand_pcg(rng) * 2.0 - 1.0, 0.0);
        if dot(p, p) < 1.0 {
            return p;
        }
    }
    return vec3<f32>(0.0);
}

// --- 3. Geometry Functions ---

// フレネル反射率の計算 (Schlick's approximation)
fn reflectance(cosine: f32, ref_idx: f32) -> f32 {
    var r0 = (1.0 - ref_idx) / (1.0 + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0 - r0) * pow((1.0 - cosine), 5.0);
}

fn ray_at(r: Ray, t: f32) -> vec3<f32> {
    return r.origin + t * r.direction;
}

fn hit_sphere(s: Sphere, r: Ray, t_min: f32, t_max: f32, rec: ptr<function, HitRecord>) -> bool {
    let oc = r.origin - s.center;
    let a = dot(r.direction, r.direction);
    let h = dot(r.direction, oc);
    let c = dot(oc, oc) - s.radius * s.radius;
    let discriminant = h * h - a * c;

    if discriminant < 0.0 { return false; }

    let sqrtd = sqrt(discriminant);
    var root = (-h - sqrtd) / a;
    if root <= t_min || t_max <= root {
        root = (-h + sqrtd) / a;
        if root <= t_min || t_max <= root {
            return false;
        }
    }

    (*rec).t = root;
    (*rec).p = ray_at(r, root);
    let outward_normal = ((*rec).p - s.center) / s.radius;
    (*rec).front_face = dot(r.direction, outward_normal) < 0.0;
    (*rec).normal = select(-outward_normal, outward_normal, (*rec).front_face);

    return true;
}

// --- 4. Ray Tracing Logic ---

fn ray_color(r_in: Ray, rng: ptr<function, RandomGenerator>) -> vec3<f32> {
    let sphere_count = arrayLength(&scene_spheres);


    var ray = r_in;
    var color = vec3<f32>(0.0);
    var throughput = vec3<f32>(1.0);

    for (var depth = 0u; depth < 10u; depth++) {
        var rec: HitRecord;
        var hit_anything = false;
        var closest_so_far = 9999999.0;
        let t_min = 0.001;
        var hit_index = -1; // どの球に当たったか


        for (var i = 0u; i < sphere_count; i++) {
            if hit_sphere(scene_spheres[i], ray, t_min, closest_so_far, &rec) {
                hit_anything = true;
                closest_so_far = rec.t;
                hit_index = i32(i);
            }
        }

        if hit_anything {
            // ヒットした球の情報を取得
            let s = scene_spheres[hit_index];
            var scattered_direction = vec3<f32>(0.0);

            if s.mat_type < 0.5 { // diffusion
                scattered_direction = rec.normal + random_unit_vector(rng);
            } else if s.mat_type < 1.5 { // metal
                scattered_direction = reflect(ray.direction, rec.normal) + s.extra * random_unit_vector(rng);
            } else {  // === 2: Dielectric (ガラス/水) ===
                // s.extra を 屈折率 (IR) として使う (例: 1.5)
                
                // 1. 屈折率の比率 (eta) を決定
                // 表から入る場合: 1.0 / 1.5, 裏から出る場合: 1.5 / 1.0
                let refraction_ratio = select(s.extra, 1.0 / s.extra, rec.front_face);

                let unit_direction = normalize(ray.direction);
                
                // 2. 全反射 (Total Internal Reflection) の判定用パラメータ
                // cos_theta: 入射角のコサイン (0.0 ~ 1.0)
                // dot(-unit, normal) と 1.0 の小さい方をとる
                let cos_theta = min(dot(-unit_direction, rec.normal), 1.0);
                let sin_theta = sqrt(1.0 - cos_theta * cos_theta);

                // 全反射するかどうか (スネルの法則が成立しない場合)
                let cannot_refract = refraction_ratio * sin_theta > 1.0;
                
                // 3. 反射するか屈折するかの決定
                // (全反射条件) OR (フレネル反射確率 > 乱数)
                let do_reflect = cannot_refract || (reflectance(cos_theta, refraction_ratio) > rand_pcg(rng));

                if do_reflect {
                    // 反射 (組み込み関数)
                    scattered_direction = reflect(unit_direction, rec.normal);
                } else {
                    // 屈折 (組み込み関数)
                    // WGSLのrefractは (入射ベクトル, 法線, eta) を取る
                    scattered_direction = refract(unit_direction, rec.normal, refraction_ratio);
                }
            }

            ray = Ray(rec.p, scattered_direction);
            throughput *= s.color;
            if length(throughput) < 0.001 { break; }
        } else {
            // ヒットなし(空)
            let unit_direction = normalize(ray.direction);
            let t = 0.5 * (unit_direction.y + 1.0);
            let sky_color = mix(vec3<f32>(1.0), vec3<f32>(0.5, 0.7, 1.0), t);
            color += throughput * sky_color;
            break;
        }
    }
    return color;
}

// --- 5. Main Compute Shader ---

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(outputTex);
    if id.x >= dims.x || id.y >= dims.y { return; }

    let pixel_idx = id.y * dims.x + id.x;

    // 乱数初期化 (ピクセル位置 + フレーム数でシード変化)
    var rng = RandomGenerator(pixel_idx * 719393u + frame.frame_count * 51234u);

    // --- カメラレイ生成 (Defocus Blur対応) ---
    
    // UV + Jitter
    let u_offset = rand_pcg(&rng);
    let v_offset = rand_pcg(&rng);
    let s = (f32(id.x) + u_offset) / f32(dims.x);
    let t = 1.0 - ((f32(id.y) + v_offset) / f32(dims.y));

    // レンズ上のオフセット (ボケ)
    var ray_origin = camera.origin;
    var ray_offset = vec3<f32>(0.0);

    if (camera.lens_radius > 0.0) {
        let rd = camera.lens_radius * random_in_unit_disk(&rng);
        ray_offset = camera.u * rd.x + camera.v * rd.y;
        ray_origin += ray_offset;
    }

    let direction = camera.lower_left_corner 
                  + (s * camera.horizontal) 
                  + (t * camera.vertical) 
                  - camera.origin
                  - ray_offset;

    let r = Ray(camera.origin, direction);

    // --- 色計算 ---
    // 蓄積レンダリング時は spp=1 で十分
    let pixel_color_linear = ray_color(r, &rng);

    // --- 蓄積処理 (Linear Space) ---
    var accumulated = vec4<f32>(0.0);
    if frame.frame_count > 1u {
        accumulated = accumulateBuffer[pixel_idx];
    }

    let new_accumulated = accumulated + vec4<f32>(pixel_color_linear, 1.0);
    accumulateBuffer[pixel_idx] = new_accumulated;

    // --- 表示処理 (Gamma Correction) ---
    let frame_f32 = f32(frame.frame_count);
    var final_color = new_accumulated.rgb / frame_f32;

    // Linear -> sRGB (簡易ガンマ 2.2近似 = sqrt)
    final_color = sqrt(final_color);
    final_color = clamp(final_color, vec3<f32>(0.0), vec3<f32>(1.0));

    textureStore(outputTex, vec2<i32>(id.xy), vec4<f32>(final_color, 1.0));
}
`,g=document.getElementById("gpu-canvas"),P=document.getElementById("render-btn"),a={create:(e,n,t)=>({x:e,y:n,z:t}),sub:(e,n)=>({x:e.x-n.x,y:e.y-n.y,z:e.z-n.z}),add:(e,n)=>({x:e.x+n.x,y:e.y+n.y,z:e.z+n.z}),scale:(e,n)=>({x:e.x*n,y:e.y*n,z:e.z*n}),cross:(e,n)=>({x:e.y*n.z-e.z*n.y,y:e.z*n.x-e.x*n.z,z:e.x*n.y-e.y*n.x}),normalize:e=>{const n=Math.sqrt(e.x*e.x+e.y*e.y+e.z*e.z);return n===0?{x:0,y:0,z:0}:{x:e.x/n,y:e.y/n,z:e.z/n}},len:e=>Math.sqrt(e.x*e.x+e.y*e.y+e.z*e.z)},f=()=>Math.random(),B=(e,n)=>e+(n-e)*Math.random();function M(e,n,t,c,r,o,i){const y=c*Math.PI/180,_=2*Math.tan(y/2)*i,w=_*r,b=a.normalize(a.sub(e,n)),l=a.normalize(a.cross(t,b)),v=a.cross(b,l),m=e,d=a.scale(l,w),u=a.scale(v,_),z=a.sub(a.sub(a.sub(m,a.scale(d,.5)),a.scale(u,.5)),a.scale(b,i)),s=i*Math.tan(o*Math.PI/360);return new Float32Array([m.x,m.y,m.z,s,z.x,z.y,z.z,0,d.x,d.y,d.z,0,u.x,u.y,u.z,0,l.x,l.y,l.z,0,v.x,v.y,v.z,0])}const p={Lambertian:0,Metal:1,Dielectric:2};function h(e,n,t,c,r=0){return[e.x,e.y,e.z,n,t.r,t.g,t.b,c,r,0,0,0]}function C(){const e=[];e.push(h({x:0,y:-1e3,z:0},1e3,{r:.5,g:.5,b:.5},p.Lambertian));for(let n=-11;n<11;n++)for(let t=-11;t<11;t++){const c=f(),r={x:n+.9*f(),y:.2,z:t+.9*f()};if(a.len(a.sub(r,{x:4,y:.2,z:0}))>.9)if(c<.8){const i=f()*f(),y=f()*f(),x=f()*f();e.push(h(r,.2,{r:i,g:y,b:x},p.Lambertian))}else if(c<.95){const i=B(.5,1),y=B(.5,1),x=B(.5,1),_=B(0,.5);e.push(h(r,.2,{r:i,g:y,b:x},p.Metal,_))}else e.push(h(r,.2,{r:1,g:1,b:1},p.Dielectric,1.5))}return e.push(h({x:0,y:1,z:0},1,{r:1,g:1,b:1},p.Dielectric,1.5)),e.push(h({x:-4,y:1,z:0},1,{r:.4,g:.2,b:.1},p.Lambertian)),e.push(h({x:4,y:1,z:0},1,{r:.7,g:.6,b:.5},p.Metal,0)),e}async function q(){if(!navigator.gpu){alert("WebGPU not supported.");return}const e=await navigator.gpu.requestAdapter();if(!e)throw new Error("No adapter found");const n=await e.requestDevice(),t=g.getContext("webgpu");if(!t)throw new Error("WebGPU context not found");t.configure({device:n,format:"rgba8unorm",usage:GPUTextureUsage.STORAGE_BINDING});const c=g.width*g.height*4*4,r=n.createBuffer({size:c,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),o=n.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),i=n.createBuffer({size:96,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});{const s={lookfrom:{x:13,y:2,z:3},lookat:{x:0,y:0,z:0},vup:{x:0,y:1,z:0},vfov:20,aspectRatio:g.width/g.height,defocusAngle:.2,focusDist:10},G=M(s.lookfrom,s.lookat,s.vup,s.vfov,s.aspectRatio,s.defocusAngle,s.focusDist);n.queue.writeBuffer(i,0,G)}const x=C().flat(),_=new Float32Array(x),w=n.createBuffer({size:_.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});n.queue.writeBuffer(w,0,_);const b=n.createShaderModule({label:"RayTracing",code:S}),l=n.createComputePipeline({label:"Main Pipeline",layout:"auto",compute:{module:b,entryPoint:"main"}}),v=l.getBindGroupLayout(0),m=new Uint32Array(1);let d=0,u=!1;const z=()=>{if(!u)return;d++,m[0]=d,n.queue.writeBuffer(o,0,m);const G=t.getCurrentTexture().createView(),L=n.createBindGroup({layout:v,entries:[{binding:0,resource:G},{binding:1,resource:{buffer:r}},{binding:2,resource:{buffer:o}},{binding:3,resource:{buffer:i}},{binding:4,resource:{buffer:w}}]}),U=n.createCommandEncoder(),R=U.beginComputePass();R.setPipeline(l),R.setBindGroup(0,L),R.dispatchWorkgroups(Math.ceil(g.width/8),Math.ceil(g.height/8)),R.end(),n.queue.submit([U.finish()]),requestAnimationFrame(z)};P.addEventListener("click",()=>{if(u)console.log("Stop Rendering"),u=!1,P.textContent="Restart Rendering";else{console.log("Reset & Start Rendering"),d=0;const s=new Float32Array(c/4);n.queue.writeBuffer(r,0,s),u=!0,z(),P.textContent="Stop Rendering"}})}q().catch(console.error);
