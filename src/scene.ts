import { vec3, rnd, rndRange, type Vec3 } from './math';
import { Sphere, Triangle, type IHittable } from './primitives';

export const MatType = {
  Lambertian: 0.0,
  Metal: 1.0,
  Dielectric: 2.0,
  Light: 3.0,
};

// --- 型定義 ---

// カメラの初期設定値
export interface CameraConfig {
  lookfrom: Vec3;
  lookat: Vec3;
  vup: Vec3;
  vfov: number;
  defocusAngle: number;
  focusDist: number;
}

export interface SceneData {
  camera: CameraConfig; // (定義は省略)
  primitives: IHittable[]; // ★ここがシンプルになる
}

// --- カメラデータ生成 (Buffer用) ---
export function createCameraData(
  cfg: CameraConfig,
  aspectRatio: number
): Float32Array<ArrayBuffer> {
  const theta = (cfg.vfov * Math.PI) / 180.0;
  const h = Math.tan(theta / 2.0);
  const viewportHeight = 2.0 * h * cfg.focusDist;
  const viewportWidth = viewportHeight * aspectRatio;

  const w = vec3.normalize(vec3.sub(cfg.lookfrom, cfg.lookat));
  const u = vec3.normalize(vec3.cross(cfg.vup, w));
  const v = vec3.cross(w, u);

  const horizontal = vec3.scale(u, viewportWidth);
  const vertical = vec3.scale(v, viewportHeight);
  const lowerLeft = vec3.sub(
    vec3.sub(vec3.sub(cfg.lookfrom, vec3.scale(horizontal, 0.5)), vec3.scale(vertical, 0.5)),
    vec3.scale(w, cfg.focusDist)
  );
  const lensRadius = cfg.focusDist * Math.tan((cfg.defocusAngle * Math.PI) / 360.0);

  return new Float32Array([
    cfg.lookfrom.x, cfg.lookfrom.y, cfg.lookfrom.z, lensRadius,
    lowerLeft.x, lowerLeft.y, lowerLeft.z, 0.0,
    horizontal.x, horizontal.y, horizontal.z, 0.0,
    vertical.x, vertical.y, vertical.z, 0.0,
    u.x, u.y, u.z, 0.0,
    v.x, v.y, v.z, 0.0
  ]);
}

// --- ヘルパー ---
function addQuad(list: IHittable[], v0: Vec3, v1: Vec3, v2: Vec3, v3: Vec3, col: Vec3, mat: number, extra: number = 0.0) {
  list.push(new Triangle(v0, v1, v2, col, mat, extra));
  list.push(new Triangle(v0, v2, v3, col, mat, extra));
}

// --- ★追加: 箱生成ヘルパー (Exportして再利用可能に) ---
export function addBox(
  list: IHittable[],
  center: Vec3,
  size: Vec3,
  angleY: number,
  col: Vec3,
  mat: number,
  extra: number = 0.0
) {
  const rad = (angleY * Math.PI) / 180.0;
  const cosA = Math.cos(rad);
  const sinA = Math.sin(rad);

  const rotate = (p: Vec3): Vec3 => ({
    x: p.x * cosA + p.z * sinA,
    y: p.y,
    z: -p.x * sinA + p.z * cosA
  });

  const hx = size.x / 2;
  const hy = size.y / 2;
  const hz = size.z / 2;

  // Local vertices
  const p0 = { x: -hx, y: -hy, z: hz };
  const p1 = { x: hx, y: -hy, z: hz };
  const p2 = { x: hx, y: hy, z: hz };
  const p3 = { x: -hx, y: hy, z: hz };
  const p4 = { x: -hx, y: -hy, z: -hz };
  const p5 = { x: hx, y: -hy, z: -hz };
  const p6 = { x: hx, y: hy, z: -hz };
  const p7 = { x: -hx, y: hy, z: -hz };

  const transform = (p: Vec3) => {
    const rot = rotate(p);
    return { x: rot.x + center.x, y: rot.y + center.y, z: rot.z + center.z };
  };

  const v0 = transform(p0), v1 = transform(p1), v2 = transform(p2), v3 = transform(p3);
  const v4 = transform(p4), v5 = transform(p5), v6 = transform(p6), v7 = transform(p7);

  // 6 faces
  addQuad(list, v0, v1, v2, v3, col, mat, extra); // Front
  addQuad(list, v5, v4, v7, v6, col, mat, extra); // Back
  addQuad(list, v4, v0, v3, v7, col, mat, extra); // Left
  addQuad(list, v1, v5, v6, v2, col, mat, extra); // Right
  addQuad(list, v3, v2, v6, v7, col, mat, extra); // Top
  addQuad(list, v0, v4, v5, v1, col, mat, extra); // Bottom
}

// ==========================================
//   各シーンの定義
// ==========================================

// 1. コーネルボックス
function getCornellBoxScene(): SceneData {
  const triangles: IHittable[] = [];
  const white = { x: 0.73, y: 0.73, z: 0.73 };
  const red = { x: 0.65, y: 0.05, z: 0.05 };
  const green = { x: 0.12, y: 0.45, z: 0.15 };
  const light = { x: 50.0, y: 50.0, z: 50.0 };

  const S = 555;
  const v = (x: number, y: number, z: number) => ({ x: (x / S) * 2 - 1, y: (y / S) * 2, z: (z / S) * 2 - 1 });
  const s = (x: number, y: number, z: number) => ({ x: (x / S) * 2, y: (y / S) * 2, z: (z / S) * 2 });

  // Walls
  addQuad(triangles, v(0, 0, 0), v(555, 0, 0), v(555, 0, 555), v(0, 0, 555), white, MatType.Lambertian);
  addQuad(triangles, v(0, 555, 0), v(0, 555, 555), v(555, 555, 555), v(555, 555, 0), white, MatType.Lambertian);
  addQuad(triangles, v(0, 0, 555), v(555, 0, 555), v(555, 555, 555), v(0, 555, 555), white, MatType.Lambertian);
  addQuad(triangles, v(0, 0, 0), v(0, 555, 0), v(0, 555, 555), v(0, 0, 555), green, MatType.Lambertian);
  addQuad(triangles, v(555, 0, 0), v(555, 0, 555), v(555, 555, 555), v(555, 555, 0), red, MatType.Lambertian);

  // Light
  const l0 = v(213, 554, 227), l1 = v(343, 554, 227), l2 = v(343, 554, 332), l3 = v(213, 554, 332);
  addQuad(triangles, l0, l3, l2, l1, light, MatType.Light);

  // Boxes (★addBoxを使用)
  addBox(triangles, v(265 + 82.5 - 50, 165, 296 + 82.5), s(165, 330, 165), -15, white, MatType.Lambertian);
  addBox(triangles, v(130 + 82.5 + 20, 82.5, 65 + 82.5), s(165, 165, 165), 18, white, MatType.Lambertian);

  return {
    camera: {
      lookfrom: { x: 0, y: 1.0, z: -2.4 },
      lookat: { x: 0, y: 1.0, z: 0 },
      vup: { x: 0, y: 1, z: 0 },
      vfov: 60.0,
      defocusAngle: 0.0,
      focusDist: 2.4
    },
    primitives: triangles
  };
}

// 2. ランダムな球 (One Weekend Final Scene風)
function getRandomSpheresScene(): SceneData {
  const spheres: IHittable[] = [];

  // Ground
  spheres.push(new Sphere({ x: 0, y: -1000, z: 0 }, 1000, { x: 0.5, y: 0.5, z: 0.5 }, MatType.Lambertian, 0));

  // sun
  spheres.push(new Sphere({ x: -50, y: 50, z: -50 }, 30, { x: 10, y: 10, z: 10 }, MatType.Light, 0));

  // Random Spheres
  for (let a = -11; a < 11; a++) {
    for (let b = -11; b < 11; b++) {
      const chooseMat = rnd();
      const center = { x: a + 0.9 * rnd(), y: 0.2, z: b + 0.9 * rnd() };
      const dist = vec3.len(vec3.sub(center, { x: 4.0, y: 0.2, z: 0.0 }));

      if (dist > 0.9) {
        if (chooseMat < 0.8) {
          const col = { x: rnd() * rnd(), y: rnd() * rnd(), z: rnd() * rnd() };
          spheres.push(new Sphere(center, 0.2, col, MatType.Lambertian, 0));
        } else if (chooseMat < 0.95) {
          const col = { x: rndRange(0.5, 1), y: rndRange(0.5, 1), z: rndRange(0.5, 1) };
          spheres.push(new Sphere(center, 0.2, col, MatType.Metal, rndRange(0, 0.5)));
        } else {
          spheres.push(new Sphere(center, 0.2, { x: 1, y: 1, z: 1 } as any, MatType.Dielectric, 1.5));
        }
      }
    }
  }

  // Big Spheres
  spheres.push(new Sphere({ x: 0, y: 1, z: 0 }, 1.0, { x: 1, y: 1, z: 1 }, MatType.Dielectric, 1.5));
  spheres.push(new Sphere({ x: -4, y: 1, z: 0 }, 1.0, { x: 0.4, y: 0.2, z: 0.1 }, MatType.Lambertian, 0));
  spheres.push(new Sphere({ x: 4, y: 1, z: 0 }, 1.0, { x: 0.7, y: 0.6, z: 0.5 }, MatType.Metal, 0.0));

  return {
    camera: {
      lookfrom: { x: 13, y: 2, z: 3 },
      lookat: { x: 0, y: 0, z: 0 },
      vup: { x: 0, y: 1, z: 0 },
      vfov: 20.0,
      defocusAngle: 0.6,
      focusDist: 10.0
    },
    primitives: spheres
  };
}

// ==========================================
//   3. ミックス (アーティスティックなシーン)
// ==========================================
function getMixedScene(): SceneData {
  const objects: IHittable[] = [];

  // --- 1. 床 (Dark Mirror Floor) ---
  // 少しだけザラつき(fuzz 0.05)のある、暗い鏡の床
  // 反射が綺麗に伸びます
  const floorCol = { x: 0.1, y: 0.1, z: 0.1 };
  addBox(objects, { x: 0, y: -1.0, z: 0 }, { x: 40, y: 2, z: 40 }, 0, floorCol, MatType.Metal, 0.05);

  // --- 2. ライティング (Cinematic 2-Point Lighting) ---

  // Key Light (暖色・強め・右上)
  // 影を落とすメインの光
  const warmLight = { x: 40.0, y: 30.0, z: 10.0 }; // 明るいオレンジ
  const lA_pos = { x: -4, y: 8, z: 4 };
  addQuad(objects,
    { x: lA_pos.x, y: lA_pos.y, z: lA_pos.z },
    { x: lA_pos.x + 2, y: lA_pos.y, z: lA_pos.z },
    { x: lA_pos.x + 2, y: lA_pos.y, z: lA_pos.z + 2 },
    { x: lA_pos.x, y: lA_pos.y, z: lA_pos.z + 2 },
    warmLight, MatType.Light
  );

  // Rim Light (寒色・弱め・左奥)
  // 輪郭を際立たせる青い光
  const coolLight = { x: 5.0, y: 10.0, z: 20.0 }; // 青
  const lB_pos = { x: 4, y: 6, z: -4 };
  addQuad(objects,
    { x: lB_pos.x, y: lB_pos.y, z: lB_pos.z },
    { x: lB_pos.x + 3, y: lB_pos.y, z: lB_pos.z },
    { x: lB_pos.x + 3, y: lB_pos.y - 3, z: lB_pos.z }, // 垂直に立ててみる
    { x: lB_pos.x, y: lB_pos.y - 3, z: lB_pos.z },
    coolLight, MatType.Light
  );

  // --- 3. 中央のオブジェ (The Tower) ---

  // 土台: 金のブロック
  const gold = { x: 0.8, y: 0.6, z: 0.2 };
  addBox(objects, { x: 0, y: 0.5, z: 0 }, { x: 2, y: 1, z: 2 }, 45, gold, MatType.Metal, 0.1);

  // 中段: ガラスの球
  // 屈折率 1.5 (ガラス)
  objects.push(new Sphere({ x: 0, y: 1.8, z: 0 }, 0.8, { x: 1, y: 1, z: 1 }, MatType.Dielectric, 1.5));
  // 中に気泡を入れる (屈折率 1.0/1.5 の空気を中に入れると泡に見える)
  objects.push(new Sphere({ x: 0, y: 1.8, z: 0 }, -0.7, { x: 1, y: 1, z: 1 }, MatType.Dielectric, 1.0));

  // 上段: 浮遊する赤いガラスキューブ
  const ruby = { x: 0.9, y: 0.1, z: 0.1 };
  // addBox(triangles, {x:0, y:3.2, z:0}, {x:1, y:1, z:1}, 30, ruby, MatType.Dielectric, 1.5);
  // キューブだと透過が計算しにくいので、ここはメタルにする
  addBox(objects, { x: 0, y: 3.0, z: 0 }, { x: 0.8, y: 0.8, z: 0.8 }, 15, ruby, MatType.Metal, 0.2);


  // --- 4. 周囲の浮遊リング (Procedural Ring) ---

  const count = 12;
  const radius = 4.0;

  for (let i = 0; i < count; i++) {
    const angle = (i / count) * Math.PI * 2;
    const x = Math.cos(angle) * radius;
    const z = Math.sin(angle) * radius;
    const y = 1.0 + Math.sin(angle * 3) * 0.5; // 波打たせる

    // 偶数は球、奇数は箱
    if (i % 2 === 0) {
      // 鏡の球
      const col = { x: 0.8, y: 0.8, z: 0.8 };
      objects.push(new Sphere({ x, y, z }, 0.4, col as any, MatType.Metal, 0.0));
    } else {
      // カラフルな箱 (Diffuse)
      const r = 0.5 + 0.5 * Math.cos(i);
      const g = 0.5 + 0.5 * Math.sin(i);
      const b = 0.8;
      addBox(objects, { x, y, z }, { x: 0.6, y: 0.6, z: 0.6 }, i * 20, { x: r, y: g, z: b }, MatType.Lambertian);
    }
  }

  // --- 5. 背景の柱 (Depth Reference) ---
  // 奥に巨大なモノリスを置いて、反射とシルエットを作る
  addBox(objects, { x: -4, y: 3, z: -6 }, { x: 1, y: 6, z: 1 }, 10, { x: 0.2, y: 0.2, z: 0.3 }, MatType.Lambertian);
  addBox(objects, { x: 4, y: 2, z: -5 }, { x: 1, y: 4, z: 1 }, -20, { x: 0.2, y: 0.2, z: 0.3 }, MatType.Lambertian);


  return {
    camera: {
      lookfrom: { x: 0, y: 3.5, z: 9 }, // 少し上から見下ろす
      lookat: { x: 0, y: 1.5, z: 0 }, // タワーの中心を見る
      vup: { x: 0, y: 1, z: 0 },
      vfov: 40.0, // 望遠気味にして歪みを減らす
      defocusAngle: 0.3, // ほんのりボケさせる (ミニチュア効果)
      focusDist: 9.0     // 中心にピントを合わせる
    },
    primitives: objects
  };
}

// --- シーン取得関数 ---
export function getSceneData(name: string): SceneData {
  switch (name) {
    case 'spheres': return getRandomSpheresScene();
    case 'mixed': return getMixedScene();
    case 'cornell':
    default: return getCornellBoxScene();
  }
}
