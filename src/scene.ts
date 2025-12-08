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

// 1. 原点(0,0,0)中心の箱を作る (まだリストには追加しない)
function createBox(size: Vec3, col: Vec3, mat: number, extra: number = 0.0): IHittable[] {
  const box: IHittable[] = [];
  const hx = size.x / 2, hy = size.y / 2, hz = size.z / 2;

  // 頂点 (原点中心)
  const p0 = { x: -hx, y: -hy, z: hz };
  const p1 = { x: hx, y: -hy, z: hz };
  const p2 = { x: hx, y: hy, z: hz };
  const p3 = { x: -hx, y: hy, z: hz };
  const p4 = { x: -hx, y: -hy, z: -hz };
  const p5 = { x: hx, y: -hy, z: -hz };
  const p6 = { x: hx, y: hy, z: -hz };
  const p7 = { x: -hx, y: hy, z: -hz };

  // Quad生成ヘルパー (内部用)
  const pushQuad = (v0: Vec3, v1: Vec3, v2: Vec3, v3: Vec3) => {
    box.push(new Triangle(v0, v1, v2, col, mat, extra));
    box.push(new Triangle(v0, v2, v3, col, mat, extra));
  };

  pushQuad(p0, p1, p2, p3); // Front
  pushQuad(p5, p4, p7, p6); // Back
  pushQuad(p4, p0, p3, p7); // Left
  pushQuad(p1, p5, p6, p2); // Right
  pushQuad(p3, p2, p6, p7); // Top
  pushQuad(p0, p4, p5, p1); // Bottom

  return box;
}

// 2. リスト内の全オブジェクトを回転・移動して、メインリストに追加する
function addTransformed(
  targetList: IHittable[],
  sourceObjects: IHittable[],
  pos: Vec3,
  rotY: number
) {
  for (const obj of sourceObjects) {
    const newObj = obj.clone(); // 元の形を壊さないように複製
    if (rotY !== 0) newObj.rotateY(rotY); // まず回転
    newObj.translate(pos); // 次に移動
    targetList.push(newObj);
  }
}

// --- ヘルパー ---
function addQuad(list: IHittable[], v0: Vec3, v1: Vec3, v2: Vec3, v3: Vec3, col: Vec3, mat: number, extra: number = 0.0) {
  list.push(new Triangle(v0, v1, v2, col, mat, extra));
  list.push(new Triangle(v0, v2, v3, col, mat, extra));
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
  const light = { x: 20.0, y: 20.0, z: 20.0 };

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
  addQuad(triangles, l0, l1, l2, l3, light, MatType.Light,0.0);

  // Boxes (★addBoxを使用)
  // 1. 背の高い箱 (Tall Box)
  // サイズ: 165 x 330 x 165
  // 位置: x(265+82.5-50)=297.5, y(165), z(296+82.5)=378.5
  // 回転: -15度
  const tallBoxGeo = createBox(s(165, 330, 165), white, MatType.Lambertian);
  addTransformed(triangles, tallBoxGeo, v(297.5, 165, 378.5), -15);

  // 2. 背の低い箱 (Short Box)
  // サイズ: 165 x 165 x 165
  // 位置: x(130+82.5+20)=232.5, y(82.5), z(65+82.5)=147.5
  // 回転: 18度
  const shortBoxGeo = createBox(s(165, 165, 165), white, MatType.Lambertian);
  addTransformed(triangles, shortBoxGeo, v(232.5, 82.5, 147.5), 18);

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
  spheres.push(new Sphere({ x: -50, y: 50, z: -50 }, 30, { x: 3, y: 2.7, z: 2.7 }, MatType.Light, 0));

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
  const box = createBox({ x: 40, y: 2, z: 40 }, floorCol, MatType.Metal, 0.05);
  addTransformed(objects, box, { x: 0, y: -1.0, z: 0 }, 0)

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
  const goldbox = createBox({ x: 2, y: 1, z: 2 }, gold, MatType.Metal, 0.1);
  addTransformed(objects, goldbox, { x: 0, y: 0.5, z: 0 }, 0);

  // 中段: ガラスの球
  // 屈折率 1.5 (ガラス)
  objects.push(new Sphere({ x: 0, y: 1.8, z: 0 }, 0.8, { x: 1, y: 1, z: 1 }, MatType.Dielectric, 1.5));
  // 中に気泡を入れる (屈折率 1.0/1.5 の空気を中に入れると泡に見える)
  objects.push(new Sphere({ x: 0, y: 1.8, z: 0 }, -0.7, { x: 1, y: 1, z: 1 }, MatType.Dielectric, 1.0));

  // 上段: 浮遊する赤いガラスキューブ
  const ruby = { x: 0.9, y: 0.1, z: 0.1 };
  // addBox(triangles, {x:0, y:3.2, z:0}, {x:1, y:1, z:1}, 30, ruby, MatType.Dielectric, 1.5);
  // キューブだと透過が計算しにくいので、ここはメタルにする
  addTransformed(objects, createBox({ x: 0.8, y: 0.8, z: 0.8 }, ruby, MatType.Metal, 0.2), { x: 0, y: 3.2, z: 0 }, 15);


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
      addTransformed(objects, createBox({ x: 0.6, y: 0.6, z: 0.6 }, { x: r, y: g, z: b }, MatType.Lambertian), { x, y, z }, i * 20)
    }
  }

  // --- 5. 背景の柱 (Depth Reference) ---
  // 奥に巨大なモノリスを置いて、反射とシルエットを作る
  // 1. 左奥の柱
  // サイズ: 1x6x1, 位置: (-4, 3, -6), 回転: 10度, 色: ダークブルーグレー
  const colPillar = { x: 0.2, y: 0.2, z: 0.3 };
  const pillar1Geo = createBox({ x: 1, y: 6, z: 1 }, colPillar, MatType.Lambertian);
  addTransformed(objects, pillar1Geo, { x: -4, y: 3, z: -6 }, 10);

  // 2. 右奥の柱
  // サイズ: 1x4x1, 位置: (4, 2, -5), 回転: -20度
  const pillar2Geo = createBox({ x: 1, y: 4, z: 1 }, colPillar, MatType.Lambertian);
  addTransformed(objects, pillar2Geo, { x: 4, y: 2, z: -5 }, -20);


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

// ==========================================
//   新しいシーン (ガラスの箱 + 中に青い光球)
// ==========================================
function getCornellBoxSpecialScene(): SceneData {
  const objects: IHittable[] = [];

  const white = { x: 0.73, y: 0.73, z: 0.73 };
  const red = { x: 0.65, y: 0.05, z: 0.05 };
  const green = { x: 0.12, y: 0.45, z: 0.15 };
  const light = { x: 20.0, y: 20.0, z: 20.0 };

  const blueLight = { x: 0.1, y: 0.1, z: 10. }; // 青を少し強調したほうが綺麗に見えます
  const glassColor = { x: 0.95, y: 0.95, z: 0.95 };

  const S = 555;
  const v = (x: number, y: number, z: number) => ({ x: (x / S) * 2 - 1, y: (y / S) * 2, z: (z / S) * 2 - 1 });
  const s = (x: number, y: number, z: number) => ({ x: (x / S) * 2, y: (y / S) * 2, z: (z / S) * 2 });

  // --- 壁・床・天井 (オリジナルと同じ) ---
  // 床を Metal (粗さ0.3) にして反射を楽しむ設定は維持
  addQuad(objects, v(0, 0, 0), v(555, 0, 0), v(555, 0, 555), v(0, 0, 555), white, MatType.Metal, 0.1);
  addQuad(objects, v(0, 555, 0), v(0, 555, 555), v(555, 555, 555), v(555, 555, 0), white, MatType.Lambertian);
  addQuad(objects, v(0, 0, 555), v(555, 0, 555), v(555, 555, 555), v(0, 555, 555), white, MatType.Lambertian);
  addQuad(objects, v(0, 0, 0), v(0, 555, 0), v(0, 555, 555), v(0, 0, 555), green, MatType.Lambertian);
  addQuad(objects, v(555, 0, 0), v(555, 0, 555), v(555, 555, 555), v(555, 555, 0), red, MatType.Lambertian);

  const l0 = v(213, 554, 227), l1 = v(343, 554, 227), l2 = v(343, 554, 332), l3 = v(213, 554, 332);
  addQuad(objects, l0, l1, l2, l3, light, MatType.Light);

  // --- 箱の配置 (再計算済み) ---

  // 1. 背の高い箱 (ガラス)
  const tallBoxCenter = v(366, 165, 383);

  const tallBoxGeo = createBox(s(165, 330, 165), glassColor, MatType.Dielectric, 1.5);
  addTransformed(objects, tallBoxGeo, tallBoxCenter, 15);

  // 2. 背の低い箱 (白メタル)
  const shortBoxCenter = v(183, 82.5, 209);

  const shortBoxGeo = createBox(s(165, 165, 165), { x: 0.73, y: 0.73, z: 0.73 }, MatType.Metal, 0.2);
  addTransformed(objects, shortBoxGeo, shortBoxCenter, -18);

  // 3. 青い球
  // TallBoxと同じ中心座標に配置
  const sphereGeo = [new Sphere({ x: 0, y: 0, z: 0 }, (60.0 / S) * 1.0, blueLight, MatType.Light)];
  addTransformed(objects, sphereGeo, tallBoxCenter, 0);

  return {
    camera: {
      // 本家のカメラ位置 (278, 278, -800) に近い比率
      lookfrom: { x: 0, y: 1.0, z: -3.9 },
      lookat: { x: 0, y: 1.0, z: 0 },
      vup: { x: 0, y: 1, z: 0 },
      vfov: 40.0,
      defocusAngle: 0.0,
      focusDist: 2.4
    },
    primitives: objects
  };
}

// --- シーン取得関数 ---
export function getSceneData(name: string): SceneData {
  switch (name) {
    case 'spheres': return getRandomSpheresScene();
    case 'mixed': return getMixedScene();
    case 'special': return getCornellBoxSpecialScene();
    case 'cornell':
    default: return getCornellBoxScene();
  }
}
