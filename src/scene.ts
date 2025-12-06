// scene.ts
import { vec3, rnd, rndRange, type Vec3 } from './math';

// --- マテリアル定義 ---
export const MatType = {
  Lambertian: 0.0,
  Metal: 1.0,
  Dielectric: 2.0,
};

// --- カメラデータ生成 ---
export function createCameraData(
  lookfrom: Vec3,
  lookat: Vec3,
  vup: Vec3,
  vfov: number,
  aspectRatio: number,
  defocusAngle: number,
  focusDist: number
): Float32Array<ArrayBuffer> {
  const theta = (vfov * Math.PI) / 180.0;
  const h = Math.tan(theta / 2.0);
  const viewportHeight = 2.0 * h * focusDist;
  const viewportWidth = viewportHeight * aspectRatio;

  const w = vec3.normalize(vec3.sub(lookfrom, lookat));
  const u = vec3.normalize(vec3.cross(vup, w));
  const v = vec3.cross(w, u);

  const origin = lookfrom;
  const horizontal = vec3.scale(u, viewportWidth);
  const vertical = vec3.scale(v, viewportHeight);

  const lowerLeftCorner = vec3.sub(
    vec3.sub(vec3.sub(origin, vec3.scale(horizontal, 0.5)), vec3.scale(vertical, 0.5)),
    vec3.scale(w, focusDist)
  );

  const lensRadius = focusDist * Math.tan((defocusAngle * Math.PI) / 360.0);

  // alignment: 16byte (4 floats)
  return new Float32Array([
    origin.x, origin.y, origin.z, lensRadius,          // offset 0
    lowerLeftCorner.x, lowerLeftCorner.y, lowerLeftCorner.z, 0.0, // offset 16
    horizontal.x, horizontal.y, horizontal.z, 0.0,     // offset 32
    vertical.x, vertical.y, vertical.z, 0.0,           // offset 48
    u.x, u.y, u.z, 0.0,                                // offset 64
    v.x, v.y, v.z, 0.0                                 // offset 80
  ]);
}

// --- 球データ生成ヘルパー ---
function createSphere(center: Vec3, radius: number, color: Vec3, matType: number, extra: number = 0.0): number[] {
  return [
    center.x, center.y, center.z, radius,    // vec4 data1
    color.x, color.y, color.z, matType,      // vec4 data2 (JSの{r,g,b}はVec3型として扱う)
    extra, 0.0, 0.0, 0.0                     // vec4 data3
  ];
}

export function makeSpheres(): number[][] {
  const spheresList: number[][] = [];

  // 1. 地面
  spheresList.push(createSphere({ x: 0.0, y: -1000.0, z: 0.0 }, 1000.0, { x: 0.5, y: 0.5, z: 0.5 } as any, MatType.Lambertian));

  // 2. 小球
  for (let a = -5; a < 5; a++) {
    for (let b = -5; b < 5; b++) {
      const chooseMat = rnd();
      const center = { x: a + 0.9 * rnd(), y: 0.2, z: b + 0.9 * rnd() };
      const dist = vec3.len(vec3.sub(center, { x: 4.0, y: 0.2, z: 0.0 }));

      if (dist > 0.9) {
        if (chooseMat < 0.8) { // Lambertian
          const r = rnd() * rnd();
          const g = rnd() * rnd();
          const b = rnd() * rnd();
          spheresList.push(createSphere(center, 0.2, { x: r, y: g, z: b } as any, MatType.Lambertian));
        } else if (chooseMat < 0.95) { // Metal
          const r = rndRange(0.5, 1.0);
          const g = rndRange(0.5, 1.0);
          const b = rndRange(0.5, 1.0);
          const fuzz = rndRange(0.0, 0.5);
          spheresList.push(createSphere(center, 0.2, { x: r, y: g, z: b } as any, MatType.Metal, fuzz));
        } else { // Glass
          spheresList.push(createSphere(center, 0.2, { x: 1, y: 1, z: 1 } as any, MatType.Dielectric, 1.5));
        }
      }
    }
  }

  // 3. 大球
  spheresList.push(createSphere({ x: 0, y: 1, z: 0 }, 1.0, { x: 1, y: 1, z: 1 } as any, MatType.Dielectric, 1.5));
  spheresList.push(createSphere({ x: -4, y: 1, z: 0 }, 1.0, { x: 0.4, y: 0.2, z: 0.1 } as any, MatType.Lambertian));
  spheresList.push(createSphere({ x: 4, y: 1, z: 0 }, 1.0, { x: 0.7, y: 0.6, z: 0.5 } as any, MatType.Metal, 0.0));

  return spheresList;
}


// 三角形データ生成 (Packed)
// extra のデフォルト値を 0.0 に設定
function createTriangle(v0: Vec3, v1: Vec3, v2: Vec3, col: Vec3, mat: number, extra: number = 0.0): number[] {
  return [
    v0.x, v0.y, v0.z, extra, // ★v0のw成分に extra を詰める
    v1.x, v1.y, v1.z, 0,     // pad
    v2.x, v2.y, v2.z, 0,     // pad
    col.x, col.y, col.z, mat // mat_type
  ];
}

// addQuad も extra を受け取れるように拡張
function addQuad(list: number[], v0: Vec3, v1: Vec3, v2: Vec3, v3: Vec3, col: Vec3, mat: number, extra: number = 0.0) {
  list.push(...createTriangle(v0, v1, v2, col, mat, extra));
  list.push(...createTriangle(v0, v2, v3, col, mat, extra));
}

// コーネルボックス生成
export function makeCornellBox(): Float32Array<ArrayBuffer> {
  const triangles: number[] = [];

  // 色定義
  const white = { x: 0.73, y: 0.73, z: 0.73 };
  const red = { x: 0.65, y: 0.05, z: 0.05 };
  const green = { x: 0.12, y: 0.45, z: 0.15 };
  const light = { x: 15.0, y: 15.0, z: 15.0 }; // 強烈な光

  const S = 555;
  const v = (x: number, y: number, z: number) => ({ x: (x / S) * 2 - 1, y: (y / S) * 2, z: (z / S) * 2 - 1 });

  // 壁 (Lambertian なので extra=0)
  addQuad(triangles, v(0, 0, 0), v(555, 0, 0), v(555, 0, 555), v(0, 0, 555), white, MatType.Lambertian);         // 床
  addQuad(triangles, v(0, 555, 0), v(0, 555, 555), v(555, 555, 555), v(555, 555, 0), white, MatType.Lambertian); // 天井
  addQuad(triangles, v(0, 0, 555), v(555, 0, 555), v(555, 555, 555), v(0, 555, 555), white, MatType.Lambertian); // 奥壁
  addQuad(triangles, v(0, 0, 0), v(0, 555, 0), v(0, 555, 555), v(0, 0, 555), green, MatType.Lambertian);         // 右壁
  addQuad(triangles, v(555, 0, 0), v(555, 0, 555), v(555, 555, 555), v(555, 555, 0), red, MatType.Lambertian);   // 左壁

  // ライト (extra不要)
  const l0 = v(213, 554, 227), l1 = v(343, 554, 227), l2 = v(343, 554, 332), l3 = v(213, 554, 332);
  addQuad(triangles, l0, l3, l2, l1, light, 3); // MatType=3 (Light)

  // ★箱1 (背の高い箱): アルミっぽい金属 (Metal, fuzz=0.1)
  // 座標変換が少し面倒なので、簡易的に回転なしの箱として配置例
  // (実際は回転させるために頂点計算が必要ですが、まずは配置)
  const box1_col = { x: 0.8, y: 0.85, z: 0.88 };
  // addCube 的な関数を作るか、Quadを6回呼ぶ
  // ここでは例として「鏡の壁」を奥に置いてみる
  // addQuad(triangles, v(100,0,400), v(455,0,400), v(455,300,400), v(100,300,400), box1_col, MatType.Metal, 0.0);

  return new Float32Array(triangles);
}
