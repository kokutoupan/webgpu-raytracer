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

// --- 回転付きの箱を追加するヘルパー ---
function addRotatedBox(
  list: number[],
  center: Vec3,
  size: Vec3,
  angleY: number, // 度数法
  col: Vec3,
  mat: number,
  extra: number = 0.0
) {
  const rad = (angleY * Math.PI) / 180.0;
  const cosA = Math.cos(rad);
  const sinA = Math.sin(rad);

  // Y軸回転関数
  const rotate = (p: Vec3): Vec3 => ({
    x: p.x * cosA + p.z * sinA,
    y: p.y,
    z: -p.x * sinA + p.z * cosA
  });

  // 箱の8頂点を計算（原点中心で作ってから回転・移動）
  const hx = size.x / 2;
  const hy = size.y / 2;
  const hz = size.z / 2;

  // 頂点ローカル座標
  const p0 = { x: -hx, y: -hy, z: hz }; // 前・左・下
  const p1 = { x: hx, y: -hy, z: hz }; // 前・右・下
  const p2 = { x: hx, y: hy, z: hz }; // 前・右・上
  const p3 = { x: -hx, y: hy, z: hz }; // 前・左・上
  const p4 = { x: -hx, y: -hy, z: -hz }; // 奥・左・下
  const p5 = { x: hx, y: -hy, z: -hz }; // 奥・右・下
  const p6 = { x: hx, y: hy, z: -hz }; // 奥・右・上
  const p7 = { x: -hx, y: hy, z: -hz }; // 奥・左・上

  // 回転 -> 移動
  const transform = (p: Vec3) => {
    const rot = rotate(p);
    return { x: rot.x + center.x, y: rot.y + center.y, z: rot.z + center.z };
  };

  const v0 = transform(p0), v1 = transform(p1), v2 = transform(p2), v3 = transform(p3);
  const v4 = transform(p4), v5 = transform(p5), v6 = transform(p6), v7 = transform(p7);

  // 6面を追加 (Quad = 2 Triangles)
  addQuad(list, v0, v1, v2, v3, col, mat, extra); // 前
  addQuad(list, v5, v4, v7, v6, col, mat, extra); // 奥
  addQuad(list, v4, v0, v3, v7, col, mat, extra); // 左
  addQuad(list, v1, v5, v6, v2, col, mat, extra); // 右
  addQuad(list, v3, v2, v6, v7, col, mat, extra); // 上
  addQuad(list, v0, v4, v5, v1, col, mat, extra); // 下
}

// コーネルボックス生成
export function makeCornellBox(): Float32Array<ArrayBuffer> {
  const triangles: number[] = [];

  // 色定義
  const white = { x: 0.73, y: 0.73, z: 0.73 };
  const red = { x: 0.65, y: 0.05, z: 0.05 };
  const green = { x: 0.12, y: 0.45, z: 0.15 };
  const light = { x: 15.0, y: 15.0, z: 15.0 }; // 強烈な光
  const toumei = { x: 1.0, y: 1.0, z: 1.0 };

  const S = 555;
  const v = (x: number, y: number, z: number) => ({ x: (x / S) * 2 - 1, y: (y / S) * 2, z: (z / S) * 2 - 1 });
  const s = (x: number, y: number, z: number) => ({ x: (x / S) * 2, y: (y / S) * 2, z: (z / S) * 2 });

  // 壁 (Lambertian なので extra=0)
  addQuad(triangles, v(0, 0, 0), v(555, 0, 0), v(555, 0, 555), v(0, 0, 555), white, MatType.Lambertian);         // 床
  addQuad(triangles, v(0, 555, 0), v(0, 555, 555), v(555, 555, 555), v(555, 555, 0), white, MatType.Lambertian); // 天井
  addQuad(triangles, v(0, 0, 555), v(555, 0, 555), v(555, 555, 555), v(0, 555, 555), white, MatType.Lambertian); // 奥壁
  addQuad(triangles, v(0, 0, 0), v(0, 555, 0), v(0, 555, 555), v(0, 0, 555), green, MatType.Lambertian);         // 右壁
  addQuad(triangles, v(555, 0, 0), v(555, 0, 555), v(555, 555, 555), v(555, 555, 0), red, MatType.Lambertian);   // 左壁

  // ライト (extra不要)
  const l0 = v(213, 554, 227), l1 = v(343, 554, 227), l2 = v(343, 554, 332), l3 = v(213, 554, 332);
  addQuad(triangles, l0, l3, l2, l1, light, 3); // MatType=3 (Light)

  // ★箱1 (Tall Box)
  // 位置: (265, 165, 296), サイズ: (165, 330, 165), 回転: -15度
  addRotatedBox(
    triangles,
    v(265 + 165 / 2 - 50, 165, 296 + 165 / 2), // 中心座標 (適当に調整)
    s(165, 330, 165), // サイズ
    -15, // 回転
    white,
    MatType.Metal
  );

  // ★箱2 (Short Box)
  // 位置: (130, 82.5, 65), サイズ: (165, 165, 165), 回転: 18度
  addRotatedBox(
    triangles,
    v(130 + 165 / 2 + 20, 82.5, 65 + 165 / 2),
    s(165, 165, 165),
    18,
    toumei,
    MatType.Dielectric, // 試しに Metal とかにしても面白いです
    1.5
  );

  return new Float32Array(triangles);
}
