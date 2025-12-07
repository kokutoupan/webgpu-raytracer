import { type Vec3 } from './math';

// GPUに送るデータサイズ (64 bytes = 16 floats)
export const PRIMITIVE_STRIDE = 16;

// AABB型
export interface AABB {
  min: Vec3;
  max: Vec3;
  center: Vec3;
}

// 共通インターフェース
export interface IHittable {
  getAABB(): AABB;
  // 引数のバッファの offset 位置にデータを書き込む
  pack(buffer: Float32Array, offset: number): void;

  clone(): IHittable;
  translate(offset: Vec3): void;
  rotateY(angle: number): void;
}


// --- ヘルパー: ベクトル演算 (math.tsを使ってもいいが、ここで簡易定義して高速化) ---
function add(v: Vec3, d: Vec3): Vec3 { return { x: v.x + d.x, y: v.y + d.y, z: v.z + d.z }; }
function rotY(v: Vec3, cos: number, sin: number): Vec3 {
  return {
    x: v.x * cos + v.z * sin,
    y: v.y,
    z: -v.x * sin + v.z * cos
  };
}

// ==========================================
//   Sphere Class
// ==========================================
export class Sphere implements IHittable {
  center: Vec3;
  radius: number;
  color: Vec3;
  matType: number;
  extra: number;

  constructor(
    center: Vec3,
    radius: number,
    color: Vec3,
    matType: number,
    extra: number = 0.0
  ) {
    this.center = { ...center }; // コピーして持つ
    this.radius = radius;
    this.color = { ...color };
    this.matType = matType;
    this.extra = extra;
  }

  getAABB(): AABB {
    const { x, y, z } = this.center;
    const r = this.radius;
    return {
      min: { x: x - r, y: y - r, z: z - r },
      max: { x: x + r, y: y + r, z: z + r },
      center: this.center
    };
  }
  pack(buffer: Float32Array, offset: number): void {
    const i = offset;
    const { x, y, z } = this.center;
    // ★修正: r,g,b ではなく x,y,z を使う
    const r = this.color.x;
    const g = this.color.y;
    const b = this.color.z;

    // Data0: Center(xyz), Radius(w)
    buffer[i + 0] = x; buffer[i + 1] = y; buffer[i + 2] = z; buffer[i + 3] = this.radius;
    // Data1: Unused(xyz), MatType(w)
    buffer[i + 4] = 0; buffer[i + 5] = 0; buffer[i + 6] = 0; buffer[i + 7] = this.matType;
    // Data2: Unused(xyz), ObjType(w) -> 1.0 = Sphere
    buffer[i + 8] = 0; buffer[i + 9] = 0; buffer[i + 10] = 0; buffer[i + 11] = 1.0;
    // Data3: Color(xyz), Extra(w)
    buffer[i + 12] = r; buffer[i + 13] = g; buffer[i + 14] = b; buffer[i + 15] = this.extra;
  }

  // ★追加: 複製
  clone(): Sphere {
    return new Sphere(this.center, this.radius, this.color, this.matType, this.extra);
  }

  // ★追加: 移動
  translate(offset: Vec3): void {
    this.center = add(this.center, offset);
  }

  // ★追加: Y軸回転 (原点中心)
  rotateY(angle: number): void {
    const rad = (angle * Math.PI) / 180.0;
    const c = Math.cos(rad);
    const s = Math.sin(rad);
    this.center = rotY(this.center, c, s);
  }

}

// ==========================================
//   Triangle Class
// ==========================================
export class Triangle implements IHittable {
  v0: Vec3;
  v1: Vec3;
  v2: Vec3;
  color: Vec3;
  matType: number;
  extra: number;

  constructor(
    v0: Vec3,
    v1: Vec3,
    v2: Vec3,
    color: Vec3,
    matType: number,
    extra: number = 0.0
  ) {
    this.v0 = { ...v0 };
    this.v1 = { ...v1 };
    this.v2 = { ...v2 };
    this.color = { ...color };
    this.matType = matType;
    this.extra = extra;
  }

  getAABB(): AABB {
    const min = {
      x: Math.min(this.v0.x, this.v1.x, this.v2.x),
      y: Math.min(this.v0.y, this.v1.y, this.v2.y),
      z: Math.min(this.v0.z, this.v1.z, this.v2.z),
    };
    const max = {
      x: Math.max(this.v0.x, this.v1.x, this.v2.x),
      y: Math.max(this.v0.y, this.v1.y, this.v2.y),
      z: Math.max(this.v0.z, this.v1.z, this.v2.z),
    };

    // 厚みゼロ対策
    const EPSILON = 0.001;
    if (max.x - min.x < EPSILON) { min.x -= EPSILON; max.x += EPSILON; }
    if (max.y - min.y < EPSILON) { min.y -= EPSILON; max.y += EPSILON; }
    if (max.z - min.z < EPSILON) { min.z -= EPSILON; max.z += EPSILON; }

    const center = {
      x: (min.x + max.x) * 0.5,
      y: (min.y + max.y) * 0.5,
      z: (min.z + max.z) * 0.5,
    };

    return { min, max, center };
  }

  pack(buffer: Float32Array, offset: number): void {
    const i = offset;
    const { x, y, z } = this.color as any;

    // Data0: V0(xyz), Unused(w)
    buffer[i + 0] = this.v0.x; buffer[i + 1] = this.v0.y; buffer[i + 2] = this.v0.z; buffer[i + 3] = 0;
    // Data1: V1(xyz), MatType(w)
    buffer[i + 4] = this.v1.x; buffer[i + 5] = this.v1.y; buffer[i + 6] = this.v1.z; buffer[i + 7] = this.matType;
    // Data2: V2(xyz), ObjType(w) -> 2.0 = Triangle
    buffer[i + 8] = this.v2.x; buffer[i + 9] = this.v2.y; buffer[i + 10] = this.v2.z; buffer[i + 11] = 2.0;
    // Data3: Color(xyz), Extra(w)
    buffer[i + 12] = x; buffer[i + 13] = y; buffer[i + 14] = z; buffer[i + 15] = this.extra;
  }

  // ★追加: 複製
  clone(): Triangle {
    return new Triangle(this.v0, this.v1, this.v2, this.color, this.matType, this.extra);
  }

  // ★追加: 移動
  translate(offset: Vec3): void {
    this.v0 = add(this.v0, offset);
    this.v1 = add(this.v1, offset);
    this.v2 = add(this.v2, offset);
  }

  // ★追加: Y軸回転 (原点中心)
  rotateY(angle: number): void {
    const rad = (angle * Math.PI) / 180.0;
    const c = Math.cos(rad);
    const s = Math.sin(rad);
    this.v0 = rotY(this.v0, c, s);
    this.v1 = rotY(this.v1, c, s);
    this.v2 = rotY(this.v2, c, s);
  }
}
