// math.ts

export type Vec3 = { x: number; y: number; z: number };

export const vec3 = {
  create: (x: number, y: number, z: number): Vec3 => ({ x, y, z }),
  sub: (a: Vec3, b: Vec3): Vec3 => ({ x: a.x - b.x, y: a.y - b.y, z: a.z - b.z }),
  add: (a: Vec3, b: Vec3): Vec3 => ({ x: a.x + b.x, y: a.y + b.y, z: a.z + b.z }),
  scale: (v: Vec3, s: number): Vec3 => ({ x: v.x * s, y: v.y * s, z: v.z * s }),
  cross: (a: Vec3, b: Vec3): Vec3 => ({
    x: a.y * b.z - a.z * b.y,
    y: a.z * b.x - a.x * b.z,
    z: a.x * b.y - a.y * b.x,
  }),
  normalize: (v: Vec3): Vec3 => {
    const len = Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    return len === 0 ? { x: 0, y: 0, z: 0 } : { x: v.x / len, y: v.y / len, z: v.z / len };
  },
  len: (v: Vec3): number => Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z),
};

export const rnd = () => Math.random();
export const rndRange = (min: number, max: number) => min + (max - min) * Math.random();
