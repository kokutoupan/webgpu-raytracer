// use wasm_bindgen::prelude::*;
use glam::{Vec3, vec3}; // glamからVec3とコンストラクタ(vec3)を使う

pub const PRIMITIVE_STRIDE: usize = 16;
// --- AABB ---
#[derive(Clone, Copy, Debug)]
pub struct AABB {
    pub min: Vec3,
    pub max: Vec3,
}

impl Default for AABB {
    fn default() -> Self {
        Self::empty()
    }
}

impl AABB {
    pub fn empty() -> Self {
        Self {
            // INFINITY も glam に定数はありますが、f32::INFINITY でOK
            min: Vec3::splat(f32::INFINITY),     // 全要素を∞に
            max: Vec3::splat(f32::NEG_INFINITY), // 全要素を-∞に
        }
    }

    pub fn grow(&mut self, p: Vec3) {
        // glam の min/max 関数で一発です
        self.min = self.min.min(p);
        self.max = self.max.max(p);
    }

    // AABB同士の結合も簡単に書けます
    pub fn union(&self, other: &AABB) -> AABB {
        AABB {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    // 表面積 (SAH用)
    pub fn area(&self) -> f32 {
        let d = self.max - self.min;
        // 負のサイズチェック（念の為）
        if d.x < 0.0 || d.y < 0.0 || d.z < 0.0 {
            0.0
        } else {
            2.0 * (d.x * d.y + d.y * d.z + d.z * d.x)
        }
    }

    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }
}

// --- Shapes ---

#[derive(Clone, Copy)]
pub struct Sphere {
    pub center: Vec3,
    pub radius: f32,
    pub color: Vec3,
    pub mat_type: u32,
    pub extra: f32,
}

#[derive(Clone, Copy)]
pub struct Triangle {
    pub v0: Vec3,
    pub v1: Vec3,
    pub v2: Vec3,
    pub color: Vec3,
    pub mat_type: u32,
    pub extra: f32,
}

#[derive(Clone, Copy)]
pub enum Primitive {
    Sphere(Sphere),
    Triangle(Triangle),
}

impl Primitive {
    pub fn aabb(&self) -> AABB {
        match self {
            Primitive::Sphere(s) => {
                let r = Vec3::splat(s.radius);
                AABB {
                    min: s.center - r,
                    max: s.center + r,
                }
            }
            Primitive::Triangle(t) => {
                let min = t.v0.min(t.v1).min(t.v2);
                let max = t.v0.max(t.v1).max(t.v2);
                // 厚みゼロ対策（AABBが潰れるのを防ぐ）
                const EPSILON: f32 = 0.001;
                let size = max - min;
                let padding = vec3(
                    if size.x < EPSILON { EPSILON } else { 0.0 },
                    if size.y < EPSILON { EPSILON } else { 0.0 },
                    if size.z < EPSILON { EPSILON } else { 0.0 },
                );

                AABB {
                    min: min - padding * 0.5,
                    max: max + padding * 0.5, // 厳密には片側だけでいいですが簡易的に
                }
            }
        }
    }

    pub fn translate(&mut self, offset: Vec3) {
        match self {
            Primitive::Sphere(s) => s.center += offset,
            Primitive::Triangle(t) => {
                t.v0 += offset;
                t.v1 += offset;
                t.v2 += offset;
            }
        }
    }

    pub fn rotate_y(&mut self, angle_deg: f32) {
        let rad = angle_deg.to_radians();
        let sin = rad.sin();
        let cos = rad.cos();

        let rot = |v: &mut Vec3| {
            let x = v.x * cos + v.z * sin;
            let z = -v.x * sin + v.z * cos;
            v.x = x;
            v.z = z;
        };

        match self {
            Primitive::Sphere(s) => rot(&mut s.center),
            Primitive::Triangle(t) => {
                rot(&mut t.v0);
                rot(&mut t.v1);
                rot(&mut t.v2);
            }
        }
    }

    // パック処理
    pub fn pack(&self) -> [f32; 16] {
        let mut data = [0.0; 16];
        match self {
            Primitive::Sphere(s) => {
                // data0: center(xyz), radius(w)
                data[0] = s.center.x;
                data[1] = s.center.y;
                data[2] = s.center.z;
                data[3] = s.radius;

                data[4] = 0.0;
                data[5] = 0.0;
                data[6] = 0.0;
                data[7] = s.mat_type as f32;

                data[8] = 0.0;
                data[9] = 0.0;
                data[10] = 0.0;
                data[11] = 1.0;

                // data3: color(xyz), extra(w)
                data[12] = s.color.x;
                data[13] = s.color.y;
                data[14] = s.color.z;
                data[15] = s.extra;
            }
            // Triangleも同様...
            Primitive::Triangle(tri) => {
                data[0] = tri.v0.x;
                data[1] = tri.v0.y;
                data[2] = tri.v0.z;
                data[3] = 0.0;

                data[4] = tri.v1.x;
                data[5] = tri.v1.y;
                data[6] = tri.v1.z;
                data[7] = tri.mat_type as f32;

                data[8] = tri.v2.x;
                data[9] = tri.v2.y;
                data[10] = tri.v2.z;
                data[11] = 2.0;

                data[12] = tri.color.x;
                data[13] = tri.color.y;
                data[14] = tri.color.z;
                data[15] = tri.extra;
            }
        }
        data
    }
}
