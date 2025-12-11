use crate::primitives::{Primitive, Triangle};
use glam::{Vec3, vec3};
use rand::Rng;

// ランダム関数
pub fn rnd() -> f32 {
    rand::rng().random()
}

pub fn rnd_range(min: f32, max: f32) -> f32 {
    rand::rng().random_range(min..max)
}

// プリミティブ追加ヘルパー
#[allow(clippy::too_many_arguments)]
pub fn add_quad(
    list: &mut Vec<Primitive>,
    v0: Vec3,
    v1: Vec3,
    v2: Vec3,
    v3: Vec3,
    col: Vec3,
    mat: u32,
    extra: f32,
) {
    list.push(Primitive::Triangle(Triangle {
        v0,
        v1,
        v2,
        color: col,
        mat_type: mat,
        extra,
    }));
    list.push(Primitive::Triangle(Triangle {
        v0,
        v1: v2,
        v2: v3,
        color: col,
        mat_type: mat,
        extra,
    }));
}

pub fn create_box(size: Vec3, col: Vec3, mat: u32, extra: f32) -> Vec<Primitive> {
    let mut list = Vec::new();
    let h = size * 0.5;
    let (hx, hy, hz) = (h.x, h.y, h.z);

    let p0 = vec3(-hx, -hy, hz);
    let p1 = vec3(hx, -hy, hz);
    let p2 = vec3(hx, hy, hz);
    let p3 = vec3(-hx, hy, hz);
    let p4 = vec3(-hx, -hy, -hz);
    let p5 = vec3(hx, -hy, -hz);
    let p6 = vec3(hx, hy, -hz);
    let p7 = vec3(-hx, hy, -hz);

    add_quad(&mut list, p0, p1, p2, p3, col, mat, extra); // Front
    add_quad(&mut list, p5, p4, p7, p6, col, mat, extra); // Back
    add_quad(&mut list, p4, p0, p3, p7, col, mat, extra); // Left
    add_quad(&mut list, p1, p5, p6, p2, col, mat, extra); // Right
    add_quad(&mut list, p3, p2, p6, p7, col, mat, extra); // Top
    add_quad(&mut list, p0, p4, p5, p1, col, mat, extra); // Bottom
    list
}

pub fn add_transformed(
    target: &mut Vec<Primitive>,
    source: &[Primitive],
    pos: Vec3,
    rot_y_deg: f32,
) {
    for prim in source {
        let mut new_prim = *prim;
        if rot_y_deg != 0.0 {
            new_prim.rotate_y(rot_y_deg);
        }
        new_prim.translate(pos);
        target.push(new_prim);
    }
}
