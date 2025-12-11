// src/scene/helpers.rs
use crate::geometry::Geometry;
use glam::{Vec3, vec3};
use rand::Rng;

// 乱数ヘルパー
pub fn rnd() -> f32 {
    rand::rng().random()
}

pub fn rnd_range(min: f32, max: f32) -> f32 {
    rand::rng().random_range(min..max)
}

// 四角形(Quad)を追加
// 2つの三角形としてGeometryに追加します
#[allow(clippy::too_many_arguments)]
pub fn add_quad(
    geom: &mut Geometry,
    v0: Vec3,
    v1: Vec3,
    v2: Vec3,
    v3: Vec3,
    col: Vec3,
    mat: u32,
    extra: f32,
) {
    geom.add_triangle(v0, v1, v2, col, mat, extra);
    geom.add_triangle(v0, v2, v3, col, mat, extra);
}

// 箱(Box)を作成して追加
// 位置(pos)とY軸回転(rot_y)を適用します
pub fn create_box(
    geom: &mut Geometry,
    size: Vec3,
    pos: Vec3,
    rot_y: f32,
    col: Vec3,
    mat: u32,
    extra: f32,
) {
    let h = size * 0.5;
    let (hx, hy, hz) = (h.x, h.y, h.z);

    // ローカル座標での頂点
    let raw_pts = [
        vec3(-hx, -hy, hz),
        vec3(hx, -hy, hz),
        vec3(hx, hy, hz),
        vec3(-hx, hy, hz),
        vec3(-hx, -hy, -hz),
        vec3(hx, -hy, -hz),
        vec3(hx, hy, -hz),
        vec3(-hx, hy, -hz),
    ];

    // 変換処理
    let rad = rot_y.to_radians();
    let sin = rad.sin();
    let cos = rad.cos();

    // 回転 + 移動
    let transform = |p: Vec3| -> Vec3 {
        let rx = p.x * cos + p.z * sin;
        let rz = -p.x * sin + p.z * cos;
        vec3(rx, p.y, rz) + pos
    };

    let p: Vec<Vec3> = raw_pts.iter().map(|&v| transform(v)).collect();

    // 6面分のQuadを追加
    add_quad(geom, p[0], p[1], p[2], p[3], col, mat, extra); // Front
    add_quad(geom, p[5], p[4], p[7], p[6], col, mat, extra); // Back
    add_quad(geom, p[4], p[0], p[3], p[7], col, mat, extra); // Left
    add_quad(geom, p[1], p[5], p[6], p[2], col, mat, extra); // Right
    add_quad(geom, p[3], p[2], p[6], p[7], col, mat, extra); // Top
    add_quad(geom, p[0], p[4], p[5], p[1], col, mat, extra); // Bottom
}
