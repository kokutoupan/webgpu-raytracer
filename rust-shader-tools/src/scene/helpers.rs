// src/scene/helpers.rs
use crate::geometry::Geometry;
use glam::{Vec3, vec2, vec3};
use rand::Rng;

pub fn add_quad(
    geom: &mut Geometry,
    a: Vec3,
    b: Vec3,
    c: Vec3,
    d: Vec3,
    color: Vec3,
    mat_type: u32,
    extra: f32,
) {
    let n = (b - a).cross(d - a).normalize();
    // UV自動割当: a(0,0), b(1,0), c(1,1), d(0,1)
    let i0 = geom.push_vertex(a, n, vec2(0., 0.));
    let i1 = geom.push_vertex(b, n, vec2(1., 0.));
    let i2 = geom.push_vertex(c, n, vec2(1., 1.));
    let i3 = geom.push_vertex(d, n, vec2(0., 1.));

    geom.indices.extend_from_slice(&[i0, i1, i2]);
    geom.push_attributes(color, mat_type, extra);

    geom.indices.extend_from_slice(&[i0, i2, i3]);
    geom.push_attributes(color, mat_type, extra);
}

pub fn create_box(
    geom: &mut Geometry,
    size: Vec3,
    center: Vec3,
    rot_y_deg: f32,
    color: Vec3,
    mat_type: u32,
    extra: f32,
) {
    let rad = rot_y_deg.to_radians();
    let cos_r = rad.cos();
    let sin_r = rad.sin();

    let transform = |p: Vec3| -> Vec3 {
        let x = p.x * cos_r + p.z * sin_r;
        let z = -p.x * sin_r + p.z * cos_r;
        vec3(x, p.y, z) + center
    };

    let dx = vec3(size.x / 2.0, 0.0, 0.0);
    let dy = vec3(0.0, size.y / 2.0, 0.0);
    let dz = vec3(0.0, 0.0, size.z / 2.0);

    // Front
    add_quad(
        geom,
        transform(-dx - dy + dz),
        transform(dx - dy + dz),
        transform(dx + dy + dz),
        transform(-dx + dy + dz),
        color,
        mat_type,
        extra,
    );
    // Back
    add_quad(
        geom,
        transform(dx - dy - dz),
        transform(-dx - dy - dz),
        transform(-dx + dy - dz),
        transform(dx + dy - dz),
        color,
        mat_type,
        extra,
    );
    // Top
    add_quad(
        geom,
        transform(-dx + dy + dz),
        transform(dx + dy + dz),
        transform(dx + dy - dz),
        transform(-dx + dy - dz),
        color,
        mat_type,
        extra,
    );
    // Bottom
    add_quad(
        geom,
        transform(-dx - dy - dz),
        transform(dx - dy - dz),
        transform(dx - dy + dz),
        transform(-dx - dy + dz),
        color,
        mat_type,
        extra,
    );
    // Right
    add_quad(
        geom,
        transform(dx - dy + dz),
        transform(dx - dy - dz),
        transform(dx + dy - dz),
        transform(dx + dy + dz),
        color,
        mat_type,
        extra,
    );
    // Left
    add_quad(
        geom,
        transform(-dx - dy - dz),
        transform(-dx - dy + dz),
        transform(-dx + dy + dz),
        transform(-dx + dy - dz),
        color,
        mat_type,
        extra,
    );
}

// Random helpers
pub fn rnd() -> f32 {
    let mut rng = rand::rng();
    rng.random::<f32>()
}

pub fn rnd_range(min: f32, max: f32) -> f32 {
    let mut rng = rand::rng();
    rng.random_range(min..max)
}

// ※ load_gltf_nodes_skins_anims を loader.rs から呼び出す場合、ここに実装するか、
// loader.rs 内に記述して呼び出す構成にしてください（相互依存回避のため loader.rs に書く方が無難です）。
// ここでは省略します。
