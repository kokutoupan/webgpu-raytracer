// src/geometry.rs
use crate::mesh::Mesh;
use glam::{Mat3, Vec3, vec3};
use std::f32::consts::PI;

#[derive(Default, Clone)]
pub struct Geometry {
    pub vertices: Vec<f32>, // [x, y, z, pad]
    pub normals: Vec<f32>,  // [nx, ny, nz, pad]

    // スキニング用 元データ
    pub base_positions: Vec<Vec3>,
    pub base_normals: Vec<Vec3>,

    pub indices: Vec<u32>,
    pub attributes: Vec<f32>,

    pub joints: Vec<u32>,
    pub weights: Vec<f32>,

    // ★追加: このジオメトリが使用するスキンのインデックス
    pub skin_index: Option<usize>,
}

impl Geometry {
    pub fn new() -> Self {
        Self::default()
    }

    fn push_vertex(&mut self, v: Vec3, n: Vec3) -> u32 {
        let index = (self.vertices.len() / 4) as u32;
        self.vertices.extend_from_slice(&[v.x, v.y, v.z, 0.0]);
        self.normals.extend_from_slice(&[n.x, n.y, n.z, 0.0]);

        self.base_positions.push(v);
        self.base_normals.push(n);

        self.joints.extend_from_slice(&[0, 0, 0, 0]);
        self.weights.extend_from_slice(&[0.0, 0.0, 0.0, 0.0]);
        index
    }

    pub fn push_vertex_skinned(&mut self, v: Vec3, n: Vec3, j: [u32; 4], w: [f32; 4]) -> u32 {
        let index = (self.vertices.len() / 4) as u32;
        self.vertices.extend_from_slice(&[v.x, v.y, v.z, 0.0]);
        self.normals.extend_from_slice(&[n.x, n.y, n.z, 0.0]);

        self.base_positions.push(v);
        self.base_normals.push(n);

        self.joints.extend_from_slice(&j);
        self.weights.extend_from_slice(&w);

        index
    }

    pub fn push_attributes(&mut self, color: Vec3, mat_type: u32, extra: f32) {
        let mat_bits = f32::from_bits(mat_type);
        self.attributes
            .extend_from_slice(&[color.x, color.y, color.z, mat_bits, extra, 0.0, 0.0, 0.0]);
    }

    // --- Primitives ---
    pub fn add_triangle(
        &mut self,
        v0: Vec3,
        v1: Vec3,
        v2: Vec3,
        color: Vec3,
        mat_type: u32,
        extra: f32,
    ) {
        let e1 = v1 - v0;
        let e2 = v2 - v0;
        let normal = e1.cross(e2).normalize_or_zero();
        let i0 = self.push_vertex(v0, normal);
        let i1 = self.push_vertex(v1, normal);
        let i2 = self.push_vertex(v2, normal);
        self.indices.extend_from_slice(&[i0, i1, i2]);
        self.push_attributes(color, mat_type, extra);
    }

    pub fn add_sphere(
        &mut self,
        center: Vec3,
        radius: f32,
        color: Vec3,
        mat_type: u32,
        extra: f32,
    ) {
        let sectors = 24;
        let stacks = 12;
        let start_index = (self.vertices.len() / 4) as u32;
        for i in 0..=stacks {
            let stack_angle = PI / 2.0 - PI * (i as f32) / (stacks as f32);
            let xy = radius * stack_angle.cos();
            let z = radius * stack_angle.sin();
            for j in 0..=sectors {
                let sector_angle = 2.0 * PI * (j as f32) / (sectors as f32);
                let x = xy * sector_angle.cos();
                let y = xy * sector_angle.sin();
                let pos = vec3(x, y, z) + center;
                let normal = vec3(x, y, z).normalize();
                self.push_vertex(pos, normal);
            }
        }
        for i in 0..stacks {
            let k1 = start_index + i * (sectors + 1);
            let k2 = k1 + sectors + 1;
            for j in 0..sectors {
                if i != 0 {
                    self.indices
                        .extend_from_slice(&[k1 + j, k2 + j, k1 + j + 1]);
                    self.push_attributes(color, mat_type, extra);
                }
                if i != (stacks - 1) {
                    self.indices
                        .extend_from_slice(&[k1 + j + 1, k2 + j, k2 + j + 1]);
                    self.push_attributes(color, mat_type, extra);
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_mesh_instance(
        &mut self,
        mesh: &Mesh,
        pos: Vec3,
        scale: f32,
        rot_y_deg: f32,
        color: Vec3,
        mat_type: u32,
        extra: f32,
    ) {
        if mesh.vertices.is_empty() {
            return;
        }
        let rad = rot_y_deg.to_radians();
        let rot_mat = Mat3::from_rotation_y(rad);
        let start_offset = (self.vertices.len() / 4) as u32;
        for (i, v) in mesh.vertices.iter().enumerate() {
            let tv = (rot_mat * (*v * scale)) + pos;
            let tn = if i < mesh.normals.len() {
                rot_mat * mesh.normals[i]
            } else {
                vec3(0., 1., 0.)
            };
            self.push_vertex(tv, tn);
        }
        for chunk in mesh.indices.chunks(3) {
            if chunk.len() == 3 {
                self.indices.push(chunk[0] + start_offset);
                self.indices.push(chunk[1] + start_offset);
                self.indices.push(chunk[2] + start_offset);
                self.push_attributes(color, mat_type, extra);
            }
        }
    }

    // src/geometry.rs の impl Geometry 内に追加

    pub fn from_mesh(mesh: &Mesh) -> Self {
        let mut geo = Self::new();
        // デフォルトのマテリアル (灰色、Lambertian)
        let color = vec3(0.8, 0.8, 0.8);
        let mat_type = 0; // LAMBERTIAN
        let extra = 0.0;

        for (i, v) in mesh.vertices.iter().enumerate() {
            let n = if i < mesh.normals.len() {
                mesh.normals[i]
            } else {
                vec3(0., 1., 0.)
            };
            geo.push_vertex(*v, n);
        }

        for chunk in mesh.indices.chunks(3) {
            if chunk.len() == 3 {
                geo.indices.extend_from_slice(chunk);
                geo.push_attributes(color, mat_type, extra);
            }
        }
        geo
    }

    pub fn normalize_scale(&mut self) {
        if self.base_positions.is_empty() {
            return;
        }
        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);

        for p in &self.base_positions {
            min = min.min(*p);
            max = max.max(*p);
        }

        let center = (min + max) * 0.5;
        let extent = max - min;
        let max_dim = extent.max_element();

        if max_dim < 1e-6 {
            return;
        }
        // [-1, 1] の範囲に収まるように正規化
        let scale = 2.0 / max_dim;

        for i in 0..self.base_positions.len() {
            let p = (self.base_positions[i] - center) * scale;
            self.base_positions[i] = p;

            // verticesバッファも同期して更新
            self.vertices[i * 4 + 0] = p.x;
            self.vertices[i * 4 + 1] = p.y;
            self.vertices[i * 4 + 2] = p.z;
        }
    }
}
