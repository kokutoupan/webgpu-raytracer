use crate::mesh::Mesh;
use glam::{Mat3, Vec3, vec3};
use std::f32::consts::PI;

#[derive(Default, Clone)]
pub struct Geometry {
    pub vertices: Vec<f32>,   // [x, y, z, pad]
    pub indices: Vec<u32>,    // [i0, i1, i2]
    pub attributes: Vec<f32>, // [r, g, b, mat_bits, extra, 0, 0, 0]
}

impl Geometry {
    pub fn new() -> Self {
        Self::default()
    }

    // 内部ヘルパー
    fn push_vertex(&mut self, v: Vec3) -> u32 {
        let index = (self.vertices.len() / 4) as u32;
        self.vertices.extend_from_slice(&[v.x, v.y, v.z, 0.0]);
        index
    }
    fn push_attributes(&mut self, color: Vec3, mat_type: u32, extra: f32) {
        let mat_bits = f32::from_bits(mat_type);
        self.attributes
            .extend_from_slice(&[color.x, color.y, color.z, mat_bits, extra, 0.0, 0.0, 0.0]);
    }

    // --- 形状追加メソッド ---

    // 三角形単体
    pub fn add_triangle(
        &mut self,
        v0: Vec3,
        v1: Vec3,
        v2: Vec3,
        color: Vec3,
        mat_type: u32,
        extra: f32,
    ) {
        let i0 = self.push_vertex(v0);
        let i1 = self.push_vertex(v1);
        let i2 = self.push_vertex(v2);
        self.indices.extend_from_slice(&[i0, i1, i2]);
        self.push_attributes(color, mat_type, extra);
    }

    // 球 (テッセレーション)
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
                self.push_vertex(vec3(x, y, z) + center);
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

    // OBJメッシュの結合
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

        for v in &mesh.vertices {
            let tv = (rot_mat * (*v * scale)) + pos;
            self.push_vertex(tv);
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
}
