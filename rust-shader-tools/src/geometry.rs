// src/geometry.rs
use crate::mesh::Mesh;
use glam::{Mat3, Vec2, Vec3, vec2, vec3};
use std::f32::consts::PI;

#[derive(Default, Clone)]
pub struct Geometry {
    pub vertices: Vec<f32>, // [x, y, z, pad]
    pub normals: Vec<f32>,  // [nx, ny, nz, pad]
    pub uvs: Vec<f32>,      // [u, v] ★追加

    // スキニング用 元データ
    pub base_positions: Vec<Vec3>,
    pub base_normals: Vec<Vec3>,
    pub base_uvs: Vec<Vec2>, // ★追加

    pub indices: Vec<u32>,
    pub attributes: Vec<f32>,

    pub joints: Vec<u32>,
    pub weights: Vec<f32>,

    pub skin_index: Option<usize>,
}

impl Geometry {
    pub fn new() -> Self {
        Self::default()
    }

    // 引数に uv: Vec2 を追加
    pub fn push_vertex(&mut self, v: Vec3, n: Vec3, uv: Vec2) -> u32 {
        self.vertices.extend_from_slice(&[v.x, v.y, v.z, 0.0]);
        self.normals.extend_from_slice(&[n.x, n.y, n.z, 0.0]);
        self.uvs.extend_from_slice(&[uv.x, uv.y]);

        self.base_positions.push(v);
        self.base_normals.push(n);
        self.base_uvs.push(uv);

        self.joints.extend_from_slice(&[0, 0, 0, 0]);
        self.weights.extend_from_slice(&[0.0, 0.0, 0.0, 0.0]);
        (self.vertices.len() / 4 - 1) as u32
    }

    pub fn push_vertex_skinned(
        &mut self,
        v: Vec3,
        n: Vec3,
        uv: Vec2,
        j: [u32; 4],
        w: [f32; 4],
    ) -> u32 {
        self.vertices.extend_from_slice(&[v.x, v.y, v.z, 0.0]);
        self.normals.extend_from_slice(&[n.x, n.y, n.z, 0.0]);
        self.uvs.extend_from_slice(&[uv.x, uv.y]);

        self.base_positions.push(v);
        self.base_normals.push(n);
        self.base_uvs.push(uv);

        self.joints.extend_from_slice(&j);
        self.weights.extend_from_slice(&w);

        (self.vertices.len() / 4 - 1) as u32
    }

    pub fn push_attributes(
        &mut self,
        base_color: Vec3,
        mat_type: u32,
        metallic: f32,
        roughness: f32,
        ior: f32,
        emissive_color: Vec3,
        tex_indices: [f32; 4], // base, met_rough, normal, emissive
        occlusion_tex: f32,
    ) {
        let mat_bits = f32::from_bits(mat_type);
        // Stride 12 (to match 16-word layout with 4 indices)
        // data0: rgba (BaseColor + MatType)
        // data1: metallic, roughness, ior, padding/extra
        // data2: base_tex, met_rough_tex, normal_tex, emissive_tex
        // data3: emissive_color (rgb), occlusion_tex
        self.attributes.extend_from_slice(&[
            base_color.x,
            base_color.y,
            base_color.z,
            mat_bits,
            metallic,
            roughness,
            ior,
            0.0, // extra/padding
            tex_indices[0],
            tex_indices[1],
            tex_indices[2],
            tex_indices[3],
            emissive_color.x,
            emissive_color.y,
            emissive_color.z,
            occlusion_tex,
        ]);
    }

    pub fn from_mesh(mesh: &Mesh) -> Self {
        let mut geo = Geometry::new();

        for (i, v) in mesh.vertices.iter().enumerate() {
            let n = mesh.normals.get(i).copied().unwrap_or(vec3(0., 1., 0.));
            let uv = mesh.uvs.get(i).copied().unwrap_or(vec2(0., 0.));
            geo.push_vertex(*v, n, uv);
        }

        for chunk in mesh.indices.chunks(3) {
            if chunk.len() == 3 {
                geo.indices.extend_from_slice(chunk);
                geo.push_attributes(
                    Vec3::new(1.0, 1.0, 1.0),
                    crate::scene::material::LAMBERTIAN,
                    0.0,
                    1.0,
                    1.5,
                    Vec3::ZERO,
                    [-1.0, -1.0, -1.0, -1.0],
                    -1.0,
                );
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
        let scale = 2.0 / max_dim;

        for i in 0..self.base_positions.len() {
            let p = (self.base_positions[i] - center) * scale;
            self.base_positions[i] = p;

            // verticesバッファも同期
            self.vertices[i * 4 + 0] = p.x;
            self.vertices[i * 4 + 1] = p.y;
            self.vertices[i * 4 + 2] = p.z;
        }
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
        texture_index: f32,
    ) {
        let e1 = v1 - v0;
        let e2 = v2 - v0;
        let normal = e1.cross(e2).normalize_or_zero();

        // 簡易的なUV
        let i0 = self.push_vertex(v0, normal, vec2(0., 0.));
        let i1 = self.push_vertex(v1, normal, vec2(1., 0.));
        let i2 = self.push_vertex(v2, normal, vec2(0., 1.));

        self.indices.extend_from_slice(&[i0, i1, i2]);

        let (metallic, roughness, ior) = match mat_type {
            1 => (1.0, extra, 1.5), // METAL
            2 => (0.0, 0.0, extra), // DIELECTRIC
            _ => (0.0, 1.0, 1.5),   // LAMBERTIAN / LIGHT
        };

        self.push_attributes(
            color,
            mat_type,
            metallic,
            roughness,
            ior,
            Vec3::ZERO,
            [texture_index, -1.0, -1.0, -1.0],
            -1.0,
        );
    }

    pub fn add_sphere(
        &mut self,
        center: Vec3,
        radius: f32,
        color: Vec3,
        mat_type: u32,
        extra: f32,
        texture_index: f32,
    ) {
        let sectors = 24;
        let stacks = 12;
        let start_index = (self.vertices.len() / 4) as u32;
        for i in 0..=stacks {
            let v_coord = i as f32 / stacks as f32;
            let stack_angle = PI / 2.0 - PI * v_coord;
            let xy = radius * stack_angle.cos();
            let z = radius * stack_angle.sin();
            for j in 0..=sectors {
                let u_coord = j as f32 / sectors as f32;
                let sector_angle = 2.0 * PI * u_coord;
                let x = xy * sector_angle.cos();
                let y = xy * sector_angle.sin();
                let pos = vec3(x, y, z) + center;
                let normal = vec3(x, y, z).normalize();
                self.push_vertex(pos, normal, vec2(u_coord, v_coord));
            }
        }
        for i in 0..stacks {
            let k1 = start_index + i * (sectors + 1);
            let k2 = k1 + sectors + 1;
            for j in 0..sectors {
                if i != 0 {
                    self.indices
                        .extend_from_slice(&[k1 + j, k2 + j, k1 + j + 1]);
                    let (metallic, roughness, ior) = match mat_type {
                        1 => (1.0, extra, 1.5),
                        2 => (0.0, 0.0, extra),
                        _ => (0.0, 1.0, 1.5),
                    };
                    self.push_attributes(
                        color,
                        mat_type,
                        metallic,
                        roughness,
                        ior,
                        Vec3::ZERO,
                        [texture_index, -1.0, -1.0, -1.0],
                        -1.0,
                    );
                }
                if i != (stacks - 1) {
                    self.indices
                        .extend_from_slice(&[k1 + j + 1, k2 + j, k2 + j + 1]);
                    let (metallic, roughness, ior) = match mat_type {
                        1 => (1.0, extra, 1.5),
                        2 => (0.0, 0.0, extra),
                        _ => (0.0, 1.0, 1.5),
                    };
                    self.push_attributes(
                        color,
                        mat_type,
                        metallic,
                        roughness,
                        ior,
                        Vec3::ZERO,
                        [texture_index, -1.0, -1.0, -1.0],
                        -1.0,
                    );
                }
            }
        }
    }

    pub fn add_mesh_instance(
        &mut self,
        mesh: &Mesh,
        pos: Vec3,
        scale: f32,
        rot_y_deg: f32,
        color: Vec3,
        mat_type: u32,
        extra: f32,
        texture_index: f32,
    ) {
        if mesh.vertices.is_empty() {
            return;
        }
        let rad = rot_y_deg.to_radians();
        let rot_mat = Mat3::from_rotation_y(rad);
        let start_offset = (self.vertices.len() / 4) as u32;
        for (i, v) in mesh.vertices.iter().enumerate() {
            let tv = (rot_mat * (*v * scale)) + pos;
            let tn = mesh
                .normals
                .get(i)
                .map(|n| rot_mat * *n)
                .unwrap_or(vec3(0., 1., 0.));
            let uv = mesh.uvs.get(i).copied().unwrap_or(vec2(0., 0.));
            self.push_vertex(tv, tn, uv);
        }
        for chunk in mesh.indices.chunks(3) {
            if chunk.len() == 3 {
                self.indices.push(chunk[0] + start_offset);
                self.indices.push(chunk[1] + start_offset);
                self.indices.push(chunk[2] + start_offset);
                let (metallic, roughness, ior) = match mat_type {
                    1 => (1.0, extra, 1.5),
                    2 => (0.0, 0.0, extra),
                    _ => (0.0, 1.0, 1.5),
                };
                self.push_attributes(
                    color,
                    mat_type,
                    metallic,
                    roughness,
                    ior,
                    Vec3::ZERO,
                    [texture_index, -1.0, -1.0, -1.0],
                    -1.0,
                );
            }
        }
    }
}
