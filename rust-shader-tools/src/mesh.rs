use crate::primitives::{Primitive, Triangle};
use glam::{Mat3, Vec3, vec3};

pub struct Mesh {
    pub vertices: Vec<Vec3>,
    pub indices: Vec<u32>,
}

impl Mesh {
    pub fn new(obj_text: &str) -> Self {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        for line in obj_text.lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.is_empty() {
                continue;
            }

            match parts[0] {
                "v" => {
                    if parts.len() >= 4 {
                        let x: f32 = parts[1].parse().unwrap_or(0.0);
                        let y: f32 = parts[2].parse().unwrap_or(0.0);
                        let z: f32 = parts[3].parse().unwrap_or(0.0);
                        vertices.push(vec3(x, y, z));
                    }
                }
                "f" => {
                    let mut face_indices = Vec::new();
                    for p in &parts[1..] {
                        let idx_str = p.split('/').next().unwrap_or("0");
                        if let Ok(idx) = idx_str.parse::<i32>() {
                            // OBJは1-based
                            face_indices.push((idx - 1) as u32);
                        }
                    }
                    // 三角形分割 (Fan)
                    if face_indices.len() >= 3 {
                        for i in 1..face_indices.len() - 1 {
                            indices.push(face_indices[0]);
                            indices.push(face_indices[i]);
                            indices.push(face_indices[i + 1]);
                        }
                    }
                }
                _ => {}
            }
        }

        let mut mesh = Self { vertices, indices };
        mesh.normalize(); // 自動正規化
        mesh
    }

    // モデルを原点中心・高さ2.0程度に正規化
    fn normalize(&mut self) {
        if self.vertices.is_empty() {
            return;
        }

        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);

        for v in &self.vertices {
            min = min.min(*v);
            max = max.max(*v);
        }

        let size = max - min;
        let center = (min + max) * 0.5;
        let max_dim = size.x.max(size.y).max(size.z);
        let scale = if max_dim > 0.0 { 2.0 / max_dim } else { 1.0 };

        for v in &mut self.vertices {
            *v = (*v - center) * scale;
        }
    }

    pub fn create_instance(
        &self,
        pos: Vec3,
        scale: f32,
        rot_y_deg: f32,
        color: Vec3,
        mat_type: u32,
        extra: f32,
    ) -> Vec<Primitive> {
        let mut primitives = Vec::new();

        let rad = rot_y_deg.to_radians();
        let rot_mat = Mat3::from_rotation_y(rad);

        // ※ インデックス処理
        // self.indices には頂点インデックスが3つずつ入っている
        for i in (0..self.indices.len()).step_by(3) {
            let idx0 = self.indices[i] as usize;
            let idx1 = self.indices[i + 1] as usize;
            let idx2 = self.indices[i + 2] as usize;

            if idx0 >= self.vertices.len()
                || idx1 >= self.vertices.len()
                || idx2 >= self.vertices.len()
            {
                continue;
            }

            // 変換関数 (Scale -> Rotate -> Translate)
            let transform = |v: Vec3| -> Vec3 {
                let scaled = v * scale;
                let rotated = rot_mat * scaled;
                rotated + pos
            };

            let v0 = transform(self.vertices[idx0]);
            let v1 = transform(self.vertices[idx1]);
            let v2 = transform(self.vertices[idx2]);

            primitives.push(Primitive::Triangle(Triangle {
                v0,
                v1,
                v2,
                color,
                mat_type,
                extra,
            }));
        }

        primitives
    }
}
