// src/mesh.rs
use glam::{Vec3, vec3};
use std::collections::HashMap;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct VertexKey {
    v_idx: usize,
    vn_idx: Option<usize>,
}

pub struct Mesh {
    pub vertices: Vec<Vec3>,
    pub normals: Vec<Vec3>, // 頂点と1:1対応する法線
    pub indices: Vec<u32>,
}

impl Mesh {
    pub fn new(obj_text: &str) -> Self {
        let mut raw_positions = Vec::new();
        let mut raw_normals = Vec::new();

        // 最終的なメッシュデータ
        let mut vertices = Vec::new();
        let mut normals = Vec::new();
        let mut indices = Vec::new();

        // 重複頂点除去用のマップ
        let mut unique_map: HashMap<VertexKey, u32> = HashMap::new();

        for line in obj_text.lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.is_empty() {
                continue;
            }

            match parts[0] {
                "v" => {
                    if parts.len() >= 4 {
                        let x = parts[1].parse().unwrap_or(0.);
                        let y = parts[2].parse().unwrap_or(0.);
                        let z = parts[3].parse().unwrap_or(0.);
                        raw_positions.push(vec3(x, y, z));
                    }
                }
                "vn" => {
                    if parts.len() >= 4 {
                        let x = parts[1].parse().unwrap_or(0.);
                        let y = parts[2].parse().unwrap_or(0.);
                        let z = parts[3].parse().unwrap_or(0.);
                        raw_normals.push(vec3(x, y, z).normalize_or_zero());
                    }
                }
                "f" => {
                    // 多角形対応 (Triangle Fan)
                    let mut face_indices = Vec::new();

                    for p in &parts[1..] {
                        let segs: Vec<&str> = p.split('/').collect();
                        // 1. 位置インデックス
                        let v_idx_raw = segs[0].parse::<i32>().unwrap_or(0);
                        let v_idx = if v_idx_raw > 0 {
                            v_idx_raw - 1
                        } else {
                            raw_positions.len() as i32 + v_idx_raw
                        } as usize;

                        // 3. 法線インデックス (v/vt/vn または v//vn)
                        let vn_idx = if segs.len() >= 3 && !segs[2].is_empty() {
                            let vn_raw = segs[2].parse::<i32>().unwrap_or(0);
                            Some(if vn_raw > 0 {
                                vn_raw - 1
                            } else {
                                raw_normals.len() as i32 + vn_raw
                            } as usize)
                        } else {
                            None
                        };

                        let key = VertexKey { v_idx, vn_idx };

                        // すでに登録済みかチェック
                        let final_idx = if let Some(&idx) = unique_map.get(&key) {
                            idx
                        } else {
                            // 新しい頂点を作成
                            let idx = vertices.len() as u32;
                            let pos = raw_positions[v_idx];

                            // 法線がない場合は仮で(0,1,0)を入れる（後で計算も可能だが今回は簡易的に）
                            let normal = if let Some(ni) = vn_idx {
                                raw_normals[ni]
                            } else {
                                vec3(0., 1., 0.)
                            };

                            vertices.push(pos);
                            normals.push(normal);
                            unique_map.insert(key, idx);
                            idx
                        };
                        face_indices.push(final_idx);
                    }

                    // 三角形分割
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

        // 法線がない場合の自動計算 (Flat shading fallback)
        // もし raw_normals が空なら、ここで三角形ごとに法線を計算して埋めると親切ですが、
        // 今回は geometry.rs 側でも計算できるので省略します。

        let mut mesh = Self {
            vertices,
            normals,
            indices,
        };
        mesh.normalize();
        mesh
    }

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
        let scale = if max_dim > 0. { 2. / max_dim } else { 1. };

        for v in &mut self.vertices {
            *v = (*v - center) * scale;
        }
    }
}
