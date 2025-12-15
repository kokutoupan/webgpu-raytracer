// src/mesh.rs
use glam::{Vec2, Vec3, vec2, vec3};

pub struct Mesh {
    pub vertices: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub uvs: Vec<Vec2>, // ★追加
    pub indices: Vec<u32>,
}

impl Mesh {
    pub fn new(source: &str) -> Self {
        let mut raw_positions = Vec::new();
        let mut raw_normals = Vec::new();
        let mut raw_uvs = Vec::new();

        let mut unique_vertices = Vec::new(); // (p, t, n) -> idx
        let mut indices = Vec::new();

        let mut final_positions = Vec::new();
        let mut final_normals = Vec::new();
        let mut final_uvs = Vec::new();

        for line in source.lines() {
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
                "vt" => {
                    if parts.len() >= 3 {
                        let u = parts[1].parse().unwrap_or(0.);
                        let v = parts[2].parse().unwrap_or(0.);
                        raw_uvs.push(vec2(u, v));
                    }
                }
                "vn" => {
                    if parts.len() >= 4 {
                        let x = parts[1].parse().unwrap_or(0.);
                        let y = parts[2].parse().unwrap_or(0.);
                        let z = parts[3].parse().unwrap_or(0.);
                        raw_normals.push(vec3(x, y, z));
                    }
                }
                "f" => {
                    let mut face_indices = Vec::new();
                    for part in parts.iter().skip(1) {
                        let segs: Vec<&str> = part.split('/').collect();
                        let p_idx = segs[0].parse::<usize>().unwrap_or(0).saturating_sub(1);

                        let t_idx = if segs.len() > 1 && !segs[1].is_empty() {
                            Some(segs[1].parse::<usize>().unwrap_or(0).saturating_sub(1))
                        } else {
                            None
                        };

                        let n_idx = if segs.len() > 2 && !segs[2].is_empty() {
                            Some(segs[2].parse::<usize>().unwrap_or(0).saturating_sub(1))
                        } else {
                            None
                        };

                        let key = (p_idx, t_idx, n_idx);
                        if let Some(idx) = unique_vertices.iter().position(|&k| k == key) {
                            face_indices.push(idx as u32);
                        } else {
                            let idx = unique_vertices.len() as u32;
                            unique_vertices.push(key);

                            if p_idx < raw_positions.len() {
                                final_positions.push(raw_positions[p_idx]);
                            } else {
                                final_positions.push(Vec3::ZERO);
                            }

                            if let Some(ti) = t_idx {
                                if ti < raw_uvs.len() {
                                    final_uvs.push(raw_uvs[ti]);
                                } else {
                                    final_uvs.push(Vec2::ZERO);
                                }
                            } else {
                                final_uvs.push(Vec2::ZERO);
                            }

                            if let Some(ni) = n_idx {
                                if ni < raw_normals.len() {
                                    final_normals.push(raw_normals[ni]);
                                } else {
                                    final_normals.push(Vec3::Y);
                                }
                            } else {
                                final_normals.push(Vec3::Y);
                            }

                            face_indices.push(idx);
                        }
                    }
                    for i in 1..face_indices.len().saturating_sub(1) {
                        indices.push(face_indices[0]);
                        indices.push(face_indices[i]);
                        indices.push(face_indices[i + 1]);
                    }
                }
                _ => {}
            }
        }

        Self {
            vertices: final_positions,
            normals: final_normals,
            uvs: final_uvs,
            indices,
        }
    }
}
