// src/loader.rs
use crate::geometry::Geometry;
use glam::{Vec3, vec3};

pub fn load_gltf(geom: &mut Geometry, glb_data: &[u8]) {
    // 1. glTF (GLB) をパース
    // import_slice はメモリ上のバイト列から読み込みます (Wasmに最適)
    let (document, buffers, _images) = gltf::import_slice(glb_data).expect("Failed to load glTF");

    // 2. メッシュを走査
    for mesh in document.meshes() {
        for primitive in mesh.primitives() {
            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

            // --- A. 頂点座標 (Positions) ---
            let positions: Vec<Vec3> = reader
                .read_positions()
                .map(|iter| iter.map(|p| vec3(p[0], p[1], p[2])).collect())
                .unwrap_or_default();

            if positions.is_empty() {
                continue;
            }
            let vertex_count = positions.len();

            // --- B. 法線 (Normals) ---
            // なければ (0, 1, 0) で埋める
            let normals: Vec<Vec3> = reader
                .read_normals()
                .map(|iter| iter.map(|n| vec3(n[0], n[1], n[2])).collect())
                .unwrap_or_else(|| vec![vec3(0., 1., 0.); vertex_count]);

            // --- C. スキニング情報 (Joints & Weights) ---
            // ジョイント (u16 -> u32 に変換)
            let joints: Vec<[u32; 4]> = reader
                .read_joints(0)
                .map(|iter| {
                    iter.into_u16()
                        .map(|j| [j[0] as u32, j[1] as u32, j[2] as u32, j[3] as u32])
                        .collect()
                })
                .unwrap_or_else(|| vec![[0, 0, 0, 0]; vertex_count]);

            // ウェイト (f32)
            let weights: Vec<[f32; 4]> = reader
                .read_weights(0)
                .map(|iter| iter.into_f32().collect())
                .unwrap_or_else(|| vec![[0.0, 0.0, 0.0, 0.0]; vertex_count]);

            // --- D. インデックス (Indices) ---
            // u32 に統一して取得
            let indices: Vec<u32> = reader
                .read_indices()
                .map(|iter| iter.into_u32().collect())
                .unwrap_or_else(|| (0..vertex_count as u32).collect());

            // --- E. Geometry に焼き込み ---

            // マテリアル (今回は簡易的に白固定。本来は primitive.material() から取得)
            let color = vec3(0.8, 0.8, 0.8);
            let mat_type = 0; // Lambertian
            let extra = 0.0;

            // Geometryの現在の頂点数をオフセットとして記録
            // (Geometryは複数のメッシュを統合するため)
            let start_vertex_offset = (geom.vertices.len() / 4) as u32;

            // 1. 頂点の追加
            for i in 0..vertex_count {
                // ここで geometry.rs の pub fn push_vertex_skinned を呼ぶ
                geom.push_vertex_skinned(positions[i], normals[i], joints[i], weights[i]);
            }

            // 2. インデックスの追加 (オフセットを加算)
            for chunk in indices.chunks(3) {
                if chunk.len() == 3 {
                    geom.indices.push(chunk[0] + start_vertex_offset);
                    geom.indices.push(chunk[1] + start_vertex_offset);
                    geom.indices.push(chunk[2] + start_vertex_offset);

                    // 属性(色など)は三角形ごとに追加
                    geom.push_attributes(color, mat_type, extra);
                }
            }
        }
    }
}
