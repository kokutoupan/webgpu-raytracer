// src/rebuilder.rs
use crate::bvh::blas::BVHBuilder;
use crate::geometry::Geometry;
use crate::render_buffers::RenderBuffers;
use crate::scene::Skin;
use glam::{Mat4, Vec3};

/// アニメーション適用後の姿勢(global_transforms)を元に、
/// 全ジオメトリの頂点を計算し、BLASを構築してバッファに詰める関数
pub fn build_blas_and_vertices(
    geometries: &[Geometry],
    skins: &[Skin],
    global_transforms: &[Mat4],
    buffers: &mut RenderBuffers,
    blas_root_offsets: &mut Vec<u32>,
) {
    // バッファをクリア（前フレームのゴミを消す）
    buffers.vertices.clear();
    buffers.normals.clear();
    buffers.indices.clear();
    buffers.attributes.clear();
    buffers.blas_nodes.clear();
    blas_root_offsets.clear();

    let mut current_node_offset = 0;
    let mut current_tri_offset = 0;

    for geom in geometries {
        // --- 1. 頂点座標と法線の準備 (スキニング or 静的コピー) ---
        
        // base_positionsが空の場合（geometry.rs更新漏れ等）の安全策
        let base_positions_backup: Vec<Vec3>;
        let base_normals_backup: Vec<Vec3>;

        let (use_positions, use_normals) = if geom.base_positions.is_empty() {
            if !geom.vertices.is_empty() {
                // 互換性のため: フラット配列からVec3へ変換
                base_positions_backup = geom
                    .vertices
                    .chunks(4)
                    .map(|c| Vec3::new(c[0], c[1], c[2]))
                    .collect();
                base_normals_backup = geom
                    .normals
                    .chunks(4)
                    .map(|c| Vec3::new(c[0], c[1], c[2]))
                    .collect();
                (&base_positions_backup, &base_normals_backup)
            } else {
                // 頂点自体がない場合はダミーオフセットを入れてスキップ
                blas_root_offsets.push(0);
                continue;
            }
        } else {
            (&geom.base_positions, &geom.base_normals)
        };

        // スキニング適用チェック
        let skin_opt = if let Some(s_idx) = geom.skin_index {
            if s_idx < skins.len() {
                Some(&skins[s_idx])
            } else {
                None
            }
        } else {
            None
        };

        let mut v_vec4 = Vec::with_capacity(use_positions.len() * 4);
        let mut n_vec4 = Vec::with_capacity(use_normals.len() * 4);

        if let Some(skin) = skin_opt {
            // --- スキニング計算 ---
            let joint_mats: Vec<Mat4> = skin
                .joints
                .iter()
                .zip(skin.inverse_bind_matrices.iter())
                .map(|(&j, &inv)| global_transforms[j] * inv)
                .collect();

            for i in 0..use_positions.len() {
                let pos = use_positions[i];
                let norm = use_normals[i];
                let j = &geom.joints[i * 4..(i + 1) * 4];
                let w = &geom.weights[i * 4..(i + 1) * 4];

                let mut mat = Mat4::ZERO;
                for k in 0..4 {
                    if w[k] > 0. {
                        mat += joint_mats[j[k] as usize] * w[k];
                    }
                }
                if mat == Mat4::ZERO {
                    mat = Mat4::IDENTITY;
                }

                let p = mat.transform_point3(pos);
                let n = mat.transform_vector3(norm).normalize();
                v_vec4.extend_from_slice(&[p.x, p.y, p.z, 1.0]);
                n_vec4.extend_from_slice(&[n.x, n.y, n.z, 0.0]);
            }
        } else {
            // --- 静的メッシュ (そのままコピー) ---
            for i in 0..use_positions.len() {
                let p = use_positions[i];
                let n = use_normals[i];
                v_vec4.extend_from_slice(&[p.x, p.y, p.z, 1.0]);
                n_vec4.extend_from_slice(&[n.x, n.y, n.z, 0.0]);
            }
        }

        // --- 2. BLAS構築 (BVH) ---
        let mut builder = BVHBuilder::new(&v_vec4, &geom.indices);
        let (mut nodes, indices, tri_ids) = builder.build_with_ids();

        // インデックスのグローバルオフセット調整
        let v_offset = (buffers.vertices.len() / 4) as u32;
        let global_indices: Vec<u32> = indices.iter().map(|&idx| idx + v_offset).collect();

        // LeafノードのトライアングルIDオフセット調整 (Global Indexing)
        for i in 0..(nodes.len() / 8) {
            if nodes[i * 8 + 7] > 0. {
                // is_leaf
                let lf = nodes[i * 8 + 3] as u32;
                nodes[i * 8 + 3] = (lf + current_tri_offset) as f32;
            }
        }

        // --- 3. 属性(Attributes)の並び替え ---
        // BVH構築時に三角形の順番が変わるので、それに合わせて属性も並べ替える
        let mut sorted_attrs = vec![0.0; geom.attributes.len()];
        let stride = 8; // vec4 * 2
        for (new_i, &old_id) in tri_ids.iter().enumerate() {
            let src = old_id * stride;
            let dst = new_i * stride;
            if src + stride <= geom.attributes.len() {
                sorted_attrs[dst..dst + stride]
                    .copy_from_slice(&geom.attributes[src..src + stride]);
            }
        }

        // --- 4. グローバルバッファへの追加 ---
        buffers.vertices.extend(v_vec4);
        buffers.normals.extend(n_vec4);
        buffers.indices.extend(global_indices);
        buffers.attributes.extend(sorted_attrs);

        // BLASルートオフセットの記録
        blas_root_offsets.push(current_node_offset);

        let node_count = (nodes.len() / 8) as u32;
        current_node_offset += node_count;
        current_tri_offset += (indices.len() / 3) as u32;

        buffers.blas_nodes.extend(nodes);
    }
}
