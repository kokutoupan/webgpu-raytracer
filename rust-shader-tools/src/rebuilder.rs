// src/rebuilder.rs
use crate::bvh::blas::BVHBuilder;
use crate::geometry::Geometry;
use crate::render_buffers::RenderBuffers;
use crate::scene::Skin;
use glam::{Mat4, Vec3};

pub fn build_blas_and_vertices(
    geometries: &[Geometry],
    skins: &[Skin],
    global_transforms: &[Mat4],
    buffers: &mut RenderBuffers,
    blas_root_offsets: &mut Vec<u32>,
) -> Vec<Vec<u32>> {
    buffers.clear();
    blas_root_offsets.clear();
    let mut emissive_lists = Vec::new(); // Added

    let mut current_node_offset = 0;

    for geom in geometries {
        // Base Data
        let (use_positions, use_normals) = if geom.base_positions.is_empty() {
            // 頂点がない場合スキップ
            blas_root_offsets.push(0);
            continue;
        } else {
            (&geom.base_positions, &geom.base_normals)
        };
        let base_uvs = &geom.base_uvs;

        // Skinning Check
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
        let mut uv_vec2 = Vec::with_capacity(base_uvs.len() * 2); // ★追加

        if let Some(skin) = skin_opt {
            let joint_mats: Vec<Mat4> = skin
                .joints
                .iter()
                .zip(skin.inverse_bind_matrices.iter())
                .map(|(&j, &inv)| global_transforms[j] * inv)
                .collect();

            for i in 0..use_positions.len() {
                let pos = use_positions[i];
                let norm = use_normals[i];
                let uv = if i < base_uvs.len() {
                    base_uvs[i]
                } else {
                    glam::Vec2::ZERO
                };
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
                let n = mat.transform_vector3(norm).normalize_or_zero();

                // Sanitize NaNs
                let p_safe = if p.is_nan() { Vec3::ZERO } else { p };
                let n_safe = if n.is_nan() { Vec3::Z } else { n };

                v_vec4.extend_from_slice(&[p_safe.x, p_safe.y, p_safe.z, 1.0]);
                n_vec4.extend_from_slice(&[n_safe.x, n_safe.y, n_safe.z, 0.0]);
                uv_vec2.extend_from_slice(&[uv.x, uv.y]);
            }
        } else {
            for i in 0..use_positions.len() {
                let p = use_positions[i];
                let n = use_normals[i];
                let uv = if i < base_uvs.len() {
                    base_uvs[i]
                } else {
                    glam::Vec2::ZERO
                };

                // Sanitize NaNs
                let p_safe = if p.is_nan() { Vec3::ZERO } else { p };
                let n_safe = if n.is_nan() { Vec3::Z } else { n };

                v_vec4.extend_from_slice(&[p_safe.x, p_safe.y, p_safe.z, 1.0]);
                n_vec4.extend_from_slice(&[n_safe.x, n_safe.y, n_safe.z, 0.0]);
                uv_vec2.extend_from_slice(&[uv.x, uv.y]);
            }
        }

        // BLAS Build
        let mut builder = BVHBuilder::new(&v_vec4, &geom.indices);
        let (mut nodes, indices, tri_ids) = builder.build_with_ids();
        let v_offset = (buffers.vertices.len() / 4) as u32;

        // global_indices removed (unused)
        // Update Leaf Tri Indices
        // Update Leaf Tri Indices
        // For mesh_topology (stride 1 -> 1 struct per tri), leaf refers to "Primitive Index"
        // In this case, primitive index is just the index in the topology array.
        // Current BLAS holds `tri_offset` in mesh_topology.
        let current_topology_start = (buffers.mesh_topology.len() / 20) as u32;

        for i in 0..(nodes.len() / 8) {
            if nodes[i * 8 + 7] > 0. {
                // is_leaf
                let lf = nodes[i * 8 + 3] as u32;
                // lf is the index in the *local* sorted_indices (chunks of 3).
                // We want global index in mesh_topology.
                nodes[i * 8 + 3] = (lf + current_topology_start) as f32;
            }
        }

        // Pack Mesh Topology
        let mut current_emissive = Vec::new();
        // Stride: 16 u32s (v0, v1, v2, pad, data0[4], data1[4], data2[4], data3[4])
        // Attribute part is 12 floats.
        for (i, &old_id) in tri_ids.iter().enumerate() {
            let base_v = i * 3;
            let v0 = indices[base_v] + v_offset;
            let v1 = indices[base_v + 1] + v_offset;
            let v2 = indices[base_v + 2] + v_offset;

            // Attributes
            let src_attr = old_id * 16;
            let attrs = &geom.attributes[src_attr..src_attr + 16];

            // 0..3: Indices + Pad
            buffers.mesh_topology.push(v0);
            buffers.mesh_topology.push(v1);
            buffers.mesh_topology.push(v2);
            buffers.mesh_topology.push(0); // Pad

            // 4..19: Attributes (16 floats total now? Wait, stride in geometry was 16?)
            // Let's re-verify geometry.rs push_attributes.
            // data0..data3 = 16 floats.
            for &f in attrs {
                buffers.mesh_topology.push(f.to_bits());
            }

            // Check for Light (Type = 3)
            // attrs[3] is mat_type (float).
            let mat_bits = attrs[3].to_bits();
            if mat_bits == 3 {
                current_emissive.push(current_topology_start + i as u32);
            }
        }
        emissive_lists.push(current_emissive);

        // Extend Global Buffers
        buffers.vertices.extend(v_vec4);
        buffers.normals.extend(n_vec4);
        buffers.uvs.extend(uv_vec2);
        // buffers.indices / attributes removed
        buffers.blas_nodes.extend(nodes);

        blas_root_offsets.push(current_node_offset);
        let node_count = (buffers.blas_nodes.len() as u32 - current_node_offset * 8) / 8;
        current_node_offset += node_count;
        // current_tri_offset removed
    }
    emissive_lists
}
