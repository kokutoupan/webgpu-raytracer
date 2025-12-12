// src/loader.rs
use crate::geometry::Geometry;
use crate::scene::animation::{Animation, Channel, ChannelOutputs, Interpolation};
use crate::scene::{Node, SceneInstance, Skin};
use glam::{Mat4, Quat, Vec3, vec3};
use std::collections::HashMap;

pub fn load_gltf(
    geometries: &mut Vec<Geometry>,
    instances: &mut Vec<SceneInstance>,
    nodes_out: &mut Vec<Node>,
    skins_out: &mut Vec<Skin>,
    animations_out: &mut Vec<Animation>,
    glb_data: &[u8],
) -> Result<(), String> {
    // expect ではなく map_err でエラーを返す
    let (document, buffers, _images) =
        gltf::import_slice(glb_data).map_err(|e| format!("GLTF Import Error: {}", e))?;

    let mut node_map = HashMap::new();
    let start_node_index = nodes_out.len();

    // 1. Nodes
    for node in document.nodes() {
        let (t, r, s) = node.transform().decomposed();
        let new_node = Node {
            name: node.name().unwrap_or("").to_string(),
            parent_index: None,
            children_indices: node
                .children()
                .map(|c| start_node_index + c.index())
                .collect(),
            translation: Vec3::from(t),
            rotation: Quat::from_array(r),
            scale: Vec3::from(s),
            global_transform: Mat4::IDENTITY,
        };
        nodes_out.push(new_node);
        node_map.insert(node.index(), start_node_index + node.index());
    }

    for node in document.nodes() {
        let my_idx = start_node_index + node.index();
        for child in node.children() {
            let child_idx = start_node_index + child.index();
            nodes_out[child_idx].parent_index = Some(my_idx);
        }
    }

    // 2. Skins
    let mut skin_map = HashMap::new();
    for skin in document.skins() {
        let reader = skin.reader(|buffer| Some(&buffers[buffer.index()]));
        let inverse_bind_matrices: Vec<Mat4> = reader
            .read_inverse_bind_matrices()
            .map(|iter| iter.map(|m| Mat4::from_cols_array_2d(&m)).collect())
            .unwrap_or_default();
        let joints: Vec<usize> = skin
            .joints()
            .map(|j| start_node_index + j.index())
            .collect();
        skins_out.push(Skin {
            joints,
            inverse_bind_matrices,
        });
        skin_map.insert(skin.index(), skins_out.len() - 1);
    }

    // 3. Meshes
    for node in document.nodes() {
        if let Some(mesh) = node.mesh() {
            let skin_index = node.skin().and_then(|s| skin_map.get(&s.index()).cloned());
            let mut geom = Geometry::new();
            geom.skin_index = skin_index;

            for primitive in mesh.primitives() {
                extract_primitive(&primitive, &buffers, &mut geom);
            }

            if !geom.vertices.is_empty() {
                geometries.push(geom);
                let geom_idx = geometries.len() - 1;
                // スキニングメッシュはIdentity
                instances.push(SceneInstance {
                    transform: Mat4::IDENTITY,
                    geometry_index: geom_idx,
                });
            }
        }
    }

    // 4. Animations
    for anim in document.animations() {
        let mut channels = Vec::new();
        let mut max_time = 0.0f32;

        for channel in anim.channels() {
            let reader = channel.reader(|buffer| Some(&buffers[buffer.index()]));
            let inputs: Vec<f32> = reader.read_inputs().unwrap().collect();
            if let Some(last) = inputs.last() {
                if *last > max_time {
                    max_time = *last;
                }
            }

            let target_node_gltf = channel.target().node();
            let target_node_name = target_node_gltf.name().unwrap_or("").to_string();

            if !node_map.contains_key(&target_node_gltf.index()) {
                continue;
            }

            let outputs = match reader.read_outputs() {
                Some(gltf::animation::util::ReadOutputs::Translations(iter)) => {
                    ChannelOutputs::Translations(iter.map(Vec3::from).collect())
                }
                Some(gltf::animation::util::ReadOutputs::Rotations(iter)) => {
                    ChannelOutputs::Rotations(iter.into_f32().map(Quat::from_array).collect())
                }
                Some(gltf::animation::util::ReadOutputs::Scales(iter)) => {
                    ChannelOutputs::Scales(iter.map(Vec3::from).collect())
                }
                _ => continue,
            };

            let interpolation = match channel.sampler().interpolation() {
                gltf::animation::Interpolation::Linear => Interpolation::Linear,
                gltf::animation::Interpolation::Step => Interpolation::Step,
                gltf::animation::Interpolation::CubicSpline => Interpolation::CubicSpline,
            };

            channels.push(Channel {
                target_node_name,
                inputs,
                outputs,
                interpolation,
            });
        }

        if !channels.is_empty() {
            animations_out.push(Animation {
                name: anim.name().unwrap_or("anim").to_string(),
                channels,
                duration: max_time,
            });
        }
    }

    Ok(())
}

fn extract_primitive(
    primitive: &gltf::Primitive,
    buffers: &[gltf::buffer::Data],
    geom: &mut Geometry,
) {
    let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
    let positions: Vec<Vec3> = reader
        .read_positions()
        .map(|iter| iter.map(|p| vec3(p[0], p[1], p[2])).collect())
        .unwrap_or_default();
    if positions.is_empty() {
        return;
    }
    let vertex_count = positions.len();
    let normals: Vec<Vec3> = reader
        .read_normals()
        .map(|iter| iter.map(|n| vec3(n[0], n[1], n[2])).collect())
        .unwrap_or_else(|| vec![vec3(0., 1., 0.); vertex_count]);
    let joints: Vec<[u32; 4]> = reader
        .read_joints(0)
        .map(|iter| {
            iter.into_u16()
                .map(|j| [j[0] as u32, j[1] as u32, j[2] as u32, j[3] as u32])
                .collect()
        })
        .unwrap_or_else(|| vec![[0, 0, 0, 0]; vertex_count]);
    let weights: Vec<[f32; 4]> = reader
        .read_weights(0)
        .map(|iter| iter.into_f32().collect())
        .unwrap_or_else(|| vec![[0.0, 0.0, 0.0, 0.0]; vertex_count]);
    let indices: Vec<u32> = reader
        .read_indices()
        .map(|iter| iter.into_u32().collect())
        .unwrap_or_else(|| (0..vertex_count as u32).collect());

    let color = vec3(0.8, 0.8, 0.8);
    let mat_type = 0;
    let extra = 0.0;

    // オフセット加算
    let vertex_offset = (geom.vertices.len() / 4) as u32;

    for i in 0..vertex_count {
        geom.push_vertex_skinned(positions[i], normals[i], joints[i], weights[i]);
    }
    for chunk in indices.chunks(3) {
        if chunk.len() == 3 {
            geom.indices.push(chunk[0] + vertex_offset);
            geom.indices.push(chunk[1] + vertex_offset);
            geom.indices.push(chunk[2] + vertex_offset);
            geom.push_attributes(color, mat_type, extra);
        }
    }
}
