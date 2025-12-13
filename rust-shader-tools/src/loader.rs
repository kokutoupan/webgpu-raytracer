// src/loader.rs
use crate::geometry::Geometry;
use crate::scene::animation::{Animation, Channel, ChannelOutputs, Interpolation}; // Interpolationを追加
use crate::scene::{Node, SceneInstance, Skin};
use glam::{Mat4, Quat, Vec2, Vec3, vec2, vec3};
use std::collections::HashMap;

pub fn load_gltf(
    geometries: &mut Vec<Geometry>,
    instances: &mut Vec<SceneInstance>,
    nodes: &mut Vec<Node>,
    skins: &mut Vec<Skin>,
    animations: &mut Vec<Animation>,
    textures: &mut Vec<Vec<u8>>, // 追加
    data: &[u8],
) -> Result<(), Box<dyn std::error::Error>> {
    let (gltf, buffers, _) = gltf::import_slice(data)?;

    // 0. Load Images -> Textures
    for image in gltf.images() {
        match image.source() {
            gltf::image::Source::View { view, mime_type: _ } => {
                let parent_buffer_data = &buffers[view.buffer().index()];
                let begin = view.offset();
                let end = begin + view.length();
                let image_data = &parent_buffer_data[begin..end];
                textures.push(image_data.to_vec());
            }
            _ => {
                // URI参照などは今回はスキップ (or Empty)
                textures.push(Vec::new());
            }
        }
    }

    // 1. Load Meshes -> Geometries
    let mut mesh_map = HashMap::new();

    for (i, mesh) in gltf.meshes().enumerate() {
        for primitive in mesh.primitives() {
            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

            let positions: Vec<Vec3> = reader
                .read_positions()
                .map(|v| v.map(|p| vec3(p[0], p[1], p[2])).collect())
                .unwrap_or_default();

            let normals: Vec<Vec3> = reader
                .read_normals()
                .map(|v| v.map(|n| vec3(n[0], n[1], n[2])).collect())
                .unwrap_or_default();

            let indices: Vec<u32> = reader
                .read_indices()
                .map(|v| v.into_u32().collect())
                .unwrap_or_default();

            // UV読み込み
            let tex_coords: Vec<Vec2> = reader
                .read_tex_coords(0)
                .map(|v| v.into_f32().map(|[u, v]| vec2(u, 1.0 - v)).collect()) // Flip V for WebGPU?
                .unwrap_or_else(|| vec![vec2(0., 0.); positions.len()]);

            let joints_u16: Vec<[u16; 4]> = reader
                .read_joints(0)
                .map(|v| v.into_u16().collect())
                .unwrap_or_else(|| vec![[0; 4]; positions.len()]);

            let weights: Vec<[f32; 4]> = reader
                .read_weights(0)
                .map(|v| v.into_f32().collect())
                .unwrap_or_else(|| vec![[0.; 4]; positions.len()]);

            let mut geom = Geometry::new();

            for k in 0..positions.len() {
                let p = positions[k];
                let n = if k < normals.len() {
                    normals[k]
                } else {
                    vec3(0., 1., 0.)
                };
                let uv = if k < tex_coords.len() {
                    tex_coords[k]
                } else {
                    vec2(0., 0.)
                };
                let j_u16 = joints_u16[k];
                let j = [
                    j_u16[0] as u32,
                    j_u16[1] as u32,
                    j_u16[2] as u32,
                    j_u16[3] as u32,
                ];
                let w = weights[k];

                geom.push_vertex_skinned(p, n, uv, j, w);
            }

            // Material情報 (Attributeに焼く)
            let mat = primitive.material();
            let pbr = mat.pbr_metallic_roughness();
            let base_color = pbr.base_color_factor();
            let col = vec3(base_color[0], base_color[1], base_color[2]);
            let mat_type = 0; // Lambertian
            let extra = 0.0;

            // Texture Index
            let tex_info = pbr.base_color_texture();
            let texture_index = if let Some(info) = tex_info {
                info.texture().index() as f32
            } else {
                -1.0
            };

            for chunk in indices.chunks(3) {
                if chunk.len() == 3 {
                    geom.indices.extend_from_slice(chunk);
                    geom.push_attributes(col, mat_type, extra, texture_index);
                }
            }

            mesh_map
                .entry(i)
                .or_insert(Vec::new())
                .push(geometries.len());
            geometries.push(geom);
        }
    }

    // 2. Load Nodes, Skins, Animations
    load_gltf_nodes_skins_anims(
        &gltf, &buffers, &mesh_map, geometries, instances, nodes, skins, animations,
    );

    Ok(())
}

fn load_gltf_nodes_skins_anims(
    gltf: &gltf::Document,
    buffers: &[gltf::buffer::Data],
    mesh_map: &HashMap<usize, Vec<usize>>,
    geometries: &mut Vec<Geometry>,
    instances: &mut Vec<SceneInstance>,
    nodes: &mut Vec<Node>,
    skins: &mut Vec<Skin>,
    animations: &mut Vec<Animation>,
) {
    // ---------------------------------------------------------
    // 1. Nodes の読み込み
    // ---------------------------------------------------------
    nodes.resize(gltf.nodes().count(), Node::default());

    for node in gltf.nodes() {
        let idx = node.index();
        let (trans, rot, scale) = node.transform().decomposed();

        nodes[idx].name = node.name().unwrap_or("").to_string();
        nodes[idx].translation = vec3(trans[0], trans[1], trans[2]);
        nodes[idx].rotation = Quat::from_array(rot); 
        nodes[idx].scale = vec3(scale[0], scale[1], scale[2]);
        nodes[idx].children_indices = node.children().map(|c| c.index()).collect();
    }

    // Parent Index の解決
    for i in 0..nodes.len() {
        let children = nodes[i].children_indices.clone();
        for child_idx in children {
            nodes[child_idx].parent_index = Some(i);
        }
    }

    // ---------------------------------------------------------
    // 2. Skins の読み込み
    // ---------------------------------------------------------
    for skin in gltf.skins() {
        let reader = skin.reader(|buffer| Some(&buffers[buffer.index()]));
        let inverse_bind_matrices: Vec<Mat4> = reader
            .read_inverse_bind_matrices()
            .map(|iter| iter.map(|m| Mat4::from_cols_array_2d(&m)).collect())
            .unwrap_or_else(|| vec![Mat4::IDENTITY; skin.joints().count()]);

        let joints: Vec<usize> = skin.joints().map(|node| node.index()).collect();

        skins.push(Skin {
            joints,
            inverse_bind_matrices,
        });
    }

    // ---------------------------------------------------------
    // 3. Instances (Mesh & Skin Link) の作成
    // ---------------------------------------------------------
    for node in gltf.nodes() {
        if let Some(mesh) = node.mesh() {
            let mesh_idx = mesh.index();
            if let Some(geo_indices) = mesh_map.get(&mesh_idx) {
                let skin_idx = node.skin().map(|s| s.index());

                for &geo_idx in geo_indices {
                    if let Some(s_idx) = skin_idx {
                        // geometryのskin_indexを更新
                        if geo_idx < geometries.len() {
                            geometries[geo_idx].skin_index = Some(s_idx);
                        }
                        // スキニング対象はIdentity配置
                        instances.push(SceneInstance {
                            transform: Mat4::IDENTITY,
                            geometry_index: geo_idx,
                        });
                    } else {
                        // 静的メッシュ (簡易的にIdentity)
                        instances.push(SceneInstance {
                            transform: Mat4::IDENTITY,
                            geometry_index: geo_idx,
                        });
                    }
                }
            }
        }
    }

    // ---------------------------------------------------------
    // 4. Animations の読み込み
    // ---------------------------------------------------------
    for anim in gltf.animations() {
        let mut channels = Vec::new();

        for channel in anim.channels() {
            let reader = channel.reader(|buffer| Some(&buffers[buffer.index()]));
            let target = channel.target();
            let target_node_idx = target.node().index();
            let target_node_name = nodes[target_node_idx].name.clone();

            // ★ Interpolation 読み込み
            let sampler = channel.sampler();
            let interpolation = match sampler.interpolation() {
                gltf::animation::Interpolation::Linear => Interpolation::Linear,
                gltf::animation::Interpolation::Step => Interpolation::Step,
                gltf::animation::Interpolation::CubicSpline => Interpolation::CubicSpline,
            };

            let inputs: Vec<f32> = reader.read_inputs().unwrap().collect();
            let outputs = reader.read_outputs().unwrap();

            let channel_outputs = match outputs {
                gltf::animation::util::ReadOutputs::Translations(iter) => {
                    let vecs: Vec<Vec3> = iter.map(|v| vec3(v[0], v[1], v[2])).collect();
                    ChannelOutputs::Translations(vecs)
                }
                gltf::animation::util::ReadOutputs::Rotations(iter) => {
                    let quats: Vec<Quat> = iter.into_f32().map(|v| Quat::from_array(v)).collect();
                    ChannelOutputs::Rotations(quats)
                }
                gltf::animation::util::ReadOutputs::Scales(iter) => {
                    let vecs: Vec<Vec3> = iter.map(|v| vec3(v[0], v[1], v[2])).collect();
                    ChannelOutputs::Scales(vecs)
                }
                _ => continue,
            };

            channels.push(Channel {
                target_node_name,
                inputs,
                outputs: channel_outputs,
                interpolation, // ★ここを設定しました
            });
        }

        let mut max_time = 0.0;
        for ch in &channels {
            if let Some(last) = ch.inputs.last() {
                if *last > max_time {
                    max_time = *last;
                }
            }
        }

        animations.push(Animation {
            name: anim.name().unwrap_or("anim").to_string(),
            duration: max_time,
            channels,
        });
    }
}
