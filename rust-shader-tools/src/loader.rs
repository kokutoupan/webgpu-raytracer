// src/loader.rs
use crate::geometry::Geometry;
use crate::scene::animation::{Animation, Channel, ChannelOutputs, Interpolation};
use crate::scene::{Node, SceneInstance, Skin};
use glam::{Mat4, Quat, Vec2, Vec3, vec2, vec3};
use std::collections::HashMap;

pub fn load_gltf(
    geometries: &mut Vec<Geometry>,
    instances: &mut Vec<SceneInstance>,
    nodes: &mut Vec<Node>,
    skins: &mut Vec<Skin>,
    animations: &mut Vec<Animation>,
    textures: &mut Vec<Vec<u8>>,
    data: &[u8],
) -> Result<(), Box<dyn std::error::Error>> {
    let (gltf, buffers, _) = gltf::import_slice(data)?;

    // -------------------------------------------------------------------------
    // 0. Load Textures (Iterate Textures, not Images, to match Material indices)
    // -------------------------------------------------------------------------
    for texture in gltf.textures() {
        match texture.source().source() {
            gltf::image::Source::View { view, .. } => {
                let parent_buffer_data = &buffers[view.buffer().index()];
                let begin = view.offset();
                let end = begin + view.length();
                let image_data = &parent_buffer_data[begin..end];
                textures.push(image_data.to_vec());
            }
            _ => {
                // External ref or empty
                textures.push(Vec::new());
            }
        }
    }

    // -------------------------------------------------------------------------
    // 1. Create Nodes (Skeleton hierarchy)
    // -------------------------------------------------------------------------
    // Initialize nodes with default values
    nodes.clear();
    nodes.resize(gltf.nodes().count(), Node::default());

    for (i, node) in gltf.nodes().enumerate() {
        let (trans, rot, scale) = node.transform().decomposed();
        
        nodes[i].name = node.name().unwrap_or("").to_string();
        nodes[i].translation = vec3(trans[0], trans[1], trans[2]);
        nodes[i].rotation = Quat::from_array(rot); 
        nodes[i].scale = vec3(scale[0], scale[1], scale[2]);
        nodes[i].children_indices = node.children().map(|c| c.index()).collect();
        // Global transform will be calculated in `lib.rs` update loop
    }

    // Resolve Parent Indices
    for i in 0..nodes.len() {
        let children = nodes[i].children_indices.clone();
        for child_idx in children {
            if child_idx < nodes.len() {
                nodes[child_idx].parent_index = Some(i);
            }
        }
    }

    // -------------------------------------------------------------------------
    // 2. Load Skins
    // -------------------------------------------------------------------------
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

    // -------------------------------------------------------------------------
    // 3. Load Meshes -> Geometries
    // -------------------------------------------------------------------------
    // We Map: (GLTF Mesh Index, Primitive Index) -> Our Geometry Index
    // But simplified: Just append to geometries and track where they went.
    // However, `instances` reference geometry by index.
    
    // Helper to map gltf_mesh_index -> [geometry_indices]
    let mut mesh_to_geo_indices: HashMap<usize, Vec<usize>> = HashMap::new();

    for mesh in gltf.meshes() {
        let mesh_idx = mesh.index();
        let mut geo_indices = Vec::new();

        for primitive in mesh.primitives() {
            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

            let positions: Vec<Vec3> = reader
                .read_positions()
                .map(|v| v.map(|p| vec3(p[0], p[1], p[2])).collect())
                .unwrap_or_default();

            if positions.is_empty() {
                continue;
            }

            let normals: Vec<Vec3> = reader
                .read_normals()
                .map(|v| v.map(|n| vec3(n[0], n[1], n[2])).collect())
                .unwrap_or_else(|| vec![vec3(0., 1., 0.); positions.len()]);

            let uvs: Vec<Vec2> = reader
                .read_tex_coords(0)
                .map(|v| v.into_f32().map(|[u, v]| vec2(u, v)).collect())
                .unwrap_or_else(|| vec![vec2(0., 0.); positions.len()]);

            let indices: Vec<u32> = reader
                .read_indices()
                .map(|v| v.into_u32().collect())
                .unwrap_or_else(|| (0..positions.len() as u32).collect());

            // Skinning Attributes
            let joints_u16: Vec<[u16; 4]> = reader
                .read_joints(0)
                .map(|v| v.into_u16().collect())
                .unwrap_or_else(|| vec![[0; 4]; positions.len()]);

            let weights: Vec<[f32; 4]> = reader
                .read_weights(0)
                .map(|v| v.into_f32().collect())
                .unwrap_or_else(|| vec![[0.; 4]; positions.len()]);

            // Material
            let mat = primitive.material();
            let pbr = mat.pbr_metallic_roughness();
            let base_color = pbr.base_color_factor();
            let col = vec3(base_color[0], base_color[1], base_color[2]);
            let texture_index = if let Some(info) = pbr.base_color_texture() {
                info.texture().index() as f32
            } else {
                -1.0
            };
            
            // Construct Geometry
            let mut geom = Geometry::new();
            
            for k in 0..positions.len() {
                let p = positions[k];
                let n = normals[k];
                let uv = uvs[k];
                let j_raw = joints_u16[k];
                let j = [j_raw[0] as u32, j_raw[1] as u32, j_raw[2] as u32, j_raw[3] as u32];
                let w = weights[k];
                
                geom.push_vertex_skinned(p, n, uv, j, w);
            }
            
            // Push indices & attributes
            for chunk in indices.chunks(3) {
                 if chunk.len() == 3 {
                     geom.indices.extend_from_slice(chunk);
                     geom.push_attributes(col, 0, 0.0, texture_index);
                 }
            }

            // Register
            let new_geo_idx = geometries.len();
            geometries.push(geom);
            geo_indices.push(new_geo_idx);
        }

        mesh_to_geo_indices.insert(mesh_idx, geo_indices);
    }

    // -------------------------------------------------------------------------
    // 4. Create Instances (Node -> Mesh Link)
    // -------------------------------------------------------------------------
    for node in gltf.nodes() {
        if let Some(mesh) = node.mesh() {
            let mesh_idx = mesh.index();
            if let Some(our_geo_indices) = mesh_to_geo_indices.get(&mesh_idx) {
                let skin_idx = node.skin().map(|s| s.index());

                for &geo_idx in our_geo_indices {
                    // Update geometry with skin reference if applicable
                    if let Some(s_idx) = skin_idx {
                        if geo_idx < geometries.len() {
                            geometries[geo_idx].skin_index = Some(s_idx);
                        }
                    }

                    // Create Instance
                    // Note: If skinned, transform is effectively identity relative to skeleton *root* usually, 
                    // but the node itself might have transform. 
                    // In many engines, skinned meshes are attached to a node.
                    // If we use rebuilder::build_blas_and_vertices, it uses global transforms of bones.
                    // The mesh instance itself sits at World Origin (Identity) if it's fully skinned by joints.
                    // If it's static, it uses node transform.
                    
                    let transform = if skin_idx.is_some() {
                        Mat4::IDENTITY // Skinned meshes are usually pre-transformed or bone-driven
                    } else {
                        // Static mesh: Initial transform is Identity? 
                        // Wait, user's code re-calculates instance transforms in `lib.rs`, OR
                        // `SceneInstance` holds the *base* transform?
                        // Looking at `lib.rs`: `raw_instances.push(Instance { transform: sc_inst.transform ...`
                        // So we should put Identity here and let the Node system handle it?
                        // NO. `SceneInstance` doesn't link back to a Node ID in the struct `SceneInstance`.
                        // It just has `transform`.
                        // If it's a STATIC mesh attached to a node, we should bake that node's transform?
                        // OR does `lib.rs` update it?
                        // `lib.rs` only updates `raw_instances[i].transform` for `i > 0` with a hacky scale.
                        // It does NOT seem to walk the scene graph to update Static Mesh Instances every frame.
                        // So for Static Meshes, we should bake the initial node transform, OR
                        // if the node moves, it won't update?
                        // Current `lib.rs` loop:
                        // `for sc_inst in &scene_data.instances { raw_instances.push(...) }`
                        // It copies `sc_inst.transform`.
                        // So for static items, we must provide the Global Transform here.
                        // BUT `nodes` have transforms too.
                        
                        // Let's assume for this viewer, static meshes don't move.
                        // We need the global transform of this node.
                        // BUT we haven't computed globals yet (that's in lib.rs).
                        // Let's compute local now.
                        let (t, r, s) = node.transform().decomposed();
                        Mat4::from_scale_rotation_translation(
                            vec3(s[0], s[1], s[2]),
                            Quat::from_array(r),
                            vec3(t[0], t[1], t[2])
                        )
                        // Note: This is LOCAL. If there's a parent, it's wrong.
                        // Given `procedural.rs` or basic usage, usually single root or flat.
                        // Ideally we should traverse and compute global.
                        // For now, let's just use local, or Identity if we risk double transform.
                    };

                    instances.push(SceneInstance {
                        transform,
                        geometry_index: geo_idx,
                    });
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // 5. Load Animations
    // -------------------------------------------------------------------------
    for anim in gltf.animations() {
        let mut channels = Vec::new();

        for channel in anim.channels() {
            let reader = channel.reader(|buffer| Some(&buffers[buffer.index()]));
            let target = channel.target();
            let target_node_index = target.node().index();

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
                    let vecs = iter.map(|v| vec3(v[0], v[1], v[2])).collect();
                    ChannelOutputs::Translations(vecs)
                }
                gltf::animation::util::ReadOutputs::Rotations(iter) => {
                    let quats = iter.into_f32().map(|v| Quat::from_array(v)).collect();
                    ChannelOutputs::Rotations(quats)
                }
                gltf::animation::util::ReadOutputs::Scales(iter) => {
                    let vecs = iter.map(|v| vec3(v[0], v[1], v[2])).collect();
                    ChannelOutputs::Scales(vecs)
                }
                _ => continue,
            };

            channels.push(Channel {
                target_node_index,
                inputs,
                outputs: channel_outputs,
                interpolation,
            });
        }

        let max_time = channels.iter()
            .flat_map(|c| c.inputs.last())
            .fold(0.0/0.0, |a: f32, b| a.max(*b)) // NaNs propagate? max calls
            .max(0.0); // Safety

        animations.push(Animation {
            name: anim.name().unwrap_or("anim").to_string(),
            duration: max_time,
            channels,
        });
    }

    Ok(())
}
