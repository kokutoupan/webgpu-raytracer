// src/lib.rs
use crate::bvh::{Instance, tlas::TLASBuilder};
use crate::mesh::Mesh;
use crate::scene::animation::{Animation, ChannelOutputs};
use crate::scene::{Node, SceneData};
use glam::{Mat4, Quat, Vec3};
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

pub mod bvh;
pub mod geometry;
pub mod loader;
pub mod mesh;
pub mod primitives;
pub mod rebuilder;
pub mod render_buffers;
pub mod scene;

use crate::render_buffers::RenderBuffers;

#[wasm_bindgen(start)]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
pub struct World {
    // 分離したバッファ
    buffers: RenderBuffers,

    // シーンデータ (Nodes, Skins, Animations, Geometries)
    scene: SceneData,

    // 内部状態 (TLAS再構築用)
    blas_root_offsets: Vec<u32>,
    instance_blas_aabbs: Vec<crate::primitives::AABB>,
    raw_instances: Vec<Instance>,
}

#[wasm_bindgen]
impl World {
    #[wasm_bindgen(constructor)]
    pub fn new(
        scene_name: &str,
        mesh_obj_source: Option<String>,
        glb_data: Option<Vec<u8>>,
    ) -> World {
        let loaded_mesh = mesh_obj_source.map(|source| Mesh::new(&source));
        let has_glb = glb_data.is_some();

        let mut scene_data: SceneData =
            scene::get_scene_data(scene_name, loaded_mesh.as_ref(), has_glb);

        if let Some(data) = glb_data {
            let _ = loader::load_gltf(
                &mut scene_data.geometries,
                &mut scene_data.instances,
                &mut scene_data.nodes,
                &mut scene_data.skins,
                &mut scene_data.animations,
                &data,
            );
        }

        let mut raw_instances = Vec::new();
        let mut instance_blas_aabbs = Vec::new();

        for sc_inst in &scene_data.instances {
            raw_instances.push(Instance {
                transform: sc_inst.transform,
                inverse_transform: sc_inst.transform.inverse(),
                blas_node_offset: 0,
                attr_offset: 0,
                instance_id: sc_inst.geometry_index as u32,
                pad: 0,
            });
            instance_blas_aabbs.push(crate::primitives::AABB::empty());
        }

        if raw_instances.is_empty() {
            raw_instances.push(Instance::default());
            instance_blas_aabbs.push(crate::primitives::AABB::empty());
        }

        let mut world = World {
            buffers: RenderBuffers::new(),
            scene: scene_data,
            blas_root_offsets: Vec::new(),
            instance_blas_aabbs,
            raw_instances,
        };

        world.update(0.0);
        world
    }

    pub fn load_animation_glb(&mut self, glb_data: &[u8]) {
        let mut temp_geoms = Vec::new();
        let mut temp_insts = Vec::new();
        let mut temp_nodes = Vec::new();
        let mut temp_skins = Vec::new();
        let mut new_anims = Vec::new();

        if loader::load_gltf(
            &mut temp_geoms,
            &mut temp_insts,
            &mut temp_nodes,
            &mut temp_skins,
            &mut new_anims,
            glb_data,
        )
        .is_ok()
        {
            self.scene.animations.extend(new_anims);
        }
    }

    pub fn update(&mut self, time: f32) {
        // 1. Animation
        if !self.scene.animations.is_empty() {
            let anim_idx = 0;
            let anim = &self.scene.animations[anim_idx];
            let t = time % anim.duration;
            self.apply_animation(anim_idx, t);
        }

        // 2. Global Transforms
        let node_count = self.scene.nodes.len();
        let mut globals = vec![Mat4::IDENTITY; node_count];
        for i in 0..node_count {
            if self.scene.nodes[i].parent_index.is_none() {
                self.update_node_global(i, Mat4::IDENTITY, &mut globals);
            }
        }
        for i in 0..node_count {
            self.scene.nodes[i].global_transform = globals[i];
        }

        // 3. Rebuild Geometry
        rebuilder::build_blas_and_vertices(
            &self.scene.geometries,
            &self.scene.skins,
            &globals,
            &mut self.buffers,
            &mut self.blas_root_offsets,
        );

        // 4. Update Instances
        for (i, inst) in self.raw_instances.iter_mut().enumerate() {
            if i > 0 {
                let model_scale = 1.0;
                let model_transform = Mat4::from_scale(Vec3::splat(model_scale));
                inst.transform = model_transform;
                inst.inverse_transform = model_transform.inverse();
            }

            let geom_idx = inst.instance_id as usize;
            if geom_idx < self.blas_root_offsets.len() {
                inst.blas_node_offset = self.blas_root_offsets[geom_idx];

                let base = inst.blas_node_offset as usize * 8;
                if base < self.buffers.blas_nodes.len() {
                    let min = Vec3::new(
                        self.buffers.blas_nodes[base],
                        self.buffers.blas_nodes[base + 1],
                        self.buffers.blas_nodes[base + 2],
                    );
                    let max = Vec3::new(
                        self.buffers.blas_nodes[base + 4],
                        self.buffers.blas_nodes[base + 5],
                        self.buffers.blas_nodes[base + 6],
                    );
                    if i < self.instance_blas_aabbs.len() {
                        self.instance_blas_aabbs[i] = crate::primitives::AABB { min, max };
                    }
                }
            }
        }

        // 5. TLAS Rebuild
        let mut tlas_builder = TLASBuilder::new(&self.raw_instances, &self.instance_blas_aabbs);
        let (tlas_nodes, sorted_insts) = tlas_builder.build();
        self.buffers.tlas_nodes = tlas_nodes;

        self.buffers.instances = unsafe {
            let ratio = std::mem::size_of::<Instance>() / std::mem::size_of::<f32>();
            let len = sorted_insts.len() * ratio;
            let ptr = sorted_insts.as_ptr() as *const f32;
            std::slice::from_raw_parts(ptr, len).to_vec()
        };
    }

    // --- Pointers for JS ---
    pub fn tlas_ptr(&self) -> *const f32 {
        self.buffers.tlas_nodes.as_ptr()
    }
    pub fn tlas_len(&self) -> usize {
        self.buffers.tlas_nodes.len()
    }

    pub fn blas_ptr(&self) -> *const f32 {
        self.buffers.blas_nodes.as_ptr()
    }
    pub fn blas_len(&self) -> usize {
        self.buffers.blas_nodes.len()
    }

    pub fn instances_ptr(&self) -> *const f32 {
        self.buffers.instances.as_ptr()
    }
    pub fn instances_len(&self) -> usize {
        self.buffers.instances.len()
    }

    pub fn vertices_ptr(&self) -> *const f32 {
        self.buffers.vertices.as_ptr()
    }
    pub fn vertices_len(&self) -> usize {
        self.buffers.vertices.len()
    }

    pub fn normals_ptr(&self) -> *const f32 {
        self.buffers.normals.as_ptr()
    }
    pub fn normals_len(&self) -> usize {
        self.buffers.normals.len()
    }

    pub fn indices_ptr(&self) -> *const u32 {
        self.buffers.indices.as_ptr()
    }
    pub fn indices_len(&self) -> usize {
        self.buffers.indices.len()
    }

    pub fn attributes_ptr(&self) -> *const f32 {
        self.buffers.attributes.as_ptr()
    }
    pub fn attributes_len(&self) -> usize {
        self.buffers.attributes.len()
    }

    pub fn camera_ptr(&self) -> *const f32 {
        self.buffers.camera_data.as_ptr()
    }

    pub fn update_camera(&mut self, width: f32, height: f32) {
        if height == 0.0 {
            return;
        }
        self.buffers.camera_data = self.scene.camera.create_buffer(width / height).to_vec();
    }

    // --- Internal Helpers ---

    fn update_node_global(&self, node_idx: usize, parent_mat: Mat4, globals: &mut Vec<Mat4>) {
        let node = &self.scene.nodes[node_idx];
        let local =
            Mat4::from_scale_rotation_translation(node.scale, node.rotation, node.translation);
        let global = parent_mat * local;
        globals[node_idx] = global;
        for &child in &node.children_indices {
            self.update_node_global(child, global, globals);
        }
    }

    fn apply_animation(&mut self, anim_idx: usize, time: f32) {
        let anim = &self.scene.animations[anim_idx];
        let mut node_name_map = HashMap::new();
        for (i, node) in self.scene.nodes.iter().enumerate() {
            node_name_map.insert(node.name.clone(), i);
        }

        for channel in &anim.channels {
            let target_name = &channel.target_node_name;

            let mut node_idx_opt = node_name_map.get(target_name);
            if node_idx_opt.is_none() {
                // Name resolving logic
                let stripped = target_name
                    .replace("mixamorig:", "")
                    .replace("mixamorig", "");
                if let Some(idx) = node_name_map.get(&stripped) {
                    node_idx_opt = Some(idx);
                } else if let Some(idx) = node_name_map.get(&format!("J_Bip_C_{}", stripped)) {
                    node_idx_opt = Some(idx);
                } else if let Some(rest) = stripped.strip_prefix("Left") {
                    // Left prefix logic
                    let vrm_l = format!("J_Bip_L_{}", rest);
                    node_idx_opt = node_name_map.get(&vrm_l);
                    if node_idx_opt.is_none() && rest == "Arm" {
                        node_idx_opt = node_name_map.get("J_Bip_L_UpperArm");
                    }
                    if node_idx_opt.is_none() && rest == "ForeArm" {
                        node_idx_opt = node_name_map.get("J_Bip_L_LowerArm");
                    }
                    if node_idx_opt.is_none() && rest == "UpLeg" {
                        node_idx_opt = node_name_map.get("J_Bip_L_UpperLeg");
                    }
                } else if let Some(rest) = stripped.strip_prefix("Right") {
                    // Right prefix logic (ここがWarning箇所)
                    let vrm_r = format!("J_Bip_R_{}", rest);
                    node_idx_opt = node_name_map.get(&vrm_r);
                    if node_idx_opt.is_none() && rest == "Arm" {
                        node_idx_opt = node_name_map.get("J_Bip_R_UpperArm");
                    }
                    if node_idx_opt.is_none() && rest == "ForeArm" {
                        node_idx_opt = node_name_map.get("J_Bip_R_LowerArm");
                    }
                    if node_idx_opt.is_none() && rest == "UpLeg" {
                        node_idx_opt = node_name_map.get("J_Bip_R_UpperLeg");
                    }
                }
            }

            let node_idx = match node_idx_opt {
                Some(&idx) => idx,
                None => continue,
            };

            let inputs = &channel.inputs;
            let count = inputs.len();
            if count == 0 {
                continue;
            }

            let mut next_idx = 0;
            while next_idx < count && inputs[next_idx] < time {
                next_idx += 1;
            }
            if next_idx == 0 {
                next_idx = 1;
            }
            if next_idx >= count {
                next_idx = 0;
            }
            let prev_idx = if next_idx == 0 {
                count - 1
            } else {
                next_idx - 1
            };

            let t0 = inputs[prev_idx];
            let t1 = inputs[next_idx];
            let dt = if t1 < t0 {
                anim.duration - t0 + t1
            } else {
                t1 - t0
            };
            let current = if t1 < t0 {
                if time >= t0 {
                    time - t0
                } else {
                    (anim.duration - t0) + time
                }
            } else {
                time - t0
            };
            let factor = if dt > 0.0001 {
                (current / dt).clamp(0.0, 1.0)
            } else {
                0.0
            };

            let node = &mut self.scene.nodes[node_idx];

            match &channel.outputs {
                ChannelOutputs::Translations(vecs) => {
                    // ★修正: if条件を結合
                    if prev_idx < vecs.len()
                        && next_idx < vecs.len()
                        && (target_name.contains("Hips") || target_name.contains("Root"))
                    {
                        node.translation = vecs[prev_idx].lerp(vecs[next_idx], factor);
                    }
                }
                ChannelOutputs::Rotations(quats) => {
                    if prev_idx < quats.len() && next_idx < quats.len() {
                        let mut q = quats[prev_idx].slerp(quats[next_idx], factor);
                        if target_name.contains("UpLeg") || target_name.contains("UpperLeg") {
                            q = q * Quat::from_rotation_x(std::f32::consts::PI);
                        }
                        node.rotation = q;
                    }
                }
                ChannelOutputs::Scales(vecs) => {
                    if prev_idx < vecs.len() && next_idx < vecs.len() {
                        node.scale = vecs[prev_idx].lerp(vecs[next_idx], factor);
                    }
                }
            }
        }
    }
}
