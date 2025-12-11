// src/lib.rs
use crate::bvh::{BVHBuilder, Instance, TLASBuilder};
use crate::mesh::Mesh;
use crate::scene::SceneData;
use glam::{Mat4, Vec3};
use wasm_bindgen::prelude::*;

pub mod bvh;
pub mod geometry;
pub mod loader;
pub mod mesh;
pub mod primitives;
pub mod scene;

#[wasm_bindgen(start)]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
pub struct World {
    tlas_nodes: Vec<f32>,
    blas_nodes: Vec<f32>,
    instances: Vec<f32>,

    vertices: Vec<f32>,
    normals: Vec<f32>,
    indices: Vec<u32>,
    attributes: Vec<f32>,

    joints: Vec<u32>,
    weights: Vec<f32>,

    camera_data: Vec<f32>,
    current_camera: scene::CameraConfig,
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
            // GLBがある場合、Geometries[1] (Model用) にロードする想定
            if scene_data.geometries.len() < 2 {
                scene_data.geometries.push(geometry::Geometry::new());
            }
            let target_geom = if scene_data.geometries.len() > 1 {
                &mut scene_data.geometries[1]
            } else {
                &mut scene_data.geometries[0]
            };
            loader::load_gltf(target_geom, &data);
        }

        // --- Global Buffers ---
        let mut all_vertices = Vec::new();
        let mut all_normals = Vec::new();
        let mut all_indices = Vec::new();
        let mut all_attributes = Vec::new();
        let mut all_joints = Vec::new();
        let mut all_weights = Vec::new();
        let mut all_blas_nodes = Vec::new();

        // BLAS Offset Tracking
        let mut blas_root_offsets = Vec::new(); // 各BLASのルートノードインデックス(Global)
        let mut current_node_offset = 0;
        let mut current_tri_offset = 0;

        // --- 1. Build Multiple BLAS ---
        for geom in &scene_data.geometries {
            if geom.vertices.is_empty() {
                blas_root_offsets.push(0);
                continue;
            }

            // Geometry.vertices/normals are already stride 4 (vec4)
            let v_vec4 = &geom.vertices;
            let n_vec4 = &geom.normals;

            // Build BVH
            let mut builder = BVHBuilder::new(v_vec4, &geom.indices);
            let (mut nodes_f32, indices, tri_ids) = builder.build_with_ids();

            // Adjust Indices
            let vertex_offset = (all_vertices.len() / 4) as u32;
            let global_indices: Vec<u32> = indices.iter().map(|&idx| idx + vertex_offset).collect();

            // Adjust Leaf Nodes
            let node_count = nodes_f32.len() / 8;
            for i in 0..node_count {
                let base = i * 8;
                let tri_count = nodes_f32[base + 7];
                if tri_count > 0.0 {
                    let local_first = nodes_f32[base + 3] as u32;
                    nodes_f32[base + 3] = (local_first + current_tri_offset) as f32;
                }
            }

            // Sort Attributes
            let attr_stride = 8;
            let mut sorted_attributes = vec![0.0; geom.attributes.len()];
            if sorted_attributes.len() >= tri_ids.len() * attr_stride {
                for (new_idx, &old_tri_id) in tri_ids.iter().enumerate() {
                    let src = old_tri_id * attr_stride;
                    let dst = new_idx * attr_stride;
                    if src + attr_stride <= geom.attributes.len() {
                        sorted_attributes[dst..dst + attr_stride]
                            .copy_from_slice(&geom.attributes[src..src + attr_stride]);
                    }
                }
            }

            all_vertices.extend(v_vec4);
            all_normals.extend(n_vec4);
            all_indices.extend(global_indices);
            all_attributes.extend(sorted_attributes);
            all_blas_nodes.extend(nodes_f32);

            // Joints/Weights (Pad if needed)
            if !geom.joints.is_empty() {
                all_joints.extend(&geom.joints);
                all_weights.extend(&geom.weights);
            } else {
                let count = v_vec4.len() / 4;
                all_joints.extend(vec![0; count * 4]);
                all_weights.extend(vec![0.0; count * 4]);
            }

            blas_root_offsets.push(current_node_offset);
            current_node_offset += node_count as u32;
            current_tri_offset += (indices.len() / 3) as u32;
        }

        // --- 2. Instance Creation ---
        let mut instances = Vec::new();
        let mut instance_blas_aabbs = Vec::new();

        // BLAS AABB Helper
        let get_blas_aabb = |nodes: &[f32], offset: usize| -> crate::primitives::AABB {
            if offset * 8 >= nodes.len() {
                return crate::primitives::AABB::empty();
            }
            let base = offset * 8;
            crate::primitives::AABB {
                min: Vec3::new(nodes[base], nodes[base + 1], nodes[base + 2]),
                max: Vec3::new(nodes[base + 4], nodes[base + 5], nodes[base + 6]),
            }
        };

        for sc_inst in &scene_data.instances {
            if sc_inst.geometry_index < blas_root_offsets.len() {
                let root_offset = blas_root_offsets[sc_inst.geometry_index];
                let aabb = get_blas_aabb(&all_blas_nodes, root_offset as usize);

                instances.push(Instance {
                    transform: sc_inst.transform,
                    inverse_transform: sc_inst.transform.inverse(),
                    blas_node_offset: root_offset,
                    attr_offset: 0,
                    instance_id: sc_inst.geometry_index as u32,
                    pad: 0,
                });
                instance_blas_aabbs.push(aabb);
            }
        }

        // Fallback
        if instances.is_empty() {
            instances.push(Instance {
                transform: Mat4::IDENTITY,
                inverse_transform: Mat4::IDENTITY,
                blas_node_offset: 0,
                attr_offset: 0,
                instance_id: 0,
                pad: 0,
            });
            instance_blas_aabbs.push(crate::primitives::AABB::empty());
        }

        // --- 3. TLAS Build ---
        let mut tlas_builder = TLASBuilder::new(&instances, &instance_blas_aabbs);
        let (tlas_nodes, sorted_instances) = tlas_builder.build();

        let instance_floats = unsafe {
            let ratio = std::mem::size_of::<Instance>() / std::mem::size_of::<f32>();
            let len = sorted_instances.len() * ratio;
            let ptr = sorted_instances.as_ptr() as *const f32;
            std::slice::from_raw_parts(ptr, len).to_vec()
        };

        let cam_buffer = scene_data.camera.create_buffer(1.5);

        World {
            tlas_nodes,
            blas_nodes: all_blas_nodes,
            instances: instance_floats,
            vertices: all_vertices,
            normals: all_normals,
            indices: all_indices,
            attributes: all_attributes,
            joints: all_joints,
            weights: all_weights,
            camera_data: cam_buffer.to_vec(),
            current_camera: scene_data.camera,
        }
    }

    // Pointers
    pub fn tlas_ptr(&self) -> *const f32 {
        self.tlas_nodes.as_ptr()
    }
    pub fn tlas_len(&self) -> usize {
        self.tlas_nodes.len()
    }
    pub fn blas_ptr(&self) -> *const f32 {
        self.blas_nodes.as_ptr()
    }
    pub fn blas_len(&self) -> usize {
        self.blas_nodes.len()
    }
    pub fn instances_ptr(&self) -> *const f32 {
        self.instances.as_ptr()
    }
    pub fn instances_len(&self) -> usize {
        self.instances.len()
    }
    pub fn vertices_ptr(&self) -> *const f32 {
        self.vertices.as_ptr()
    }
    pub fn vertices_len(&self) -> usize {
        self.vertices.len()
    }
    pub fn normals_ptr(&self) -> *const f32 {
        self.normals.as_ptr()
    }
    pub fn normals_len(&self) -> usize {
        self.normals.len()
    }
    pub fn indices_ptr(&self) -> *const u32 {
        self.indices.as_ptr()
    }
    pub fn indices_len(&self) -> usize {
        self.indices.len()
    }
    pub fn attributes_ptr(&self) -> *const f32 {
        self.attributes.as_ptr()
    }
    pub fn attributes_len(&self) -> usize {
        self.attributes.len()
    }
    pub fn joints_ptr(&self) -> *const u32 {
        self.joints.as_ptr()
    }
    pub fn joints_len(&self) -> usize {
        self.joints.len()
    }
    pub fn weights_ptr(&self) -> *const f32 {
        self.weights.as_ptr()
    }
    pub fn weights_len(&self) -> usize {
        self.weights.len()
    }
    pub fn camera_ptr(&self) -> *const f32 {
        self.camera_data.as_ptr()
    }

    pub fn update_camera(&mut self, width: f32, height: f32) {
        if height == 0.0 {
            return;
        }
        self.camera_data = self.current_camera.create_buffer(width / height).to_vec();
    }
}
