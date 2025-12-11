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
            loader::load_gltf(&mut scene_data.geometry, &data);
        }

        // --- 1. BLAS Build ---
        // Currently treating the entire scene geometry as one Mesh (one BLAS)
        let mut builder =
            BVHBuilder::new(&scene_data.geometry.vertices, &scene_data.geometry.indices);
        let (packed_nodes, sorted_indices, sorted_tri_ids) = builder.build_with_ids();
        let root_aabb = builder.root_aabb();

        // Attribute Sorting
        let attr_stride = 8;
        let mut sorted_attributes = vec![0.0; scene_data.geometry.attributes.len()];
        for (new_idx, &old_tri_id) in sorted_tri_ids.iter().enumerate() {
            let src = old_tri_id * attr_stride;
            let dst = new_idx * attr_stride;
            if src + attr_stride <= scene_data.geometry.attributes.len() {
                sorted_attributes[dst..dst + attr_stride]
                    .copy_from_slice(&scene_data.geometry.attributes[src..src + attr_stride]);
            }
        }

        // Since we only have 1 BLAS, offsets are 0
        let blas_node_offset = 0;
        let attr_offset = 0;

        // --- 2. Instance Creation ---
        // Creating 2 instances for demonstration
        let mut instances = Vec::new();
        let mut instance_blas_aabbs = Vec::new();

        // Instance 0: Identity
        instances.push(Instance {
            transform: Mat4::IDENTITY,
            inverse_transform: Mat4::IDENTITY,
            blas_node_offset,
            attr_offset,
            instance_id: 0,
            pad: 0,
        });
        instance_blas_aabbs.push(root_aabb);

        // Instance 1: Shifted (if not pure viewer mode)
        // If scene is 'viewer', we might just want 1 instance centered.
        if scene_name != "viewer" {
            let shift = Mat4::from_translation(Vec3::new(
                1.2 * (root_aabb.max.x - root_aabb.min.x),
                0.0,
                0.0,
            ));
            instances.push(Instance {
                transform: shift,
                inverse_transform: shift.inverse(),
                blas_node_offset,
                attr_offset,
                instance_id: 1,
                pad: 0,
            });
            instance_blas_aabbs.push(root_aabb);
        }

        // --- 3. TLAS Build ---
        let mut tlas_builder = TLASBuilder::new(&instances, &instance_blas_aabbs);
        let (tlas_nodes, sorted_instances) = tlas_builder.build();

        // Convert Instances to f32 for GPU
        let instance_floats = unsafe {
            let ratio = std::mem::size_of::<Instance>() / std::mem::size_of::<f32>();
            let len = sorted_instances.len() * ratio;
            let ptr = sorted_instances.as_ptr() as *const f32;
            std::slice::from_raw_parts(ptr, len).to_vec()
        };

        let cam_buffer = scene_data.camera.create_buffer(1.5);

        World {
            tlas_nodes,
            blas_nodes: packed_nodes,
            instances: instance_floats,
            vertices: scene_data.geometry.vertices,
            normals: scene_data.geometry.normals,
            indices: sorted_indices,
            attributes: sorted_attributes,
            joints: scene_data.geometry.joints,
            weights: scene_data.geometry.weights,
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
