// src/render_buffers.rs
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
#[derive(Default)]
pub struct RenderBuffers {
    pub(crate) vertices: Vec<f32>,
    pub(crate) normals: Vec<f32>,
    pub(crate) uvs: Vec<f32>,
    pub(crate) mesh_topology: Vec<u32>, // Consolidated indices + attributes
    pub(crate) tlas_nodes: Vec<f32>,
    pub(crate) blas_nodes: Vec<f32>,
    pub(crate) instances: Vec<f32>,
    pub(crate) lights: Vec<u32>, // [inst_idx, tri_idx, ...]
    pub(crate) camera_data: Vec<f32>,
}

impl RenderBuffers {
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            normals: Vec::new(),
            uvs: Vec::new(),
            mesh_topology: Vec::new(),
            tlas_nodes: Vec::new(),
            blas_nodes: Vec::new(),
            instances: Vec::new(),
            lights: Vec::new(),
            camera_data: Vec::new(),
        }
    }

    pub fn clear(&mut self) {
        self.vertices.clear();
        self.normals.clear();
        self.uvs.clear();
        self.mesh_topology.clear();
        self.blas_nodes.clear();
        self.lights.clear();
    }
}
