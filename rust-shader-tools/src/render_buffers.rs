// src/render_buffers.rs
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
#[derive(Default)]
pub struct RenderBuffers {
    pub(crate) vertices: Vec<f32>,
    pub(crate) normals: Vec<f32>,
    pub(crate) uvs: Vec<f32>, // ★追加: [u, v, u, v, ...]
    pub(crate) indices: Vec<u32>,
    pub(crate) attributes: Vec<f32>,
    pub(crate) tlas_nodes: Vec<f32>,
    pub(crate) blas_nodes: Vec<f32>,
    pub(crate) instances: Vec<f32>,
    pub(crate) camera_data: Vec<f32>,
}

impl RenderBuffers {
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            normals: Vec::new(),
            uvs: Vec::new(),
            indices: Vec::new(),
            attributes: Vec::new(),
            tlas_nodes: Vec::new(),
            blas_nodes: Vec::new(),
            instances: Vec::new(),
            camera_data: Vec::new(),
        }
    }

    pub fn clear(&mut self) {
        self.vertices.clear();
        self.normals.clear();
        self.uvs.clear();
        self.indices.clear();
        self.attributes.clear();
        self.blas_nodes.clear();
    }
}
