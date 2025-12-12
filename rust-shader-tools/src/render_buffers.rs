// src/render_buffers.rs
use wasm_bindgen::prelude::*;

// GPUに送るバッファ群をまとめて管理する構造体
#[wasm_bindgen]
#[derive(Default)]
pub struct RenderBuffers {
    // これらのフィールドはWASM経由でJSからアクセスされるため public にする
    // (ただしRust側では直接操作するため、WASM用getterはlib.rsでラップする形でもOKですが、
    //  ここでは内部データ保持用として定義します)
    pub(crate) vertices: Vec<f32>,
    pub(crate) normals: Vec<f32>,
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
        self.indices.clear();
        self.attributes.clear();
        self.blas_nodes.clear();
        // tlas, instances, cameraは毎フレーム全書き換えに近いのでここでクリアしなくても良いが、
        // 必要に応じてクリアする運用にします
    }
}
