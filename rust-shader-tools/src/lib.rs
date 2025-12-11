use crate::bvh::BVHBuilder;
use crate::mesh::Mesh;
use crate::scene::SceneData; // SceneDataを使う
use wasm_bindgen::prelude::*;

pub mod bvh;
pub mod mesh;
pub mod primitives;
pub mod scene;

#[wasm_bindgen(start)]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
pub struct World {
    bvh_nodes: Vec<f32>,
    primitives: Vec<f32>,
    camera_data: Vec<f32>,
    // カメラ設定を保持して、リサイズ時に再計算できるようにする
    current_camera: scene::CameraConfig,
}

#[wasm_bindgen]
impl World {
    #[wasm_bindgen(constructor)]
    pub fn new(scene_name: &str, mesh_obj_source: Option<String>) -> World {
        // メッシュ生成
        let loaded_mesh = mesh_obj_source.map(|source| Mesh::new(&source));

        // シーン取得
        let scene_data: SceneData = scene::get_scene_data(scene_name, loaded_mesh.as_ref());

        // BVH構築
        let mut builder = BVHBuilder::new(scene_data.primitives);
        let (packed_nodes, packed_prims) = builder.build();

        // カメラデータ作成 (初期アスペクト比は仮で1.5。JS側で直後にupdate_cameraを呼んでも良い)
        let cam_buffer = scene_data.camera.create_buffer(1.5);

        World {
            bvh_nodes: packed_nodes,
            primitives: packed_prims,
            camera_data: cam_buffer.to_vec(),
            current_camera: scene_data.camera, // 保持
        }
    }

    // --- JS連携メソッド ---

    pub fn bvh_ptr(&self) -> *const f32 {
        self.bvh_nodes.as_ptr()
    }

    pub fn bvh_len(&self) -> usize {
        self.bvh_nodes.len()
    }

    pub fn prim_ptr(&self) -> *const f32 {
        self.primitives.as_ptr()
    }

    pub fn prim_len(&self) -> usize {
        self.primitives.len()
    }

    pub fn camera_ptr(&self) -> *const f32 {
        self.camera_data.as_ptr()
    }

    // ★追加: アスペクト比が変わったときにカメラバッファを更新する
    pub fn update_camera(&mut self, width: f32, height: f32) {
        if height == 0.0 {
            return;
        }
        let aspect = width / height;

        // 新しいバッファを生成して上書き
        let new_buffer = self.current_camera.create_buffer(aspect);
        self.camera_data = new_buffer.to_vec();
    }
}
