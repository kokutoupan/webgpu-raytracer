// src/lib.rs
use crate::bvh::BVHBuilder;
use crate::mesh::Mesh;
use crate::scene::SceneData;
use wasm_bindgen::prelude::*;

pub mod bvh;
pub mod geometry;
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
    vertices: Vec<f32>,
    indices: Vec<u32>,
    attributes: Vec<f32>,
    camera_data: Vec<f32>,
    current_camera: scene::CameraConfig,
}

#[wasm_bindgen]
impl World {
    #[wasm_bindgen(constructor)]
    pub fn new(scene_name: &str, mesh_obj_source: Option<String>) -> World {
        let loaded_mesh = mesh_obj_source.map(|source| Mesh::new(&source));
        let scene_data: SceneData = scene::get_scene_data(scene_name, loaded_mesh.as_ref());

        // BVH構築
        // bvh::BVHBuilder は tri_indices (三角形IDの並び順) を内部で計算しています。
        // これを公開するように bvh.rs を少し修正するか、
        // ここでは「ソート済みインデックス」だけ受け取って、属性の整合性は一旦無視するか...
        //
        // ★修正方針: BVHBuilder::build は (packed_nodes, sorted_indices, sorted_tri_ids) を返すように変更するのがベストですが、
        // 今回は「geometry.attributes」のデータ構造が [tri0_attr, tri1_attr...] と並んでいるため、
        // インデックスバッファだけソートしても、属性との対応がずれてしまいます。

        // 簡易対応:
        // bvh.rs の build() が返す sorted_indices は「頂点インデックス」の羅列です。
        // これを使えば頂点は正しく引けますが、「この三角形の色は？」が分からなくなります。

        // なので、World内で「属性も並び替える」処理が必要です。
        // bvh.rs の build() を少し改造して「ソートされた三角形IDリスト」も返すようにします。

        let mut builder =
            BVHBuilder::new(&scene_data.geometry.vertices, &scene_data.geometry.indices);

        // ※ bvh.rs の build() が (nodes, indices, tri_ids) を返すように修正したと仮定
        // 実際には bvh.rs の最後に `self.tri_indices` を返せばOKです。
        let (packed_nodes, sorted_indices, sorted_tri_ids) = builder.build_with_ids();

        // 属性の並び替え
        let attr_stride = 8; // float x 8
        let mut sorted_attributes = vec![0.0; scene_data.geometry.attributes.len()];

        for (new_idx, &old_tri_id) in sorted_tri_ids.iter().enumerate() {
            let src_start = old_tri_id * attr_stride;
            let dst_start = new_idx * attr_stride;
            sorted_attributes[dst_start..dst_start + attr_stride].copy_from_slice(
                &scene_data.geometry.attributes[src_start..src_start + attr_stride],
            );
        }

        let cam_buffer = scene_data.camera.create_buffer(1.5);

        World {
            bvh_nodes: packed_nodes,
            vertices: scene_data.geometry.vertices, // 頂点プールは不動
            indices: sorted_indices,                // ソート済み
            attributes: sorted_attributes,          // ソート済み
            camera_data: cam_buffer.to_vec(),
            current_camera: scene_data.camera,
        }
    }

    // ... getter ...
    pub fn bvh_ptr(&self) -> *const f32 {
        self.bvh_nodes.as_ptr()
    }
    pub fn bvh_len(&self) -> usize {
        self.bvh_nodes.len()
    }
    pub fn vertices_ptr(&self) -> *const f32 {
        self.vertices.as_ptr()
    }
    pub fn vertices_len(&self) -> usize {
        self.vertices.len()
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
    pub fn camera_ptr(&self) -> *const f32 {
        self.camera_data.as_ptr()
    }

    pub fn update_camera(&mut self, width: f32, height: f32) {
        if height == 0.0 {
            return;
        }
        let aspect = width / height;
        let new_buffer = self.current_camera.create_buffer(aspect);
        self.camera_data = new_buffer.to_vec();
    }
}
