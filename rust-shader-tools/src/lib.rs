// src/lib.rs
use crate::bvh::{BVHBuilder, Instance, TLASBuilder};
use crate::mesh::Mesh;
use crate::primitives::AABB; // AABBをインポート
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

    // ★追加: 更新のために保持するデータ
    raw_instances: Vec<Instance>,
    instance_blas_aabbs: Vec<AABB>,

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
            loader::load_gltf(&mut scene_data.geometries, &mut scene_data.instances, &data);
        }

        // --- Global Buffers ---
        let mut all_vertices = Vec::new();
        let mut all_normals = Vec::new();
        let mut all_indices = Vec::new();
        let mut all_attributes = Vec::new();
        let mut all_joints = Vec::new();
        let mut all_weights = Vec::new();
        let mut all_blas_nodes = Vec::new();

        let mut blas_root_offsets = Vec::new();
        let mut current_node_offset = 0;
        let mut current_tri_offset = 0;

        // --- 1. Build Multiple BLAS ---
        for geom in &scene_data.geometries {
            if geom.vertices.is_empty() {
                blas_root_offsets.push(0);
                continue;
            }

            let v_vec4 = &geom.vertices;
            let n_vec4 = &geom.normals;

            let mut builder = BVHBuilder::new(v_vec4, &geom.indices);
            let (mut nodes_f32, indices, tri_ids) = builder.build_with_ids();

            let vertex_offset = (all_vertices.len() / 4) as u32;
            let global_indices: Vec<u32> = indices.iter().map(|&idx| idx + vertex_offset).collect();

            let node_count = nodes_f32.len() / 8;
            for i in 0..node_count {
                let base = i * 8;
                let tri_count = nodes_f32[base + 7];
                if tri_count > 0.0 {
                    let local_first = nodes_f32[base + 3] as u32;
                    nodes_f32[base + 3] = (local_first + current_tri_offset) as f32;
                }
            }

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

            if !geom.joints.is_empty() {
                all_joints.extend(&geom.joints);
                all_weights.extend(&geom.weights);
            } else {
                let count = v_vec4.len() / 4;
                all_joints.extend(vec![0; count * 4]);
                all_weights.extend(vec![0.0; count * 4]);
            }

            blas_root_offsets.push(current_node_offset as u32);
            current_node_offset += node_count as u32;
            current_tri_offset += (indices.len() / 3) as u32;
        }

        // --- 2. Instance Creation ---
        let mut raw_instances = Vec::new();
        let mut instance_blas_aabbs = Vec::new();

        let get_blas_aabb = |nodes: &[f32], offset: usize| -> AABB {
            if offset * 8 >= nodes.len() {
                return AABB::empty();
            }
            let base = offset * 8;
            AABB {
                min: Vec3::new(nodes[base], nodes[base + 1], nodes[base + 2]),
                max: Vec3::new(nodes[base + 4], nodes[base + 5], nodes[base + 6]),
            }
        };

        for sc_inst in &scene_data.instances {
            if sc_inst.geometry_index < blas_root_offsets.len() {
                let root_offset = blas_root_offsets[sc_inst.geometry_index];
                let aabb = get_blas_aabb(&all_blas_nodes, root_offset as usize);

                raw_instances.push(Instance {
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

        if raw_instances.is_empty() {
            raw_instances.push(Instance {
                transform: Mat4::IDENTITY,
                inverse_transform: Mat4::IDENTITY,
                blas_node_offset: 0,
                attr_offset: 0,
                instance_id: 0,
                pad: 0,
            });
            instance_blas_aabbs.push(AABB::empty());
        }

        // --- 3. TLAS Build ---
        let mut tlas_builder = TLASBuilder::new(&raw_instances, &instance_blas_aabbs);
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
            // 状態保持
            raw_instances,
            instance_blas_aabbs,

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

    // --- Update Method ---
    // 時間を受け取り、モデル（環境以外）を回転させ、TLASを再構築する
    pub fn update(&mut self, time: f32) {
        // インスタンス数が1以下（環境のみなど）なら何もしない
        if self.raw_instances.len() <= 1 {
            return;
        }

        // 簡易アニメーション: Y軸回転
        // index 0 は環境(Environment)と仮定し、1以降(モデルのパーツ)を回転させる
        let rotation = Mat4::from_rotation_y(time * 0.5); // 0.5 rad/sec

        // raw_instances[0] は動かさない
        for i in 1..self.raw_instances.len() {
            // 初期姿勢に対して回転を掛ける必要があるが、
            // ここでは簡易的に「原点中心に回転」させる
            // 本来は各インスタンスの初期Transformを保持しておくべきだが、
            // loaderが「ワールド座標」にベイクしているため、ここでの回転は
            // 「モデル全体を原点中心に回す」動きになる（モデルが原点にあればOK）

            // 既存のTransformに回転を乗算すると、毎フレーム回転が累積してしまうので注意。
            // 正しくは「初期Transform * 回転」だが、初期Transformを保持していない。
            // ここでは「loaderで生成された時点がt=0」とし、今回は簡易的に
            // 毎回回転行列を生成して上書きする（※これはモデルが原点にある前提）
            // もしモデルが原点からずれていれば、公転してしまう。

            // より安全な方法:
            // 今回は「動くこと」を確認するため、単純にrotation行列を入れる（位置情報が消えるリスクあり）
            // いや、loader.rsで transform = parent * local しているので、位置情報も入っている。
            // これを上書きすると位置がリセットされる。

            // 修正案: 現在の値を保持し続けるのは難しいので、
            // 「インスタンスのTransformの回転成分だけを更新」するアプローチをとるか、
            // 単純に「TLASテスト」として、すべてのインスタンスに回転行列を掛けてみる。

            // 一旦、t=0の状態を保持していないため、
            // 毎フレーム「現在のTransform」に対して微小回転を加える
            let delta_rot = Mat4::from_rotation_y(0.01); // 毎フレーム少し回す
            self.raw_instances[i].transform = delta_rot * self.raw_instances[i].transform;
            self.raw_instances[i].inverse_transform = self.raw_instances[i].transform.inverse();
        }

        // TLAS再構築 (高速)
        let mut tlas_builder = TLASBuilder::new(&self.raw_instances, &self.instance_blas_aabbs);
        let (nodes, sorted_insts) = tlas_builder.build();

        self.tlas_nodes = nodes;

        // インスタンス配列の更新 (Shaderへ送る用)
        self.instances = unsafe {
            let ratio = std::mem::size_of::<Instance>() / std::mem::size_of::<f32>();
            let len = sorted_insts.len() * ratio;
            let ptr = sorted_insts.as_ptr() as *const f32;
            std::slice::from_raw_parts(ptr, len).to_vec()
        };
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
