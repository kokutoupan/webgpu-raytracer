// src/lib.rs
use crate::bvh::{BVHBuilder, Instance, TLASBuilder};
use crate::mesh::Mesh;
use crate::scene::animation::{Animation, ChannelOutputs};
use crate::scene::{Node, SceneData, Skin};
use glam::{Mat4, Quat, Vec3};
use std::collections::HashMap;
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
    // GPU Buffers (WasmからJSへ渡すポインタ用)
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

    // CPU Side Data (アニメーション計算用)
    nodes: Vec<Node>,
    skins: Vec<Skin>,
    animations: Vec<Animation>,
    scene_geometries: Vec<geometry::Geometry>,

    // Internal State (TLAS再構築用)
    blas_root_offsets: Vec<u32>,
    instance_blas_aabbs: Vec<crate::primitives::AABB>,
    raw_instances: Vec<Instance>, // CPU側で管理するインスタンス情報

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

        // GLBロード (エラーはログ出力して無視し、処理を続行)
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

        // --- 1. Prepare Initial Instances ---
        // scene_dataをWorldにムーブする前に、そこから初期インスタンスリストを作る
        let mut raw_instances = Vec::new();
        let mut instance_blas_aabbs = Vec::new();

        for sc_inst in &scene_data.instances {
            raw_instances.push(Instance {
                transform: sc_inst.transform,
                inverse_transform: sc_inst.transform.inverse(),
                blas_node_offset: 0, // rebuildで計算して埋める
                attr_offset: 0,      // Global Indexingなので0でOK
                instance_id: sc_inst.geometry_index as u32, // geometry_indexを保持しておく
                pad: 0,
            });
            // AABBはまだ不明なので空を入れておく (rebuildで計算)
            instance_blas_aabbs.push(crate::primitives::AABB::empty());
        }

        // インスタンスが空の場合のフォールバック (安全策)
        if raw_instances.is_empty() {
            raw_instances.push(Instance::default());
            instance_blas_aabbs.push(crate::primitives::AABB::empty());
        }

        // --- 2. Create World Struct ---
        let mut world = World {
            // Buffer初期化 (空)
            tlas_nodes: vec![],
            blas_nodes: vec![],
            instances: vec![],
            vertices: vec![],
            normals: vec![],
            indices: vec![],
            attributes: vec![],
            joints: vec![],
            weights: vec![],
            camera_data: vec![],

            // データ移動
            nodes: scene_data.nodes,
            skins: scene_data.skins,
            animations: scene_data.animations,
            scene_geometries: scene_data.geometries,

            // 内部状態
            blas_root_offsets: vec![],
            instance_blas_aabbs,
            raw_instances,

            current_camera: scene_data.camera,
        };

        // --- 3. Initial Calculation ---

        // ノードのグローバル変換行列を初期計算
        let node_count = world.nodes.len();
        let mut globals = vec![Mat4::IDENTITY; node_count];
        // 階層構造に従って計算
        for i in 0..node_count {
            if world.nodes[i].parent_index.is_none() {
                world.update_node_global(i, Mat4::IDENTITY, &mut globals);
            }
        }
        for i in 0..node_count {
            world.nodes[i].global_transform = globals[i];
        }

        // ★重要: 初回のフルビルドを実行
        // これにより vertices, blas_nodes, blas_root_offsets 等が生成される
        world.rebuild_world_from_animation(&globals);

        // ★重要: TLASを構築 (raw_instancesの位置とAABBを使って)
        world.rebuild_tlas();

        // カメラバッファ更新
        world.update_camera(1.0, 1.0); // 初期アスペクト比は仮

        world
    }

    // 後からアニメーションGLBを追加ロードする関数
    pub fn load_animation_glb(&mut self, glb_data: &[u8]) {
        // ダミーのコンテナを用意してloaderを呼ぶ
        let mut temp_geoms = Vec::new();
        let mut temp_insts = Vec::new();
        let mut temp_nodes = Vec::new();
        let mut temp_skins = Vec::new();
        let mut new_anims = Vec::new();

        if let Ok(_) = loader::load_gltf(
            &mut temp_geoms,
            &mut temp_insts,
            &mut temp_nodes,
            &mut temp_skins,
            &mut new_anims,
            glb_data,
        ) {
            // 成功したらアニメーションリストに追加
            self.animations.extend(new_anims);
        }
    }

    // 毎フレームの更新処理
    // src/lib.rs (update関数のみ抜粋)

    pub fn update(&mut self, time: f32) {
        if self.nodes.is_empty() {
            return;
        }

        // 1. Animation Application
        if !self.animations.is_empty() {
            let anim_idx = 0;
            let anim = &self.animations[anim_idx];
            let t = time % anim.duration;
            self.apply_animation(anim_idx, t);
        }

        // 2. Global Transforms
        let node_count = self.nodes.len();
        let mut globals = vec![Mat4::IDENTITY; node_count];
        for i in 0..node_count {
            if self.nodes[i].parent_index.is_none() {
                self.update_node_global(i, Mat4::IDENTITY, &mut globals);
            }
        }
        for i in 0..node_count {
            self.nodes[i].global_transform = globals[i];
        }

        // 3. Rebuild World (CPU Skinning)
        self.rebuild_world_from_animation(&globals);

        // ★修正: 強制縮小を削除 (Scale = 1.0)
        // もしサイズ調整が必要ならここを 1.5 とか 0.5 に変えてください
        let model_scale = 1.0;
        let model_transform = Mat4::from_scale(Vec3::splat(model_scale));

        if self.raw_instances.len() > 1 {
            for i in 1..self.raw_instances.len() {
                self.raw_instances[i].transform = model_transform;
                self.raw_instances[i].inverse_transform = model_transform.inverse();
            }
        }

        // 5. TLAS Rebuild
        let mut tlas_builder = TLASBuilder::new(&self.raw_instances, &self.instance_blas_aabbs);
        let (tlas_nodes, sorted_insts) = tlas_builder.build();
        self.tlas_nodes = tlas_nodes;

        self.instances = unsafe {
            let ratio = std::mem::size_of::<Instance>() / std::mem::size_of::<f32>();
            let len = sorted_insts.len() * ratio;
            let ptr = sorted_insts.as_ptr() as *const f32;
            std::slice::from_raw_parts(ptr, len).to_vec()
        };
    }

    // 内部: TLAS構築処理
    fn rebuild_tlas(&mut self) {
        // raw_instances と instance_blas_aabbs を使ってTLASをビルド
        let mut tlas_builder = TLASBuilder::new(&self.raw_instances, &self.instance_blas_aabbs);
        let (tlas_nodes, sorted_insts) = tlas_builder.build();
        self.tlas_nodes = tlas_nodes;

        // Shaderに送るインスタンス配列を更新
        self.instances = unsafe {
            let ratio = std::mem::size_of::<Instance>() / std::mem::size_of::<f32>();
            let len = sorted_insts.len() * ratio;
            let ptr = sorted_insts.as_ptr() as *const f32;
            std::slice::from_raw_parts(ptr, len).to_vec()
        };
    }

    fn apply_animation(&mut self, anim_idx: usize, time: f32) {
        let anim = &self.animations[anim_idx];
        let mut node_name_map = HashMap::new();
        for (i, node) in self.nodes.iter().enumerate() {
            node_name_map.insert(node.name.clone(), i);
        }

        for channel in &anim.channels {
            let target_name = &channel.target_node_name;

            // リターゲティング (Mixamo -> VRM)
            let mut node_idx_opt = node_name_map.get(target_name);
            if node_idx_opt.is_none() {
                let stripped = target_name
                    .replace("mixamorig:", "")
                    .replace("mixamorig", "");
                // そのまま検索
                if let Some(idx) = node_name_map.get(&stripped) {
                    node_idx_opt = Some(idx);
                }
                // VRM標準名 "J_Bip_C_Hips" など
                else if let Some(idx) = node_name_map.get(&format!("J_Bip_C_{}", stripped)) {
                    node_idx_opt = Some(idx);
                }
                // 左右推定 "LeftArm" -> "J_Bip_L_UpperArm" など
                // (簡易実装: プレフィックス置換)
                else if stripped.starts_with("Left") {
                    let rest = &stripped[4..];
                    let vrm_l = format!("J_Bip_L_{}", rest);
                    node_idx_opt = node_name_map.get(&vrm_l);
                    // Upper/Lower対応
                    if node_idx_opt.is_none() && rest == "Arm" {
                        node_idx_opt = node_name_map.get("J_Bip_L_UpperArm");
                    }
                    if node_idx_opt.is_none() && rest == "ForeArm" {
                        node_idx_opt = node_name_map.get("J_Bip_L_LowerArm");
                    }
                    if node_idx_opt.is_none() && rest == "UpLeg" {
                        node_idx_opt = node_name_map.get("J_Bip_L_UpperLeg");
                    }
                } else if stripped.starts_with("Right") {
                    let rest = &stripped[5..];
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

            let node = &mut self.nodes[node_idx];

            match &channel.outputs {
                ChannelOutputs::Translations(vecs) => {
                    if prev_idx < vecs.len() && next_idx < vecs.len() {
                        if target_name.contains("Hips") || target_name.contains("Root") {
                            let v0 = vecs[prev_idx];
                            let v1 = vecs[next_idx];
                            // 移動量のスケール調整 (Mixamoはデカいことが多いので0.01倍など)
                            // ここではそのまま lerp していますが、必要なら * 0.01 してください
                            node.translation = v0.lerp(v1, factor);
                        }
                    }
                }
                ChannelOutputs::Rotations(quats) => {
                    if prev_idx < quats.len() && next_idx < quats.len() {
                        let mut q = quats[prev_idx].slerp(quats[next_idx], factor);

                        // ★足の180度回転修正
                        // Mixamoの足ボーン軸が逆向きの場合があるため補正
                        if target_name.contains("UpLeg") || target_name.contains("UpperLeg") {
                            let fix_rot = Quat::from_rotation_x(std::f32::consts::PI);
                            q = q * fix_rot;
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

    fn update_node_global(&self, node_idx: usize, parent_mat: Mat4, globals: &mut Vec<Mat4>) {
        let node = &self.nodes[node_idx];
        let local =
            Mat4::from_scale_rotation_translation(node.scale, node.rotation, node.translation);
        let global = parent_mat * local;
        globals[node_idx] = global;
        for &child in &node.children_indices {
            self.update_node_global(child, global, globals);
        }
    }

    // 全ジオメトリの頂点変形、BLAS再構築、Globalバッファへの結合を行う
    fn rebuild_world_from_animation(&mut self, globals: &[Mat4]) {
        // バッファクリア
        self.vertices.clear();
        self.normals.clear();
        self.indices.clear();
        self.attributes.clear();
        self.blas_nodes.clear();
        self.blas_root_offsets.clear();
        self.instance_blas_aabbs.clear();

        let mut current_node_offset = 0;
        let mut current_tri_offset = 0;

        for geom in &self.scene_geometries {
            // base_positionsが空の場合（geometry.rs更新漏れ等）の安全策
            let base_positions_backup: Vec<Vec3>;
            let base_normals_backup: Vec<Vec3>;

            let (use_positions, use_normals) = if geom.base_positions.is_empty() {
                if !geom.vertices.is_empty() {
                    base_positions_backup = geom
                        .vertices
                        .chunks(4)
                        .map(|c| Vec3::new(c[0], c[1], c[2]))
                        .collect();
                    base_normals_backup = geom
                        .normals
                        .chunks(4)
                        .map(|c| Vec3::new(c[0], c[1], c[2]))
                        .collect();
                    (&base_positions_backup, &base_normals_backup)
                } else {
                    // 頂点自体がない場合はダミーオフセットを入れてスキップ
                    self.blas_root_offsets.push(0);
                    continue;
                }
            } else {
                (&geom.base_positions, &geom.base_normals)
            };

            // スキニング適用チェック
            // 「スキンインデックスを持っている」場合に適用
            let skin_opt = if let Some(s_idx) = geom.skin_index {
                if s_idx < self.skins.len() {
                    Some(&self.skins[s_idx])
                } else {
                    None
                }
            } else {
                None
            };

            let mut v_vec4 = Vec::with_capacity(use_positions.len() * 4);
            let mut n_vec4 = Vec::with_capacity(use_normals.len() * 4);

            if let Some(skin) = skin_opt {
                // スキニング計算
                let joint_mats: Vec<Mat4> = skin
                    .joints
                    .iter()
                    .zip(skin.inverse_bind_matrices.iter())
                    .map(|(&j, &inv)| globals[j] * inv)
                    .collect();

                for i in 0..use_positions.len() {
                    let pos = use_positions[i];
                    let norm = use_normals[i];
                    let j = &geom.joints[i * 4..(i + 1) * 4];
                    let w = &geom.weights[i * 4..(i + 1) * 4];

                    let mut mat = Mat4::ZERO;
                    for k in 0..4 {
                        if w[k] > 0. {
                            mat += joint_mats[j[k] as usize] * w[k];
                        }
                    }
                    if mat == Mat4::ZERO {
                        mat = Mat4::IDENTITY;
                    }

                    let p = mat.transform_point3(pos);
                    let n = mat.transform_vector3(norm).normalize();
                    v_vec4.extend_from_slice(&[p.x, p.y, p.z, 1.0]);
                    n_vec4.extend_from_slice(&[n.x, n.y, n.z, 0.0]);
                }
            } else {
                // 静的メッシュ (そのままコピー)
                for i in 0..use_positions.len() {
                    let p = use_positions[i];
                    let n = use_normals[i];
                    v_vec4.extend_from_slice(&[p.x, p.y, p.z, 1.0]);
                    n_vec4.extend_from_slice(&[n.x, n.y, n.z, 0.0]);
                }
            }

            // BLAS構築
            let mut builder = BVHBuilder::new(&v_vec4, &geom.indices);
            let (mut nodes, indices, tri_ids) = builder.build_with_ids();

            // インデックスのグローバルオフセット調整
            let v_offset = (self.vertices.len() / 4) as u32;
            let global_indices: Vec<u32> = indices.iter().map(|&idx| idx + v_offset).collect();

            // LeafノードのトライアングルIDオフセット調整
            for i in 0..(nodes.len() / 8) {
                if nodes[i * 8 + 7] > 0. {
                    // is_leaf
                    let lf = nodes[i * 8 + 3] as u32;
                    nodes[i * 8 + 3] = (lf + current_tri_offset) as f32;
                }
            }

            // 属性の並び替え
            let mut sorted_attrs = vec![0.0; geom.attributes.len()];
            let stride = 8;
            for (new_i, &old_id) in tri_ids.iter().enumerate() {
                let src = old_id * stride;
                let dst = new_i * stride;
                if src + stride <= geom.attributes.len() {
                    sorted_attrs[dst..dst + stride]
                        .copy_from_slice(&geom.attributes[src..src + stride]);
                }
            }

            // グローバルバッファへの追加
            self.vertices.extend(v_vec4);
            self.normals.extend(n_vec4);
            self.indices.extend(global_indices);
            self.attributes.extend(sorted_attrs);

            // BLASルートオフセットの記録
            self.blas_root_offsets.push(current_node_offset);

            let node_count = (nodes.len() / 8) as u32;
            current_node_offset += node_count;
            current_tri_offset += (indices.len() / 3) as u32;

            self.blas_nodes.extend(nodes);
        }

        // インスタンス情報の更新 (BLASオフセットとAABB)
        // raw_instances の instance_id には geometry_index が入っている前提
        for inst in &mut self.raw_instances {
            let geom_idx = inst.instance_id as usize;

            if geom_idx < self.blas_root_offsets.len() {
                inst.blas_node_offset = self.blas_root_offsets[geom_idx];

                // ルートノードからAABBを取得
                let base = inst.blas_node_offset as usize * 8;
                let aabb = if base < self.blas_nodes.len() {
                    crate::primitives::AABB {
                        min: Vec3::new(
                            self.blas_nodes[base],
                            self.blas_nodes[base + 1],
                            self.blas_nodes[base + 2],
                        ),
                        max: Vec3::new(
                            self.blas_nodes[base + 4],
                            self.blas_nodes[base + 5],
                            self.blas_nodes[base + 6],
                        ),
                    }
                } else {
                    crate::primitives::AABB::empty()
                };
                self.instance_blas_aabbs.push(aabb);
            } else {
                self.instance_blas_aabbs
                    .push(crate::primitives::AABB::empty());
            }
        }
    }

    // Pointers ... (省略、変更なし)
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
