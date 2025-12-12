// src/scene/mod.rs
use crate::geometry::Geometry;
use glam::Mat4;

// ★追加: animationモジュール
pub mod animation;
pub mod camera;
pub mod helpers;
pub mod scenes;

pub use camera::CameraConfig;
pub use scenes::get_scene_data;

pub mod mat_type {
    pub const LAMBERTIAN: u32 = 0;
    pub const METAL: u32 = 1;
    pub const DIELECTRIC: u32 = 2;
    pub const LIGHT: u32 = 3;
}

// --- Animation / Skeleton Support ---

#[derive(Clone, Debug)]
pub struct Node {
    pub name: String,
    pub parent_index: Option<usize>,
    pub children_indices: Vec<usize>,

    // Transform (Local)
    pub translation: glam::Vec3,
    pub rotation: glam::Quat,
    pub scale: glam::Vec3,

    // Computed World Transform (Updateごとに計算)
    pub global_transform: Mat4,
}

impl Default for Node {
    fn default() -> Self {
        Self {
            name: "Node".to_string(),
            parent_index: None,
            children_indices: Vec::new(),
            translation: glam::Vec3::ZERO,
            rotation: glam::Quat::IDENTITY,
            scale: glam::Vec3::ONE,
            global_transform: Mat4::IDENTITY,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Skin {
    pub joints: Vec<usize>, // Node indices
    pub inverse_bind_matrices: Vec<Mat4>,
}

// シーンインスタンス
pub struct SceneInstance {
    pub transform: Mat4,
    pub geometry_index: usize,
}

// シーンデータ構造体
pub struct SceneData {
    pub camera: CameraConfig,
    pub geometries: Vec<Geometry>,
    pub instances: Vec<SceneInstance>,

    // ★追加フィールド
    pub nodes: Vec<Node>,
    pub skins: Vec<Skin>,
    pub animations: Vec<animation::Animation>,
}
