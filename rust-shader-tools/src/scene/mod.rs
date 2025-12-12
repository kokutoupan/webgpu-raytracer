// src/scene/mod.rs
pub mod animation;
pub mod camera;
pub mod helpers;
pub mod material;
pub mod node;
pub mod procedural;
pub mod factory; // scenes -> factory に変更

use crate::geometry::Geometry;
pub use camera::CameraConfig;
pub use node::{Node, SceneInstance, Skin};

// lib.rs から scene::get_scene_data() で呼べるように再エクスポート
pub use factory::get_scene_data;

// マテリアル定数へのエイリアス
pub mod mat_type {
    pub use super::material::*;
}

// シーン全体を表すデータコンテナ
pub struct SceneData {
    pub camera: CameraConfig,
    pub geometries: Vec<Geometry>,
    pub instances: Vec<SceneInstance>,
    
    // Animation Support
    pub nodes: Vec<Node>,
    pub skins: Vec<Skin>,
    pub animations: Vec<animation::Animation>,
}
