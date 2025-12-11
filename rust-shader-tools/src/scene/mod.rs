// src/scene/mod.rs
use crate::geometry::Geometry;
use glam::Mat4;

// サブモジュールの公開
pub mod camera;
pub mod helpers;
pub mod scenes;

// 外部からアクセスしやすいように再エクスポート
pub use camera::CameraConfig;
pub use scenes::get_scene_data;

// マテリアルタイプ定数
pub mod mat_type {
    pub const LAMBERTIAN: u32 = 0;
    pub const METAL: u32 = 1;
    pub const DIELECTRIC: u32 = 2;
    pub const LIGHT: u32 = 3;
}

// シーンインスタンス構造体
pub struct SceneInstance {
    pub transform: Mat4,
    pub geometry_index: usize, // ★追加: どのGeometry(BLAS)を使うか
}

// シーンデータ構造体
pub struct SceneData {
    pub camera: CameraConfig,
    pub geometries: Vec<Geometry>, // ★変更: 複数のGeometryを持てるようにする
    pub instances: Vec<SceneInstance>,
}
