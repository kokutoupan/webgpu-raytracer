use crate::primitives::Primitive;

// サブモジュールの公開
pub mod camera;
pub mod helpers;
pub mod scenes;

// 外部から使うときは scene::CameraConfig でアクセスできるように再エクスポート
pub use camera::CameraConfig;
// get_scene_data もここで再エクスポートしておくと便利
pub use scenes::get_scene_data;

// マテリアルタイプ定数
pub mod mat_type {
    pub const LAMBERTIAN: u32 = 0;
    pub const METAL: u32 = 1;
    pub const DIELECTRIC: u32 = 2;
    pub const LIGHT: u32 = 3;
}

// シーンデータ (戻り値用)
pub struct SceneData {
    pub camera: CameraConfig,
    pub primitives: Vec<Primitive>,
}
