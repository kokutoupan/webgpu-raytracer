// src/scene/node.rs
use glam::{Mat4, Quat, Vec3};

// --- Scene Graph Nodes ---

#[derive(Clone, Debug)]
pub struct Node {
    pub name: String,
    pub parent_index: Option<usize>,
    pub children_indices: Vec<usize>,

    // Local Transform (Animation target)
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,

    // Computed Global Transform
    pub global_transform: Mat4,
}

impl Default for Node {
    fn default() -> Self {
        Self {
            name: "Node".to_string(),
            parent_index: None,
            children_indices: Vec::new(),
            translation: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
            global_transform: Mat4::IDENTITY,
        }
    }
}

// --- Skinning ---

#[derive(Clone, Debug)]
pub struct Skin {
    pub joints: Vec<usize>, // Node indices
    pub inverse_bind_matrices: Vec<Mat4>,
}

// --- High Level Instance ---

// シーン内に配置される「実体」
pub struct SceneInstance {
    pub transform: Mat4,
    pub geometry_index: usize,
}
