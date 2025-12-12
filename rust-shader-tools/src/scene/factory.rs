// src/scene/factory.rs
use super::{procedural, SceneData};
use crate::mesh::Mesh;

pub fn get_scene_data(scene_name: &str, mesh: Option<&Mesh>, has_glb: bool) -> SceneData {
    match scene_name {
        "spheres" => procedural::create_random_spheres(),
        "mixed" => procedural::create_mixed_scene(),
        "special" => procedural::create_cornell_box_special(),
        "mesh" => procedural::create_mesh_scene(),
        "viewer" => procedural::create_model_viewer_scene(mesh, has_glb),
        "cornell" | _ => procedural::create_cornell_box(None),
    }
}
