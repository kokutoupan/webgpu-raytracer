// src/bvh/mod.rs
pub mod blas;
pub mod tlas;

use glam::Mat4;

// --- Common Structures ---

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct StacklessBVHNode {
    pub min_b: [f32; 3],
    pub skip_pointer: u32,
    pub max_b: [f32; 3],
    pub data: u32,
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Instance {
    pub transform: Mat4,
    pub inverse_transform: Mat4,
    pub blas_node_offset: u32,
    pub attr_offset: u32,
    pub instance_id: u32,
    pub pad: u32,
}

impl Default for Instance {
    fn default() -> Self {
        Self {
            transform: Mat4::IDENTITY,
            inverse_transform: Mat4::IDENTITY,
            blas_node_offset: 0,
            attr_offset: 0,
            instance_id: 0,
            pad: 0,
        }
    }
}
