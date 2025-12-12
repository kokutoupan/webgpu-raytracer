// src/bvh/tlas.rs
use super::{BVHNode, Instance};
use crate::primitives::AABB;
use glam::Vec3;
use std::cmp::Ordering;

pub type TLASNode = BVHNode;

pub struct TLASBuilder<'a> {
    pub nodes: Vec<TLASNode>,
    pub instances: &'a [Instance],
    pub instance_indices: Vec<usize>,
    pub instance_aabbs: Vec<AABB>,
    pub instance_centers: Vec<Vec3>,
}

impl<'a> TLASBuilder<'a> {
    pub fn new(instances: &'a [Instance], blas_aabbs: &[AABB]) -> Self {
        let count = instances.len();
        let mut instance_aabbs = Vec::with_capacity(count);
        let mut instance_centers = Vec::with_capacity(count);
        let instance_indices = (0..count).collect();

        for i in 0..count {
            let local_aabb = blas_aabbs[i];
            let world_aabb = local_aabb.transform(instances[i].transform);
            instance_aabbs.push(world_aabb);
            instance_centers.push(world_aabb.center());
        }

        Self {
            nodes: Vec::new(),
            instances,
            instance_indices,
            instance_aabbs,
            instance_centers,
        }
    }

    pub fn build(&mut self) -> (Vec<f32>, Vec<Instance>) {
        self.nodes.clear();
        if self.instances.is_empty() {
            return (vec![], vec![]);
        }

        let root = TLASNode {
            left_first: 0,
            tri_count: self.instances.len() as u32,
            ..Default::default()
        };
        self.nodes.push(root);

        let mut root_aabb = AABB::empty();
        for aabb in &self.instance_aabbs {
            root_aabb = root_aabb.union(aabb);
        }
        self.nodes[0].aabb = root_aabb;

        self.subdivide(0);

        let packed = self.pack_nodes();

        let mut sorted_instances = Vec::with_capacity(self.instances.len());
        for &idx in &self.instance_indices {
            sorted_instances.push(self.instances[idx]);
        }

        (packed, sorted_instances)
    }

    fn subdivide(&mut self, node_idx: usize) {
        let node = self.nodes[node_idx];
        let count = node.tri_count as usize;
        let first = node.left_first as usize;

        if count == 1 {
            self.nodes[node_idx].tri_count = 1;
            self.nodes[node_idx].left_first = first as u32;
            return;
        }

        let extent = node.aabb.max - node.aabb.min;
        let axis = if extent.y > extent.x { 1 } else if extent.z > extent.x && extent.z > extent.y { 2 } else { 0 };

        let slice = &mut self.instance_indices[first..first + count];
        slice.sort_by(|&a, &b| {
            let ca = self.instance_centers[a][axis];
            let cb = self.instance_centers[b][axis];
            ca.partial_cmp(&cb).unwrap_or(Ordering::Equal)
        });

        let mid = count / 2;
        let left_count = mid;
        let right_count = count - mid;

        let left_child_idx = self.nodes.len();
        self.nodes.push(Default::default());
        self.nodes.push(Default::default());

        self.nodes[node_idx].tri_count = 0;
        self.nodes[node_idx].left_first = left_child_idx as u32;

        let mut left_aabb = AABB::empty();
        for i in 0..left_count {
            left_aabb = left_aabb.union(&self.instance_aabbs[self.instance_indices[first + i]]);
        }
        self.nodes[left_child_idx].aabb = left_aabb;
        self.nodes[left_child_idx].left_first = first as u32;
        self.nodes[left_child_idx].tri_count = left_count as u32;

        let mut right_aabb = AABB::empty();
        for i in 0..right_count {
            right_aabb = right_aabb.union(&self.instance_aabbs[self.instance_indices[first + mid + i]]);
        }
        self.nodes[left_child_idx + 1].aabb = right_aabb;
        self.nodes[left_child_idx + 1].left_first = (first + mid) as u32;
        self.nodes[left_child_idx + 1].tri_count = right_count as u32;

        self.subdivide(left_child_idx);
        self.subdivide(left_child_idx + 1);
    }

    fn pack_nodes(&self) -> Vec<f32> {
        let mut data = vec![0.; self.nodes.len() * 8];
        for (i, node) in self.nodes.iter().enumerate() {
            let off = i * 8;
            data[off + 0] = node.aabb.min.x;
            data[off + 1] = node.aabb.min.y;
            data[off + 2] = node.aabb.min.z;
            data[off + 3] = node.left_first as f32;
            data[off + 4] = node.aabb.max.x;
            data[off + 5] = node.aabb.max.y;
            data[off + 6] = node.aabb.max.z;
            data[off + 7] = node.tri_count as f32;
        }
        data
    }
}
