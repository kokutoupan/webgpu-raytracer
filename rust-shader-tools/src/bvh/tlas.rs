// src/bvh/tlas.rs
use super::{Instance, StacklessBVHNode};
use crate::primitives::AABB;
use glam::Vec3;
use std::cmp::Ordering;

pub type TLASNode = StacklessBVHNode;

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

        self.subdivide(0, self.instances.len());

        let packed = self.pack_nodes();

        let mut sorted_instances = Vec::with_capacity(self.instances.len());
        for &idx in &self.instance_indices {
            sorted_instances.push(self.instances[idx]);
        }

        (packed, sorted_instances)
    }

    fn subdivide(&mut self, first: usize, count: usize) {
        let node_idx = self.nodes.len();
        self.nodes.push(TLASNode::default());

        let mut aabb = AABB::empty();
        for i in 0..count {
            aabb = aabb.union(&self.instance_aabbs[self.instance_indices[first + i]]);
        }
        self.nodes[node_idx].min_b = aabb.min.to_array();
        self.nodes[node_idx].max_b = aabb.max.to_array();

        if count == 1 {
            self.nodes[node_idx].data = ((first as u32) << 3) | 1;
            self.nodes[node_idx].skip_pointer = self.nodes.len() as u32;
            return;
        }

        let extent = aabb.max - aabb.min;
        let axis = if extent.y > extent.x { 1 } else if extent.z > extent.x && extent.z > extent.y { 2 } else { 0 };

        let slice = &mut self.instance_indices[first..first + count];
        slice.sort_by(|&a, &b| {
            let ca = self.instance_centers[a][axis];
            let cb = self.instance_centers[b][axis];
            ca.partial_cmp(&cb).unwrap_or(Ordering::Equal)
        });

        let mid = count / 2;
        let mut l_count = mid;
        let mut r_count = count - mid;

        let mut l_aabb = AABB::empty();
        for i in 0..l_count {
            l_aabb = l_aabb.union(&self.instance_aabbs[self.instance_indices[first + i]]);
        }
        let mut r_aabb = AABB::empty();
        for i in 0..r_count {
            r_aabb = r_aabb.union(&self.instance_aabbs[self.instance_indices[first + mid + i]]);
        }

        if r_aabb.area() * r_count as f32 > l_aabb.area() * l_count as f32 {
            self.instance_indices[first..first + count].rotate_left(l_count);
            let temp = l_count;
            l_count = r_count;
            r_count = temp;
        }

        self.nodes[node_idx].data = 0; // internal

        self.subdivide(first, l_count);
        self.subdivide(first + l_count, r_count);

        self.nodes[node_idx].skip_pointer = self.nodes.len() as u32;
    }

    fn pack_nodes(&self) -> Vec<f32> {
        let mut data = vec![0.; self.nodes.len() * 8];
        for (i, node) in self.nodes.iter().enumerate() {
            let off = i * 8;
            data[off + 0] = node.min_b[0];
            data[off + 1] = node.min_b[1];
            data[off + 2] = node.min_b[2];
            data[off + 3] = f32::from_bits(node.skip_pointer);
            data[off + 4] = node.max_b[0];
            data[off + 5] = node.max_b[1];
            data[off + 6] = node.max_b[2];
            data[off + 7] = f32::from_bits(node.data);
        }
        data
    }
}
