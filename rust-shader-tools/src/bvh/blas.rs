// src/bvh/blas.rs
use super::StacklessBVHNode;
use crate::primitives::AABB;
use glam::{vec3, Vec3};

#[derive(Clone, Copy, Debug, Default)]
struct Bin {
    bounds: AABB,
    count: u32,
}

pub struct BVHBuilder<'a> {
    pub nodes: Vec<StacklessBVHNode>,
    indices: &'a [u32],
    tri_indices: Vec<usize>,
    tri_aabbs: Vec<AABB>,
    tri_centers: Vec<Vec3>,
}

impl<'a> BVHBuilder<'a> {
    pub fn new(vertices: &'a [f32], indices: &'a [u32]) -> Self {
        let tri_count = indices.len() / 3;
        let mut tri_aabbs = Vec::with_capacity(tri_count);
        let mut tri_centers = Vec::with_capacity(tri_count);

        for i in 0..tri_count {
            let i0 = indices[i * 3] as usize;
            let i1 = indices[i * 3 + 1] as usize;
            let i2 = indices[i * 3 + 2] as usize;

            let get_v = |idx: usize| {
                let b = idx * 4;
                vec3(vertices[b], vertices[b + 1], vertices[b + 2])
            };
            let v0 = get_v(i0);
            let v1 = get_v(i1);
            let v2 = get_v(i2);

            let min = v0.min(v1).min(v2);
            let max = v0.max(v1).max(v2);
            
            let size = max - min;
            let eps = 1e-5;
            let pad = vec3(
                if size.x < eps { eps } else { 0. },
                if size.y < eps { eps } else { 0. },
                if size.z < eps { eps } else { 0. },
            );

            let aabb = AABB {
                min: min - pad * 0.5,
                max: max + pad * 0.5,
            };
            tri_aabbs.push(aabb);
            tri_centers.push(aabb.center());
        }

        Self {
            nodes: Vec::new(),
            indices,
            tri_indices: Vec::new(),
            tri_aabbs,
            tri_centers,
        }
    }

    pub fn build_with_ids(&mut self) -> (Vec<f32>, Vec<u32>, Vec<usize>) {
        self.nodes.clear();
        self.tri_indices = (0..self.indices.len() / 3).collect();

        if !self.tri_indices.is_empty() {
            self.subdivide(0, self.tri_indices.len());
        }

        let packed_nodes = self.pack_nodes();
        let mut sorted_indices = Vec::with_capacity(self.indices.len());
        
        for &tri_idx in &self.tri_indices {
            let base = tri_idx * 3;
            sorted_indices.push(self.indices[base]);
            sorted_indices.push(self.indices[base + 1]);
            sorted_indices.push(self.indices[base + 2]);
        }

        (packed_nodes, sorted_indices, self.tri_indices.clone())
    }

    fn subdivide(&mut self, first: usize, count: usize) {
        let node_idx = self.nodes.len();
        self.nodes.push(StacklessBVHNode::default());

        let mut aabb = AABB::empty();
        for i in 0..count {
            aabb = aabb.union(&self.tri_aabbs[self.tri_indices[first + i]]);
        }
        self.nodes[node_idx].min_b = aabb.min.to_array();
        self.nodes[node_idx].max_b = aabb.max.to_array();

        if count <= 4 {
            self.nodes[node_idx].data = ((first as u32) << 3) | count as u32;
            self.nodes[node_idx].skip_pointer = self.nodes.len() as u32;
            return;
        }

        let extent = aabb.max - aabb.min;
        let axis = if extent.y > extent.x { 1 } else if extent.z > extent.x && extent.z > extent.y { 2 } else { 0 };

        let split_len = extent[axis];
        let split_min = aabb.min[axis];

        if split_len < 1e-6 {
            self.nodes[node_idx].data = ((first as u32) << 3) | count as u32;
            self.nodes[node_idx].skip_pointer = self.nodes.len() as u32;
            return;
        }

        const BINS: usize = 16;
        let mut bins = [Bin::default(); BINS];
        let scale = BINS as f32 / split_len;
        
        let get_bin_idx = |val: f32| -> usize {
            let idx = ((val - split_min) * scale) as usize;
            idx.min(BINS - 1)
        };

        for i in 0..count {
            let tri_id = self.tri_indices[first + i];
            let pos = self.tri_centers[tri_id][axis];
            bins[get_bin_idx(pos)].count += 1;
            bins[get_bin_idx(pos)].bounds = bins[get_bin_idx(pos)].bounds.union(&self.tri_aabbs[tri_id]);
        }

        let mut left_area = [0.; BINS];
        let mut left_count = [0; BINS];
        let mut right_area = [0.; BINS];
        let mut right_count = [0; BINS];
        
        let mut curr_box = AABB::empty();
        let mut curr_sum = 0;
        for i in 0..BINS {
            curr_sum += bins[i].count;
            curr_box = curr_box.union(&bins[i].bounds);
            left_area[i] = curr_box.area();
            left_count[i] = curr_sum;
        }

        curr_box = AABB::empty();
        curr_sum = 0;
        for i in (0..BINS).rev() {
            curr_sum += bins[i].count;
            curr_box = curr_box.union(&bins[i].bounds);
            right_area[i] = curr_box.area();
            right_count[i] = curr_sum;
        }

        let mut best_cost = f32::INFINITY;
        let mut best_split = usize::MAX;
        for i in 0..(BINS - 1) {
            if left_count[i] == 0 || right_count[i + 1] == 0 { continue; }
            let cost = left_area[i] * (left_count[i] as f32) + right_area[i + 1] * (right_count[i + 1] as f32);
            if cost < best_cost {
                best_cost = cost;
                best_split = i;
            }
        }

        if best_split == usize::MAX {
            self.nodes[node_idx].data = ((first as u32) << 3) | count as u32;
            self.nodes[node_idx].skip_pointer = self.nodes.len() as u32;
            return;
        }

        let mut i = first;
        let mut j = first + count - 1;
        while i <= j {
            let pos = self.tri_centers[self.tri_indices[i]][axis];
            if get_bin_idx(pos) <= best_split {
                i += 1;
            } else {
                let pos_j = self.tri_centers[self.tri_indices[j]][axis];
                if get_bin_idx(pos_j) > best_split {
                    if j == 0 { break; }
                    j -= 1;
                } else {
                    self.tri_indices.swap(i, j);
                    i += 1;
                    if j == 0 { break; }
                    j -= 1;
                }
            }
        }

        let mut l_count = i - first;
        let mut r_count = count - l_count;

        if l_count == 0 || l_count == count {
            self.nodes[node_idx].data = ((first as u32) << 3) | count as u32;
            self.nodes[node_idx].skip_pointer = self.nodes.len() as u32;
            return;
        }

        // --- SORT CHILDREN FOR STACKLESS BVH (Static Front-to-Back heuristic) ---
        let l_cost = left_area[best_split] * l_count as f32;
        let r_cost = right_area[best_split + 1] * r_count as f32;
        if r_cost > l_cost {
            self.tri_indices[first..first + count].rotate_left(l_count);
            let temp = l_count;
            l_count = r_count;
            r_count = temp;
        }

        self.nodes[node_idx].data = 0;

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
