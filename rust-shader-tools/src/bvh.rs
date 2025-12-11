// src/bvh.rs
use crate::primitives::AABB;
use glam::{Vec3, vec3};

#[derive(Clone, Copy, Debug, Default)]
pub struct BVHNode {
    pub aabb: AABB,
    pub left_first: u32,
    pub tri_count: u32,
}

#[derive(Clone, Copy, Debug, Default)]
struct Bin {
    bounds: AABB,
    count: u32,
}

impl Bin {
    fn new() -> Self {
        Self {
            bounds: AABB::empty(),
            count: 0,
        }
    }
}

pub struct BVHBuilder<'a> {
    pub nodes: Vec<BVHNode>,
    vertices: &'a [f32],
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
        let mut tri_indices = Vec::with_capacity(tri_count);

        for i in 0..tri_count {
            tri_indices.push(i);
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
            vertices,
            indices,
            tri_indices,
            tri_aabbs,
            tri_centers,
        }
    }

    pub fn build(&mut self) -> (Vec<f32>, Vec<u32>) {
        self.nodes.clear();
        let root = BVHNode {
            left_first: 0,
            tri_count: self.tri_indices.len() as u32,
            ..Default::default()
        };
        self.nodes.push(root);
        self.update_node_bounds(0);
        self.subdivide(0);

        let packed_nodes = self.pack_nodes();
        let mut sorted_indices = Vec::with_capacity(self.indices.len());
        for &tri_idx in &self.tri_indices {
            let base = tri_idx * 3;
            sorted_indices.push(self.indices[base]);
            sorted_indices.push(self.indices[base + 1]);
            sorted_indices.push(self.indices[base + 2]);
        }
        (packed_nodes, sorted_indices)
    }
    pub fn build_with_ids(&mut self) -> (Vec<f32>, Vec<u32>, Vec<usize>) {
        self.nodes.clear();
        let root = BVHNode {
            left_first: 0,
            tri_count: self.tri_indices.len() as u32,
            ..Default::default()
        };
        self.nodes.push(root);
        self.update_node_bounds(0);
        self.subdivide(0);

        let packed_nodes = self.pack_nodes();
        let mut sorted_indices = Vec::with_capacity(self.indices.len());
        for &tri_idx in &self.tri_indices {
            let base = tri_idx * 3;
            sorted_indices.push(self.indices[base]);
            sorted_indices.push(self.indices[base + 1]);
            sorted_indices.push(self.indices[base + 2]);
        }

        // 最後にこれも返す
        (packed_nodes, sorted_indices, self.tri_indices.clone())
    }

    fn update_node_bounds(&mut self, node_idx: usize) {
        let node = self.nodes[node_idx];
        let mut aabb = AABB::empty();
        for i in 0..node.tri_count {
            let tri_id = self.tri_indices[(node.left_first + i) as usize];
            aabb = aabb.union(&self.tri_aabbs[tri_id]);
        }
        self.nodes[node_idx].aabb = aabb;
    }

    fn subdivide(&mut self, node_idx: usize) {
        let node = self.nodes[node_idx];
        if node.tri_count <= 4 {
            return;
        }

        let extent = node.aabb.max - node.aabb.min;
        let axis = if extent.y > extent.x {
            1
        } else if extent.z > extent.x && extent.z > extent.y {
            2
        } else {
            0
        };
        let split_len = match axis {
            0 => extent.x,
            1 => extent.y,
            _ => extent.z,
        };
        let split_min = match axis {
            0 => node.aabb.min.x,
            1 => node.aabb.min.y,
            _ => node.aabb.min.z,
        };

        if split_len < 1e-6 {
            return;
        }

        const BINS: usize = 16;
        let mut bins = [Bin::new(); BINS];
        let first = node.left_first as usize;
        let count = node.tri_count as usize;

        for i in 0..count {
            let tri_id = self.tri_indices[first + i];
            let pos = match axis {
                0 => self.tri_centers[tri_id].x,
                1 => self.tri_centers[tri_id].y,
                _ => self.tri_centers[tri_id].z,
            };
            let mut bin_idx = (((pos - split_min) / split_len) * (BINS as f32)) as usize;
            if bin_idx >= BINS {
                bin_idx = BINS - 1;
            }
            bins[bin_idx].count += 1;
            bins[bin_idx].bounds = bins[bin_idx].bounds.union(&self.tri_aabbs[tri_id]);
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
            if left_count[i] == 0 || right_count[i + 1] == 0 {
                continue;
            }
            let cost = left_area[i] * (left_count[i] as f32)
                + right_area[i + 1] * (right_count[i + 1] as f32);
            if cost < best_cost {
                best_cost = cost;
                best_split = i;
            }
        }

        if best_split == usize::MAX {
            return;
        }

        let mut i = first;
        let mut j = first + count - 1;
        while i <= j {
            let get_bin = |tri_id: usize| {
                let pos = match axis {
                    0 => self.tri_centers[tri_id].x,
                    1 => self.tri_centers[tri_id].y,
                    _ => self.tri_centers[tri_id].z,
                };
                let mut idx = (((pos - split_min) / split_len) * (BINS as f32)) as usize;
                if idx >= BINS {
                    idx = BINS - 1;
                }
                idx
            };

            let bin_i = get_bin(self.tri_indices[i]);

            if bin_i <= best_split {
                i += 1;
            } else if get_bin(self.tri_indices[j]) > best_split {
                if j == 0 {
                    break;
                }
                j -= 1;
            } else {
                self.tri_indices.swap(i, j);
                i += 1;
                if j == 0 {
                    break;
                }
                j -= 1;
            }
        }

        let left_count = i - first;
        if left_count == 0 || left_count == count {
            return;
        }

        let left_child_idx = self.nodes.len();
        self.nodes.push(BVHNode {
            left_first: first as u32,
            tri_count: left_count as u32,
            ..Default::default()
        });
        self.nodes.push(BVHNode {
            left_first: i as u32,
            tri_count: (count - left_count) as u32,
            ..Default::default()
        });

        self.nodes[node_idx].left_first = left_child_idx as u32;
        self.nodes[node_idx].tri_count = 0;

        self.update_node_bounds(left_child_idx);
        self.update_node_bounds(left_child_idx + 1);
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
