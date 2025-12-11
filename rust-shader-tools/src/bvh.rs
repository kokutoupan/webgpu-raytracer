// use glam::{vec3, Vec3};

use crate::primitives::{AABB, PRIMITIVE_STRIDE, Primitive};

#[derive(Clone, Copy, Debug)]
pub struct BVHNode {
    pub aabb: AABB,
    pub left_first: u32,
    pub tri_count: u32,
}

impl Default for BVHNode {
    fn default() -> Self {
        Self {
            aabb: AABB::empty(),
            left_first: 0,
            tri_count: 0,
        }
    }
}

// SAH計算用のビン情報
#[derive(Clone, Copy, Debug)]
struct Bin {
    bounds: AABB,
    count: u32,
}

impl Default for Bin {
    fn default() -> Self {
        Self {
            bounds: AABB::empty(),
            count: 0,
        }
    }
}

pub struct BVHBuilder {
    pub nodes: Vec<BVHNode>,
    pub primitives: Vec<Primitive>, // ソート用に保持
}

impl BVHBuilder {
    pub fn new(primitives: Vec<Primitive>) -> Self {
        Self {
            nodes: Vec::new(),
            primitives,
        }
    }

    pub fn build(&mut self) -> (Vec<f32>, Vec<f32>) {
        self.nodes.clear();

        // ルートノード作成
        let root = BVHNode {
            left_first: 0,
            tri_count: self.primitives.len() as u32,
            ..Default::default()
        };
        self.nodes.push(root);

        // ルートのAABB計算
        self.update_node_bounds(0);

        // 再帰的に分割
        self.subdivide(0);

        // --- データパック ---

        // ノードデータ (Float32Array[8 * N])
        let packed_nodes = self.pack_nodes();

        // プリミティブデータ (Float32Array[16 * N])
        // 並び替えられた primitives を順番にパック
        let mut packed_prims = vec![0.0; self.primitives.len() * PRIMITIVE_STRIDE];
        for (i, prim) in self.primitives.iter().enumerate() {
            let data = prim.pack(); // [f32; 16]
            let offset = i * PRIMITIVE_STRIDE;
            packed_prims[offset..offset + PRIMITIVE_STRIDE].copy_from_slice(&data);
        }

        (packed_nodes, packed_prims)
    }

    // ノードのAABBを再計算する
    fn update_node_bounds(&mut self, node_idx: usize) {
        let (first, count) = {
            let node = &self.nodes[node_idx];
            (node.left_first as usize, node.tri_count as usize)
        };

        let mut node_aabb = AABB::empty();

        for i in 0..count {
            let prim_idx = first + i;
            let prim_aabb = self.primitives[prim_idx].aabb();
            // ★修正: union は新しいインスタンスを返すので再代入する
            node_aabb = node_aabb.union(&prim_aabb);
        }

        self.nodes[node_idx].aabb = node_aabb;
    }

    // 再帰的分割 (Binning SAH)
    fn subdivide(&mut self, node_idx: usize) {
        let node = self.nodes[node_idx];

        // 要素数が少なければ葉ノードとして確定
        if node.tri_count <= 4 {
            return;
        }

        // AABBのサイズ計算
        let extent = node.aabb.max - node.aabb.min;

        // 最長軸を選択 (0:x, 1:y, 2:z)
        let mut axis = 0;
        if extent.y > extent.x {
            axis = 1;
        }
        if extent.z > extent.x && extent.z > extent.y {
            axis = 2;
        }

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

        // 軸の長さがほぼゼロなら分割不能
        if split_len < 1e-6 {
            return;
        }

        // --- 1. Binning ---
        const BINS: usize = 16;
        let mut bins = [Bin::default(); BINS];

        let first = node.left_first as usize;
        let count = node.tri_count as usize;

        // 各プリミティブをビンに振り分け
        for i in 0..count {
            let prim = &self.primitives[first + i];
            let center = prim.aabb().center();
            let pos = match axis {
                0 => center.x,
                1 => center.y,
                _ => center.z,
            };

            let mut bin_idx = (((pos - split_min) / split_len) * (BINS as f32)) as usize;
            if bin_idx >= BINS {
                bin_idx = BINS - 1;
            }

            bins[bin_idx].count += 1;
            // ★修正: 再代入
            bins[bin_idx].bounds = bins[bin_idx].bounds.union(&prim.aabb());
        }

        // --- 2. SAH コスト計算 ---
        let mut left_area = [0.0; BINS];
        let mut left_count = [0; BINS];
        let mut right_area = [0.0; BINS];
        let mut right_count = [0; BINS];

        // Left Sweep
        let mut current_box = AABB::empty();
        let mut current_count = 0;
        for i in 0..BINS {
            if bins[i].count > 0 {
                // ★修正: 再代入
                current_box = current_box.union(&bins[i].bounds);
                current_count += bins[i].count;
            }
            left_area[i] = current_box.area();
            left_count[i] = current_count;
        }

        // Right Sweep
        current_box = AABB::empty();
        current_count = 0;
        for i in (0..BINS).rev() {
            if bins[i].count > 0 {
                // ★修正: 再代入
                current_box = current_box.union(&bins[i].bounds);
                current_count += bins[i].count;
            }
            right_area[i] = current_box.area();
            right_count[i] = current_count;
        }

        // ベストな分割位置を探す
        let mut best_cost = f32::INFINITY;
        let mut best_split = usize::MAX;

        for i in 0..(BINS - 1) {
            // 片方が空になる分割は無意味
            if left_count[i] == 0 || right_count[i + 1] == 0 {
                continue;
            }

            // SAH Cost
            let cost = left_area[i] * (left_count[i] as f32)
                + right_area[i + 1] * (right_count[i + 1] as f32);

            if cost < best_cost {
                best_cost = cost;
                best_split = i;
            }
        }

        // 良い分割が見つからない場合は葉ノードのまま終了
        if best_split == usize::MAX {
            return;
        }

        // --- 3. パーティショニング (並び替え) ---
        let mut i = first;
        let mut j = first + count - 1;

        while i <= j {
            let get_bin_idx = |prim: &Primitive| -> usize {
                let center = prim.aabb().center();
                let pos = match axis {
                    0 => center.x,
                    1 => center.y,
                    _ => center.z,
                };
                let mut idx = (((pos - split_min) / split_len) * (BINS as f32)) as usize;
                if idx >= BINS {
                    idx = BINS - 1;
                }
                idx
            };

            let bin_i = get_bin_idx(&self.primitives[i]);

            if bin_i <= best_split {
                i += 1;
            } else {
                let bin_j = get_bin_idx(&self.primitives[j]);
                if bin_j > best_split {
                    if j == 0 {
                        break;
                    }
                    j -= 1;
                } else {
                    self.primitives.swap(i, j);
                    i += 1;
                    if j == 0 {
                        break;
                    }
                    j -= 1;
                }
            }
        }

        let left_count_real = i - first;

        if left_count_real == 0 || left_count_real == count {
            return;
        }

        // 子ノードの作成
        let left_child_idx = self.nodes.len();

        // 左の子
        self.nodes.push(BVHNode {
            left_first: first as u32,
            tri_count: left_count_real as u32,
            ..Default::default()
        });
        // 右の子
        self.nodes.push(BVHNode {
            left_first: i as u32,
            tri_count: (count - left_count_real) as u32,
            ..Default::default()
        });

        // 現在のノードを内部ノード化
        self.nodes[node_idx].left_first = left_child_idx as u32;
        self.nodes[node_idx].tri_count = 0;

        // 再帰処理
        self.update_node_bounds(left_child_idx);
        self.update_node_bounds(left_child_idx + 1);

        self.subdivide(left_child_idx);
        self.subdivide(left_child_idx + 1);
    }

    // ノードデータをGPU用のフラットな配列にパック
    fn pack_nodes(&self) -> Vec<f32> {
        let mut data = vec![0.0; self.nodes.len() * 8];
        for (i, node) in self.nodes.iter().enumerate() {
            let offset = i * 8;
            data[offset + 0] = node.aabb.min.x;
            data[offset + 1] = node.aabb.min.y;
            data[offset + 2] = node.aabb.min.z;
            data[offset + 3] = node.left_first as f32;

            data[offset + 4] = node.aabb.max.x;
            data[offset + 5] = node.aabb.max.y;
            data[offset + 6] = node.aabb.max.z;
            data[offset + 7] = node.tri_count as f32;
        }
        data
    }
}
