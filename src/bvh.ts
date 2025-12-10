// bvh.ts
import { type Vec3 } from './math';
import { type IHittable, type AABB, PRIMITIVE_STRIDE } from './primitives';

export interface BVHNode {
  min: Vec3; max: Vec3;
  leftFirst: number; // 参照リストの開始インデックス
  triCount: number;  // 要素数
}

// --- ヘルパー関数: AABB操作 ---

// 空のAABBを作成 (無限小)
function emptyAABB(): AABB {
  return {
    min: { x: Infinity, y: Infinity, z: Infinity },
    max: { x: -Infinity, y: -Infinity, z: -Infinity },
    center: { x: 0, y: 0, z: 0 }
  };
}

// AABBを拡張して他のAABBを含める
function growAABB(target: AABB, add: AABB) {
  target.min.x = Math.min(target.min.x, add.min.x);
  target.min.y = Math.min(target.min.y, add.min.y);
  target.min.z = Math.min(target.min.z, add.min.z);
  target.max.x = Math.max(target.max.x, add.max.x);
  target.max.y = Math.max(target.max.y, add.max.y);
  target.max.z = Math.max(target.max.z, add.max.z);
}

// AABBの表面積を計算 (Cost計算用)
function aabbArea(aabb: AABB): number {
  const w = aabb.max.x - aabb.min.x;
  const h = aabb.max.y - aabb.min.y;
  const d = aabb.max.z - aabb.min.z;
  // 無効なボックス(初期値)の場合は0を返す
  if (w < 0 || h < 0 || d < 0) return 0;
  return 2 * (w * h + h * d + d * w);
}

// SAH用のビン情報
interface Bin {
  bounds: AABB;
  count: number;
}

export class BVHBuilder {
  nodes: BVHNode[] = [];
  sortedPrims: IHittable[] = [];

  // 入力: 球の配列と三角形の配列（生データ）を受け取る
  build(primitives: IHittable[]): { bvhNodes: Float32Array<ArrayBuffer>, unifiedPrimitives: Float32Array<ArrayBuffer> } {
    this.nodes = [];
    this.sortedPrims = [...primitives];


    // --- 2. BVH構築 ---
    const root = { min: { x: 0, y: 0, z: 0 }, max: { x: 0, y: 0, z: 0 }, leftFirst: 0, triCount: this.sortedPrims.length };
    this.nodes.push(root);

    // 全体のAABB
    this.updateNodeBounds(0);

    // 再帰的に分割開始
    // ※ console.time で計測するとSAHの効果（ビルド時間増 vs 描画負荷減）が見えて面白いです
    console.time("BVH Build");
    this.subdivide(0);
    console.timeEnd("BVH Build");

    // --- 3. データパック ---

    // ノードデータ (Float32)
    const bvhNodeData = this.packNodes();

    // プリミティブの一括パック
    const unifiedData = new Float32Array(this.sortedPrims.length * PRIMITIVE_STRIDE);
    for (let i = 0; i < this.sortedPrims.length; i++) {
      // 各オブジェクトの pack メソッドを呼ぶだけ！
      this.sortedPrims[i].pack(unifiedData, i * PRIMITIVE_STRIDE);
    }

    return { bvhNodes: bvhNodeData, unifiedPrimitives: unifiedData };
  }


  updateNodeBounds(nodeIdx: number) {
    const node = this.nodes[nodeIdx];
    let first = true;
    for (let i = 0; i < node.triCount; i++) {
      const aabb = this.sortedPrims[node.leftFirst + i].getAABB();
      if (first) {
        node.min = { ...aabb.min };
        node.max = { ...aabb.max };
        first = false;
      } else {
        node.min.x = Math.min(node.min.x, aabb.min.x);
        node.min.y = Math.min(node.min.y, aabb.min.y);
        node.min.z = Math.min(node.min.z, aabb.min.z);
        node.max.x = Math.max(node.max.x, aabb.max.x);
        node.max.y = Math.max(node.max.y, aabb.max.y);
        node.max.z = Math.max(node.max.z, aabb.max.z);
      }
    }
  }

  // ★改善ポイント: Binning SAH による分割
  subdivide(nodeIdx: number) {
    const node = this.nodes[nodeIdx];

    // 要素数が少なければ分割終了（葉ノード確定）
    if (node.triCount <= 4) return;

    // 現在のノードの大きさ
    const extent = {
      x: node.max.x - node.min.x,
      y: node.max.y - node.min.y,
      z: node.max.z - node.min.z,
    };

    // 最長軸を選択
    let axis = 0;
    if (extent.y > extent.x) axis = 1;
    if (extent.z > extent.x && extent.z > extent.y) axis = 2;

    const minVal = axis === 0 ? node.min.x : axis === 1 ? node.min.y : node.min.z;
    const length = axis === 0 ? extent.x : axis === 1 ? extent.y : extent.z;

    // 軸の長さがほぼゼロ（全ポリゴンが同じ位置など）なら分割不能なので終了
    if (length < 1e-6) return;

    // --- 1. Binning (ビニング) ---
    const BINS = 16; // ビン数 (増やしすぎるとビルドが遅くなる。16程度が丁度いい)
    const bins: Bin[] = [];
    for (let i = 0; i < BINS; i++) {
      bins.push({ bounds: emptyAABB(), count: 0 });
    }

    // 全プリミティブをビンに振り分け
    for (let i = 0; i < node.triCount; i++) {
      const prim = this.sortedPrims[node.leftFirst + i];
      const center = prim.getAABB().center;
      const pos = axis === 0 ? center.x : axis === 1 ? center.y : center.z;

      let binIdx = Math.floor(((pos - minVal) / length) * BINS);
      if (binIdx >= BINS) binIdx = BINS - 1; // 誤差対策
      if (binIdx < 0) binIdx = 0;

      bins[binIdx].count++;
      growAABB(bins[binIdx].bounds, prim.getAABB());
    }

    // --- 2. SAH コスト計算 ---
    // 左側と右側の累積データを計算
    const leftArea = new Float32Array(BINS);
    const leftCount = new Float32Array(BINS);
    const rightArea = new Float32Array(BINS);
    const rightCount = new Float32Array(BINS);

    // Left Sweep
    let currentBox = emptyAABB();
    let currentCount = 0;
    for (let i = 0; i < BINS; i++) {
      if (bins[i].count > 0) {
        growAABB(currentBox, bins[i].bounds);
        currentCount += bins[i].count;
      }
      leftArea[i] = aabbArea(currentBox);
      leftCount[i] = currentCount;
    }

    // Right Sweep
    currentBox = emptyAABB();
    currentCount = 0;
    for (let i = BINS - 1; i >= 0; i--) {
      if (bins[i].count > 0) {
        growAABB(currentBox, bins[i].bounds);
        currentCount += bins[i].count;
      }
      rightArea[i] = aabbArea(currentBox);
      rightCount[i] = currentCount;
    }

    // ベストな分割位置を探す
    let bestCost = Infinity;
    let bestSplit = -1; // ビンのインデックス (bestSplit番目のビンの右で切る)

    // ビンの境目 (0番目～BINS-2番目の後ろ) で試行
    for (let i = 0; i < BINS - 1; i++) {
      // 片方が空になる分割は意味がないのでスキップ
      if (leftCount[i] === 0 || rightCount[i + 1] === 0) continue;

      // SAH Cost = LeftArea * LeftCount + RightArea * RightCount
      const cost = leftArea[i] * leftCount[i] + rightArea[i + 1] * rightCount[i + 1];

      if (cost < bestCost) {
        bestCost = cost;
        bestSplit = i;
      }
    }

    // 良い分割が見つからなかった場合（コスト改善しない、または片寄る）は葉ノードにする
    if (bestSplit === -1) {
      return;
    }

    // --- 3. パーティショニング (並び替え) ---
    // クイックソートの Partition のような処理
    // bestSplit 以下のビンに入るものを左へ、それ以外を右へ集める
    let i = node.leftFirst;
    let j = i + node.triCount - 1;

    while (i <= j) {
      // 左側にあるべきものを探してスキップ
      const primI = this.sortedPrims[i];
      const centerI = primI.getAABB().center;
      const posI = axis === 0 ? centerI.x : axis === 1 ? centerI.y : centerI.z;
      let binIdxI = Math.floor(((posI - minVal) / length) * BINS);
      if (binIdxI >= BINS) binIdxI = BINS - 1;
      if (binIdxI < 0) binIdxI = 0;

      if (binIdxI <= bestSplit) {
        i++;
      } else {
        // 右側にあるべきものを見つけたので、右側から交換候補を探す
        const primJ = this.sortedPrims[j];
        const centerJ = primJ.getAABB().center;
        const posJ = axis === 0 ? centerJ.x : axis === 1 ? centerJ.y : centerJ.z;
        let binIdxJ = Math.floor(((posJ - minVal) / length) * BINS);
        if (binIdxJ >= BINS) binIdxJ = BINS - 1;
        if (binIdxJ < 0) binIdxJ = 0;

        if (binIdxJ > bestSplit) {
          j--;
        } else {
          // 交換 (Swap)
          const temp = this.sortedPrims[i];
          this.sortedPrims[i] = this.sortedPrims[j];
          this.sortedPrims[j] = temp;
          i++;
          j--;
        }
      }
    }

    // 左側の子の要素数
    const leftCountReal = i - node.leftFirst;

    // 安全策: 万が一どちらかが0個になってしまったら分割失敗として終了
    if (leftCountReal === 0 || leftCountReal === node.triCount) return;

    // 子ノードの作成
    const leftChildIdx = this.nodes.length;
    // 左の子
    this.nodes.push({
      min: { x: 0, y: 0, z: 0 }, max: { x: 0, y: 0, z: 0 },
      leftFirst: node.leftFirst,
      triCount: leftCountReal
    });
    // 右の子
    this.nodes.push({
      min: { x: 0, y: 0, z: 0 }, max: { x: 0, y: 0, z: 0 },
      leftFirst: i,
      triCount: node.triCount - leftCountReal
    });

    // 現在のノードを内部ノード化
    node.leftFirst = leftChildIdx;
    node.triCount = 0;

    // 再帰処理 (AABBを再計算してから潜る)
    this.updateNodeBounds(leftChildIdx);
    this.updateNodeBounds(leftChildIdx + 1);
    this.subdivide(leftChildIdx);
    this.subdivide(leftChildIdx + 1);
  }

  packNodes() {
    const data = new Float32Array(this.nodes.length * 8);
    for (let i = 0; i < this.nodes.length; i++) {
      const n = this.nodes[i];
      const off = i * 8;
      data[off] = n.min.x; data[off + 1] = n.min.y; data[off + 2] = n.min.z; data[off + 3] = n.leftFirst;
      data[off + 4] = n.max.x; data[off + 5] = n.max.y; data[off + 6] = n.max.z; data[off + 7] = n.triCount;
    }
    return data;
  }
}
