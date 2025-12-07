// bvh.ts
import { type Vec3 } from './math';
import { type IHittable, PRIMITIVE_STRIDE } from './primitives';

export interface BVHNode {
  min: Vec3; max: Vec3;
  leftFirst: number; // 参照リストの開始インデックス
  triCount: number;  // 要素数
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

    // 再帰的に分割
    this.subdivide(0);

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

  subdivide(nodeIdx: number) {
    const node = this.nodes[nodeIdx];
    if (node.triCount <= 4) return;

    const extent = {
      x: node.max.x - node.min.x,
      y: node.max.y - node.min.y,
      z: node.max.z - node.min.z,
    };
    let axis = 0;
    if (extent.y > extent.x) axis = 1;
    if (extent.z > extent.x && extent.z > extent.y) axis = 2;

    const splitPos = axis === 0 ? node.min.x + extent.x * 0.5
      : axis === 1 ? node.min.y + extent.y * 0.5
        : node.min.z + extent.z * 0.5;

    let i = node.leftFirst;
    let j = i + node.triCount - 1;

    while (i <= j) {
      // getAABB().center を使う
      const center = this.sortedPrims[i].getAABB().center;
      const pos = axis === 0 ? center.x : axis === 1 ? center.y : center.z;

      if (pos < splitPos) {
        i++;
      } else {
        const temp = this.sortedPrims[i];
        this.sortedPrims[i] = this.sortedPrims[j];
        this.sortedPrims[j] = temp;
        j--;
      }
    }

    const leftCount = i - node.leftFirst;
    if (leftCount === 0 || leftCount === node.triCount) return;

    const leftChildIdx = this.nodes.length;
    this.nodes.push({ min: { x: 0, y: 0, z: 0 }, max: { x: 0, y: 0, z: 0 }, leftFirst: node.leftFirst, triCount: leftCount });
    this.nodes.push({ min: { x: 0, y: 0, z: 0 }, max: { x: 0, y: 0, z: 0 }, leftFirst: i, triCount: node.triCount - leftCount });

    node.leftFirst = leftChildIdx;
    node.triCount = 0;

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
