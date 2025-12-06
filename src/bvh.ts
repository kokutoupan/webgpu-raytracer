// bvh.ts
import { type Vec3 } from './math';

export interface BVHNode {
  min: Vec3; max: Vec3;
  leftFirst: number; // 参照リストの開始インデックス
  triCount: number;  // 要素数
}

// 構築用データ（種類とインデックスを持つ）
export interface PrimitiveInfo {
  min: Vec3; max: Vec3; center: Vec3;
  type: number; // 0: Sphere, 1: Triangle
  originalIndex: number; // 元のバッファでのインデックス
}

export class BVHBuilder {
  nodes: BVHNode[] = [];
  sortedPrims: PrimitiveInfo[] = [];

  // 入力: 球の配列と三角形の配列（生データ）を受け取る
  build(spheres: Float32Array, triangles: Float32Array): { bvhNodes: Float32Array<ArrayBuffer>, primitiveRefs: Uint32Array<ArrayBuffer> } {
    this.nodes = [];
    this.sortedPrims = [];

    // --- 1. 全プリミティブのAABB計算 ---

    // A. 球 (Stride = 12 floats)
    // Center(3), Radius(1), Color(3), Mat(1), Extra(1), Pad(3)
    const sphereStride = 12;
    for (let i = 0; i < spheres.length / sphereStride; i++) {
      const off = i * sphereStride;
      const r = spheres[off + 3]; // radius
      // 半径0（ダミー）は無視するか、極小の箱にする
      if (r <= 0) continue;

      const cx = spheres[off], cy = spheres[off + 1], cz = spheres[off + 2];
      this.sortedPrims.push({
        min: { x: cx - r, y: cy - r, z: cz - r },
        max: { x: cx + r, y: cy + r, z: cz + r },
        center: { x: cx, y: cy, z: cz },
        type: 0, // Sphere
        originalIndex: i
      });
    }

    // B. 三角形 (Stride = 16 floats)
    // v0(4), v1(4), v2(4), mat(4)
    const triStride = 16;
    for (let i = 0; i < triangles.length / triStride; i++) {
      const off = i * triStride;
      const v0x = triangles[off], v0y = triangles[off + 1], v0z = triangles[off + 2];
      const v1x = triangles[off + 4], v1y = triangles[off + 5], v1z = triangles[off + 6];
      const v2x = triangles[off + 8], v2y = triangles[off + 9], v2z = triangles[off + 10];

      const min = {
        x: Math.min(v0x, v1x, v2x), y: Math.min(v0y, v1y, v2y), z: Math.min(v0z, v1z, v2z),
      };
      const max = {
        x: Math.max(v0x, v1x, v2x), y: Math.max(v0y, v1y, v2y), z: Math.max(v0z, v1z, v2z),
      };

      // パディング (厚みゼロ対策)
      const EPSILON = 0.001;
      if (max.x - min.x < EPSILON) { min.x -= EPSILON; max.x += EPSILON; }
      if (max.y - min.y < EPSILON) { min.y -= EPSILON; max.y += EPSILON; }
      if (max.z - min.z < EPSILON) { min.z -= EPSILON; max.z += EPSILON; }

      const center = {
        x: (min.x + max.x) * 0.5, y: (min.y + max.y) * 0.5, z: (min.z + max.z) * 0.5,
      };

      this.sortedPrims.push({ min, max, center, type: 1, originalIndex: i }); // Type 1: Triangle
    }

    // --- 2. BVH構築 ---
    const root = { min: { x: 0, y: 0, z: 0 }, max: { x: 0, y: 0, z: 0 }, leftFirst: 0, triCount: this.sortedPrims.length };
    this.nodes.push(root);
    this.updateNodeBounds(0);
    this.subdivide(0);

    // --- 3. データパック ---

    // ノードデータ (Float32)
    const bvhNodeData = this.packNodes();

    // 参照リスト (Uint32)
    // [type, index, type, index, ...] 
    const refData = new Uint32Array(this.sortedPrims.length * 2);
    for (let i = 0; i < this.sortedPrims.length; i++) {
      refData[i * 2] = this.sortedPrims[i].type;
      refData[i * 2 + 1] = this.sortedPrims[i].originalIndex;
    }

    return { bvhNodes: bvhNodeData, primitiveRefs: refData };
  }

  // ... (updateNodeBounds, subdivide, packNodes は以前と同じロジックでOK) ...
  // ただし subdivide 内の参照は this.sortedTris ではなく this.sortedPrims に変えること

  updateNodeBounds(nodeIdx: number) {
    const node = this.nodes[nodeIdx];
    let first = true;
    for (let i = 0; i < node.triCount; i++) {
      const p = this.sortedPrims[node.leftFirst + i];
      if (first) {
        node.min = { ...p.min }; node.max = { ...p.max }; first = false;
      } else {
        node.min.x = Math.min(node.min.x, p.min.x); node.min.y = Math.min(node.min.y, p.min.y); node.min.z = Math.min(node.min.z, p.min.z);
        node.max.x = Math.max(node.max.x, p.max.x); node.max.y = Math.max(node.max.y, p.max.y); node.max.z = Math.max(node.max.z, p.max.z);
      }
    }
  }

  subdivide(nodeIdx: number) {
    const node = this.nodes[nodeIdx];
    if (node.triCount <= 4) return;

    const extent = { x: node.max.x - node.min.x, y: node.max.y - node.min.y, z: node.max.z - node.min.z };
    let axis = 0;
    if (extent.y > extent.x) axis = 1;
    if (extent.z > extent.x && extent.z > extent.y) axis = 2;

    const splitPos = axis === 0 ? node.min.x + extent.x * 0.5 : axis === 1 ? node.min.y + extent.y * 0.5 : node.min.z + extent.z * 0.5;

    let i = node.leftFirst;
    let j = i + node.triCount - 1;

    while (i <= j) {
      const t = this.sortedPrims[i];
      const pos = axis === 0 ? t.center.x : axis === 1 ? t.center.y : t.center.z;
      if (pos < splitPos) { i++; } else {
        const temp = this.sortedPrims[i]; this.sortedPrims[i] = this.sortedPrims[j]; this.sortedPrims[j] = temp;
        j--;
      }
    }

    const leftCount = i - node.leftFirst;
    if (leftCount === 0 || leftCount === node.triCount) return;

    const leftChildIdx = this.nodes.length;
    this.nodes.push({ min: { x: 0, y: 0, z: 0 }, max: { x: 0, y: 0, z: 0 }, leftFirst: node.leftFirst, triCount: leftCount });
    this.nodes.push({ min: { x: 0, y: 0, z: 0 }, max: { x: 0, y: 0, z: 0 }, leftFirst: i, triCount: node.triCount - leftCount });
    node.leftFirst = leftChildIdx; node.triCount = 0;
    this.updateNodeBounds(leftChildIdx); this.updateNodeBounds(leftChildIdx + 1);
    this.subdivide(leftChildIdx); this.subdivide(leftChildIdx + 1);
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
