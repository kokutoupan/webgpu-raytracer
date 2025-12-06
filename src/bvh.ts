// bvh.ts
import { type Vec3 } from './math';

export interface BVHNode {
  min: Vec3;
  max: Vec3;
  leftFirst: number;
  triCount: number;
}

// 内部計算用の三角形データ
interface Tri {
  min: Vec3; max: Vec3; center: Vec3;
  originalIndex: number; // 元の配列でのインデックス（これで追跡する）
}

export class BVHBuilder {
  nodes: BVHNode[] = [];
  // ソートされた三角形リストを保持する
  sortedTris: Tri[] = [];

  // ★戻り値を変更: ノード配列と、並び替え済みの三角形データのセットを返す
  buildAndReorder(originalTriangles: Float32Array): { bvhNodes: Float32Array<ArrayBuffer>, reorderedTriangles: Float32Array<ArrayBuffer> } {
    this.nodes = [];
    this.sortedTris = [];

    // 1. 生データから作業用Triオブジェクトを作る
    // 1トライアングル = 16 floats (64 bytes)
    const stride = 16;
    const triCount = originalTriangles.length / stride;

    for (let i = 0; i < triCount; i++) {
      const off = i * stride;
      // 頂点座標だけ読み取ってAABBを作る
      const v0x = originalTriangles[off], v0y = originalTriangles[off + 1], v0z = originalTriangles[off + 2];
      const v1x = originalTriangles[off + 4], v1y = originalTriangles[off + 5], v1z = originalTriangles[off + 6];
      const v2x = originalTriangles[off + 8], v2y = originalTriangles[off + 9], v2z = originalTriangles[off + 10];

      const min = {
        x: Math.min(v0x, v1x, v2x), y: Math.min(v0y, v1y, v2y), z: Math.min(v0z, v1z, v2z),
      };
      const max = {
        x: Math.max(v0x, v1x, v2x), y: Math.max(v0y, v1y, v2y), z: Math.max(v0z, v1z, v2z),
      };
      // 箱が薄すぎるとレイが当たらないので、最低限の厚み(0.001)を持たせる

      const EPSILON = 0.001;
      if (max.x - min.x < EPSILON) { min.x -= EPSILON; max.x += EPSILON; }
      if (max.y - min.y < EPSILON) { min.y -= EPSILON; max.y += EPSILON; }
      if (max.z - min.z < EPSILON) { min.z -= EPSILON; max.z += EPSILON; }

      const center = {
        x: (min.x + max.x) * 0.5, y: (min.y + max.y) * 0.5, z: (min.z + max.z) * 0.5,
      };

      // originalIndex を記録しておく！
      this.sortedTris.push({ min, max, center, originalIndex: i });
    }

    // 2. BVH構築（ここで this.sortedTris の中身がソートされまくる）
    const root = { min: { x: 0, y: 0, z: 0 }, max: { x: 0, y: 0, z: 0 }, leftFirst: 0, triCount: triCount };
    this.nodes.push(root);
    this.updateNodeBounds(0);
    this.subdivide(0);

    // 3. ノード配列を GPU用 (Float32Array) に変換
    const bvhNodeData = this.packNodes();

    // 4. ★重要: 並び替えられた sortedTris の順序に従って、
    //    オリジナルの triangleData をコピーして作り直す
    const reorderedTriangles = new Float32Array(originalTriangles.length);

    for (let i = 0; i < triCount; i++) {
      const srcIdx = this.sortedTris[i].originalIndex; // 元の場所
      const dstIdx = i;                                // 新しい場所

      // 16 float 分をコピー
      const srcOff = srcIdx * stride;
      const dstOff = dstIdx * stride;
      for (let k = 0; k < stride; k++) {
        reorderedTriangles[dstOff + k] = originalTriangles[srcOff + k];
      }
    }

    return {
      bvhNodes: bvhNodeData,
      reorderedTriangles: reorderedTriangles
    };
  }

  updateNodeBounds(nodeIdx: number) {
    const node = this.nodes[nodeIdx];
    // ノードに含まれる全三角形を含むAABBを計算
    let first = true;
    for (let i = 0; i < node.triCount; i++) {
      const tri = this.sortedTris[node.leftFirst + i];
      if (first) {
        node.min = { ...tri.min };
        node.max = { ...tri.max };
        first = false;
      } else {
        node.min.x = Math.min(node.min.x, tri.min.x);
        node.min.y = Math.min(node.min.y, tri.min.y);
        node.min.z = Math.min(node.min.z, tri.min.z);
        node.max.x = Math.max(node.max.x, tri.max.x);
        node.max.y = Math.max(node.max.y, tri.max.y);
        node.max.z = Math.max(node.max.z, tri.max.z);
      }
    }
  }

  subdivide(nodeIdx: number) {
    const node = this.nodes[nodeIdx];
    // ★ここで分割を止める数を調整（4個以下なら葉ノードにする）
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
      const t = this.sortedTris[i];
      const pos = axis === 0 ? t.center.x : axis === 1 ? t.center.y : t.center.z;
      if (pos < splitPos) {
        i++;
      } else {
        // swap
        const temp = this.sortedTris[i];
        this.sortedTris[i] = this.sortedTris[j];
        this.sortedTris[j] = temp;
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
    // 1ノード 32 bytes (8 floats)
    // min(3), leftFirst(1)
    // max(3), triCount(1)
    const data = new Float32Array(this.nodes.length * 8);
    for (let i = 0; i < this.nodes.length; i++) {
      const n = this.nodes[i];
      const off = i * 8;
      data[off + 0] = n.min.x; data[off + 1] = n.min.y; data[off + 2] = n.min.z;
      data[off + 3] = n.leftFirst;
      data[off + 4] = n.max.x; data[off + 5] = n.max.y; data[off + 6] = n.max.z;
      data[off + 7] = n.triCount;
    }
    return data;
  }
}
