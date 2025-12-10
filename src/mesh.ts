// src/mesh.ts
import { type Vec3 } from './math';
import { Triangle, type IHittable } from './primitives';

export class Mesh {
  vertices: Vec3[] = [];
  indices: number[] = [];

  constructor(objText: string) {
    this.parseObj(objText);
  }

  // OBJパース
  private parseObj(text: string) {
    const lines = text.split('\n');

    // 1始まりのインデックス対応のため、ダミー頂点を入れておく
    // (OBJは通常1-basedですが、0-basedの場合もあるので補正は読み込み時に行う)
    const rawVertices: Vec3[] = [];

    for (const line of lines) {
      const parts = line.trim().split(/\s+/);
      const type = parts[0];

      if (type === 'v') {
        rawVertices.push({
          x: parseFloat(parts[1]),
          y: parseFloat(parts[2]),
          z: parseFloat(parts[3])
        });
      } else if (type === 'f') {
        const faceIndices: number[] = [];
        for (let i = 1; i < parts.length; i++) {
          const seg = parts[i].split('/');
          const idx = parseInt(seg[0]);
          if (!isNaN(idx)) {
            // OBJは1始まりが基本。負の値は相対指定だが今回は非対応で簡易化
            faceIndices.push(idx - 1);
          }
        }
        // 三角形分割 (Fan)
        for (let i = 1; i < faceIndices.length - 1; i++) {
          this.indices.push(faceIndices[0]);
          this.indices.push(faceIndices[i]);
          this.indices.push(faceIndices[i + 1]);
        }
      }
    }
    this.vertices = rawVertices;
  }

  // ★重要: モデルを原点中心・高さ2.0程度のサイズに正規化する
  normalize() {
    if (this.vertices.length === 0) return;

    // AABB計算
    let min = { x: Infinity, y: Infinity, z: Infinity };
    let max = { x: -Infinity, y: -Infinity, z: -Infinity };

    for (const v of this.vertices) {
      min.x = Math.min(min.x, v.x); min.y = Math.min(min.y, v.y); min.z = Math.min(min.z, v.z);
      max.x = Math.max(max.x, v.x); max.y = Math.max(max.y, v.y); max.z = Math.max(max.z, v.z);
    }

    const size = { x: max.x - min.x, y: max.y - min.y, z: max.z - min.z };
    const center = {
      x: (min.x + max.x) * 0.5,
      y: (min.y + max.y) * 0.5,
      z: (min.z + max.z) * 0.5
    };

    // 最大辺の長さを探す
    const maxDim = Math.max(size.x, Math.max(size.y, size.z));
    const scale = maxDim > 0 ? 2.0 / maxDim : 1.0; // 高さ2.0くらいに収める

    // 全頂点を移動・拡縮
    for (let i = 0; i < this.vertices.length; i++) {
      const v = this.vertices[i];
      this.vertices[i] = {
        x: (v.x - center.x) * scale,
        y: (v.y - center.y) * scale,
        z: (v.z - center.z) * scale
      };
    }
  }

  // インスタンス生成
  createInstance(
    pos: Vec3,
    scale: number,
    rotY_deg: number,
    color: Vec3,
    matType: number,
    extra: number = 0.0
  ): IHittable[] {
    const triangles: IHittable[] = [];
    const transform = this.getTransformFn(pos, scale, rotY_deg);

    for (let i = 0; i < this.indices.length; i += 3) {
      const idx0 = this.indices[i];
      const idx1 = this.indices[i + 1];
      const idx2 = this.indices[i + 2];

      // インデックスが範囲外でないかチェック
      if (!this.vertices[idx0] || !this.vertices[idx1] || !this.vertices[idx2]) continue;

      const v0 = transform({ ...this.vertices[idx0] });
      const v1 = transform({ ...this.vertices[idx1] });
      const v2 = transform({ ...this.vertices[idx2] });

      triangles.push(new Triangle(v0, v1, v2, color, matType, extra));
    }
    return triangles;
  }

  private getTransformFn(pos: Vec3, scale: number, rotY_deg: number) {
    const rad = (rotY_deg * Math.PI) / 180.0;
    const c = Math.cos(rad);
    const s = Math.sin(rad);

    return (v: Vec3): Vec3 => {
      let x = v.x * scale;
      let y = v.y * scale;
      let z = v.z * scale;
      const rx = x * c + z * s;
      const rz = -x * s + z * c;
      return { x: rx + pos.x, y: y + pos.y, z: rz + pos.z };
    };
  }
}
