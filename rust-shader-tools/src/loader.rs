// src/loader.rs
use crate::geometry::Geometry;
use crate::scene::SceneInstance;
use glam::{Mat4, Quat, Vec3, vec3};
use std::collections::HashMap;

pub fn load_gltf(
    geometries: &mut Vec<Geometry>,
    instances: &mut Vec<SceneInstance>,
    glb_data: &[u8],
) {
    // 1. glTF (GLB) をパース
    let (document, buffers, _images) = gltf::import_slice(glb_data).expect("Failed to load glTF");

    // glTF Mesh Index -> List of (Geometry Index in 'geometries')
    // 1つのMeshが複数のPrimitiveを持つ場合、それぞれを別のGeometryとしてロードします
    let mut mesh_map: HashMap<usize, Vec<usize>> = HashMap::new();

    // 2. すべてのメッシュ・プリミティブをGeometryとしてロード
    for mesh in document.meshes() {
        let mut geom_indices = Vec::new();
        for primitive in mesh.primitives() {
            let mut geom = Geometry::new();
            extract_primitive(&primitive, &buffers, &mut geom);

            // 頂点がある場合のみ登録
            if !geom.vertices.is_empty() {
                geometries.push(geom);
                geom_indices.push(geometries.len() - 1);
            }
        }
        mesh_map.insert(mesh.index(), geom_indices);
    }

    // 3. シーンブラフ(ノード階層)をトラバースしてインスタンスを生成
    for scene in document.scenes() {
        for node in scene.nodes() {
            traverse_node(&node, Mat4::IDENTITY, &mesh_map, instances);
        }
    }
}

// ノードを再帰的にトラバース
fn traverse_node(
    node: &gltf::Node,
    parent_transform: Mat4,
    mesh_map: &HashMap<usize, Vec<usize>>,
    instances: &mut Vec<SceneInstance>,
) {
    // ローカル変換行列を計算
    let (t, r, s) = node.transform().decomposed();
    let local_transform =
        Mat4::from_scale_rotation_translation(Vec3::from(s), Quat::from_array(r), Vec3::from(t));
    let world_transform = parent_transform * local_transform;

    // メッシュがあればインスタンスを生成
    if let Some(mesh) = node.mesh() {
        if let Some(geom_indices) = mesh_map.get(&mesh.index()) {
            for &g_idx in geom_indices {
                instances.push(SceneInstance {
                    transform: world_transform,
                    geometry_index: g_idx,
                });
            }
        }
    }

    // 子ノードへ
    for child in node.children() {
        traverse_node(&child, world_transform, mesh_map, instances);
    }
}

// プリミティブからデータを抽出してGeometryに流し込む
fn extract_primitive(
    primitive: &gltf::Primitive,
    buffers: &[gltf::buffer::Data],
    geom: &mut Geometry,
) {
    let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

    // --- A. 頂点座標 ---
    let positions: Vec<Vec3> = reader
        .read_positions()
        .map(|iter| iter.map(|p| vec3(p[0], p[1], p[2])).collect())
        .unwrap_or_default();

    if positions.is_empty() {
        return;
    }
    let vertex_count = positions.len();

    // --- B. 法線 ---
    let normals: Vec<Vec3> = reader
        .read_normals()
        .map(|iter| iter.map(|n| vec3(n[0], n[1], n[2])).collect())
        .unwrap_or_else(|| vec![vec3(0., 1., 0.); vertex_count]);

    // --- C. スキニング ---
    let joints: Vec<[u32; 4]> = reader
        .read_joints(0)
        .map(|iter| {
            iter.into_u16()
                .map(|j| [j[0] as u32, j[1] as u32, j[2] as u32, j[3] as u32])
                .collect()
        })
        .unwrap_or_else(|| vec![[0, 0, 0, 0]; vertex_count]);

    let weights: Vec<[f32; 4]> = reader
        .read_weights(0)
        .map(|iter| iter.into_f32().collect())
        .unwrap_or_else(|| vec![[0.0, 0.0, 0.0, 0.0]; vertex_count]);

    // --- D. インデックス ---
    let indices: Vec<u32> = reader
        .read_indices()
        .map(|iter| iter.into_u32().collect())
        .unwrap_or_else(|| (0..vertex_count as u32).collect());

    // --- E. 書き込み ---
    // マテリアル等は簡易実装 (本来はprimitive.material()から取得)
    let color = vec3(0.8, 0.8, 0.8);
    let mat_type = 0;
    let extra = 0.0;

    // 頂点の追加
    // Geometryは新規作成されたものなのでオフセットは常に0から始まる
    for i in 0..vertex_count {
        geom.push_vertex_skinned(positions[i], normals[i], joints[i], weights[i]);
    }

    // インデックスの追加
    for chunk in indices.chunks(3) {
        if chunk.len() == 3 {
            geom.indices.push(chunk[0]);
            geom.indices.push(chunk[1]);
            geom.indices.push(chunk[2]);
            geom.push_attributes(color, mat_type, extra);
        }
    }
}
