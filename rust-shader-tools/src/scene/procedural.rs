// src/scene/procedural.rs
use super::{CameraConfig, SceneData, SceneInstance, helpers, mat_type};
use crate::geometry::Geometry;
use crate::mesh::Mesh;
use glam::{Mat4, Vec3, vec3};

// Helper: 単一のIdentityインスタンスを作成
fn create_instances() -> Vec<SceneInstance> {
    vec![SceneInstance {
        transform: Mat4::IDENTITY,
        geometry_index: 0,
    }]
}

// --- 1. Cornell Box ---
pub fn create_cornell_box(loaded_mesh: Option<&Mesh>) -> SceneData {
    let mut geom = Geometry::new();
    let white = vec3(0.73, 0.73, 0.73);
    let red = vec3(0.65, 0.05, 0.05);
    let green = vec3(0.12, 0.45, 0.15);
    let light = vec3(20.0, 20.0, 20.0);

    let s = 555.0;
    let v = |x: f32, y: f32, z: f32| vec3(x / s * 2. - 1., y / s * 2., z / s * 2. - 1.);
    let sz = |x: f32, y: f32, z: f32| vec3(x / s * 2., y / s * 2., z / s * 2.);

    // Walls
    helpers::add_quad(
        &mut geom,
        v(0., 0., 0.),
        v(555., 0., 0.),
        v(555., 0., 555.),
        v(0., 0., 555.),
        white,
        mat_type::LAMBERTIAN,
        0.,
        -1.0,
    );
    helpers::add_quad(
        &mut geom,
        v(0., 555., 0.),
        v(0., 555., 555.),
        v(555., 555., 555.),
        v(555., 555., 0.),
        white,
        mat_type::LAMBERTIAN,
        0.,
        -1.0,
    );
    helpers::add_quad(
        &mut geom,
        v(0., 0., 555.),
        v(555., 0., 555.),
        v(555., 555., 555.),
        v(0., 555., 555.),
        white,
        mat_type::LAMBERTIAN,
        0.,
        -1.0,
    );
    helpers::add_quad(
        &mut geom,
        v(0., 0., 0.),
        v(0., 555., 0.),
        v(0., 555., 555.),
        v(0., 0., 555.),
        green,
        mat_type::LAMBERTIAN,
        0.,
        -1.0,
    );
    helpers::add_quad(
        &mut geom,
        v(555., 0., 0.),
        v(555., 0., 555.),
        v(555., 555., 555.),
        v(555., 555., 0.),
        red,
        mat_type::LAMBERTIAN,
        0.,
        -1.0,
    );

    // Light
    helpers::add_quad(
        &mut geom,
        v(213., 554., 227.),
        v(343., 554., 227.),
        v(343., 554., 332.),
        v(213., 554., 332.),
        light,
        mat_type::LIGHT,
        0.,
        -1.0,
    );

    // Objects or Default Boxes
    if let Some(mesh) = loaded_mesh {
        let mut mesh_geo = Geometry::from_mesh(mesh);
        mesh_geo.normalize_scale();

        let geometries = vec![geom, mesh_geo];
        let instances = vec![
            SceneInstance {
                transform: Mat4::IDENTITY,
                geometry_index: 0,
            },
            SceneInstance {
                transform: Mat4::from_translation(vec3(0.0, 1.0, 0.0))
                    * Mat4::from_scale(Vec3::splat(2.0)),
                geometry_index: 1,
            },
        ];

        return SceneData {
            camera: CameraConfig {
                lookfrom: vec3(0., 1., -1.0),
                lookat: vec3(0., 1., 0.),
                vup: vec3(0., 1., 0.),
                vfov: 60.,
                defocus_angle: 0.,
                focus_dist: 2.4,
            },
            geometries,
            instances,
            nodes: Vec::new(),
            skins: Vec::new(),
            animations: Vec::new(),
            textures: Vec::new(),
        };
    } else {
        // Default Boxes
        helpers::create_box(
            &mut geom,
            sz(165., 330., 165.),
            v(297.5, 165., 378.5),
            -15.,
            white,
            mat_type::LAMBERTIAN,
            0.,
            -1.0,
        );
        helpers::create_box(
            &mut geom,
            sz(165., 165., 165.),
            v(232.5, 82.5, 147.5),
            18.,
            white,
            mat_type::LAMBERTIAN,
            0.,
            -1.0,
        );
    }

    SceneData {
        camera: CameraConfig {
            lookfrom: vec3(0., 1., -2.4),
            lookat: vec3(0., 1., 0.),
            vup: vec3(0., 1., 0.),
            vfov: 60.,
            defocus_angle: 0.,
            focus_dist: 2.4,
        },
        geometries: vec![geom],
        instances: create_instances(),
        nodes: Vec::new(),
        skins: Vec::new(),
        animations: Vec::new(),
        textures: Vec::new(),
    }
}

// --- 2. Random Spheres ---
pub fn create_random_spheres() -> SceneData {
    let mut geom = Geometry::new();

    geom.add_sphere(
        vec3(0., -1000., 0.),
        1000.,
        vec3(0.5, 0.5, 0.5),
        mat_type::LAMBERTIAN,
        0.,
        -1.0,
    );
    geom.add_sphere(
        vec3(-50., 50., -50.),
        30.,
        vec3(3., 2.7, 2.7),
        mat_type::LIGHT,
        0.,
        -1.0,
    );

    for a in -11..11 {
        for b in -11..11 {
            let choose_mat = helpers::rnd();
            let center = vec3(
                a as f32 + 0.9 * helpers::rnd(),
                0.2,
                b as f32 + 0.9 * helpers::rnd(),
            );
            if (center - vec3(4., 0.2, 0.)).length() > 0.9 {
                if choose_mat < 0.8 {
                    let col = vec3(
                        helpers::rnd().powi(2),
                        helpers::rnd().powi(2),
                        helpers::rnd().powi(2),
                    );
                    geom.add_sphere(center, 0.2, col, mat_type::LAMBERTIAN, 0., -1.0);
                } else if choose_mat < 0.95 {
                    let col = vec3(
                        helpers::rnd_range(0.5, 1.),
                        helpers::rnd_range(0.5, 1.),
                        helpers::rnd_range(0.5, 1.),
                    );
                    geom.add_sphere(
                        center,
                        0.2,
                        col,
                        mat_type::METAL,
                        helpers::rnd_range(0., 0.5),
                        -1.0,
                    );
                } else {
                    geom.add_sphere(center, 0.2, vec3(1., 1., 1.), mat_type::DIELECTRIC, 1.5, -1.0);
                }
            }
        }
    }

    geom.add_sphere(
        vec3(0., 1., 0.),
        1.,
        vec3(1., 1., 1.),
        mat_type::DIELECTRIC,
        1.5,
        -1.0,
    );
    geom.add_sphere(
        vec3(-4., 1., 0.),
        1.,
        vec3(0.4, 0.2, 0.1),
        mat_type::LAMBERTIAN,
        0.,
        -1.0,
    );
    geom.add_sphere(
        vec3(4., 1., 0.),
        1.,
        vec3(0.7, 0.6, 0.5),
        mat_type::METAL,
        0.,
        -1.0,
    );

    SceneData {
        camera: CameraConfig {
            lookfrom: vec3(13., 2., 3.),
            lookat: vec3(0., 0., 0.),
            vup: vec3(0., 1., 0.),
            vfov: 20.,
            defocus_angle: 0.6,
            focus_dist: 10.,
        },
        geometries: vec![geom],
        instances: create_instances(),
        nodes: Vec::new(),
        skins: Vec::new(),
        animations: Vec::new(),
        textures: Vec::new(),
    }
}

// --- 3. Mixed Scene ---
pub fn create_mixed_scene() -> SceneData {
    let mut geom = Geometry::new();

    helpers::create_box(
        &mut geom,
        vec3(40., 2., 40.),
        vec3(0., -1.0, 0.),
        0.,
        vec3(0.1, 0.1, 0.1),
        mat_type::METAL,
        0.05,
        -1.0,
    );

    let warm = vec3(40., 30., 10.);
    let la = vec3(-4., 8., 4.);
    helpers::add_quad(
        &mut geom,
        la,
        la + vec3(2., 0., 0.),
        la + vec3(2., 0., 2.),
        la + vec3(0., 0., 2.),
        warm,
        mat_type::LIGHT,
        0.,
        -1.0,
    );

    let cool = vec3(5., 10., 20.);
    let lb = vec3(4., 6., -4.);
    helpers::add_quad(
        &mut geom,
        lb,
        lb + vec3(3., 0., 0.),
        lb + vec3(3., -3., 0.),
        lb + vec3(0., -3., 0.),
        cool,
        mat_type::LIGHT,
        0.,
        -1.0,
    );

    helpers::create_box(
        &mut geom,
        vec3(2., 1., 2.),
        vec3(0., 0.5, 0.),
        0.,
        vec3(0.8, 0.6, 0.2),
        mat_type::METAL,
        0.1,
        -1.0,
    );
    geom.add_sphere(
        vec3(0., 1.8, 0.),
        0.8,
        vec3(1., 1., 1.),
        mat_type::DIELECTRIC,
        1.5,
        -1.0,
    );
    geom.add_sphere(
        vec3(0., 1.8, 0.),
        -0.7,
        vec3(1., 1., 1.),
        mat_type::DIELECTRIC,
        1.0,
        -1.0,
    );

    helpers::create_box(
        &mut geom,
        vec3(0.8, 0.8, 0.8),
        vec3(0., 3.2, 0.),
        15.,
        vec3(0.9, 0.1, 0.1),
        mat_type::METAL,
        0.2,
        -1.0,
    );

    for i in 0..12 {
        let fi = i as f32;
        let angle = fi / 12.0 * std::f32::consts::PI * 2.0;
        let pos = vec3(
            angle.cos() * 4.0,
            1.0 + (angle * 3.0).sin() * 0.5,
            angle.sin() * 4.0,
        );

        if i % 2 == 0 {
            geom.add_sphere(pos, 0.4, vec3(0.8, 0.8, 0.8), mat_type::METAL, 0., -1.0);
        } else {
            let col = vec3(0.5 + 0.5 * fi.cos(), 0.5 + 0.5 * fi.sin(), 0.8);
            helpers::create_box(
                &mut geom,
                vec3(0.6, 0.6, 0.6),
                pos,
                fi * 20.,
                col,
                mat_type::LAMBERTIAN,
                0.,
                -1.0,
            );
        }
    }

    helpers::create_box(
        &mut geom,
        vec3(1., 6., 1.),
        vec3(-4., 3., -6.),
        10.,
        vec3(0.2, 0.2, 0.3),
        mat_type::LAMBERTIAN,
        0.,
        -1.0,
    );
    helpers::create_box(
        &mut geom,
        vec3(1., 4., 1.),
        vec3(4., 2., -5.),
        -20.,
        vec3(0.2, 0.2, 0.3),
        mat_type::LAMBERTIAN,
        0.,
        -1.0,
    );

    SceneData {
        camera: CameraConfig {
            lookfrom: vec3(0., 3.5, 9.),
            lookat: vec3(0., 1.5, 0.),
            vup: vec3(0., 1., 0.),
            vfov: 40.,
            defocus_angle: 0.3,
            focus_dist: 9.0,
        },
        geometries: vec![geom],
        instances: create_instances(),
        nodes: Vec::new(),
        skins: Vec::new(),
        animations: Vec::new(),
        textures: Vec::new(),
    }
}

// --- 4. Special (Glass Box) Scene ---
pub fn create_cornell_box_special() -> SceneData {
    let mut geom = Geometry::new();
    let white = vec3(0.73, 0.73, 0.73);
    let red = vec3(0.65, 0.05, 0.05);
    let green = vec3(0.12, 0.45, 0.15);
    let light = vec3(20.0, 20.0, 20.0);

    let s = 555.0;
    let v = |x: f32, y: f32, z: f32| vec3(x / s * 2. - 1., y / s * 2., z / s * 2. - 1.);
    let sz = |x: f32, y: f32, z: f32| vec3(x / s * 2., y / s * 2., z / s * 2.);

    helpers::add_quad(
        &mut geom,
        v(0., 0., 0.),
        v(555., 0., 0.),
        v(555., 0., 555.),
        v(0., 0., 555.),
        white,
        mat_type::METAL,
        0.1,
        -1.0,
    );
    helpers::add_quad(
        &mut geom,
        v(0., 555., 0.),
        v(0., 555., 555.),
        v(555., 555., 555.),
        v(555., 555., 0.),
        white,
        mat_type::LAMBERTIAN,
        0.,
        -1.0,
    );
    helpers::add_quad(
        &mut geom,
        v(0., 0., 555.),
        v(555., 0., 555.),
        v(555., 555., 555.),
        v(0., 555., 555.),
        white,
        mat_type::LAMBERTIAN,
        0.,
        -1.0,
    );
    helpers::add_quad(
        &mut geom,
        v(0., 0., 0.),
        v(0., 555., 0.),
        v(0., 555., 555.),
        v(0., 0., 555.),
        green,
        mat_type::LAMBERTIAN,
        0.,
        -1.0,
    );
    helpers::add_quad(
        &mut geom,
        v(555., 0., 0.),
        v(555., 0., 555.),
        v(555., 555., 555.),
        v(555., 555., 0.),
        red,
        mat_type::LAMBERTIAN,
        0.,
        -1.0,
    );
    helpers::add_quad(
        &mut geom,
        v(213., 554., 227.),
        v(343., 554., 227.),
        v(343., 554., 332.),
        v(213., 554., 332.),
        light,
        mat_type::LIGHT,
        0.,
        -1.0,
    );

    let tall_pos = v(366., 165., 383.);
    helpers::create_box(
        &mut geom,
        sz(165., 330., 165.),
        tall_pos,
        15.0,
        vec3(0.95, 0.95, 0.95),
        mat_type::DIELECTRIC,
        1.5,
        -1.0,
    );
    let short_pos = v(183., 82.5, 209.);
    helpers::create_box(
        &mut geom,
        sz(165., 165., 165.),
        short_pos,
        -18.0,
        white,
        mat_type::METAL,
        0.2,
        -1.0,
    );
    geom.add_sphere(
        tall_pos,
        (60.0 / s) * 1.0,
        vec3(0.1, 0.1, 10.),
        mat_type::LIGHT,
        0.,
        -1.0,
    );

    SceneData {
        camera: CameraConfig {
            lookfrom: vec3(0., 1., -3.9),
            lookat: vec3(0., 1., 0.),
            vup: vec3(0., 1., 0.),
            vfov: 40.,
            defocus_angle: 0.,
            focus_dist: 2.4,
        },
        geometries: vec![geom],
        instances: create_instances(),
        nodes: Vec::new(),
        skins: Vec::new(),
        animations: Vec::new(),
        textures: Vec::new(),
    }
}

// --- 5. Mesh Scene ---
const CUBE_OBJ: &str = "v -1 -1 1\nv 1 -1 1\nv -1 1 1\nv 1 1 1\nv -1 -1 -1\nv 1 -1 -1\nv -1 1 -1\nv 1 1 -1\nf 1 2 4 3\nf 3 4 8 7\nf 7 8 6 5\nf 5 6 2 1\nf 3 7 5 1\nf 8 4 2 6";

pub fn create_mesh_scene() -> SceneData {
    let mut geom = Geometry::new();
    let mesh = Mesh::new(CUBE_OBJ);

    geom.add_sphere(
        vec3(0., -1000., 0.),
        1000.,
        vec3(0.5, 0.5, 0.5),
        mat_type::LAMBERTIAN,
        0.,
        -1.0,
    );
    geom.add_mesh_instance(
        &mesh,
        vec3(-2., 1., 0.),
        1.0,
        45.,
        vec3(0.8, 0.2, 0.2),
        mat_type::METAL,
        0.2,
        -1.0,
    );
    geom.add_mesh_instance(
        &mesh,
        vec3(0., 1., 1.5),
        1.2,
        0.,
        vec3(1., 1., 1.),
        mat_type::DIELECTRIC,
        1.5,
        -1.0,
    );

    for i in 0..5 {
        let fi = i as f32;
        geom.add_mesh_instance(
            &mesh,
            vec3(2. + fi * 0.5, 0.5 + fi * 0.5, -fi),
            0.5,
            fi * 30.,
            vec3(0.2, 0.4, 0.8),
            mat_type::LAMBERTIAN,
            0.,
            -1.0,
        );
    }

    geom.add_sphere(
        vec3(0., 10., 0.),
        3.,
        vec3(10., 10., 10.),
        mat_type::LIGHT,
        0.,
        -1.0,
    );

    SceneData {
        camera: CameraConfig {
            lookfrom: vec3(0., 3., 6.),
            lookat: vec3(0., 1., 0.),
            vup: vec3(0., 1., 0.),
            vfov: 40.,
            defocus_angle: 0.,
            focus_dist: 6.,
        },
        geometries: vec![geom],
        instances: create_instances(),
        nodes: Vec::new(),
        skins: Vec::new(),
        animations: Vec::new(),
        textures: Vec::new(),
    }
}

// --- 6. Viewer Scene ---
pub fn create_model_viewer_scene(mesh: Option<&Mesh>, has_glb: bool) -> SceneData {
    // Geometry 0: Environment
    let mut geom_env = Geometry::new();
    geom_env.add_sphere(
        vec3(0., -1000., 0.),
        1000.,
        vec3(0.2, 0.2, 0.2),
        mat_type::LAMBERTIAN,
        0.,
        -1.0,
    );
    geom_env.add_sphere(
        vec3(5., 10., 5.),
        3.,
        vec3(15., 15., 15.),
        mat_type::LIGHT,
        0.,
        -1.0,
    );
    geom_env.add_sphere(vec3(-5., 5., 5.), 1., vec3(3., 3., 5.), mat_type::LIGHT, 0., -1.0);

    // Geometry 1: Model
    let mut geom_model = Geometry::new();
    let should_add_dummy = mesh.is_none() && !has_glb;

    if let Some(m) = mesh {
        geom_model.add_mesh_instance(
            m,
            vec3(0., 1., 0.),
            1.,
            0.,
            vec3(0.8, 0.8, 0.8),
            mat_type::LAMBERTIAN,
            0.,
            -1.0,
        );
    } else if should_add_dummy {
        geom_model.add_sphere(
            vec3(0., 1., 0.),
            1.,
            vec3(1., 0., 1.),
            mat_type::LAMBERTIAN,
            0.,
            -1.0,
        );
    }

    // Instances
    let mut instances = Vec::new();
    instances.push(SceneInstance {
        transform: Mat4::IDENTITY,
        geometry_index: 0,
    });

    // Model Instance (if geometry exists)
    if !geom_model.vertices.is_empty() {
        instances.push(SceneInstance {
            transform: Mat4::IDENTITY,
            geometry_index: 1,
        });
    }

    SceneData {
        camera: CameraConfig {
            lookfrom: vec3(0., 3., -3.),
            lookat: vec3(0., 1., 0.),
            vup: vec3(0., 1., 0.),
            vfov: 35.,
            defocus_angle: 0.,
            focus_dist: 6.,
        },
        geometries: vec![geom_env, geom_model],
        instances,
        nodes: Vec::new(),
        skins: Vec::new(),
        animations: Vec::new(),
        textures: Vec::new(),
    }
}
