use super::{CameraConfig, SceneData, helpers, mat_type};
use crate::mesh::Mesh;
use crate::primitives::{Primitive, Sphere};
use glam::vec3; // 親モジュールからインポート

// --- 1. Cornell Box ---
pub fn get_cornell_box_scene() -> SceneData {
    let mut prims = Vec::new();
    let white = vec3(0.73, 0.73, 0.73);
    let red = vec3(0.65, 0.05, 0.05);
    let green = vec3(0.12, 0.45, 0.15);
    let light = vec3(20.0, 20.0, 20.0);

    let s_val = 555.0;
    let v = |x: f32, y: f32, z: f32| {
        vec3(
            x / s_val * 2.0 - 1.0,
            y / s_val * 2.0,
            z / s_val * 2.0 - 1.0,
        )
    };
    let s = |x: f32, y: f32, z: f32| vec3(x / s_val * 2.0, y / s_val * 2.0, z / s_val * 2.0);

    helpers::add_quad(
        &mut prims,
        v(0., 0., 0.),
        v(555., 0., 0.),
        v(555., 0., 555.),
        v(0., 0., 555.),
        white,
        mat_type::LAMBERTIAN,
        0.0,
    );
    helpers::add_quad(
        &mut prims,
        v(0., 555., 0.),
        v(0., 555., 555.),
        v(555., 555., 555.),
        v(555., 555., 0.),
        white,
        mat_type::LAMBERTIAN,
        0.0,
    );
    helpers::add_quad(
        &mut prims,
        v(0., 0., 555.),
        v(555., 0., 555.),
        v(555., 555., 555.),
        v(0., 555., 555.),
        white,
        mat_type::LAMBERTIAN,
        0.0,
    );
    helpers::add_quad(
        &mut prims,
        v(0., 0., 0.),
        v(0., 555., 0.),
        v(0., 555., 555.),
        v(0., 0., 555.),
        green,
        mat_type::LAMBERTIAN,
        0.0,
    );
    helpers::add_quad(
        &mut prims,
        v(555., 0., 0.),
        v(555., 0., 555.),
        v(555., 555., 555.),
        v(555., 555., 0.),
        red,
        mat_type::LAMBERTIAN,
        0.0,
    );
    helpers::add_quad(
        &mut prims,
        v(213., 554., 227.),
        v(343., 554., 227.),
        v(343., 554., 332.),
        v(213., 554., 332.),
        light,
        mat_type::LIGHT,
        0.0,
    );

    let tall_box = helpers::create_box(s(165., 330., 165.), white, mat_type::LAMBERTIAN, 0.0);
    helpers::add_transformed(&mut prims, &tall_box, v(297.5, 165., 378.5), -15.0);

    let short_box = helpers::create_box(s(165., 165., 165.), white, mat_type::LAMBERTIAN, 0.0);
    helpers::add_transformed(&mut prims, &short_box, v(232.5, 82.5, 147.5), 18.0);

    SceneData {
        camera: CameraConfig {
            lookfrom: vec3(0.0, 1.0, -2.4),
            lookat: vec3(0.0, 1.0, 0.0),
            vup: vec3(0.0, 1.0, 0.0),
            vfov: 60.0,
            defocus_angle: 0.0,
            focus_dist: 2.4,
        },
        primitives: prims,
    }
}

// --- 2. Random Spheres ---
pub fn get_random_spheres_scene() -> SceneData {
    let mut prims = Vec::new();
    prims.push(Primitive::Sphere(Sphere {
        center: vec3(0., -1000., 0.),
        radius: 1000.,
        color: vec3(0.5, 0.5, 0.5),
        mat_type: mat_type::LAMBERTIAN,
        extra: 0.,
    }));
    prims.push(Primitive::Sphere(Sphere {
        center: vec3(-50., 50., -50.),
        radius: 30.,
        color: vec3(3., 2.7, 2.7),
        mat_type: mat_type::LIGHT,
        extra: 0.,
    }));

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
                        helpers::rnd() * helpers::rnd(),
                        helpers::rnd() * helpers::rnd(),
                        helpers::rnd() * helpers::rnd(),
                    );
                    prims.push(Primitive::Sphere(Sphere {
                        center,
                        radius: 0.2,
                        color: col,
                        mat_type: mat_type::LAMBERTIAN,
                        extra: 0.,
                    }));
                } else if choose_mat < 0.95 {
                    let col = vec3(
                        helpers::rnd_range(0.5, 1.),
                        helpers::rnd_range(0.5, 1.),
                        helpers::rnd_range(0.5, 1.),
                    );
                    prims.push(Primitive::Sphere(Sphere {
                        center,
                        radius: 0.2,
                        color: col,
                        mat_type: mat_type::METAL,
                        extra: helpers::rnd_range(0., 0.5),
                    }));
                } else {
                    prims.push(Primitive::Sphere(Sphere {
                        center,
                        radius: 0.2,
                        color: vec3(1., 1., 1.),
                        mat_type: mat_type::DIELECTRIC,
                        extra: 1.5,
                    }));
                }
            }
        }
    }
    prims.push(Primitive::Sphere(Sphere {
        center: vec3(0., 1., 0.),
        radius: 1.,
        color: vec3(1., 1., 1.),
        mat_type: mat_type::DIELECTRIC,
        extra: 1.5,
    }));
    prims.push(Primitive::Sphere(Sphere {
        center: vec3(-4., 1., 0.),
        radius: 1.,
        color: vec3(0.4, 0.2, 0.1),
        mat_type: mat_type::LAMBERTIAN,
        extra: 0.,
    }));
    prims.push(Primitive::Sphere(Sphere {
        center: vec3(4., 1., 0.),
        radius: 1.,
        color: vec3(0.7, 0.6, 0.5),
        mat_type: mat_type::METAL,
        extra: 0.,
    }));

    SceneData {
        camera: CameraConfig {
            lookfrom: vec3(13., 2., 3.),
            lookat: vec3(0., 0., 0.),
            vup: vec3(0., 1., 0.),
            vfov: 20.0,
            defocus_angle: 0.6,
            focus_dist: 10.0,
        },
        primitives: prims,
    }
}

// --- 3. Mixed Scene ---
pub fn get_mixed_scene() -> SceneData {
    let mut prims = Vec::new();

    let floor_col = vec3(0.1, 0.1, 0.1);
    let floor_box = helpers::create_box(vec3(40., 2., 40.), floor_col, mat_type::METAL, 0.05);
    helpers::add_transformed(&mut prims, &floor_box, vec3(0., -1.0, 0.), 0.);

    let warm_light = vec3(40., 30., 10.);
    let la_pos = vec3(-4., 8., 4.);
    helpers::add_quad(
        &mut prims,
        la_pos,
        la_pos + vec3(2., 0., 0.),
        la_pos + vec3(2., 0., 2.),
        la_pos + vec3(0., 0., 2.),
        warm_light,
        mat_type::LIGHT,
        0.,
    );

    let cool_light = vec3(5., 10., 20.);
    let lb_pos = vec3(4., 6., -4.);
    helpers::add_quad(
        &mut prims,
        lb_pos,
        lb_pos + vec3(3., 0., 0.),
        lb_pos + vec3(3., -3., 0.),
        lb_pos + vec3(0., -3., 0.),
        cool_light,
        mat_type::LIGHT,
        0.,
    );

    let gold = vec3(0.8, 0.6, 0.2);
    let gold_box = helpers::create_box(vec3(2., 1., 2.), gold, mat_type::METAL, 0.1);
    helpers::add_transformed(&mut prims, &gold_box, vec3(0., 0.5, 0.), 0.);

    prims.push(Primitive::Sphere(Sphere {
        center: vec3(0., 1.8, 0.),
        radius: 0.8,
        color: vec3(1., 1., 1.),
        mat_type: mat_type::DIELECTRIC,
        extra: 1.5,
    }));
    prims.push(Primitive::Sphere(Sphere {
        center: vec3(0., 1.8, 0.),
        radius: -0.7,
        color: vec3(1., 1., 1.),
        mat_type: mat_type::DIELECTRIC,
        extra: 1.0,
    }));

    let ruby = vec3(0.9, 0.1, 0.1);
    let ruby_box = helpers::create_box(vec3(0.8, 0.8, 0.8), ruby, mat_type::METAL, 0.2);
    helpers::add_transformed(&mut prims, &ruby_box, vec3(0., 3.2, 0.), 15.);

    for i in 0..12 {
        let fi = i as f32;
        let angle = fi / 12.0 * std::f32::consts::PI * 2.0;
        let x = angle.cos() * 4.0;
        let z = angle.sin() * 4.0;
        let y = 1.0 + (angle * 3.0).sin() * 0.5;

        if i % 2 == 0 {
            prims.push(Primitive::Sphere(Sphere {
                center: vec3(x, y, z),
                radius: 0.4,
                color: vec3(0.8, 0.8, 0.8),
                mat_type: mat_type::METAL,
                extra: 0.,
            }));
        } else {
            let r = 0.5 + 0.5 * fi.cos();
            let g = 0.5 + 0.5 * fi.sin();
            let b_val = 0.8;
            let colored_box = helpers::create_box(
                vec3(0.6, 0.6, 0.6),
                vec3(r, g, b_val),
                mat_type::LAMBERTIAN,
                0.,
            );
            helpers::add_transformed(&mut prims, &colored_box, vec3(x, y, z), fi * 20.);
        }
    }

    let col_pillar = vec3(0.2, 0.2, 0.3);
    let pillar1 = helpers::create_box(vec3(1., 6., 1.), col_pillar, mat_type::LAMBERTIAN, 0.);
    helpers::add_transformed(&mut prims, &pillar1, vec3(-4., 3., -6.), 10.);
    let pillar2 = helpers::create_box(vec3(1., 4., 1.), col_pillar, mat_type::LAMBERTIAN, 0.);
    helpers::add_transformed(&mut prims, &pillar2, vec3(4., 2., -5.), -20.);

    SceneData {
        camera: CameraConfig {
            lookfrom: vec3(0., 3.5, 9.),
            lookat: vec3(0., 1.5, 0.),
            vup: vec3(0., 1., 0.),
            vfov: 40.0,
            defocus_angle: 0.3,
            focus_dist: 9.0,
        },
        primitives: prims,
    }
}

// --- 4. Special Cornell Box ---
pub fn get_cornell_box_special_scene() -> SceneData {
    let mut prims = Vec::new();
    let white = vec3(0.73, 0.73, 0.73);
    let red = vec3(0.65, 0.05, 0.05);
    let green = vec3(0.12, 0.45, 0.15);
    let light = vec3(20.0, 20.0, 20.0);
    let blue_light = vec3(0.1, 0.1, 10.);
    let glass_col = vec3(0.95, 0.95, 0.95);

    let s_val = 555.0;
    let v = |x: f32, y: f32, z: f32| {
        vec3(
            x / s_val * 2.0 - 1.0,
            y / s_val * 2.0,
            z / s_val * 2.0 - 1.0,
        )
    };
    let s = |x: f32, y: f32, z: f32| vec3(x / s_val * 2.0, y / s_val * 2.0, z / s_val * 2.0);

    helpers::add_quad(
        &mut prims,
        v(0., 0., 0.),
        v(555., 0., 0.),
        v(555., 0., 555.),
        v(0., 0., 555.),
        white,
        mat_type::METAL,
        0.1,
    );
    helpers::add_quad(
        &mut prims,
        v(0., 555., 0.),
        v(0., 555., 555.),
        v(555., 555., 555.),
        v(555., 555., 0.),
        white,
        mat_type::LAMBERTIAN,
        0.0,
    );
    helpers::add_quad(
        &mut prims,
        v(0., 0., 555.),
        v(555., 0., 555.),
        v(555., 555., 555.),
        v(0., 555., 555.),
        white,
        mat_type::LAMBERTIAN,
        0.0,
    );
    helpers::add_quad(
        &mut prims,
        v(0., 0., 0.),
        v(0., 555., 0.),
        v(0., 555., 555.),
        v(0., 0., 555.),
        green,
        mat_type::LAMBERTIAN,
        0.0,
    );
    helpers::add_quad(
        &mut prims,
        v(555., 0., 0.),
        v(555., 0., 555.),
        v(555., 555., 555.),
        v(555., 555., 0.),
        red,
        mat_type::LAMBERTIAN,
        0.0,
    );
    helpers::add_quad(
        &mut prims,
        v(213., 554., 227.),
        v(343., 554., 227.),
        v(343., 554., 332.),
        v(213., 554., 332.),
        light,
        mat_type::LIGHT,
        0.0,
    );

    let tall_box_center = v(366., 165., 383.);
    let tall_box = helpers::create_box(s(165., 330., 165.), glass_col, mat_type::DIELECTRIC, 1.5);
    helpers::add_transformed(&mut prims, &tall_box, tall_box_center, 15.0);

    let short_box_center = v(183., 82.5, 209.);
    let short_box = helpers::create_box(s(165., 165., 165.), white, mat_type::METAL, 0.2);
    helpers::add_transformed(&mut prims, &short_box, short_box_center, -18.0);

    prims.push(Primitive::Sphere(Sphere {
        center: tall_box_center,
        radius: (60.0 / s_val) * 1.0,
        color: blue_light,
        mat_type: mat_type::LIGHT,
        extra: 0.0,
    }));

    SceneData {
        camera: CameraConfig {
            lookfrom: vec3(0.0, 1.0, -3.9),
            lookat: vec3(0.0, 1.0, 0.0),
            vup: vec3(0.0, 1.0, 0.0),
            vfov: 40.0,
            defocus_angle: 0.0,
            focus_dist: 2.4,
        },
        primitives: prims,
    }
}

// --- 5. Mesh Scene ---
const CUBE_OBJ_DATA: &str = r#"
v -1 -1  1
v  1 -1  1
v -1  1  1
v  1  1  1
v -1 -1 -1
v  1 -1 -1
v -1  1 -1
v  1  1 -1
f 1 2 4 3
f 3 4 8 7
f 7 8 6 5
f 5 6 2 1
f 3 7 5 1
f 8 4 2 6
"#;

pub fn get_mesh_scene() -> SceneData {
    let mut prims = Vec::new();
    let mesh = Mesh::new(CUBE_OBJ_DATA);

    prims.push(Primitive::Sphere(Sphere {
        center: vec3(0., -1000., 0.),
        radius: 1000.,
        color: vec3(0.5, 0.5, 0.5),
        mat_type: mat_type::LAMBERTIAN,
        extra: 0.,
    }));

    prims.extend(mesh.create_instance(
        vec3(-2., 1., 0.),
        1.0,
        45.,
        vec3(0.8, 0.2, 0.2),
        mat_type::METAL,
        0.2,
    ));
    prims.extend(mesh.create_instance(
        vec3(0., 1., 1.5),
        1.2,
        0.,
        vec3(1., 1., 1.),
        mat_type::DIELECTRIC,
        1.5,
    ));

    for i in 0..5 {
        let fi = i as f32;
        prims.extend(mesh.create_instance(
            vec3(2. + fi * 0.5, 0.5 + fi * 0.5, -fi),
            0.5,
            fi * 30.,
            vec3(0.2, 0.4, 0.8),
            mat_type::LAMBERTIAN,
            0.,
        ));
    }

    prims.push(Primitive::Sphere(Sphere {
        center: vec3(0., 10., 0.),
        radius: 3.,
        color: vec3(10., 10., 10.),
        mat_type: mat_type::LIGHT,
        extra: 0.,
    }));

    SceneData {
        camera: CameraConfig {
            lookfrom: vec3(0., 3., 6.),
            lookat: vec3(0., 1., 0.),
            vup: vec3(0., 1., 0.),
            vfov: 40.0,
            defocus_angle: 0.0,
            focus_dist: 6.0,
        },
        primitives: prims,
    }
}

// --- 6. Model Viewer Scene ---
pub fn get_model_viewer_scene(mesh: Option<&Mesh>) -> SceneData {
    let mut prims = Vec::new();

    let floor_col = vec3(0.2, 0.2, 0.2);
    prims.push(Primitive::Sphere(Sphere {
        center: vec3(0., -1000., 0.),
        radius: 1000.,
        color: floor_col,
        mat_type: mat_type::LAMBERTIAN,
        extra: 0.,
    }));

    prims.push(Primitive::Sphere(Sphere {
        center: vec3(5., 10., 5.),
        radius: 3.0,
        color: vec3(15., 15., 15.),
        mat_type: mat_type::LIGHT,
        extra: 0.,
    }));
    prims.push(Primitive::Sphere(Sphere {
        center: vec3(-5., 5., 5.),
        radius: 1.0,
        color: vec3(3., 3., 5.),
        mat_type: mat_type::LIGHT,
        extra: 0.,
    }));

    if let Some(m) = mesh {
        prims.extend(m.create_instance(
            vec3(0., 1.0, 0.),
            1.0,
            0.,
            vec3(0.8, 0.8, 0.8),
            mat_type::LAMBERTIAN,
            0.,
        ));
        prims.extend(m.create_instance(
            vec3(-2.5, 1.0, -1.0),
            0.8,
            30.,
            vec3(1.0, 1.0, 1.0),
            mat_type::DIELECTRIC,
            1.5,
        ));
        prims.extend(m.create_instance(
            vec3(2.5, 1.0, -1.0),
            0.8,
            -30.,
            vec3(0.8, 0.6, 0.2),
            mat_type::METAL,
            0.1,
        ));
    } else {
        prims.push(Primitive::Sphere(Sphere {
            center: vec3(0., 1., 0.),
            radius: 1.,
            color: vec3(1., 0., 1.),
            mat_type: mat_type::LAMBERTIAN,
            extra: 0.,
        }));
    }

    SceneData {
        camera: CameraConfig {
            lookfrom: vec3(0., 3., 6.),
            lookat: vec3(0., 1., 0.),
            vup: vec3(0., 1., 0.),
            vfov: 35.0,
            defocus_angle: 0.0,
            focus_dist: 6.0,
        },
        primitives: prims,
    }
}

// --- ディスパッチャ ---
pub fn get_scene_data(name: &str, uploaded_mesh: Option<&Mesh>) -> SceneData {
    match name {
        "spheres" => get_random_spheres_scene(),
        "mixed" => get_mixed_scene(),
        "special" => get_cornell_box_special_scene(),
        "mesh" => get_mesh_scene(),
        "viewer" => get_model_viewer_scene(uploaded_mesh),
        _ => get_cornell_box_scene(),
    }
}
