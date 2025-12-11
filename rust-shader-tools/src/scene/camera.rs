use glam::Vec3;

#[derive(Clone, Copy, Debug)]
pub struct CameraConfig {
    pub lookfrom: Vec3,
    pub lookat: Vec3,
    pub vup: Vec3,
    pub vfov: f32,
    pub defocus_angle: f32,
    pub focus_dist: f32,
}

impl CameraConfig {
    pub fn create_buffer(&self, aspect_ratio: f32) -> [f32; 24] {
        let theta = self.vfov.to_radians();
        let h = (theta / 2.0).tan();
        let viewport_height = 2.0 * h * self.focus_dist;
        let viewport_width = viewport_height * aspect_ratio;

        let w = (self.lookfrom - self.lookat).normalize();
        let u = self.vup.cross(w).normalize();
        let v = w.cross(u);

        let horizontal = u * viewport_width;
        let vertical = v * viewport_height;
        let lower_left = self.lookfrom - horizontal * 0.5 - vertical * 0.5 - w * self.focus_dist;

        let lens_radius = self.focus_dist * (self.defocus_angle.to_radians() / 2.0).tan();

        [
            self.lookfrom.x,
            self.lookfrom.y,
            self.lookfrom.z,
            lens_radius,
            lower_left.x,
            lower_left.y,
            lower_left.z,
            0.0,
            horizontal.x,
            horizontal.y,
            horizontal.z,
            0.0,
            vertical.x,
            vertical.y,
            vertical.z,
            0.0,
            u.x,
            u.y,
            u.z,
            0.0,
            v.x,
            v.y,
            v.z,
            0.0,
        ]
    }
}
