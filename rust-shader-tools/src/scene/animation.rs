// src/scene/animation.rs
use glam::{Quat, Vec3};

#[derive(Clone, Debug)]
pub enum Interpolation {
    Linear,
    Step,
    CubicSpline,
}

#[derive(Clone, Debug)]
pub struct Channel {
    // ターゲットをインデックスではなく「名前」で持つことで、
    // 別のモデルのモーションを適用(リターゲティング)しやすくする
    pub target_node_name: String,
    pub inputs: Vec<f32>, // Time keys
    pub outputs: ChannelOutputs,
    pub interpolation: Interpolation,
}

#[derive(Clone, Debug)]
pub enum ChannelOutputs {
    Translations(Vec<Vec3>),
    Rotations(Vec<Quat>),
    Scales(Vec<Vec3>),
}

#[derive(Clone, Debug)]
pub struct Animation {
    pub name: String,
    pub channels: Vec<Channel>,
    pub duration: f32,
}
