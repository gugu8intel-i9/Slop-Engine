// src/animation.rs
// Handles skeletal animation, blend trees, keyframes, interpolation

use glam::{Mat4, Quat, Vec3};
use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;

/// A single keyframe in an animation track
#[derive(Clone, Debug)]
pub struct Keyframe {
    pub time: f32,
    pub position: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Keyframe {
    pub fn new(time: f32, position: Vec3, rotation: Quat, scale: Vec3) -> Self {
        Self { time, position, rotation, scale }
    }

    /// Linear interpolation between two keyframes
    pub fn lerp(&self, other: &Keyframe, t: f32) -> Keyframe {
        let t = t.clamp(0.0, 1.0);
        Keyframe {
            time: self.time + (other.time - self.time) * t,
            position: self.position.lerp(other.position, t),
            rotation: self.rotation.slerp(other.rotation, t),
            scale: self.scale.lerp(other.scale, t),
        }
    }
}

/// Animation track for a single bone
#[derive(Clone, Debug)]
pub struct AnimationTrack {
    pub bone_id: u32,
    pub keyframes: Vec<Keyframe>,
}

impl AnimationTrack {
    pub fn new(bone_id: u32, keyframes: Vec<Keyframe>) -> Self {
        Self { bone_id, keyframes }
    }

    /// Sample the track at a given time
    pub fn sample(&self, time: f32) -> Keyframe {
        if self.keyframes.is_empty() {
            return Keyframe::new(0.0, Vec3::ZERO, Quat::IDENTITY, Vec3::ONE);
        }

        if self.keyframes.len() == 1 {
            return self.keyframes[0].clone();
        }

        // Find surrounding keyframes
        let mut prev_idx = 0;
        let mut next_idx = self.keyframes.len() - 1;

        for i in 0..self.keyframes.len() - 1 {
            if self.keyframes[i].time <= time && self.keyframes[i + 1].time > time {
                prev_idx = i;
                next_idx = i + 1;
                break;
            }
        }

        let prev = &self.keyframes[prev_idx];
        let next = &self.keyframes[next_idx];

        let duration = next.time - prev.time;
        let t = if duration > 0.0 { (time - prev.time) / duration } else { 0.0 };

        prev.lerp(next, t)
    }
}

/// Complete animation clip
#[derive(Clone, Debug)]
pub struct AnimationClip {
    pub name: String,
    pub duration: f32,
    pub tracks: Vec<AnimationTrack>,
    pub loop_mode: LoopMode,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LoopMode {
    Once,
    Loop,
    PingPong,
}

impl AnimationClip {
    pub fn new(name: String, duration: f32, tracks: Vec<AnimationTrack>, loop_mode: LoopMode) -> Self {
        Self { name, duration, tracks, loop_mode }
    }

    /// Sample all tracks at a given time
    pub fn sample(&self, time: f32) -> HashMap<u32, Keyframe> {
        let mut result = HashMap::new();
        let adjusted_time = self.adjust_time(time);

        for track in &self.tracks {
            let keyframe = track.sample(adjusted_time);
            result.insert(track.bone_id, keyframe);
        }

        result
    }

    fn adjust_time(&self, time: f32) -> f32 {
        match self.loop_mode {
            LoopMode::Once => time.min(self.duration),
            LoopMode::Loop => {
                if self.duration > 0.0 {
                    time.rem_euclid(self.duration)
                } else {
                    0.0
                }
            }
            LoopMode::PingPong => {
                if self.duration > 0.0 {
                    let cycle = time / self.duration;
                    let phase = cycle.fract();
                    if cycle.floor() as i32 % 2 == 0 {
                        phase * self.duration
                    } else {
                        (1.0 - phase) * self.duration
                    }
                } else {
                    0.0
                }
            }
        }
    }
}

/// Blend tree node for combining animations
#[derive(Clone)]
pub enum BlendTreeNode {
    /// Single animation clip
    Clip {
        clip_id: String,
        weight: f32,
        playback_speed: f32,
        current_time: f32,
    },
    /// Linear blend between two nodes
    Blend1D {
        parameter: String,
        threshold_min: f32,
        threshold_max: f32,
        node_a: Box<BlendTreeNode>,
        node_b: Box<BlendTreeNode>,
    },
    /// 2D blend space (e.g., for movement)
    Blend2D {
        parameter_x: String,
        parameter_y: String,
        points: Vec<BlendPoint>,
    },
    /// Additive overlay (e.g., upper body aim)
    Additive {
        base: Box<BlendTreeNode>,
        additive: Box<BlendTreeNode>,
        weight: f32,
    },
    /// State machine transition
    StateMachine {
        states: Vec<String>,
        current_state: usize,
        transitions: Vec<StateTransition>,
    },
}

#[derive(Clone)]
pub struct BlendPoint {
    pub x: f32,
    pub y: f32,
    pub node: BlendTreeNode,
}

#[derive(Clone)]
pub struct StateTransition {
    pub from_state: usize,
    pub to_state: usize,
    pub condition: TransitionCondition,
    pub blend_duration: f32,
}

#[derive(Clone)]
pub enum TransitionCondition {
    ParameterGreater(String, f32),
    ParameterLess(String, f32),
    ParameterEqual(String, f32),
    AnimationFinished,
    Custom(Box<dyn Fn(&HashMap<String, f32>) -> bool + Send + Sync>),
}

impl BlendTreeNode {
    /// Evaluate the blend tree and return combined pose
    pub fn evaluate(
        &self,
        clips: &HashMap<String, Arc<AnimationClip>>,
        parameters: &HashMap<String, f32>,
        bone_count: usize,
    ) -> Vec<Mat4> {
        let mut poses = vec![Mat4::IDENTITY; bone_count];

        self.evaluate_recursive(clips, parameters, &mut poses, 1.0);

        poses
    }

    fn evaluate_recursive(
        &self,
        clips: &HashMap<String, Arc<AnimationClip>>,
        parameters: &HashMap<String, f32>,
        poses: &mut [Mat4],
        parent_weight: f32,
    ) {
        match self {
            BlendTreeNode::Clip { clip_id, weight, playback_speed, current_time } => {
                if let Some(clip) = clips.get(clip_id) {
                    let sampled = clip.sample(*current_time * playback_speed);
                    let effective_weight = parent_weight * weight;

                    for (bone_id, keyframe) in sampled {
                        if bone_id as usize < poses.len() {
                            let transform = Mat4::from_scale_rotation_translation(
                                keyframe.scale,
                                keyframe.rotation,
                                keyframe.position,
                            );

                            // Apply weighted blend
                            let current = poses[bone_id as usize];
                            poses[bone_id as usize] = current.lerp(transform, effective_weight);
                        }
                    }
                }
            }
            BlendTreeNode::Blend1D { parameter, threshold_min, threshold_max, node_a, node_b } => {
                let value = parameters.get(parameter).copied().unwrap_or(0.0);
                let t = ((value - threshold_min) / (threshold_max - threshold_min)).clamp(0.0, 1.0);

                node_a.evaluate_recursive(clips, parameters, poses, parent_weight * (1.0 - t));
                node_b.evaluate_recursive(clips, parameters, poses, parent_weight * t);
            }
            BlendTreeNode::Blend2D { parameter_x, parameter_y, points } => {
                let x = parameters.get(parameter_x).copied().unwrap_or(0.0);
                let y = parameters.get(parameter_y).copied().unwrap_or(0.0);

                // Bilinear interpolation based on nearest 4 points
                for point in points {
                    let dx = (point.x - x).abs();
                    let dy = (point.y - y).abs();
                    let dist = dx + dy;
                    let weight = (1.0 - dist.min(1.0)).max(0.0);

                    point.node.evaluate_recursive(clips, parameters, poses, parent_weight * weight);
                }
            }
            BlendTreeNode::Additive { base, additive, weight } => {
                let mut additive_poses = vec![Mat4::IDENTITY; poses.len()];

                base.evaluate_recursive(clips, parameters, poses, parent_weight);
                additive.evaluate_recursive(clips, parameters, &mut additive_poses, parent_weight * weight);

                // Apply additive transforms
                for i in 0..poses.len() {
                    let base_transform = poses[i];
                    let additive_transform = additive_poses[i];

                    // Extract delta from identity
                    let delta_pos = additive_transform.translation;
                    let delta_rot = additive_transform.to_scale_rotation().1;
                    let delta_scale = additive_transform.to_scale_rotation().0;

                    // Apply delta to base
                    poses[i] = Mat4::from_scale_rotation_translation(
                        base_transform.to_scale_rotation().0 * delta_scale,
                        base_transform.to_scale_rotation().1 * delta_rot,
                        base_transform.translation + delta_pos,
                    );
                }
            }
            BlendTreeNode::StateMachine { states: _, current_state, transitions } => {
                // Simple state machine - just use current state index as clip selection
                // In a full implementation, this would be more sophisticated
                for transition in transitions {
                    if transition.from_state == *current_state {
                        // Check condition (simplified)
                        // Actual implementation would evaluate conditions
                    }
                }
            }
        }
    }

    /// Update animation time
    pub fn update(&mut self, dt: f32, clips: &HashMap<String, Arc<AnimationClip>>) {
        match self {
            BlendTreeNode::Clip { clip_id, playback_speed, current_time, .. } => {
                if let Some(clip) = clips.get(clip_id) {
                    *current_time += dt * playback_speed;

                    match clip.loop_mode {
                        LoopMode::Once => {
                            *current_time = current_time.min(clip.duration);
                        }
                        LoopMode::Loop | LoopMode::PingPong => {
                            // Time adjustment handled in sample()
                        }
                    }
                }
            }
            BlendTreeNode::Blend1D { node_a, node_b, .. } => {
                node_a.update(dt, clips);
                node_b.update(dt, clips);
            }
            BlendTreeNode::Blend2D { points, .. } => {
                for point in points {
                    // Clone to avoid borrow issues
                    let mut node = point.node.clone();
                    node.update(dt, clips);
                }
            }
            BlendTreeNode::Additive { base, additive, .. } => {
                base.update(dt, clips);
                additive.update(dt, clips);
            }
            BlendTreeNode::StateMachine { .. } => {
                // State machine updates handled separately
            }
        }
    }
}

/// Animation controller for managing playback
#[derive(Clone)]
pub struct AnimationController {
    pub blend_tree_root: BlendTreeNode,
    pub parameters: HashMap<String, f32>,
    pub active_clips: HashMap<String, Arc<AnimationClip>>,
    pub is_playing: bool,
}

impl AnimationController {
    pub fn new(blend_tree_root: BlendTreeNode) -> Self {
        Self {
            blend_tree_root,
            parameters: HashMap::new(),
            active_clips: HashMap::new(),
            is_playing: true,
        }
    }

    pub fn set_parameter(&mut self, name: &str, value: f32) {
        self.parameters.insert(name.to_string(), value);
    }

    pub fn get_parameter(&self, name: &str) -> f32 {
        self.parameters.get(name).copied().unwrap_or(0.0)
    }

    pub fn add_clip(&mut self, id: String, clip: Arc<AnimationClip>) {
        self.active_clips.insert(id, clip);
    }

    pub fn update(&mut self, dt: f32) {
        if !self.is_playing {
            return;
        }

        self.blend_tree_root.update(dt, &self.active_clips);
    }

    pub fn evaluate(&self, bone_count: usize) -> Vec<Mat4> {
        self.blend_tree_root.evaluate(&self.active_clips, &self.parameters, bone_count)
    }
}

/// GPU-ready animated bone data
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuBone {
    pub matrix: [[f32; 4]; 4],
}

impl From<Mat4> for GpuBone {
    fn from(mat: Mat4) -> Self {
        Self {
            matrix: mat.to_cols_array_2d(),
        }
    }
}

/// Skeletal animation system
pub struct AnimationSystem {
    pub controllers: RwLock<HashMap<u32, AnimationController>>,
    pub clips: RwLock<HashMap<String, Arc<AnimationClip>>>,
    pub gpu_bones_buffer: RwLock<Option<wgpu::Buffer>>,
    pub max_bones: usize,
}

impl AnimationSystem {
    pub fn new(max_bones: usize, device: &wgpu::Device) -> Self {
        let buffer_size = max_bones * std::mem::size_of::<GpuBone>() as u64;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Animation Bones Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            controllers: RwLock::new(HashMap::new()),
            clips: RwLock::new(HashMap::new()),
            gpu_bones_buffer: RwLock::new(Some(buffer)),
            max_bones,
        }
    }

    pub fn register_clip(&self, clip: AnimationClip) {
        let name = clip.name.clone();
        self.clips.write().insert(name, Arc::new(clip));
    }

    pub fn create_controller(&self, entity_id: u32, root_node: BlendTreeNode) {
        let mut controllers = self.controllers.write();
        let mut controller = AnimationController::new(root_node);

        // Copy all registered clips to this controller
        for (id, clip) in self.clips.read().iter() {
            controller.add_clip(id.clone(), clip.clone());
        }

        controllers.insert(entity_id, controller);
    }

    pub fn update(&self, dt: f32) {
        let mut controllers = self.controllers.write();
        for controller in controllers.values_mut() {
            controller.update(dt);
        }
    }

    pub fn get_bone_matrices(&self, entity_id: u32, bone_count: usize) -> Vec<Mat4> {
        let controllers = self.controllers.read();
        if let Some(controller) = controllers.get(&entity_id) {
            controller.evaluate(bone_count)
        } else {
            vec![Mat4::IDENTITY; bone_count]
        }
    }

    pub fn upload_bones(&self, device: &wgpu::Device, queue: &wgpu::Queue, bones: &[GpuBone]) {
        if let Some(ref buffer) = *self.gpu_bones_buffer.read() {
            queue.write_buffer(
                buffer,
                0,
                bytemuck::cast_slice(bones),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keyframe_interpolation() {
        let kf1 = Keyframe::new(0.0, Vec3::ZERO, Quat::IDENTITY, Vec3::ONE);
        let kf2 = Keyframe::new(1.0, Vec3::X, Quat::from_rotation_y(std::f32::consts::PI), Vec3::splat(2.0));

        let mid = kf1.lerp(&kf2, 0.5);

        assert!((mid.position.x - 0.5).abs() < 0.001);
        assert!((mid.scale.x - 1.5).abs() < 0.001);
    }

    #[test]
    fn test_animation_track_sampling() {
        let keyframes = vec![
            Keyframe::new(0.0, Vec3::ZERO, Quat::IDENTITY, Vec3::ONE),
            Keyframe::new(1.0, Vec3::X, Quat::IDENTITY, Vec3::ONE),
        ];

        let track = AnimationTrack::new(0, keyframes);
        let result = track.sample(0.5);

        assert!((result.position.x - 0.5).abs() < 0.001);
    }
}
