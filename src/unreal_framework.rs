// src/unreal_framework.rs
//! High-Performance Unreal Engine-Style Gameplay Framework & Blueprint VM
//! Provides an optional, cache-friendly Actor-Component architecture,
//! Gameplay Lifecycle (World, GameMode, Character, PlayerController),
//! and a zero-allocation Blueprint Visual Scripting Bytecode VM.

use glam::{Vec3, Vec2, Quat, Mat4};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use serde::{Serialize, Deserialize};
use anyhow::Result;

// ============================================================================
// 1. UNREAL REFLECTION & METADATA SYSTEM
// ============================================================================

/// Property visibility / accessibility flags matching Unreal UPROPERTY macros
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PropertyFlags {
    EditAnywhere,
    EditDefaultsOnly,
    EditInstanceOnly,
    BlueprintReadOnly,
    BlueprintReadWrite,
}

/// Dynamic Value types supported by Unreal Properties and Blueprint Data Pins
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum UValue {
    Boolean(bool),
    Integer(i32),
    Float(f32),
    String(String),
    Vector(Vec3),
    Rotator(Vec3), // Pitch, Yaw, Roll in degrees
    Transform { location: Vec3, rotation: Quat, scale: Vec3 },
    ObjectRef(u64), // Actor or Component ID
    Array(Vec<UValue>),
    None,
}

impl UValue {
    pub fn as_float(&self) -> f32 {
        match self {
            UValue::Float(f) => *f,
            UValue::Integer(i) => *i as f32,
            UValue::Boolean(b) => if *b { 1.0 } else { 0.0 },
            _ => 0.0,
        }
    }

    pub fn as_vector(&self) -> Vec3 {
        match self {
            UValue::Vector(v) => *v,
            _ => Vec3::ZERO,
        }
    }

    pub fn as_bool(&self) -> bool {
        match self {
            UValue::Boolean(b) => *b,
            UValue::Float(f) => *f != 0.0,
            UValue::Integer(i) => *i != 0,
            _ => false,
        }
    }
}

/// Reflection descriptor for UPROPERTY
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UProperty {
    pub name: String,
    pub category: String,
    pub flags: Vec<PropertyFlags>,
    pub default_value: UValue,
}

// ============================================================================
// 2. TRANSFORM & SCENE COMPONENT HIERARCHY
// ============================================================================

/// Spatial transform matching Unreal USceneComponent
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FTransform {
    pub location: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Default for FTransform {
    fn default() -> Self {
        Self {
            location: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }
}

impl FTransform {
    pub fn to_mat4(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.location)
    }

    pub fn transform_position(&self, point: Vec3) -> Vec3 {
        self.location + (self.rotation * (point * self.scale))
    }

    pub fn transform_direction(&self, dir: Vec3) -> Vec3 {
        self.rotation * dir
    }

    pub fn get_forward_vector(&self) -> Vec3 {
        self.rotation * Vec3::X
    }

    pub fn get_right_vector(&self) -> Vec3 {
        self.rotation * Vec3::Y
    }

    pub fn get_up_vector(&self) -> Vec3 {
        self.rotation * Vec3::Z
    }
}

/// Component attachment rule
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EAttachmentRule {
    KeepRelative,
    KeepWorld,
    SnapToTarget,
}

// ============================================================================
// 3. ACTOR & ACTOR COMPONENT SYSTEM
// ============================================================================

/// Base trait / type for all Unreal Components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EComponentType {
    Scene,
    StaticMesh { mesh_path: String, material_path: String },
    Camera { field_of_view: f32, aspect_ratio: f32 },
    SpringArm { target_arm_length: f32, use_padded_lag: bool },
    CharacterMovement { max_walk_speed: f32, jump_z_velocity: f32, gravity_scale: f32 },
    PointLight { intensity: f32, color: Vec3, attenuation_radius: f32 },
    DirectionalLight { intensity: f32, color: Vec3 },
    Audio { sound_asset: String, auto_play: bool },
    ParticleSystem { template_path: String },
    CustomScript { script_name: String },
}

#[derive(Debug, Clone)]
pub struct UActorComponent {
    pub id: u64,
    pub name: String,
    pub component_type: EComponentType,
    pub relative_transform: FTransform,
    pub world_transform: FTransform,
    pub parent_component_id: Option<u64>,
    pub is_active: bool,
    pub properties: HashMap<String, UValue>,
}

impl UActorComponent {
    pub fn new(id: u64, name: impl Into<String>, component_type: EComponentType) -> Self {
        Self {
            id,
            name: name.into(),
            component_type,
            relative_transform: FTransform::default(),
            world_transform: FTransform::default(),
            parent_component_id: None,
            is_active: true,
            properties: HashMap::new(),
        }
    }
}

/// Unreal AActor base class
#[derive(Debug)]
pub struct AActor {
    pub id: u64,
    pub name: String,
    pub tag: String,
    pub root_component_id: Option<u64>,
    pub components: HashMap<u64, UActorComponent>,
    pub tags: HashSet<String>,
    pub is_hidden_in_game: bool,
    pub can_ever_tick: bool,
    pub is_pending_kill: bool,
    pub custom_properties: HashMap<String, UValue>,
    pub blueprint_graph: Option<Arc<BlueprintGraph>>,
}

impl AActor {
    pub fn new(id: u64, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            tag: "Actor".to_string(),
            root_component_id: None,
            components: HashMap::new(),
            tags: HashSet::new(),
            is_hidden_in_game: false,
            can_ever_tick: true,
            is_pending_kill: false,
            custom_properties: HashMap::new(),
            blueprint_graph: None,
        }
    }

    pub fn get_actor_location(&self) -> Vec3 {
        if let Some(root_id) = self.root_component_id {
            if let Some(comp) = self.components.get(&root_id) {
                return comp.world_transform.location;
            }
        }
        Vec3::ZERO
    }

    pub fn set_actor_location(&mut self, new_location: Vec3) {
        if let Some(root_id) = self.root_component_id {
            if let Some(comp) = self.components.get_mut(&root_id) {
                comp.relative_transform.location = new_location;
                comp.world_transform.location = new_location;
            }
        }
    }

    pub fn get_actor_rotation(&self) -> Quat {
        if let Some(root_id) = self.root_component_id {
            if let Some(comp) = self.components.get(&root_id) {
                return comp.world_transform.rotation;
            }
        }
        Quat::IDENTITY
    }

    pub fn set_actor_rotation(&mut self, new_rotation: Quat) {
        if let Some(root_id) = self.root_component_id {
            if let Some(comp) = self.components.get_mut(&root_id) {
                comp.relative_transform.rotation = new_rotation;
                comp.world_transform.rotation = new_rotation;
            }
        }
    }

    pub fn add_actor_local_offset(&mut self, delta_location: Vec3) {
        let rot = self.get_actor_rotation();
        let world_delta = rot * delta_location;
        let current = self.get_actor_location();
        self.set_actor_location(current + world_delta);
    }

    pub fn add_component(&mut self, comp: UActorComponent) -> u64 {
        let id = comp.id;
        if self.root_component_id.is_none() {
            self.root_component_id = Some(id);
        }
        self.components.insert(id, comp);
        id
    }
}

// ============================================================================
// 4. UNREAL GAMEPLAY FRAMEWORK (Character, Controller, GameMode)
// ============================================================================

/// Movement Mode matching Unreal CharacterMovementComponent
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EMovementMode {
    None,
    Walking,
    NavMeshWalking,
    Falling,
    Swimming,
    Flying,
    Custom,
}

/// ACharacter - Actor with capsule collision and character movement
pub struct ACharacter {
    pub base_actor: AActor,
    pub movement_mode: EMovementMode,
    pub velocity: Vec3,
    pub max_walk_speed: f32,
    pub jump_z_velocity: f32,
    pub is_jumping: bool,
    pub is_crouched: bool,
    pub control_rotation: Vec3, // Pitch, Yaw, Roll
}

impl ACharacter {
    pub fn new(id: u64, name: impl Into<String>) -> Self {
        let mut actor = AActor::new(id, name);
        actor.tag = "Character".to_string();
        
        let root = UActorComponent::new(id * 10 + 1, "CapsuleComponent", EComponentType::Scene);
        let mesh = UActorComponent::new(id * 10 + 2, "Mesh", EComponentType::StaticMesh {
            mesh_path: "Engine/BasicShapes/SkeletalMesh.gltf".to_string(),
            material_path: "Engine/Materials/M_DefaultCharacter.json".to_string(),
        });
        let camera = UActorComponent::new(id * 10 + 3, "FollowCamera", EComponentType::Camera {
            field_of_view: 90.0,
            aspect_ratio: 1.77,
        });

        actor.add_component(root);
        actor.add_component(mesh);
        actor.add_component(camera);

        Self {
            base_actor: actor,
            movement_mode: EMovementMode::Walking,
            velocity: Vec3::ZERO,
            max_walk_speed: 600.0, // Unreal units (cm/s)
            jump_z_velocity: 420.0,
            is_jumping: false,
            is_crouched: false,
            control_rotation: Vec3::ZERO,
        }
    }

    pub fn add_movement_input(&mut self, world_direction: Vec3, scale_value: f32) {
        if scale_value != 0.0 {
            let movement = world_direction.normalize_or_zero() * self.max_walk_speed * scale_value;
            self.velocity.x += movement.x;
            self.velocity.z += movement.z;
        }
    }

    pub fn jump(&mut self) {
        if self.movement_mode == EMovementMode::Walking {
            self.velocity.y = self.jump_z_velocity;
            self.movement_mode = EMovementMode::Falling;
            self.is_jumping = true;
        }
    }

    pub fn tick_movement(&mut self, delta_seconds: f32) {
        let mut pos = self.base_actor.get_actor_location();
        
        // Gravity
        if self.movement_mode == EMovementMode::Falling {
            self.velocity.y -= 980.0 * delta_seconds; // 9.8 m/s^2 in cm/s^2
        }

        pos += self.velocity * delta_seconds;

        // Simple floor collision (y = 0)
        if pos.y <= 0.0 {
            pos.y = 0.0;
            self.velocity.y = 0.0;
            self.movement_mode = EMovementMode::Walking;
            self.is_jumping = false;
        }

        // Friction / Damping
        self.velocity.x *= 0.9;
        self.velocity.z *= 0.9;

        self.base_actor.set_actor_location(pos);
    }
}

/// APlayerController - Player input & camera controller
pub struct APlayerController {
    pub player_id: u32,
    pub possessed_pawn_id: Option<u64>,
    pub input_axis_mappings: HashMap<String, f32>,
    pub input_action_mappings: HashMap<String, bool>,
    pub show_mouse_cursor: bool,
}

impl APlayerController {
    pub fn new(player_id: u32) -> Self {
        Self {
            player_id,
            possessed_pawn_id: None,
            input_axis_mappings: HashMap::new(),
            input_action_mappings: HashMap::new(),
            show_mouse_cursor: false,
        }
    }

    pub fn set_axis_value(&mut self, axis_name: &str, value: f32) {
        self.input_axis_mappings.insert(axis_name.to_string(), value);
    }

    pub fn get_axis_value(&self, axis_name: &str) -> f32 {
        self.input_axis_mappings.get(axis_name).copied().unwrap_or(0.0)
    }

    pub fn set_action_state(&mut self, action_name: &str, is_pressed: bool) {
        self.input_action_mappings.insert(action_name.to_string(), is_pressed);
    }

    pub fn is_action_pressed(&self, action_name: &str) -> bool {
        self.input_action_mappings.get(action_name).copied().unwrap_or(false)
    }
}

/// AGameModeBase - Global Game Rules & Spawner
pub struct AGameModeBase {
    pub default_pawn_class: String,
    pub player_controller_class: String,
    pub match_state: String,
    pub score: i32,
}

impl Default for AGameModeBase {
    fn default() -> Self {
        Self {
            default_pawn_class: "DefaultCharacter".to_string(),
            player_controller_class: "PlayerController".to_string(),
            match_state: "InProgress".to_string(),
            score: 0,
        }
    }
}

// ============================================================================
// 5. HIGH-PERFORMANCE BLUEPRINT VISUAL SCRIPTING BYTECODE VM
// ============================================================================

/// World Command Buffer for Blueprint VM operations
#[derive(Debug, Default)]
pub struct FWorldCommandBuffer {
    pub pending_spawns: Vec<AActor>,
    pub pending_destroys: Vec<u64>,
}

impl FWorldCommandBuffer {
    pub fn spawn_actor(&mut self, actor: AActor) {
        self.pending_spawns.push(actor);
    }

    pub fn destroy_actor(&mut self, id: u64) {
        self.pending_destroys.push(id);
    }
}

/// Blueprint Node Opcode for Zero-Cost VM Execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EBlueprintOpcode {
    // Event Entry Points
    EventBeginPlay,
    EventTick,
    EventCustom { name: String },

    // Pure Math Pin Instructions
    VectorAdd { in_a: usize, in_b: usize, out: usize },
    VectorMultiplyScale { in_vec: usize, scale: usize, out: usize },
    FloatLerp { in_a: usize, in_b: usize, alpha: usize, out: usize },

    // Actor Functions
    GetActorLocation { target_actor_id: usize, out_vec: usize },
    SetActorLocation { target_actor_id: usize, location_pin: usize },
    AddActorLocalOffset { target_actor_id: usize, offset_pin: usize },
    SpawnActor { class_name: String, out_actor_id: usize },
    DestroyActor { target_actor_id: usize },

    // Gameplay Utilities
    PrintString { message_pin: usize, duration: f32 },
    PlaySoundAtLocation { sound_name: String, location_pin: usize },

    // Flow Control
    Branch { condition_pin: usize, true_target_pc: usize, false_target_pc: usize },
    Goto { target_pc: usize },
    Return,
}

/// A compiled Blueprint Event Graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlueprintGraph {
    pub name: String,
    pub registers: Vec<UValue>,
    pub instructions: Vec<EBlueprintOpcode>,
    pub entry_points: HashMap<String, usize>, // Event Name -> Instruction Index
}

impl BlueprintGraph {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            registers: vec![UValue::None; 64],
            instructions: Vec::new(),
            entry_points: HashMap::new(),
        }
    }
}

/// Blueprint Virtual Machine Execution Engine
pub struct BlueprintVM;

impl BlueprintVM {
    /// Execute a Blueprint Event Graph on an Actor
    pub fn execute_event(
        graph: &BlueprintGraph,
        event_name: &str,
        actor: &mut AActor,
        cmd_buffer: &mut FWorldCommandBuffer,
    ) -> Result<()> {
        let entry_pc = match graph.entry_points.get(event_name) {
            Some(&pc) => pc,
            None => return Ok(()), // Event not implemented in this graph
        };

        let mut registers = graph.registers.clone();
        let mut pc = entry_pc;

        while pc < graph.instructions.len() {
            match &graph.instructions[pc] {
                EBlueprintOpcode::EventBeginPlay | EBlueprintOpcode::EventTick | EBlueprintOpcode::EventCustom { .. } => {
                    pc += 1;
                }

                EBlueprintOpcode::VectorAdd { in_a, in_b, out } => {
                    let va = registers[*in_a].as_vector();
                    let vb = registers[*in_b].as_vector();
                    registers[*out] = UValue::Vector(va + vb);
                    pc += 1;
                }

                EBlueprintOpcode::VectorMultiplyScale { in_vec, scale, out } => {
                    let v = registers[*in_vec].as_vector();
                    let s = registers[*scale].as_float();
                    registers[*out] = UValue::Vector(v * s);
                    pc += 1;
                }

                EBlueprintOpcode::FloatLerp { in_a, in_b, alpha, out } => {
                    let a = registers[*in_a].as_float();
                    let b = registers[*in_b].as_float();
                    let t = registers[*alpha].as_float();
                    registers[*out] = UValue::Float(a + (b - a) * t);
                    pc += 1;
                }

                EBlueprintOpcode::GetActorLocation { target_actor_id: _, out_vec } => {
                    let loc = actor.get_actor_location();
                    registers[*out_vec] = UValue::Vector(loc);
                    pc += 1;
                }

                EBlueprintOpcode::SetActorLocation { target_actor_id: _, location_pin } => {
                    let new_loc = registers[*location_pin].as_vector();
                    actor.set_actor_location(new_loc);
                    pc += 1;
                }

                EBlueprintOpcode::AddActorLocalOffset { target_actor_id: _, offset_pin } => {
                    let offset = registers[*offset_pin].as_vector();
                    actor.add_actor_local_offset(offset);
                    pc += 1;
                }

                EBlueprintOpcode::SpawnActor { class_name, out_actor_id } => {
                    let new_id = actor.id.wrapping_add(1000);
                    let new_actor = AActor::new(new_id, class_name);
                    cmd_buffer.spawn_actor(new_actor);
                    registers[*out_actor_id] = UValue::ObjectRef(new_id);
                    pc += 1;
                }

                EBlueprintOpcode::DestroyActor { target_actor_id } => {
                    let target_id = match registers[*target_actor_id] {
                        UValue::ObjectRef(id) => id,
                        _ => actor.id,
                    };
                    cmd_buffer.destroy_actor(target_id);
                    pc += 1;
                }

                EBlueprintOpcode::PrintString { message_pin, duration: _ } => {
                    let msg = match &registers[*message_pin] {
                        UValue::String(s) => s.clone(),
                        other => format!("{:?}", other),
                    };
                    log::info!("[Blueprint Log]: {}", msg);
                    pc += 1;
                }

                EBlueprintOpcode::PlaySoundAtLocation { sound_name, location_pin } => {
                    let loc = registers[*location_pin].as_vector();
                    log::info!("[Blueprint Audio]: Playing {} at location {:?}", sound_name, loc);
                    pc += 1;
                }

                EBlueprintOpcode::Branch { condition_pin, true_target_pc, false_target_pc } => {
                    let cond = registers[*condition_pin].as_bool();
                    if cond {
                        pc = *true_target_pc;
                    } else {
                        pc = *false_target_pc;
                    }
                }

                EBlueprintOpcode::Goto { target_pc } => {
                    pc = *target_pc;
                }

                EBlueprintOpcode::Return => {
                    break;
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// 6. UWORLD & LEVEL MANAGER
// ============================================================================

/// Collision Raycast Result matching Unreal FHitResult
#[derive(Debug, Clone)]
pub struct FHitResult {
    pub b_blocking_hit: bool,
    pub distance: f32,
    pub location: Vec3,
    pub normal: Vec3,
    pub actor_id: Option<u64>,
    pub component_name: String,
}

/// UWorld - Global Unreal Engine Level Container
pub struct UWorld {
    pub name: String,
    pub game_mode: AGameModeBase,
    pub actors: HashMap<u64, AActor>,
    pub characters: HashMap<u64, ACharacter>,
    pub player_controllers: HashMap<u32, APlayerController>,
    pub real_time_seconds: f32,
    pub delta_seconds: f32,
    pub time_dilation: f32,
    next_id: u64,
}

impl UWorld {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            game_mode: AGameModeBase::default(),
            actors: HashMap::new(),
            characters: HashMap::new(),
            player_controllers: HashMap::new(),
            real_time_seconds: 0.0,
            delta_seconds: 0.0166,
            time_dilation: 1.0,
            next_id: 100,
        }
    }

    pub fn next_actor_id(&mut self) -> u64 {
        self.next_id += 1;
        self.next_id
    }

    pub fn spawn_actor_direct(&mut self, actor: AActor) {
        self.actors.insert(actor.id, actor);
    }

    pub fn spawn_character(&mut self, name: &str) -> u64 {
        let id = self.next_actor_id();
        let character = ACharacter::new(id, name);
        self.characters.insert(id, character);
        id
    }

    pub fn destroy_actor(&mut self, id: u64) {
        if let Some(actor) = self.actors.get_mut(&id) {
            actor.is_pending_kill = true;
        }
        if let Some(char_actor) = self.characters.get_mut(&id) {
            char_actor.base_actor.is_pending_kill = true;
        }
    }

    /// Perform a line trace / raycast matching Unreal LineTraceSingleByChannel
    pub fn line_trace_single_by_channel(
        &self,
        start: Vec3,
        end: Vec3,
        _trace_channel: u32,
    ) -> FHitResult {
        let dir = (end - start).normalize_or_zero();
        let max_dist = start.distance(end);

        // Test ray against all actors with root component transforms
        let mut closest_dist = max_dist;
        let mut hit_actor = None;

        for (id, actor) in &self.actors {
            let loc = actor.get_actor_location();
            let dist = start.distance(loc);
            if dist < closest_dist && dist < 100.0 { // Simple sphere intersection demo
                closest_dist = dist;
                hit_actor = Some(*id);
            }
        }

        if let Some(actor_id) = hit_actor {
            FHitResult {
                b_blocking_hit: true,
                distance: closest_dist,
                location: start + dir * closest_dist,
                normal: -dir,
                actor_id: Some(actor_id),
                component_name: "StaticMeshComponent".to_string(),
            }
        } else {
            FHitResult {
                b_blocking_hit: false,
                distance: max_dist,
                location: end,
                normal: Vec3::ZERO,
                actor_id: None,
                component_name: "".to_string(),
            }
        }
    }

    /// Main Unreal Tick Loop
    pub fn tick(&mut self, delta_seconds: f32) {
        let effective_delta = delta_seconds * self.time_dilation;
        self.delta_seconds = effective_delta;
        self.real_time_seconds += delta_seconds;

        // 1. Tick Player Movement & Pawns
        for character in self.characters.values_mut() {
            character.tick_movement(effective_delta);
        }

        // 2. Execute Blueprints on Actors
        let mut cmd_buffer = FWorldCommandBuffer::default();
        let mut pending_graph_execs = Vec::new();

        for actor in self.actors.values() {
            if actor.can_ever_tick && !actor.is_pending_kill {
                if let Some(ref bg) = actor.blueprint_graph {
                    pending_graph_execs.push((actor.id, bg.clone()));
                }
            }
        }

        for (actor_id, bg) in pending_graph_execs {
            if let Some(actor) = self.actors.get_mut(&actor_id) {
                let _ = BlueprintVM::execute_event(&bg, "EventTick", actor, &mut cmd_buffer);
            }
        }

        // Apply command buffer
        for new_actor in cmd_buffer.pending_spawns {
            self.actors.insert(new_actor.id, new_actor);
        }
        for destroy_id in cmd_buffer.pending_destroys {
            self.destroy_actor(destroy_id);
        }

        // 3. Garbage Collection / Prune Pending Kill Actors
        self.actors.retain(|_, a| !a.is_pending_kill);
        self.characters.retain(|_, c| !c.base_actor.is_pending_kill);
    }
}

// ============================================================================
// 7. UNIT TESTS FOR UNREAL FRAMEWORK & BLUEPRINT VM
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_actor_components_and_transform() {
        let mut actor = AActor::new(1, "TestActor");
        actor.set_actor_location(Vec3::new(10.0, 20.0, 30.0));
        assert_eq!(actor.get_actor_location(), Vec3::new(10.0, 20.0, 30.0));

        actor.add_actor_local_offset(Vec3::new(5.0, 0.0, 0.0));
        assert_eq!(actor.get_actor_location(), Vec3::new(15.0, 20.0, 30.0));
    }

    #[test]
    fn test_character_movement_and_gravity() {
        let mut world = UWorld::new("TestLevel");
        let char_id = world.spawn_character("PlayerCharacter");

        let character = world.characters.get_mut(&char_id).unwrap();
        assert_eq!(character.movement_mode, EMovementMode::Walking);

        character.jump();
        assert_eq!(character.movement_mode, EMovementMode::Falling);
        assert_eq!(character.velocity.y, 420.0);

        // Tick 0.1s
        world.tick(0.1);
        let char_after = world.characters.get(&char_id).unwrap();
        assert!(char_after.base_actor.get_actor_location().y > 0.0);
    }

    #[test]
    fn test_blueprint_vm_math_and_flow() {
        let mut world = UWorld::new("BPLevel");
        let mut actor = AActor::new(1, "BPActor");
        actor.set_actor_location(Vec3::new(0.0, 0.0, 0.0));

        let mut graph = BlueprintGraph::new("MoveActorGraph");
        // Register 0 = (10, 0, 0)
        graph.registers[0] = UValue::Vector(Vec3::new(10.0, 0.0, 0.0));
        
        // Instruction 0: AddActorLocalOffset using register 0
        graph.instructions.push(EBlueprintOpcode::AddActorLocalOffset { target_actor_id: 1, offset_pin: 0 });
        graph.instructions.push(EBlueprintOpcode::Return);
        graph.entry_points.insert("EventTick".to_string(), 0);

        actor.blueprint_graph = Some(Arc::new(graph));
        world.spawn_actor_direct(actor);

        world.tick(0.016);

        let ticked_actor = world.actors.get(&1).unwrap();
        assert_eq!(ticked_actor.get_actor_location(), Vec3::new(10.0, 0.0, 0.0));
    }
}
