# 📖 Slop Engine — Complete Developer Usage Guide

Welcome to **Slop Engine**, a hyper-optimized, modern real-time WebGPU game engine built in Rust. This guide provides comprehensive, step-by-step instructions on how to build games using Slop Engine—from basic window setup to the **Unreal Engine-Style Gameplay Framework**, **Blueprint VM**, **TDSP Prediction**, **CDR Save System**, and **WebGPU Predictive Rendering**.

---

## 📋 Table of Contents

1. [Installation & Cargo Setup](#1-installation--cargo-setup)
2. [Quick Start Engine Loop](#2-quick-start-engine-loop)
3. [Unreal Engine Framework (`unreal_framework`)](#3-unreal-engine-framework-unreal_framework)
   - [World & GameMode Setup](#world--gamemode-setup)
   - [Spawning & Controlling Characters](#spawning--controlling-characters)
   - [Creating Actors & Components](#creating-actors--components)
   - [Raycasting & Line Tracing](#raycasting--line-tracing)
4. [Blueprint Visual Scripting System](#4-blueprint-visual-scripting-system)
   - [Understanding Blueprint Graphs](#understanding-blueprint-graphs)
   - [Creating & Attaching a Blueprint Graph](#creating--attaching-a-blueprint-graph)
5. [Core Systems Guide](#5-core-systems-guide)
   - [TDSP Engine (Intent Prediction)](#tdsp-engine-intent-prediction)
   - [Predictive Rendering](#predictive-rendering)
   - [CDR Save System (Causal Divergence Recording)](#cdr-save-system-causal-divergence-recording)
   - [Network System](#network-system)
   - [Resource & Memory Management](#resource--memory-management)
6. [Building for Web (WASM) & Native](#6-building-for-web-wasm--native)

---

## 1. Installation & Cargo Setup

Add `slop_engine` to your project's `Cargo.toml`:

```toml
[dependencies]
slop_engine = { git = "https://github.com/gugu8intel-i9/Slop-Engine.git", features = ["unreal_framework"] }
glam = "0.24"
bytemuck = "1.16"
winit = "0.30"
log = "0.4"
```

To enable the optional Unreal Engine Actor & Blueprint framework, ensure the `unreal_framework` feature flag is active.

---

## 2. Quick Start Engine Loop

Here is a minimal native application initializing `Slop Engine`:

```rust
use slop_engine::{EngineConfig, EngineState};
use std::sync::Arc;

fn main() {
    // 1. Create engine configuration
    let config = EngineConfig::default();

    // 2. Initialize Engine State
    let mut engine = EngineState::new(config);

    // 3. Main simulation tick loop
    let delta_time = 0.0166; // 60 FPS tick
    let current_time = 0.0;

    for frame in 0..600 {
        let time = current_time + (frame as f64 * delta_time);
        
        // Update simulation
        engine.update(time);
        
        println!("Frame {}: FPS target = 60", frame);
    }
}
```

---

## 3. Unreal Engine Framework (`unreal_framework`)

Slop Engine provides an optional, high-performance gameplay paradigm modeled directly after **Unreal Engine 5**. It includes native `AActor`, `ACharacter`, `UActorComponent`, `APlayerController`, `AGameModeBase`, and `UWorld`.

### World & GameMode Setup

The `UWorld` struct acts as the primary level container:

```rust
use slop_engine::unreal_framework::{UWorld, AGameModeBase};

fn init_level() -> UWorld {
    let mut world = UWorld::new("Level_01");

    // Configure GameMode rules
    world.game_mode = AGameModeBase {
        default_pawn_class: "HeroCharacter".to_string(),
        player_controller_class: "DefaultPlayerController".to_string(),
        match_state: "InProgress".to_string(),
        score: 0,
    };

    world
}
```

### Spawning & Controlling Characters

Use `ACharacter` for pawns requiring walking physics, jumping, and camera controls:

```rust
use slop_engine::unreal_framework::{UWorld, ACharacter, APlayerController, EMovementMode};
use glam::Vec3;

fn setup_player(world: &mut UWorld) {
    // 1. Spawn player character
    let char_id = world.spawn_character("PlayerHero");

    // 2. Spawn & setup player controller
    let mut controller = APlayerController::new(1);
    controller.possessed_pawn_id = Some(char_id);
    world.player_controllers.insert(1, controller);

    // 3. Apply movement input (e.g. Move Forward)
    if let Some(character) = world.characters.get_mut(&char_id) {
        // Add movement input in world direction (Forward = X)
        character.add_movement_input(Vec3::X, 1.0);
        
        // Trigger jump physics
        character.jump();
    }
}
```

### Creating Actors & Components

You can assemble custom Actors with attached components:

```rust
use slop_engine::unreal_framework::{AActor, UActorComponent, EComponentType, FTransform};
use glam::Vec3;

fn spawn_light_actor(world: &mut UWorld) -> u64 {
    let actor_id = world.next_actor_id();
    let mut actor = AActor::new(actor_id, "StreetLamp");

    // Root Scene Component
    let root = UActorComponent::new(actor_id * 10 + 1, "RootComponent", EComponentType::Scene);
    
    // Point Light Component
    let light = UActorComponent::new(actor_id * 10 + 2, "LampLight", EComponentType::PointLight {
        intensity: 5000.0,
        color: Vec3::new(1.0, 0.9, 0.7), // Warm light
        attenuation_radius: 1200.0,
    });

    actor.add_component(root);
    actor.add_component(light);

    actor.set_actor_location(Vec3::new(100.0, 250.0, 0.0));

    world.spawn_actor_direct(actor);
    actor_id
}
```

### Raycasting & Line Tracing

Perform 3D collision line traces matching Unreal's `LineTraceSingleByChannel`:

```rust
use slop_engine::unreal_framework::UWorld;
use glam::Vec3;

fn perform_weapon_trace(world: &UWorld, camera_pos: Vec3, camera_forward: Vec3) {
    let trace_start = camera_pos;
    let trace_end = camera_pos + camera_forward * 10000.0; // 100 meters
    let channel = 1; // Visibility channel

    let hit_result = world.line_trace_single_by_channel(trace_start, trace_end, channel);

    if hit_result.b_blocking_hit {
        println!(
            "Hit Actor ID: {:?} at location {:?} (Distance: {:.2} cm)",
            hit_result.actor_id, hit_result.location, hit_result.distance
        );
    } else {
        println!("Line trace missed.");
    }
}
```

---

## 4. Blueprint Visual Scripting System

Slop Engine features a **zero-allocation Bytecode VM** that executes Unreal-style Blueprint Event Graphs on Actors without runtime allocations.

### Understanding Blueprint Graphs

A `BlueprintGraph` contains:
- **Registers (`Vec<UValue>`)**: Local pin storage for vectors, floats, booleans, strings, and object references.
- **Opcodes (`EBlueprintOpcode`)**: High-performance instructions representing Blueprint nodes.
- **Entry Points**: Event hooks (`EventBeginPlay`, `EventTick`, `EventCustom`).

### Creating & Attaching a Blueprint Graph

Here is how to create a Blueprint that moves an Actor continuously on `EventTick`:

```rust
use slop_engine::unreal_framework::{
    AActor, BlueprintGraph, EBlueprintOpcode, UValue, UWorld
};
use glam::Vec3;
use std::sync::Arc;

fn attach_movement_blueprint(actor: &mut AActor) {
    let mut graph = BlueprintGraph::new("ContinuousMoveBP");

    // Register 0: Local movement velocity vector (10 cm/frame forward)
    graph.registers[0] = UValue::Vector(Vec3::new(10.0, 0.0, 0.0));

    // Instruction 0: AddActorLocalOffset using Register 0
    graph.instructions.push(EBlueprintOpcode::AddActorLocalOffset {
        target_actor_id: actor.id as usize,
        offset_pin: 0,
    });

    // Instruction 1: Return
    graph.instructions.push(EBlueprintOpcode::Return);

    // Bind EventTick to Instruction 0
    graph.entry_points.insert("EventTick".to_string(), 0);

    // Attach Blueprint to Actor
    actor.blueprint_graph = Some(Arc::new(graph));
}
```

When `world.tick(delta_seconds)` is called, the `BlueprintVM` automatically executes all attached Blueprints.

---

## 5. Core Systems Guide

### TDSP Engine (Intent Prediction)

The **TDSP System** decouples hardware polling from frame rendering and uses intent prediction to eliminate perceived input latency:

```rust
use slop_engine::tdsp_engine::{TDSPEngine, TDSPConfig, InputEvent, InputState};
use glam::Vec2;

fn process_tdsp_input(tdsp: &mut TDSPEngine) {
    let event = InputEvent {
        timestamp_ns: 1000000,
        scancode: 87, // 'W' key
        state: InputState::Pressed,
        velocity: Vec2::new(0.0, 1.0),
        acceleration: Vec2::ZERO,
    };

    tdsp.update(event);

    // Fetch intent prediction stats
    let stats = tdsp.get_stats();
    println!("Intent Confidence: {:.1}%", stats.intent_confidence * 100.0);
}
```

### Predictive Rendering

Predictive rendering updates only hot screen micro-tiles (16x16 pixels) to reduce GPU load by 30-70%:

```rust
use slop_engine::predictive_renderer::{PredictiveRenderer, PredictiveRenderConfig};
use wgpu::Device;

fn setup_predictive_renderer(device: &Device) -> PredictiveRenderer {
    let config = PredictiveRenderConfig {
        enabled: true,
        tile_size: 16,
        error_threshold: 0.05,
        max_accumulated_error: 0.2,
        history_capacity: 4,
    };

    PredictiveRenderer::new(device, config, 1920, 1080)
}
```

### CDR Save System (Causal Divergence Recording)

CDR stores only causal divergence events rather than full world state snapshots, achieving **400x disk savings**:

```rust
use slop_engine::causal_save::{CausalSaveFile, WorldSeed, PlayerSeed};

fn create_save_file() -> CausalSaveFile {
    CausalSaveFile::new(
        WorldSeed { seed: 1337 },
        PlayerSeed { class_id: 1, customization_flags: vec![2, 5, 8] },
    )
}
```

### Network System

The `NetworkSystem` handles client prediction, server reconciliation, and interest management:

```rust
use slop_engine::network::{NetworkSystem, NetworkConfig, NetworkRole};

fn setup_networking() -> NetworkSystem {
    let config = NetworkConfig {
        enabled: true,
        role: NetworkRole::Client,
        tick_rate: 60,
    };

    NetworkSystem::new(config)
}
```

### Resource & Memory Management

Slop Engine uses a **W-TinyLFU cache** and handle pools to guarantee zero heap allocation on hot execution paths:

```rust
use slop_engine::resource_manager::{ResourceManager, ResourceConfig};
use std::sync::Arc;
use wgpu::{Device, Queue};

fn init_resources(device: Arc<Device>, queue: Arc<Queue>) -> ResourceManager {
    let config = ResourceConfig {
        max_texture_bytes: 2 * 1024 * 1024 * 1024, // 2GB VRAM budget
        staging_buffer_size: 64 * 1024 * 1024,
        max_bind_group_cache: 1024,
        max_texture_handles: 4096,
        max_mesh_handles: 2048,
        max_material_handles: 2048,
        deduplication_enabled: true,
    };

    ResourceManager::new(device, queue, config)
}
```

---

## 6. Building for Web (WASM) & Native

### Building Native (Linux, Windows, macOS)

```bash
cargo build --release --features unreal_framework
```

### Building for WebGPU (WASM)

```bash
wasm-pack build --target web --release --features unreal_framework
```

---

## 💡 Summary

Slop Engine combines **ultra-low-latency WebGPU graphics**, **intent-predictive simulation**, and **Unreal Engine-style gameplay architecture**. For further questions or contributions, check out the [GitHub Repository](https://github.com/gugu8intel-i9/Slop-Engine).
