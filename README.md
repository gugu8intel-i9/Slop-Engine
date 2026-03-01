# **Slop Engine**  
### *A hyper‑optimized, modern real‑time rendering engine built for performance, clarity, and raw power.*

Slop Engine is a fully‑modular, high‑performance real‑time engine designed around **modern GPU pipelines**, **zero‑waste memory usage**, and **hot‑reloadable systems**. Every subsystem is engineered for **predictable performance**, **low latency**, and **maximum feature density** without sacrificing clarity or extensibility.

This engine is built for developers who want **full control**, **cutting‑edge rendering**, and **no compromises**.

---

## **✨ Core Features**

### **⚡ Ultra‑Fast Renderer**
- Modern **bindless‑leaning architecture**
- GPU‑driven rendering paths
- High‑performance **Mesh + Material** system
- 2D & 3D shader support with unified pipeline
- Full PBR with advanced BRDF, clearcoat, sheen, anisotropy, transmission, subsurface

---

### **🌑 Advanced Lighting**
- Directional, point, and spot lights
- Physically‑based shading
- Shadow atlas with:
  - Cascaded Shadow Maps (CSM)
  - PCF3/PCF5
  - VSM / EVSM
  - Stable texel snapping
  - Tile‑based allocation
- Contact shadows hook
- Light LOD system

---

### **🌌 Sky & IBL**
- HDR environment loading (EXR/HDR/PNG)
- GPU‑accelerated PMREM generation
- Prefiltered specular cubemap (GGX)
- Irradiance convolution
- BRDF LUT generation
- Procedural sky with atmospheric scattering

---

### **🎨 Materials**
- Full PBR material system
- ORM packing (Occlusion/Roughness/Metallic)
- Normal, emissive, clearcoat, sheen, transmission, subsurface maps
- Bitflag‑driven feature toggles (no wasted texture fetches)
- Bind group caching
- Hot‑reloadable shaders

---

### **🧱 Scene Graph / ECS**
- High‑performance hierarchical scene graph
- Optional ECS mode for large‑scale worlds
- Transform propagation optimized for cache locality
- Culling hooks (frustum, occlusion, LOD)

---

### **📦 Resource Manager**
A hyper‑optimized resource system designed for:
- Zero‑copy GPU uploads
- Streaming‑friendly asset loading
- Deduplication of textures, meshes, shaders
- Reference‑counted GPU resources
- Extremely low RAM footprint
- Async loading support

---

### **🎮 Input System**
- High‑performance, event‑driven input
- Keyboard, mouse, controller abstraction
- Input mapping layer
- Low‑latency polling mode

---

### **📷 Camera Controller**
- Free‑fly, orbit, follow, cinematic spline, orthographic
- Collision‑aware camera movement
- Smooth damping, acceleration, input smoothing
- GPU‑ready camera uniform
- Frustum extraction for culling

---

### **🌫️ Post‑Processing**
- HDR pipeline
- ACES/Filmic tonemapping
- Bloom (multi‑scale)
- TAA with jitter + history buffer
- SSAO (HBAO‑style)
- SSR (hierarchical raymarch)
- Motion blur
- Depth of Field (CoC)
- Color grading (LUT)
- Lens effects (vignette, chromatic aberration, grain)
- Modular pass chain

---

### **🔥 Shader Hot Reload**
- Watches WGSL files in real time
- Debounced rebuilds
- Safe pipeline recreation
- Include support (`#include "file.wgsl"`)
- Error overlay support
- Zero‑downtime pipeline swapping

---
### ⚖️ Anticheat (Rust) — Behaviour, Network, Process & Memory Verification

A Rust-based anti-cheat module designed to track player state, validate network packets, analyse behaviour, and (optionally) perform lightweight process and memory integrity checks on Windows.

> **Note:** This code is a *framework/demo-style implementation* and includes placeholder logic (e.g. pattern scanning). It is not production-ready without hardening, correct time handling, proper process association, and careful security review.

---

## Contents

- [Features](#features)
- [How it works](#how-it-works)
  - [Player lifecycle](#player-lifecycle)
  - [State verification](#state-verification)
  - [Network packet verification (HMAC)](#network-packet-verification-hmac)
  - [Behaviour analysis](#behaviour-analysis)
  - [Memory signature & anomaly scan (Windows)](#memory-signature--anomaly-scan-windows)
  - [Process scanning](#process-scanning)
  - [Violations & banning](#violations--banning)
  - [Events](#events)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Initialising](#initialising)
  - [Registering a player](#registering-a-player)
  - [Updating state](#updating-state)
  - [Verifying packets](#verifying-packets)
  - [Listening for events](#listening-for-events)
  - [Adding memory patterns and known-good processes](#adding-memory-patterns-and-known-good-processes)
- [Testing](#testing)
- [Platform notes](#platform-notes)
- [Security & implementation notes](#security--implementation-notes)
- [Licence](#licence)

---

## Features

- **Player registration and tracking**
  - Stores per-player `PlayerState`
  - Maintains violation history and a behaviour score

- **Movement & action validation**
  - Speed hack detection (distance over time vs velocity tolerance)
  - Teleport detection (large displacement in tiny time window)
  - No-clip heuristic (example: below ground)

- **Action checks**
  - Rapid fire detection
  - Recoil manipulation heuristic

- **Checkpoint verification**
  - Enforces sequential checkpoints
  - Detects checkpoints reached too quickly

- **Network verification**
  - HMAC-SHA256 packet signing and verification
  - Packet size enforcement

- **Behaviour analysis**
  - Collects movement and shot patterns
  - Detects suspiciously precise aiming and impossible average speeds

- **Process scanning**
  - Computes a hash of running processes
  - Flags suspicious/known-cheat process names

- **Memory integrity (Windows)**
  - Generates a simple “memory signature”
  - Optional anomaly scan using `ReadProcessMemory` (pattern matching is stubbed)

- **Event-driven design**
  - Emits `AnticheatEvent` via Tokio MPSC channel

---

## How it works

### Player lifecycle

- `register_player(player_id, initial_state)`
  - Inserts player into internal store
  - Initialises `behaviour_score`
  - Computes:
    - `memory_signature` (if enabled)
    - `process_hash` (if enabled)
  - Emits `PlayerConnected`

- `unregister_player(player_id)`
  - Removes player
  - Emits `PlayerDisconnected`

### State verification

`update_player_state(player_id, new_state)` performs checks via `verify_state_update`:

- `check_movement`
  - Speed tolerance based on old velocity plus jitter buffer
  - Teleport check (distance threshold and time window)
  - No-clip heuristic

- `check_actions`
  - Rapid fire check
  - Recoil manipulation check

- `verify_checkpoints`
  - Validates checkpoint timing and order
  - Emits `CheckpointPassed` for legitimate transitions

### Network packet verification (HMAC)

`verify_packet(player_id, packet, signature)`:

- Validates `signature` using **HMAC-SHA256** and `config.hmac_key`
- Rejects packets larger than `MAX_PACKET_SIZE`

### Behaviour analysis

`check_behavior(player_id)` maintains a `BehaviorProfile`:

- `movement_patterns`: sampled velocities
- `shot_patterns`: simplified rotation deltas on firing

`analyze_behavior` currently flags:
- **Aimbot heuristic**: extremely small average rotation change
- **Impossible movement**: high average speed across samples

### Memory signature & anomaly scan (Windows)

- `generate_memory_signature`
  - Hashes process metadata (PID, name, memory usage) with SHA-256

- `scan_memory_for_anomalies`
  - Uses `OpenProcess` + `ReadProcessMemory`
  - Compares bytes read against stored patterns
  - Emits `MemoryScanResult`
  - Records `MemoryTampering` if anomalies exist

> The pattern search function `find_pattern_in_process` is currently a stub returning `None`.

### Process scanning

- `generate_process_hash`
  - Hashes the list of running processes (PID + name) to detect changes

- `scan_for_unauthorised_processes`
  - Looks for process names containing suspicious tooling keywords (e.g. debuggers, cheat tools)

### Violations & banning

`record_violation`:
- Reduces the player’s `behaviour_score` by `severity`
- Stores `(ViolationType, timestamp)`
- Emits `ViolationDetected`
- Prints a message when violations exceed `max_violations_before_ban`


### Events

Events are represented by:

- `PlayerConnected`
- `PlayerDisconnected`
- `ViolationDetected`
- `HeartbeatReceived`
- `CheckpointPassed`
- `MemoryScanResult`

---

## Configuration

The module is controlled via `AnticheatConfig`:

- `enable_memory_scans`
- `enable_process_scans`
- `enable_behavior_analysis`
- `enable_network_verification`
- `enable_checkpoints`
- `max_violations_before_ban`
- `violation_cooldown`
- `memory_scan_interval`
- `process_scan_interval`
- `behavior_analysis_interval`
- `hmac_key`

A default configuration is provided via `Default`.

---

## Usage

### Initialising

```rust
let config = AnticheatConfig::default();
let anticheat = Anticheat::new(config);
---

## **📁 Project Structure**

```
/src
  camera_controller.rs
  input.rs
  material.rs
  mesh.rs
  pbr_materials.rs
  post_processing.rs
  resource_manager.rs
  scene_graph.rs
  shader_hot_reload.rs
  shadows.rs
  skybox.rs
  renderer.rs
  engine.rs

/shaders
  pbr_full.wgsl
  shadow_depth.wgsl
  shadow_sampling.wgsl
  skybox_vert.wgsl
  skybox_frag.wgsl
  prefilter_env.wgsl
  irradiance_conv.wgsl
  brdf_lut.wgsl
  post_blit.wgsl
  bloom_*.wgsl
  taa.wgsl
  ssao.wgsl
  ssr.wgsl
  dof.wgsl
```

---

## **🚀 Getting Started**

### **1. Clone**
```
git clone https://github.com/yourname/slop-engine
cd slop-engine
```

### **2. Build**
```
cargo run --release
```

### **3. Edit Shaders Live**
Just open any `.wgsl` file — the engine will hot‑reload it instantly.

---

## **🧪 Roadmap**
- GPU‑driven culling (meshlet / cluster)
- Bindless textures (when stable in wgpu)
- GPU particle system
- Virtual texturing
- Ray tracing backend (DXR / Vulkan RT)
- Editor UI (ImGui or custom)

---

## **📜 License**
GNU Affero General Public License

---

## **🤝 Contributing**
Contributions are welcome — especially around:
- New rendering passes
- Optimization
- Shader improvements
- Documentation

---

## **💬 Final Notes**
Slop Engine is built for developers who want **full control**, **maximum performance**, and **modern rendering techniques** without the bloat of traditional engines. Every subsystem is slopped for clarity, speed, and extensibility.

   
   * If you wish to contribute, please be a sponsor, because these API costs are getting expensive
   
   * If any issue occurs, please put an issue out
---
> Please note that this will not work, because it is pure AI slop
