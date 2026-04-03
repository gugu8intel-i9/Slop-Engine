# Slop Engine  


https://github.com/user-attachments/assets/fb64874c-fde1-4500-a9d6-959070eba808

### The specs are

 * ```Intel(R) Core(TM) i5-2400S CPU @ 2.50GHz```

* ```AMD Radeon HD 6750M```

* ```Size: 8 GB Type: DDR3 Speed: 1333 MHz```

A hyper‑optimized, modern real‑time rendering engine built for performance, clarity, and raw power.

Slop Engine is a fully‑modular, high‑performance real‑time engine designed around modern GPU pipelines, zero‑waste memory usage, and hot‑reloadable systems. Every subsystem is engineered for predictable performance, low latency, and maximum feature density without sacrificing clarity or extensibility.

This engine is built for developers who want full control, cutting‑edge rendering, and no compromises.

✨ Core Features

⚡ Ultra‑Fast Renderer
- Modern bindless‑leaning architecture
- GPU‑driven rendering paths
- High‑performance Mesh + Material system
- 2D & 3D shader support with unified pipeline
- Full PBR with advanced BRDF, clearcoat, sheen, anisotropy, transmission, subsurface

🌑 Advanced Lighting
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

🌌 Sky & IBL
- HDR environment loading (EXR/HDR/PNG)
- GPU‑accelerated PMREM generation
- Prefiltered specular cubemap (GGX)
- Irradiance convolution
- BRDF LUT generation
- Procedural sky with atmospheric scattering

🎨 Materials
- Full PBR material system
- ORM packing (Occlusion/Roughness/Metallic)
- Normal, emissive, clearcoat, sheen, transmission, subsurface maps
- Bitflag‑driven feature toggles
- Bind group caching
- Hot‑reloadable shaders

🧱 Scene Graph / ECS
- High‑performance hierarchical scene graph
- Optional ECS mode for large‑scale worlds
- Transform propagation optimized for cache locality
- Culling hooks (frustum, occlusion, LOD)

📦 Resource Manager
- Zero‑copy GPU uploads
- Streaming‑friendly asset loading
- Deduplication of textures, meshes, shaders
- Reference‑counted GPU resources
- Extremely low RAM footprint
- Async loading support

🎮 Input System
- High‑performance, event‑driven input
- Keyboard, mouse, controller abstraction
- Input mapping layer
- Low‑latency polling mode

📷 Camera Controller
- Free‑fly, orbit, follow, cinematic spline, orthographic
- Collision‑aware camera movement
- Smooth damping, acceleration, input smoothing
- GPU‑ready camera uniform
- Frustum extraction for culling

🌫️ Post‑Processing
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

🔥 Shader Hot Reload
- Watches WGSL files in real time
- Debounced rebuilds
- Safe pipeline recreation
- Include support (#include "file.wgsl")
- Error overlay support
- Zero‑downtime pipeline swapping

🚀 Getting Started

1. Clone  
``` git clone https://github.com/yourname/slop-engine ```
``` cd slop-engine ```

2. Build  
``` cargo run --release ```

3. Edit Shaders Live  
Open any .wgsl file — the engine will hot‑reload it instantly.

🧪 Roadmap
- GPU‑driven culling (meshlet / cluster)
- Bindless textures (when stable in wgpu)
- GPU particle system
- Virtual texturing
- Ray tracing backend (DXR / Vulkan RT)
- Editor UI (ImGui or custom)

📜 License (AGPLv3)

``` Slop Engine is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License version 3 as published by the Free Software Foundation. Slop Engine is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. You should have received a copy of the GNU Affero General Public License along with this program. If not, see: https://www.gnu.org/licenses/ ```

🤝 Contributing
Contributions are welcome — especially around:
- New rendering passes
- Optimization
- Shader improvements
- Documentation

💬 Final Notes
Slop Engine is built for developers who want full control, maximum performance, and modern rendering techniques without the bloat of traditional engines. Every subsystem is slopped for clarity, speed, and extensibility.

* If you wish to contribute, please be a sponsor, because these API costs are getting expensive  
* If any issue occurs, please put an issue out  
* Please note that this will not work, because it is pure AI slop
