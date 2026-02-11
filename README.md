# Slop-Engine

# To make this functional:

Install ```wasm-pack```: ```cargo install wasm-pack```

Run: ```wasm-pack build --target web --release```

Push the files and the /pkg folder to your repo.



# Slop Engine

**Slop Engine** is a high-performance, local-first game engine built for the 2026 web. It leverages **Rust**, **WebAssembly**, and **WebGPU** to deliver native-grade performance directly in the browser.

## Features
* **Zero-Latency Core:** Rust/Wasm architecture with no Garbage Collection (GC) pauses.
* **GPU-Driven:** Leveraging WebGPU for massive-scale rendering.
* **Offline-First:** Fully functional without an internet connection via Service Workers and OPFS.
* **Lightweight:** Minimal binary footprint optimized for mobile and desktop browsers.

---

## Toolchain Setup

To build and contribute to Slop Engine, you need the following installed:

1. **Rust Toolchain**: [Install Rust](https://rustup.rs/)
2. **wasm-pack**: The bridge for Rust-to-JS compilation.
   ```bash cargo install wasm-pack```

### Please note that this will not work, because it is pure AI slop
