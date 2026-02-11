// main.js - The Slop Engine Bridge
import init, { run } from './pkg/slop_engine.js';

async function bootstrap() {
    try {
        // 1. Initialize the Wasm module
        // This loads the .wasm file and prepares the memory space
        await init();

        // 2. Performance Check: Verify WebGPU support before launching
        if (!navigator.gpu) {
            console.error("Slop Engine Error: WebGPU not supported in this browser.");
            document.body.innerHTML = "<h1>Please use a browser that supports WebGPU.</h1>";
            return;
        }

        console.log("Slop Engine: Wasm Loaded. Launching Core...");

        // 3. Hand over control to the Rust 'run' function in lib.rs
        run();

    } catch (error) {
        console.error("Slop Engine failed to initialize:", error);
    }
}

// Start the engine
bootstrap();