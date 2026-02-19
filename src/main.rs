// src/main.rs

#![cfg(not(target_arch = "wasm32"))]

use std::panic;
use log::LevelFilter;

fn main() {
    // Native-only entry point
    // WASM uses #[wasm_bindgen(start)] in lib.rs

    setup_runtime();

    // Call your engine
    slop_engine::run_native();
}

fn setup_runtime() {
    // Better panic output (native)
    panic::set_hook(Box::new(|info| {
        eprintln!("\n=== ENGINE PANIC ===");
        eprintln!("{info}");
    }));

    // Optimized logging config
    env_logger::Builder::new()
        .filter_level(
            if cfg!(debug_assertions) {
                LevelFilter::Debug
            } else {
                LevelFilter::Warn
            }
        )
        .format_timestamp_millis()
        .init();

    // CPU performance hint (optional but safe)
    #[cfg(feature = "high_priority")]
    {
        let _ = thread_priority::set_current_thread_priority(
            thread_priority::ThreadPriority::Max,
        );
    }
}
