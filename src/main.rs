#![cfg(not(target_arch = "wasm32"))]

use std::fs::File;
use std::io::Write;
use std::panic;
use std::backtrace::Backtrace;
use log::{LevelFilter, info, error};

// 1. HIGH PERFORMANCE ALLOCATOR
// mimalloc is significantly faster than the default system allocator 
// for multi-threaded game engines (ECS, Asset streaming, etc).
// Add `mimalloc = { version = "0.1", optional = true }` to Cargo.toml
#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() {
    // 2. OS & CPU LEVEL OPTIMIZATIONS
    setup_os_optimizations();
    setup_cpu_math();

    // 3. ROBUST DIAGNOSTICS & CRASH HANDLING
    setup_diagnostics();

    // 4. BACKGROUND ASYNC RUNTIME (Optional but recommended)
    // winit requires running on the main thread, so we spawn Tokio in the background
    // for handling async tasks like multiplayer networking or heavy file I/O.
    let _bg_runtime = setup_async_runtime();

    info!("Starting SlopEngine (Native)...");

    // 5. START ENGINE
    // run_native() will likely use pollster::block_on() or block the main thread with winit.
    if let Err(e) = std::panic::catch_unwind(|| {
        slop_engine::run_native();
    }) {
        error!("Engine terminated abruptly: {:?}", e);
        std::process::exit(1);
    }
}

/// Sets up high-performance OS-level constraints
fn setup_os_optimizations() {
    #[cfg(feature = "high_priority")]
    {
        // Elevate thread priority for the main render/event loop thread
        let _ = thread_priority::set_current_thread_priority(
            thread_priority::ThreadPriority::Max,
        );
        info!("Elevated main thread priority to Max.");
    }

    // Windows high-resolution timer fix. Ensures std::thread::sleep is accurate to 1ms
    // preventing fixed-timestep physics from stuttering on Windows.
    #[cfg(target_os = "windows")]
    unsafe {
        // Requires `winapi` or `windows-sys` crate in your Cargo.toml if you uncomment this
        // windows_sys::Win32::Media::timeBeginPeriod(1);
    }
}

/// Optimizes CPU Floating Point math
fn setup_cpu_math() {
    // Flush-To-Zero (FTZ) and Denormals-Are-Zero (DAZ).
    // Game engines (especially physics and audio) can suffer massive CPU spikes 
    // when floating-point numbers get incredibly close to zero. This hardware flag forces them to 0.0.
    #[cfg(target_arch = "x86_64")]
    {
        // Note: You can use the `no-denormals` crate to safely do this cross-platform.
        // Doing this manually via inline assembly or core::arch ensures maximum speed.
        unsafe {
            let mut mxcsr: u32;
            std::arch::asm!("stmxcsr [{}]", out(reg) mxcsr);
            mxcsr |= (1 << 15) | (1 << 6); // Set FZ and DAZ flags
            std::arch::asm!("ldmxcsr [{}]", in(reg) mxcsr);
        }
        info!("Enabled FTZ/DAZ for CPU floating point optimizations.");
    }
}

/// Sets up logging and advanced crash-dumping
fn setup_diagnostics() {
    // 1. Initialize Logger
    // In a production engine, you'd output to both stdout AND a file.
    env_logger::Builder::new()
        .filter_level(if cfg!(debug_assertions) {
            LevelFilter::Debug
        } else {
            LevelFilter::Warn // Release mode performance
        })
        .format_timestamp_millis()
        .format_target(false)
        .parse_default_env()
        .init();

    // 2. Advanced Panic Hook
    panic::set_hook(Box::new(|panic_info| {
        let backtrace = Backtrace::force_capture();
        
        let msg = match panic_info.payload().downcast_ref::<&'static str>() {
            Some(s) => *s,
            None => match panic_info.payload().downcast_ref::<String>() {
                Some(s) => &s[..],
                None => "Box<dyn Any>",
            },
        };

        let location = panic_info.location().map_or("Unknown location".to_string(), |loc| {
            format!("{}:{}", loc.file(), loc.line())
        });

        let crash_msg = format!(
            "=== ENGINE CRASH ===\nReason: {}\nLocation: {}\n\nStack Trace:\n{}",
            msg, location, backtrace
        );

        // Print to console with color (if supported)
        eprintln!("\x1b[31;1m{}\x1b[0m", crash_msg);

        // Write to crash file for post-mortem debugging
        if let Ok(mut file) = File::create("engine_crash.log") {
            let _ = file.write_all(crash_msg.as_bytes());
            eprintln!("Crash report saved to engine_crash.log");
        }
    }));
}

/// Sets up a dedicated background async runtime.
/// winit/wgpu must run on the OS Main Thread. By moving Toko to a background thread,
/// you prevent networking, asset streaming, or DB calls from dropping your frame rate.
fn setup_async_runtime() -> Option<tokio::runtime::Runtime> {
    // Requires `tokio = { version = "1", features = ["rt-multi-thread"] }`
    #[cfg(feature = "tokio_runtime")]
    {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2) // Leave CPU cores for rendering/physics
            .thread_name("slop-engine-bg-worker")
            .enable_all()
            .build()
            .expect("Failed to initialize background Tokio runtime");
        
        info!("Background Async Runtime started.");
        return Some(rt);
    }
    
    #[cfg(not(feature = "tokio_runtime"))]
    None
}
