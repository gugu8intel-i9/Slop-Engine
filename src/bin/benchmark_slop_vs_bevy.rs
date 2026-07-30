// src/bin/benchmark_slop_vs_bevy.rs
//! Live Real-Hardware Benchmark Comparing Slop Engine vs Bevy ECS 0.14
//! Runs 10,000 entities position update simulation on this exact machine and measures real CPU times.

use slop_engine::unreal_framework::*;
use glam::Vec3;
use std::time::Instant;
use bevy_ecs::prelude::*;

// ---------- Bevy Components & Systems ----------
#[derive(Component)]
struct BevyPosition(Vec3);

#[derive(Component)]
struct BevyVelocity(Vec3);

fn bevy_movement_system(mut query: Query<(&mut BevyPosition, &BevyVelocity)>) {
    for (mut pos, vel) in query.iter_mut() {
        pos.0 += vel.0 * 0.0166;
    }
}

fn main() {
    println!("===========================================================");
    println!("   SLOP ENGINE vs BEVY ECS 0.14 — LIVE HARDWARE BENCHMARK  ");
    println!("===========================================================\n");

    let num_entities = 10_000;
    let ticks = 500;

    println!("Hardware Execution Environment:");
    println!("  CPU Cores: {}", num_cpus());
    println!("  Entities Simulated: {}", num_entities);
    println!("  Simulation Frames: {}", ticks);
    println!("-----------------------------------------------------------\n");

    // ========================================================================
    // 1. SLOP ENGINE BENCHMARK
    // ========================================================================
    println!("1. SLOP ENGINE (UWorld / AActor Framework):");
    let slop_spawn_start = Instant::now();
    let mut slop_world = UWorld::new("SlopBenchWorld");

    for i in 0..num_entities {
        let mut actor = AActor::new(slop_world.next_actor_id(), format!("Actor_{}", i));
        actor.set_actor_location(Vec3::new(i as f32, 0.0, 0.0));
        slop_world.spawn_actor_direct(actor);
    }
    let slop_spawn_time = slop_spawn_start.elapsed();

    let slop_tick_start = Instant::now();
    for _ in 0..ticks {
        slop_world.tick(0.0166);
    }
    let slop_tick_total = slop_tick_start.elapsed();
    let slop_avg_tick_ms = (slop_tick_total.as_secs_f64() * 1000.0) / ticks as f64;
    let slop_fps = 1000.0 / slop_avg_tick_ms;

    println!("  • Spawn Time (10,000 Actors): {:?}", slop_spawn_time);
    println!("  • Total Tick Time (500 frames): {:?}", slop_tick_total);
    println!("  • Avg CPU Time per Frame: {:.4} ms", slop_avg_tick_ms);
    println!("  • Max Simulated Rate: {:.1} FPS\n", slop_fps);

    // ========================================================================
    // 2. BEVY ECS 0.14 BENCHMARK
    // ========================================================================
    println!("2. BEVY ECS v0.14:");
    let bevy_spawn_start = Instant::now();
    let mut bevy_world = World::new();
    let mut bevy_schedule = Schedule::default();
    bevy_schedule.add_systems(bevy_movement_system);

    for i in 0..num_entities {
        bevy_world.spawn((
            BevyPosition(Vec3::new(i as f32, 0.0, 0.0)),
            BevyVelocity(Vec3::new(10.0, 0.0, 0.0)),
        ));
    }
    let bevy_spawn_time = bevy_spawn_start.elapsed();

    let bevy_tick_start = Instant::now();
    for _ in 0..ticks {
        bevy_schedule.run(&mut bevy_world);
    }
    let bevy_tick_total = bevy_tick_start.elapsed();
    let bevy_avg_tick_ms = (bevy_tick_total.as_secs_f64() * 1000.0) / ticks as f64;
    let bevy_fps = 1000.0 / bevy_avg_tick_ms;

    println!("  • Spawn Time (10,000 Entities): {:?}", bevy_spawn_time);
    println!("  • Total Tick Time (500 frames): {:?}", bevy_tick_total);
    println!("  • Avg CPU Time per Frame: {:.4} ms", bevy_avg_tick_ms);
    println!("  • Max Simulated Rate: {:.1} FPS\n", bevy_fps);

    // ========================================================================
    // 3. COMPARISON SUMMARY
    // ========================================================================
    println!("===========================================================");
    println!("                  LIVE BENCHMARK SUMMARY                   ");
    println!("===========================================================");
    println!("  Metric                     │ Slop Engine  │ Bevy ECS 0.14");
    println!(" ────────────────────────────┼──────────────┼──────────────");
    println!("  10,000 Entity Spawn Time   │ {:>9.2?} │ {:>9.2?}", slop_spawn_time, bevy_spawn_time);
    println!("  Avg Frame Tick Time (CPU)  │ {:>8.4} ms │ {:>8.4} ms", slop_avg_tick_ms, bevy_avg_tick_ms);
    println!("  Simulated Frame Rate       │ {:>8.1} FPS│ {:>8.1} FPS", slop_fps, bevy_fps);
    println!("===========================================================");
}

fn num_cpus() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}
