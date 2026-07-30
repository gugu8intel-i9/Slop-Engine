// src/bin/slop_editor.rs
//! Standalone Executable Launch Application for Slop Engine Editor

use slop_engine::editor::SlopEngineEditor;
use slop_engine::unreal_framework::EComponentType;

fn main() {
    println!("===========================================================");
    println!("              SLOP ENGINE GUI EDITOR v2.2                  ");
    println!("  High-Performance Cross-Platform Editor (Linux/Mac/Win/Web) ");
    println!("===========================================================");

    let mut editor = SlopEngineEditor::new();
    println!("Loaded Level: {}", editor.world.name);
    println!("Initial Actor Count: {}", editor.world.actors.len());

    // Demonstrate spawning an actor
    let light_id = editor.spawn_new_actor("EditorSpotLight", EComponentType::Scene);
    println!("Spawned new Actor in Editor, ID: {}", light_id);

    // Attach Blueprint
    editor.attach_movement_blueprint_to_selected();

    // Start Play-In-Editor (PIE)
    editor.play();
    println!("Play-In-Editor (PIE) Mode Activated.");

    // Simulate 60 frames
    for _ in 0..60 {
        editor.tick(0.0166);
    }

    println!("Simulation Frame: {}", editor.frame_counter);
    println!("Editor FPS: {:.1}", editor.fps);

    if let Ok(json_save) = editor.save_project_to_json() {
        println!("Project State Saved:\n{}", json_save);
    }

    println!("===========================================================");
    println!(" Slop Engine Editor execution completed successfully. ");
    println!("===========================================================");
}
