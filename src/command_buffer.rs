use crossbeam_channel::{bounded, Receiver, Sender, TryRecvError};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

// ============================================================================
// 1. DOMAIN TYPES (Strong Typing for Safety & Performance)
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EntityId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AssetId(pub u64); // Better than &'static str for performance (hashed strings)

pub type Vector3 = [f32; 3];

// ============================================================================
// 2. COMMAND ENUM (Optimized for Size)
// ============================================================================

/// Represents a generic unit of work for the engine.
/// 
/// OPTIMIZATION: Enums take the size of their largest variant. 
/// We use `Box` for large payloads (like LevelData) to keep the base enum small, 
/// which drastically improves CPU Cache Locality when moving through the channel.
#[derive(Debug)]
pub enum EngineCommand {
    UpdateTransform(EntityId, Vector3),
    SpawnEntity(EntityId),
    DestroyEntity(EntityId),
    PlaySound(AssetId),
    
    // Feature: Grouping multiple physics updates
    PhysicsUpdate(EntityId, Vector3, Vector3), // Entity, Pos, Velocity
    
    // Feature: Boxed payload to prevent enum size bloat
    LoadLevel(Box<LevelData>), 
    
    // Feature: Engine lifecycle controls
    Quit,
}

#[derive(Debug)]
pub struct LevelData {
    pub level_name: String,
    pub entity_count: usize,
    pub data_blob: Vec<u8>,
}

// ============================================================================
// 3. THE COMMAND BUS (Global State via OnceLock)
// ============================================================================

/// Global metrics for engine profiling
pub struct CommandMetrics {
    pub emitted: AtomicUsize,
    pub processed: AtomicUsize,
    pub dropped: AtomicUsize, // Commands dropped due to full queue
}

impl CommandMetrics {
    const fn new() -> Self {
        Self {
            emitted: AtomicUsize::new(0),
            processed: AtomicUsize::new(0),
            dropped: AtomicUsize::new(0),
        }
    }
}

pub static METRICS: CommandMetrics = CommandMetrics::new();

struct CommandBus {
    sender: Sender<EngineCommand>,
    receiver: Receiver<EngineCommand>,
}

/// Native Rust alternative to `lazy_static!`. Zero external dependencies.
static COMMAND_BUS: OnceLock<CommandBus> = OnceLock::new();

/// Initializes the command bus. 
/// OPTIMIZATION: We use a `bounded` channel. Game engines should fail predictably.
/// Unbounded channels can cause OOM errors if producers outpace the main thread.
const CHANNEL_CAPACITY: usize = 100_000;

fn get_bus() -> &'static CommandBus {
    COMMAND_BUS.get_or_init(|| {
        // Bounded is generally faster than unbounded in crossbeam when capacity is large
        let (sender, receiver) = bounded(CHANNEL_CAPACITY);
        CommandBus { sender, receiver }
    })
}

// ============================================================================
// 4. PRODUCER API (Thread-Safe, Non-Blocking)
// ============================================================================

/// Push a single command from any thread without locking the main loop.
#[inline(always)] // Force inline for hot-path performance
pub fn emit_command(cmd: EngineCommand) {
    let bus = get_bus();
    match bus.sender.try_send(cmd) {
        Ok(_) => {
            METRICS.emitted.fetch_add(1, Ordering::Relaxed);
        }
        Err(_) => {
            // Queue is full! In a real engine, log an error or expand queue.
            METRICS.dropped.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// FEATURE: Batch emitting. 
/// Pushing multiple commands at once is more cache-friendly and reduces function call overhead.
pub fn emit_batch<I>(commands: I)
where
    I: IntoIterator<Item = EngineCommand>,
{
    let bus = get_bus();
    let mut count = 0;
    
    for cmd in commands {
        if bus.sender.try_send(cmd).is_ok() {
            count += 1;
        } else {
            METRICS.dropped.fetch_add(1, Ordering::Relaxed);
        }
    }
    METRICS.emitted.fetch_add(count, Ordering::Relaxed);
}

// ============================================================================
// 5. CONSUMER API (Main Thread processing)
// ============================================================================

/// Drains the queue completely. Best for initialization or non-critical loops.
pub fn flush_commands_all<F>(mut processor: F)
where
    F: FnMut(EngineCommand),
{
    let bus = get_bus();
    let mut count = 0;

    while let Ok(cmd) = bus.receiver.try_recv() {
        processor(cmd);
        count += 1;
    }
    
    METRICS.processed.fetch_add(count, Ordering::Relaxed);
}

/// FEATURE: Time-budgeted flushing.
/// Crucial for game engines! If the queue has millions of events, processing them all
/// will cause a lag spike. This function processes events until a time limit is hit.
pub fn flush_commands_budgeted<F>(max_duration: Duration, mut processor: F) -> bool
where
    F: FnMut(EngineCommand),
{
    let bus = get_bus();
    let start_time = Instant::now();
    let mut count = 0;
    let mut finished_draining = true;

    loop {
        match bus.receiver.try_recv() {
            Ok(cmd) => {
                processor(cmd);
                count += 1;

                // Check budget every 16 commands to avoid syscall overhead of `Instant::now()`
                if count % 16 == 0 && start_time.elapsed() >= max_duration {
                    finished_draining = false;
                    break;
                }
            }
            Err(TryRecvError::Empty) => break, // Queue empty
            Err(TryRecvError::Disconnected) => break,
        }
    }

    METRICS.processed.fetch_add(count, Ordering::Relaxed);
    finished_draining // Returns false if we ran out of time before emptying the queue
}

// ============================================================================
// 6. DEMONSTRATION / USAGE
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn engine_simulation() {
        // 1. Spawn worker threads (e.g., Physics, Network, Audio)
        let mut handles = vec![];

        for t in 0..3 {
            handles.push(thread::spawn(move || {
                // Thread generates some work
                let mut batch = Vec::new();
                for i in 0..1000 {
                    batch.push(EngineCommand::UpdateTransform(
                        EntityId(i + (t * 1000)),
                        [1.0, 2.0, 3.0],
                    ));
                }
                
                // Emitting in batch is much faster
                emit_batch(batch);
                
                // Emitting single high-priority event
                emit_command(EngineCommand::PlaySound(AssetId(424242)));
            }));
        }

        // Wait for workers to finish simulating work
        for h in handles {
            h.join().unwrap();
        }

        // 2. Main Loop Simulation
        let time_budget = Duration::from_millis(2); // 2ms budget per frame
        
        let mut frame = 0;
        let mut running = true;

        while running && frame < 10 {
            frame += 1;
            
            // Drain queue with a time budget
            let completed = flush_commands_budgeted(time_budget, |cmd| {
                match cmd {
                    EngineCommand::UpdateTransform(id, _pos) => {
                        // In reality, update your ECS here
                    }
                    EngineCommand::PlaySound(_asset) => {
                        // Play audio
                    }
                    EngineCommand::Quit => {
                        running = false;
                    }
                    _ => {}
                }
            });

            if !completed {
                println!("Frame {}: Hit time budget! Deferring remaining commands.", frame);
            }
        }

        // 3. Print Metrics
        println!(
            "Metrics - Emitted: {}, Processed: {}, Dropped: {}",
            METRICS.emitted.load(Ordering::Relaxed),
            METRICS.processed.load(Ordering::Relaxed),
            METRICS.dropped.load(Ordering::Relaxed)
        );
    }
}
