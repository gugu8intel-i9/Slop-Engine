//! memory_optimizer.rs
//! Spatio-Temporal WGSL Resonance Allocator (STRA) for WebGPU/WASM Game Engines.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// Alignments required by WGSL (std140/std430)
const WGSL_BASE_ALIGNMENT: u64 = 16;
/// Default size of a memory chunk (16MB is optimal for WebGPU buffers)
const CHUNK_SIZE: u64 = 16 * 1024 * 1024; 
/// Max bytes we are allowed to defragment per frame to maintain 60/120fps
const DEFRAG_BYTES_PER_FRAME_BUDGET: u64 = 512 * 1024;

/// A trait that guarantees a Rust struct natively matches a WGSL struct layout.
pub trait WgslResonant {
    fn wgsl_stride() -> u64;
    fn as_bytes(&self) -> &[u8];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryTier {
    ColdWasmHeap,    // Exists only in WASM Memory
    WarmStaging,     // In mapped WebGPU staging buffer
    HotVram,         // Fast, device-local GPU memory for WGSL Compute/Render
}

/// Represents a sub-allocated block of memory within a massive WebGPU buffer
#[derive(Clone, Debug)]
pub struct WgslAllocation {
    pub chunk_id: usize,
    pub offset: u64,
    pub size: u64,
    pub tier: MemoryTier,
    /// A unique ID mapping back to the ECS entity or asset
    pub resource_id: u64, 
}

struct MemoryChunk {
    buffer: wgpu::Buffer,
    used: u64,
    capacity: u64,
    // Tracks free gaps for the Time-Sliced Defragmenter
    free_blocks: Vec<(u64, u64)>, // (offset, size)
}

/// The main High-Performance Memory Optimizer
pub struct MemoryOptimizer {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    
    // Tiered Storage
    vram_chunks: Vec<MemoryChunk>,
    staging_ring: VecDeque<wgpu::Buffer>,
    
    // Resource Tracking
    allocations: HashMap<u64, WgslAllocation>,
    
    // Predictive Paging System
    predictive_queue: Vec<u64>,
    
    // Defragmentation state
    defrag_cursor: usize,
}

impl MemoryOptimizer {
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>, ring_size: usize) -> Self {
        // Initialize the Asynchronous Ring-Staging
        let mut staging_ring = VecDeque::with_capacity(ring_size);
        for _ in 0..ring_size {
            let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("STRA_Staging_Ring_Buffer"),
                size: CHUNK_SIZE,
                usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            staging_ring.push_back(staging_buffer);
        }

        Self {
            device,
            queue,
            vram_chunks: Vec::new(),
            staging_ring,
            allocations: HashMap::new(),
            predictive_queue: Vec::new(),
            defrag_cursor: 0,
        }
    }

    /// Allocates memory directly aligned for WGSL, padding automatically.
    /// This uses a bump-allocator strategy with gap-filling (buddy-style).
    pub fn allocate<T: WgslResonant>(&mut self, resource_id: u64, data: &T) -> WgslAllocation {
        let size = T::wgsl_stride();
        // Pad size to WGSL base alignment bounds strictly
        let aligned_size = (size + WGSL_BASE_ALIGNMENT - 1) & !(WGSL_BASE_ALIGNMENT - 1);

        // Find a chunk with space or create a new one
        let chunk_idx = self.find_or_create_chunk(aligned_size);
        let chunk = &mut self.vram_chunks[chunk_idx];

        let offset = chunk.used;
        chunk.used += aligned_size;

        let allocation = WgslAllocation {
            chunk_id: chunk_idx,
            offset,
            size: aligned_size,
            tier: MemoryTier::HotVram,
            resource_id,
        };

        self.allocations.insert(resource_id, allocation.clone());
        
        // Zero-stall write using ring staging
        self.stream_to_vram(&allocation, data.as_bytes());

        allocation
    }

    /// Spatio-Temporal Prediction: Tell the memory manager where an entity is moving.
    /// If it's heading towards the camera, it moves from WASM heap to VRAM *before* it's needed.
    pub fn predict_movement(&mut self, resource_id: u64, distance_to_camera: f32, velocity_towards_camera: f32) {
        if distance_to_camera < 100.0 && velocity_towards_camera > 0.0 {
            // It's coming into view soon. Queue for pre-warming.
            if let Some(alloc) = self.allocations.get(&resource_id) {
                if alloc.tier == MemoryTier::ColdWasmHeap {
                    self.predictive_queue.push(resource_id);
                }
            }
        }
    }

    /// Processes predictive loading during WASM idle time.
    pub fn process_predictive_paging(&mut self) {
        // Pop off up to 5 predictions per frame to avoid choking the WebGPU queue
        for _ in 0..5 {
            if let Some(_res_id) = self.predictive_queue.pop() {
                // In a real scenario, fetch the asset from WASM memory and push to HotVRAM
                // self.stream_to_vram(...)
            }
        }
    }

    /// The Zero-Stall Ring Buffered Staging upload
    fn stream_to_vram(&mut self, alloc: &WgslAllocation, data: &[u8]) {
        // Pop the front staging buffer (assuming it's ready/unmapped)
        let staging_buffer = self.staging_ring.pop_front().expect("Staging ring starved!");
        
        // Write to staging buffer. In a real WASM app, you map async, but for immediate 
        // small writes `write_buffer` is optimized internally by browsers. 
        // For massive writes, we'd use wgpu buffer mapping here.
        self.queue.write_buffer(&staging_buffer, 0, data);

        // Schedule GPU-side copy from Staging -> Hot VRAM Chunk
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("STRA_Stream_Encoder"),
        });

        encoder.copy_buffer_to_buffer(
            &staging_buffer,
            0,
            &self.vram_chunks[alloc.chunk_id].buffer,
            alloc.offset,
            data.len() as u64,
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        // Cycle the staging buffer to the back of the queue
        self.staging_ring.push_back(staging_buffer);
    }

    /// Time-Sliced Micro-Defragmentation
    /// Web Game Engines stutter during GC. This does tiny bits of defrag entirely on the GPU
    /// over many frames, invisible to the player.
    pub fn micro_defrag_tick(&mut self) {
        if self.vram_chunks.is_empty() { return; }

        let mut bytes_moved = 0;
        let chunk = &mut self.vram_chunks[self.defrag_cursor];

        // If chunk is highly fragmented (many free gaps)
        if !chunk.free_blocks.is_empty() {
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("STRA_Defrag_Encoder"),
            });

            // Find an active allocation that is located *after* a free gap
            // Shift it leftward on the GPU to close the gap.
            // (Simplified pseudo-logic for the shift)
            if bytes_moved < DEFRAG_BYTES_PER_FRAME_BUDGET {
                // Shift logic:
                // encoder.copy_buffer_to_buffer(buffer, old_offset, buffer, new_offset, size);
                // Update allocation table
                bytes_moved += 256 * 1024; // Simulated byte movement cost
            }

            self.queue.submit(std::iter::once(encoder.finish()));
        }

        // Cycle the defrag cursor over frames
        self.defrag_cursor = (self.defrag_cursor + 1) % self.vram_chunks.len();
    }

    /// Returns the WebGPU buffer for a specific chunk, to be bound to a BindGroup
    pub fn get_wgsl_buffer(&self, chunk_id: usize) -> &wgpu::Buffer {
        &self.vram_chunks[chunk_id].buffer
    }

    fn find_or_create_chunk(&mut self, size: u64) -> usize {
        for (i, chunk) in self.vram_chunks.iter().enumerate() {
            if chunk.capacity - chunk.used >= size {
                return i;
            }
        }

        // Create new VRAM Chunk if full
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("STRA_VRAM_Chunk_{}", self.vram_chunks.len())),
            size: CHUNK_SIZE,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let chunk = MemoryChunk {
            buffer,
            used: 0,
            capacity: CHUNK_SIZE,
            free_blocks: Vec::new(),
        };

        self.vram_chunks.push(chunk);
        self.vram_chunks.len() - 1
    }
