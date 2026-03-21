//! memory_optimizer.rs
//! STRA v2.0: Spatio-Temporal WGSL Resonance Allocator

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use wgpu::util::DeviceExt;
use crossbeam_channel::{unbounded, Sender, Receiver}; // Optimal for WASM thread bridging

const WGSL_BASE_ALIGNMENT: u64 = 16;
const CHUNK_SIZE: u64 = 16 * 1024 * 1024;
const DEFRAG_BYTES_PER_FRAME_BUDGET: u64 = 512 * 1024;
const LARGE_UPLOAD_THRESHOLD: u64 = 256 * 1024; // > 256KB uses async DMA

pub trait WgslResonant {
    fn wgsl_stride() -> u64;
    fn as_bytes(&self) -> &[u8];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryTier {
    ColdWasmHeap,
    WarmStaging,
    HotVram,
}

#[derive(Clone, Debug)]
pub struct WgslAllocation {
    pub chunk_id: usize,
    pub offset: u64,
    pub size: u64,
    pub tier: MemoryTier,
    pub resource_id: u64,
}

#[derive(Clone, Debug)]
struct FreeBlock {
    offset: u64,
    size: u64,
}

struct MemoryChunk {
    buffer: wgpu::Buffer,
    used: u64,
    capacity: u64,
    free_blocks: Vec<FreeBlock>,
    // Needed to quickly find the allocation adjacent to a free block
    allocations_by_offset: Vec<u64>, // Stores resource_ids sorted by offset
}

/// Tracks the async state machine for zero-stall WASM uploads/downloads
enum AsyncDmaTask {
    UploadPendingMap { resource_id: u64, data: Vec<u8>, staging_buf: Arc<wgpu::Buffer> },
    EvictPendingRead { resource_id: u64, readback_buf: Arc<wgpu::Buffer> },
}

pub struct MemoryOptimizer {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    
    vram_chunks: Vec<MemoryChunk>,
    allocations: HashMap<u64, WgslAllocation>,
    
    // Asynchronous State-Machine DMA
    dma_sender: Sender<AsyncDmaTask>,
    dma_receiver: Receiver<AsyncDmaTask>,
    
    // Backup data for evicted resources (Cold Storage)
    wasm_cold_storage: HashMap<u64, Vec<u8>>,
    
    defrag_cursor: usize,
}

impl MemoryOptimizer {
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        let (dma_sender, dma_receiver) = unbounded();
        
        Self {
            device,
            queue,
            vram_chunks: Vec::new(),
            allocations: HashMap::new(),
            dma_sender,
            dma_receiver,
            wasm_cold_storage: HashMap::new(),
            defrag_cursor: 0,
        }
    }

    // ==========================================
    // 4. ASYNC MAPPING FOR LARGE UPLOADS
    // ==========================================
    pub fn allocate<T: WgslResonant>(&mut self, resource_id: u64, data: &T) -> WgslAllocation {
        let size = T::wgsl_stride();
        let aligned_size = (size + WGSL_BASE_ALIGNMENT - 1) & !(WGSL_BASE_ALIGNMENT - 1);

        let chunk_idx = self.find_or_create_chunk(aligned_size);
        let chunk = &mut self.vram_chunks[chunk_idx];

        // Advanced allocation: Try to fit in a free block first (Best-Fit)
        let offset = if let Some(free_idx) = chunk.free_blocks.iter().position(|b| b.size >= aligned_size) {
            let free_block = &mut chunk.free_blocks[free_idx];
            let assigned_offset = free_block.offset;
            free_block.offset += aligned_size;
            free_block.size -= aligned_size;
            if free_block.size == 0 { chunk.free_blocks.remove(free_idx); }
            assigned_offset
        } else {
            let assigned_offset = chunk.used;
            chunk.used += aligned_size;
            assigned_offset
        };

        let allocation = WgslAllocation { chunk_id: chunk_idx, offset, size: aligned_size, tier: MemoryTier::HotVram, resource_id };
        self.allocations.insert(resource_id, allocation.clone());
        self.insert_allocation_sorted(chunk_idx, resource_id);

        let bytes = data.as_bytes().to_vec();

        // Branch based on payload size
        if aligned_size > LARGE_UPLOAD_THRESHOLD {
            self.stream_large_async(&allocation, bytes);
        } else {
            // Small payloads use the immediate queue (browser optimizes this internally)
            self.queue.write_buffer(&self.vram_chunks[chunk_idx].buffer, offset, &bytes);
        }

        allocation
    }

    fn stream_large_async(&self, alloc: &WgslAllocation, data: Vec<u8>) {
        let staging_buffer = Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("STRA_Async_Staging"),
            size: data.len() as u64,
            usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));

        let slice = staging_buffer.slice(..);
        let sender = self.dma_sender.clone();
        let res_id = alloc.resource_id;
        let buf_clone = Arc::clone(&staging_buffer);

        // Async callback: Pushes state to the DMA receiver safely across WASM bounds
        slice.map_async(wgpu::MapMode::Write, move |result| {
            if result.is_ok() {
                let _ = sender.send(AsyncDmaTask::UploadPendingMap { resource_id: res_id, data, staging_buf: buf_clone });
            }
        });
    }

    /// Call this once per frame in the main loop to process resolved async DMA tasks
    pub fn poll_dma_tasks(&mut self) {
        while let Ok(task) = self.dma_receiver.try_recv() {
            match task {
                AsyncDmaTask::UploadPendingMap { resource_id, data, staging_buf } => {
                    // Buffer is now mapped. Write data, unmap, and execute copy.
                    {
                        let mut mapped_view = staging_buf.slice(..).get_mapped_range_mut();
                        mapped_view.copy_from_slice(&data);
                    }
                    staging_buf.unmap();

                    if let Some(alloc) = self.allocations.get(&resource_id) {
                        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
                        encoder.copy_buffer_to_buffer(
                            &staging_buf, 0,
                            &self.vram_chunks[alloc.chunk_id].buffer, alloc.offset,
                            data.len() as u64
                        );
                        self.queue.submit(std::iter::once(encoder.finish()));
                    }
                },
                AsyncDmaTask::EvictPendingRead { resource_id, readback_buf } => {
                    // VRAM has been copied to readback_buf, and it is now mapped!
                    let data = {
                        let mapped_view = readback_buf.slice(..).get_mapped_range();
                        mapped_view.to_vec()
                    };
                    readback_buf.unmap();
                    
                    // Store locally in WASM, officially freeing the VRAM space
                    self.wasm_cold_storage.insert(resource_id, data);
                    self.free_vram_allocation(resource_id);
                }
            }
        }
    }

    // ==========================================
    // 3. EVICTION / RESIDENCY DOWNGRADE PATH
    // ==========================================
    /// Evicts an entity from Hot VRAM back to the WASM Heap if it hasn't been seen
    pub fn evict_to_cold_storage(&mut self, resource_id: u64) {
        let alloc = match self.allocations.get_mut(&resource_id) {
            Some(a) if a.tier == MemoryTier::HotVram => a,
            _ => return, // Already evicted or doesn't exist
        };

        alloc.tier = MemoryTier::WarmStaging; // Mark as transitioning

        let readback_buf = Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("STRA_Readback"),
            size: alloc.size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(
            &self.vram_chunks[alloc.chunk_id].buffer, alloc.offset,
            &readback_buf, 0,
            alloc.size
        );
        self.queue.submit(std::iter::once(encoder.finish()));

        // Ask WebGPU to map the buffer for reading asynchronously
        let slice = readback_buf.slice(..);
        let sender = self.dma_sender.clone();
        let buf_clone = Arc::clone(&readback_buf);
        
        slice.map_async(wgpu::MapMode::Read, move |res| {
            if res.is_ok() {
                let _ = sender.send(AsyncDmaTask::EvictPendingRead { resource_id, readback_buf: buf_clone });
            }
        });
    }

    fn free_vram_allocation(&mut self, resource_id: u64) {
        if let Some(alloc) = self.allocations.remove(&resource_id) {
            let chunk = &mut self.vram_chunks[alloc.chunk_id];
            chunk.free_blocks.push(FreeBlock { offset: alloc.offset, size: alloc.size });
            chunk.allocations_by_offset.retain(|&id| id != resource_id);
            
            // Trigger the coalescer!
            self.coalesce_free_blocks(alloc.chunk_id);
        }
    }

    // ==========================================
    // 1. FREE-BLOCK COALESCING
    // ==========================================
    /// Contiguous-Spectrum Coalescing: Merges adjacent free blocks to fight long-term fragmentation
    fn coalesce_free_blocks(&mut self, chunk_id: usize) {
        let chunk = &mut self.vram_chunks[chunk_id];
        if chunk.free_blocks.is_empty() { return; }

        // Sort by offset so adjacent blocks are next to each other in the array
        chunk.free_blocks.sort_unstable_by_key(|b| b.offset);

        let mut coalesced = Vec::with_capacity(chunk.free_blocks.len());
        let mut current = chunk.free_blocks[0].clone();

        for block in chunk.free_blocks.iter().skip(1) {
            if current.offset + current.size == block.offset {
                // They touch! Merge them.
                current.size += block.size;
            } else {
                coalesced.push(current);
                current = block.clone();
            }
        }
        coalesced.push(current);
        chunk.free_blocks = coalesced;
    }

    // ==========================================
    // 2. REAL DEFRAGMENTATION LOGIC
    // ==========================================
    /// GPU-Timeline Shift Defragmentation
    /// Finds gaps in VRAM and shifts right-side allocations to the left using `copy_buffer_to_buffer`.
    pub fn micro_defrag_tick(&mut self) {
        if self.vram_chunks.is_empty() { return; }

        let chunk_id = self.defrag_cursor;
        let chunk = &mut self.vram_chunks[chunk_id];
        
        // Coalesce before checking gaps to ensure maximum shifting efficiency
        self.coalesce_free_blocks(chunk_id);

        if let Some(first_gap) = chunk.free_blocks.first().cloned() {
            // Find the allocation sitting immediately *after* the gap
            let target_res_id = chunk.allocations_by_offset.iter()
                .find(|&&id| self.allocations[&id].offset >= first_gap.offset + first_gap.size)
                .cloned();

            if let Some(res_id) = target_res_id {
                let mut alloc = self.allocations.get_mut(&res_id).unwrap().clone();
                
                // Safety boundary: Ensure we don't exceed frame timing
                if alloc.size <= DEFRAG_BYTES_PER_FRAME_BUDGET {
                    let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("STRA_Defrag") });

                    // Execute the shift on the GPU timeline! Zero CPU iteration.
                    encoder.copy_buffer_to_buffer(
                        &chunk.buffer, alloc.offset,       // Source (old position)
                        &chunk.buffer, first_gap.offset,   // Destination (start of gap)
                        alloc.size
                    );
                    self.queue.submit(std::iter::once(encoder.finish()));

                    // Update Rust state
                    let old_offset = alloc.offset;
                    alloc.offset = first_gap.offset;
                    
                    // Shift the free block rightward
                    chunk.free_blocks[0].offset += alloc.size; 
                    
                    self.allocations.insert(res_id, alloc);
                    
                    // Re-sort allocation tracking
                    chunk.allocations_by_offset.sort_unstable_by_key(|&id| self.allocations[&id].offset);
                }
            }
        }

        self.defrag_cursor = (self.defrag_cursor + 1) % self.vram_chunks.len();
    }

    // ==========================================
    // Internal Utilities
    // ==========================================
    fn find_or_create_chunk(&mut self, size: u64) -> usize {
        for (i, chunk) in self.vram_chunks.iter().enumerate() {
            if chunk.capacity - chunk.used >= size { return i; }
            if chunk.free_blocks.iter().any(|b| b.size >= size) { return i; } // Check for fitting gap
        }

        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("STRA_VRAM_Chunk_{}", self.vram_chunks.len())),
            size: CHUNK_SIZE,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        self.vram_chunks.push(MemoryChunk {
            buffer, used: 0, capacity: CHUNK_SIZE,
            free_blocks: Vec::new(),
            allocations_by_offset: Vec::new(),
        });
        
        self.vram_chunks.len() - 1
    }

    fn insert_allocation_sorted(&mut self, chunk_id: usize, resource_id: u64) {
        let chunk = &mut self.vram_chunks[chunk_id];
        chunk.allocations_by_offset.push(resource_id);
        chunk.allocations_by_offset.sort_unstable_by_key(|&id| self.allocations[&id].offset);
    }
}
