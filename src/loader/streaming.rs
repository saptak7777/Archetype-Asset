//! Streaming buffer implementation for efficient memory management.
//!
//! This module provides a streaming vertex buffer that loads data in chunks on-demand.

use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use thiserror::Error;

/// Error type for streaming buffer operations
#[derive(Error, Debug)]
pub enum StreamingError {
    #[error("Chunk index out of bounds")]
    ChunkOutOfBounds,

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Asset error: {0}")]
    Asset(String),
}

/// A streaming vertex buffer that loads data in chunks
pub struct StreamingVertexBuffer<L: ChunkLoader> {
    buffer_id: u32,
    total_size: usize,
    chunk_size: usize,
    loaded_chunks: RwLock<HashMap<usize, Vec<u8>>>,
    chunk_loader: L,
}

/// Trait for loading chunks of data
/// 
/// Uses async-trait for dyn compatibility
#[async_trait::async_trait]
pub trait ChunkLoader: Send + Sync {
    /// Load a specific chunk of data asynchronously
    async fn load_chunk(&self, chunk_index: usize) -> Result<Vec<u8>, StreamingError>;
}

impl<L: ChunkLoader> StreamingVertexBuffer<L> {
    /// Creates a new streaming vertex buffer
    pub fn new(total_size: usize, chunk_loader: L, chunk_size: Option<usize>) -> Self {
        static BUFFER_ID_COUNTER: AtomicU32 = AtomicU32::new(1);

        Self {
            buffer_id: BUFFER_ID_COUNTER.fetch_add(1, Ordering::Relaxed),
            total_size,
            chunk_size: chunk_size.unwrap_or(64 * 1024),
            loaded_chunks: RwLock::new(HashMap::new()),
            chunk_loader,
        }
    }

    /// Get the total size of the buffer in bytes
    pub fn total_size(&self) -> usize {
        self.total_size
    }

    /// Get the size of each chunk in bytes
    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    /// Get the total number of chunks
    pub fn chunk_count(&self) -> usize {
        self.total_size.div_ceil(self.chunk_size)
    }

    /// Check if a chunk is currently loaded
    pub fn is_chunk_loaded(&self, chunk_index: usize) -> bool {
        self.loaded_chunks.read().contains_key(&chunk_index)
    }

    /// Get the number of currently loaded chunks
    pub fn loaded_chunk_count(&self) -> usize {
        self.loaded_chunks.read().len()
    }

    /// Ensure a specific chunk is loaded
    pub async fn ensure_chunk_loaded(&self, chunk_index: usize) -> Result<(), StreamingError> {
        if chunk_index >= self.chunk_count() {
            return Err(StreamingError::ChunkOutOfBounds);
        }

        if self.is_chunk_loaded(chunk_index) {
            return Ok(());
        }

        let chunk_data = self.chunk_loader.load_chunk(chunk_index).await?;
        self.loaded_chunks.write().insert(chunk_index, chunk_data);

        Ok(())
    }

    /// Get a reference to a loaded chunk
    pub fn get_chunk(&self, chunk_index: usize) -> Option<Vec<u8>> {
        self.loaded_chunks.read().get(&chunk_index).cloned()
    }

    /// Unload a specific chunk to free memory
    pub fn unload_chunk(&self, chunk_index: usize) -> bool {
        self.loaded_chunks.write().remove(&chunk_index).is_some()
    }

    /// Unload all chunks to free memory
    pub fn unload_all_chunks(&self) {
        self.loaded_chunks.write().clear();
    }

    /// Get the buffer ID
    pub fn buffer_id(&self) -> u32 {
        self.buffer_id
    }
}

/// Simple file-based chunk loader
pub struct FileChunkLoader {
    base_path: std::path::PathBuf,
    chunk_size: usize,
}

impl FileChunkLoader {
    /// Create a new file chunk loader
    pub fn new(base_path: impl Into<std::path::PathBuf>, chunk_size: usize) -> Self {
        Self {
            base_path: base_path.into(),
            chunk_size,
        }
    }
}

#[async_trait::async_trait]
impl ChunkLoader for FileChunkLoader {
    async fn load_chunk(&self, chunk_index: usize) -> Result<Vec<u8>, StreamingError> {
        // Simple implementation - read from file
        let offset = chunk_index * self.chunk_size;
        let data = std::fs::read(&self.base_path)?;
        
        let end = (offset + self.chunk_size).min(data.len());
        if offset >= data.len() {
            return Err(StreamingError::ChunkOutOfBounds);
        }
        
        Ok(data[offset..end].to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestChunkLoader {
        chunk_size: usize,
    }

    #[async_trait::async_trait]
    impl ChunkLoader for TestChunkLoader {
        async fn load_chunk(&self, chunk_index: usize) -> Result<Vec<u8>, StreamingError> {
            let start = chunk_index * self.chunk_size;
            let end = (chunk_index + 1) * self.chunk_size;
            Ok((start..end).map(|i| (i % 256) as u8).collect())
        }
    }

    #[test]
    fn test_streaming_buffer_creation() {
        let loader = TestChunkLoader { chunk_size: 1024 };
        let buffer = StreamingVertexBuffer::new(4096, loader, Some(1024));
        
        assert_eq!(buffer.total_size(), 4096);
        assert_eq!(buffer.chunk_size(), 1024);
        assert_eq!(buffer.chunk_count(), 4);
    }
}
