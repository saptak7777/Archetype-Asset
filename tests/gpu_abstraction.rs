//! Integration tests for GPU abstraction layer

use archetype_asset::{BufferUsage, GpuDevice, MockGpu};

#[test]
fn test_mock_gpu_integration() {
    let gpu = MockGpu::new();

    // Create a buffer
    let buffer = gpu.allocate_buffer(1024, BufferUsage::Vertex).unwrap();

    // Upload data
    let data = vec![1u8, 2, 3, 4];
    gpu.upload_buffer_data(&buffer, 0, &data).unwrap();

    // Verify GPU operations work
    assert!(buffer.size() >= 1024);
}

#[test]
fn test_gpu_trait_object_safety() {
    // Ensure GpuDevice can be used as a trait bound
    fn use_gpu<G: GpuDevice>(gpu: &G) {
        let _ = gpu.allocate_buffer(64, BufferUsage::Uniform);
    }

    let gpu = MockGpu::new();
    use_gpu(&gpu);
}
