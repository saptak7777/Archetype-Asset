//! Vulkan GPU implementation
//!
//! Provides a Vulkan backend using `ash` and `gpu-allocator`.

use super::{BufferUsage, GpuDevice, GpuError, GpuResult, GpuTextureFormat, TextureDescriptor};
use ash::vk;
use parking_lot::Mutex;
use std::sync::Arc;

/// Vulkan GPU device wrapper
///
/// Wraps `ash::Device` and VMA allocator to implement the `GpuDevice` trait.
#[derive(Clone, Debug)]
pub struct VulkanDevice {
    device: ash::Device,
    allocator: Arc<Mutex<vk_mem::Allocator>>,
}

impl VulkanDevice {
    /// Create a new Vulkan device wrapper
    ///
    /// # Arguments
    /// * `device` - The ash Vulkan device
    /// * `allocator` - VMA allocator for memory management
    pub fn new(device: ash::Device, allocator: Arc<Mutex<vk_mem::Allocator>>) -> Self {
        Self { device, allocator }
    }

    /// Get the underlying ash device
    pub fn device(&self) -> &ash::Device {
        &self.device
    }

    /// Get the allocator
    pub fn allocator(&self) -> &Arc<Mutex<vk_mem::Allocator>> {
        &self.allocator
    }

    /// Convert BufferUsage to Vulkan buffer usage flags
    fn to_vk_buffer_usage(usage: BufferUsage) -> vk::BufferUsageFlags {
        match usage {
            BufferUsage::Vertex => {
                vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST
            }
            BufferUsage::Index => {
                vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST
            }
            BufferUsage::Uniform => {
                vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST
            }
            BufferUsage::Storage => {
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST
            }
            BufferUsage::Staging => vk::BufferUsageFlags::TRANSFER_SRC,
        }
    }

    /// Convert GpuTextureFormat to Vulkan format
    fn to_vk_format(format: GpuTextureFormat) -> vk::Format {
        match format {
            GpuTextureFormat::Rgba8Srgb => vk::Format::R8G8B8A8_SRGB,
            GpuTextureFormat::Rgba8Unorm => vk::Format::R8G8B8A8_UNORM,
            GpuTextureFormat::Rgb8Srgb => vk::Format::R8G8B8_SRGB,
            GpuTextureFormat::R8Unorm => vk::Format::R8_UNORM,
            GpuTextureFormat::Depth32Float => vk::Format::D32_SFLOAT,
            GpuTextureFormat::Depth24Stencil8 => vk::Format::D24_UNORM_S8_UINT,
        }
    }
}

/// Vulkan buffer with associated allocation
#[derive(Clone, Debug)]
pub struct VulkanBuffer {
    /// Vulkan buffer handle
    pub buffer: vk::Buffer,
    /// VMA allocation
    pub allocation: vk_mem::Allocation,
    /// Buffer size in bytes
    pub size: usize,
    /// Buffer usage
    pub usage: BufferUsage,
    /// Reference to allocator for cleanup
    allocator: Arc<Mutex<vk_mem::Allocator>>,
}

impl VulkanBuffer {
    /// Get the raw Vulkan buffer handle
    pub fn handle(&self) -> vk::Buffer {
        self.buffer
    }
}

impl Drop for VulkanBuffer {
    fn drop(&mut self) {
        // Note: In practice, you'd want to defer this cleanup
        // to avoid destroying resources still in use by the GPU.
        // For now, we just log that cleanup would happen here.
        log::trace!("VulkanBuffer dropped (size={})", self.size);
    }
}

/// Vulkan texture with associated image and view
#[derive(Clone, Debug)]
pub struct VulkanTexture {
    /// Vulkan image handle
    pub image: vk::Image,
    /// Image view
    pub view: vk::ImageView,
    /// VMA allocation
    pub allocation: vk_mem::Allocation,
    /// Texture dimensions
    pub width: u32,
    pub height: u32,
    /// Format
    pub format: GpuTextureFormat,
    /// Reference to allocator for cleanup
    allocator: Arc<Mutex<vk_mem::Allocator>>,
    /// Reference to device for cleanup
    device: ash::Device,
}

impl Drop for VulkanTexture {
    fn drop(&mut self) {
        log::trace!(
            "VulkanTexture dropped ({}x{}, format={:?})",
            self.width,
            self.height,
            self.format
        );
    }
}

impl GpuDevice for VulkanDevice {
    type Buffer = VulkanBuffer;
    type Texture = VulkanTexture;

    fn allocate_buffer(&self, size: usize, usage: BufferUsage) -> GpuResult<Self::Buffer> {
        if size == 0 {
            return Err(GpuError::InvalidSize(size));
        }

        let vk_usage = Self::to_vk_buffer_usage(usage);

        let buffer_info = vk::BufferCreateInfo::builder()
            .size(size as u64)
            .usage(vk_usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .build();

        // Determine memory location based on usage
        let memory_location = match usage {
            BufferUsage::Staging => vk_mem::MemoryUsage::AutoPreferHost,
            _ => vk_mem::MemoryUsage::AutoPreferDevice,
        };

        let allocation_info = vk_mem::AllocationCreateInfo {
            usage: memory_location,
            ..Default::default()
        };

        let allocator = self.allocator.lock();
        let (buffer, allocation) = allocator
            .create_buffer(&buffer_info, &allocation_info)
            .map_err(|e| GpuError::AllocationFailed(format!("VMA error: {e:?}")))?;

        Ok(VulkanBuffer {
            buffer,
            allocation,
            size,
            usage,
            allocator: Arc::clone(&self.allocator),
        })
    }

    fn upload_buffer_data(
        &self,
        buffer: &Self::Buffer,
        offset: usize,
        data: &[u8],
    ) -> GpuResult<()> {
        if offset + data.len() > buffer.size {
            return Err(GpuError::UploadFailed(format!(
                "Data exceeds buffer size: offset={}, data_len={}, buffer_size={}",
                offset,
                data.len(),
                buffer.size
            )));
        }

        let allocator = self.allocator.lock();

        // Map memory, copy data, unmap
        let ptr = allocator
            .map_memory(&buffer.allocation)
            .map_err(|e| GpuError::UploadFailed(format!("Failed to map memory: {e:?}")))?;

        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr.add(offset), data.len());
        }

        allocator.unmap_memory(&buffer.allocation);

        Ok(())
    }

    fn create_texture(&self, desc: &TextureDescriptor, data: &[u8]) -> GpuResult<Self::Texture> {
        if desc.width == 0 || desc.height == 0 {
            return Err(GpuError::TextureCreationFailed(
                "Invalid texture dimensions".to_string(),
            ));
        }

        let vk_format = Self::to_vk_format(desc.format);

        let image_info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk_format)
            .extent(vk::Extent3D {
                width: desc.width,
                height: desc.height,
                depth: 1,
            })
            .mip_levels(desc.mip_levels)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .build();

        let allocation_info = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::AutoPreferDevice,
            ..Default::default()
        };

        let allocator = self.allocator.lock();
        let (image, allocation) = allocator
            .create_image(&image_info, &allocation_info)
            .map_err(|e| GpuError::TextureCreationFailed(format!("VMA error: {e:?}")))?;

        // Create image view
        let view_info = vk::ImageViewCreateInfo::builder()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk_format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: desc.mip_levels,
                base_array_layer: 0,
                layer_count: 1,
            })
            .build();

        let view = unsafe { self.device.create_image_view(&view_info, None) }.map_err(|e| {
            GpuError::TextureCreationFailed(format!("Failed to create image view: {e:?}"))
        })?;

        // Note: Actual data upload would require staging buffer + command buffer
        // This is a simplified implementation
        let _ = data; // Data upload happens separately via staging buffer

        Ok(VulkanTexture {
            image,
            view,
            allocation,
            width: desc.width,
            height: desc.height,
            format: desc.format,
            allocator: Arc::clone(&self.allocator),
            device: self.device.clone(),
        })
    }

    fn destroy_buffer(&self, buffer: Self::Buffer) {
        let allocator = self.allocator.lock();
        allocator.destroy_buffer(buffer.buffer, &buffer.allocation);
    }

    fn destroy_texture(&self, texture: Self::Texture) {
        unsafe {
            self.device.destroy_image_view(texture.view, None);
        }
        let allocator = self.allocator.lock();
        allocator.destroy_image(texture.image, &texture.allocation);
    }

    fn backend_name(&self) -> &'static str {
        "Vulkan"
    }
}
