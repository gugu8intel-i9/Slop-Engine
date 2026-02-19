use anyhow::*;
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgba};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use wgpu::util::DeviceExt;

// ─────────────────────────────────────────────────────────────────────────────
// Texture configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Controls how a texture is created and sampled.
#[derive(Debug, Clone)]
pub struct TextureConfig {
    pub label: Option<String>,
    pub format: wgpu::TextureFormat,
    pub usage: wgpu::TextureUsages,
    pub address_mode: wgpu::AddressMode,
    pub mag_filter: wgpu::FilterMode,
    pub min_filter: wgpu::FilterMode,
    pub mipmap_filter: wgpu::FilterMode,
    pub generate_mipmaps: bool,
    pub sample_count: u32,
    pub dimension: wgpu::TextureDimension,
    pub comparison: Option<wgpu::CompareFunction>,
    pub anisotropy_clamp: u16,
    pub lod_min_clamp: f32,
    pub lod_max_clamp: f32,
    pub border_color: Option<wgpu::SamplerBorderColor>,
    pub srgb: bool,
    pub premultiply_alpha: bool,
    pub max_dimension: Option<u32>,
}

impl Default for TextureConfig {
    fn default() -> Self {
        Self {
            label: None,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            address_mode: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            generate_mipmaps: true,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            comparison: None,
            anisotropy_clamp: 1,
            lod_min_clamp: 0.0,
            lod_max_clamp: 32.0,
            border_color: None,
            srgb: true,
            premultiply_alpha: false,
            max_dimension: None,
        }
    }
}

impl TextureConfig {
    pub fn pixel_art() -> Self {
        Self {
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            generate_mipmaps: false,
            ..Default::default()
        }
    }

    pub fn normal_map() -> Self {
        Self {
            format: wgpu::TextureFormat::Rgba8Unorm,
            srgb: false,
            ..Default::default()
        }
    }

    pub fn hdr() -> Self {
        Self {
            format: wgpu::TextureFormat::Rgba16Float,
            srgb: false,
            ..Default::default()
        }
    }

    pub fn render_target(width: u32, height: u32) -> Self {
        Self {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            generate_mipmaps: false,
            ..Default::default()
        }
    }

    pub fn repeating() -> Self {
        Self {
            address_mode: wgpu::AddressMode::Repeat,
            ..Default::default()
        }
    }

    pub fn mirrored() -> Self {
        Self {
            address_mode: wgpu::AddressMode::MirrorRepeat,
            ..Default::default()
        }
    }

    pub fn with_label(mut self, label: &str) -> Self {
        self.label = Some(label.to_string());
        self
    }

    pub fn with_anisotropy(mut self, clamp: u16) -> Self {
        self.anisotropy_clamp = clamp;
        self
    }

    pub fn with_max_dimension(mut self, max: u32) -> Self {
        self.max_dimension = Some(max);
        self
    }

    pub fn with_format(mut self, format: wgpu::TextureFormat) -> Self {
        self.format = format;
        self
    }

    pub fn with_usage(mut self, usage: wgpu::TextureUsages) -> Self {
        self.usage = usage;
        self
    }

    pub fn with_address_mode(mut self, mode: wgpu::AddressMode) -> Self {
        self.address_mode = mode;
        self
    }

    pub fn with_premultiply_alpha(mut self, premultiply: bool) -> Self {
        self.premultiply_alpha = premultiply;
        self
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Core texture struct
// ─────────────────────────────────────────────────────────────────────────────

pub struct Texture {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
    pub size: wgpu::Extent3d,
    pub format: wgpu::TextureFormat,
    pub mip_level_count: u32,
}

impl Texture {
    // ── Dimensions & utilities ───────────────────────────────────────────

    #[inline]
    pub fn width(&self) -> u32 {
        self.size.width
    }

    #[inline]
    pub fn height(&self) -> u32 {
        self.size.height
    }

    #[inline]
    pub fn depth(&self) -> u32 {
        self.size.depth_or_array_layers
    }

    #[inline]
    pub fn aspect_ratio(&self) -> f32 {
        self.size.width as f32 / self.size.height as f32
    }

    /// Maximum number of mip levels for a given dimension pair.
    #[inline]
    pub fn max_mip_levels(width: u32, height: u32) -> u32 {
        (width.max(height) as f32).log2().floor() as u32 + 1
    }

    /// Bytes per pixel for common formats (returns 0 for compressed).
    #[inline]
    pub fn bytes_per_pixel(format: wgpu::TextureFormat) -> u32 {
        match format {
            wgpu::TextureFormat::R8Unorm | wgpu::TextureFormat::R8Snorm | wgpu::TextureFormat::R8Uint | wgpu::TextureFormat::R8Sint => 1,
            wgpu::TextureFormat::Rg8Unorm | wgpu::TextureFormat::Rg8Snorm | wgpu::TextureFormat::Rg8Uint | wgpu::TextureFormat::Rg8Sint => 2,
            wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb | wgpu::TextureFormat::Rgba8Snorm | wgpu::TextureFormat::Rgba8Uint | wgpu::TextureFormat::Rgba8Sint | wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb => 4,
            wgpu::TextureFormat::R16Float | wgpu::TextureFormat::R16Uint | wgpu::TextureFormat::R16Sint => 2,
            wgpu::TextureFormat::Rg16Float | wgpu::TextureFormat::Rg16Uint | wgpu::TextureFormat::Rg16Sint => 4,
            wgpu::TextureFormat::Rgba16Float | wgpu::TextureFormat::Rgba16Uint | wgpu::TextureFormat::Rgba16Sint => 8,
            wgpu::TextureFormat::R32Float | wgpu::TextureFormat::R32Uint | wgpu::TextureFormat::R32Sint => 4,
            wgpu::TextureFormat::Rg32Float | wgpu::TextureFormat::Rg32Uint | wgpu::TextureFormat::Rg32Sint => 8,
            wgpu::TextureFormat::Rgba32Float | wgpu::TextureFormat::Rgba32Uint | wgpu::TextureFormat::Rgba32Sint => 16,
            wgpu::TextureFormat::Depth32Float => 4,
            wgpu::TextureFormat::Depth24Plus | wgpu::TextureFormat::Depth24PlusStencil8 => 4,
            wgpu::TextureFormat::Depth16Unorm => 2,
            _ => 0,
        }
    }

    // ── Construction: from bytes ─────────────────────────────────────────

    /// Convenience: decode image bytes with default config.
    pub fn from_bytes(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bytes: &[u8],
        label: &str,
    ) -> Result<Self> {
        Self::from_bytes_with_config(device, queue, bytes, TextureConfig::default().with_label(label))
    }

    /// Decode image bytes with a full config.
    pub fn from_bytes_with_config(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bytes: &[u8],
        config: TextureConfig,
    ) -> Result<Self> {
        let img = image::load_from_memory(bytes)
            .context("Failed to decode image from bytes")?;
        Self::from_image(device, queue, &img, config)
    }

    // ── Construction: from file path ─────────────────────────────────────

    pub fn from_path(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        path: impl AsRef<Path>,
        label: &str,
    ) -> Result<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path)
            .with_context(|| format!("Failed to read texture file: {}", path.display()))?;
        Self::from_bytes_with_config(device, queue, &bytes, TextureConfig::default().with_label(label))
    }

    pub fn from_path_with_config(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        path: impl AsRef<Path>,
        config: TextureConfig,
    ) -> Result<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path)
            .with_context(|| format!("Failed to read texture file: {}", path.display()))?;
        Self::from_bytes_with_config(device, queue, &bytes, config)
    }

    // ── Construction: from DynamicImage ──────────────────────────────────

    pub fn from_image(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        img: &DynamicImage,
        config: TextureConfig,
    ) -> Result<Self> {
        // Optionally downscale before uploading to GPU
        let img = if let Some(max) = config.max_dimension {
            let (w, h) = img.dimensions();
            if w > max || h > max {
                std::borrow::Cow::Owned(img.resize(max, max, image::imageops::FilterType::Lanczos3))
            } else {
                std::borrow::Cow::Borrowed(img)
            }
        } else {
            std::borrow::Cow::Borrowed(img)
        };

        let mut rgba = img.to_rgba8();
        let (width, height) = img.dimensions();

        // Premultiply alpha if requested (improves blending quality)
        if config.premultiply_alpha {
            premultiply_alpha_inplace(&mut rgba);
        }

        let mip_level_count = if config.generate_mipmaps {
            Self::max_mip_levels(width, height)
        } else {
            1
        };

        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        // If we generate mipmaps, we need RENDER_ATTACHMENT to blit each level
        let mut usage = config.usage | wgpu::TextureUsages::COPY_DST;
        if config.generate_mipmaps && mip_level_count > 1 {
            usage |= wgpu::TextureUsages::RENDER_ATTACHMENT;
        }

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: config.label.as_deref(),
            size,
            mip_level_count,
            sample_count: config.sample_count,
            dimension: config.dimension,
            format: config.format,
            usage,
            view_formats: &[],
        });

        // Upload mip 0
        queue.write_texture(
            wgpu::ImageCopyTexture {
                aspect: wgpu::TextureAspect::All,
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            &rgba,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * width),
                rows_per_image: Some(height),
            },
            size,
        );

        // CPU-side mipmap generation (no extra render pass overhead)
        if config.generate_mipmaps && mip_level_count > 1 {
            generate_mipmaps_cpu(device, queue, &texture, &rgba, width, height, mip_level_count);
        }

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: config.label.as_deref().map(|l| {
                // Tiny allocation here is fine; it only happens at load time
                // We cannot return a reference to a local, so we leak a small string.
                // In practice you'd use a bump allocator or just pass a &'static str.
                // For simplicity, we duplicate the pattern from wgpu examples.
                l
            }),
            address_mode_u: config.address_mode,
            address_mode_v: config.address_mode,
            address_mode_w: config.address_mode,
            mag_filter: config.mag_filter,
            min_filter: config.min_filter,
            mipmap_filter: config.mipmap_filter,
            lod_min_clamp: config.lod_min_clamp,
            lod_max_clamp: config.lod_max_clamp,
            compare: config.comparison,
            anisotropy_clamp: config.anisotropy_clamp,
            border_color: config.border_color,
        });

        Ok(Self {
            texture,
            view,
            sampler,
            size,
            format: config.format,
            mip_level_count,
        })
    }

    // ── Construction: from raw RGBA bytes ────────────────────────────────

    pub fn from_rgba(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        data: &[u8],
        width: u32,
        height: u32,
        config: TextureConfig,
    ) -> Result<Self> {
        let expected = (width * height * 4) as usize;
        ensure!(
            data.len() >= expected,
            "RGBA data too short: expected {} bytes, got {}",
            expected,
            data.len()
        );

        let mip_level_count = if config.generate_mipmaps {
            Self::max_mip_levels(width, height)
        } else {
            1
        };

        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        let mut usage = config.usage | wgpu::TextureUsages::COPY_DST;
        if config.generate_mipmaps && mip_level_count > 1 {
            usage |= wgpu::TextureUsages::RENDER_ATTACHMENT;
        }

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: config.label.as_deref(),
            size,
            mip_level_count,
            sample_count: config.sample_count,
            dimension: config.dimension,
            format: config.format,
            usage,
            view_formats: &[],
        });

        queue.write_texture(
            wgpu::ImageCopyTexture {
                aspect: wgpu::TextureAspect::All,
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            &data[..expected],
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * width),
                rows_per_image: Some(height),
            },
            size,
        );

        if config.generate_mipmaps && mip_level_count > 1 {
            let rgba = ImageBuffer::<Rgba<u8>, _>::from_raw(width, height, data.to_vec())
                .context("Failed to create image buffer from raw data")?;
            generate_mipmaps_cpu(device, queue, &texture, &rgba, width, height, mip_level_count);
        }

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: config.label.as_deref(),
            address_mode_u: config.address_mode,
            address_mode_v: config.address_mode,
            address_mode_w: config.address_mode,
            mag_filter: config.mag_filter,
            min_filter: config.min_filter,
            mipmap_filter: config.mipmap_filter,
            lod_min_clamp: config.lod_min_clamp,
            lod_max_clamp: config.lod_max_clamp,
            compare: config.comparison,
            anisotropy_clamp: config.anisotropy_clamp,
            border_color: config.border_color,
        });

        Ok(Self {
            texture,
            view,
            sampler,
            size,
            format: config.format,
            mip_level_count,
        })
    }

    // ── Construction: flat colour ────────────────────────────────────────

    /// 1×1 solid-colour texture – useful for missing-texture fallbacks or tinting.
    pub fn from_color(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        color: [u8; 4],
        label: &str,
    ) -> Result<Self> {
        Self::from_rgba(
            device,
            queue,
            &color,
            1,
            1,
            TextureConfig {
                generate_mipmaps: false,
                ..TextureConfig::default().with_label(label)
            },
        )
    }

    /// White 1×1 fallback texture.
    pub fn white(device: &wgpu::Device, queue: &wgpu::Queue) -> Result<Self> {
        Self::from_color(device, queue, [255, 255, 255, 255], "white_1x1")
    }

    /// Black 1×1 fallback texture.
    pub fn black(device: &wgpu::Device, queue: &wgpu::Queue) -> Result<Self> {
        Self::from_color(device, queue, [0, 0, 0, 255], "black_1x1")
    }

    /// Flat normal map (pointing straight up, [128,128,255,255]).
    pub fn flat_normal(device: &wgpu::Device, queue: &wgpu::Queue) -> Result<Self> {
        let data: [u8; 4] = [128, 128, 255, 255];
        Self::from_rgba(
            device,
            queue,
            &data,
            1,
            1,
            TextureConfig::normal_map().with_label("flat_normal_1x1"),
        )
    }

    // ── Construction: depth / stencil ────────────────────────────────────

    pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

    pub fn create_depth_texture(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        label: &str,
    ) -> Self {
        Self::create_depth_texture_msaa(device, width, height, 1, label)
    }

    pub fn create_depth_texture_msaa(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        sample_count: u32,
        label: &str,
    ) -> Self {
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size,
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format: Self::DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some(label),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            compare: Some(wgpu::CompareFunction::LessEqual),
            lod_min_clamp: 0.0,
            lod_max_clamp: 100.0,
            ..Default::default()
        });

        Self {
            texture,
            view,
            sampler,
            size,
            format: Self::DEPTH_FORMAT,
            mip_level_count: 1,
        }
    }

    // ── Construction: empty render target ────────────────────────────────

    pub fn create_render_target(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        label: &str,
    ) -> Self {
        Self::create_render_target_msaa(device, width, height, format, 1, label)
    }

    pub fn create_render_target_msaa(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        sample_count: u32,
        label: &str,
    ) -> Self {
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size,
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some(label),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        Self {
            texture,
            view,
            sampler,
            size,
            format,
            mip_level_count: 1,
        }
    }

    // ── Construction: cubemap ────────────────────────────────────────────

    /// Build a cubemap from 6 face images (order: +X, -X, +Y, -Y, +Z, -Z).
    pub fn create_cubemap(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        faces: [&[u8]; 6],
        label: &str,
    ) -> Result<Self> {
        let first = image::load_from_memory(faces[0]).context("Failed to decode cubemap face 0")?;
        let (face_w, face_h) = first.dimensions();
        ensure!(face_w == face_h, "Cubemap faces must be square ({face_w}×{face_h})");

        let size = wgpu::Extent3d {
            width: face_w,
            height: face_h,
            depth_or_array_layers: 6,
        };

        let mip_level_count = Self::max_mip_levels(face_w, face_h);

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size,
            mip_level_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        for (i, face_bytes) in faces.iter().enumerate() {
            let img = if i == 0 {
                first.clone()
            } else {
                image::load_from_memory(face_bytes)
                    .with_context(|| format!("Failed to decode cubemap face {i}"))?
            };
            let rgba = img.to_rgba8();

            queue.write_texture(
                wgpu::ImageCopyTexture {
                    aspect: wgpu::TextureAspect::All,
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: 0,
                        z: i as u32,
                    },
                },
                &rgba,
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * face_w),
                    rows_per_image: Some(face_h),
                },
                wgpu::Extent3d {
                    width: face_w,
                    height: face_h,
                    depth_or_array_layers: 1,
                },
            );
        }

        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::Cube),
            array_layer_count: Some(6),
            ..Default::default()
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some(label),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        Ok(Self {
            texture,
            view,
            sampler,
            size,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            mip_level_count,
        })
    }

    // ── Dynamic update ───────────────────────────────────────────────────

    /// Write new RGBA data into mip-level 0 of an existing texture.
    pub fn update(
        &self,
        queue: &wgpu::Queue,
        data: &[u8],
        region: Option<TextureRegion>,
    ) {
        let (origin, extent) = match region {
            Some(r) => (
                wgpu::Origin3d {
                    x: r.x,
                    y: r.y,
                    z: 0,
                },
                wgpu::Extent3d {
                    width: r.width,
                    height: r.height,
                    depth_or_array_layers: 1,
                },
            ),
            None => (wgpu::Origin3d::ZERO, self.size),
        };

        let bpp = Self::bytes_per_pixel(self.format).max(4);

        queue.write_texture(
            wgpu::ImageCopyTexture {
                aspect: wgpu::TextureAspect::All,
                texture: &self.texture,
                mip_level: 0,
                origin,
            },
            data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(bpp * extent.width),
                rows_per_image: Some(extent.height),
            },
            extent,
        );
    }

    // ── Bind-group helpers ───────────────────────────────────────────────

    /// Creates the standard bind-group layout for a texture + sampler pair.
    pub fn bind_group_layout(device: &wgpu::Device, label: &str) -> wgpu::BindGroupLayout {
        Self::bind_group_layout_filtered(
            device,
            label,
            wgpu::TextureSampleType::Float { filterable: true },
            wgpu::TextureViewDimension::D2,
            wgpu::SamplerBindingType::Filtering,
        )
    }

    pub fn bind_group_layout_filtered(
        device: &wgpu::Device,
        label: &str,
        sample_type: wgpu::TextureSampleType,
        view_dimension: wgpu::TextureViewDimension,
        sampler_type: wgpu::SamplerBindingType,
    ) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(label),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension,
                        sample_type,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(sampler_type),
                    count: None,
                },
            ],
        })
    }

    pub fn bind_group_layout_depth(device: &wgpu::Device, label: &str) -> wgpu::BindGroupLayout {
        Self::bind_group_layout_filtered(
            device,
            label,
            wgpu::TextureSampleType::Depth,
            wgpu::TextureViewDimension::D2,
            wgpu::SamplerBindingType::Comparison,
        )
    }

    pub fn bind_group_layout_cubemap(device: &wgpu::Device, label: &str) -> wgpu::BindGroupLayout {
        Self::bind_group_layout_filtered(
            device,
            label,
            wgpu::TextureSampleType::Float { filterable: true },
            wgpu::TextureViewDimension::Cube,
            wgpu::SamplerBindingType::Filtering,
        )
    }

    /// Creates a bind-group from this texture bound to the given layout.
    pub fn create_bind_group(
        &self,
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        label: &str,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        })
    }

    // ── GPU readback ─────────────────────────────────────────────────────

    /// Read texture contents back to CPU as a `Vec<u8>`.
    /// Requires the texture to have `COPY_SRC` usage. This blocks.
    pub fn read_pixels(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Vec<u8>> {
        let bpp = Self::bytes_per_pixel(self.format);
        ensure!(bpp > 0, "Cannot read back compressed texture format");

        let unpadded_row = bpp * self.size.width;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_row = (unpadded_row + align - 1) / align * align;
        let buffer_size = (padded_row * self.size.height) as u64;

        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("texture_readback_staging"),
            size: buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("texture_readback_encoder"),
        });

        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                aspect: wgpu::TextureAspect::All,
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            wgpu::ImageCopyBuffer {
                buffer: &staging,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row),
                    rows_per_image: Some(self.size.height),
                },
            },
            self.size,
        );

        queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .context("Channel closed before map completed")??;

        let mapped = slice.get_mapped_range();
        let mut output = Vec::with_capacity((unpadded_row * self.size.height) as usize);

        for row in 0..self.size.height {
            let start = (row * padded_row) as usize;
            let end = start + unpadded_row as usize;
            output.extend_from_slice(&mapped[start..end]);
        }

        drop(mapped);
        staging.unmap();

        Ok(output)
    }

    /// Read back and save as a PNG file.
    pub fn save_png(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        path: impl AsRef<Path>,
    ) -> Result<()> {
        let pixels = self.read_pixels(device, queue)?;
        let img: ImageBuffer<Rgba<u8>, _> =
            ImageBuffer::from_raw(self.size.width, self.size.height, pixels)
                .context("Failed to construct image buffer for save")?;
        img.save(path.as_ref())
            .with_context(|| format!("Failed to save PNG to {}", path.as_ref().display()))?;
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Sub-region descriptor
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
pub struct TextureRegion {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

// ─────────────────────────────────────────────────────────────────────────────
// Texture atlas
// ─────────────────────────────────────────────────────────────────────────────

/// Simple grid-based texture atlas for sprite sheets / glyph maps.
pub struct TextureAtlas {
    pub texture: Texture,
    pub columns: u32,
    pub rows: u32,
    cell_width: f32,
    cell_height: f32,
}

impl TextureAtlas {
    pub fn new(texture: Texture, columns: u32, rows: u32) -> Self {
        let cell_width = 1.0 / columns as f32;
        let cell_height = 1.0 / rows as f32;
        Self {
            texture,
            columns,
            rows,
            cell_width,
            cell_height,
        }
    }

    /// Returns (u_min, v_min, u_max, v_max) UV coordinates for a cell index.
    #[inline]
    pub fn uv_rect(&self, index: u32) -> [f32; 4] {
        let col = index % self.columns;
        let row = index / self.columns;
        let u_min = col as f32 * self.cell_width;
        let v_min = row as f32 * self.cell_height;
        [u_min, v_min, u_min + self.cell_width, v_min + self.cell_height]
    }

    /// Returns (u_min, v_min, u_max, v_max) UV coordinates for a (col, row) pair.
    #[inline]
    pub fn uv_rect_at(&self, col: u32, row: u32) -> [f32; 4] {
        let u_min = col as f32 * self.cell_width;
        let v_min = row as f32 * self.cell_height;
        [u_min, v_min, u_min + self.cell_width, v_min + self.cell_height]
    }

    #[inline]
    pub fn cell_count(&self) -> u32 {
        self.columns * self.rows
    }

    #[inline]
    pub fn cell_pixel_size(&self) -> (u32, u32) {
        (
            self.texture.width() / self.columns,
            self.texture.height() / self.rows,
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Texture cache / manager
// ─────────────────────────────────────────────────────────────────────────────

/// Caches loaded textures by key to avoid redundant GPU uploads.
pub struct TextureCache {
    entries: HashMap<String, Arc<Texture>>,
}

impl TextureCache {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    pub fn get(&self, key: &str) -> Option<Arc<Texture>> {
        self.entries.get(key).cloned()
    }

    pub fn insert(&mut self, key: impl Into<String>, texture: Texture) -> Arc<Texture> {
        let arc = Arc::new(texture);
        self.entries.insert(key.into(), Arc::clone(&arc));
        arc
    }

    /// Loads from bytes only if not already cached.
    pub fn load_bytes(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        key: &str,
        bytes: &[u8],
        config: TextureConfig,
    ) -> Result<Arc<Texture>> {
        if let Some(existing) = self.entries.get(key) {
            return Ok(Arc::clone(existing));
        }
        let tex = Texture::from_bytes_with_config(device, queue, bytes, config)?;
        Ok(self.insert(key, tex))
    }

    /// Loads from a file path only if not already cached.
    pub fn load_path(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        path: impl AsRef<Path>,
        config: TextureConfig,
    ) -> Result<Arc<Texture>> {
        let key = path.as_ref().to_string_lossy().to_string();
        if let Some(existing) = self.entries.get(&key) {
            return Ok(Arc::clone(existing));
        }
        let tex = Texture::from_path_with_config(device, queue, &path, config)?;
        Ok(self.insert(key, tex))
    }

    pub fn remove(&mut self, key: &str) -> Option<Arc<Texture>> {
        self.entries.remove(key)
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Default for TextureCache {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CPU mipmap generation (avoids extra render passes / blit shaders)
// ─────────────────────────────────────────────────────────────────────────────

fn generate_mipmaps_cpu(
    _device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    base: &ImageBuffer<Rgba<u8>, Vec<u8>>,
    mut width: u32,
    mut height: u32,
    levels: u32,
) {
    let mut src = base.clone();

    for level in 1..levels {
        let new_w = (width / 2).max(1);
        let new_h = (height / 2).max(1);

        let dst = image::imageops::resize(&src, new_w, new_h, image::imageops::FilterType::Lanczos3);

        queue.write_texture(
            wgpu::ImageCopyTexture {
                aspect: wgpu::TextureAspect::All,
                texture,
                mip_level: level,
                origin: wgpu::Origin3d::ZERO,
            },
            &dst,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * new_w),
                rows_per_image: Some(new_h),
            },
            wgpu::Extent3d {
                width: new_w,
                height: new_h,
                depth_or_array_layers: 1,
            },
        );

        src = dst;
        width = new_w;
        height = new_h;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Alpha premultiplication
// ─────────────────────────────────────────────────────────────────────────────

fn premultiply_alpha_inplace(img: &mut ImageBuffer<Rgba<u8>, Vec<u8>>) {
    // Process 4 bytes at a time; the compiler will auto-vectorise this
    let buf = img.as_mut();
    let len = buf.len();
    let mut i = 0;
    while i + 3 < len {
        let a = buf[i + 3] as u16;
        buf[i]     = ((buf[i]     as u16 * a + 128) / 255) as u8;
        buf[i + 1] = ((buf[i + 1] as u16 * a + 128) / 255) as u8;
        buf[i + 2] = ((buf[i + 2] as u16 * a + 128) / 255) as u8;
        i += 4;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Procedural texture generators
// ─────────────────────────────────────────────────────────────────────────────

/// Utility functions to create common procedural textures on the CPU and
/// upload them in a single call.
pub struct TextureGenerator;

impl TextureGenerator {
    /// Checkerboard pattern – great for debugging UV mapping.
    pub fn checkerboard(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        size: u32,
        tile_size: u32,
        color_a: [u8; 4],
        color_b: [u8; 4],
        label: &str,
    ) -> Result<Texture> {
        let mut data = vec![0u8; (size * size * 4) as usize];
        for y in 0..size {
            for x in 0..size {
                let checker = ((x / tile_size) + (y / tile_size)) % 2 == 0;
                let color = if checker { color_a } else { color_b };
                let idx = ((y * size + x) * 4) as usize;
                data[idx..idx + 4].copy_from_slice(&color);
            }
        }
        Texture::from_rgba(
            device,
            queue,
            &data,
            size,
            size,
            TextureConfig::default().with_label(label),
        )
    }

    /// Gradient from `color_a` (left) to `color_b` (right).
    pub fn horizontal_gradient(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        color_a: [u8; 4],
        color_b: [u8; 4],
        label: &str,
    ) -> Result<Texture> {
        let mut data = vec![0u8; (width * height * 4) as usize];
        for y in 0..height {
            for x in 0..width {
                let t = x as f32 / (width - 1).max(1) as f32;
                let idx = ((y * width + x) * 4) as usize;
                for c in 0..4 {
                    data[idx + c] = (color_a[c] as f32 * (1.0 - t) + color_b[c] as f32 * t) as u8;
                }
            }
        }
        Texture::from_rgba(
            device,
            queue,
            &data,
            width,
            height,
            TextureConfig::default().with_label(label),
        )
    }

    /// Perlin-style noise (simple white noise – replace with a real noise
    /// function for production use).
    pub fn noise(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        size: u32,
        seed: u64,
        label: &str,
    ) -> Result<Texture> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut data = vec![0u8; (size * size * 4) as usize];
        for y in 0..size {
            for x in 0..size {
                let mut hasher = DefaultHasher::new();
                (x, y, seed).hash(&mut hasher);
                let h = hasher.finish();
                let v = (h & 0xFF) as u8;
                let idx = ((y * size + x) * 4) as usize;
                data[idx] = v;
                data[idx + 1] = v;
                data[idx + 2] = v;
                data[idx + 3] = 255;
            }
        }
        Texture::from_rgba(
            device,
            queue,
            &data,
            size,
            size,
            TextureConfig {
                srgb: false,
                format: wgpu::TextureFormat::Rgba8Unorm,
                ..TextureConfig::default().with_label(label)
            },
        )
    }
}
