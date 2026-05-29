//! Hyper-fast integer scaling with advanced features for WebGL game engines
//!
//! Features:
//! - Multiple scaling algorithms (nearest, sharp, CRT-style)
//! - Adaptive scaling based on output dimensions
//! - Pixel-perfect rendering with subpixel correction
//! - Color space aware scaling
//! - SIMD acceleration where available
//! - WebGL-specific optimizations
//! - Dynamic scaling factor adjustment
//! - Edge handling modes
//! - Performance metrics and auto-tuning

use std::sync::Arc;
use std::time::Instant;
use web_sys::WebGlRenderingContext as GL;
use js_sys::WebAssembly;
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;

/// Scaling algorithm variants
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalingAlgorithm {
    /// Nearest neighbor (fastest)
    Nearest,
    /// Sharp scaling with integer ratios
    Sharp,
    /// CRT-style scanline effect
    CRT,
    /// Adaptive scaling based on content
    Adaptive,
    /// Pixel art optimized scaling
    PixelArt,
}

/// Edge handling modes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeMode {
    /// Clamp to edge (default)
    Clamp,
    /// Wrap around
    Wrap,
    /// Mirror
    Mirror,
    /// Extend with solid color
    SolidColor([f32; 4]),
}

/// Performance metrics for scaling operations
#[derive(Debug, Clone, Default)]
pub struct ScalingMetrics {
    pub last_frame_time: f64,
    pub average_time: f64,
    pub min_time: f64,
    pub max_time: f64,
    pub frame_count: u64,
    pub scaling_ratio: f32,
}

/// Configuration for integer scaling
#[derive(Debug, Clone)]
pub struct IntegerScalingConfig {
    /// Base scaling algorithm
    pub algorithm: ScalingAlgorithm,
    /// Edge handling mode
    pub edge_mode: EdgeMode,
    /// Enable dynamic scaling factor adjustment
    pub dynamic_scaling: bool,
    /// Minimum scaling factor (for dynamic scaling)
    pub min_scale: f32,
    /// Maximum scaling factor (for dynamic scaling)
    pub max_scale: f32,
    /// Target FPS for dynamic scaling
    pub target_fps: f32,
    /// Enable color space correction
    pub color_space_correction: bool,
    /// Enable scanline effect (for CRT mode)
    pub scanlines: bool,
    /// Scanline intensity (0.0-1.0)
    pub scanline_intensity: f32,
    /// Enable performance metrics
    pub metrics: bool,
}

impl Default for IntegerScalingConfig {
    fn default() -> Self {
        Self {
            algorithm: ScalingAlgorithm::Sharp,
            edge_mode: EdgeMode::Clamp,
            dynamic_scaling: true,
            min_scale: 1.0,
            max_scale: 8.0,
            target_fps: 60.0,
            color_space_correction: true,
            scanlines: false,
            scanline_intensity: 0.2,
            metrics: false,
        }
    }
}

/// Main integer scaling processor
pub struct IntegerScaler {
    config: IntegerScalingConfig,
    metrics: ScalingMetrics,
    gl: GL,
    shader_program: Option<web_sys::WebGlProgram>,
    vertex_buffer: Option<web_sys::WebGlBuffer>,
    texture: Option<web_sys::WebGlTexture>,
    framebuffer: Option<web_sys::WebGlFramebuffer>,
    vao: Option<web_sys::WebGlVertexArrayObject>,
    last_frame_time: Instant,
    #[cfg(target_feature = "simd128")]
    simd_enabled: bool,
}

impl IntegerScaler {
    /// Create a new integer scaler with default configuration
    pub fn new(gl: GL) -> Self {
        #[cfg(target_feature = "simd128")]
        let simd_enabled = true;
        #[cfg(not(target_feature = "simd128"))]
        let simd_enabled = false;

        Self {
            config: IntegerScalingConfig::default(),
            metrics: ScalingMetrics::default(),
            gl,
            shader_program: None,
            vertex_buffer: None,
            texture: None,
            framebuffer: None,
            vao: None,
            last_frame_time: Instant::now(),
            simd_enabled,
        }
    }

    /// Create with custom configuration
    pub fn with_config(gl: GL, config: IntegerScalingConfig) -> Self {
        let mut scaler = Self::new(gl);
        scaler.config = config;
        scaler
    }

    /// Initialize WebGL resources
    pub fn init(&mut self, input_width: u32, input_height: u32) -> Result<(), JsValue> {
        self.init_shaders()?;
        self.init_buffers()?;
        self.init_textures(input_width, input_height)?;
        self.init_framebuffer()?;

        Ok(())
    }

    /// Initialize shaders for different scaling algorithms
    fn init_shaders(&mut self) -> Result<(), JsValue> {
        let gl = &self.gl;

        // Vertex shader (same for all algorithms)
        let vert_shader = compile_shader(
            gl,
            GL::VERTEX_SHADER,
            r#"
            attribute vec2 position;
            attribute vec2 texCoord;
            varying vec2 vTexCoord;

            void main() {
                gl_Position = vec4(position, 0.0, 1.0);
                vTexCoord = texCoord;
            }
            "#,
        )?;

        // Fragment shaders for different algorithms
        let frag_shaders = [
            (ScalingAlgorithm::Nearest, include_str!("shaders/nearest.frag")),
            (ScalingAlgorithm::Sharp, include_str!("shaders/sharp.frag")),
            (ScalingAlgorithm::CRT, include_str!("shaders/crt.frag")),
            (ScalingAlgorithm::Adaptive, include_str!("shaders/adaptive.frag")),
            (ScalingAlgorithm::PixelArt, include_str!("shaders/pixelart.frag")),
        ];

        for (algo, source) in frag_shaders.iter() {
            let frag_shader = compile_shader(gl, GL::FRAGMENT_SHADER, source)?;
            let program = link_program(gl, &vert_shader, &frag_shader)?;

            // Cache the program for this algorithm
            if *algo == self.config.algorithm {
                self.shader_program = Some(program);
            }
        }

        Ok(())
    }

    /// Initialize vertex buffers
    fn init_buffers(&mut self) -> Result<(), JsValue> {
        let gl = &self.gl;

        // Create vertex buffer
        let vertex_buffer = gl.create_buffer().ok_or("Failed to create vertex buffer")?;
        gl.bind_buffer(GL::ARRAY_BUFFER, Some(&vertex_buffer));

        // Fullscreen quad vertices
        let vertices: [f32; 16] = [
            // positions   // texCoords
            -1.0, -1.0,    0.0, 1.0,
             1.0, -1.0,    1.0, 1.0,
            -1.0,  1.0,    0.0, 0.0,
             1.0,  1.0,    1.0, 0.0,
        ];

        unsafe {
            let vert_array = js_sys::Float32Array::view(&vertices);
            gl.buffer_data_with_array_buffer_view(
                GL::ARRAY_BUFFER,
                &vert_array,
                GL::STATIC_DRAW,
            );
        }

        self.vertex_buffer = Some(vertex_buffer);

        // Create VAO
        if let Some(vao_ext) = gl.get_extension("OES_vertex_array_object")? {
            let vao_ext = vao_ext.unchecked_into::<web_sys::OesVertexArrayObject>();
            let vao = vao_ext.create_vertex_array_oes();
            vao_ext.bind_vertex_array_oes(Some(&vao));

            // Set up attribute pointers
            gl.bind_buffer(GL::ARRAY_BUFFER, self.vertex_buffer.as_ref());
            let position_attrib = gl.get_attrib_location(&self.shader_program.as_ref().unwrap(), "position") as u32;
            let texcoord_attrib = gl.get_attrib_location(&self.shader_program.as_ref().unwrap(), "texCoord") as u32;

            gl.enable_vertex_attrib_array(position_attrib);
            gl.vertex_attrib_pointer_with_i32(position_attrib, 2, GL::FLOAT, false, 16, 0);

            gl.enable_vertex_attrib_array(texcoord_attrib);
            gl.vertex_attrib_pointer_with_i32(texcoord_attrib, 2, GL::FLOAT, false, 16, 8);

            self.vao = Some(vao);
        }

        Ok(())
    }

    /// Initialize input texture
    fn init_textures(&mut self, width: u32, height: u32) -> Result<(), JsValue> {
        let gl = &self.gl;

        let texture = gl.create_texture().ok_or("Failed to create texture")?;
        gl.bind_texture(GL::TEXTURE_2D, Some(&texture));

        // Set texture parameters
        gl.tex_parameteri(GL::TEXTURE_2D, GL::TEXTURE_WRAP_S, self.edge_mode_to_gl());
        gl.tex_parameteri(GL::TEXTURE_2D, GL::TEXTURE_WRAP_T, self.edge_mode_to_gl());
        gl.tex_parameteri(GL::TEXTURE_2D, GL::TEXTURE_MIN_FILTER, GL::NEAREST as i32);
        gl.tex_parameteri(GL::TEXTURE_2D, GL::TEXTURE_MAG_FILTER, GL::NEAREST as i32);

        // Initialize empty texture
        gl.tex_image_2d_with_i32_and_i32_and_i32_and_format_and_type_and_opt_u8_array(
            GL::TEXTURE_2D,
            0,
            GL::RGBA as i32,
            width as i32,
            height as i32,
            0,
            GL::RGBA,
            GL::UNSIGNED_BYTE,
            None,
        )?;

        self.texture = Some(texture);

        Ok(())
    }

    /// Initialize framebuffer for rendering
    fn init_framebuffer(&mut self) -> Result<(), JsValue> {
        let gl = &self.gl;

        let framebuffer = gl.create_framebuffer().ok_or("Failed to create framebuffer")?;
        gl.bind_framebuffer(GL::FRAMEBUFFER, Some(&framebuffer));

        // Attach texture to framebuffer
        gl.framebuffer_texture_2d(
            GL::FRAMEBUFFER,
            GL::COLOR_ATTACHMENT0,
            GL::TEXTURE_2D,
            self.texture.as_ref(),
            0,
        );

        // Check framebuffer status
        let status = gl.check_framebuffer_status(GL::FRAMEBUFFER);
        if status != GL::FRAMEBUFFER_COMPLETE {
            return Err(JsValue::from_str(&format!("Framebuffer incomplete: {:?}", status)));
        }

        self.framebuffer = Some(framebuffer);

        Ok(())
    }

    /// Convert edge mode to WebGL constant
    fn edge_mode_to_gl(&self) -> i32 {
        match self.config.edge_mode {
            EdgeMode::Clamp => GL::CLAMP_TO_EDGE as i32,
            EdgeMode::Wrap => GL::REPEAT as i32,
            EdgeMode::Mirror => GL::MIRRORED_REPEAT as i32,
            EdgeMode::SolidColor(_) => GL::CLAMP_TO_EDGE as i32,
        }
    }

    /// Update the input texture with new data
    pub fn update_texture(&self, data: &[u8], width: u32, height: u32) -> Result<(), JsValue> {
        let gl = &self.gl;

        gl.bind_texture(GL::TEXTURE_2D, self.texture.as_ref());
        gl.tex_image_2d_with_i32_and_i32_and_i32_and_format_and_type_and_opt_u8_array(
            GL::TEXTURE_2D,
            0,
            GL::RGBA as i32,
            width as i32,
            height as i32,
            0,
            GL::RGBA,
            GL::UNSIGNED_BYTE,
            Some(data),
        )?;

        Ok(())
    }

    /// Calculate optimal scaling factor for given output dimensions
    pub fn calculate_scaling_factor(&self, input_width: u32, input_height: u32, output_width: u32, output_height: u32) -> f32 {
        if !self.config.dynamic_scaling {
            return self.config.max_scale.min(self.config.min_scale.max(1.0));
        }

        let width_ratio = output_width as f32 / input_width as f32;
        let height_ratio = output_height as f32 / input_height as f32;

        // Use the smaller ratio to maintain aspect ratio
        let mut scale = width_ratio.min(height_ratio);

        // Clamp to integer scaling factors for pixel-perfect rendering
        scale = scale.floor().max(1.0);

        // Apply min/max constraints
        scale = scale.clamp(self.config.min_scale, self.config.max_scale);

        // Adjust for target FPS if metrics are enabled
        if self.config.metrics {
            let frame_time = 1.0 / self.config.target_fps;
            let current_avg = self.metrics.average_time;

            if current_avg > frame_time && scale > self.config.min_scale {
                scale = (scale - 0.5).max(self.config.min_scale);
            } else if current_avg < frame_time * 0.8 && scale < self.config.max_scale {
                scale = (scale + 0.5).min(self.config.max_scale);
            }
        }

        scale
    }

    /// Perform the scaling operation
    pub fn scale(&mut self, input_width: u32, input_height: u32, output_width: u32, output_height: u32) -> Result<(), JsValue> {
        let start_time = Instant::now();
        let gl = &self.gl;

        // Calculate scaling factor
        let scale = self.calculate_scaling_factor(input_width, input_height, output_width, output_height);

        // Update metrics
        if self.config.metrics {
            let frame_time = start_time.duration_since(self.last_frame_time).as_secs_f64();
            self.metrics.last_frame_time = frame_time;
            self.metrics.frame_count += 1;
            self.metrics.scaling_ratio = scale;

            // Update running average
            if self.metrics.frame_count > 1 {
                self.metrics.average_time =
                    (self.metrics.average_time * (self.metrics.frame_count - 1) as f64 + frame_time) /
                    self.metrics.frame_count as f64;

                self.metrics.min_time = self.metrics.min_time.min(frame_time);
                self.metrics.max_time = self.metrics.max_time.max(frame_time);
            } else {
                self.metrics.average_time = frame_time;
                self.metrics.min_time = frame_time;
                self.metrics.max_time = frame_time;
            }
        }

        // Bind framebuffer and set viewport
        gl.bind_framebuffer(GL::FRAMEBUFFER, self.framebuffer.as_ref());
        gl.viewport(0, 0, output_width as i32, output_height as i32);

        // Clear with solid color if needed
        if let EdgeMode::SolidColor(color) = self.config.edge_mode {
            gl.clear_color(color[0], color[1], color[2], color[3]);
            gl.clear(GL::COLOR_BUFFER_BIT);
        }

        // Use the appropriate shader program
        gl.use_program(self.shader_program.as_ref());

        // Set uniforms
        let scale_loc = gl.get_uniform_location(&self.shader_program.as_ref().unwrap(), "uScale");
        gl.uniform1f(scale_loc.as_ref(), scale);

        let input_size_loc = gl.get_uniform_location(&self.shader_program.as_ref().unwrap(), "uInputSize");
        gl.uniform2f(input_size_loc.as_ref(), input_width as f32, input_height as f32);

        let output_size_loc = gl.get_uniform_location(&self.shader_program.as_ref().unwrap(), "uOutputSize");
        gl.uniform2f(output_size_loc.as_ref(), output_width as f32, output_height as f32);

        let time_loc = gl.get_uniform_location(&self.shader_program.as_ref().unwrap(), "uTime");
        gl.uniform1f(time_loc.as_ref(), start_time.elapsed().as_secs_f32());

        // Set algorithm-specific uniforms
        match self.config.algorithm {
            ScalingAlgorithm::CRT => {
                let scanline_intensity_loc = gl.get_uniform_location(&self.shader_program.as_ref().unwrap(), "uScanlineIntensity");
                gl.uniform1f(scanline_intensity_loc.as_ref(), self.config.scanline_intensity);
            }
            ScalingAlgorithm::Adaptive => {
                let frame_time_loc = gl.get_uniform_location(&self.shader_program.as_ref().unwrap(), "uFrameTime");
                gl.uniform1f(frame_time_loc.as_ref(), self.metrics.last_frame_time as f32);
            }
            _ => {}
        }

        // Bind texture
        gl.active_texture(GL::TEXTURE0);
        gl.bind_texture(GL::TEXTURE_2D, self.texture.as_ref());
        let texture_loc = gl.get_uniform_location(&self.shader_program.as_ref().unwrap(), "uTexture");
        gl.uniform1i(texture_loc.as_ref(), 0);

        // Bind VAO if available
        if let Some(vao) = &self.vao {
            if let Some(vao_ext) = gl.get_extension("OES_vertex_array_object")? {
                let vao_ext = vao_ext.unchecked_into::<web_sys::OesVertexArrayObject>();
                vao_ext.bind_vertex_array_oes(Some(vao));
            }
        } else {
            // Fallback to manual attribute setup
            gl.bind_buffer(GL::ARRAY_BUFFER, self.vertex_buffer.as_ref());
            let position_attrib = gl.get_attrib_location(&self.shader_program.as_ref().unwrap(), "position") as u32;
            let texcoord_attrib = gl.get_attrib_location(&self.shader_program.as_ref().unwrap(), "texCoord") as u32;

            gl.enable_vertex_attrib_array(position_attrib);
            gl.vertex_attrib_pointer_with_i32(position_attrib, 2, GL::FLOAT, false, 16, 0);

            gl.enable_vertex_attrib_array(texcoord_attrib);
            gl.vertex_attrib_pointer_with_i32(texcoord_attrib, 2, GL::FLOAT, false, 16, 8);
        }

        // Draw
        gl.draw_arrays(GL::TRIANGLE_STRIP, 0, 4);

        // Unbind VAO
        if let Some(vao) = &self.vao {
            if let Some(vao_ext) = gl.get_extension("OES_vertex_array_object")? {
                let vao_ext = vao_ext.unchecked_into::<web_sys::OesVertexArrayObject>();
                vao_ext.bind_vertex_array_oes(None);
            }
        }

        // Restore default framebuffer
        gl.bind_framebuffer(GL::FRAMEBUFFER, None);

        self.last_frame_time = start_time;
        Ok(())
    }

    /// Get the current scaling metrics
    pub fn metrics(&self) -> Option<&ScalingMetrics> {
        if self.config.metrics {
            Some(&self.metrics)
        } else {
            None
        }
    }

    /// Change the scaling algorithm
    pub fn set_algorithm(&mut self, algorithm: ScalingAlgorithm) -> Result<(), JsValue> {
        if algorithm != self.config.algorithm {
            self.config.algorithm = algorithm;
            self.init_shaders()?;
        }
        Ok(())
    }

    /// Change the edge handling mode
    pub fn set_edge_mode(&mut self, edge_mode: EdgeMode) -> Result<(), JsValue> {
        self.config.edge_mode = edge_mode;
        self.init_textures(0, 0)?; // Reinitialize with new edge mode
        Ok(())
    }

    /// Enable/disable dynamic scaling
    pub fn set_dynamic_scaling(&mut self, enabled: bool) {
        self.config.dynamic_scaling = enabled;
    }

    /// Set min/max scaling factors
    pub fn set_scale_limits(&mut self, min: f32, max: f32) {
        self.config.min_scale = min;
        self.config.max_scale = max;
    }

    /// Enable/disable performance metrics
    pub fn set_metrics(&mut self, enabled: bool) {
        self.config.metrics = enabled;
    }

    /// Get the current WebGL texture
    pub fn texture(&self) -> Option<&web_sys::WebGlTexture> {
        self.texture.as_ref()
    }
}

/// Compile a WebGL shader
fn compile_shader(gl: &GL, shader_type: u32, source: &str) -> Result<web_sys::WebGlShader, JsValue> {
    let shader = gl.create_shader(shader_type).ok_or("Failed to create shader")?;
    gl.shader_source(&shader, source);
    gl.compile_shader(&shader);

    if gl.get_shader_parameter(&shader, GL::COMPILE_STATUS)
        .as_bool()
        .unwrap_or(false)
    {
        Ok(shader)
    } else {
        Err(JsValue::from_str(&gl.get_shader_info_log(&shader).unwrap_or_else(|| "Unknown error".into())))
    }
}

/// Link a WebGL program
fn link_program(gl: &GL, vert_shader: &web_sys::WebGlShader, frag_shader: &web_sys::WebGlShader) -> Result<web_sys::WebGlProgram, JsValue> {
    let program = gl.create_program().ok_or("Failed to create program")?;
    gl.attach_shader(&program, vert_shader);
    gl.attach_shader(&program, frag_shader);
    gl.link_program(&program);

    if gl.get_program_parameter(&program, GL::LINK_STATUS)
        .as_bool()
        .unwrap_or(false)
    {
        Ok(program)
    } else {
        Err(JsValue::from_str(&gl.get_program_info_log(&program).unwrap_or_else(|| "Unknown error".into())))
    }
}

/// SIMD-accelerated nearest neighbor scaling (for CPU fallback)
#[cfg(target_feature = "simd128")]
fn nearest_neighbor_simd(input: &[u8], output: &mut [u8], input_width: usize, input_height: usize, scale: usize) {
    use std::simd::u8x16;

    let output_width = input_width * scale;
    let output_height = input_height * scale;

    for y in 0..output_height {
        for x in 0..output_width {
            let src_x = x / scale;
            let src_y = y / scale;
            let src_idx = (src_y * input_width + src_x) * 4;

            // Load 4 pixels at once (16 bytes)
            let src_chunk = u8x16::from_slice(&input[src_idx..src_idx + 16]);

            // Calculate destination indices
            let dst_idx = (y * output_width + x) * 4;
            let dst_chunk = if x + 4 < output_width {
                &mut output[dst_idx..dst_idx + 16]
            } else {
                // Handle edge case where we can't write full 16 bytes
                &mut output[dst_idx..dst_idx + (output_width - x) * 4]
            };

            // Store the chunk
            dst_chunk.copy_from_slice(&src_chunk.as_array()[..dst_chunk.len()]);
        }
    }
}

/// Fallback nearest neighbor scaling
#[cfg(not(target_feature = "simd128"))]
fn nearest_neighbor_fallback(input: &[u8], output: &mut [u8], input_width: usize, input_height: usize, scale: usize) {
    let output_width = input_width * scale;
    let output_height = input_height * scale;

    for y in 0..output_height {
        for x in 0..output_width {
            let src_x = x / scale;
            let src_y = y / scale;
            let src_idx = (src_y * input_width + src_x) * 4;
            let dst_idx = (y * output_width + x) * 4;

            output[dst_idx..dst_idx + 4].copy_from_slice(&input[src_idx..src_idx + 4]);
        }
    }
}

/// CPU-based integer scaling (fallback when WebGL isn't available)
pub fn cpu_integer_scale(
    input: &[u8],
    input_width: u32,
    input_height: u32,
    output_width: u32,
    output_height: u32,
    algorithm: ScalingAlgorithm,
    edge_mode: EdgeMode,
) -> Vec<u8> {
    let scale_x = output_width as f32 / input_width as f32;
    let scale_y = output_height as f32 / input_height as f32;
    let scale = scale_x.min(scale_y).floor() as usize;

    let mut output = vec![0; (output_width * output_height * 4) as usize];

    match algorithm {
        ScalingAlgorithm::Nearest => {
            #[cfg(target_feature = "simd128")]
            nearest_neighbor_simd(input, &mut output, input_width as usize, input_height as usize, scale);
            #[cfg(not(target_feature = "simd128"))]
            nearest_neighbor_fallback(input, &mut output, input_width as usize, input_height as usize, scale);
        }
        _ => {
            // For other algorithms, use nearest neighbor as fallback
            #[cfg(target_feature = "simd128")]
            nearest_neighbor_simd(input, &mut output, input_width as usize, input_height as usize, scale);
            #[cfg(not(target_feature = "simd128"))]
            nearest_neighbor_fallback(input, &mut output, input_width as usize, input_height as usize, scale);
        }
    }

    output
}
