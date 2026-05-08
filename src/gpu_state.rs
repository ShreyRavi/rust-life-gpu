use std::sync::Arc;
use wgpu::util::DeviceExt;
use crate::simulation::{GridConfig, SimState};

pub struct GpuState {
    pub surface:         wgpu::Surface<'static>,
    pub device:          wgpu::Device,
    pub queue:           wgpu::Queue,
    pub surface_config:  wgpu::SurfaceConfiguration,

    pub uniform_buf:     wgpu::Buffer,

    // Ping-pong state buffers: [A, B]
    state_bufs:          [wgpu::Buffer; 2],
    // pixel_texture: rgba8unorm, written by blit pass, sampled by render pass
    pixel_texture:       wgpu::Texture,
    pixel_texture_view:  wgpu::TextureView,
    cell_sampler:        wgpu::Sampler,

    // Compute pipeline: simulation
    sim_pipeline:        wgpu::ComputePipeline,
    // compute_bg[i]: reads state_bufs[i], writes state_bufs[1-i]
    compute_bg:          [wgpu::BindGroup; 2],

    // Blit pipeline: state buffer → pixel_texture
    blit_pipeline:       wgpu::ComputePipeline,
    // blit_bg[i]: reads state_bufs[i], writes pixel_texture
    blit_bg:             [wgpu::BindGroup; 2],

    // Display pipeline: pixel_texture → surface
    display_pipeline:    wgpu::RenderPipeline,
    display_bg:          wgpu::BindGroup,

    pub grid_width:      u32,
    pub grid_height:     u32,
}

impl GpuState {
    pub async fn new(
        window: Arc<winit::window::Window>,
        sim: &SimState,
    ) -> anyhow::Result<Self> {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = instance.create_surface(window)?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| anyhow::anyhow!(
                "No compatible GPU adapter found. Try Chrome 113+ on hardware with WebGPU support."
            ))?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await?;

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width:  size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        let (grid_width, grid_height) = (sim.config.width, sim.config.height);

        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("uniform"),
            contents: bytemuck::bytes_of(&sim.config),
            usage:    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Check storage buffer limit; warn if grid exceeds 128 MB
        let buf_bytes = (grid_width * grid_height * 4) as u64;
        let max_storage = device.limits().max_storage_buffer_binding_size as u64;
        if buf_bytes > max_storage {
            log::warn!(
                "State buffer {buf_bytes}B exceeds device limit {max_storage}B — grid may be clamped"
            );
        }

        let initial_state = sim.random_state();
        let state_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("state_a"),
            contents: bytemuck::cast_slice(&initial_state),
            usage:    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let state_b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("state_b"),
            contents: bytemuck::cast_slice(&vec![0u32; (grid_width * grid_height) as usize]),
            usage:    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let pixel_texture = device.create_texture(&wgpu::TextureDescriptor {
            label:           Some("pixel_texture"),
            size:            wgpu::Extent3d { width: grid_width, height: grid_height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count:    1,
            dimension:       wgpu::TextureDimension::D2,
            format:          wgpu::TextureFormat::Rgba8Unorm,
            usage:           wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats:    &[],
        });
        let pixel_texture_view = pixel_texture.create_view(&Default::default());

        let cell_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter:    wgpu::FilterMode::Nearest,
            min_filter:    wgpu::FilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        // ── Simulation compute pipeline ──────────────────────────────────────
        let sim_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("life.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/life.wgsl").into()),
        });

        let sim_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("sim_bgl"),
            entries: &[
                // uniform GridConfig
                wgpu::BindGroupLayoutEntry {
                    binding:    0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty:         wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
                // state_in (read)
                wgpu::BindGroupLayoutEntry {
                    binding:    1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty:         wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
                // state_out (read_write)
                wgpu::BindGroupLayoutEntry {
                    binding:    2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty:         wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
            ],
        });

        let sim_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:                Some("sim_pl"),
            bind_group_layouts:   &[&sim_bgl],
            push_constant_ranges: &[],
        });

        let sim_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label:       Some("sim"),
            layout:      Some(&sim_pipeline_layout),
            module:      &sim_shader,
            entry_point: "main",
            compilation_options: Default::default(),
            cache: None,
        });

        // compute_bg[0]: reads A, writes B
        // compute_bg[1]: reads B, writes A
        let compute_bg = [
            make_sim_bg(&device, &sim_bgl, &uniform_buf, &state_a, &state_b, "compute_bg[0]"),
            make_sim_bg(&device, &sim_bgl, &uniform_buf, &state_b, &state_a, "compute_bg[1]"),
        ];

        // ── Blit compute pipeline ────────────────────────────────────────────
        let blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("render_blit.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/render_blit.wgsl").into()),
        });

        let blit_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("blit_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding:    0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty:         wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding:    1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty:         wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding:    2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty:         wgpu::BindingType::StorageTexture {
                        access:       wgpu::StorageTextureAccess::WriteOnly,
                        format:       wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let blit_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:                Some("blit_pl"),
            bind_group_layouts:   &[&blit_bgl],
            push_constant_ranges: &[],
        });

        let blit_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label:       Some("blit"),
            layout:      Some(&blit_pipeline_layout),
            module:      &blit_shader,
            entry_point: "main",
            compilation_options: Default::default(),
            cache: None,
        });

        // blit_bg[i]: reads state_bufs[i], writes pixel_texture
        let blit_bg = [
            make_blit_bg(&device, &blit_bgl, &uniform_buf, &state_a, &pixel_texture_view, "blit_bg[0]"),
            make_blit_bg(&device, &blit_bgl, &uniform_buf, &state_b, &pixel_texture_view, "blit_bg[1]"),
        ];

        // ── Display render pipeline ──────────────────────────────────────────
        let display_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("display.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/display.wgsl").into()),
        });

        let display_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("display_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding:    0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty:         wgpu::BindingType::Texture {
                        sample_type:    wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled:   false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding:    1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty:         wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let display_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("display_bg"),
            layout:  &display_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&pixel_texture_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&cell_sampler) },
            ],
        });

        let display_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:                Some("display_pl"),
            bind_group_layouts:   &[&display_bgl],
            push_constant_ranges: &[],
        });

        let display_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("display"),
            layout: Some(&display_pipeline_layout),
            vertex: wgpu::VertexState {
                module:      &display_shader,
                entry_point: "vs_main",
                buffers:     &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module:      &display_shader,
                entry_point: "fs_main",
                targets:     &[Some(wgpu::ColorTargetState {
                    format:     surface_format,
                    blend:      Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive:    wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample:  wgpu::MultisampleState::default(),
            multiview:    None,
            cache:        None,
        });

        Ok(Self {
            surface,
            device,
            queue,
            surface_config,
            uniform_buf,
            state_bufs: [state_a, state_b],
            pixel_texture,
            pixel_texture_view,
            cell_sampler,
            sim_pipeline,
            compute_bg,
            blit_pipeline,
            blit_bg,
            display_pipeline,
            display_bg,
            grid_width,
            grid_height,
        })
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 { return; }
        self.surface_config.width  = new_size.width;
        self.surface_config.height = new_size.height;
        self.surface.configure(&self.device, &self.surface_config);
    }

    // Rebuild all grid-dependent GPU resources after a grid size change or reset.
    pub fn rebuild_grid(&mut self, sim: &SimState) {
        let (w, h) = (sim.config.width, sim.config.height);
        self.grid_width  = w;
        self.grid_height = h;

        self.queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&sim.config));

        let initial = sim.random_state();
        let zeros   = vec![0u32; (w * h) as usize];

        // Recreate state buffers at new size.
        let new_a = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("state_a"),
            contents: bytemuck::cast_slice(&initial),
            usage:    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let new_b = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("state_b"),
            contents: bytemuck::cast_slice(&zeros),
            usage:    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // Recreate pixel_texture at new dimensions.
        let new_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label:           Some("pixel_texture"),
            size:            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count:    1,
            dimension:       wgpu::TextureDimension::D2,
            format:          wgpu::TextureFormat::Rgba8Unorm,
            usage:           wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats:    &[],
        });
        let new_texture_view = new_texture.create_view(&Default::default());

        // Rebuild bind groups — layouts are unchanged.
        let sim_bgl   = self.sim_pipeline.get_bind_group_layout(0);
        let blit_bgl  = self.blit_pipeline.get_bind_group_layout(0);
        let disp_bgl  = self.display_pipeline.get_bind_group_layout(0);

        self.compute_bg = [
            make_sim_bg(&self.device, &sim_bgl,  &self.uniform_buf, &new_a, &new_b, "compute_bg[0]"),
            make_sim_bg(&self.device, &sim_bgl,  &self.uniform_buf, &new_b, &new_a, "compute_bg[1]"),
        ];
        self.blit_bg = [
            make_blit_bg(&self.device, &blit_bgl, &self.uniform_buf, &new_a, &new_texture_view, "blit_bg[0]"),
            make_blit_bg(&self.device, &blit_bgl, &self.uniform_buf, &new_b, &new_texture_view, "blit_bg[1]"),
        ];
        self.display_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("display_bg"),
            layout:  &disp_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&new_texture_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.cell_sampler) },
            ],
        });

        self.state_bufs        = [new_a, new_b];
        self.pixel_texture      = new_texture;
        self.pixel_texture_view = new_texture_view;
    }

    // Upload just the uniform buffer (rule change, no buffer resize needed).
    pub fn update_uniform(&self, config: &GridConfig) {
        self.queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(config));
    }

    // Reset grid in-place: refill state_a with new random data, zero state_b.
    pub fn reset_state(&self, sim: &SimState) {
        let initial = sim.random_state();
        let zeros   = vec![0u32; (self.grid_width * self.grid_height) as usize];
        self.queue.write_buffer(&self.state_bufs[0], 0, bytemuck::cast_slice(&initial));
        self.queue.write_buffer(&self.state_bufs[1], 0, bytemuck::cast_slice(&zeros));
    }

    // Encode sim + blit + display passes. Returns encoder + view + surface output.
    // Caller adds egui pass, then calls queue.submit + output.present().
    // Splitting avoids the egui_renderer lifetime conflict with encoder.
    pub fn begin_frame(
        &self,
        sim: &SimState,
    ) -> Result<(wgpu::CommandEncoder, wgpu::TextureView, wgpu::SurfaceTexture), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view   = output.texture.create_view(&Default::default());

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("frame"),
        });

        let workgroups_x = (self.grid_width  + 7) / 8;
        let workgroups_y = (self.grid_height + 7) / 8;

        if !sim.is_paused {
            for i in 0..sim.steps_per_frame {
                let idx = ((sim.step_count + i as u64) % 2) as usize;
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("sim"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.sim_pipeline);
                cpass.set_bind_group(0, &self.compute_bg[idx], &[]);
                cpass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
            }
        }

        let blit_idx = ((sim.step_count
            + if sim.is_paused { 0 } else { sim.steps_per_frame as u64 }) % 2) as usize;
        {
            let mut bpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("blit"),
                timestamp_writes: None,
            });
            bpass.set_pipeline(&self.blit_pipeline);
            bpass.set_bind_group(0, &self.blit_bg[blit_idx], &[]);
            bpass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        // Display render pass — clear to dead-cell color then draw the grid texture
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("display"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view:           &view,
                    resolve_target: None,
                    ops:            wgpu::Operations {
                        load:  wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.039, g: 0.039, b: 0.059, a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set:      None,
                timestamp_writes:         None,
            });
            rpass.set_pipeline(&self.display_pipeline);
            rpass.set_bind_group(0, &self.display_bg, &[]);
            rpass.draw(0..3, 0..1);
        }

        Ok((encoder, view, output))
    }
}

fn make_sim_bg(
    device:      &wgpu::Device,
    layout:      &wgpu::BindGroupLayout,
    uniform_buf: &wgpu::Buffer,
    state_in:    &wgpu::Buffer,
    state_out:   &wgpu::Buffer,
    label:       &str,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label:   Some(label),
        layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniform_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: state_in.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: state_out.as_entire_binding() },
        ],
    })
}

fn make_blit_bg(
    device:       &wgpu::Device,
    layout:       &wgpu::BindGroupLayout,
    uniform_buf:  &wgpu::Buffer,
    state:        &wgpu::Buffer,
    texture_view: &wgpu::TextureView,
    label:        &str,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label:   Some(label),
        layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniform_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: state.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(texture_view) },
        ],
    })
}
