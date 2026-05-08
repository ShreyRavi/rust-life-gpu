use std::sync::Arc;
use web_time::Instant;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, KeyEvent, WindowEvent},
    event_loop::ActiveEventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};
use crate::{
    gpu_state::GpuState,
    simulation::{Rule, SimState},
};

const DEFAULT_GRID: u32 = 4096;
const GRID_OPTIONS: &[(u32, u32, &str)] = &[
    (512,  512,  "512×512"),
    (1024, 1024, "1024×1024"),
    (2048, 2048, "2048×2048"),
    (4096, 4096, "4096×4096"),
];

pub struct App {
    window:        Option<Arc<Window>>,
    gpu:           Option<GpuState>,
    egui_ctx:      egui::Context,
    egui_state:    Option<egui_winit::State>,
    egui_renderer: Option<egui_wgpu::Renderer>,

    pub sim:       SimState,
    generation:    u64,
    last_frame:    Instant,
    fps:           f32,
    frame_times:   Vec<f32>,

    pending_grid_size:   Option<(u32, u32)>,
    pending_reset:       bool,
    pending_rule_change: bool,
}

impl App {
    pub fn new() -> Self {
        let egui_ctx = egui::Context::default();
        egui_ctx.set_visuals(egui::Visuals::dark());

        Self {
            window:        None,
            gpu:           None,
            egui_ctx,
            egui_state:    None,
            egui_renderer: None,
            sim:           SimState::new(DEFAULT_GRID, DEFAULT_GRID),
            generation:    0,
            last_frame:    Instant::now(),
            fps:           0.0,
            frame_times:   Vec::with_capacity(60),
            pending_grid_size:   None,
            pending_reset:       false,
            pending_rule_change: false,
        }
    }

    pub fn init_gpu(&mut self, gpu: GpuState) {
        let surface_format = gpu.surface_config.format;
        let egui_renderer  = egui_wgpu::Renderer::new(&gpu.device, surface_format, None, 1, false);
        if let Some(window) = &self.window {
            let egui_state = egui_winit::State::new(
                self.egui_ctx.clone(),
                egui::ViewportId::ROOT,
                window.as_ref(),
                None,
                None,
                None,
            );
            self.egui_state = Some(egui_state);
        }
        self.egui_renderer = Some(egui_renderer);
        self.gpu = Some(gpu);
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        #[cfg(not(target_arch = "wasm32"))]
        let window_attrs = Window::default_attributes()
            .with_title("GPU Life")
            .with_inner_size(winit::dpi::LogicalSize::new(1280u32, 720u32));

        // On WASM, size the canvas to the actual viewport so the egui panel
        // (anchored to the right edge of the surface) stays on-screen.
        #[cfg(target_arch = "wasm32")]
        let window_attrs = {
            use winit::platform::web::WindowAttributesExtWebSys;
            let web_win = web_sys::window().expect("no window");
            let vw = web_win.inner_width().ok().and_then(|v| v.as_f64()).unwrap_or(1280.0) as u32;
            let vh = web_win.inner_height().ok().and_then(|v| v.as_f64()).unwrap_or(720.0) as u32;
            Window::default_attributes()
                .with_title("GPU Life")
                .with_inner_size(winit::dpi::LogicalSize::new(vw, vh))
                .with_append(true)
        };

        let window = Arc::new(
            event_loop.create_window(window_attrs).expect("create window")
        );
        self.window = Some(window.clone());

        #[cfg(not(target_arch = "wasm32"))]
        {
            match pollster::block_on(GpuState::new(window, &self.sim)) {
                Ok(gpu) => self.init_gpu(gpu),
                Err(e)  => {
                    log::error!("GPU init failed: {e}");
                    event_loop.exit();
                }
            }
        }

        #[cfg(target_arch = "wasm32")]
        {
            let sim_config = self.sim.config;
            let sim_density = self.sim.density;

            // Clone the egui context to move into the async block
            // We'll init egui_state in init_gpu when gpu resolves
            let window_clone = window.clone();

            // Use a channel-like approach: store result in a thread-local cell.
            // spawn_local is single-threaded on WASM so Rc is safe.
            use std::rc::Rc;
            use std::cell::RefCell;

            let pending: Rc<RefCell<Option<Result<GpuState, anyhow::Error>>>> =
                Rc::new(RefCell::new(None));
            let pending_clone = Rc::clone(&pending);

            // Store pending ref where we can poll it each frame
            // We use a thread-local for simplicity
            PENDING_GPU.with(|p| {
                *p.borrow_mut() = Some(pending);
            });

            wasm_bindgen_futures::spawn_local(async move {
                // Build a temporary SimState matching the current config
                let mut tmp_sim = SimState::new(sim_config.width, sim_config.height);
                tmp_sim.density = sim_density;

                let result = GpuState::new(window_clone, &tmp_sim).await;
                *pending_clone.borrow_mut() = Some(result);
            });
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id:        WindowId,
        event:      WindowEvent,
    ) {
        // Poll for pending GPU init on WASM
        #[cfg(target_arch = "wasm32")]
        if self.gpu.is_none() {
            PENDING_GPU.with(|p| {
                if let Some(pending) = &*p.borrow() {
                    if let Some(result) = pending.borrow_mut().take() {
                        match result {
                            Ok(gpu) => {
                                // Successful init — clear any retry counter
                                if let Some(win) = web_sys::window() {
                                    if let Ok(Some(s)) = win.session_storage() {
                                        let _ = s.remove_item("gpu_init_attempts");
                                    }
                                }
                                self.init_gpu(gpu);
                            }
                            Err(e) => {
                                log::error!("GPU init failed: {e}");
                                // Auto-reload up to 5 times; sessionStorage survives
                                // reload but not tab close, preventing infinite loops.
                                if let Some(win) = web_sys::window() {
                                    if let Ok(Some(storage)) = win.session_storage() {
                                        let attempts: u32 = storage
                                            .get_item("gpu_init_attempts")
                                            .ok().flatten()
                                            .and_then(|s| s.parse().ok())
                                            .unwrap_or(0);
                                        if attempts < 5 {
                                            let _ = storage.set_item(
                                                "gpu_init_attempts",
                                                &(attempts + 1).to_string(),
                                            );
                                            let _ = win.location().reload();
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            });
        }

        // Forward events to egui
        if let (Some(state), Some(window)) = (&mut self.egui_state, &self.window) {
            let response = state.on_window_event(window, &event);
            if response.consumed { return; }
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::KeyboardInput {
                event: KeyEvent {
                    physical_key: PhysicalKey::Code(KeyCode::Space),
                    state: ElementState::Pressed,
                    ..
                },
                ..
            } => {
                self.sim.is_paused = !self.sim.is_paused;
                if let Some(w) = &self.window { w.request_redraw(); }
            }

            WindowEvent::Resized(size) => {
                if let Some(gpu) = &mut self.gpu {
                    gpu.resize(size);
                }
            }

            WindowEvent::RedrawRequested => {
                if self.gpu.is_some() {
                    self.render_frame();
                }
                if let Some(w) = &self.window { w.request_redraw(); }
            }

            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(w) = &self.window { w.request_redraw(); }
    }
}

impl App {
    fn render_frame(&mut self) {
        let now = Instant::now();
        let dt  = now.duration_since(self.last_frame).as_secs_f32();
        self.last_frame = now;
        self.frame_times.push(dt);
        if self.frame_times.len() > 60 { self.frame_times.remove(0); }
        let avg = self.frame_times.iter().sum::<f32>() / self.frame_times.len() as f32;
        self.fps = if avg > 0.0 { 1.0 / avg } else { 0.0 };

        // Apply pending changes
        if let Some((w, h)) = self.pending_grid_size.take() {
            self.sim.config.width  = w;
            self.sim.config.height = h;
            self.sim.step_count    = 0;
            self.generation        = 0;
            if let Some(gpu) = &mut self.gpu {
                gpu.rebuild_grid(&self.sim);
            }
        }
        if self.pending_reset {
            self.pending_reset  = false;
            self.sim.step_count = 0;
            self.generation     = 0;
            if let Some(gpu) = &self.gpu { gpu.reset_state(&self.sim); }
        }
        if self.pending_rule_change {
            self.pending_rule_change = false;
            if let Some(gpu) = &self.gpu { gpu.update_uniform(&self.sim.config); }
        }

        let (Some(gpu), Some(egui_state), Some(egui_renderer), Some(window)) = (
            self.gpu.as_mut(),
            self.egui_state.as_mut(),
            self.egui_renderer.as_mut(),
            self.window.as_ref(),
        ) else {
            return;
        };

        let raw_input  = egui_state.take_egui_input(window);
        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            build_ui(
                ctx,
                &mut self.sim,
                self.fps,
                self.generation,
                &mut self.pending_grid_size,
                &mut self.pending_reset,
                &mut self.pending_rule_change,
            );
        });
        egui_state.handle_platform_output(window, full_output.platform_output.clone());

        let screen_desc = egui_wgpu::ScreenDescriptor {
            size_in_pixels:   [gpu.surface_config.width, gpu.surface_config.height],
            pixels_per_point: window.scale_factor() as f32,
        };

        // Tessellate egui shapes into paint jobs here (not inside gpu_state)
        let paint_jobs = self.egui_ctx.tessellate(
            full_output.shapes,
            screen_desc.pixels_per_point,
        );

        // begin_frame needs step_count to reflect steps done BEFORE this frame so that
        // the ping-pong bind group index is correct. Increment AFTER GPU submission.
        let frame_result = gpu.begin_frame(&self.sim);
        match frame_result {
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                let size = window.inner_size();
                gpu.resize(size);
                return;
            }
            Err(wgpu::SurfaceError::OutOfMemory) => { log::error!("OOM"); return; }
            Err(e) => { log::warn!("Surface error: {e}"); return; }
            Ok((mut encoder, view, output)) => {
                for td in &full_output.textures_delta.set {
                    egui_renderer.update_texture(&gpu.device, &gpu.queue, td.0, &td.1);
                }
                egui_renderer.update_buffers(
                    &gpu.device, &gpu.queue, &mut encoder, &paint_jobs, &screen_desc,
                );
                {
                    // forget_lifetime() converts RenderPass<'encoder> → RenderPass<'static>
                    // so egui_renderer.render()'s 'rp constraint is satisfied.
                    // We drop rpass before encoder.finish() to maintain correctness.
                    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("egui"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view:           &view,
                            resolve_target: None,
                            ops:            wgpu::Operations {
                                load:  wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        occlusion_query_set:      None,
                        timestamp_writes:         None,
                    }).forget_lifetime();
                    egui_renderer.render(&mut rpass, &paint_jobs, &screen_desc);
                    drop(rpass);
                }
                for id in &full_output.textures_delta.free {
                    egui_renderer.free_texture(id);
                }
                gpu.queue.submit(std::iter::once(encoder.finish()));
                output.present();
                if !self.sim.is_paused {
                    self.generation     += self.sim.steps_per_frame as u64;
                    self.sim.step_count += self.sim.steps_per_frame as u64;
                }
            }
        }
    }
}

#[cfg(target_arch = "wasm32")]
use std::{cell::RefCell, rc::Rc};

#[cfg(target_arch = "wasm32")]
thread_local! {
    static PENDING_GPU: RefCell<Option<Rc<RefCell<Option<Result<GpuState, anyhow::Error>>>>>> =
        RefCell::new(None);
}

fn build_ui(
    ctx:                 &egui::Context,
    sim:                 &mut SimState,
    fps:                 f32,
    generation:          u64,
    pending_grid_size:   &mut Option<(u32, u32)>,
    pending_reset:       &mut bool,
    pending_rule_change: &mut bool,
) {
    egui::Window::new("GPU Life")
        .anchor(egui::Align2::RIGHT_TOP, [-10.0, 10.0])
        .default_width(280.0)
        .resizable(false)
        .collapsible(false)
        .show(ctx, |ui| {
            if sim.is_paused {
                ui.colored_label(egui::Color32::from_rgb(57, 255, 20), "⏸ PAUSED");
                ui.separator();
            }

            ui.horizontal(|ui| {
                ui.monospace(format!("{:.1} fps", fps));
                ui.separator();
                ui.monospace(format!("gen: {}", generation));
            });
            ui.separator();

            ui.label("Rule");
            egui::ComboBox::from_id_salt("rule_preset")
                .selected_text(sim.rule.name())
                .show_ui(ui, |ui| {
                    for preset in Rule::all_presets() {
                        if ui.selectable_label(sim.rule == *preset, preset.name()).clicked() {
                            sim.set_rule(*preset);
                            *pending_rule_change = true;
                        }
                    }
                    if matches!(sim.rule, Rule::Custom { .. }) {
                        let _ = ui.selectable_label(true, "Custom");
                    }
                });

            ui.horizontal(|ui| {
                ui.label("B:");
                for bit in 0u32..=8 {
                    let mut checked = (sim.config.birth_mask >> bit) & 1 == 1;
                    if ui.checkbox(&mut checked, bit.to_string()).changed() {
                        sim.set_custom_bit(true, bit, checked);
                        *pending_rule_change = true;
                    }
                }
            });
            ui.add(egui::Label::new(
                egui::RichText::new("birth if exactly N live neighbors").small().weak()
            ));

            ui.horizontal(|ui| {
                ui.label("S:");
                for bit in 0u32..=8 {
                    let mut checked = (sim.config.survival_mask >> bit) & 1 == 1;
                    if ui.checkbox(&mut checked, bit.to_string()).changed() {
                        sim.set_custom_bit(false, bit, checked);
                        *pending_rule_change = true;
                    }
                }
            });
            ui.add(egui::Label::new(
                egui::RichText::new("survive if exactly N live neighbors").small().weak()
            ));

            ui.separator();

            ui.label("Steps/frame");
            let mut spf = sim.steps_per_frame;
            if ui.add(egui::Slider::new(&mut spf, 1..=10)).changed() {
                sim.steps_per_frame = spf;
            }
            ui.separator();

            ui.label("Density");
            let mut density_pct = (sim.density * 100.0) as u32;
            if ui.add(egui::Slider::new(&mut density_pct, 0..=100).suffix("%")).changed() {
                sim.density = density_pct as f32 / 100.0;
            }

            if ui.button("Reset").clicked() {
                *pending_reset = true;
            }

            ui.label("Grid size");
            let current_label = GRID_OPTIONS
                .iter()
                .find(|(w, h, _)| *w == sim.config.width && *h == sim.config.height)
                .map(|(_, _, l)| *l)
                .unwrap_or("Custom");
            egui::ComboBox::from_id_salt("grid_size")
                .selected_text(current_label)
                .show_ui(ui, |ui| {
                    for (w, h, label) in GRID_OPTIONS {
                        if ui.selectable_label(
                            sim.config.width == *w && sim.config.height == *h,
                            *label,
                        ).clicked() {
                            *pending_grid_size = Some((*w, *h));
                        }
                    }
                });

            ui.separator();
            ui.add(egui::Label::new(
                egui::RichText::new("Space: pause/resume").small().weak()
            ));
        });
}
