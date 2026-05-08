mod simulation;
mod gpu_state;
mod app;

use winit::event_loop::{ControlFlow, EventLoop};

pub async fn run() {
    let event_loop = EventLoop::new().expect("event loop");
    event_loop.set_control_flow(ControlFlow::Poll);

    #[allow(unused_mut)]
    let mut app_state = app::App::new();

    #[cfg(not(target_arch = "wasm32"))]
    {
        event_loop.run_app(&mut app_state).expect("run_app");
    }

    #[cfg(target_arch = "wasm32")]
    {
        use winit::platform::web::EventLoopExtWebSys;
        event_loop.spawn_app(app_state);
    }
}

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn run_app() {
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    console_log::init_with_level(log::Level::Warn).expect("logger");
    wasm_bindgen_futures::spawn_local(run());
}
