# rust-life-gpu

GPU-accelerated cellular automaton simulator written in Rust. Runs natively via wgpu and in the browser via WebAssembly + WebGPU.

<center><img width="1493" height="823" alt="image" src="https://github.com/user-attachments/assets/6c9d2bda-139c-42c2-a67d-b6080a39a224" />
</center>

## Live demo

https://shreyravi.github.io/rust-life-gpu/

Requires a browser with WebGPU support. Safari and Firefox work reliably. Chrome 113+ works but may show a blank/green screen on some GPU configurations — refresh once or twice, or switch to Safari if the issue persists.

## What it does

Simulates cellular automata on the GPU using a compute shader ping-pong pattern. Each frame, a compute pass reads the current cell grid, applies the selected rule, and writes the next generation into an alternate buffer. A blit pass then copies the result to a render texture, which a display pass draws to the screen.

Supported rules:

- Conway (B3/S23) -- the classic
- HighLife (B36/S23) -- produces replicators
- Day and Night (B3678/S34678) -- symmetric, glassy patterns
- Custom -- toggle any birth/survival neighbor count in the UI

The egui control panel (top right) exposes rule selection, steps per frame, density, grid size, and a reset button. Space bar pauses and resumes.

## Architecture

```
src/
  simulation.rs   -- GridConfig, Rule enum, SimState (CPU side)
  gpu_state.rs    -- wgpu device/queue/surface, buffers, pipelines, bind groups
  app.rs          -- winit ApplicationHandler, egui integration, frame orchestration
  lib.rs          -- WASM entry point (wasm-bindgen)
  main.rs         -- native entry point

  shaders/
    life.wgsl         -- compute: apply birth/survival rule via bit masks
    render_blit.wgsl  -- compute: copy cell buffer to RGBA texture
    display.wgsl      -- vertex + fragment: draw texture to screen
```

Grid sizes: 512x512 (default), 1024x1024, 2048x2048, 4096x4096. All fit in GPU memory; larger grids run slower depending on hardware.

## Local setup

### Native (fastest)

```
cargo run --release
```

Requires a GPU with Vulkan, Metal, or DX12 support. wgpu picks the best available backend automatically.

### Browser (WASM)

Install wasm-pack:

```
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
```

Build and serve:

```
wasm-pack build --target web --release
npx serve .
```

Then open http://localhost:3000 in Chrome 113+ or another WebGPU-capable browser.

## Dependencies

| Crate | Purpose |
|---|---|
| wgpu 22 | GPU abstraction (Vulkan/Metal/DX12/WebGPU) |
| winit 0.30 | Cross-platform window and event loop |
| egui 0.29 + egui-wgpu + egui-winit | Immediate-mode UI |
| bytemuck | Zero-copy casting for GPU buffer uploads |
| web-time | Portable Instant (std::time::Instant does not work on wasm32) |
| wasm-bindgen / wasm-bindgen-futures | WASM glue and async runtime |

## Deployment

Pushing to `main` triggers the GitHub Actions workflow at `.github/workflows/deploy.yml`. It builds with wasm-pack and deploys the output to GitHub Pages automatically.
