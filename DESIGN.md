# Design System — GPU Life

Established from /plan-design-review on the `main` branch, 2026-05-08.
This is the baseline for all future UI decisions. New UI work should extend this, not replace it.

## Visual Identity

**Concept:** GPU performance tool meets living organism. The simulation is the product. The UI is a service to the simulation, not a frame around it.

**Aesthetic:** Terminal-aesthetic dark canvas, bright neon-green cells, minimal chrome. Looks like something you'd see in a GPU benchmark or a shader dev environment.

## Color System

| Token | Hex | Float (WGSL) | Usage |
|---|---|---|---|
| `cell-alive` | `#39FF14` | `vec4(0.224, 1.0, 0.078, 1.0)` | Live cells in render_blit.wgsl |
| `cell-dead` | `#0A0A0F` | `vec4(0.039, 0.039, 0.059, 1.0)` | Dead cells; also page background |
| `text-primary` | egui default (white) | — | Panel labels, status |
| `text-dim` | egui default (gray) | — | Secondary labels, tooltips |

Rules:
- `cell-dead` is used for the page background (`body { background: #0A0A0F }`) so there is no color flash between page load and first WebGPU frame.
- All cell colors are in `render_blit.wgsl`. They are not defined elsewhere.
- Color LUT / age-based coloring is post-MVP. Any future color change routes through `render_blit.wgsl`.

## egui Theme and Panel

**Theme:** Dark (`ctx.set_visuals(egui::Visuals::dark())`). Called once in `app.rs` before the first frame. No custom Visuals override in v1.

**Panel:**
- Window title: `"GPU Life"`
- Position: top-right corner (`egui::Window::new("GPU Life").anchor(egui::Align2::RIGHT_TOP, [-10.0, 10.0])`)
- Width: fixed 280px (`default_width(280.0)`)
- Non-resizable, non-collapsible in v1

**Control layout (top to bottom):**
1. Pause indicator — `"⏸ PAUSED"` text in egui accent color, only when paused; hidden when running
2. Status row — FPS (`"60.1 fps"`) + generation count (`"gen: 12345"`) — small text, `egui::TextStyle::Monospace`
3. Separator
4. **Rule section** — label "Rule"; `ComboBox` with options: Conway / HighLife / Day & Night / Custom; when any B/S bit is manually toggled, label becomes "Custom"; re-selecting a preset resets all bits to that preset
5. B/S bit toggles — 9 checkboxes each, labeled "B0"–"B8" and "S0"–"S8"; hover tooltip: "B = born if exactly N neighbors alive; S = survives if exactly N neighbors alive"
6. Separator
7. **Speed** — label "Steps/frame"; `Slider` 1–10, default 1
8. Separator
9. **Grid** — density `Slider` 0–100%, default 50%; Reset button (randomizes with current density); grid size `ComboBox` (512×512 / 1024×1024 / 2048×2048 / 4096×4096, default 512×512)
10. Footer — `"Space: pause/resume"` (small, dimmed text)

## Simulation Canvas

- Full-screen, no borders, no letterboxing in v1 (grid stretches to fill viewport)
- The canvas IS the product. No other visual elements on the canvas surface.
- Default grid: 512×512 — cells are ~3.75px each at 1080p, individually visible
- Scaling up to 4096×4096 via grid size ComboBox is the performance demo arc

## index.html

- Page title: `"GPU Life"`
- Background: `#0A0A0F` in CSS (prevents flash-of-white)
- CSS reset: `* { margin: 0; padding: 0 }`, `html, body { overflow: hidden }`
- `#status` div: centered monospace text for loading/error states
  - Loading: `"Loading GPU simulation..."`
  - WebGPU absent: `"WebGPU required. Use Chrome 113+ or Arc Browser."`
  - Init failure: `"WebGPU failed to initialize. Your GPU or driver may not be supported."`
  - Running: `#status` hidden (`display: none`)

## Interaction States

| Feature | Loading | Running | Error | Paused |
|---|---|---|---|---|
| Page load | "Loading GPU simulation..." | Canvas visible, status hidden | Error message in #status | — |
| Simulation | (initializing) | Cells animating | — | Cells frozen, "⏸ PAUSED" in panel |
| Grid resize | Brief pause, no indicator | Normal | — | Stays paused |
| Reset | Instant | Simulation restarts | — | Stays paused |

## Decisions NOT in Scope (v1)

- Pan/zoom: fit-to-screen only; 1:1 cell pixels and zoom controls in v2
- Letterboxing: stretch-to-fill in v1; aspect-correct letterbox in v2
- Color themes / LUT: v1 ships hardcoded `#39FF14` on `#0A0A0F`
- Semi-transparent / custom egui panel: dark theme is enough for v1
- Mobile layout: WebGPU on mobile is near-zero; desktop only
