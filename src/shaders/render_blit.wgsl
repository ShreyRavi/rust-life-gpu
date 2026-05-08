// Blit compute shader: converts state buffer to rgba8unorm texture.
// Runs after the simulation compute pass. Hardware texture cache improves
// render pass efficiency vs direct buffer indexing in the fragment shader.

struct Config {
    width:         u32,
    height:        u32,
    depth:         u32,
    birth_mask:    u32,
    survival_mask: u32,
    _pad0:         u32,
    _pad1:         u32,
    _pad2:         u32,
}

@group(0) @binding(0) var<uniform>             config:    Config;
@group(0) @binding(1) var<storage, read>       state:     array<u32>;
@group(0) @binding(2) var                      pixel_out: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x;
    let y = id.y;
    if x >= config.width || y >= config.height { return; }

    let alive = state[y * config.width + x];
    // dead = #0A0A0F, alive = #39FF14
    let color = select(
        vec4<f32>(0.039, 0.039, 0.059, 1.0),  // dead — #0A0A0F
        vec4<f32>(0.224, 1.0,   0.078, 1.0),  // alive — #39FF14
        alive == 1u
    );
    textureStore(pixel_out, vec2<i32>(i32(x), i32(y)), color);
}
