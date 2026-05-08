// Display render shader: samples pixel_texture via textureSample.
// Uses a fullscreen triangle (3 vertices, no vertex buffer needed).

@group(0) @binding(0) var cell_texture: texture_2d<f32>;
@group(0) @binding(1) var cell_sampler: sampler;

struct VertexOut {
    @builtin(position) pos: vec4<f32>,
    @location(0)       uv:  vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOut {
    // Fullscreen triangle covering NDC [-1,1]x[-1,1].
    let x = f32((vi << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(vi & 2u) * 2.0 - 1.0;
    let uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return VertexOut(vec4<f32>(x, y, 0.0, 1.0), uv);
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    return textureSample(cell_texture, cell_sampler, in.uv);
}
