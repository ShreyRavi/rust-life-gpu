// Simulation compute shader: reads state_in, writes state_out.
// Implements outer-totalistic cellular automaton rules via birth/survival bitmasks.

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

@group(0) @binding(0) var<uniform>              config:    Config;
@group(0) @binding(1) var<storage, read>        state_in:  array<u32>;
@group(0) @binding(2) var<storage, read_write>  state_out: array<u32>;

fn idx(x: u32, y: u32) -> u32 {
    return y * config.width + x;
}

// Use i32 offsets to avoid u32 underflow when x==0 or y==0.
// (x - 1u when x==0 wraps to u32::MAX; modulo then gives wrong result
// for non-power-of-2 grid widths.)
fn cell_at(x: u32, y: u32, dx: i32, dy: i32) -> u32 {
    let nx = (i32(x) + dx + i32(config.width))  % i32(config.width);
    let ny = (i32(y) + dy + i32(config.height)) % i32(config.height);
    return state_in[u32(ny) * config.width + u32(nx)];
}

fn count_neighbors(x: u32, y: u32) -> u32 {
    return cell_at(x, y, -1, -1) + cell_at(x, y,  0, -1) + cell_at(x, y,  1, -1)
         + cell_at(x, y, -1,  0)                          + cell_at(x, y,  1,  0)
         + cell_at(x, y, -1,  1) + cell_at(x, y,  0,  1) + cell_at(x, y,  1,  1);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x;
    let y = id.y;
    if x >= config.width || y >= config.height { return; }

    let n     = count_neighbors(x, y);
    let alive = state_in[idx(x, y)];
    let birth   = (config.birth_mask    >> n) & 1u;
    let survive = (config.survival_mask >> n) & 1u;
    // select(val_if_false, val_if_true, condition)
    state_out[idx(x, y)] = select(birth, survive, alive == 1u);
}
