fn main() {
    env_logger::init();
    pollster::block_on(rust_life_gpu::run());
}
