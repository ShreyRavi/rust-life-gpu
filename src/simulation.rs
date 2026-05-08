use bytemuck::{Pod, Zeroable};

// WebGPU uniform buffer must be a multiple of 16 bytes.
// 5 × u32 = 20 bytes → pad to 32 bytes with 3 _pad fields.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GridConfig {
    pub width:         u32,
    pub height:        u32,
    pub depth:         u32,  // reserved for 3D; always 1 in 2D
    pub birth_mask:    u32,
    pub survival_mask: u32,
    pub _pad0:         u32,
    pub _pad1:         u32,
    pub _pad2:         u32,
}

impl GridConfig {
    pub fn new(width: u32, height: u32, rule: Rule) -> Self {
        let (birth_mask, survival_mask) = rule.masks();
        Self {
            width,
            height,
            depth: 1,
            birth_mask,
            survival_mask,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Rule {
    Conway,
    HighLife,
    DayAndNight,
    Custom { birth_mask: u32, survival_mask: u32 },
}

impl Rule {
    // Returns (birth_mask, survival_mask).
    // Bit N set = born/survives with exactly N live neighbors.
    pub fn masks(self) -> (u32, u32) {
        match self {
            // Conway B3/S23
            Rule::Conway => (1 << 3, (1 << 2) | (1 << 3)),
            // HighLife B36/S23
            Rule::HighLife => ((1 << 3) | (1 << 6), (1 << 2) | (1 << 3)),
            // Day & Night B3678/S34678
            Rule::DayAndNight => (
                (1 << 3) | (1 << 6) | (1 << 7) | (1 << 8),
                (1 << 3) | (1 << 4) | (1 << 6) | (1 << 7) | (1 << 8),
            ),
            Rule::Custom { birth_mask, survival_mask } => (birth_mask, survival_mask),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Rule::Conway => "Conway",
            Rule::HighLife => "HighLife",
            Rule::DayAndNight => "Day & Night",
            Rule::Custom { .. } => "Custom",
        }
    }

    pub fn all_presets() -> &'static [Rule] {
        &[Rule::Conway, Rule::HighLife, Rule::DayAndNight]
    }
}

#[allow(dead_code)]
pub fn apply_rule(alive: u32, neighbors: u32, cfg: &GridConfig) -> u32 {
    let birth   = (cfg.birth_mask    >> neighbors) & 1;
    let survive = (cfg.survival_mask >> neighbors) & 1;
    if alive == 1 { survive } else { birth }
}

#[allow(dead_code)]
pub fn ping_pong_index(step_count: u64) -> usize {
    (step_count % 2) as usize
}

pub struct SimState {
    pub config: GridConfig,
    pub rule: Rule,
    pub step_count: u64,
    pub is_paused: bool,
    pub steps_per_frame: u32,
    pub density: f32, // 0.0–1.0
}

impl SimState {
    pub fn new(width: u32, height: u32) -> Self {
        let rule = Rule::Conway;
        Self {
            config: GridConfig::new(width, height, rule),
            rule,
            step_count: 0,
            is_paused: false,
            steps_per_frame: 1,
            density: 0.5,
        }
    }

    pub fn set_rule(&mut self, rule: Rule) {
        self.rule = rule;
        let (birth_mask, survival_mask) = rule.masks();
        self.config.birth_mask = birth_mask;
        self.config.survival_mask = survival_mask;
    }

    pub fn set_custom_bit(&mut self, is_birth: bool, bit: u32, set: bool) {
        if is_birth {
            if set { self.config.birth_mask    |= 1 << bit; }
            else   { self.config.birth_mask    &= !(1 << bit); }
        } else {
            if set { self.config.survival_mask |= 1 << bit; }
            else   { self.config.survival_mask &= !(1 << bit); }
        }
        // If current masks no longer match any preset, show Custom.
        let matching = Rule::all_presets()
            .iter()
            .find(|r| r.masks() == (self.config.birth_mask, self.config.survival_mask))
            .copied();
        self.rule = matching.unwrap_or(Rule::Custom {
            birth_mask:    self.config.birth_mask,
            survival_mask: self.config.survival_mask,
        });
    }

    pub fn random_state(&self) -> Vec<u32> {
        let n = (self.config.width * self.config.height) as usize;
        let mut cells = vec![0u32; n];
        // Simple xorshift64 RNG — no std::time needed, deterministic per session.
        let mut rng: u64 = 0xdeadbeef_cafebabe;
        let threshold = (self.density * u32::MAX as f32) as u64;
        for cell in cells.iter_mut() {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            *cell = if (rng & 0xFFFF_FFFF) < threshold { 1 } else { 0 };
        }
        cells
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn conway() -> GridConfig {
        GridConfig::new(64, 64, Rule::Conway)
    }

    #[test]
    fn conway_survives_2() {
        assert_eq!(apply_rule(1, 2, &conway()), 1);
    }

    #[test]
    fn conway_survives_3() {
        assert_eq!(apply_rule(1, 3, &conway()), 1);
    }

    #[test]
    fn conway_dies_1() {
        assert_eq!(apply_rule(1, 1, &conway()), 0);
    }

    #[test]
    fn conway_dies_4() {
        assert_eq!(apply_rule(1, 4, &conway()), 0);
    }

    #[test]
    fn conway_born_3() {
        assert_eq!(apply_rule(0, 3, &conway()), 1);
    }

    #[test]
    fn conway_stays_dead_2() {
        assert_eq!(apply_rule(0, 2, &conway()), 0);
    }

    #[test]
    fn highlife_born_6() {
        let cfg = GridConfig::new(64, 64, Rule::HighLife);
        assert_eq!(apply_rule(0, 6, &cfg), 1);
    }

    #[test]
    fn day_and_night_dies_isolated() {
        let cfg = GridConfig::new(64, 64, Rule::DayAndNight);
        assert_eq!(apply_rule(1, 0, &cfg), 0);
    }

    #[test]
    fn ping_pong_step0() {
        assert_eq!(ping_pong_index(0), 0);
    }

    #[test]
    fn ping_pong_step1() {
        assert_eq!(ping_pong_index(1), 1);
    }

    #[test]
    fn ping_pong_wraps() {
        assert_eq!(ping_pong_index(2), 0);
        assert_eq!(ping_pong_index(3), 1);
    }
}
