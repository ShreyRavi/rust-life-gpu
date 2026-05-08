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

    // --- Rule::masks ---

    #[test]
    fn conway_masks() {
        let (b, s) = Rule::Conway.masks();
        assert_eq!(b, 1 << 3);
        assert_eq!(s, (1 << 2) | (1 << 3));
    }

    #[test]
    fn highlife_masks() {
        let (b, s) = Rule::HighLife.masks();
        assert_eq!(b, (1 << 3) | (1 << 6));
        assert_eq!(s, (1 << 2) | (1 << 3));
    }

    #[test]
    fn day_and_night_masks() {
        let (b, s) = Rule::DayAndNight.masks();
        assert_eq!(b, (1 << 3) | (1 << 6) | (1 << 7) | (1 << 8));
        assert_eq!(s, (1 << 3) | (1 << 4) | (1 << 6) | (1 << 7) | (1 << 8));
    }

    #[test]
    fn custom_masks_passthrough() {
        let r = Rule::Custom { birth_mask: 0b101, survival_mask: 0b011 };
        assert_eq!(r.masks(), (0b101, 0b011));
    }

    // --- Rule::name ---

    #[test]
    fn rule_names() {
        assert_eq!(Rule::Conway.name(), "Conway");
        assert_eq!(Rule::HighLife.name(), "HighLife");
        assert_eq!(Rule::DayAndNight.name(), "Day & Night");
        assert_eq!(Rule::Custom { birth_mask: 0, survival_mask: 0 }.name(), "Custom");
    }

    // --- Rule::all_presets ---

    #[test]
    fn all_presets_has_three_entries() {
        let p = Rule::all_presets();
        assert_eq!(p.len(), 3);
        assert!(p.contains(&Rule::Conway));
        assert!(p.contains(&Rule::HighLife));
        assert!(p.contains(&Rule::DayAndNight));
    }

    // --- GridConfig::new ---

    #[test]
    fn grid_config_fields() {
        let cfg = GridConfig::new(128, 64, Rule::Conway);
        assert_eq!(cfg.width, 128);
        assert_eq!(cfg.height, 64);
        assert_eq!(cfg.depth, 1);
        let (b, s) = Rule::Conway.masks();
        assert_eq!(cfg.birth_mask, b);
        assert_eq!(cfg.survival_mask, s);
        assert_eq!(cfg._pad0, 0);
        assert_eq!(cfg._pad1, 0);
        assert_eq!(cfg._pad2, 0);
    }

    // --- SimState::new ---

    #[test]
    fn simstate_defaults() {
        let s = SimState::new(100, 200);
        assert_eq!(s.config.width, 100);
        assert_eq!(s.config.height, 200);
        assert_eq!(s.rule, Rule::Conway);
        assert_eq!(s.step_count, 0);
        assert!(!s.is_paused);
        assert_eq!(s.steps_per_frame, 1);
        assert!((s.density - 0.5).abs() < f32::EPSILON * 4.0);
    }

    // --- SimState::set_rule ---

    #[test]
    fn set_rule_updates_rule_and_masks() {
        let mut s = SimState::new(64, 64);
        s.set_rule(Rule::HighLife);
        assert_eq!(s.rule, Rule::HighLife);
        let (b, sv) = Rule::HighLife.masks();
        assert_eq!(s.config.birth_mask, b);
        assert_eq!(s.config.survival_mask, sv);
    }

    #[test]
    fn set_rule_day_and_night() {
        let mut s = SimState::new(64, 64);
        s.set_rule(Rule::DayAndNight);
        assert_eq!(s.rule, Rule::DayAndNight);
        let (b, sv) = Rule::DayAndNight.masks();
        assert_eq!(s.config.birth_mask, b);
        assert_eq!(s.config.survival_mask, sv);
    }

    // --- SimState::set_custom_bit ---

    #[test]
    fn set_custom_bit_birth_set() {
        let mut s = SimState::new(64, 64);
        s.set_custom_bit(true, 5, true);
        assert_ne!(s.config.birth_mask & (1 << 5), 0);
    }

    #[test]
    fn set_custom_bit_birth_clear() {
        let mut s = SimState::new(64, 64);
        // Conway birth has bit 3 set; clear it
        s.set_custom_bit(true, 3, false);
        assert_eq!(s.config.birth_mask & (1 << 3), 0);
    }

    #[test]
    fn set_custom_bit_survival_set() {
        let mut s = SimState::new(64, 64);
        s.set_custom_bit(false, 5, true);
        assert_ne!(s.config.survival_mask & (1 << 5), 0);
    }

    #[test]
    fn set_custom_bit_survival_clear() {
        let mut s = SimState::new(64, 64);
        // Conway survival has bit 2 set; clear it
        s.set_custom_bit(false, 2, false);
        assert_eq!(s.config.survival_mask & (1 << 2), 0);
    }

    #[test]
    fn set_custom_bit_snaps_to_preset() {
        let mut s = SimState::new(64, 64);
        // Conway + birth bit 6 = HighLife (same survival mask)
        s.set_custom_bit(true, 6, true);
        assert_eq!(s.rule, Rule::HighLife);
    }

    #[test]
    fn set_custom_bit_becomes_custom_when_no_preset_matches() {
        let mut s = SimState::new(64, 64);
        s.set_custom_bit(true, 7, true);
        assert!(matches!(s.rule, Rule::Custom { .. }));
    }

    #[test]
    fn set_custom_bit_custom_stores_masks() {
        let mut s = SimState::new(64, 64);
        s.set_custom_bit(true, 7, true);
        if let Rule::Custom { birth_mask, survival_mask } = s.rule {
            assert_eq!(birth_mask, s.config.birth_mask);
            assert_eq!(survival_mask, s.config.survival_mask);
        } else {
            panic!("expected Custom rule");
        }
    }

    // --- SimState::random_state ---

    #[test]
    fn random_state_correct_length() {
        let s = SimState::new(16, 32);
        assert_eq!(s.random_state().len(), 16 * 32);
    }

    #[test]
    fn random_state_values_are_binary() {
        let s = SimState::new(64, 64);
        assert!(s.random_state().iter().all(|&c| c == 0 || c == 1));
    }

    #[test]
    fn random_state_density_zero_all_dead() {
        let mut s = SimState::new(64, 64);
        s.density = 0.0;
        assert!(s.random_state().iter().all(|&c| c == 0));
    }

    #[test]
    fn random_state_density_one_all_alive() {
        let mut s = SimState::new(64, 64);
        s.density = 1.0;
        assert!(s.random_state().iter().all(|&c| c == 1));
    }

    #[test]
    fn random_state_density_half_roughly_half_alive() {
        let mut s = SimState::new(256, 256);
        s.density = 0.5;
        let cells = s.random_state();
        let alive = cells.iter().filter(|&&c| c == 1).count();
        let total = cells.len();
        // Within 10% of 50%
        assert!(alive > total * 40 / 100 && alive < total * 60 / 100);
    }

    // --- apply_rule exhaustive ---

    #[test]
    fn conway_apply_rule_all_neighbors() {
        let cfg = conway();
        for n in 0u32..=8 {
            let expected_dead = if n == 3 { 1 } else { 0 };
            let expected_alive = if n == 2 || n == 3 { 1 } else { 0 };
            assert_eq!(apply_rule(0, n, &cfg), expected_dead, "dead cell n={}", n);
            assert_eq!(apply_rule(1, n, &cfg), expected_alive, "alive cell n={}", n);
        }
    }

    #[test]
    fn highlife_apply_rule_all_neighbors() {
        let cfg = GridConfig::new(64, 64, Rule::HighLife);
        for n in 0u32..=8 {
            let expected_dead = if n == 3 || n == 6 { 1 } else { 0 };
            let expected_alive = if n == 2 || n == 3 { 1 } else { 0 };
            assert_eq!(apply_rule(0, n, &cfg), expected_dead, "dead cell n={}", n);
            assert_eq!(apply_rule(1, n, &cfg), expected_alive, "alive cell n={}", n);
        }
    }

    #[test]
    fn day_and_night_apply_rule_all_neighbors() {
        let cfg = GridConfig::new(64, 64, Rule::DayAndNight);
        let born_at    = [3u32, 6, 7, 8];
        let survive_at = [3u32, 4, 6, 7, 8];
        for n in 0u32..=8 {
            let expected_dead  = if born_at.contains(&n) { 1 } else { 0 };
            let expected_alive = if survive_at.contains(&n) { 1 } else { 0 };
            assert_eq!(apply_rule(0, n, &cfg), expected_dead, "dead cell n={}", n);
            assert_eq!(apply_rule(1, n, &cfg), expected_alive, "alive cell n={}", n);
        }
    }

    #[test]
    fn custom_rule_apply() {
        let cfg = GridConfig::new(4, 4, Rule::Custom { birth_mask: 1 << 1, survival_mask: 1 << 5 });
        assert_eq!(apply_rule(0, 1, &cfg), 1);
        assert_eq!(apply_rule(0, 2, &cfg), 0);
        assert_eq!(apply_rule(1, 5, &cfg), 1);
        assert_eq!(apply_rule(1, 4, &cfg), 0);
    }

    // --- legacy spot-checks (kept for regression) ---

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

    // --- ping_pong_index ---

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

    #[test]
    fn ping_pong_large_values() {
        assert_eq!(ping_pong_index(u64::MAX), 1);     // odd
        assert_eq!(ping_pong_index(u64::MAX - 1), 0); // even
    }
}
