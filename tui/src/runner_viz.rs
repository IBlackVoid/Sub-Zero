use crate::animation::{Cell, Frame};
use ratatui::style::Color;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VizMode {
    #[default]
    Original,
    Emerge,
    Generative,
}

impl VizMode {
    pub fn label(self) -> &'static str {
        match self {
            VizMode::Original => "original",
            VizMode::Emerge => "emerge",
            VizMode::Generative => "generative",
        }
    }

    pub fn next(self) -> Self {
        match self {
            VizMode::Original => VizMode::Emerge,
            VizMode::Emerge => VizMode::Generative,
            VizMode::Generative => VizMode::Original,
        }
    }
}

pub struct Viz {
    pub mode: VizMode,
    emerge_perm: Vec<u32>,
    emerge_perm_for: Option<String>,
    particles: Vec<Particle>,
    canvas: Vec<f32>,
    canvas_w: usize,
    canvas_h: usize,
    last_chunks_done: u32,
}

#[derive(Debug, Clone, Copy)]
struct Particle {
    x: f32,
    y: f32,
    age: u32,
}

impl Default for Viz {
    fn default() -> Self {
        Self::new()
    }
}

impl Viz {
    pub fn new() -> Self {
        Self {
            mode: VizMode::Original,
            emerge_perm: Vec::new(),
            emerge_perm_for: None,
            particles: Vec::new(),
            canvas: Vec::new(),
            canvas_w: 0,
            canvas_h: 0,
            last_chunks_done: 0,
        }
    }

    pub fn cycle_mode(&mut self) {
        self.mode = self.mode.next();
        if self.mode == VizMode::Generative {
            self.canvas.fill(0.0);
            self.particles.clear();
        }
    }

    pub fn reset_for_run(&mut self) {
        self.emerge_perm.clear();
        self.emerge_perm_for = None;
        self.particles.clear();
        self.canvas.fill(0.0);
        self.last_chunks_done = 0;
    }

    pub fn step(&mut self, area_w: u16, area_h: u16, phase: f32, chunks_done: u32, chunks_total: u32) {
        if self.mode != VizMode::Generative {
            return;
        }
        let dot_w = (area_w as usize) * 2;
        let dot_h = (area_h as usize) * 4;
        if dot_w == 0 || dot_h == 0 {
            return;
        }
        if self.canvas_w != dot_w || self.canvas_h != dot_h {
            self.canvas_w = dot_w;
            self.canvas_h = dot_h;
            self.canvas = vec![0.0; dot_w * dot_h];
            self.particles.clear();
        }

        if self.particles.is_empty() {
            let n = ((dot_w * dot_h) as f32 / 300.0).clamp(40.0, 220.0) as usize;
            for i in 0..n {
                self.particles.push(Particle {
                    x: hash_to_unit(i as u32, 1) * dot_w as f32,
                    y: hash_to_unit(i as u32, 2) * dot_h as f32,
                    age: 0,
                });
            }
        }

        if chunks_done > self.last_chunks_done {
            let burst = 16;
            let corner_x = if (chunks_done % 2) == 0 { 0.0 } else { (dot_w - 1) as f32 };
            let corner_y = if ((chunks_done / 2) % 2) == 0 { 0.0 } else { (dot_h - 1) as f32 };
            for i in 0..burst {
                let theta = hash_to_unit(chunks_done.wrapping_mul(7) ^ i as u32, 3)
                    * std::f32::consts::TAU;
                self.particles.push(Particle {
                    x: corner_x + theta.cos() * 2.0,
                    y: corner_y + theta.sin() * 2.0,
                    age: 0,
                });
            }
            self.last_chunks_done = chunks_done;
        }

        for a in self.canvas.iter_mut() {
            *a *= 0.88;
        }

        let progress = if chunks_total > 0 {
            (chunks_done as f32 / chunks_total as f32).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let field_phase = phase * 0.4 + progress * std::f32::consts::TAU;

        for p in self.particles.iter_mut() {
            p.age = p.age.saturating_add(1);
            let (vx, vy) = sample_field(p.x, p.y, field_phase, dot_w as f32, dot_h as f32);
            p.x += vx;
            p.y += vy;
            if p.x < 0.0 {
                p.x += dot_w as f32;
            }
            if p.x >= dot_w as f32 {
                p.x -= dot_w as f32;
            }
            if p.y < 0.0 {
                p.y += dot_h as f32;
            }
            if p.y >= dot_h as f32 {
                p.y -= dot_h as f32;
            }
            let dx = p.x as usize;
            let dy = p.y as usize;
            if dx < dot_w && dy < dot_h {
                let idx = dy * dot_w + dx;
                self.canvas[idx] = (self.canvas[idx] + 0.55).min(1.0);
            }
        }

        const MAX_PARTICLES: usize = 320;
        if self.particles.len() > MAX_PARTICLES {
            self.particles.sort_by_key(|p| std::cmp::Reverse(p.age));
            self.particles.truncate(MAX_PARTICLES);
        }
    }

    pub fn sample_cell(&self, cell_x: usize, cell_y: usize) -> (char, f32) {
        let dot_w = self.canvas_w;
        let dot_h = self.canvas_h;
        if dot_w == 0 || dot_h == 0 {
            return ('\u{2800}', 0.0);
        }
        const DOT_BITS: [(usize, usize, u8); 8] = [
            (0, 0, 0), (0, 1, 1), (0, 2, 2),
            (1, 0, 3), (1, 1, 4), (1, 2, 5),
            (0, 3, 6), (1, 3, 7),
        ];
        let mut bits = 0u8;
        let mut total = 0.0_f32;
        let mut lit = 0u8;
        let bx = cell_x * 2;
        let by = cell_y * 4;
        for (dx, dy, bit) in DOT_BITS {
            let x = bx + dx;
            let y = by + dy;
            if x >= dot_w || y >= dot_h {
                continue;
            }
            let a = self.canvas[y * dot_w + x];
            if a > 0.18 {
                bits |= 1 << bit;
                total += a;
                lit += 1;
            }
        }
        let avg = if lit > 0 { total / lit as f32 } else { 0.0 };
        let ch = char::from_u32(0x2800 + bits as u32).unwrap_or('\u{2800}');
        (ch, avg)
    }

    pub fn ensure_emerge_perm(&mut self, key: &str, n: u32) {
        if self.emerge_perm_for.as_deref() != Some(key)
            || self.emerge_perm.len() as u32 != n
        {
            self.emerge_perm = permutation_for(key, n);
            self.emerge_perm_for = Some(key.to_string());
        }
    }

    pub fn emerge_visible(&self, cell_index: usize, progress: f32) -> bool {
        if self.emerge_perm.is_empty() {
            return true;
        }
        let threshold = (progress.clamp(0.0, 1.0) * self.emerge_perm.len() as f32) as u32;
        let h = mix_u32(cell_index as u32 ^ self.emerge_perm_seed());
        let rank = h % self.emerge_perm.len() as u32;
        rank < threshold
    }

    fn emerge_perm_seed(&self) -> u32 {
        match self.emerge_perm_for.as_deref() {
            Some(k) => {
                let mut s: u32 = 0x9E37_79B9;
                for b in k.as_bytes() {
                    s = s.wrapping_mul(31).wrapping_add(*b as u32);
                }
                s
            }
            None => 0,
        }
    }
}

fn sample_field(x: f32, y: f32, phase: f32, w: f32, h: f32) -> (f32, f32) {
    let nx = x / w * std::f32::consts::TAU;
    let ny = y / h * std::f32::consts::TAU;
    let theta = (nx * 1.2 + phase).sin()
        + (ny * 1.8 - phase * 0.7).cos()
        + (phase * 0.3).sin();
    (theta.cos() * 0.7, theta.sin() * 0.7)
}

fn hash_to_unit(seed: u32, stream: u32) -> f32 {
    let mut h = seed.wrapping_mul(2_654_435_761).wrapping_add(stream.wrapping_mul(40_503));
    h ^= h.wrapping_shr(15);
    h = h.wrapping_mul(0x85ebca6b);
    h ^= h.wrapping_shr(13);
    h = h.wrapping_mul(0xc2b2ae35);
    h ^= h.wrapping_shr(16);
    (h & 0x00ff_ffff) as f32 / 16_777_215.0
}

pub fn permutation_for(key: &str, n: u32) -> Vec<u32> {
    let mut seed: u32 = 0x9E37_79B9;
    for b in key.as_bytes() {
        seed = seed.wrapping_mul(31).wrapping_add(*b as u32);
    }
    let mut items: Vec<(u32, u32)> = (0..n)
        .map(|i| (i, mix_u32(i ^ seed)))
        .collect();
    items.sort_by_key(|(_, k)| *k);
    items.into_iter().map(|(i, _)| i).collect()
}

fn mix_u32(mut x: u32) -> u32 {
    x ^= x.wrapping_shr(16);
    x = x.wrapping_mul(0x7feb352d);
    x ^= x.wrapping_shr(15);
    x = x.wrapping_mul(0x846ca68b);
    x ^= x.wrapping_shr(16);
    x
}

pub fn fade(c: Color, alpha: f32) -> Color {
    let a = alpha.clamp(0.0, 1.0);
    match c {
        Color::Rgb(r, g, b) => Color::Rgb(
            (r as f32 * a) as u8,
            (g as f32 * a) as u8,
            (b as f32 * a) as u8,
        ),
        other => other,
    }
}

pub fn cell_or_empty(cell: &Cell, visible: bool) -> (char, u8, u8, u8) {
    if visible {
        (cell.ch, cell.r, cell.g, cell.b)
    } else {
        ('\u{2800}', 24, 24, 24)
    }
}

pub fn frame_cell_count(frame: &Frame) -> u32 {
    (frame.width as u32) * (frame.height as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_cycle_three_then_loops() {
        let mut m = VizMode::Original;
        m = m.next();
        assert_eq!(m, VizMode::Emerge);
        m = m.next();
        assert_eq!(m, VizMode::Generative);
        m = m.next();
        assert_eq!(m, VizMode::Original);
    }

    #[test]
    fn permutation_is_a_permutation() {
        let p = permutation_for("file.srt", 1024);
        let mut sorted = p.clone();
        sorted.sort();
        let expected: Vec<u32> = (0..1024).collect();
        assert_eq!(sorted, expected);
    }

    #[test]
    fn permutation_is_deterministic() {
        let a = permutation_for("file.srt", 256);
        let b = permutation_for("file.srt", 256);
        assert_eq!(a, b);
    }

    #[test]
    fn permutation_differs_per_input() {
        let a = permutation_for("alpha.srt", 256);
        let b = permutation_for("beta.srt", 256);
        assert_ne!(a, b);
    }

    #[test]
    fn emerge_visible_grows_with_progress() {
        let mut v = Viz::new();
        v.ensure_emerge_perm("input.srt", 1000);
        let count_at = |p: f32| -> usize {
            (0..1000).filter(|i| v.emerge_visible(*i, p)).count()
        };
        assert!(count_at(0.0) < count_at(0.5));
        assert!(count_at(0.5) < count_at(1.0));
        assert!(count_at(1.0) >= 950);
    }

    #[test]
    fn hash_to_unit_in_range() {
        for i in 0..1000 {
            let v = hash_to_unit(i, 1);
            assert!((0.0..1.0).contains(&v));
        }
    }
}
