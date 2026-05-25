use std::path::Path;

pub struct Envelope {
    pub samples: Vec<f32>,
    pub interval_ms: u32,
}

impl Envelope {
    pub const INTERVAL_MS: u32 = 50;

    pub fn for_wav(wav_path: &Path) -> Option<Self> {
        let env_path = wav_path.with_extension("env.bin");
        let bytes = std::fs::read(&env_path).ok()?;
        if bytes.len() % 4 != 0 {
            return None;
        }
        let mut samples = Vec::with_capacity(bytes.len() / 4);
        for chunk in bytes.chunks_exact(4) {
            let v = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            samples.push(v.clamp(0.0, 1.0));
        }
        Some(Self {
            samples,
            interval_ms: Self::INTERVAL_MS,
        })
    }

    pub fn window(&self, cursor_ms: u64, window_ms: u32, bars: usize) -> Vec<f32> {
        if bars == 0 || self.samples.is_empty() {
            return Vec::new();
        }
        let interval = self.interval_ms.max(1) as u64;
        let start_ms = cursor_ms.saturating_sub(window_ms as u64);
        let mut out = Vec::with_capacity(bars);
        let bar_span_ms = (window_ms as f64 / bars as f64).max(1.0);
        for i in 0..bars {
            let from_ms = start_ms + ((i as f64 * bar_span_ms) as u64);
            let to_ms = start_ms + (((i + 1) as f64 * bar_span_ms) as u64);
            let from_idx = (from_ms / interval) as usize;
            let to_idx = ((to_ms / interval) as usize).min(self.samples.len());
            if from_idx >= self.samples.len() || to_idx <= from_idx {
                out.push(0.0);
                continue;
            }
            let slice = &self.samples[from_idx..to_idx];
            let mean = slice.iter().copied().sum::<f32>() / slice.len() as f32;
            out.push(mean);
        }
        out
    }
}

pub fn block_for(value: f32) -> char {
    const GLYPHS: [char; 9] = [
        ' ', '\u{2581}', '\u{2582}', '\u{2583}', '\u{2584}', '\u{2585}', '\u{2586}', '\u{2587}',
        '\u{2588}',
    ];
    let idx = ((value.clamp(0.0, 1.0) * 8.0).round() as usize).min(8);
    GLYPHS[idx]
}

pub fn fallback_bars(bars: usize, phase: f64) -> Vec<f32> {
    if bars == 0 {
        return Vec::new();
    }
    (0..bars)
        .map(|i| {
            let x = i as f64 / bars as f64;
            let slow = ((phase * 0.7 + x * std::f64::consts::TAU).sin() * 0.5 + 0.5) as f32;
            let fast = ((phase * 4.0 + x * std::f64::consts::TAU * 3.0).sin() * 0.5 + 0.5) as f32;
            ((slow * 0.65) + (fast * 0.35)).clamp(0.0, 1.0)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_for_extremes() {
        assert_eq!(block_for(0.0), ' ');
        assert_eq!(block_for(1.0), '\u{2588}');
    }

    #[test]
    fn block_for_monotonic() {
        let mut prev: usize = 0;
        for i in 0..=10 {
            let v = i as f32 / 10.0;
            let g = block_for(v) as usize;
            assert!(g >= prev);
            prev = g;
        }
    }

    #[test]
    fn window_handles_empty_envelope() {
        let env = Envelope {
            samples: vec![],
            interval_ms: 50,
        };
        let bars = env.window(1000, 800, 16);
        assert!(bars.is_empty());
    }

    #[test]
    fn window_handles_normal_case() {
        let samples = (0..20).map(|i| (i as f32) / 19.0).collect();
        let env = Envelope {
            samples,
            interval_ms: 50,
        };
        let bars = env.window(1000, 800, 8);
        assert_eq!(bars.len(), 8);
        assert!(bars[bars.len() - 1] > 0.8);
    }

    #[test]
    fn fallback_bars_length() {
        let b = fallback_bars(16, 0.0);
        assert_eq!(b.len(), 16);
        for v in b {
            assert!((0.0..=1.0).contains(&v));
        }
    }
}
