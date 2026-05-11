use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;

#[derive(Debug)]
pub struct LiveHistogramReplanner {
    max_workers: usize,
    min_workers: usize,
    active_limit: AtomicUsize,
    timeout_scale_bits: AtomicU64,
    state: Mutex<ReplannerState>,
}

#[derive(Debug)]
struct ReplannerState {
    asr_secs_per_audio: VecDeque<f64>,
    last_limit: usize,
    last_scale: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct ReplanDecision {
    pub new_asr_limit: usize,
    pub new_timeout_scale: f64,
    pub asr_p95_secs_per_audio: f64,
}

impl LiveHistogramReplanner {
    pub fn new(max_workers: usize) -> Self {
        let max_workers = max_workers.max(1);
        Self {
            max_workers,
            min_workers: 1,
            active_limit: AtomicUsize::new(max_workers),
            timeout_scale_bits: AtomicU64::new(1.0f64.to_bits()),
            state: Mutex::new(ReplannerState {
                asr_secs_per_audio: VecDeque::new(),
                last_limit: max_workers,
                last_scale: 1.0,
            }),
        }
    }

    pub fn asr_limit(&self) -> usize {
        self.active_limit
            .load(Ordering::Relaxed)
            .clamp(self.min_workers, self.max_workers)
    }

    pub fn timeout_scale(&self) -> f64 {
        f64::from_bits(self.timeout_scale_bits.load(Ordering::Relaxed)).clamp(0.50, 3.00)
    }

    pub fn note_asr_sample(
        &self,
        chunk_len_secs: f64,
        asr_elapsed_secs: f64,
    ) -> Option<ReplanDecision> {
        if !chunk_len_secs.is_finite()
            || !asr_elapsed_secs.is_finite()
            || chunk_len_secs <= 0.0
            || asr_elapsed_secs < 0.0
        {
            return None;
        }

        let secs_per_audio = (asr_elapsed_secs / chunk_len_secs).clamp(0.0, 60.0);

        let mut guard = self.state.lock().ok()?;
        push_bounded(&mut guard.asr_secs_per_audio, secs_per_audio, 96);
        let p95 = p95_of(&guard.asr_secs_per_audio);

        // Heuristic: treat 1.0x realtime as the inflection point.
        // Slow ASR (p95 > 1.2x realtime): reduce concurrency to avoid thrash/timeouts.
        // Fast ASR (p95 < 0.45x realtime): we can safely increase concurrency up to max.
        let current_limit = self.asr_limit();
        let mut new_limit = current_limit;
        if p95 > 1.20 {
            new_limit = current_limit.saturating_sub(1).max(self.min_workers);
        } else if p95 < 0.45 {
            new_limit = (current_limit + 1).min(self.max_workers);
        }

        // Timeout scaling: keep headroom above p95.
        let target_scale = (p95 / 0.85).clamp(0.75, 2.50);

        let changed =
            new_limit != guard.last_limit || (target_scale - guard.last_scale).abs() > 0.10;
        if !changed {
            return None;
        }

        self.active_limit.store(new_limit, Ordering::Relaxed);
        self.timeout_scale_bits
            .store(target_scale.to_bits(), Ordering::Relaxed);
        guard.last_limit = new_limit;
        guard.last_scale = target_scale;

        Some(ReplanDecision {
            new_asr_limit: new_limit,
            new_timeout_scale: target_scale,
            asr_p95_secs_per_audio: p95,
        })
    }
}

fn push_bounded<T>(deque: &mut VecDeque<T>, item: T, max_len: usize) {
    deque.push_back(item);
    while deque.len() > max_len {
        deque.pop_front();
    }
}

fn p95_of(samples: &VecDeque<f64>) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut v = samples.iter().copied().collect::<Vec<f64>>();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((v.len() - 1) as f64 * 0.95).round() as usize;
    v[idx.min(v.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::LiveHistogramReplanner;

    #[test]
    fn replanner_scales_down_on_slow_asr() {
        let replanner = LiveHistogramReplanner::new(6);
        assert_eq!(replanner.asr_limit(), 6);
        for _ in 0..20 {
            // 2.0x realtime: should reduce.
            let _ = replanner.note_asr_sample(10.0, 20.0);
        }
        assert!(replanner.asr_limit() < 6);
        assert!(replanner.timeout_scale() > 1.0);
    }

    #[test]
    fn replanner_scales_up_on_fast_asr() {
        let replanner = LiveHistogramReplanner::new(4);
        // Force it down first.
        for _ in 0..20 {
            let _ = replanner.note_asr_sample(10.0, 20.0);
        }
        let down = replanner.asr_limit();
        for _ in 0..40 {
            // 0.2x realtime: should increase toward max.
            let _ = replanner.note_asr_sample(10.0, 2.0);
        }
        assert!(replanner.asr_limit() >= down);
        assert!(replanner.asr_limit() <= 4);
    }
}
