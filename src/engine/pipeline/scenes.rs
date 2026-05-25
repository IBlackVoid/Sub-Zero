use super::{
    assess_srt_health, cue_has_adjacent_repeat, cue_has_low_function_word_coverage,
    cue_has_malformed_contraction, normalize_health_text, parse_srt_timing_line,
    token_has_double_apostrophe, tokenize_ascii_words,
};
use crate::engine::srt::SubtitleCue;

#[derive(Debug, Clone, Copy)]
pub(super) struct SceneQualityReport {
    pub(super) scene_count: usize,
    pub(super) low_quality_ratio: f64,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct LowQualitySceneRange {
    pub(super) start: usize,
    pub(super) end: usize,
    pub(super) score: f64,
    pub(super) floor: f64,
}

pub(super) fn assess_scene_quality(cues: &[SubtitleCue]) -> SceneQualityReport {
    let scenes = split_scenes(cues);
    if scenes.is_empty() {
        return SceneQualityReport {
            scene_count: 0,
            low_quality_ratio: 0.0,
        };
    }

    let mut low_quality = 0usize;
    for scene in &scenes {
        let (difficulty, score) = scene_quality(scene);
        let floor = scene_floor_for_difficulty(difficulty);
        if score < floor {
            low_quality += 1;
        }
    }

    SceneQualityReport {
        scene_count: scenes.len(),
        low_quality_ratio: low_quality as f64 / scenes.len() as f64,
    }
}

pub(super) fn split_scenes(cues: &[SubtitleCue]) -> Vec<Vec<&SubtitleCue>> {
    split_scene_ranges(cues)
        .into_iter()
        .map(|(start, end)| cues[start..end].iter().collect())
        .collect()
}

pub(super) fn split_scene_ranges(cues: &[SubtitleCue]) -> Vec<(usize, usize)> {
    if cues.is_empty() {
        return Vec::new();
    }
    let mut ranges = Vec::<(usize, usize)>::new();
    let mut scene_start: Option<usize> = None;
    let mut scene_end_exclusive = 0usize;
    let mut prev_end = 0.0f64;
    for (idx, cue) in cues.iter().enumerate() {
        let Ok((start, end)) = parse_srt_timing_line(&cue.timing) else {
            continue;
        };
        let gap = (start - prev_end).max(0.0);
        if scene_start.is_none() {
            scene_start = Some(idx);
        } else if gap >= 1.5 || (end - prev_end) >= 420.0 {
            if let Some(start_idx) = scene_start {
                ranges.push((start_idx, scene_end_exclusive));
            }
            scene_start = Some(idx);
        }
        scene_end_exclusive = idx + 1;
        prev_end = end.max(prev_end);
    }
    if let Some(start_idx) = scene_start {
        ranges.push((start_idx, scene_end_exclusive));
    }
    ranges
}

pub(super) fn scene_quality(scene: &[&SubtitleCue]) -> (f64, f64) {
    if scene.is_empty() {
        return (0.5, 1.0);
    }
    let mut anomaly = 0usize;
    let mut malformed = 0usize;
    let mut total_tokens = 0usize;
    let mut durations = 0.0f64;
    let mut prev_end = 0.0f64;
    let mut fast_turns = 0usize;

    for cue in scene {
        let tokens = tokenize_ascii_words(&cue.text);
        total_tokens += tokens.len();
        if cue_has_malformed_contraction(&cue.text) || token_has_double_apostrophe(&tokens) {
            malformed += 1;
            anomaly += 1;
        }
        if cue_has_adjacent_repeat(&tokens) || cue_has_low_function_word_coverage(&tokens) {
            anomaly += 1;
        }
        if let Ok((start, end)) = parse_srt_timing_line(&cue.timing) {
            durations += (end - start).max(0.0);
            let gap = (start - prev_end).max(0.0);
            if gap < 0.15 {
                fast_turns += 1;
            }
            prev_end = end;
        }
    }

    let cues = scene.len().max(1) as f64;
    let anomaly_ratio = anomaly as f64 / cues;
    let malformed_ratio = malformed as f64 / cues;
    let avg_tokens = total_tokens as f64 / cues;
    let avg_duration = durations / cues.max(1.0);
    let fast_turn_ratio = fast_turns as f64 / cues;

    let difficulty = (0.20_f64
        + if avg_duration <= 1.8 {
            0.28_f64
        } else {
            0.10_f64
        }
        + if avg_tokens >= 8.0 {
            0.22_f64
        } else {
            0.08_f64
        }
        + if fast_turn_ratio >= 0.35 {
            0.22_f64
        } else {
            0.05_f64
        })
    .clamp(0.10_f64, 0.95_f64);

    let score = (1.0 - (anomaly_ratio * 1.35 + malformed_ratio * 1.55)).clamp(0.0, 1.0);
    (difficulty, score)
}

pub(super) fn scene_quality_for_slice(scene: &[SubtitleCue]) -> (f64, f64) {
    let refs = scene.iter().collect::<Vec<_>>();
    scene_quality(&refs)
}

pub(super) fn scene_floor_for_difficulty(difficulty: f64) -> f64 {
    if difficulty >= 0.70 {
        0.70
    } else if difficulty <= 0.30 {
        0.90
    } else {
        0.80
    }
}

pub(super) fn collect_low_quality_scene_ranges(cues: &[SubtitleCue]) -> Vec<LowQualitySceneRange> {
    let mut scenes = Vec::<LowQualitySceneRange>::new();
    for (start, end) in split_scene_ranges(cues) {
        if end <= start {
            continue;
        }
        let (difficulty, score) = scene_quality_for_slice(&cues[start..end]);
        let floor = scene_floor_for_difficulty(difficulty);
        if score < floor {
            scenes.push(LowQualitySceneRange {
                start,
                end,
                score,
                floor,
            });
        }
    }
    scenes
}

pub(super) fn collect_low_quality_source_scene_ranges(
    cues: &[SubtitleCue],
) -> Vec<LowQualitySceneRange> {
    let mut scenes = Vec::<LowQualitySceneRange>::new();
    for (start, end) in split_scene_ranges(cues) {
        if end <= start {
            continue;
        }
        let scene = &cues[start..end];
        let score = source_scene_quality_score(scene);
        let floor = source_scene_quality_floor(scene.len());
        if score < floor {
            scenes.push(LowQualitySceneRange {
                start,
                end,
                score,
                floor,
            });
        }
    }
    scenes
}

fn source_scene_quality_floor(scene_len: usize) -> f64 {
    if scene_len >= 8 {
        0.82
    } else if scene_len >= 4 {
        0.76
    } else {
        0.70
    }
}

pub(super) fn source_scene_quality_score(scene: &[SubtitleCue]) -> f64 {
    if scene.is_empty() {
        return 1.0;
    }

    let Ok(health) = assess_srt_health(scene) else {
        return 0.0;
    };

    let mut short = 0usize;
    let mut very_short = 0usize;
    let mut adjacent_duplicates = 0usize;
    let mut prev = String::new();
    for cue in scene {
        if let Ok((start, end)) = parse_srt_timing_line(&cue.timing) {
            let duration = (end - start).max(0.0);
            if duration < 0.45 {
                short += 1;
            }
            if duration < 0.20 {
                very_short += 1;
            }
        }
        let normalized = normalize_health_text(&cue.text);
        if !normalized.is_empty() && normalized == prev {
            adjacent_duplicates += 1;
        }
        prev = normalized;
    }

    let total = scene.len().max(1) as f64;
    let short_ratio = short as f64 / total;
    let very_short_ratio = very_short as f64 / total;
    let duplicate_ratio = adjacent_duplicates as f64 / total;

    (1.0 - (health.top_line_ratio * 1.45
        + health.overlap_ratio * 1.35
        + (1.0 - health.non_empty_ratio) * 1.6
        + short_ratio * 0.55
        + very_short_ratio * 0.8
        + duplicate_ratio * 0.95))
        .clamp(0.0, 1.0)
}
